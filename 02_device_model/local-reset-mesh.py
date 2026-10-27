"""Local spatial refinement of the fixed positive-window scenario.

Frozen v3 ionic equations; cell-integrated reflected Gaussian electronic
sampling on a nonuniform mesh. No parameter changes or state resets in a run.
"""
from pathlib import Path
from dataclasses import replace,asdict
import importlib.util,sys,json,argparse,time,hashlib
import numpy as np
from scipy.special import expit,logit,ndtr
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix
R=Path(__file__).resolve().parent;O=R/'outputs'
s=importlib.util.spec_from_file_location('local_mesh_v3',R/'formed-channel-set-v3.py')
m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)

def fingerprint(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def kernel(edges,sigma):
    """Double-cell integral of the Neumann Gaussian kernel, divided by target h.

    Image terms k=-1,0,1 suffice here because each material exceeds 20 sigma.
    W*1=1 and diag(h)*W is symmetric before floating-point roundoff.
    """
    e=edges-edges[0];length=e[-1];h=np.diff(e)
    assert length>=20*sigma
    a=e[:-1,None];b=e[1:,None];c=e[None,:-1];d=e[None,1:]
    H=lambda z:z*ndtr(z/sigma)+sigma/np.sqrt(2*np.pi)*np.exp(-.5*(z/sigma)**2)
    mass=np.zeros((len(h),len(h)))
    for k in (-1,0,1):
        shift=2*k*length
        mass+=H(b-c+shift)-H(a-c+shift)-H(b-d+shift)+H(a-d+shift)
        mass+=H(b+d+shift)-H(a+d+shift)-H(b+c+shift)+H(a+c+shift)
    mass=(mass+mass.T)/2
    # Ignore only cancellation-level far-tail entries, then restore exact row mass
    # on the diagonal. Symmetry and weighted conservation are both retained.
    mass[mass<1e-12*np.min(h)]=0
    mass[np.diag_indices(len(h))]+=h-mass.sum(axis=1)
    assert mass.min()>-1e-12 and np.max(abs(mass.sum(axis=1)/h-1))<1e-10
    return csr_matrix(mass/h[:,None])

class Model(m.Model):
    def __init__(self,p,edges):
        p=replace(p,cells=len(edges)-1);super().__init__(p)
        self.edges=np.array(edges);self.h=np.diff(edges);self.x=(edges[1:]+edges[:-1])/2
        self.ti=self.x<p.TiOx_nm;self.N=len(self.h)
        assert np.min(abs(edges-p.TiOx_nm))<1e-12
        interface=np.flatnonzero(np.diff(self.ti.astype(int)))[0]
        # Parent harmonic D is exact at the material face when half-widths agree.
        assert abs(self.h[interface]-self.h[interface+1])<1e-10
        self.total_sites=np.where(self.ti,1.,1.4);self.S=np.where(self.ti,.10,.028)
        self.bound=np.where(self.ti,.06*expit((self.x-p.active_Ti_nm)/.15),.042) if p.backbone else np.zeros(self.N)
        self.C=self.h*self.S;self.G=np.where(self.ti,0.,.02)
        self.D=np.where(self.ti,p.diffusion_Ti_nm2_per_tau,p.diffusion_Al_nm2_per_tau)
        self.xi=np.where(self.ti,p.localization_Ti_nm,p.localization_Al_nm)
        self.Ea_e=np.where(self.ti,p.electron_activation_Ti_eV,p.electron_activation_Al_eV)
        th=np.where(self.ti,.2,expit(np.log(.2/.8)-.02/m.KT))
        self.initial=np.r_[logit(th),logit(p.Ti_initial_oxygen_fraction)]
        self.initial_defects=self.S*th+self.bound
        self.weights=[]
        for mask in (self.ti,~self.ti):
            ids=np.flatnonzero(mask);ee=self.edges[ids[0]:ids[-1]+2]
            self.weights.append((mask,kernel(ee,p.electronic_sampling_nm)))
        self.W=np.zeros((self.N,self.N))
        for mask,W in self.weights:self.W[np.ix_(mask,mask)]=W.toarray()

    def electrical(self,y,V):
        th=expit(y[:-1]);st=expit(y[-1]);cv=self.S*th+self.bound
        cv_e=cv if self.p.vacancy_conductivity_feedback else self.initial_defects
        avg=np.empty_like(cv_e)
        for mask,W in self.weights:avg[mask]=W@cv_e[mask]
        spacing=(50*avg)**(-1/3)
        sigma=1e-12+avg*np.exp(-2*spacing/self.xi-self.Ea_e/m.KT)
        rr=self.h/sigma;total=rr.sum();J=V/total;drop=J*rr
        faces=np.r_[0.,np.cumsum(drop)];phi=(faces[:-1]+faces[1:])/2
        return dict(theta=th,sTi=st,cv=cv,sigma=sigma,R=total,G=1/total,J=J,phi=phi,E=-drop/self.h,drop=drop,faces=faces)

    @staticmethod
    def bernoulli_derivative(x):
        x=np.asarray(x);u=abs(x);small=u<1e-4
        q=np.empty_like(u);q[small]=-.5+u[small]/6-u[small]**3/180+u[small]**5/5040
        v=u[~small];e=np.exp(-v)
        q[~small]=e*(1-v-e)/(-np.expm1(-v))**2
        return np.where(x>=0,q,-q-1)

    def jacobian(self,y,V):
        """Exact chain rule through sampling, series voltage division and fluxes."""
        if not self.p.vacancy_conductivity_feedback:
            raise NotImplementedError('Analytic Jacobian currently requires composition-dependent conductivity.')
        n=self.N;d=self.electrical(y,V);th=d['theta'];s=d['sTi'];dt=th*(1-th);ds=s*(1-s)
        cv=self.S*th+self.bound;avg=self.W@cv;spacing=(50*avg)**(-1/3)
        sigprime=(d['sigma']-1e-12)/avg*(1+2*spacing/(3*self.xi))
        drr=-(self.h/d['sigma']**2*sigprime)[:,None]*self.W*(self.S*dt)[None,:]
        rr=self.h/d['sigma'];dropgrad=d['J']*drr+rr[:,None]*(-d['J']/d['R']*drr.sum(0))[None,:]
        phigrad=np.cumsum(dropgrad,axis=0)-.5*dropgrad
        dp=np.diff(d['phi']);dx=np.diff(self.x);delta=(np.diff(self.G)+2*dp)/m.KT
        g=abs(dp)*self.p.lattice_hop_nm/(dx*m.KT)
        small=g<1e-5;mob=np.ones_like(g);logprime=np.zeros_like(g)
        mob[small]=1+g[small]**2/6
        logprime[small]=(g[small]/3)/(1+g[small]**2/6)
        q=g[~small];cap=self.p.hop_barrier_eV/m.KT
        mob[~small]=np.exp(np.minimum(q,cap))*(-np.expm1(-2*q))/(2*q)
        e=np.exp(-2*q);logprime[~small]=(q<cap).astype(float)+2*e/(-np.expm1(-2*q))-1/q
        if not self.p.high_field_hopping:mob[:]=1;logprime[:]=0
        Df=2*self.D[:-1]*self.D[1:]/(self.D[:-1]+self.D[1:])
        pref=Df*np.sqrt(self.S[:-1]*self.S[1:])/dx*mob
        Bp=self.bernoulli(delta);Bm=self.bernoulli(-delta)
        left=th[:-1]*(1-th[1:]);right=th[1:]*(1-th[:-1])
        flux=pref*(Bp*left-Bm*right)
        fieldprime=pref*(self.bernoulli_derivative(delta)*left+self.bernoulli_derivative(-delta)*right)*2/m.KT
        fieldprime+=flux*logprime*np.sign(dp)*self.p.lattice_hop_nm/(dx*m.KT)
        fg=np.zeros((n-1,n+1));fg[:,:n]=fieldprime[:,None]*np.diff(phigrad,axis=0)
        ii=np.arange(n-1)
        fg[ii,ii]+=pref*(Bp*(1-th[1:])+Bm*th[1:])*dt[:-1]
        fg[ii,ii+1]-=pref*(Bp*th[:-1]+Bm*(1-th[:-1]))*dt[1:]
        k=self.p.Ti_exchange_nm_per_tau*self.S[0];dg=(self.G[0]+self.p.Ti_reaction_standard_eV)/m.KT
        A=k*np.exp(-dg/2)*(1-s);B=k*np.exp(dg/2)*s
        fdelta=np.array([2*d['phi'][0]/m.KT]);bp=self.bernoulli(fdelta)[0];bm=self.bernoulli(-fdelta)[0]
        K=2*self.D[0]*self.S[0]/self.h[0];C=K*bp*(1-th[0]);D=K*bm*th[0]
        den=A+B+C+D;exchange=(A*C-B*D)/den
        eA=(C-exchange)/den;eB=(-D-exchange)/den;eC=(A-exchange)/den;eD=(-B-exchange)/den
        eg=np.zeros(n+1)
        eg[:n]=(eC*K*self.bernoulli_derivative(fdelta)[0]*(1-th[0])-eD*K*self.bernoulli_derivative(-fdelta)[0]*th[0])*2/m.KT*phigrad[0]
        eg[0]+=(-eC*K*bp+eD*K*bm)*dt[0]
        eg[-1]=(-eA*k*np.exp(-dg/2)+eB*k*np.exp(dg/2))*ds
        rates=-np.diff(np.r_[exchange,flux,0.])
        rg=-np.diff(np.vstack([eg,fg,np.zeros(n+1)]),axis=0)
        jac=np.empty((n+1,n+1));jac[:n]=rg/(self.C*dt)[:,None]
        jac[np.arange(n),np.arange(n)]-=rates/(self.C*dt)*(1-2*th)
        jac[-1]=eg/(self.Cti*ds);jac[-1,-1]-=exchange/(self.Cti*ds)*(1-2*s)
        if not self.p.vacancy_transport:jac[:]=0
        return jac

def grid(local,outer,core=0):
    near=np.linspace(0,2,local+1) if not core else np.r_[np.linspace(0,.5,core+1),np.linspace(.5,2,3*local//4+1)[1:]]
    return np.r_[near,np.linspace(2,20,18*outer+1)[1:],np.linspace(20,25,5*outer+1)[1:]]

def project(source_edges,source_theta,target_edges):
    # Conservatively transfer cell-average filling separately at aligned material
    # interfaces; S is constant within each material, so this also preserves ions.
    overlap=np.maximum(0,np.minimum(target_edges[1:,None],source_edges[None,1:])-np.maximum(target_edges[:-1,None],source_edges[None,:-1]))
    return overlap@source_theta/np.diff(target_edges)

def run(local,outer,mode,rtol,analytic=False,core=0):
    src=O/'positive-window-xi175-DTi00005-C5-N800-R1e-08.npz'
    a=np.load(src);meta=json.loads(src.with_suffix('.json').read_text());assert fingerprint(src)==meta['file_sha256']
    p=m.Parameters(**meta['parameters']);model=m.Model(replace(p,cells=local)) if mode=='legacy' else Model(p,grid(local,outer,core))
    stem=f'local-reset-{mode}-L{local}-O{outer}-R{rtol:g}'+(f'-B{core}' if core else '')+('-J' if analytic else '')
    if (O/(stem+'.json')).exists():print('EXISTS',stem,flush=True);return
    if mode in ('reset','legacy'):
        idx=np.flatnonzero(a['kind']=='positive-fall-3')[-1]
        th=project(a['edges'],a['theta'][idx],model.edges);z=np.r_[logit(th),a['z'][idx,-1]]
        source_invariant=float(m.Model(p).C@a['theta'][idx]-model.Cti*expit(z[-1]))
        assert abs(model.C@th-model.Cti*expit(z[-1])-source_invariant)<1e-11
        rows=[(10,0,-1,'negative-rise-3'),(10,-1,0,'negative-fall-3')]
        start_time=140.
    else:
        z=model.initial.copy();rows=[(10,0,-1,'prepare-rise'),(10,-1,0,'prepare-fall'),(.2,0,.02,'read-rise'),(1,.02,.02,'read'),(.2,.02,0,'read-fall')]
        rows += [(10,v0,v1,f'{k}-{c}') for c in range(5) for k,v0,v1 in [('positive-rise',0,1),('positive-fall',1,0),('negative-rise',0,-1),('negative-fall',-1,0)]]
        start_time=0.;idx=None
    ts=[];zs=[];vs=[];kinds=[];clock=start_time;segments=[];begin=time.perf_counter()
    for dur,v0,v1,k in rows:
        vf=lambda t:m.wave(t,0,dur,v0,v1)
        kw=dict(jac=lambda t,y:model.jacobian(y,vf(t))) if analytic else {}
        sol=solve_ivp(lambda t,y:model.evaluate(y,vf(t)),(0,dur),z,method='BDF',rtol=rtol,atol=rtol*.01,dense_output=True,**kw)
        if not sol.success:raise RuntimeError(sol.message)
        tt=np.linspace(0,dur,61);zz=sol.sol(tt).T;vv=vf(tt);z=sol.y[:,-1]
        if ts:tt,zz,vv=tt[1:],zz[1:],vv[1:]
        ts.extend(tt+clock);zs.extend(zz);vs.extend(vv);kinds.extend([k]*len(tt))
        segments.append(dict(kind=k,left=clock,right=clock+dur,nfev=sol.nfev,steps=len(sol.t)-1));clock+=dur
        print(stem,k,clock,'seconds',round(time.perf_counter()-begin,1),flush=True)
    ds=[model.evaluate(y,v,True) for y,v in zip(zs,vs)]
    out=dict(time=np.array(ts),z=np.array(zs),V=np.array(vs),kind=np.array(kinds),edges=model.edges)
    for key in ('G','J','theta','sTi','cv','phi','E','oxygen_invariant','diss','thermodynamic_residual'):
        out[key]=np.array([d[key] for d in ds])
    np.savez(O/(stem+'.npz'),**out)
    stats=dict(parameters=asdict(model.p),local_cells=local,core_cells_0_to_half_nm=core,analytic_jacobian=analytic,outer_cells_per_nm=outer,mode=mode,rtol=rtol,segments=segments,
        source_sha256=fingerprint(src),source_index=int(idx) if idx is not None else None,
        source_time=float(a['time'][idx]) if idx is not None else None,
        file_sha256=fingerprint(O/(stem+'.npz')),script_sha256=fingerprint(Path(__file__)),core_sha256=fingerprint(R/'formed-channel-set-v3.py'),
        mass_drift=float(np.ptp(out['oxygen_invariant'])),theta_range=[float(out['theta'].min()),float(out['theta'].max())],
        diss_min=float(out['diss'].min()),thermodynamic_residual=float(np.max(abs(out['thermodynamic_residual']))),
        model_parameters_changed=False,electronic_quadrature=('legacy uniform discrete Gaussian' if mode=='legacy' else 'double-cell-integrated reflected Gaussian, fixed 0.25nm width'))
    assert stats['mass_drift']<1e-6 and 0<stats['theta_range'][0]<stats['theta_range'][1]<1 and stats['diss_min']>-1e-12
    (O/(stem+'.json')).write_text(json.dumps(stats,indent=2));print('SAVED',stem,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--local',type=int,default=128);ap.add_argument('--outer',type=int,default=16)
    ap.add_argument('--mode',choices=['reset','full','legacy'],default='reset');ap.add_argument('--rtol',type=float,default=1e-8)
    ap.add_argument('--analytic',action='store_true');ap.add_argument('--core',type=int,default=0);a=ap.parse_args()
    if a.analytic and a.mode=='legacy':ap.error('Analytic Jacobian is implemented only for the adaptive quadrature.')
    run(a.local,a.outer,a.mode,a.rtol,a.analytic,a.core)
