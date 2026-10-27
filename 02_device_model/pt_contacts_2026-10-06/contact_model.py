"""Back-to-back thermionic contact boundary extension of the frozen Pt model.

Contact saturation currents are scenario inputs, not fitted Pt properties.
No Poisson/depletion, image-force lowering, tunneling or oxygen exchange.
"""
from pathlib import Path
import importlib.util,sys,json,hashlib,argparse,time
from dataclasses import dataclass,asdict
import numpy as np
from scipy.special import expit,logit
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix
P=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('contacts_pt_base',P.parent/'pt_blocking_2026-10-06/pt_model.py')
base=importlib.util.module_from_spec(s);sys.modules[s.name]=base;s.loader.exec_module(base)
f=base.b.f;KT=f.KT

@dataclass(frozen=True)
class Contacts:
    # Sum of reference zero-bias contact resistances / virgin oxide resistance.
    resistance_ratio:float=.01
    bottom_share:float=.2
    vacancy_lowering_eV:float=.10
    averaging_depth_nm:float=.5

CASES={'weak':Contacts(1e-4),'medium':Contacts(.01),'strong':Contacts(1.),
       'medium_fixed':Contacts(.01,vacancy_lowering_eV=0.)}

def pair(vc,Isb,Ist):
    """Two oppositely oriented ideal contacts; common current, stable voltages."""
    u=vc/KT;den=Isb*np.exp(u)+Ist
    I=Isb*Ist*np.expm1(u)/den
    p=Isb*np.exp(u)/den
    derivative=Isb*Ist*(Isb+Ist)*np.exp(u)/(den*den*KT)
    logden=np.logaddexp(np.log(Isb)+u,np.log(Ist))
    logs=np.logaddexp(np.log(Isb),np.log(Ist))
    vb=KT*(logs+u-logden);vt=KT*(logden-logs)
    return I,derivative,p,vb,vt

class ContactOxide(f.Oxide):
    def __init__(self,old,cp):
        super().__init__(old.p,old.edges,old.mode);self.cp=cp
        self.wb=np.maximum(0,np.minimum(self.edges[1:],cp.averaging_depth_nm)-self.edges[:-1])/cp.averaging_depth_nm
        assert abs(self.wb.sum()-1)<1e-12 and 0<cp.bottom_share<1
        self.theta_ref=float(self.wb@expit(self.initial[:-1]))
        self.Rref=super().electrical(self.initial,0)['R']
        self.Isb_ref=KT/(self.Rref*cp.resistance_ratio*cp.bottom_share)
        self.Ist_ref=KT/(self.Rref*cp.resistance_ratio*(1-cp.bottom_share))

    def electrical(self,y,V):
        e=super().electrical(y,0.);cp=self.cp
        dphi=-cp.vacancy_lowering_eV*(self.wb@e['theta']-self.theta_ref)
        Isb=self.Isb_ref*np.exp(-dphi/KT);Ist=self.Ist_ref
        if V==0:vc=0.
        else:vc=brentq(lambda c:c+e['R']*pair(c,Isb,Ist)[0]-V,min(0,V),max(0,V),xtol=1e-22,rtol=1e-14)
        I,di,p,vb,vt=pair(vc,Isb,Ist)
        drop=I*self.h/e['sigma'];faces=np.r_[0.,np.cumsum(drop)]
        e.update(J=I,drop=drop,faces=faces,phi=(faces[:-1]+faces[1:])/2,E=-drop/self.h,
                 G=1/(e['R']+KT/Isb+KT/Ist),bulkG=1/e['R'],vc=vc,vb=vb,vt=vt,
                 Isb=Isb,Ist=Ist,di_dvc=di,contact_p=p,barrier_change_eV=dphi)
        return e

    def evaluate(self,y,V,diagnostics=False):
        e=super().evaluate(y,V,diagnostics)
        if diagnostics:
            e['bulk_joule']=e['electronic_dissipation']
            e['contact_power']=e['J']*(e['vb']+e['vt'])
            e['electronic_dissipation']+=e['contact_power']
            e['voltage_sum_residual']=float(e['drop'].sum()+e['vb']+e['vt']-V)
        return e

    def jacobian(self,y,V):
        n=self.N;e=self.electrical(y,V);th=e['theta'];dt=th*(1-th)
        avg=self.W@(self.S*th+self.bound);spacing=(50*avg)**(-1/3)
        sigprime=(e['sigma']-1e-12)/avg*(1+2*spacing/(3*self.xi))
        drr=-(self.h/e['sigma']**2*sigprime)[:,None]*self.W*(self.S*dt)[None,:]
        # Implicit differentiation of V = Rbulk*I + Vcontacts.
        dlogIsb=self.cp.vacancy_lowering_eV/KT*self.wb*dt
        Iparam=e['J']*(1-e['contact_p'])*dlogIsb
        Igrad=(Iparam-e['di_dvc']*e['J']*drr.sum(0))/(1+e['R']*e['di_dvc'])
        dg=e['J']*drr+(self.h/e['sigma'])[:,None]*Igrad
        facesgrad=np.vstack([np.zeros(n),np.cumsum(dg,axis=0)])
        phigrad=(facesgrad[:-1]+facesgrad[1:])/2
        dx=np.diff(self.x);delta=(np.diff(self.G)+2*np.diff(e['phi']))/KT
        work,g,M,lp=self.field_mobility(e)
        Df=2*self.D[:-1]*self.D[1:]/(self.D[:-1]+self.D[1:])
        pref=Df*np.sqrt(self.S[:-1]*self.S[1:])/dx*M
        bp=self.bernoulli(delta);bm=self.bernoulli(-delta)
        flux=pref*(bp*th[:-1]*(1-th[1:])-bm*th[1:]*(1-th[:-1]))
        ap=pref*(self.bernoulli_derivative(delta)*th[:-1]*(1-th[1:])+self.bernoulli_derivative(-delta)*th[1:]*(1-th[:-1]))*2/KT
        fg=ap[:,None]*np.diff(phigrad,axis=0)
        fg+=(flux*lp*np.sign(work)/KT)[:,None]*(self.span@facesgrad)
        if self.p.high_field_hopping:
            localg=abs(np.diff(e['phi']))*self.p.lattice_hop_nm/(dx*KT)
            original,_=f.mobility(g,self.p.hop_barrier_eV/KT)
            active=original>f.drift_upper(localg,self.p.hop_barrier_eV/KT)
            fg[active]-=(flux[active]/np.diff(e['phi'])[active])[:,None]*np.diff(phigrad,axis=0)[active]
        ii=np.arange(n-1)
        fg[ii,ii]+=pref*(bp*(1-th[1:])+bm*th[1:])*dt[:-1]
        fg[ii,ii+1]-=pref*(bp*th[:-1]+bm*(1-th[:-1]))*dt[1:]
        rates=-np.diff(np.r_[0.,flux,0.]);rg=-np.diff(np.vstack([np.zeros(n),fg,np.zeros(n)]),axis=0)
        jac=np.zeros((n+1,n+1));jac[:n,:n]=rg/(self.C*dt)[:,None]
        jac[np.arange(n),np.arange(n)]-=rates/(self.C*dt)*(1-2*th)
        return jac

class Model(base.PtModel):
    def __init__(self,cp=Contacts()):
        super().__init__();self.ox=ContactOxide(self.ox,cp)
    def diag(self,y,V):
        d=self.ox.evaluate(self.q(y),V,True)
        return dict(G=d['G'],bulkG=d['bulkG'],I=d['J'],N=float(self.ox.C@d['theta']),
            Vbottom=d['vb'],Vtop=d['vt'],Voxide=float(d['drop'].sum()),
            TiOx_drop=float(d['drop'][self.ox.ti].sum()),AlOx_drop=float(d['drop'][~self.ox.ti].sum()),
            power=d['electronic_power'],joule=d['electronic_dissipation'],bulk_joule=d['bulk_joule'],contact_power=d['contact_power'],
            barrier_change_eV=d['barrier_change_eV'],Is_bottom=d['Isb'],Is_top=d['Ist'],
            voltage_residual=d['voltage_sum_residual'],diss=d['diss'],energy_residual=d['thermodynamic_residual'],exchange=d['exchange'])

def fingerprint(m):
    files=[Path(__file__),P.parent/'pt_blocking_2026-10-06/pt_model.py',P.parent/'bounded-span-field.py',P.parent/'local-reset-mesh.py']
    return dict(parameters=asdict(m.ox.cp),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})

def solve(m,y,rows,name,rtol=1e-9):
    file=P/(name+'.npz');meta=P/(name+'.json');fp=fingerprint(m)
    if file.exists():
        assert json.loads(meta.read_text())['fingerprint']==fp
        return dict(np.load(file))
    t0=0.;parts=[];audits=[];tic=time.perf_counter();initialN=m.ox.C@expit(y)
    for T,a,v,label,shape in rows:
        vf=lambda t,a=a,v=v,T=T,shape=shape:a+(v-a)*((t/T)**2*(3-2*t/T) if shape=='smooth' else t/T)
        sol=solve_ivp(lambda t,z:m.rhs(t,z,vf(t)),(0,T),y,method='BDF',jac=lambda t,z:m.jac(t,z,vf(t)),rtol=rtol,atol=.01*rtol,dense_output=True)
        if not sol.success:raise RuntimeError(sol.message)
        th=expit(sol.y);mass=float(abs(m.ox.C@th-initialN).max());assert mass<1e-6 and np.all((th>0)&(th<1))
        tt=np.r_[0,np.geomspace(.001,T,120)] if T>100 else np.linspace(0,T,121)
        yy=sol.sol(tt).T;sl=slice(None) if not parts else slice(1,None)
        parts.append(dict(t=tt[sl]+t0,y=yy[sl],V=vf(tt[sl]),kind=np.full(len(tt[sl]),label)))
        y=sol.y[:,-1];t0+=T;audits.append(dict(label=label,mass_error=mass,steps=len(sol.t)-1))
        print(name,label,round(time.perf_counter()-tic,1),flush=True)
    out={k:np.concatenate([x[k] for x in parts]) for k in parts[0]}
    ds=[m.diag(y,v) for y,v in zip(out['y'],out['V'])]
    for k in ds[0]:out[k]=np.asarray([d[k] for d in ds])
    out['edges']=m.ox.edges
    assert out['diss'].min()>-1e-9 and abs(out['energy_residual']).max()<1e-9
    assert abs(out['voltage_residual']).max()<1e-10
    assert abs(out['power']-out['joule']).max()<max(abs(out['power']).max(),1e-30)*1e-9
    assert out['contact_power'].min()>-1e-25 and np.all(out['exchange']==0)
    np.savez_compressed(file,**out)
    meta.write_text(json.dumps(dict(fingerprint=fp,rows=rows,audits=audits,rtol=rtol,physical_contact_calibration=False),indent=2))
    return out

def verify():
    rng=np.random.default_rng(47);errors=[];port=[]
    for cp in [CASES['weak'],CASES['medium'],CASES['strong'],Contacts(.01,bottom_share=.8)]:
        m=Model(cp);y=m.initial+rng.normal(0,.1,m.n);u=rng.normal(size=m.n);u/=np.linalg.norm(u)
        for V in [0.,.01,.37,-.63,1.,-1.]:
            h=2e-6;fd=(m.rhs(0,y+h*u,V)-m.rhs(0,y-h*u,V))/(2*h);ja=m.jac(0,y,V)@u
            errors.append(float(np.linalg.norm(fd-ja)/max(np.linalg.norm(ja),1e-25)))
            e=m.ox.electrical(m.q(y),V);I=e['J']
            rb=I-e['Isb']*np.expm1(e['vb']/KT);rt=I+e['Ist']*np.expm1(-e['vt']/KT)
            port.append(float(max(abs(rb),abs(rt))/max(e['Isb'],e['Ist'])))
            assert I*V>=-1e-25 and abs(e['vc']+I*e['R']-V)<1e-10
            assert abs((m.ox.C*expit(y)*(1-expit(y)))@m.rhs(0,y,V))<1e-10
        h=1e-7;g=(m.diag(y,h)['I']-m.diag(y,-h)['I'])/(2*h)
        assert abs(g/m.diag(y,0)['G']-1)<1e-5
    assert max(errors)<3e-5 and max(port)<1e-10
    ideal=base.PtModel();tiny=Model(Contacts(1e-9,vacancy_lowering_eV=0.))
    rel=float(abs(tiny.diag(tiny.initial,.6)['I']/ideal.diag(ideal.initial,.6)['I']-1))
    assert rel<1e-7
    out=dict(jacobian_relative=max(errors),contact_current_continuity_scaled=max(port),ideal_contact_limit_relative=rel,all_pass=True)
    (P/'unit-checks.json').write_text(json.dumps(out,indent=2));print(out)

def run(case):
    m=Model(CASES[case]);d=solve(m,m.initial,base.protocol(),case+'-history')
    for sign in ['pos','neg']:
        idx=np.flatnonzero(d['kind']==f'C5-{sign}-fall')[-1]
        solve(m,d['y'][idx].copy(),[(10,0,0,'hold10','linear'),(90,0,0,'hold100','linear'),(900,0,0,'hold1000','linear')],case+'-hold-'+sign)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['verify','run']);a.add_argument('--case',choices=CASES,default='medium');args=a.parse_args()
    verify() if args.action=='verify' else run(args.case)
