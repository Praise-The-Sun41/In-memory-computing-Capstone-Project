"""Lumped-temperature + external circuit extension; all dimensional scales are scenarios.

Frozen source files are never edited. Each evaluation temporarily updates the
private imported legacy module graph's kBT and transport coefficients, restoring
them on exit. Synchronous, single-thread evaluation only (one process per run).
This is NOT a spatial heat FEM or an irreversible aging model.
"""
from pathlib import Path
from dataclasses import dataclass, asdict, replace
from contextlib import contextmanager
import importlib.util, sys, json, hashlib, argparse, time
import numpy as np
from scipy.special import expit
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix

P=Path(__file__).resolve().parent; R=P.parent
sp=importlib.util.spec_from_file_location('et_private_contacts',R/'pt_contacts_2026-10-06/contact_model.py')
ct=importlib.util.module_from_spec(sp);sys.modules[sp.name]=ct;sp.loader.exec_module(ct)
f=ct.f; KB=8.617333262145e-5; TREF=300.

@dataclass(frozen=True)
class Settings:
    bath_K:float=300.
    reference_G_S:float=100e-6
    series_ohm:float=1000.
    compliance_A:float=300e-6
    thermal_R_K_W:float=1e5
    thermal_tau:float=.05
    oxide_migration_Ea_eV:float=.65
    metal_migration_Ea_eV:float=.5
    exchange_Ea_eV:float=.4
    contact_reference_barrier_eV:float=.5

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

class Device:
    def __init__(self,kind='Ti',mode='thermal',settings=Settings(),L=256,M=256):
        assert kind in ('Ti','Pt','Pt_contact') and mode in ('original','limited','thermal')
        self.kind=kind;self.mode=mode;self.cfg=settings
        self.core=ct.base.b.make_model('ref',L,M) if kind=='Ti' else ct.base.PtModel(L)
        if kind=='Pt_contact':self.core.ox=ct.ContactOxide(self.core.ox,ct.Contacts())
        self.ox=self.core.ox;self.n=len(self.core.initial);self.no=self.ox.N
        self.initial=np.r_[self.core.initial,settings.bath_K,0.,0.]
        # One common dimensional mapping, anchored to the saved Ti C0 conductance.
        anchor=R/'outputs/bounded-comsol-linear10-L256-sparse-raw.csv'
        with anchor.open(encoding='utf-8-sig') as stream:
            row=next(line for line in stream if not line.startswith('%'))
        self.Ganchor=float(row.split(',')[2]);self.scale=settings.reference_G_S/self.Ganchor
        self.Rs=0 if mode=='original' else settings.series_ohm
        self.Icc=np.inf if mode=='original' else settings.compliance_A
        self.Dref=self.ox.D.copy();self.pref=self.ox.p
        self.mods=[ct,f,f.d,f.d.c,f.d.c.w.m,f.d.c.ti]
        if kind=='Ti':self.mref=self.core.metal.base_jac.copy()
        if kind=='Pt_contact':self.Isref=(self.ox.Isb_ref,self.ox.Ist_ref)

    @contextmanager
    def at_temperature(self,T):
        if not 150<T<1500:raise ValueError(f'Temperature outside scenario safety domain: {T}')
        old=[m.KT for m in self.mods]; oldD=self.ox.D;oldp=self.ox.p
        factor=lambda E:np.exp(E/KB*(1/TREF-1/T))
        try:
            for m in self.mods:m.KT=KB*T
            self.ox.D=self.Dref*factor(self.cfg.oxide_migration_Ea_eV)
            self.ox.p=replace(self.pref,Ti_exchange_nm_per_tau=self.pref.Ti_exchange_nm_per_tau*factor(self.cfg.exchange_Ea_eV))
            if self.kind=='Ti':self.core.metal.base_jac=self.mref*factor(self.cfg.metal_migration_Ea_eV)
            if self.kind=='Pt_contact':
                fac=(T/TREF)**2*factor(self.cfg.contact_reference_barrier_eV)
                self.ox.Isb_ref,self.ox.Ist_ref=[v*fac for v in self.Isref]
            yield
        finally:
            for m,k in zip(self.mods,old):m.KT=k
            self.ox.D=oldD;self.ox.p=oldp
            if self.kind=='Ti':self.core.metal.base_jac=self.mref
            if self.kind=='Pt_contact':self.ox.Isb_ref,self.ox.Ist_ref=self.Isref

    def port(self,x):return self.core.port(x) if self.kind=='Ti' else self.core.q(x)

    def circuit(self,x,Vs):
        # Current through physical series resistance and oxide, with ideal SMU CC.
        q=self.port(x);e=f.d.w.Model.electrical(self.ox,q,0.)
        rb=e['R']/self.scale; vc=0.; vb=0.;vt=0.
        if self.kind=='Pt_contact':
            cp=self.ox.cp;dphi=-cp.vacancy_lowering_eV*(self.ox.wb@e['theta']-self.ox.theta_ref)
            ib=self.ox.Isb_ref*np.exp(-dphi/ct.KT);it=self.ox.Ist_ref
            current=lambda v:self.scale*ct.pair(v,ib,it)[0]
            if Vs:vc=brentq(lambda v:v+(rb+self.Rs)*current(v)-Vs,min(0.,Vs),max(0.,Vs),xtol=1e-15)
            I=current(vc)
            if abs(I)>self.Icc:
                I=np.copysign(self.Icc,I)
                vc=brentq(lambda v:current(v)-I,min(0.,Vs),max(0.,Vs),xtol=1e-15)
            _,_,_,vb,vt=ct.pair(vc,ib,it)
        else:I=np.clip(Vs/(rb+self.Rs),-self.Icc,self.Icc)
        Vdev=I*rb+vc;Vout=Vdev+I*self.Rs
        limited=abs(I)>=self.Icc*(1-1e-10)
        return dict(I=float(I),Vdev=float(Vdev),Vout=float(Vout),Vsource=float(Vs),
                    Voxide=float(I*rb),Vcontact=float(vc),Vbottom=float(vb),Vtop=float(vt),
                    Pdev=float(I*Vdev),Pbulk=float(I*I*rb),Pcontact=float(I*vc),
                    Pseries=float(I*I*self.Rs),limited=bool(limited))

    def rhs(self,t,y,Vs):
        x=y[:self.n];T=y[self.n]
        with self.at_temperature(T):
            e=self.circuit(x,Vs);dx=self.core.rhs(t,x,e['Vdev'])
        loss=(T-self.cfg.bath_K)/self.cfg.thermal_R_K_W
        dT=(e['Pdev']-loss)*self.cfg.thermal_R_K_W/self.cfg.thermal_tau if self.mode=='thermal' else 0.
        # In isothermal controls an ideal thermostat removes all Joule power.
        return np.r_[dx,dT,e['Pdev'],loss if self.mode=='thermal' else e['Pdev']]

    def jac(self,t,y,Vs):
        x=y[:self.n];T=y[self.n];n=self.n
        with self.at_temperature(T):
            e=self.circuit(x,Vs);v=e['Vdev'];q=self.port(x);d=self.ox.electrical(q,v)
            th=d['theta'];dt=th*(1-th);avg=self.ox.W@(self.ox.S*th+self.ox.bound)
            spacing=(50*avg)**(-1/3)
            sigp=(d['sigma']-1e-12)/avg*(1+2*spacing/(3*self.ox.xi))
            dr=np.sum(-(self.ox.h/d['sigma']**2*sigp)[:,None]*self.ox.W,axis=0)*self.ox.S*dt
            iy=np.zeros(n)
            if self.kind=='Pt_contact':
                iv=d['di_dvc']/(1+d['di_dvc']*d['R'])*self.scale
                ip=d['J']*(1-d['contact_p'])*self.ox.cp.vacancy_lowering_eV/ct.KT*self.ox.wb*dt
                iy[:self.no]=(ip-d['di_dvc']*d['J']*dr)/(1+d['di_dvc']*d['R'])*self.scale
            else:
                iv=self.scale/d['R'];iy[:self.no]=-e['I']/d['R']*dr
            vy=-iy/iv if e['limited'] else -self.Rs*iy/(1+self.Rs*iv)
            dv=1e-6
            fv=(self.core.rhs(t,x,v+dv)-self.core.rhs(t,x,v-dv))/(2*dv)
            jj=self.core.jac(t,x,v)
            jj=jj.toarray() if hasattr(jj,'toarray') else jj
            a=np.zeros((n+3,n+3));a[:n,:n]=jj+np.outer(fv,vy)
            py=e['Vdev']*iy+(e['I']+e['Vdev']*iv)*vy
            a[n+1,:n]=py
            if self.mode=='thermal':
                a[n,:n]=py*self.cfg.thermal_R_K_W/self.cfg.thermal_tau
                a[n+2,n]=1/self.cfg.thermal_R_K_W
            else:a[n+2,:n]=py
        h=1e-3;yp=y.copy();ym=y.copy();yp[n]+=h;ym[n]-=h
        a[:,n]=(self.rhs(t,yp,Vs)-self.rhs(t,ym,Vs))/(2*h)
        return csc_matrix(a)

    def diagnostic(self,y,Vs):
        x=y[:self.n];T=y[self.n]
        with self.at_temperature(T):
            e=self.circuit(x,Vs);d=self.ox.electrical(self.port(x),e['Vdev'])
            e.update(G=d['G']*self.scale,TiOx_drop=float(d['drop'][self.ox.ti].sum()),
                     AlOx_drop=float(d['drop'][~self.ox.ti].sum()),T=float(T))
        inv=float(self.ox.C@expit(x[:self.no]))
        if self.kind=='Ti':inv-=float(self.core.metal.C@x[self.no:])
        e.update(inventory=inv,Ti_surface=float(x[self.no]) if self.kind=='Ti' else np.nan,
                 thermal_balance=(self.cfg.thermal_tau/self.cfg.thermal_R_K_W*(T-self.cfg.bath_K)-(y[n:=self.n+1]-y[n+1])) if self.mode=='thermal' else float(y[-2]-y[-1]),
                 voltage_balance=e['Vout']-e['Vdev']-e['I']*self.Rs,
                 power_balance=e['I']*e['Vout']-e['Pdev']-e['Pseries'],
                 device_power_balance=e['Pdev']-e['Pbulk']-e['Pcontact'])
        return e

def protocol(cycles=5):
    return ct.base.protocol()[:5]+[(2,a,b,f'C{k}-{label}','linear') for k in range(1,cycles+1)
        for a,b,label in [(0,1,'pos-rise'),(1,0,'pos-fall'),(0,-1,'neg-rise'),(-1,0,'neg-fall')]]

def solve(m,rows,y0,name,rtol=2e-8):
    outpath=P/(name+'.npz');metapath=P/(name+'.json')
    if outpath.exists():
        meta=json.loads(metapath.read_text());assert meta['source_sha']==sha(Path(__file__))
        return dict(np.load(outpath)),meta
    y=y0.copy();t0=0.;parts=[];stats=[];tic=time.perf_counter()
    for duration,a,b,label,shape in rows:
        def vf(t):
            u=t/duration;return a+(b-a)*(u*u*(3-2*u) if shape=='smooth' else u)
        sol=solve_ivp(lambda t,z:m.rhs(t,z,vf(t)),(0,duration),y,method='BDF',
                      jac=lambda t,z:m.jac(t,z,vf(t)),rtol=rtol,atol=rtol*1e-3,
                      max_step=.05 if duration<100 else 2.,dense_output=True)
        if not sol.success:raise RuntimeError(sol.message)
        tt=np.linspace(0,duration,121);sl=slice(None) if not parts else slice(1,None)
        parts.append(dict(t=tt[sl]+t0,y=sol.sol(tt[sl]).T,Vs=vf(tt[sl]),kind=np.full(len(tt[sl]),label)))
        y=sol.y[:,-1];t0+=duration
        stats.append(dict(label=label,steps=len(sol.t),nfev=sol.nfev,njev=sol.njev))
        print(name,label,round(time.perf_counter()-tic,1),flush=True)
    out={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    ds=[m.diagnostic(z,v) for z,v in zip(out['y'],out['Vs'])]
    out.update({k:np.array([d[k] for d in ds]) for k in ds[0]})
    initial_inv=m.diagnostic(y0,rows[0][1])['inventory']
    checks=dict(mass_error=float(np.max(abs(out['inventory']-initial_inv))),
        thermal_balance_Wtau=float(np.max(abs(out['thermal_balance']))),
        voltage_residual_V=float(np.max(abs(out['voltage_balance']))),
        power_residual_W=float(max(np.max(abs(out['power_balance'])),np.max(abs(out['device_power_balance'])))),
        maximum_current_A=float(max(abs(out['I']))),maximum_T_K=float(max(out['T'])),
        compliance_excess_A=float(max(0,max(abs(out['I']))-m.Icc)),
        minimum_occupancy=float(expit(out['y'][:,:m.no]).min()),maximum_occupancy=float(expit(out['y'][:,:m.no]).max()))
    assert checks['mass_error']<2e-6 and checks['thermal_balance_Wtau']<1e-9
    assert checks['power_residual_W']<1e-12 and checks['voltage_residual_V']<1e-10
    assert checks['compliance_excess_A']<1e-12 and checks['minimum_occupancy']>0 and checks['maximum_occupancy']<1
    if m.kind=='Ti':assert np.all((out['y'][:,m.no:m.n]>0)&(out['y'][:,m.no:m.n]<1))
    meta=dict(source_sha=sha(Path(__file__)),kind=m.kind,mode=m.mode,settings=asdict(m.cfg),
        current_mapping_S_per_model_G=m.scale,anchor_model_G=m.Ganchor,rtol=rtol,rows=rows,checks=checks,stats=stats,
        spatial_temperature=False,COMSOL_verified=False,experiment_calibrated=False,irreversible_aging=False,
        thermal_scope='Electronic Joule power only; neglect reaction heat and ionic heating. Uniform lumped device temperature.',
        thermal_capacity_units='Cth/t0 = thermal_tau/thermal_R, W tau / K; t0 seconds per tau remains unknown',
        elapsed_s=time.perf_counter()-tic)
    np.savez_compressed(outpath,**out);meta['data_sha']=sha(outpath)
    metapath.write_text(json.dumps(meta,indent=2));return out,meta

def checks():
    results=[];rng=np.random.default_rng(38)
    for kind in ('Ti','Pt','Pt_contact'):
        m=Device(kind);z=m.initial.copy();z[m.n]=317.
        for voltage in (.15,1.):
            u=rng.normal(size=len(z));u[:m.no]*=.1
            if kind=='Ti':u[m.no:m.n]*=1e-3
            u[m.n:]=[.3,0,0];u/=np.linalg.norm(u)
            h=1e-5;fd=(m.rhs(0,z+h*u,voltage)-m.rhs(0,z-h*u,voltage))/(2*h)
            jj=m.jac(0,z,voltage)@u;err=np.linalg.norm(fd-jj)/max(np.linalg.norm(fd),1e-30)
            assert err<2e-5,(kind,voltage,err)
            with m.at_temperature(317.):
                e=m.circuit(z[:m.n],voltage);actual=m.ox.electrical(m.port(z[:m.n]),e['Vdev'])['J']*m.scale
            assert abs(actual-e['I'])<1e-12
            results.append(dict(kind=kind,voltage=voltage,jac_relative=float(err)))
        iso=Device(kind,'original');old=iso.core.rhs(0,iso.core.initial,.4)
        new=iso.rhs(0,iso.initial,.4)[:iso.n]
        assert np.array_equal(old,new),kind
        # Exact frozen-resistance analytical lumped heating and subsequent cooling.
        power=1e-4;cfg=m.cfg;dur=.3
        sol=solve_ivp(lambda t,y:[(power-(y[0]-300)/cfg.thermal_R_K_W)*cfg.thermal_R_K_W/cfg.thermal_tau],
                      (0,dur),[300.],rtol=1e-11,atol=1e-11)
        exact=300+power*cfg.thermal_R_K_W*(1-np.exp(-dur/cfg.thermal_tau))
        assert abs(sol.y[0,-1]-exact)<1e-6
    result=dict(all_pass=True,jacobians=results,exact_300K_regression=True,frozen_thermal_analytic=True,
                checks_scope='RHS/Jacobian/circuit/unit checks; dynamic conservation separately checked on every run')
    (P/'unit_checks.json').write_text(json.dumps(result,indent=2));print(result,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['check','run']);p.add_argument('--kind',default='Ti')
    p.add_argument('--mode',default='thermal');p.add_argument('--cycles',type=int,default=5);p.add_argument('--rtol',type=float,default=2e-8)
    a=p.parse_args()
    if a.action=='check':checks()
    else:
        m=Device(a.kind,a.mode);solve(m,protocol(a.cycles),m.initial,f'{a.kind}-{a.mode}-C{a.cycles}-R{a.rtol:g}',a.rtol)
