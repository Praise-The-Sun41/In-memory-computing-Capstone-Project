"""Conservative two-pool Ti interface hypothesis on the frozen thermal model.

Local binding is a reversible coarse-grained reaction, NOT an identified TiO2
phase. Capacities partition existing Ti sites, rather than adding oxygen sites.
No fitted parameters, imposed per-cycle damage, or zero reverse reaction rate.
"""
from pathlib import Path
from dataclasses import dataclass, asdict, replace
from contextlib import contextmanager
import importlib.util, sys, json, hashlib, time, argparse
import numpy as np
from scipy.special import expit, logit, xlogy
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from scipy.sparse import diags, csc_matrix

P=Path(__file__).resolve().parent
BASE=P.parent/'electrothermal_2026-10-10'
spec=importlib.util.spec_from_file_location('chemistry_private_thermal',BASE/'coupled.py')
et=importlib.util.module_from_spec(spec);sys.modules[spec.name]=et;spec.loader.exec_module(et)
KB=et.KB
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()

@dataclass(frozen=True)
class Chemistry:
    interface_depth_nm:float=1.
    bound_site_fraction:float=.25
    binding_delta_g_eV:float=-.08
    capture_rate_per_tau:float=.1
    capture_Ea_eV:float=.45
    passivation_barrier_eV:float=0.

CASES={
    'baseline':Chemistry(bound_site_fraction=0.,capture_rate_per_tau=0.),
    'partition_only':Chemistry(capture_rate_per_tau=0.),
    'binding':Chemistry(),
    'static_barrier':Chemistry(capture_rate_per_tau=0.,passivation_barrier_eV=.10),
    'binding_barrier':Chemistry(passivation_barrier_eV=.10),
}

class Device:
    def __init__(self,chem=Chemistry(),mode='thermal',L=256,M=256):
        if not (0<=chem.bound_site_fraction<1 and 0<chem.interface_depth_nm<=30
                and chem.capture_rate_per_tau>=0 and chem.capture_Ea_eV>=max(0,chem.binding_delta_g_eV)
                and chem.passivation_barrier_eV>=0):raise ValueError('Invalid chemistry parameters')
        self.chem=chem;self.base=et.Device('Ti',mode,L=L,M=M)
        d=self.base;metal=d.core.metal
        self.no=d.no;self.n=d.n;self.nb0=len(d.initial);self.Tindex=d.n
        self.Ctotal=metal.C.copy()
        overlap=np.maximum(0.,np.minimum(metal.edges[1:],chem.interface_depth_nm)-metal.edges[:-1])
        cb=chem.bound_site_fraction*metal.p.density_per_nm*overlap
        self.active=np.flatnonzero(cb>0);self.Cb=cb[self.active];self.Cm=self.Ctotal-cb
        self.nb=len(self.active);self.mi=self.no+self.active
        self.bi=self.nb0+np.arange(self.nb)
        self.weights=self.Cb/self.Cb.sum() if self.nb else np.array([])
        initial=d.initial.copy();binit=[]
        # Preserve each cell's original oxygen inventory; partition at local
        # reaction equilibrium at 300 K, not an initially empty trap reservoir.
        for j in self.active:
            frac=cb[j]/self.Ctotal[j];total=initial[self.no+j]
            def bound(s):return expit(logit(s)-chem.binding_delta_g_eV/(KB*300))
            s=brentq(lambda s:(1-frac)*s+frac*bound(s)-total,1e-14,1-1e-14,xtol=1e-14)
            initial[self.no+j]=s;binit.append(bound(s))
        if self.nb:
            # Positive face mobility and ideal-site chemical potentials retain
            # detailed balance even across changes of mobile site density.
            mobile_fraction=self.Cm/self.Ctotal
            ff=2*mobile_fraction[:-1]*mobile_fraction[1:]/(mobile_fraction[:-1]+mobile_fraction[1:])
            K=metal.K*ff
            diagonal=-np.r_[K[0],K[:-1]+K[1:],K[-1]]/self.Cm
            metal.C=self.Cm.copy();metal.K=K
            metal.base_jac=diags([K/self.Cm[1:],diagonal,K/self.Cm[:-1]],[-1,0,1],format='csc')
            d.mref=metal.base_jac.copy()
        self.initial=np.r_[initial,binit]
        self.original_pref=d.pref

    def barrier_factor(self,b,T):
        return float(np.exp(-self.chem.passivation_barrier_eV*(self.weights@b)/(KB*T))) if self.nb else 1.

    @contextmanager
    def exchange_context(self,b,T,log_shift=0.):
        old=self.base.pref
        self.base.pref=replace(self.original_pref,Ti_exchange_nm_per_tau=
            self.original_pref.Ti_exchange_nm_per_tau*self.barrier_factor(b,T)*np.exp(log_shift))
        try:yield
        finally:self.base.pref=old

    def reaction(self,s,b,T):
        p=self.chem
        nu=p.capture_rate_per_tau*np.exp(p.capture_Ea_eV/KB*(1/300-1/T))
        reverse=np.exp(p.binding_delta_g_eV/(KB*T))
        forward=self.Cb*nu*s*(1-b)
        backward=self.Cb*nu*reverse*b*(1-s)
        return forward-backward,forward,backward,nu,reverse

    def rhs(self,t,y,Vs):
        z=y[:self.nb0];b=y[self.nb0:];T=z[self.Tindex]
        with self.exchange_context(b,T):out=self.base.rhs(t,z,Vs)
        j,*_=self.reaction(z[self.mi],b,T)
        out[self.mi]-=j/self.Cm[self.active]
        return np.r_[out,j/self.Cb]

    def jac(self,t,y,Vs):
        z=y[:self.nb0];b=y[self.nb0:];T=z[self.Tindex]
        with self.exchange_context(b,T):old=self.base.jac(t,z,Vs).toarray()
        a=np.zeros((len(y),len(y)));a[:self.nb0,:self.nb0]=old
        if self.nb:
            s=z[self.mi];_,_,_,nu,r=self.reaction(s,b,T)
            js=self.Cb*nu*((1-b)+r*b)
            jb=self.Cb*nu*(-s-r*(1-s))
            a[self.mi,self.mi]-=js/self.Cm[self.active]
            a[self.mi,self.bi]-=jb/self.Cm[self.active]
            a[self.bi,self.mi]=js/self.Cb
            a[self.bi,self.bi]=jb/self.Cb
            if self.chem.passivation_barrier_eV:
                h=1e-4
                with self.exchange_context(b,T,h):plus=self.base.rhs(t,z,Vs)
                with self.exchange_context(b,T,-h):minus=self.base.rhs(t,z,Vs)
                a[:self.nb0,self.nb0:]+=np.outer((plus-minus)/(2*h),
                    -self.chem.passivation_barrier_eV/(KB*T)*self.weights)
        h=1e-3;yp=y.copy();ym=y.copy();yp[self.Tindex]+=h;ym[self.Tindex]-=h
        a[:,self.Tindex]=(self.rhs(t,yp,Vs)-self.rhs(t,ym,Vs))/(2*h)
        return csc_matrix(a)

    def diagnostic(self,y,Vs):
        z=y[:self.nb0];b=y[self.nb0:];T=z[self.Tindex]
        with self.exchange_context(b,T):out=self.base.diagnostic(z,Vs)
        j,fp,bp,_,_=self.reaction(z[self.mi],b,T)
        affinity=-self.chem.binding_delta_g_eV+KB*T*(logit(z[self.mi])-logit(b))
        out.update(inventory=out['inventory']-float(self.Cb@b),
            bound_oxygen=float(self.Cb@b),mobile_oxygen=float(self.Cm@z[self.no:self.n]),
            bound_fraction=float(self.weights@b) if self.nb else 0.,
            exchange_factor=self.barrier_factor(b,T),binding_flux=float(j.sum()),
            binding_dissipation=float(j@affinity),
            forward_binding=float(fp.sum()),reverse_binding=float(bp.sum()))
        return out

    def state_G(self,y):
        y=y.copy();y[self.Tindex]=300.
        return self.diagnostic(y,0.)['G']

def solve(m,rows,y0,name,rtol=2e-9):
    path=P/(name+'.npz');metadata=P/(name+'.json')
    config=dict(chemistry=asdict(m.chem),thermal=asdict(m.base.cfg),mode=m.base.mode,
                L=m.base.core.local,M=len(m.Cm),rtol=rtol,rows=rows,
                source_sha=sha(__file__),thermal_source_sha=sha(BASE/'coupled.py'),
                initial_sha=hashlib.sha256(y0.tobytes()).hexdigest())
    if path.exists():
        meta=json.loads(metadata.read_text())
        assert meta['config']==json.loads(json.dumps(config)), 'Cache configuration mismatch'
        assert meta['data_sha']==sha(path)
        return dict(np.load(path)),meta
    y=y0.copy();parts=[];stats=[];t0=0.;tic=time.perf_counter();node_mass=0.;node_min=1.;node_max=0.
    inv0=m.diagnostic(y0,rows[0][1])['inventory']
    for duration,v0,v1,label,shape in rows:
        def voltage(t):
            u=t/duration;return v0+(v1-v0)*(u*u*(3-2*u) if shape=='smooth' else u)
        sol=solve_ivp(lambda t,z:m.rhs(t,z,voltage(t)),(0,duration),y,method='BDF',
            jac=lambda t,z:m.jac(t,z,voltage(t)),rtol=rtol,atol=rtol*1e-3,
            max_step=.05 if duration<10 else .5,dense_output=True)
        if not sol.success:raise RuntimeError(sol.message)
        occ=np.r_[expit(sol.y[:m.no]).ravel(),sol.y[m.no:m.n].ravel(),sol.y[m.nb0:].ravel()]
        node_min=min(node_min,float(occ.min()));node_max=max(node_max,float(occ.max()))
        iv=m.base.ox.C@expit(sol.y[:m.no])-m.Cm@sol.y[m.no:m.n]-m.Cb@sol.y[m.nb0:]
        node_mass=max(node_mass,float(abs(iv-inv0).max()))
        assert 0<node_min and node_max<1 and node_mass<2e-6
        tt=np.linspace(0,duration,121);sl=slice(None) if not parts else slice(1,None)
        parts.append(dict(t=t0+tt[sl],y=sol.sol(tt[sl]).T,Vs=voltage(tt[sl]),kind=np.full(len(tt[sl]),label)))
        stats.append(dict(label=label,steps=len(sol.t),nfev=sol.nfev,njev=sol.njev))
        y=sol.y[:,-1];t0+=duration
        print(name,label,round(time.perf_counter()-tic,1),flush=True)
    out={k:np.concatenate([v[k] for v in parts]) for k in parts[0]}
    ds=[m.diagnostic(y,v) for y,v in zip(out['y'],out['Vs'])]
    out.update({k:np.array([d[k] for d in ds]) for k in ds[0]})
    out['G300']=np.array([m.state_G(y) for y in out['y']])
    checks=dict(internal_node_mass_error=node_mass,internal_node_minimum=node_min,internal_node_maximum=node_max,
        sampled_mass_error=float(abs(out['inventory']-inv0).max()),
        thermal_balance_Wtau=float(abs(out['thermal_balance']).max()),
        power_balance_W=float(max(abs(out['power_balance']).max(),abs(out['device_power_balance']).max())),
        voltage_balance_V=float(abs(out['voltage_balance']).max()),
        current_excess_A=float(max(0,abs(out['I']).max()-m.base.Icc)),
        minimum_binding_dissipation=float(out['binding_dissipation'].min()))
    assert checks['thermal_balance_Wtau']<1e-8 and checks['power_balance_W']<1e-12
    assert checks['voltage_balance_V']<1e-10 and checks['current_excess_A']<1e-12
    assert checks['minimum_binding_dissipation']>-1e-13
    np.savez_compressed(path,**out)
    meta=dict(config=config,checks=checks,stats=stats,data_sha=sha(path),elapsed_s=time.perf_counter()-tic,
        COMSOL_verified=False,experiment_calibrated=False,phase_identified=False,
        thermal_scope='Electronic Joule heat only; reaction heat not converted without an absolute oxygen amount scale.')
    metadata.write_text(json.dumps(meta,indent=2));return out,meta

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',choices=CASES,required=True)
    p.add_argument('--cycles',type=int,default=10);p.add_argument('--rtol',type=float,default=2e-9)
    p.add_argument('--L',type=int,default=256);p.add_argument('--M',type=int,default=256)
    a=p.parse_args();m=Device(CASES[a.case],L=a.L,M=a.M)
    stem=f'{a.case}-C{a.cycles}-L{a.L}-M{a.M}-R{a.rtol:g}'
    solve(m,et.protocol(a.cycles),m.initial,stem,a.rtol)
