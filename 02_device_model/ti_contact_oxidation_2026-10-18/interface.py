"""State-dependent resistive Ti interface with consistent redox voltage work.

This is a hypothesis test, not a calibrated Ti oxide phase or Schottky model.
Positive current is the inherited top-bias plotting convention. The oxide-side
bottom potential is Vc=I Rc relative to Ti. The vacancy charge is +2, so the
interfacial reaction free-energy shift is +2 Vc eV. Bulk transport sees Voxide.
Frozen chemistry/thermal kernels are imported, never modified.
"""
from pathlib import Path
from dataclasses import dataclass, asdict, replace
from contextlib import contextmanager
import importlib.util, sys, hashlib, json
import numpy as np
from scipy.special import expit, logit
from scipy.sparse import csc_matrix

P=Path(__file__).resolve().parent
OLD=P.parent/'ti_chemistry_2026-10-18'
sp=importlib.util.spec_from_file_location('interface_private_chemistry',OLD/'chemistry.py')
ch=importlib.util.module_from_spec(sp);sys.modules[sp.name]=ch;sp.loader.exec_module(ch)
et=ch.et;KB=ch.KB;sha=ch.sha

@dataclass(frozen=True)
class Contact:
    reference_ohm:float=500.
    chemical_sensitivity_eV:float=.2

CASES={
    'no_contact':Contact(0.,0.),
    'static_500':Contact(500.,0.),
    'chemical_500':Contact(500.,.2),
    'static_5000':Contact(5000.,0.),
    'chemical_5000':Contact(5000.,.2),
}

class Device(ch.Device):
    def __init__(self,contact=Contact(),reference_b=None,**kwargs):
        super().__init__(ch.CASES['binding_barrier'],**kwargs)
        if contact.reference_ohm<0 or contact.chemical_sensitivity_eV<0:
            raise ValueError('Nonnegative contact parameters required')
        self.contact=contact
        self.Bref=float(self.weights@self.initial[self.bi]) if reference_b is None else float(reference_b)
        metal=self.base.core.metal
        overlap=np.maximum(0.,np.minimum(metal.edges[1:],self.chem.interface_depth_nm)-metal.edges[:-1])
        self.region_total=metal.p.density_per_nm*overlap
        self.region_mobile=self.region_total.copy();self.region_mobile[self.active]-=self.Cb
        self.region_capacity=float(self.region_total.sum())

    def resistance(self,y):
        B=float(self.weights@y[self.bi]);T=y[self.Tindex]
        return self.contact.reference_ohm*np.exp(self.contact.chemical_sensitivity_eV*(B-self.Bref)/(KB*T))

    def electrical(self,y,Vs):
        # Caller must set the temperature context before computing oxide R.
        d=self.base; x=y[:self.n]
        e=et.f.d.w.Model.electrical(d.ox,d.port(x),0.)
        rb=e['R']/d.scale;rc=self.resistance(y);rt=rb+rc+d.Rs
        I=float(np.clip(Vs/rt,-d.Icc,d.Icc));vo=I*rb;vc=I*rc;vd=vo+vc
        return dict(I=I,Rbulk=rb,Rcontact=rc,Voxide=vo,Vcontact=vc,Vdev=vd,
            Vout=vd+I*d.Rs,Vsource=Vs,Vbottom=vc,Vtop=0.,
            Pdev=I*vd,Pbulk=I*vo,Pcontact=I*vc,Pseries=I*I*d.Rs,
            limited=bool(abs(I)>=d.Icc*(1-1e-10))),e

    @contextmanager
    def context(self,y,voltage_shift):
        with self.exchange_context(y[self.bi],y[self.Tindex]):
            p=self.base.pref
            self.base.pref=replace(p,Ti_reaction_standard_eV=p.Ti_reaction_standard_eV+2*voltage_shift)
            try:
                with self.base.at_temperature(y[self.Tindex]):yield
            finally:self.base.pref=p

    def core_rhs(self,t,y,vo,vc):
        with self.context(y,vc):return self.base.core.rhs(t,y[:self.n],vo)

    def rhs(self,t,y,Vs):
        with self.base.at_temperature(y[self.Tindex]):e,_=self.electrical(y,Vs)
        dx=self.core_rhs(t,y,e['Voxide'],e['Vcontact'])
        j,*_=self.reaction(y[self.mi],y[self.bi],y[self.Tindex])
        dx[self.mi]-=j/self.Cm[self.active]
        cfg=self.base.cfg;loss=(y[self.Tindex]-cfg.bath_K)/cfg.thermal_R_K_W
        dT=(e['Pdev']-loss)*cfg.thermal_R_K_W/cfg.thermal_tau if self.base.mode=='thermal' else 0.
        return np.r_[dx,dT,e['Pdev'],loss if self.base.mode=='thermal' else e['Pdev'],j/self.Cb]

    def jac(self,t,y,Vs):
        d=self.base;n=len(y);T=y[self.Tindex];a=np.zeros((n,n))
        with d.at_temperature(T):
            e,bulk=self.electrical(y,Vs)
            th=bulk['theta'];dt=th*(1-th);avg=d.ox.W@(d.ox.S*th+d.ox.bound)
            spacing=(50*avg)**(-1/3)
            sigp=(bulk['sigma']-1e-12)/avg*(1+2*spacing/(3*d.ox.xi))
            rb_y=np.zeros(n)
            rb_y[:self.no]=np.sum(-(d.ox.h/bulk['sigma']**2*sigp)[:,None]*d.ox.W,axis=0)*d.ox.S*dt/d.scale
        rc_y=np.zeros(n);rc_y[self.bi]=e['Rcontact']*self.contact.chemical_sensitivity_eV/(KB*T)*self.weights
        iy=np.zeros(n) if e['limited'] else -e['I']/(e['Rbulk']+e['Rcontact']+d.Rs)*(rb_y+rc_y)
        voy=e['Rbulk']*iy+e['I']*rb_y;vcy=e['Rcontact']*iy+e['I']*rc_y
        vo=e['Voxide'];vc=e['Vcontact'];h=1e-6
        with self.context(y,vc):
            basejac=d.core.jac(t,y[:self.n],vo)
        a[:self.n,:self.n]=basejac.toarray() if hasattr(basejac,'toarray') else basejac
        fvo=(self.core_rhs(t,y,vo+h,vc)-self.core_rhs(t,y,vo-h,vc))/(2*h)
        fvc=(self.core_rhs(t,y,vo,vc+h)-self.core_rhs(t,y,vo,vc-h))/(2*h)
        a[:self.n]+=np.outer(fvo,voy)+np.outer(fvc,vcy)
        hlog=1e-4
        with self.exchange_context(y[self.bi],T,hlog):
            p=d.pref
            d.pref=replace(p,Ti_reaction_standard_eV=p.Ti_reaction_standard_eV+2*vc)
            with d.at_temperature(T):plus=d.core.rhs(t,y[:self.n],vo)
        with self.exchange_context(y[self.bi],T,-hlog):
            p=d.pref
            d.pref=replace(p,Ti_reaction_standard_eV=p.Ti_reaction_standard_eV+2*vc)
            with d.at_temperature(T):minus=d.core.rhs(t,y[:self.n],vo)
        a[:self.n,self.nb0:]+=np.outer((plus-minus)/(2*hlog),-self.chem.passivation_barrier_eV/(KB*T)*self.weights)
        _,_,_,nu,r=self.reaction(y[self.mi],y[self.bi],T);s=y[self.mi];b=y[self.bi]
        js=self.Cb*nu*((1-b)+r*b);jb=self.Cb*nu*(-s-r*(1-s))
        a[self.mi,self.mi]-=js/self.Cm[self.active];a[self.mi,self.bi]-=jb/self.Cm[self.active]
        a[self.bi,self.mi]=js/self.Cb;a[self.bi,self.bi]=jb/self.Cb
        py=(2*e['I']*(e['Rbulk']+e['Rcontact']))*iy+e['I']**2*(rb_y+rc_y)
        a[self.n+1]=py
        if d.mode=='thermal':
            a[self.n]=py*d.cfg.thermal_R_K_W/d.cfg.thermal_tau
            a[self.n+2,self.n]=1/d.cfg.thermal_R_K_W
        else:a[self.n+2]=py
        ht=1e-3;yp=y.copy();ym=y.copy();yp[self.Tindex]+=ht;ym[self.Tindex]-=ht
        a[:,self.Tindex]=(self.rhs(t,yp,Vs)-self.rhs(t,ym,Vs))/(2*ht)
        return csc_matrix(a)

    def diagnostic(self,y,Vs):
        T=y[self.Tindex];d=self.base
        with d.at_temperature(T):e,_=self.electrical(y,Vs)
        with self.context(y,e['Vcontact']):q=d.ox.evaluate(d.port(y[:self.n]),e['Voxide'],True)
        j,fp,bp,_,_=self.reaction(y[self.mi],y[self.bi],T)
        bound=float(self.Cb@y[self.bi]);mobile=float(self.Cm@y[self.no:self.n])
        region_mobile=float(self.region_mobile@y[self.no:self.n]);B=float(self.weights@y[self.bi])
        affinity=-self.chem.binding_delta_g_eV+KB*T*(logit(y[self.mi])-logit(y[self.bi]))
        # Transfer of O to Ti creates a vacancy: invariant is N_v - N_O,Ti.
        e.update(G=1/(e['Rbulk']+e['Rcontact']),Gbulk=1/e['Rbulk'],
            TiOx_drop=float(q['drop'][d.ox.ti].sum()),AlOx_drop=float(q['drop'][~d.ox.ti].sum()),T=float(T),
            inventory=float(d.ox.C@expit(y[:self.no])-mobile-bound),Ti_surface=float(y[self.no]),
            bound_oxygen=bound,mobile_oxygen=mobile,total_Ti_oxygen=mobile+bound,
            region_oxygen=region_mobile+bound,region_oxygen_fraction=(region_mobile+bound)/self.region_capacity,
            mean_Ti_oxygen_fraction=(mobile+bound)/self.Ctotal.sum(),bound_fraction=B,
            exchange_factor=self.barrier_factor(y[self.bi],T),binding_flux=float(j.sum()),
            binding_dissipation=float(j@affinity),forward_binding=float(fp.sum()),reverse_binding=float(bp.sum()),
            exchange_flux=float(q['exchange']),ionic_dissipation=float(q['diss']),
            redox_voltage_work_eV=2*e['Vcontact'],
            thermal_balance=(d.cfg.thermal_tau/d.cfg.thermal_R_K_W*(T-d.cfg.bath_K)-(y[self.n+1]-y[self.n+2])) if d.mode=='thermal' else float(y[self.n+1]-y[self.n+2]),
            voltage_balance=e['Vout']-e['Vdev']-e['I']*d.Rs,
            power_balance=e['I']*e['Vout']-e['Pdev']-e['Pseries'],
            device_power_balance=e['Pdev']-e['Pbulk']-e['Pcontact'],
            oxide_voltage_balance=q['drop'].sum()-e['Voxide'])
        return e

def reference_state():
    path=OLD/'binding_barrier-C10-L256-M256-R2e-09.npz'
    meta=json.loads(path.with_suffix('.json').read_text())
    assert meta['data_sha']==sha(path) and meta['config']['source_sha']==sha(OLD/'chemistry.py')
    a=np.load(path);i=np.flatnonzero(np.isclose(a['t'],21.4,rtol=0,atol=1e-10));assert len(i)==1
    y=a['y'][i[0]].copy() # retain all chemical/ionic history
    # New thermal energy ledger is relative to the shared branch point.
    m=Device();T=y[m.Tindex];y[m.n+1]=m.base.cfg.thermal_tau/m.base.cfg.thermal_R_K_W*(T-300);y[m.n+2]=0.
    return y,dict(parent_sha=sha(path),parent_index=int(i[0]),parent_time_tau=21.4)
