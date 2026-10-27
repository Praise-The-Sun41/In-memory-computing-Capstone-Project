"""Controlled Ti/Pt histories; frozen kernels, no fitted lifetime or per-cycle damage."""
from pathlib import Path
import sys, json, hashlib
from dataclasses import replace, asdict
import numpy as np
from scipy.special import expit
from scipy.sparse import csc_matrix
P=Path(__file__).resolve().parent
SRC=P.parent/'ti_contact_oxidation_2026-10-18'
sys.path.insert(0,str(SRC))
import interface as it
et=it.et
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
CASES={
 'Ti_frozen500':('Ti',500.,.2,True,None),
 'Ti_static500':('Ti',500.,0.,False,'static_500'),
 'Ti_dynamic500':('Ti',500.,.2,False,'chemical_500'),
 'Pt_matched500':('Pt',500.,0.,False,None),
 'Ti_static5000':('Ti',5000.,0.,False,'static_5000'),
 'Ti_dynamic5000':('Ti',5000.,.2,False,'chemical_5000'),
 'Pt_matched5000':('Pt',5000.,0.,False,None),
 'Pt_barrier':('Pt_contact',0.,0.,False,None),
}

class PtFixed:
    """No electrode oxygen states; matched lumped electronic resistance heats device."""
    def __init__(self,Rc):
        self.base=et.Device('Pt');self.no=self.n=self.base.n;self.Tindex=self.n;self.Rc=Rc
    def electrical(self,y,V):
        d=self.base
        e=et.f.d.w.Model.electrical(d.ox,d.port(y[:self.n]),0.)
        rb=e['R']/d.scale;I=float(np.clip(V/(rb+self.Rc+d.Rs),-d.Icc,d.Icc))
        vo=I*rb;vc=I*self.Rc;vd=vo+vc
        return dict(I=I,Rbulk=rb,Rcontact=self.Rc,Voxide=vo,Vcontact=vc,Vdev=vd,
            Vout=vd+I*d.Rs,Vsource=V,Pdev=I*vd,Pbulk=I*vo,Pcontact=I*vc,Pseries=I*I*d.Rs,
            limited=abs(I)>=d.Icc*(1-1e-10)),e
    def rhs(self,t,y,V):
        d=self.base
        with d.at_temperature(y[self.n]):
            e,_=self.electrical(y,V);dx=d.core.rhs(t,y[:self.n],e['Voxide'])
        loss=(y[self.n]-d.cfg.bath_K)/d.cfg.thermal_R_K_W
        return np.r_[dx,(e['Pdev']-loss)*d.cfg.thermal_R_K_W/d.cfg.thermal_tau,e['Pdev'],loss]
    def jac(self,t,y,V):
        d=self.base;n=self.n;a=np.zeros((n+3,n+3))
        with d.at_temperature(y[n]):
            e,b=self.electrical(y,V);th=b['theta'];dt=th*(1-th)
            avg=d.ox.W@(d.ox.S*th+d.ox.bound);spacing=(50*avg)**(-1/3)
            sigp=(b['sigma']-1e-12)/avg*(1+2*spacing/(3*d.ox.xi))
            ry=np.sum(-(d.ox.h/b['sigma']**2*sigp)[:,None]*d.ox.W,axis=0)*d.ox.S*dt/d.scale
            iy=np.zeros(n) if e['limited'] else -e['I']*ry/(e['Rbulk']+self.Rc+d.Rs)
            vy=e['Rbulk']*iy+e['I']*ry;h=1e-6;vo=e['Voxide']
            fv=(d.core.rhs(t,y[:n],vo+h)-d.core.rhs(t,y[:n],vo-h))/(2*h)
            a[:n,:n]=d.core.jac(t,y[:n],vo).toarray()+np.outer(fv,vy)
            py=2*e['I']*(e['Rbulk']+self.Rc)*iy+e['I']**2*ry
            a[n,:n]=py*d.cfg.thermal_R_K_W/d.cfg.thermal_tau;a[n+1,:n]=py
            a[n+2,n]=1/d.cfg.thermal_R_K_W
        h=1e-3;yp=y.copy();ym=y.copy();yp[n]+=h;ym[n]-=h
        a[:,n]=(self.rhs(t,yp,V)-self.rhs(t,ym,V))/(2*h)
        return csc_matrix(a)
    def diagnostic(self,y,V):
        d=self.base;T=y[self.n]
        with d.at_temperature(T):
            e,_=self.electrical(y,V);q=d.ox.evaluate(d.port(y[:self.n]),e['Voxide'],True)
        e.update(G=1/(e['Rbulk']+self.Rc),T=T,inventory=float(d.ox.C@expit(y[:self.n])),
            exchange_flux=float(q['exchange']),binding_flux=0.,ionic_dissipation=float(q['diss']),
            thermal_balance=d.cfg.thermal_tau/d.cfg.thermal_R_K_W*(T-d.cfg.bath_K)-(y[self.n+1]-y[self.n+2]),
            power_balance=e['I']*e['Vout']-e['Pdev']-e['Pseries'],
            device_power_balance=e['Pdev']-e['Pbulk']-e['Pcontact'],
            voltage_balance=e['Vout']-e['Vdev']-e['I']*d.Rs,
            TiOx_drop=float(q['drop'][d.ox.ti].sum()),AlOx_drop=float(q['drop'][~d.ox.ti].sum()))
        return e

class System:
    def __init__(self,name):
        self.name=name;kind,Rc,eta,frozen,self.reuse=CASES[name];self.kind=kind
        common,self.origin=it.reference_state();template=it.Device();B=float(template.weights@common[template.bi])
        if kind=='Ti':
            self.m=it.Device(it.Contact(Rc,eta),B)
            if frozen:self.m.chem=replace(self.m.chem,capture_rate_per_tau=0.)
            self.base=self.m.base;self.y0=common.copy()
        elif kind=='Pt':
            self.m=PtFixed(Rc);self.base=self.m.base
            self.y0=np.r_[common[:template.no],common[template.n:template.n+3]]
        else:
            self.m=et.Device('Pt_contact');self.base=self.m
            self.y0=np.r_[common[:template.no],common[template.n:template.n+3]]
        self.no=self.base.no;self.n=self.base.n;self.Tindex=self.n
        self.rhs=self.m.rhs;self.jac=self.m.jac
        self.config=dict(case=name,kind=kind,reference_R=Rc,chemical_sensitivity_eV=eta,
            binding_frozen=frozen,initial_sha=hashlib.sha256(self.y0.tobytes()).hexdigest(),
            origin=self.origin,thermal=asdict(self.base.cfg),shared_oxide_initial=True,
            Pt_oxygen_states=0,irreversible_damage=False,new_history_COMSOL_verified=False)
        if kind=='Pt_contact':self.config['barrier_parameters']=asdict(self.base.ox.cp)
    def diagnostic(self,y,V):
        d=self.m.diagnostic(y,V)
        if self.kind=='Pt_contact':
            with self.base.at_temperature(y[self.n]):
                q=self.base.ox.evaluate(self.base.port(y[:self.n]),d['Vdev'],True)
            d.update(Rcontact=(et.ct.KT/q['Isb']+et.ct.KT/q['Ist'])/self.base.scale,
                exchange_flux=float(q['exchange']),binding_flux=0.,ionic_dissipation=float(q['diss']))
            # KT restored to 300 K after context: evaluate differential R consistently with T.
            d['Rcontact']=(et.KB*y[self.n]/q['Isb']+et.KB*y[self.n]/q['Ist'])/self.base.scale
        return d
    def G300(self,y):
        z=y.copy();z[self.n]=300.;return self.diagnostic(z,0.)['G']
    def occupancy(self,Y):
        out=[expit(Y[:,:self.no]).ravel()]
        if self.kind=='Ti':out.extend([Y[:,self.no:self.n].ravel(),Y[:,self.m.nb0:].ravel()])
        return np.concatenate(out)
    def inventory(self,Y):
        v=expit(Y[:,:self.no])@self.base.ox.C
        if self.kind=='Ti':v-=Y[:,self.no:self.n]@self.m.Cm+Y[:,self.m.nb0:]@self.m.Cb
        return v

def sources():
    paths=[Path(__file__),SRC/'interface.py',it.OLD/'chemistry.py',it.ch.BASE/'coupled.py',
           P.parent/'pt_contacts_2026-10-06/contact_model.py',P.parent/'pt_blocking_2026-10-06/pt_model.py',
           P.parent/'bounded-span-field.py']
    return {str(p.resolve()):sha(p) for p in paths}
