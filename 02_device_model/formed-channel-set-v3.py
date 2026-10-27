"""Zero-storage Ti-face post-forming vacancy/hopping candidate, not the Poisson baseline.

No imposed SET threshold or gap ODE. A stable filament backbone is an explicit
initial-structure hypothesis; the near-Ti mobile-vacancy profile evolves by
finite-site detailed-balance fluxes. Electron current is quasistationary.
All kinetic time and electronic conductance scales remain scenario inputs.
"""
from pathlib import Path
from dataclasses import dataclass,asdict
import argparse,json,hashlib,time
import numpy as np
from scipy.special import expit,xlogy
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter1d

R=Path(__file__).resolve().parent;O=R/'outputs'
KT=8.617333262145e-5*300.

@dataclass
class Parameters:
    cells: int=200
    electronic_sampling_nm: float=.25
    Ti_nm: float=30.
    TiOx_nm: float=20.
    AlOx_nm: float=5.
    active_Ti_nm: float=2.
    backbone: bool=True
    diffusion_Ti_nm2_per_tau: float=.05
    diffusion_Al_nm2_per_tau: float=.0005
    lattice_hop_nm: float=.25
    hop_barrier_eV: float=.65
    localization_Ti_nm: float=.25
    localization_Al_nm: float=.25
    electron_activation_Ti_eV: float=.08
    electron_activation_Al_eV: float=.12
    Ti_exchange_nm_per_tau: float=.0001
    Ti_reaction_standard_eV: float=.06
    Ti_initial_oxygen_fraction: float=.3
    vacancy_transport: bool=True
    vacancy_conductivity_feedback: bool=True
    high_field_hopping: bool=True

class Model:
    def __init__(self,p):
        self.p=p;self.edges=np.linspace(0,p.TiOx_nm+p.AlOx_nm,p.cells+1)
        self.h=np.diff(self.edges);self.x=(self.edges[1:]+self.edges[:-1])/2
        self.ti=self.x<p.TiOx_nm;self.N=p.cells
        # Normalized to 5e28 sites/m3 * 1 nm. Effective mobile and fixed
        # reservoirs are separate, so a mobile filling is not total O deficit.
        self.total_sites=np.where(self.ti,1.,1.4)
        self.S=np.where(self.ti,.10,.028)
        stable_Ti=.06*expit((self.x-p.active_Ti_nm)/.15)
        self.bound=np.where(self.ti,stable_Ti,.042) if p.backbone else np.zeros(self.N)
        self.C=self.h*self.S
        self.Cti=1.14*p.Ti_nm # one O site per Ti, explicit scenario capacity
        self.G=np.where(self.ti,0.,.02)
        self.D=np.where(self.ti,p.diffusion_Ti_nm2_per_tau,p.diffusion_Al_nm2_per_tau)
        self.xi=np.where(self.ti,p.localization_Ti_nm,p.localization_Al_nm)
        self.Ea_e=np.where(self.ti,p.electron_activation_Ti_eV,p.electron_activation_Al_eV)
        theta=np.where(self.ti,.2,expit(np.log(.2/.8)-.02/KT))
        self.initial=np.r_[np.log(theta/(1-theta)),np.log(p.Ti_initial_oxygen_fraction/(1-p.Ti_initial_oxygen_fraction))]
        self.initial_defects=self.S*theta+self.bound

    def electrical(self,y,V):
        theta=expit(y[:-1]);s=expit(y[-1])
        cv=self.S*theta+self.bound
        electronic_cv=cv if self.p.vacancy_conductivity_feedback else self.initial_defects
        # A carrier samples a finite localization volume; do not infer an
        # independent conducting link from an arbitrarily thin numerical cell.
        # Reflect at each material interface, not across different site systems.
        if self.p.electronic_sampling_nm > 0:
            electronic_cv=electronic_cv.copy()
            for mask in (self.ti,~self.ti):
                electronic_cv[mask]=gaussian_filter1d(electronic_cv[mask],self.p.electronic_sampling_nm/self.h[0],mode='reflect',truncate=5.)
        # Mean localized-site distance; 5e28/m3 = 50/nm3. Miller-Abrahams
        # distance/activation scaling, with an explicit weak leakage floor.
        spacing=(50*electronic_cv)**(-1/3)
        sigma=1e-12+electronic_cv*np.exp(-2*spacing/self.xi-self.Ea_e/KT)
        resistance=self.h/sigma;Rtot=sum(resistance);current=V/Rtot
        drop=current*resistance
        faces=np.r_[0.,np.cumsum(drop)];phi=(faces[:-1]+faces[1:])/2
        return dict(theta=theta,sTi=s,cv=cv,sigma=sigma,R=Rtot,G=1/Rtot,J=current,
                    phi=phi,E=-drop/self.h,drop=drop,faces=faces)

    @staticmethod
    def bernoulli(x):
        x=np.asarray(x);out=np.empty_like(x);small=abs(x)<1e-5
        out[small]=1-x[small]/2+x[small]**2/12-x[small]**4/720
        z=x[~small];out[~small]=np.where(z>50,z*np.exp(-z),z/np.expm1(z))
        return out

    def evaluate(self,y,V,diagnostics=False):
        d=self.electrical(y,V);th=d['theta'];s=d['sTi']
        dx=np.diff(self.x);delta=(np.diff(self.G)+2*np.diff(d['phi']))/KT
        Df=2*self.D[:-1]*self.D[1:]/(self.D[:-1]+self.D[1:])
        mobility=np.ones_like(dx)
        if self.p.high_field_hopping:
            gamma=abs(2*np.diff(d['phi'])/dx*self.p.lattice_hop_nm/(2*KT))
            small=gamma<1e-5;mobility[small]=1+gamma[small]**2/6
            g=gamma[~small]
            # Fixed lattice length, never the numerical mesh spacing.
            # Above-barrier drift saturates instead of exceeding attempt rate.
            mobility[~small]=np.exp(np.minimum(g,self.p.hop_barrier_eV/KT))*(-np.expm1(-2*g))/(2*g)
        pref=Df*np.sqrt(self.S[:-1]*self.S[1:])/dx*mobility
        forward=pref*self.bernoulli(delta)*th[:-1]*(1-th[1:])
        reverse=pref*self.bernoulli(-delta)*th[1:]*(1-th[:-1])
        flux=forward-reverse
        # Ti oxygen uptake creates a film vacancy; both share one reaction.
        dg=(self.G[0]+self.p.Ti_reaction_standard_eV)/KT
        k=self.p.Ti_exchange_nm_per_tau*self.S[0]
        A=k*np.exp(-dg/2)*(1-s);B=k*np.exp(dg/2)*s
        delta_face=2*d['phi'][0]/KT
        K=2*self.D[0]*self.S[0]/self.h[0]
        C=K*self.bernoulli(np.array([delta_face]))[0]*(1-th[0])
        D=K*self.bernoulli(np.array([-delta_face]))[0]*th[0]
        den=A+B+C+D
        theta_face=(A+D)/den
        f=A*(1-theta_face);b=B*theta_face
        exchange=(A*C-B*D)/den
        if not self.p.vacancy_transport:
            flux*=0;forward*=0;reverse*=0;exchange*=0;f*=0;b*=0
        inventory_rate=-np.diff(np.r_[exchange,flux,0.])
        derivative=np.r_[inventory_rate/(self.C*th*(1-th)),exchange/(self.Cti*s*(1-s))]
        if not diagnostics:return derivative
        mu=self.G+KT*y[:-1]+2*d['phi'];muTi=self.p.Ti_reaction_standard_eV+KT*y[-1]
        diss=float(np.dot(flux,-np.diff(mu))+exchange*(-muTi-mu[0]))
        chem_rate=float(np.dot(self.G+KT*y[:-1],inventory_rate)+muTi*exchange)
        ion_power=float(-2*np.dot(flux,np.diff(d['phi']))-2*d['phi'][0]*exchange)
        mixing=lambda t:xlogy(t,t)+xlogy(1-t,1-t)
        F=float(self.C@(self.G*th+KT*mixing(th))+self.Cti*(self.p.Ti_reaction_standard_eV*s+KT*mixing(s)))
        d.update(derivative=derivative,flux=flux,exchange=exchange,diss=diss,chemical_rate=chem_rate,
                 ionic_power=ion_power,chemical_free_energy=F,
                 oxygen_invariant=float(self.C@th-self.Cti*s),
                 thermodynamic_residual=chem_rate-ion_power+diss,
                 oxygen_rate_residual=float(sum(inventory_rate)-exchange),
                 voltage_sum_residual=float(sum(d['drop'])-V),
                 electronic_power=float(V*d['J']),
                 electronic_dissipation=float(np.sum(d['sigma']*d['E']**2*self.h)),
                 x_TiOx_or_AlOx=np.where(self.ti,2.,1.5)*(1-d['cv']/self.total_sites))
        return d

def wave(t,left,right,v0,v1):
    u=np.clip((t-left)/(right-left),0,1);return v0+(v1-v0)*u*u*(3-2*u)

def protocol(quarter=10.,cycles=2,blank=False):
    rows=[];t=0.
    def add(duration,v0,v1,kind):
        nonlocal t
        rows.append(dict(left=t,right=t+duration,v0=v0,v1=v1,kind=kind));t+=duration
    # Post-forming channel -> negative conditioning creates its own HRS.
    add(quarter,0,-1,'prepare-negative-rise');add(quarter,-1,0,'prepare-negative-fall')
    add(.2,0,.02,'initial-read-rise');add(1,.02,.02,'initial-read');add(.2,.02,0,'initial-read-fall')
    for c in range(cycles):
        amp=0 if blank else 1.
        add(quarter,0,amp,f'positive-{c}-rise');add(quarter,amp,0,f'positive-{c}-fall')
        add(.2,0,.02,f'set-read-{c}-rise');add(1,.02,.02,f'set-read-{c}');add(.2,.02,0,f'set-read-{c}-fall')
        add(quarter,0,-amp,f'negative-{c}-rise');add(quarter,-amp,0,f'negative-{c}-fall')
        add(.2,0,.02,f'reset-read-{c}-rise');add(1,.02,.02,f'reset-read-{c}');add(.2,.02,0,f'reset-read-{c}-fall')
    add(20,0,0,'retention')
    return rows

def run(p,quarter=10.,cycles=2,blank=False,rtol=1e-7,tag='candidate'):
    m=Model(p);z=m.initial.copy();rows=protocol(quarter,cycles,blank);ts=[];zs=[];vs=[];kinds=[];stats=[]
    start=time.perf_counter()
    for row in rows:
        l,r=row['left'],row['right'];voltage=lambda t:wave(t,l,r,row['v0'],row['v1'])
        sol=solve_ivp(lambda t,y:m.evaluate(y,voltage(t)),(l,r),z,method='BDF',rtol=rtol,atol=rtol*.01,dense_output=True)
        if not sol.success:
            np.savez(O/f'formed-channel-v3-failure-{tag}-N{p.cells}.npz',time=sol.t,z=sol.y,V=voltage(sol.t))
            raise RuntimeError(f"{row['kind']} at {sol.t[-1]}: {sol.message}")
        target=np.linspace(l,r,31);target=target if not ts else target[1:]
        ts.extend(target);zs.extend(sol.sol(target).T);vs.extend(voltage(target));kinds.extend([row['kind']]*len(target));z=sol.y[:,-1]
        stats.append(dict(**row,nfev=sol.nfev,steps=len(sol.t)-1))
        print(tag,row['kind'],r,'G',m.electrical(z,0)['G'],'secs',time.perf_counter()-start,flush=True)
    ds=[m.evaluate(y,V,True) for y,V in zip(zs,vs)]
    name=f'formed-channel-v3-{tag}-N{p.cells}-R{rtol:g}'+('-blank' if blank else '')
    data=dict(time=np.array(ts),z=np.array(zs),V=np.array(vs),kind=np.array(kinds),edges=m.edges)
    for k in ds[0]:
        if k!='derivative':data[k]=np.array([d[k] for d in ds])
    np.savez(O/(name+'.npz'),**data)
    readings=[]
    for i,k in enumerate(kinds):
        if (k=='initial-read' or (k.startswith(('set-read-','reset-read-')) and k.split('-')[-1].isdigit())) and (i==len(kinds)-1 or kinds[i+1]!=k):
            readings.append(dict(kind=k,time=ts[i],G=float(data['G'][i]),J=float(data['J'][i]),sTi=float(data['sTi'][i])))
    out=dict(parameters=asdict(p),quarter=quarter,cycles=cycles,blank=blank,rtol=rtol,readings=readings,
             elapsed_seconds=time.perf_counter()-start,segments=stats,
             max_oxygen_drift=float(np.max(abs(data['oxygen_invariant']-data['oxygen_invariant'][0]))),
             max_thermodynamic_residual=float(np.max(abs(data['thermodynamic_residual']))),
             minimum_dissipation=float(data['diss'].min()),
             theta_range=[float(data['theta'].min()),float(data['theta'].max())],
             peak_field_V_per_nm=float(abs(data['E']).max()),
             file_sha256=hashlib.sha256((O/(name+'.npz')).read_bytes()).hexdigest(),
             runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             postforming_backbone_assumed=p.backbone,pristine_forming_solved=False,
             old_Poisson_model_replaced=False,material_calibrated=False,clock_calibrated=False,
             full_electron_ion_terminal_energy_ledger=False,electron_current_quasistatic=True)
    (O/(name+'.json')).write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cells',type=int,default=200);ap.add_argument('--quarter',type=float,default=10);ap.add_argument('--cycles',type=int,default=2);ap.add_argument('--blank',action='store_true');ap.add_argument('--rtol',type=float,default=1e-7);ap.add_argument('--tag',default='candidate');ap.add_argument('--no-backbone',action='store_true');ap.add_argument('--frozen',action='store_true');ap.add_argument('--no-feedback',action='store_true');a=ap.parse_args()
    run(Parameters(cells=a.cells,backbone=not a.no_backbone,vacancy_transport=not a.frozen,vacancy_conductivity_feedback=not a.no_feedback),a.quarter,a.cycles,a.blank,a.rtol,a.tag)
