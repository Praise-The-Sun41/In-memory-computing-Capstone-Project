"""Finite-bias post-forming candidate with spatial, closed Ti oxygen inventory.

Frozen oxide equations are reused, including their high-field approximation.
State: oxide vacancy logits followed by direct metal oxygen occupancies.
No material/time/current calibration or full terminal-energy claim.
"""
from pathlib import Path
import importlib.util, sys, json
from dataclasses import asdict
import numpy as np
from scipy.special import expit, logit, xlogy
from scipy.sparse import bmat, csc_matrix
from scipy.integrate import solve_ivp

R=Path(__file__).resolve().parent; O=R/'outputs'
spec=importlib.util.spec_from_file_location('spatial_device_closed',R/'ti-finite-oxide-coupling.py')
c=importlib.util.module_from_spec(spec);sys.modules[spec.name]=c;spec.loader.exec_module(c)
w=c.w;KT=c.KT

class Model:
    def __init__(self,local=256,nmetal=256,D=1e-6):
        p=w.m.Parameters(**json.loads((O/'ti-capacity-T30-L256-O8-C10-R1e-08.json').read_text())['parameters'])
        self.ox=w.Model(p,w.grid(local,8));self.n=self.ox.N
        self.metal=c.ti.Reservoir(c.ti.Parameters(cells=nmetal,D_nm2_per_tau=D,grid_power=2.,
            thickness_nm=p.Ti_nm,mu0_eV=p.Ti_reaction_standard_eV))
        self.nm=nmetal;self.local=local
        self.initial=np.r_[self.ox.initial[:-1],np.full(nmetal,p.Ti_initial_oxygen_fraction)]

    def port(self,y):
        return np.r_[y[:self.n],logit(y[self.n])]

    def rhs(self,t,y,V):
        s=y[self.n:];q=self.port(y);d=self.ox.evaluate(q,V)
        exchange=d[-1]*self.ox.Cti*s[0]*(1-s[0])
        metalrate=self.metal.base_jac@s
        metalrate[0]+=exchange/self.metal.C[0]
        return np.r_[d[:-1],metalrate]

    def jac(self,t,y,V):
        s=y[self.n];ds=s*(1-s);q=self.port(y)
        jj=self.ox.jacobian(q,V);rate=self.ox.evaluate(q,V)
        # Convert old logit-reservoir derivative back to interfacial flux gradient.
        eg=jj[-1]*self.ox.Cti*ds
        eg[-1]+=rate[-1]*self.ox.Cti*ds*(1-2*s)
        eg[-1]/=ds
        col=np.zeros((self.n,self.nm));col[:,0]=jj[:-1,-1]/ds
        row=np.zeros((self.nm,self.n));row[0]=eg[:-1]/self.metal.C[0]
        mt=self.metal.base_jac.copy().tolil();mt[0,0]+=eg[-1]/self.metal.C[0]
        return bmat([[csc_matrix(jj[:-1,:-1]),csc_matrix(col)],
                     [csc_matrix(row),mt.tocsc()]],format='csc')

    def inventory(self,y):
        return self.ox.C@expit(y[:self.n])-self.metal.C@y[self.n:]

    def electrical(self,y,V):return self.ox.electrical(self.port(y),V)

    def diagnostics(self,y,V):
        s=y[self.n:]
        if np.any(s<=0) or np.any(s>=1):raise ValueError('Metal occupancy outside bounds; no clipping.')
        d=self.ox.evaluate(self.port(y),V,True);th=d['theta']
        mu=self.metal.mu(s);flux=self.metal.K*(s[:-1]-s[1:])
        dti=float(flux@(mu[:-1]-mu[1:]))
        dy=self.rhs(0,y,V)
        chem=float((self.ox.C*(self.ox.G+KT*y[:self.n]))@(th*(1-th)*dy[:self.n])+(self.metal.C*mu)@dy[self.n:])
        mix=lambda x:xlogy(x,x)+xlogy(1-x,1-x)
        free=float(self.ox.C@(self.ox.G*th+KT*mix(th))+self.metal.C@(self.metal.p.mu0_eV*s+KT*mix(s)))
        # Split oxide half-cell transport from the interface reaction. Metal
        # is equipotential at zero: only oxide ionic field work enters here.
        k=self.ox.p.Ti_exchange_nm_per_tau*self.ox.S[0]
        dg=(self.ox.G[0]+self.metal.p.mu0_eV)/KT
        A=k*np.exp(-dg/2)*(1-s[0]);B=k*np.exp(dg/2)*s[0]
        delta=np.array([2*d['phi'][0]/KT]);K=2*self.ox.D[0]*self.ox.S[0]/self.ox.h[0]
        C=K*self.ox.bernoulli(delta)[0]*(1-th[0]);D=K*self.ox.bernoulli(-delta)[0]*th[0]
        face=(A+D)/(A+B+C+D)
        muf=self.ox.G[0]+KT*logit(face)
        mucell=self.ox.G[0]+KT*y[0]+2*d['phi'][0]
        reaction=-d['exchange']*(muf+mu[0]);half=d['exchange']*(muf-mucell)
        bulk=float(d['flux']@(-np.diff(self.ox.G+KT*y[:self.n]+2*d['phi'])))
        return dict(G=d['G'],J=d['J'],surface=float(s[0]),mean=float(self.metal.C@s/self.metal.C.sum()),
            exchange=d['exchange'],invariant=float(self.inventory(y)),free_energy=free,
            ionic_power=d['ionic_power'],chemical_rate=chem,dissipation=bulk+reaction+half+dti,
            energy_residual=chem-d['ionic_power']+bulk+reaction+half+dti,
            oxide_diss=bulk,reaction_diss=float(reaction),half_cell_diss=float(half),Ti_diss=dti,
            detailed_balance=float(np.log(A*(1-face)/(B*face))+(muf+mu[0])/KT),
            electronic_power=d['electronic_power'],electronic_power_residual=d['electronic_power']-d['electronic_dissipation'],
            voltage_residual=d['voltage_sum_residual'],peak_field=float(abs(d['E']).max()),
            invariant_rate=float((self.ox.C*th*(1-th))@dy[:self.n]-self.metal.C@dy[self.n:]))

def solve(model,y,duration,v0,v1,rtol=1e-8,targets=None):
    vf=lambda t:w.m.wave(t,0,duration,v0,v1)
    sol=solve_ivp(lambda t,z:model.rhs(t,z,vf(t)),(0,duration),y,method='BDF',
        jac=lambda t,z:model.jac(t,z,vf(t)),rtol=rtol,atol=rtol*.01,dense_output=True)
    if not sol.success:raise RuntimeError(sol.message)
    theta=expit(sol.y[:model.n]);metal=sol.y[model.n:]
    inv=model.ox.C@theta-model.metal.C@metal
    audit=dict(steps=len(sol.t)-1,nfev=sol.nfev,
        mass_error=float(np.max(abs(inv-model.inventory(y)))),
        theta_min=float(theta.min()),theta_max=float(theta.max()),
        metal_min=float(metal.min()),metal_max=float(metal.max()))
    assert audit['mass_error']<1e-6 and 0<theta.min()<=theta.max()<1 and 0<metal.min()<=metal.max()<1
    tt=np.linspace(0,duration,121) if targets is None else np.asarray(targets)
    return tt,sol.sol(tt).T,vf(tt),sol.y[:,-1],audit
