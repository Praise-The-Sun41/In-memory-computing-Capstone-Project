"""Closed, zero-voltage TiOx/AlOx + spatial Ti oxygen coupling.

Same vacancy transport and Ti-face reaction as the frozen post-forming core at
V=0; spatial Ti replaces only its uniformly mixed oxygen variable. No source
holds the oxide fixed. No field-driven/device-sweep validation is claimed.
"""
from pathlib import Path
from dataclasses import replace
import importlib.util, sys
import numpy as np
from scipy.special import expit, logit, xlogy
from scipy.sparse import coo_matrix, block_diag
from scipy.optimize import brentq

R=Path(__file__).resolve().parent


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    x=importlib.util.module_from_spec(spec); sys.modules[name]=x; spec.loader.exec_module(x)
    return x


w=module('closed_oxide_core',R/'local-reset-mesh.py')
ti=module('closed_ti_core',R/'ti-spatial-reservoir.py')
KT=ti.KT


class Coupled:
    def __init__(self,oxide,metal,exchange_enabled=True):
        if not metal.p.surface_node or metal.reaction is not None or metal.imposed_flux != 0:
            raise ValueError('Use the standalone surface-node metal without external supply.')
        if metal.p.mu0_eV != oxide.p.Ti_reaction_standard_eV:
            raise ValueError('Ti chemical reference mismatch.')
        self.ox=oxide; self.metal=metal; self.n=oxide.N; self.nm=len(metal.C)
        self.exchange_enabled=exchange_enabled
        self.capacity=np.r_[oxide.C,metal.C]
        dx=np.diff(oxide.x)
        delta=np.diff(oxide.G)/KT
        df=2*oxide.D[:-1]*oxide.D[1:]/(oxide.D[:-1]+oxide.D[1:])
        pref=df*np.sqrt(oxide.S[:-1]*oxide.S[1:])/dx
        self.fp=pref*oxide.bernoulli(delta); self.fm=pref*oxide.bernoulli(-delta)
        k=oxide.p.Ti_exchange_nm_per_tau*oxide.S[0]
        dg=(oxide.G[0]+metal.p.mu0_eV)/KT
        self.a=k*np.exp(-dg/2); self.b=k*np.exp(dg/2)
        self.K=2*oxide.D[0]*oxide.S[0]/oxide.h[0]

    def interface(self,t,s):
        A=self.a*(1-s); B=self.b*s; C=self.K*(1-t); D=self.K*t
        den=A+B+C+D
        face=(A+D)/den
        j=(A*C-B*D)/den
        jt=-self.K*(A+B)/den
        js=(-self.a*C-self.b*D-j*(self.b-self.a))/den
        f=A*(1-face); b=B*face
        if not self.exchange_enabled:
            j=jt=js=f=b=0.;face=t
        return j,jt,js,face,f,b

    def rhs(self,t,y):
        th=y[:self.n]; s=y[self.n:]
        j,*_=self.interface(th[0],s[0])
        flux=self.fp*th[:-1]*(1-th[1:])-self.fm*th[1:]*(1-th[:-1])
        oxrate=-np.diff(np.r_[j,flux,0.])
        metalrate=-np.diff(np.r_[j,self.metal.K*(s[:-1]-s[1:]),0.])
        return np.r_[oxrate/self.ox.C,metalrate/self.metal.C]

    def jac(self,t,y):
        th=y[:self.n]; s=y[self.n:]
        left=self.fp*(1-th[1:])+self.fm*th[1:]
        right=-self.fp*th[:-1]-self.fm*(1-th[:-1])
        i=np.arange(self.n-1)
        rows=np.r_[i,i,i+1,i+1]; cols=np.r_[i,i+1,i,i+1]
        vals=np.r_[-left/self.ox.C[:-1],-right/self.ox.C[:-1],left/self.ox.C[1:],right/self.ox.C[1:]]
        oj=coo_matrix((vals,(rows,cols)),shape=(self.n,self.n)).tocsc()
        out=block_diag((oj,self.metal.base_jac),format='csc')
        _,jt,js,*_=self.interface(th[0],s[0])
        rr=[0,0,self.n,self.n]; cc=[0,self.n,0,self.n]
        v=[jt/self.ox.C[0],js/self.ox.C[0],jt/self.metal.C[0],js/self.metal.C[0]]
        return out+coo_matrix((v,(rr,cc)),shape=out.shape).tocsc()

    def equilibrium(self,y):
        q=float(self.ox.C@y[:self.n]-self.metal.C@y[self.n:])
        def state(mu):
            return np.r_[expit((mu-self.ox.G)/KT),
                          np.full(self.nm,expit((-mu-self.metal.p.mu0_eV)/KT))]
        def residual(mu):
            z=state(mu)
            return self.ox.C@z[:self.n]-self.metal.C@z[self.n:]-q
        root=brentq(residual,-2,2,xtol=1e-14)
        return state(root)

    def diagnostics(self,y):
        if np.any(y<=0) or np.any(y>=1):raise ValueError('Nonphysical occupation; no clipping.')
        th=y[:self.n];s=y[self.n:]; dy=self.rhs(0,y)
        j,_,_,face,f,b=self.interface(th[0],s[0])
        muv=self.ox.G+KT*logit(th); mum=self.metal.mu(s)
        flux=self.fp*th[:-1]*(1-th[1:])-self.fm*th[1:]*(1-th[:-1])
        mtflux=self.metal.K*(s[:-1]-s[1:])
        dox=float(flux@(muv[:-1]-muv[1:])); dti=float(mtflux@(mum[:-1]-mum[1:]))
        muf=self.ox.G[0]+KT*logit(face)
        reaction=float(-j*(muf+mum[0])); half=float(j*(muf-muv[0]))
        mixing=lambda u:xlogy(u,u)+xlogy(1-u,1-u)
        F=float(self.ox.C@(self.ox.G*th+KT*mixing(th))+self.metal.C@(self.metal.p.mu0_eV*s+KT*mixing(s)))
        Fdot=float((self.ox.C*muv)@dy[:self.n]+(self.metal.C*mum)@dy[self.n:])
        M=float(self.ox.C@th);N=float(self.metal.C@s)
        # N_oxide oxygen = constant - vacancy count, hence N-M is conserved.
        out=dict(oxide_vacancies=M,Ti_oxygen=N,invariant=M-N,
                 surface=float(s[0]),mean=float(N/self.metal.C.sum()),back=float(s[-1]),exchange=float(j),
                 free_energy=F,free_energy_rate=Fdot,oxide_diss=dox,Ti_diss=dti,
                 reaction_diss=reaction,oxide_half_cell_diss=half,
                 energy_residual=float(Fdot+dox+dti+reaction+half),
                 invariant_rate=float(self.ox.C@dy[:self.n]-self.metal.C@dy[self.n:]),
                 oxide_rate_residual=float(self.ox.C@dy[:self.n]-j),
                 Ti_rate_residual=float(self.metal.C@dy[self.n:]-j))
        if self.exchange_enabled:
            out['detailed_balance_residual']=float(np.log(f/b)+(muf+mum[0])/KT)
        return out
