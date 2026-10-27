"""Bounded fixed-span electrical-work closure for the existing continuum flux.

The same positive prefactor multiplies forward/reverse rates. It is bounded by
exp(Ea/kT)/(2*g_local), preserving the old uniform-filling drift saturation.
This is a constitutive closure, not state/current clipping and not atomistic
validation. A constant field recovers the original capped sinh law.

This is NOT a microscopic nonlocal hopping master equation. The numerical
finite-volume links and their thermodynamic affinities are unchanged. Only the
positive, forward/reverse-symmetric high-field mobility is changed. The old
attempt/barrier cap is retained. Boundary completion is a scenario choice.

span_shift: translate a centered a-long sampling interval inside the oxide.
span_interior: use an a-long interval only when it fits, else retain old field.
local: exact old electrical-work expression, for implementation regression.

An interval crossing TiOx/AlOx samples the continuous piecewise potential.
The original material free-energy jump and harmonic diffusivity remain in the
local finite-volume link. No new interfacial barrier is being claimed.
"""
from pathlib import Path
import importlib.util,sys
import numpy as np
from scipy.special import expit,logit,xlogy
from scipy.sparse import csr_matrix

R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('bounded_span_base',R/'ti-spatial-device.py')
d=importlib.util.module_from_spec(spec);sys.modules[spec.name]=d;spec.loader.exec_module(d)
KT=d.KT

def sample_matrix(edges,centers,a,mode):
    assert mode in ('local','span_shift','span_interior')
    assert 0<a<=edges[-1]-edges[0]
    mids=(centers[1:]+centers[:-1])/2;dx=np.diff(centers)
    lo=mids-a/2;hi=mids+a/2
    outside=(lo<edges[0])|(hi>edges[-1])
    use_local=np.ones(len(mids),bool) if mode=='local' else (outside if mode=='span_interior' else np.zeros(len(mids),bool))
    if mode=='span_shift':
        lo=np.clip(lo,edges[0],edges[-1]-a);hi=lo+a
    rows=[];cols=[];vals=[]
    def interp(row,x,sgn):
        j=int(np.clip(np.searchsorted(edges,x,side='right')-1,0,len(edges)-2))
        f=(x-edges[j])/(edges[j+1]-edges[j])
        rows.extend([row,row]);cols.extend([j,j+1]);vals.extend([sgn*(1-f),sgn*f])
    for i in range(len(mids)):
        if use_local[i]:
            rows.extend([i,i]);cols.extend([i,i+2]);vals.extend([-.5*a/dx[i],.5*a/dx[i]])
        else:interp(i,hi[i],1);interp(i,lo[i],-1)
    W=csr_matrix((vals,(rows,cols)),shape=(len(mids),len(edges)))
    return W,dict(left=lo,right=hi,local_fallback=use_local,original_outside=outside)

def mobility(g,cap):
    """Exactly the frozen capped sinh(g)/g factor and its log derivative."""
    g=np.asarray(g);small=g<1e-5;M=np.ones_like(g);lp=np.zeros_like(g)
    M[small]=1+g[small]**2/6;lp[small]=(g[small]/3)/M[small]
    q=g[~small];e=np.exp(-2*q)
    M[~small]=np.exp(np.minimum(q,cap))*(-np.expm1(-2*q))/(2*q)
    lp[~small]=(q<cap).astype(float)+2*e/(-np.expm1(-2*q))-1/q
    return M,lp

def drift_upper(g,cap):
    upper=np.full_like(g,np.inf)
    np.divide(np.exp(cap),2*g,out=upper,where=g>np.exp(cap)/np.finfo(float).max/2)
    return upper

class Oxide(d.w.Model):
    def __init__(self,p,edges,mode):
        super().__init__(p,edges);self.mode=mode
        self.span,self.sampling=sample_matrix(self.edges,self.x,p.lattice_hop_nm,mode)

    def field_mobility(self,e):
        work=self.span@e['faces'];g=abs(work)/KT
        # Match original arithmetic on local rows to avoid subtractive roundoff
        # in almost-balanced net fluxes. This is not a change in the field law.
        mask=self.sampling['local_fallback']
        local_work=2*np.diff(e['phi'])/np.diff(self.x)*self.p.lattice_hop_nm/2
        work[mask]=local_work[mask]
        local_g=abs(2*np.diff(e['phi'])/np.diff(self.x)*self.p.lattice_hop_nm/(2*KT))
        g[mask]=local_g[mask]
        M,lp=mobility(g,self.p.hop_barrier_eV/KT)
        if self.p.high_field_hopping and self.mode!='local':
            upper=drift_upper(local_g,self.p.hop_barrier_eV/KT)
            active=M>upper
            M=np.minimum(M,upper);lp[active]=0.
        if not self.p.high_field_hopping:M[:]=1;lp[:]=0
        return work,g,M,lp

    def evaluate(self,y,V,diagnostics=False):
        e=self.electrical(y,V);th=e['theta'];s=e['sTi']
        dx=np.diff(self.x);delta=(np.diff(self.G)+2*np.diff(e['phi']))/KT
        Df=2*self.D[:-1]*self.D[1:]/(self.D[:-1]+self.D[1:])
        work,g,M,lp=self.field_mobility(e)
        pref=Df*np.sqrt(self.S[:-1]*self.S[1:])/dx*M
        forward=pref*self.bernoulli(delta)*th[:-1]*(1-th[1:])
        reverse=pref*self.bernoulli(-delta)*th[1:]*(1-th[:-1]);flux=forward-reverse
        dg=(self.G[0]+self.p.Ti_reaction_standard_eV)/KT
        k=self.p.Ti_exchange_nm_per_tau*self.S[0]
        A=k*np.exp(-dg/2)*(1-s);B=k*np.exp(dg/2)*s
        df=2*e['phi'][0]/KT;K=2*self.D[0]*self.S[0]/self.h[0]
        C=K*self.bernoulli(np.array([df]))[0]*(1-th[0]);D=K*self.bernoulli(np.array([-df]))[0]*th[0]
        exchange=(A*C-B*D)/(A+B+C+D)
        if not self.p.vacancy_transport:flux*=0;forward*=0;reverse*=0;exchange*=0
        rate=-np.diff(np.r_[exchange,flux,0.])
        derivative=np.r_[rate/(self.C*th*(1-th)),exchange/(self.Cti*s*(1-s))]
        if not diagnostics:return derivative
        mu=self.G+KT*y[:-1]+2*e['phi'];muTi=self.p.Ti_reaction_standard_eV+KT*y[-1]
        diss=float(flux@(-np.diff(mu))+exchange*(-muTi-mu[0]))
        chem=float((self.G+KT*y[:-1])@rate+muTi*exchange)
        power=float(-2*flux@np.diff(e['phi'])-2*e['phi'][0]*exchange)
        mix=lambda z:xlogy(z,z)+xlogy(1-z,1-z)
        free=float(self.C@(self.G*th+KT*mix(th))+self.Cti*(self.p.Ti_reaction_standard_eV*s+KT*mix(s)))
        e.update(derivative=derivative,flux=flux,forward=forward,reverse=reverse,exchange=exchange,
            field_gamma=g,mobility=M,diss=diss,chemical_rate=chem,ionic_power=power,chemical_free_energy=free,
            oxygen_invariant=float(self.C@th-self.Cti*s),thermodynamic_residual=chem-power+diss,
            oxygen_rate_residual=float(rate.sum()-exchange),voltage_sum_residual=float(e['drop'].sum()-V),
            electronic_power=float(V*e['J']),electronic_dissipation=float(np.sum(e['sigma']*e['E']**2*self.h)),
            x_TiOx_or_AlOx=np.where(self.ti,2.,1.5)*(1-e['cv']/self.total_sites))
        return e

    def jacobian(self,y,V):
        n=self.N;e=self.electrical(y,V);th=e['theta'];s=e['sTi'];dt=th*(1-th);ds=s*(1-s)
        if not self.p.vacancy_conductivity_feedback:raise NotImplementedError('Feedback-off Jacobian is outside this candidate.')
        avg=self.W@(self.S*th+self.bound);spacing=(50*avg)**(-1/3)
        sigprime=(e['sigma']-1e-12)/avg*(1+2*spacing/(3*self.xi))
        drr=-(self.h/e['sigma']**2*sigprime)[:,None]*self.W*(self.S*dt)[None,:]
        rr=self.h/e['sigma'];dropgrad=e['J']*drr+rr[:,None]*(-e['J']/e['R']*drr.sum(0))[None,:]
        facesgrad=np.vstack([np.zeros(n),np.cumsum(dropgrad,axis=0)])
        phigrad=(facesgrad[:-1]+facesgrad[1:])/2
        delta=(np.diff(self.G)+2*np.diff(e['phi']))/KT;dx=np.diff(self.x)
        work,g,M,lp=self.field_mobility(e)
        Df=2*self.D[:-1]*self.D[1:]/(self.D[:-1]+self.D[1:])
        pref=Df*np.sqrt(self.S[:-1]*self.S[1:])/dx*M
        bp=self.bernoulli(delta);bm=self.bernoulli(-delta)
        left=th[:-1]*(1-th[1:]);right=th[1:]*(1-th[:-1]);flux=pref*(bp*left-bm*right)
        affinityprime=pref*(self.bernoulli_derivative(delta)*left+self.bernoulli_derivative(-delta)*right)*2/KT
        fg=np.zeros((n-1,n+1));fg[:,:n]=affinityprime[:,None]*np.diff(phigrad,axis=0)
        fg[:,:n]+=(flux*lp*np.sign(work)/KT)[:,None]*(self.span@facesgrad)
        if self.p.high_field_hopping and self.mode!='local':
            local_g=abs(np.diff(e['phi']))*self.p.lattice_hop_nm/(dx*KT)
            original,_=mobility(g,self.p.hop_barrier_eV/KT)
            upper=drift_upper(local_g,self.p.hop_barrier_eV/KT)
            active=original>upper
            # On a limited link, log M = const - log |Delta phi|.
            fg[active,:n]-=(flux[active]/np.diff(e['phi'])[active])[:,None]*np.diff(phigrad,axis=0)[active]
        ii=np.arange(n-1)
        fg[ii,ii]+=pref*(bp*(1-th[1:])+bm*th[1:])*dt[:-1]
        fg[ii,ii+1]-=pref*(bp*th[:-1]+bm*(1-th[:-1]))*dt[1:]
        # The electrode reaction and its oxide half-cell access are unchanged.
        k=self.p.Ti_exchange_nm_per_tau*self.S[0];dg=(self.G[0]+self.p.Ti_reaction_standard_eV)/KT
        A=k*np.exp(-dg/2)*(1-s);B=k*np.exp(dg/2)*s
        fd=np.array([2*e['phi'][0]/KT]);bp=self.bernoulli(fd)[0];bm=self.bernoulli(-fd)[0]
        K=2*self.D[0]*self.S[0]/self.h[0];C=K*bp*(1-th[0]);D=K*bm*th[0]
        den=A+B+C+D;exchange=(A*C-B*D)/den
        eA=(C-exchange)/den;eB=(-D-exchange)/den;eC=(A-exchange)/den;eD=(-B-exchange)/den
        eg=np.zeros(n+1)
        eg[:n]=(eC*K*self.bernoulli_derivative(fd)[0]*(1-th[0])-eD*K*self.bernoulli_derivative(-fd)[0]*th[0])*2/KT*phigrad[0]
        eg[0]+=(-eC*K*bp+eD*K*bm)*dt[0]
        eg[-1]=(-eA*k*np.exp(-dg/2)+eB*k*np.exp(dg/2))*ds
        rates=-np.diff(np.r_[exchange,flux,0.]);rg=-np.diff(np.vstack([eg,fg,np.zeros(n+1)]),axis=0)
        jac=np.empty((n+1,n+1));jac[:n]=rg/(self.C*dt)[:,None]
        jac[np.arange(n),np.arange(n)]-=rates/(self.C*dt)*(1-2*th)
        jac[-1]=eg/(self.Cti*ds);jac[-1,-1]-=exchange/(self.Cti*ds)*(1-2*s)
        if not self.p.vacancy_transport:jac[:]=0
        return jac

class Model(d.Model):
    def __init__(self,local=512,nmetal=512,D=1e-6,mode='span_shift'):
        super().__init__(local,nmetal,D)
        self.ox=Oxide(self.ox.p,self.ox.edges,mode);self.n=self.ox.N;self.mode=mode
        self.initial=np.r_[self.ox.initial[:-1],np.full(nmetal,self.ox.p.Ti_initial_oxygen_fraction)]

    def diagnostics(self,y,V):
        out=super().diagnostics(y,V);e=self.ox.electrical(self.port(y),V)
        _,g,M,_=self.ox.field_mobility(e)
        gl=abs(np.diff(e['phi']))/np.diff(self.ox.x)*self.ox.p.lattice_hop_nm/KT
        same=np.diff(self.ox.G)==0
        out['drift_cap_ratio']=float(np.max((2*gl*M/np.exp(self.ox.p.hop_barrier_eV/KT))[same]))
        original,_=mobility(g,self.ox.p.hop_barrier_eV/KT)
        out['limited_link_fraction']=float(np.mean(M<original*(1-1e-12)))
        return out
