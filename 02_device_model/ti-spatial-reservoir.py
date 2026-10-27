"""Independent 1-D ideal Ti oxygen reservoir; no fitted material coefficients.

z=0 is the oxide-facing boundary; z=L is blocked. Fraction s is a model
occupancy, not a measured O/Ti stoichiometry. No bound pool, phase changes,
electric field in metal, or device coupling is introduced here.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq
from scipy.special import xlogy
from scipy.sparse import diags

KT = 8.617333262145e-5*300.


@dataclass(frozen=True)
class Reaction:
    """j+ = A(1-s)/(d+e*s), j- = B*s/(d+e*s), evaluated at the surface.

    A/B fixes equilibrium activity; positive common denominator controls
    kinetics without changing local detailed balance. Flux units are the
    same normalized oxygen inventory per area per tau as the old core.
    """
    A: float
    B: float
    d: float = 1.
    e: float = 0.
    mu0: float = .06

    def __post_init__(self):
        if min(self.A, self.B, self.d, self.d+self.e) <= 0:
            raise ValueError('Positive rates and denominator are required.')

    @property
    def equilibrium(self):
        return self.A/(self.A+self.B)

    @property
    def bath_mu(self):
        return self.mu0+KT*np.log(self.A/self.B)

    def rates(self, s):
        den = self.d+self.e*s
        return self.A*(1-s)/den, self.B*s/den

    def flux(self, s):
        # One numerator avoids subtraction of large nearly equal rates.
        return (self.A-(self.A+self.B)*s)/(self.d+self.e*s)

    def slope(self, s):
        return -((self.A+self.B)*self.d+self.A*self.e)/(self.d+self.e*s)**2


@dataclass(frozen=True)
class Parameters:
    thickness_nm: float = 30.
    cells: int = 128
    D_nm2_per_tau: float = 1.
    density_per_nm: float = 1.14
    mu0_eV: float = .06
    surface_node: bool = True
    grid_power: float = 1.


class Reservoir:
    def __init__(self, p, reaction=None, imposed_flux=0.):
        self.p=p; self.reaction=reaction; self.imposed_flux=imposed_flux
        if p.D_nm2_per_tau <= 0 or p.thickness_nm <= 0 or p.cells < 2:
            raise ValueError('Positive length, diffusivity and >=2 cells required.')
        if reaction is not None and (imposed_flux != 0 or reaction.mu0 != p.mu0_eV):
            raise ValueError('Use either reaction or prescribed flux; match chemical references.')
        if p.surface_node:
            # Vertex-centred FV: both material surfaces carry finite dual-cell
            # inventory. No algebraic half-cell quasi-steady jump at startup.
            self.x=p.thickness_nm*np.linspace(0,1,p.cells)**p.grid_power
            self.edges=np.r_[0.,(self.x[:-1]+self.x[1:])/2,p.thickness_nm]
        else:
            self.edges=np.linspace(0,p.thickness_nm,p.cells+1)
            self.x=(self.edges[1:]+self.edges[:-1])/2
        self.h=np.diff(self.edges)
        self.C=p.density_per_nm*self.h
        self.K=p.density_per_nm*p.D_nm2_per_tau/np.diff(self.x)
        self.g=p.density_per_nm*p.D_nm2_per_tau/(self.h[0]/2)
        diag=-np.r_[self.K[0],self.K[:-1]+self.K[1:],self.K[-1]]/self.C
        self.base_jac=diags([self.K/self.C[1:],diag,self.K/self.C[:-1]],[-1,0,1],format='csc')

    def mu(self,s):
        return self.p.mu0_eV+KT*np.log(s/(1-s))

    def boundary(self,s0):
        if self.p.surface_node:
            if self.reaction is None:return self.imposed_flux,s0,0.
            return self.reaction.flux(s0),s0,self.reaction.slope(s0)
        if self.reaction is None:
            j=self.imposed_flux
            return j,s0+j/self.g,0.
        r=self.reaction
        # Zero-storage face: chemical reaction equals half-cell diffusion.
        sf=brentq(lambda q:r.flux(q)-self.g*(q-s0),0.,1.,xtol=5e-15,rtol=1e-14)
        j=r.flux(sf)
        jp=r.slope(sf)
        return j,sf,jp*self.g/(self.g-jp)

    def rhs(self,t,s):
        j,_,_=self.boundary(s[0])
        faces=np.r_[j,self.K*(s[:-1]-s[1:]),0.]
        return -np.diff(faces)/self.C

    def jac(self,t,s):
        _,_,jp=self.boundary(s[0])
        out=self.base_jac.copy()
        out[0,0]+=jp/self.C[0]
        return out

    def diagnostics(self,s):
        if np.any(s<=0) or np.any(s>=1):
            raise ValueError('Occupancy outside open interval; never clip states.')
        j,sf,_=self.boundary(s[0]); mu=self.mu(s); ds=self.rhs(0,s)
        bulk=float(np.dot(self.K*(s[:-1]-s[1:]),mu[:-1]-mu[1:]))
        free=float(self.C@(self.p.mu0_eV*s+KT*(xlogy(s,s)+xlogy(1-s,1-s))))
        fdot=float((self.C*mu)@ds)
        face_mu=float(self.mu(sf))
        diffusion_face=j*(face_mu-mu[0])
        reaction_diss=j*(self.reaction.bath_mu-face_mu) if self.reaction else 0.
        supply_mu=self.reaction.bath_mu if self.reaction else face_mu
        work=supply_mu*j
        out=dict(inventory=float(self.C@s),mean=float(self.C@s/self.C.sum()),
                 surface=float(sf),flux=float(j),free_energy=free,free_energy_rate=fdot,
                 external_chemical_power=float(work),bulk_dissipation=bulk,
                 half_cell_dissipation=float(diffusion_face),reaction_dissipation=float(reaction_diss),
                 balance_residual=float(fdot-work+bulk+diffusion_face+reaction_diss),
                 mass_rate_residual=float(self.C@ds-j),
                 boundary_flux_residual=float(0. if self.p.surface_node else j-self.g*(sf-s[0])))
        if self.reaction:
            f,b=self.reaction.rates(sf)
            out['detailed_balance_residual']=float(np.log(f/b)-(self.reaction.bath_mu-face_mu)/KT)
            out['grand_potential']=float(free-self.reaction.bath_mu*(self.C@s))
        return out
