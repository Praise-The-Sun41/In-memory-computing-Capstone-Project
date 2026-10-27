from model import *

def main():
    result=[];rng=np.random.default_rng(72)
    for name in ['Ti_frozen500','Pt_matched500','Pt_matched5000','Pt_barrier']:
        s=System(name);z=s.y0.copy();z[s.n]=312.
        for V,cc in [(.3,300e-6),(-1.,300e-6),(1.,5e-6)]:
            s.base.Icc=cc;u=rng.normal(size=len(z));u[:s.no]*=.2
            if s.kind=='Ti':u[s.no:s.n]*=.002;u[s.m.bi]*=.002
            u[s.n:s.n+3]=[.3,0,0];u/=np.linalg.norm(u);h=1e-5
            fd=(s.rhs(0,z+h*u,V)-s.rhs(0,z-h*u,V))/(2*h);j=s.jac(0,z,V)@u
            err=float(np.linalg.norm(fd-j)/max(np.linalg.norm(fd),1e-30));assert err<2e-5,(name,V,err)
            d=s.diagnostic(z,V);assert abs(d['power_balance'])<1e-12 and d['Pcontact']>=0
            assert d['limited']==(cc==5e-6)
            result.append(dict(case=name,V=V,jac_error=err))
    # Limiting-case identity: no Ti oxygen exchange + frozen binding equals
    # blocking Pt for the common oxide, circuit and thermal variables.
    for R,name in [(500,'Pt_matched500'),(5000,'Pt_matched5000')]:
        pt=System(name);ti=System('Ti_frozen500');ti.m.contact=it.Contact(R,.2)
        ti.m.original_pref=replace(ti.m.original_pref,Ti_exchange_nm_per_tau=0.)
        for V in [-1.,0.,1.]:
            a=ti.rhs(0,ti.y0,V);b=pt.rhs(0,pt.y0,V)
            err=float(abs(np.r_[a[:ti.no],a[ti.n:ti.n+3]]-b).max());assert err<1e-9,(R,V,err)
            result.append(dict(test='blocked_Ti_equals_Pt',R=R,V=V,max_error=err))
    # Zero extra contact restores the existing electrothermal Pt equations.
    pt=System('Pt_matched500');pt.m.Rc=0.;old=et.Device('Pt')
    for V in [-1.,0.,1.]:assert np.allclose(pt.rhs(0,pt.y0,V),old.rhs(0,pt.y0,V),rtol=1e-12,atol=1e-10)
    out=dict(all_pass=True,checks=result,sources=sources())
    (P/'unit_checks.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
