"""Predefined chemistry gates, independent of whether cycling improves or worsens."""
from chemistry import *

def main():
    result={};rng=np.random.default_rng(1810)
    m=Device(CASES['baseline']);old=et.Device('Ti')
    for V in (0.,1.,-.9):
        assert np.array_equal(m.rhs(0,m.initial,V),old.rhs(0,old.initial,V))
    result['disabled_exact_RHS']=True
    errors=[];balances=[]
    for case in ('binding','binding_barrier'):
        d=Device(CASES[case]);z=d.initial.copy();z[d.Tindex]=317.
        z[d.mi]=rng.uniform(.07,.8,d.nb);z[d.bi]=rng.uniform(.1,.94,d.nb)
        j,fp,bp,nu,r=d.reaction(z[d.mi],z[d.bi],317.)
        aff=-d.chem.binding_delta_g_eV+KB*317*(logit(z[d.mi])-logit(z[d.bi]))
        err=float(abs(np.log(fp/bp)-aff/(KB*317)).max())
        assert err<1e-12 and np.min(j*aff)>=0
        beq=expit(logit(z[d.mi])-d.chem.binding_delta_g_eV/(KB*317))
        assert abs(d.reaction(z[d.mi],beq,317.)[0]).max()<1e-15
        for V in (0.,1.,-.9):
            u=rng.normal(size=len(z));u[:d.no]*=.05;u[d.no:d.n]*=.002;u[d.bi]*=.002
            u[d.Tindex:d.nb0]=[.3,0,0];u/=np.linalg.norm(u);h=1e-5
            fd=(d.rhs(0,z+h*u,V)-d.rhs(0,z-h*u,V))/(2*h)
            analytic=d.jac(0,z,V)@u
            je=float(np.linalg.norm(fd-analytic)/np.linalg.norm(fd));assert je<3e-5,(case,V,je)
            dy=d.rhs(0,z,V);th=expit(z[:d.no])
            mass=float((d.base.ox.C*th*(1-th))@dy[:d.no]-d.Cm@dy[d.no:d.n]-d.Cb@dy[d.nb0:])
            assert abs(mass)<1e-9,(case,V,mass)
            errors.append(dict(case=case,V=V,jac_error=je,mass_rate_residual=mass,
                               limited=d.diagnostic(z,V)['limited']))
        balances.append(err)
    # Include a known conducting state where the real circuit activates CC.
    source=dict(np.load(BASE/'Ti-thermal-C5-R2e-09.npz'))
    ix=np.flatnonzero(source['limited'])[0];d=Device(CASES['binding_barrier'])
    z=np.r_[source['y'][ix],d.initial[d.nb0:]];V=float(source['Vs'][ix])
    assert d.diagnostic(z,V)['limited']
    u=rng.normal(size=len(z));u[d.no:]*=.001;u[d.Tindex]=.1;u[d.Tindex+1:d.nb0]=0;u/=np.linalg.norm(u)
    h=1e-5;fd=(d.rhs(0,z+h*u,V)-d.rhs(0,z-h*u,V))/(2*h)
    err=float(np.linalg.norm(fd-d.jac(0,z,V)@u)/np.linalg.norm(fd));assert err<3e-5,err
    result.update(jacobians=errors,active_CC_jacobian_error=err,detailed_balance_errors=balances)
    # Closed two-pool relaxation, solved independently of the complete device.
    cm=.75;cb=.25;dg=-.08;T=300.;nu=.1;r=np.exp(dg/(KB*T));pair_results=[]
    mix=lambda x:xlogy(x,x)+xlogy(1-x,1-x)
    for initial in ([.3,.3],[.005,.99]):
        def rhs(t,y):
            s,b=y;j=cb*nu*(s*(1-b)-r*b*(1-s));return [-j/cm,j/cb]
        total=cm*initial[0]+cb*initial[1]
        ss=brentq(lambda s:cm*s+cb*expit(logit(s)-dg/(KB*T))-total,1e-14,1-1e-14)
        expected=np.array([ss,expit(logit(ss)-dg/(KB*T))])
        sol=solve_ivp(rhs,(0,5000),initial,method='BDF',rtol=1e-10,atol=1e-12)
        assert sol.success
        s,b=sol.y;energy=cm*KB*T*mix(s)+cb*(dg*b+KB*T*mix(b))
        assert abs(cm*s+cb*b-total).max()<1e-12 and np.diff(energy).max()<1e-12
        error=float(abs(sol.y[:,-1]-expected).max());assert error<1e-7
        pair_results.append(dict(initial=initial,initial_binding_rate=rhs(0,initial)[1],
                                  equilibrium_error=error,final=sol.y[:,-1].tolist()))
    assert pair_results[0]['initial_binding_rate']>0 and pair_results[1]['initial_binding_rate']<0
    result['closed_pair_binds_and_releases']=pair_results
    capacities=[]
    for M in (128,256,512):
        d=Device(M=M);total_initial=d.Cm@d.initial[d.no:d.n]+d.Cb@d.initial[d.nb0:]
        assert abs(total_initial-.3*d.Ctotal.sum())<1e-12
        assert abs(d.Cb.sum()-.285)<1e-13
        capacities.append(dict(M=M,bound_capacity=float(d.Cb.sum()),initial_oxygen=float(total_initial)))
    result['capacity_and_initial_inventory']=capacities
    result['all_pass']=True;result['source_sha']=sha(P/'chemistry.py')
    (P/'unit_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
