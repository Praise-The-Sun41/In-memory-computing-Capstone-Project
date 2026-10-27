from interface import *

def main():
    y,origin=reference_state();base=Device();Bref=float(base.weights@y[base.bi]);checks=[]
    old=ch.Device(ch.CASES['binding_barrier']);off=Device(CASES['no_contact'],Bref)
    for v in [-1.,0.,.4,1.]:
        err=float(abs(off.rhs(0,y,v)-old.rhs(0,y,v)).max())
        assert err<1e-10,err
        checks.append(dict(test='zero_contact_regression',V=v,max_rhs_error=err))
    rng=np.random.default_rng(17)
    for name in ['static_500','chemical_500','chemical_5000']:
        m=Device(CASES[name],Bref)
        for V,cc in [(.2,300e-6),(-1.,300e-6),(1.,5e-6)]:
            m.base.Icc=cc;z=y.copy();z[m.Tindex]=312.
            u=rng.normal(size=len(z));u[:m.no]*=.2;u[m.no:m.n]*=.002;u[m.bi]*=.002
            u[m.Tindex]=.3;u[m.n+1:m.n+3]=0.;u/=np.linalg.norm(u)
            h=1e-5;fd=(m.rhs(0,z+h*u,V)-m.rhs(0,z-h*u,V))/(2*h)
            exact=m.jac(0,z,V)@u;err=float(np.linalg.norm(fd-exact)/max(np.linalg.norm(fd),1e-30))
            assert err<2e-5,(name,V,err)
            d=m.diagnostic(z,V);assert abs(d['voltage_balance'])<1e-12
            assert d['Pcontact']>=0 and abs(d['device_power_balance'])<1e-12
            assert d['limited']==(cc==5e-6),(name,V,d['I'])
            checks.append(dict(test='coupled_jacobian',case=name,V=V,limited=d['limited'],relative_error=err))
    # Surface reaction plus access link obey equilibrium including +2 Vcontact.
    m=Device(CASES['chemical_500'],Bref);ox=m.base.ox
    for vc in [-.1,0.,.1]:
        z=y.copy();s=z[m.no];kt=KB*z[m.Tindex]
        target=-(ox.G[0]+m.original_pref.Ti_reaction_standard_eV+2*vc)/kt-logit(s)
        z[0]=target
        with m.context(z,vc):e=ox.evaluate(m.base.port(z[:m.n]),0.,True)
        assert abs(e['exchange'])<1e-13,(vc,e['exchange'])
        checks.append(dict(test='interface_detailed_balance',Vcontact=vc,flux=float(e['exchange'])))
    # Changing binding at fixed local oxygen changes R, not total oxidation ledger.
    m=Device(CASES['chemical_500'],Bref);z=y.copy();delta=.01
    # Exclude the one dual cell straddling 1 nm: its uniform mobile field also
    # represents oxygen outside the reporting region.
    inside=m.base.core.metal.edges[m.active+1]<=m.chem.interface_depth_nm
    z[m.bi[inside]]+=delta;z[m.mi[inside]]-=m.Cb[inside]*delta/m.Cm[m.active[inside]]
    a=m.diagnostic(y,0);b=m.diagnostic(z,0)
    assert abs(a['total_Ti_oxygen']-b['total_Ti_oxygen'])<1e-13
    assert abs(a['region_oxygen']-b['region_oxygen'])<1e-12
    assert b['Rcontact']>a['Rcontact']
    checks.append(dict(test='binding_is_not_oxygen_uptake',delta_total_oxygen=b['total_Ti_oxygen']-a['total_Ti_oxygen'],R_ratio=b['Rcontact']/a['Rcontact']))
    (P/'unit_checks.json').write_text(json.dumps(dict(all_pass=True,checks=checks),indent=2));print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
