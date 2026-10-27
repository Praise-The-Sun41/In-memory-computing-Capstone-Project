"""New branch runner; all cases start from the same complete C0 history."""
from interface import *
import time, argparse
from scipy.integrate import solve_ivp

def solve(m,rows,y0,name,rtol=2e-9):
    path=P/(name+'.npz');metadata=P/(name+'.json')
    config=dict(chemistry=asdict(m.chem),thermal=asdict(m.base.cfg),mode=m.base.mode,
                L=m.base.core.local,M=len(m.Cm),rtol=rtol,rows=rows,
                source_sha=sha(P/'interface.py'),runner_sha=sha(__file__),chemistry_source_sha=sha(OLD/'chemistry.py'),thermal_source_sha=sha(ch.BASE/'coupled.py'),contact=asdict(m.contact),reference_b=m.Bref,
                initial_sha=hashlib.sha256(y0.tobytes()).hexdigest())
    if path.exists():
        meta=json.loads(metadata.read_text())
        assert meta['config']==json.loads(json.dumps(config)), 'Cache configuration mismatch'
        assert meta['data_sha']==sha(path)
        return dict(np.load(path)),meta
    y=y0.copy();parts=[];stats=[];t0=0.;tic=time.perf_counter();node_mass=0.;node_min=1.;node_max=0.
    inv0=m.diagnostic(y0,rows[0][1])['inventory']
    for duration,v0,v1,label,shape in rows:
        def voltage(t):
            u=t/duration;return v0+(v1-v0)*(u*u*(3-2*u) if shape=='smooth' else u)
        sol=solve_ivp(lambda t,z:m.rhs(t,z,voltage(t)),(0,duration),y,method='BDF',
            jac=lambda t,z:m.jac(t,z,voltage(t)),rtol=rtol,atol=rtol*1e-3,
            max_step=.05 if duration<10 else .5,dense_output=True)
        if not sol.success:raise RuntimeError(sol.message)
        occ=np.r_[expit(sol.y[:m.no]).ravel(),sol.y[m.no:m.n].ravel(),sol.y[m.nb0:].ravel()]
        node_min=min(node_min,float(occ.min()));node_max=max(node_max,float(occ.max()))
        iv=m.base.ox.C@expit(sol.y[:m.no])-m.Cm@sol.y[m.no:m.n]-m.Cb@sol.y[m.nb0:]
        node_mass=max(node_mass,float(abs(iv-inv0).max()))
        assert 0<node_min and node_max<1 and node_mass<2e-6
        tt=np.linspace(0,duration,121);sl=slice(None) if not parts else slice(1,None)
        parts.append(dict(t=t0+tt[sl],y=sol.sol(tt[sl]).T,Vs=voltage(tt[sl]),kind=np.full(len(tt[sl]),label)))
        stats.append(dict(label=label,steps=len(sol.t),nfev=sol.nfev,njev=sol.njev))
        y=sol.y[:,-1];t0+=duration
        print(name,label,round(time.perf_counter()-tic,1),flush=True)
    out={k:np.concatenate([v[k] for v in parts]) for k in parts[0]}
    ds=[m.diagnostic(y,v) for y,v in zip(out['y'],out['Vs'])]
    out.update({k:np.array([d[k] for d in ds]) for k in ds[0]})
    out['G300']=np.array([m.state_G(y) for y in out['y']])
    checks=dict(internal_node_mass_error=node_mass,internal_node_minimum=node_min,internal_node_maximum=node_max,
        sampled_mass_error=float(abs(out['inventory']-inv0).max()),
        thermal_balance_Wtau=float(abs(out['thermal_balance']).max()),
        power_balance_W=float(max(abs(out['power_balance']).max(),abs(out['device_power_balance']).max())),
        voltage_balance_V=float(abs(out['voltage_balance']).max()),
        current_excess_A=float(max(0,abs(out['I']).max()-m.base.Icc)),
        minimum_binding_dissipation=float(out['binding_dissipation'].min()))
    assert checks['thermal_balance_Wtau']<1e-8 and checks['power_balance_W']<1e-12
    assert checks['voltage_balance_V']<1e-10 and checks['current_excess_A']<1e-12
    assert checks['minimum_binding_dissipation']>-1e-13
    checks['minimum_ionic_dissipation']=float(out['ionic_dissipation'].min())
    checks['oxide_voltage_balance_V']=float(abs(out['oxide_voltage_balance']).max())
    assert checks['minimum_ionic_dissipation']>-1e-11 and checks['oxide_voltage_balance_V']<1e-10
    np.savez_compressed(path,**out)
    meta=dict(config=config,checks=checks,stats=stats,data_sha=sha(path),elapsed_s=time.perf_counter()-tic,
        COMSOL_verified=False,experiment_calibrated=False,phase_identified=False,absolute_oxygen_calibrated=False,
        thermal_scope='Electronic Joule heat only; reaction heat not converted without an absolute oxygen amount scale.')
    metadata.write_text(json.dumps(meta,indent=2));return out,meta


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',choices=CASES,required=True)
    p.add_argument('--cycles',type=int,default=5);p.add_argument('--rtol',type=float,default=2e-9)
    p.add_argument('--history',action='store_true');a=p.parse_args()
    y,origin=reference_state();temp=Device();B=float(temp.weights@y[temp.bi])
    m=Device(CASES[a.case],reference_b=B)
    stem=f'{a.case}-C{a.cycles}-R{a.rtol:g}'
    cyc,meta=solve(m,et.protocol(a.cycles)[5:],y,stem,a.rtol)
    meta['origin']=origin;(P/(stem+'.json')).write_text(json.dumps(meta,indent=2))
    if a.history:
        rows=[]
        for label,V in [('SET',1.),('RESET',-1.)]:
            rows += [(2,0,V,label+'-rise','linear'),(2,V,0,label+'-fall','linear'),
                (100,0,0,label+'-hold','linear'),(.2,0,.02,label+'-read-rise','linear'),
                (1,.02,.02,label+'-read','linear'),(.2,.02,0,label+'-read-fall','linear')]
        _,meta=solve(m,rows,cyc['y'][-1],a.case+'-write-hold',a.rtol)
        meta['parent_sha']=sha(P/(stem+'.npz'))
        (P/(a.case+'-write-hold.json')).write_text(json.dumps(meta,indent=2))
