"""Continuous cycles, same-age idle control, and equally aged forked retention."""
from model import *
from scipy.integrate import solve_ivp
import time,argparse

def solve(s,rows,y0,stem,rtol=2e-9):
    folder=P/s.name;folder.mkdir(exist_ok=True);p=folder/(stem+'.npz')
    cfg=dict(system=s.config,rows=rows,rtol=rtol,sources=sources(),runner_sha=sha(__file__),
             initial_sha=hashlib.sha256(y0.tobytes()).hexdigest())
    if p.exists():
        meta=json.loads(p.with_suffix('.json').read_text());assert meta['config']==json.loads(json.dumps(cfg))
        assert sha(p)==meta['data_sha'];return dict(np.load(p)),meta
    y=y0.copy();parts=[];t0=0.;tic=time.perf_counter();mass=0.;lo=1.;hi=0.;stats=[]
    inv0=s.inventory(y[None,:])[0]
    for duration,v0,v1,label in rows:
        vf=lambda t:v0+(v1-v0)*t/duration
        sol=solve_ivp(lambda t,z:s.rhs(t,z,vf(t)),(0,duration),y,method='BDF',jac=lambda t,z:s.jac(t,z,vf(t)),
            rtol=rtol,atol=rtol*1e-3,max_step=.05 if duration<10 else 1.,dense_output=True)
        if not sol.success:raise RuntimeError(sol.message)
        occ=s.occupancy(sol.y.T);lo=min(lo,float(occ.min()));hi=max(hi,float(occ.max()))
        mass=max(mass,float(abs(s.inventory(sol.y.T)-inv0).max()))
        assert 0<lo and hi<1 and mass<2e-6,(lo,hi,mass)
        tt=np.linspace(0,duration,121);sl=slice(None) if not parts else slice(1,None)
        parts.append(dict(t=t0+tt[sl],y=sol.sol(tt[sl]).T,Vs=vf(tt[sl]),kind=np.full(len(tt[sl]),label)))
        stats.append(dict(label=label,steps=len(sol.t),nfev=sol.nfev,njev=sol.njev))
        t0+=duration;y=sol.y[:,-1]
        print(s.name,stem,label,round(time.perf_counter()-tic,1),flush=True)
    out={k:np.concatenate([v[k] for v in parts]) for k in parts[0]}
    ds=[s.diagnostic(y,V) for y,V in zip(out['y'],out['Vs'])]
    out.update({k:np.array([d[k] for d in ds]) for k in ds[0]})
    out['G300']=np.array([s.G300(y) for y in out['y']])
    check=dict(mass_error=mass,min_occupancy=lo,max_occupancy=hi,
        heat_error=float(abs(out['thermal_balance']).max()),power_error=float(max(abs(out['power_balance']).max(),abs(out['device_power_balance']).max())),
        voltage_error=float(abs(out['voltage_balance']).max()),max_T=float(out['T'].max()),
        current_excess=float(max(0,abs(out['I']).max()-s.base.Icc)),min_ionic_dissipation=float(out['ionic_dissipation'].min()))
    assert check['heat_error']<1e-8 and check['power_error']<1e-12 and check['voltage_error']<1e-10
    assert check['current_excess']<1e-12 and check['min_ionic_dissipation']>-1e-10
    if s.kind=='Ti':
        check['min_binding_dissipation']=float(out['binding_dissipation'].min());assert check['min_binding_dissipation']>-1e-13
    else:assert np.all(out['exchange_flux']==0) and np.all(out['binding_flux']==0)
    assert np.isfinite(out['I']).all() and np.isfinite(out['y']).all()
    np.savez_compressed(p,**out)
    meta=dict(config=cfg,checks=check,stats=stats,data_sha=sha(p),elapsed_s=time.perf_counter()-tic)
    p.with_suffix('.json').write_text(json.dumps(meta,indent=2));return out,meta

def cycles(first,last):
    return [(2,a,b,f'C{k}-{label}') for k in range(first,last+1) for a,b,label in
        [(0,1,'pos-rise'),(1,0,'pos-fall'),(0,-1,'neg-rise'),(-1,0,'neg-fall')]]

def run(case):
    s=System(case);folder=P/case;folder.mkdir(exist_ok=True)
    parents={}
    if s.reuse:
        p=SRC/(s.reuse+'-C5-R2e-09.npz');meta=json.loads(p.with_suffix('.json').read_text())
        assert meta['data_sha']==sha(p) and meta['config']['source_sha']==sha(SRC/'interface.py')
        assert meta['config']['chemistry_source_sha']==sha(it.OLD/'chemistry.py')
        assert meta['config']['thermal_source_sha']==sha(it.ch.BASE/'coupled.py')
        old=dict(np.load(p));assert np.array_equal(old['y'][0],s.y0)
        assert abs(s.diagnostic(old['y'][-1],0)['G']/old['G'][-1]-1)<1e-12
        parents[str(p)]=sha(p)
    else:old,_=solve(s,cycles(1,5),s.y0,'C1-C5')
    ext,_=solve(s,cycles(6,20),old['y'][-1],'C6-C20');ext['t']+=40
    shared=set(old)&set(ext);c={k:np.concatenate([old[k],ext[k][1:]]) for k in shared}
    np.savez_compressed(folder/'cycles.npz',**c)
    idle,_=solve(s,[(160,0,0,'idle160')],s.y0,'idle160')
    # Both writes start at EXACTLY the same C20 state, have equal duration,
    # and retain their own full chemical and thermal state during the hold.
    for name,V in [('positive',1.),('negative',-1.)]:
        w,_=solve(s,[(2,0,V,'write-rise'),(2,V,0,'write-fall')],c['y'][-1],name+'-write')
        h,_=solve(s,[(1,0,0,'hold1'),(9,0,0,'hold10'),(90,0,0,'hold100'),(900,0,0,'hold1000')],w['y'][-1],name+'-hold')
        # Terminal read branches do not feed back into the unperturbed hold curve.
        for age,index in [(0,0),(1000,-1)]:
            solve(s,[(.2,0,.02,'read-rise'),(1,.02,.02,'read'),(.2,.02,0,'read-fall')],h['y'][index],f'{name}-read{age}')
    manifest=dict(case=case,system=s.config,sources=sources(),parents=parents,cycles_sha=sha(folder/'cycles.npz'),
        completed=True,cycles=20,retention_ages_tau=[0,1,10,100,1000],matched_age_retention=True,
        experimental_lifetime_claim=False,new_COMSOL_history=False)
    (folder/'complete.json').write_text(json.dumps(manifest,indent=2));print('CASE_COMPLETE',case,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--case',choices=CASES,required=True);v=a.parse_args();run(v.case)
