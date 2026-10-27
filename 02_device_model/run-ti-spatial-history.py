"""Continuous multi-cycle IV and explicitly forked read/hold histories."""
from pathlib import Path
import importlib.util,sys,json,argparse,time,hashlib
from dataclasses import asdict
import numpy as np
from scipy.special import expit
R=Path(__file__).resolve().parent;O=R/'outputs'
spec=importlib.util.spec_from_file_location('history_device',R/'ti-spatial-device.py')
d=importlib.util.module_from_spec(spec);sys.modules[spec.name]=d;spec.loader.exec_module(d)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def name(q,L,M,D,rtol,cycles):return f'ti-spatial-IV-q{q:g}-L{L}-M{M}-D{D:g}-C{cycles}-R{rtol:g}'

def save(stem,m,rows,y0,rtol,metadata,points=241):
    path=O/(stem+'.npz');mp=path.with_suffix('.json')
    if mp.exists():
        meta=json.loads(mp.read_text());assert sha(path)==meta['file_sha256']
        assert meta['device_sha256']==sha(R/'ti-spatial-device.py')
        print('EXISTS',stem,flush=True);return dict(np.load(path)),meta
    ts=[];ys=[];vs=[];kinds=[];stats=[];clock=0.;y=y0.copy();begin=time.perf_counter()
    for T,v0,v1,kind in rows:
        targets=np.linspace(0,T,points)
        if T>100 and v0==v1==0:targets=np.r_[0,np.geomspace(.001,T,points-1)]
        tt,yy,vv,y,st=d.solve(m,y,T,v0,v1,rtol,targets)
        if ts:tt,yy,vv=tt[1:],yy[1:],vv[1:]
        ts.extend(tt+clock);ys.extend(yy);vs.extend(vv);kinds.extend([kind]*len(tt))
        stats.append(dict(kind=kind,left=clock,right=clock+T,**st));clock+=T
        print(stem,kind,'elapsed',round(time.perf_counter()-begin,1),flush=True)
    yy=np.asarray(ys);vv=np.asarray(vs);dd=[m.diagnostics(z,v) for z,v in zip(yy,vv)]
    data=dict(time=np.asarray(ts),y=yy,V=vv,kind=np.asarray(kinds),oxide_edges=m.ox.edges,metal_x=m.metal.x)
    for k in dd[0]:data[k]=np.asarray([x[k] for x in dd])
    meta=dict(**metadata,parameters=asdict(m.ox.p),metal_parameters=asdict(m.metal.p),rtol=rtol,
        local_cells=m.local,segments=stats,elapsed_seconds=time.perf_counter()-begin,
        mass_drift=float(np.max(abs(data['invariant']-m.inventory(y0)))),
        max_energy_residual=float(abs(data['energy_residual']).max()),
        min_component_dissipation=float(min(data[k].min() for k in ('oxide_diss','Ti_diss','reaction_diss','half_cell_diss'))),
        max_detailed_balance=float(abs(data['detailed_balance']).max()),
        max_voltage_residual=float(abs(data['voltage_residual']).max()),
        max_electronic_power_residual=float(abs(data['electronic_power_residual']).max()),
        current_calibrated=False,clock_calibrated=False,instrument_compliance=False,
        full_local_transient_convergence_claimed=False,full_terminal_energy_ledger=False,
        device_sha256=sha(R/'ti-spatial-device.py'),core_sha256=sha(R/'formed-channel-set-v3.py'),
        script_sha256=sha(Path(__file__)))
    np.savez_compressed(path,**data);meta['file_sha256']=sha(path)
    mp.write_text(json.dumps(meta,indent=2))
    assert meta['mass_drift']<1e-6 and meta['min_component_dissipation']>-1e-9
    assert meta['max_energy_residual']<1e-7
    print('SAVED',stem,flush=True)
    return data,meta

def run(q,L,M,D,rtol,cycles,history=False):
    gate=json.loads((O/'ti-spatial-bias-verification.json').read_text())
    assert all(gate['gates'].values())
    model=d.Model(L,M,D);stem=name(q,L,M,D,rtol,cycles)
    rows=[(10,0,-1,'prepare-rise'),(10,-1,0,'prepare-fall'),
          (.2,0,.02,'initial-read-rise'),(1,.02,.02,'initial-read'),(.2,.02,0,'initial-read-fall')]
    rows += [(q,a,b,f'{kind}-{cy}') for cy in range(cycles) for kind,a,b in
        [('positive-rise',0,1),('positive-fall',1,0),('negative-rise',0,-1),('negative-fall',-1,0)]]
    data,meta=save(stem,model,rows,model.initial,rtol,
        dict(quarter_tau=q,cycles=cycles,protocol='common negative preparation and initial 20mV read; uninterrupted 0,+1,0,-1,0 cycles',
             initial_condition='old uniform oxide initial mobile filling and Ti s=0.3; all preparation rerun with spatial Ti',
             experimental_60nm_data_used=False))
    if not history:return
    for cy in sorted(set([0,cycles-1])):
        for state,kind in [('positive','positive-fall'),('negative','negative-fall')]:
            idx=np.flatnonzero(data['kind']==f'{kind}-{cy}')[-1];z=data['y'][idx].copy()
            delays=[0,10,100,1000] if cy==cycles-1 else [0]
            for sham in (False,True):
                seq=[];clock=0.
                for delay in delays:
                    if delay>clock:seq.append((delay-clock,0,0,f'hold-to-{delay}'))
                    amp=0 if sham else .02
                    seq.extend([(.2,0,amp,f'read-{delay}-rise'),(1,amp,amp,f'read-{delay}-plateau'),(.2,amp,0,f'read-{delay}-fall')]);clock=delay+1.4
                save(stem+f'-C{cy+1}-{state}-'+('sham' if sham else 'reads'),model,seq,z,min(rtol,1e-9),
                    dict(source=stem,source_sha256=meta['file_sha256'],source_time=float(data['time'][idx]),
                         cycle=cy+1,state=state,read_delays_tau=delays,sham=sham,
                         protocol='one persistent branch; read pulses feed forward into subsequent holds/reads; sham uses same durations at V=0'),points=81)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--q',type=float,default=10);p.add_argument('--local',type=int,default=256)
    p.add_argument('--metal',type=int,default=256);p.add_argument('--D',type=float,default=1e-6)
    p.add_argument('--rtol',type=float,default=1e-8);p.add_argument('--cycles',type=int,default=5);p.add_argument('--history',action='store_true')
    a=p.parse_args();run(a.q,a.local,a.metal,a.D,a.rtol,a.cycles,a.history)
