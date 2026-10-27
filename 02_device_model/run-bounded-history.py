"""Bounded closure full histories; independent Ti capacity/availability/transport cases."""
from pathlib import Path
from dataclasses import replace
import importlib.util,sys,json,hashlib,argparse
import numpy as np
R=Path(__file__).resolve().parent;O=R/'outputs'
def module(name,file):
    s=importlib.util.spec_from_file_location(name,R/file);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
f=module('bounded_history_model','bounded-span-field.py');h=module('bounded_history_io','run-ti-spatial-history.py')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
BASE=dict(capacity_scale=1.,initial_oxygen=.3,metal_D=1e-6,Ti_nm=30.)
CASES={k:dict(BASE,**v) for k,v in {
 'ref':{},'capacity025':{'capacity_scale':.25},'capacity4':{'capacity_scale':4.},
 'oxygen01':{'initial_oxygen':.1},'oxygen08':{'initial_oxygen':.8},
 'Dslow':{'metal_D':1e-8},'Dfast':{'metal_D':1e-4},
 'Ti15':{'Ti_nm':15.},'Ti60':{'Ti_nm':60.}}.items()}

def make_model(case,L,M):
    c=CASES[case];m=f.Model(L,M,D=c['metal_D'])
    p=replace(m.ox.p,Ti_nm=c['Ti_nm'],Ti_initial_oxygen_fraction=c['initial_oxygen'])
    m.ox=f.Oxide(p,m.ox.edges,'span_shift');m.n=m.ox.N
    nodes=int(round((M-1)*np.sqrt(c['Ti_nm']/30)))+1
    mp=replace(m.metal.p,cells=nodes,thickness_nm=c['Ti_nm'],density_per_nm=1.14*c['capacity_scale'],D_nm2_per_tau=c['metal_D'])
    m.metal=f.d.c.ti.Reservoir(mp);m.nm=nodes
    m.initial=np.r_[m.ox.initial[:-1],np.full(nodes,c['initial_oxygen'])]
    return m

def stem(case,L,M,rtol):return f'bounded-q2-{case}-L{L}-M{M}-C5-R{rtol:g}'
def read(st):
    path=O/(st+'.npz');meta=json.loads(path.with_suffix('.json').read_text())
    assert sha(path)==meta['file_sha256']
    assert meta['bounded_sha256']==sha(R/'bounded-span-field.py')
    assert meta['driver_sha256']==sha(Path(__file__))
    return dict(np.load(path)),meta
def save(st,m,rows,y,rtol,metadata,points):
    if (O/(st+'.json')).exists():print('EXISTS',st,flush=True);return read(st)
    out,meta=h.save(st,m,rows,y,rtol,dict(metadata,bounded_sha256=sha(R/'bounded-span-field.py'),driver_sha256=sha(Path(__file__))),points=points)
    assert np.max(out['drift_cap_ratio'])<=1+1e-11
    return out,meta

def hold_rows(sham=False):
    rows=[]
    for phase,amp in [('SET',1.),('RESET',-1.)]:
        rows.extend([(2,0,amp,phase+'-rise'),(2,amp,0,phase+'-fall')]);clock=0.
        for delay in (0,10,100,1000):
            if delay>clock:rows.append((delay-clock,0,0,f'{phase}-hold-to-{delay}'))
            v=0 if sham else .02
            rows.extend([(.2,0,v,f'{phase}-read-{delay}-rise'),(1,v,v,f'{phase}-read-{delay}-plateau'),(.2,v,0,f'{phase}-read-{delay}-fall')]);clock=delay+1.4
    return rows

def run(case,L,M,rtol,sham=False):
    gate=json.loads((O/'bounded-span-verification.json').read_text())
    assert gate['all_pass'] and gate['candidate_sha256']==sha(R/'bounded-span-field.py')
    m=make_model(case,L,M);st=stem(case,L,M,rtol)
    seq=[(10,0,-1,'prepare-rise'),(10,-1,0,'prepare-fall'),(.2,0,.02,'initial-read-rise'),(1,.02,.02,'initial-read'),(.2,.02,0,'initial-read-fall')]
    seq += [(2,a,b,f'{kind}-{cy}') for cy in range(5) for kind,a,b in [('positive-rise',0,1),('positive-fall',1,0),('negative-rise',0,-1),('negative-fall',-1,0)]]
    data,meta=save(st,m,seq,m.initial,rtol,dict(case=case,Ti_scenario=CASES[case],quarter_tau=2.,cycles=5,
        protocol='own fresh initial state, 10+10 tau negative preparation, initial read, five uninterrupted bipolar q2 cycles',
        synthetic_parameters=True,experimental_60nm_data_used=False,initial_state_reprepared=True),241)
    for blank in ([False,True] if sham else [False]):
        tail=st+('-hold-sham' if blank else '-hold-reads')
        save(tail,m,hold_rows(blank),data['y'][-1].copy(),rtol,dict(case=case,Ti_scenario=CASES[case],source=st,
            source_sha256=meta['file_sha256'],source_time=float(data['time'][-1]),sham=blank,
            protocol='after own C5: SET, 1000 tau hold with read slots, RESET, 1000 tau hold; complete state continued',
            read_start_delays_tau=[0,10,100,1000],read_voltage_V=0 if blank else .02,last_read_ends_tau=1001.4,
            synthetic_parameters=True,experimental_60nm_data_used=False),81)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',choices=list(CASES),default='ref');p.add_argument('--local',type=int,default=256)
    p.add_argument('--metal',type=int,default=256);p.add_argument('--rtol',type=float,default=1e-9);p.add_argument('--sham',action='store_true')
    a=p.parse_args();run(a.case,a.local,a.metal,a.rtol,a.sham)
