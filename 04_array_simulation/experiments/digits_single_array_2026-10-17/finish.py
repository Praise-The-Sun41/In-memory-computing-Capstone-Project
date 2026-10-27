"""Add an explicit row-serial architecture boundary; do not extrapolate its accuracy."""
import run as s
import json,hashlib,time
import numpy as np
from sklearn.datasets import load_digits

def main():
    started=time.perf_counter();r=json.loads((s.R/'results.json').read_text());base=r['configurations'][0]
    z=np.load(s.PRE/'binary_model.npz');p={k:z[k] for k in ['w1','w2','a1','a2','b1','b2']}
    ts=s.tiles([np.where(p[k]>=0,1.,-1.) for k in ['w1','w2']]);pixels=load_digits().data[z['test']].astype(int)
    c=dict(base['config'],name='row_serial_programming',programming_rows_parallel=1)
    sch=s.schedule(c,ts,pixels,[np.exp(p['a1']),np.exp(p['a2'])],[p['b1'],p['b2']])
    extra=450*10*15*43.48*c['tau_s']
    assert abs(sch['total_s']-base['timing']['total_s']-extra)<1e-9
    row=dict(config=c,timing={k:v for k,v in sch.items() if k not in ['trace','starts','replayed_scores']},
             max_read_age_tau=base['max_read_age_tau']+15*43.48,within_marginal_table_envelope=False,scenarios=[],host_runtime_s=time.perf_counter()-started)
    r['configurations']=[v for v in r['configurations'] if v['config']['name']!=c['name']]+[row]
    note='Row-serial sensitivity repeats the complete43.48 tau conditioning protocol for each row (not an optimized interleaved programming scheduler). Earliest rows then wait15x43.48 tau. It has timing only: native retention data cannot support accuracy there. Full-array independently masked writing is an optimistic architecture bound, not a demonstrated conventional1T1R implementation.'
    if note not in r['scope']:r['scope'].append(note)
    r['row_serial_timing_check_passed']=True
    s.dump('results.json',r)
    protocol=json.loads((s.R/'protocol.json').read_text());protocol['configurations']=[v['config'] for v in r['configurations']];protocol['scope']=r['scope'];s.dump('protocol.json',protocol)
    trace=json.loads((s.R/'trace_first_image.json').read_text());s.report(r,trace)
    provenance=json.loads((s.R/'provenance.json').read_text())
    for item in provenance:item['sha256']=hashlib.sha256(s.Path(item['path']).read_bytes()).hexdigest()
    for name in ['audit.py','finish.py']:provenance.append(dict(path=str(s.R/name),sha256=hashlib.sha256((s.R/name).read_bytes()).hexdigest()))
    s.dump('provenance.json',provenance)
    print(json.dumps(row,indent=2))

if __name__=='__main__':main()
