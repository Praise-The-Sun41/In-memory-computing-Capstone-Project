"""Read-only response lookup benchmark on the recorded reference path only."""
from pathlib import Path
import json,time,numpy as np
from analyse_native import load,read_levels
R=Path(__file__).resolve().parent
h=json.loads((R/'history_summary.json').read_text(encoding='utf-8'));count=h['selected'] or h['candidates'][-1]['reset_count']
g=read_levels(load(f'probe{count}_history0'));ids=np.random.default_rng(42).integers(0,len(g),size=(16,16));trials=[]
for _ in range(7):
    start=time.perf_counter()
    for k in range(10000):current=.02*g[ids]
    trials.append((time.perf_counter()-start)/10000)
np.testing.assert_allclose(current,.02*np.take(g,ids),rtol=0,atol=0)
result=dict(devices=256,median_batch_seconds=float(np.median(trials)),history_gate_pass=h['selected'] is not None,
 scope='Read-only lookup of existing pulse-count states on one recorded reference path; excludes circuit solve, programming, memory scheduling, loading and history coverage.',
 deployment_allowed=False)
(R/'lookup_benchmark.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
