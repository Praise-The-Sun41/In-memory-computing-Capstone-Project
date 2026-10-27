import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from native_io import R,load,reads,snapshot,dump
from native_worker import run
from validate_levels import refine_worker,probe_reads

start=time.monotonic()
while 'FAR_LEVELS_COMPLETE' not in (R/'far_levels.log').read_text(errors='replace'):
    if 'Traceback (most recent call last)' in (R/'far_levels.log').read_text(errors='replace'):raise RuntimeError('Far-level worker failed')
    if time.monotonic()-start>3600:raise TimeoutError('Far-level worker incomplete')
    time.sleep(3)

def worker(h):
    refine_worker(32,10,h)
    error=float(np.max(abs(probe_reads(32,10,h,True)-probe_reads(32,10,h))/abs(probe_reads(32,10,h,True))))
    checks=[]
    for k in [19,20]:
        y=snapshot(load(f'ref_tail_s32_w10_h{h}'),.14+.64*(k-16))
        tag=f'boundary_read8_l{k}_h{h}'
        if not (R/'results'/(tag+'.csv')).exists():
            run(f'BoundaryL{k}H{h}',y,f'List<double[]> p=new ArrayList<>();for(int q=0;q<8;q++)read(p);solve("{tag}",y,p,1e-9);')
        a=reads(load(tag));b=reads(load(f'valid_s32_w10_l{k}_h{h}_read8'))
        checks.append(dict(reset_count=k,relative_error=float(np.max(abs(a-b)/abs(a))),refined_last_G=float(a[-1])))
    return dict(history=h,maximum_program_G_relative_error=error,read8_checks=checks)

with ThreadPoolExecutor(max_workers=2) as pool:out=list(pool.map(worker,[0,3]))
dump('boundary_reference.json',dict(scope='Post-result numerical sensitivity diagnostic for histories 0 and 3, which produced the short-envelope extrema at RESET19 and RESET20. Last eight SET pulses, wait, RESET0-32 and both eight-READ branches repeated at rtol=1e-9. Histories1/2 and other stability branches were not repeated. This is not an accepted level release.',histories=out,maximum_relative_G_error=max([x['maximum_program_G_relative_error'] for x in out]+[c['relative_error'] for x in out for c in x['read8_checks']])))
print('BOUNDARY_REFERENCE_COMPLETE',flush=True)
