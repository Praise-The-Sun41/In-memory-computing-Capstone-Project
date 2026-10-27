import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from native_io import R,load,dump
from native_worker import run
from screen_protocol import separated_sets
from validate_levels import probe_reads,validate,refine

def main(force=None):
    result=json.loads((R/'campaign_result.json').read_text(encoding='utf-8'))
    if result['accepted']:return
    options=[]
    for n in result['screened_set_counts']:
        screen=json.loads((R/f'screen_{n}.json').read_text(encoding='utf-8'))
        for c in screen['candidates']:
            if force is not None and (n,c['wait_tau'])!=tuple(force):continue
            ids=c['screening_passing_counts']
            if c['wait_tau'] not in [1,10] or not c['stationary_set_response'] or len(ids)<2:continue
            means=np.array(c['means'])[ids];span=float(np.ptp(means)/means.mean())
            options.append((span,-n,-c['wait_tau'],c))
    if not options:
        dump('extension_decision.json',dict(executed=False,reason='No stationary fixed-wait candidate had at least two 1%-history-passing counts.'))
        return
    _,_,_,base=max(options,key=lambda x:x[:3]);n=base['set_count'];w=base['wait_tau']
    reason='Widest span among history-passing counts, without changing acceptance gates.' if force is None else 'Bounded adaptive follow-up of the protocol with already demonstrated short-window stable levels; same RESET limit of 32 and unchanged gates.'
    dump('extension_decision.json',dict(executed=True,set_count=n,wait_tau=w,reason=reason,extra_checkpoint_read_after_reset16=True))
    def worker(h):
        tag=f'tail_s{n}_w{w}_h{h}'
        if (R/'results'/(tag+'.csv')).exists():load(tag);return
        y=load(f'reset_s{n}_w{w}_h{h}')[-1,4:]
        run(f'ExtendS{n}W{w}H{h}',y,f'solve("{tag}",y,pulses(-.8,16,true),1e-8);')
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(worker,range(4)))
    curves=np.array([probe_reads(n,w,h) for h in range(4)]);assert curves.shape==(4,33)
    means=curves.mean(axis=0);spread=np.ptp(curves,axis=0)/means;ids=np.flatnonzero(spread<=.01).tolist()
    candidate=dict(base,curves=curves.tolist(),means=means.tolist(),relative_spread=spread.tolist(),screening_passing_counts=ids,separated_groups=separated_sets(ids,means),extended_reset_max=32)
    dump('extension_screen.json',candidate)
    print('EXTENSION_SCREEN',ids,candidate['separated_groups'][:1],flush=True)
    proposal=validate(candidate) if candidate['separated_groups'] else None
    numerical=refine(proposal) if proposal else None
    if 'extension' in result:
        result.setdefault('prior_extension_attempts',[]).append(result['extension'])
    result['extension']=dict(set_count=n,wait_tau=w,maximum_reset_count=32,stable_selection=proposal,numerical=numerical)
    if proposal and numerical['passed']:
        result['accepted']=True;result['selection']=dict(**proposal,numerical=numerical,extended_reset_max=32)
    dump('campaign_result.json',result);dump('campaign_progress.json',result)
    print('RESET_EXTENSION_COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':main()
