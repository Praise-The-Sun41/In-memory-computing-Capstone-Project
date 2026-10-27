from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import numpy as np
from native_io import R,load,reads,original,dump
from native_worker import run

def prepare(count,h):
    done=[n for n in range(8,count+1,8) if (R/'results'/f'set_n{n}_h{h}.csv').exists()]
    previous=max(done,default=0)
    if previous==count:
        assert len(reads(load(f'set_n{count}_h{h}')))==9
        return
    y=original(h) if previous==0 else load(f'set_n{previous}_h{h}')[-1,4:]
    body='\n'.join(f'y=solve("set_n{n}_h{h}",y,pulses(.8,8,true),1e-8);' for n in range(previous+8,count+1,8))
    run(f'Prepare{count}H{h}',y,body)

def probe(count,h):
    y=load(f'set_n{count}_h{h}')[-1,4:];parts=[]
    for wait in [0,1,10]:
        tag=f'reset_s{count}_w{wait}_h{h}'
        if (R/'results'/(tag+'.csv')).exists():
            assert len(reads(load(tag)))==17
            continue
        parts.append('{double[] z=y.clone();')
        if wait:parts.append(f'List<double[]> p=new ArrayList<>();p.add(new double[]{{{wait},0,0}});z=solve("wait_s{count}_w{wait}_h{h}",z,p,1e-8);')
        parts.append(f'solve("{tag}",z,pulses(-.8,16,true),1e-8);}}')
    if parts:run(f'Probe{count}H{h}',y,'\n'.join(parts))

def separated_sets(ids,means,max_levels=4):
    for size in range(min(max_levels,len(ids)),1,-1):
        groups=[list(c) for c in combinations(sorted(ids),size) if all((means[a]-means[b])/((means[a]+means[b])/2)>=.10 for a,b in zip(c,c[1:]))]
        if groups:return sorted(groups,key=lambda c:(-(means[c[0]]-means[c[-1]]),sum(c),c))
    return []

def summarise(count):
    stationary=[]
    for h in range(4):
        g=reads(load(f'set_n{count}_h{h}'));assert len(g)==9
        stationary.append(float(np.max(abs(np.diff(g[-5:]))/abs(g[-5:-1]))))
    candidates=[]
    for wait in [0,1,10]:
        g=np.array([reads(load(f'reset_s{count}_w{wait}_h{h}')) for h in range(4)]);assert g.shape==(4,17)
        means=g.mean(axis=0);spread=np.ptp(g,axis=0)/means;ids=np.flatnonzero(spread<=.01).tolist();groups=separated_sets(ids,means)
        candidates.append(dict(set_count=count,wait_tau=wait,curves=g.tolist(),means=means.tolist(),relative_spread=spread.tolist(),screening_passing_counts=ids,separated_groups=groups,stationary_set_response=bool(max(stationary)<=.001),set_last_four_max_step_change=stationary))
    result=dict(set_count=count,candidates=candidates)
    dump(f'screen_{count}.json',result)
    print('SCREEN_RESULT',[(x['wait_tau'],x['screening_passing_counts'],x['separated_groups'][:1],x['stationary_set_response']) for x in candidates],flush=True)
    return result

def execute(count):
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda h:prepare(count,h),range(4)))
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda h:probe(count,h),range(4)))
    return summarise(count)

if __name__=='__main__':
    import sys
    execute(int(sys.argv[1]))
