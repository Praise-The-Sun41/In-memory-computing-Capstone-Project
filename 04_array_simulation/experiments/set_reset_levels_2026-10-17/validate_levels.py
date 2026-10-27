from concurrent.futures import ThreadPoolExecutor
import numpy as np
from native_io import R,load,reads,snapshot,original,dump
from native_worker import run
from screen_protocol import separated_sets

CASES=['ready','hold1','hold10','read8','sham8']

def prefix(n,w,k,h):return f'valid_s{n}_w{w}_l{k}_h{h}_'

def probe_reads(n,w,h,ref=False):
    prefix='ref_' if ref else ''
    values=reads(load(f'{prefix}reset_s{n}_w{w}_h{h}'))
    tag=f'{prefix}tail_s{n}_w{w}_h{h}'
    if (R/'results'/(tag+'.csv')).exists():values=np.r_[values,reads(load(tag))[1:]]
    return values

def probe_snapshot(n,w,h,k):
    if k<=16:return snapshot(load(f'reset_s{n}_w{w}_h{h}'),.14+.64*k)
    return snapshot(load(f'tail_s{n}_w{w}_h{h}'),.14+.64*(k-16))

def worker(n,w,k,h):
    y=probe_snapshot(n,w,h,k)
    pieces=[]
    for case in CASES:
        tag=prefix(n,w,k,h)+case
        if (R/'results'/(tag+'.csv')).exists():
            a=load(tag);assert len(reads(a))==(8 if case=='read8' else 1)
            continue
        body='List<double[]> p=new ArrayList<>();'
        if case.startswith('hold'):body+=f'p.add(new double[]{{{int(case[4:])},0,0}});read(p);'
        elif case=='read8':body+='for(int q=0;q<8;q++)read(p);'
        elif case=='sham8':body+='p.add(new double[]{.98,0,0});read(p);'
        else:body+='read(p);'
        pieces.append('{'+body+f'solve("{tag}",y,p,1e-8);'+'}')
    if pieces:run(f'ValidateS{n}W{w}L{k}H{h}',y,'\n'.join(pieces))

def analyse_level(n,w,k):
    matrix=[]
    for h in range(4):
        programmed=probe_reads(n,w,h)[k]
        matrix.append([programmed]+[reads(load(prefix(n,w,k,h)+case))[-1] for case in CASES])
    a=np.array(matrix);short=a[:,[0,1,2,4,5]]
    range_short=float(np.ptp(short)/short.mean());range_extended=float(np.ptp(a)/a.mean())
    result=dict(reset_pulses=k,columns=['program_verify']+CASES,conductance_by_history=a.tolist(),nominal_G=float(a[:,1].mean()),short_envelope_relative_range=range_short,extended_envelope_relative_range=range_extended,short_pass=range_short<=.01,extended_pass=range_extended<=.01,maximum_read_extra_change=float(np.max(abs(a[:,4]-a[:,5])/abs(a[:,5]))),maximum_read_total_change=float(np.max(abs(a[:,4]-a[:,1])/abs(a[:,1]))))
    dump(f'level_s{n}_w{w}_l{k}.json',result)
    print('LEVEL_RESULT',n,w,k,range_short,range_extended,flush=True)
    return result

def choose(results):
    means=np.zeros(max([16]+[x['reset_pulses'] for x in results])+1)
    for x in results:means[x['reset_pulses']]=x['nominal_G']
    options=[]
    for window in ['extended','short']:
        ids=[x['reset_pulses'] for x in results if x[window+'_pass']]
        groups=separated_sets(ids,means)
        if groups:options.append(dict(window=window,levels=groups[0]))
    return max(options,key=lambda x:(len(x['levels']),x['window']=='extended')) if options else None

def validate(candidate):
    n=candidate['set_count'];w=candidate['wait_tau'];first=candidate['separated_groups'][0]
    order=first+[k for k in candidate['screening_passing_counts'] if k not in first];results=[]
    for k in order:
        with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda h:worker(n,w,k,h),range(4)))
        results.append(analyse_level(n,w,k))
        dump(f'validation_s{n}_w{w}.json',dict(set_count=n,wait_tau=w,levels=results,selection=choose(results)))
        if len(results)>=len(first):
            selection=choose(results)
            if selection:return dict(set_count=n,wait_tau=w,validated_levels=results,**selection)
    return None

def refine_worker(n,w,h):
    y=original(h) if n==8 else load(f'set_n{n-8}_h{h}')[-1,4:]
    tag=f'ref_reset_s{n}_w{w}_h{h}'
    extended=(R/'results'/f'tail_s{n}_w{w}_h{h}.csv').exists()
    if (R/'results'/(tag+'.csv')).exists() and (not extended or (R/'results'/f'ref_tail_s{n}_w{w}_h{h}.csv').exists()):
        assert len(reads(load(tag)))==17
        return
    body=f'y=solve("ref_set_s{n}_h{h}",y,pulses(.8,8,true),1e-9);'
    body+=f'List<double[]> p=new ArrayList<>();p.add(new double[]{{{w},0,0}});y=solve("ref_wait_s{n}_w{w}_h{h}",y,p,1e-9);'
    body+=f'y=solve("{tag}",y,pulses(-.8,16,true),1e-9);'
    if extended:body+=f'solve("ref_tail_s{n}_w{w}_h{h}",y,pulses(-.8,16,true),1e-9);'
    run(f'RefineS{n}W{w}H{h}',y,body)

def refine(selection):
    n=selection['set_count'];w=selection['wait_tau']
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda h:refine_worker(n,w,h),range(4)))
    errors=[]
    for h in range(4):
        a=probe_reads(n,w,h,ref=True);b=probe_reads(n,w,h)
        errors.append(float(np.max(abs(a-b)/abs(a))))
    result=dict(relative_errors_by_history=errors,passed=max(errors)<=.001,scope='Last eight SET pulses, fixed wait and entire RESET probe repeated at rtol=1e-9 from the same preceding native checkpoint. Earlier preparation prefix and stability-window trajectories are not independently repeated here.')
    dump('numerical_validation.json',result)
    return result
