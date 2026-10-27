"""Single physical array, sample-major execution; conditional empirical LUT study."""
from pathlib import Path
from collections import OrderedDict, Counter
import json, time, hashlib, sys
import numpy as np
from sklearn.datasets import load_digits
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parent
PRE=R.parent/'digits_hidden_binary_2026-10-17'
PHY=R.parent/'set_reset_levels_2026-10-17'
sys.path.insert(0,str(PHY))
from native_io import load_path, read_ids

def dump(name,x): (R/name).write_text(json.dumps(x,indent=2),encoding='utf-8')

class Memory:
    """Serialized, write-through, no-write-allocate memory with real byte storage."""
    def __init__(self,c):
        self.c=c; self.data=bytearray(131072); self.next=0; self.cache=OrderedDict(); self.count=Counter()
    def allocate(self,a):
        a=np.ascontiguousarray(a); p=(self.next+63)//64*64; self.next=p+a.nbytes
        assert self.next<=len(self.data)
        self.data[p:self.next]=a.tobytes() # Pre-existing dataset/weights: not inference traffic.
        return p
    def access(self,p,n,write=None):
        assert 0<=p<=p+n<=self.next
        dt=0.; self.count['write_requests' if write is not None else 'read_requests']+=1
        self.count['write_bytes' if write is not None else 'read_bytes']+=n
        if write is not None:
            assert len(write)==n; self.data[p:p+n]=write
            dt=self.c['latency_s']+n/self.c['bandwidth_Bps'];self.count['dram_write_bytes']+=n
            # Cache holds line tags; bytes always come from the coherent backing store.
        else:
            for line in range(p//64,(p+n-1)//64+1):
                if line in self.cache:
                    self.count['hits']+=1;dt+=2e-9;self.cache.move_to_end(line)
                else:
                    self.count['misses']+=1;self.count['dram_read_bytes']+=64
                    dt+=self.c['latency_s']+64/self.c['bandwidth_Bps']
                    if self.c['cache_bytes']:
                        self.cache[line]=True
                        if len(self.cache)>self.c['cache_bytes']//64:self.cache.popitem(last=False);self.count['evictions']+=1
        return dt,bytes(self.data[p:p+n])

def tiles(signs):
    out=[]
    for l,s in enumerate(signs):
        for rb in range(s.shape[0]//16):
            for cb in range((s.shape[1]+7)//8):
                ss=s[rb*16:rb*16+16,cb*8:min(cb*8+8,s.shape[1])]
                high=np.zeros((16,16),bool)
                high[:,:2*ss.shape[1]:2]=ss>0;high[:,1:2*ss.shape[1]:2]=ss<0
                out.append(dict(layer=l,row=rb,col=cb,outputs=ss.shape[1],high=high,packed=np.packbits(high).tobytes()))
    return out

def program_events():
    events=[]
    for block in range(4):
        events.append(('SET checkpoint read',.14))
        for k in range(8):events.extend([('SET pulse + gap',.5),('SET verify read',.14)])
    events.extend([('Fixed preparation hold',10.),('RESET checkpoint read',.14)])
    for k in range(1,20):
        if k==17:events.append(('RESET16 extra checkpoint read',.14))
        events.extend([(f'RESET {k} pulse + gap',.5),(f'RESET {k} verify read',.14)])
    assert abs(sum(x[1] for x in events)-43.48)<1e-10
    return events

def schedule(c,ts,pixels,gains,biases,compute=None):
    m=Memory(c);weights=[m.allocate(np.frombuffer(t['packed'],np.uint8)) for t in ts]
    inp=m.allocate(pixels.astype(np.uint8));hidden=m.allocate(np.zeros(16,np.uint8))
    acc=m.allocate(np.zeros(16,np.float64));outputs=m.allocate(np.zeros((len(pixels),10),np.float64))
    params=m.allocate(np.concatenate(gains+biases).astype(np.float64))
    now=0.;totals=Counter();trace=[];starts=np.zeros((len(pixels),10));bounds=np.zeros((len(pixels),2));replayed=[]
    def event(kind,dt,sample=-1,tile=-1,detail='',address=None,n=None):
        nonlocal now
        if sample<1:trace.append(dict(start_s=now,duration_s=dt,kind=kind,sample=sample,tile=tile,detail=detail,address=address,bytes=n))
        now+=dt;totals[kind]+=dt
    def mem(p,n,write=None,sample=-1,tile=-1,detail=''):
        dt,data=m.access(p,n,write);event('Memory',dt,sample,tile,detail,p,n);return data
    mem(params,8*52,detail='Load fixed digital gains and biases into registers')
    for i in range(len(pixels)):
        bounds[i,0]=now
        mem(acc,128,bytes(128),i,detail='Clear hidden partial sums')
        for j,t in enumerate(ts):
            assert mem(weights[j],32,sample=i,tile=j,detail='Fetch packed two-level tile')==t['packed']
            for group in range(16//c.get('programming_rows_parallel',16)):
                for name,duration in program_events():event('Programming',duration*c['tau_s'],i,j,f'Group {group}: {name}')
            # The high level finished 4.62 tau before this common endpoint.
            program_end=now
            p=inp+i*64+t['row']*16 if t['layer']==0 else hidden
            data=mem(p,16,sample=i,tile=j,detail='Fetch 16 input activations into DAC bit-plane buffer')
            if t['layer']==0:assert np.array_equal(np.frombuffer(data,np.uint8),pixels[i,t['row']*16:(t['row']+1)*16])
            dest=acc+t['col']*64 if t['layer']==0 else outputs+i*80+t['col']*64
            partial=mem(dest,t['outputs']*8,sample=i,tile=j,detail='Fetch partial sums') if t['layer']==0 else bytes(t['outputs']*8)
            starts[i,j]=(now-program_end)/c['tau_s']
            for bit in range(5 if t['layer']==0 else 8):
                event('Readout and control',c['slot_overhead_s'],i,j,f'Bit {bit}: DAC / ADC / digital accumulation budget')
                event('Array reads',.14*c['tau_s'],i,j,f'Bit {bit}: fixed +0.02 V pulse')
            payload=bytes(t['outputs']*8) if compute is None else (np.frombuffer(partial,np.float64)+compute(j,np.frombuffer(data,np.uint8),starts[i,j])).astype(np.float64).tobytes()
            mem(dest,t['outputs']*8,payload,i,j,'Store partial sums or output logits')
            if j==7:
                sums=mem(acc,128,sample=i,detail='Fetch complete hidden sums')
                event('Activation',c['activation_s'],i,detail='Bias, clipped ReLU, 8-bit conversion')
                payload=bytes(16) if compute is None else np.rint(np.clip(np.frombuffer(sums,np.float64)+biases[0],0,1)*255).astype(np.uint8).tobytes()
                mem(hidden,16,payload,i,detail='Store hidden activations')
        logits=mem(outputs+i*80,80,sample=i,detail='Fetch logits for argmax')
        if compute is not None:replayed.append(np.frombuffer(logits,np.float64)+biases[1])
        event('Activation',c['activation_s'],i,detail='Final bias and argmax')
        bounds[i,1]=now
    return dict(total_s=now,breakdown_s=dict(totals),memory=dict(m.count),allocated_bytes=m.next,
                sample_latency_s=dict(minimum=float(np.diff(bounds).min()),mean=float(np.diff(bounds).mean()),maximum=float(np.diff(bounds).max())),
                trace=trace,starts=starts,replayed_scores=replayed)

def lut():
    levels=[json.loads((PHY/f'level_s32_w10_l{k}.json').read_text()) for k in [12,19]]
    values=np.array([v['conductance_by_history'] for v in levels]); times=np.array([-.02,.12,1.10,1.12,10.12])
    hold=values[:,:,[0,1,5,2,3]]; reads=np.empty((2,4,8));files=[]
    for l,k in enumerate([12,19]):
        for h in range(4):
            f=PHY/'results'/f'valid_s32_w10_l{k}_h{h}_read8.csv';a=load_path(f);ids=read_ids(a)
            assert len(ids)==8; assert np.allclose(a[ids,0],.12+.14*np.arange(8))
            reads[l,h]=a[ids,1];files.append(f)
    residual=np.empty_like(reads)
    for l in range(2):
        for h in range(4):residual[l,h]=reads[l,h]-np.interp(.12+.14*np.arange(8),times,hold[l,h])
    assert np.max(abs(residual[:,:,0]))<1e-20
    return levels,times,hold,residual,files

def infer(mode,c,sch,ts,p,pixels,hmap,table):
    levels,times,hold,residual,_=table; delta=levels[0]['nominal_G']-levels[1]['nominal_G']
    gains=[np.exp(p['a1']),np.exp(p['a2'])];N=len(pixels);hidden_sum=np.zeros((N,16));scores=np.zeros((N,10));hidden=None
    for j,t in enumerate(ts):
        l=t['layer'];o=t['outputs'];high=t['high'][:,:2*o];hm=hmap[:,:2*o]
        x=pixels[:,t['row']*16:(t['row']+1)*16] if l==0 else hidden
        counts=np.zeros((N,16),int);out=np.zeros((N,o));bits=5 if l==0 else 8
        for b in range(bits):
            on=(x>>b)&1;counts+=on
            if mode=='ideal':g=np.where(high,levels[0]['nominal_G'],levels[1]['nominal_G'])[None,:,:]
            elif mode=='static':g=np.where(high,hold[0,hm,1],hold[1,hm,1])[None,:,:]
            else:
                age=sch['starts'][:,j]+c['slot_overhead_s']/c['tau_s']*(b+1)+.14*b+.12
                g=np.empty((N,16,2*o))
                for level in range(2):
                    for h in range(4):
                        mask=((high if level==0 else ~high)&(hm==h))
                        if not mask.any():continue
                        aa=age+(4.62 if level==0 else 0.)
                        assert aa.min()>=times[0] and aa.max()<=times[-1]+1e-8
                        v=np.interp(aa,times,hold[level,h])[:,None,None]+np.zeros((N,16,2*o))
                        if mode=='dynamic':
                            # Empirical excess read effect vs equal-age hold. Transfer to
                            # gapped reads is a separability assumption, not native validation.
                            extra=residual[level,h,np.maximum(counts-1,0)]
                            v+=extra[:,:,None]
                        g[:,mask]=v[:,mask]
            diff=g[:,:,0::2]-g[:,:,1::2]
            out+=np.einsum('nr,nro->no',on,diff)/(delta)*(2**b/(16 if l==0 else 255))
        target=hidden_sum if l==0 else scores;cc=slice(t['col']*8,t['col']*8+o)
        target[:,cc]+=out*gains[l][cc]
        if j==7:hidden=np.rint(np.clip(hidden_sum+p['b1'],0,1)*255).astype(int)
    return scores+p['b2']

def main():
    started=time.perf_counter();z=np.load(PRE/'binary_model.npz');p={k:z[k] for k in ['w1','w2','a1','a2','b1','b2']}
    d=load_digits();pixels=d.data[z['test']].astype(int);y=d.target[z['test']];signs=[np.where(p[k]>=0,1.,-1.) for k in ['w1','w2']];ts=tiles(signs);table=lut()
    base=dict(name='baseline',tau_s=1e-6,latency_s=50e-9,bandwidth_Bps=1e9,cache_bytes=512,slot_overhead_s=20e-9,activation_s=100e-9)
    configs=[base,dict(base,name='slow_memory',latency_s=200e-9,bandwidth_Bps=1e8),dict(base,name='fast_memory',latency_s=10e-9,bandwidth_Bps=1e10),dict(base,name='no_cache',cache_bytes=0),dict(base,name='tau_100ns',tau_s=1e-7),dict(base,name='tau_10ns',tau_s=1e-8),dict(base,name='tau_10us',tau_s=1e-5)]
    scope=[
      'Conditional system simulation using existing native COMSOL tables; no new device solve, retraining or measured hardware timing.',
      'Tau is not calibrated to seconds. All tau-to-second conversions, memory speeds and peripheral budgets are explicit assumptions.',
      'One 16x16 array, sample-major: reload all 10 tiles for every image. Packed 1-bit/device weight map, 8-bit activations, float64 digital sums. Serial execution, no prefetch overlap.',
      'Ideal independently masked devices (1T1R-like isolation), fully parallel SET and masked RESET across the array. No transistor, wire, ADC noise, half-select or supply-current model. Serial per-cell writing would have much longer unvalidated retention.',
      'Each tile: four SET8 blocks including verification, hold10, RESET19 including verification and extra RESET16 checkpoint. High cells stop after RESET12 and wait4.62 tau. Programming cost43.48 tau; fixed pulse counts, no adaptive verify retry.',
      'A physical-cell history map is reused on all tiles. Every reprogramming is assumed to return to the corresponding characterized protocol-conditioned state. Closure under4500 arbitrary rewrite cycles is NOT established by the four original histories.',
      'Hold conductance is piecewise-linear through program/ready/sham8/hold1/hold10 observations. Excess read disturbance is interpolated from read8 minus equal-age hold, indexed by energized-row read count. Additivity and transfer to gapped reads are unvalidated approximations.',
      'Coverage is only the marginal age <=10.12 tau and <=8 reads envelope, not validation of all joint waiting/read histories. Out-of-envelope configurations have timing only and no accuracy.',
      'All100 mixed seeds plus16 coherent branch-pair combinations use the same four history labels. The physical mapping differs from earlier independent logical-weight maps because one array is reused.',
      'The450-image test cohort is reused. No test-based training or calibration. Setup population of weights/dataset is excluded; inference parameter load, input/output traffic and all reprogramming are included.',
      'Conductance/current units are uncalibrated. Fixed read amplitude0.02 V is modeled with ideal linear current sums; memory latency limits are simulated, host Python runtime is reported separately.'
    ]
    dump('protocol.json',dict(configurations=configs,scope=scope,architecture=[64,16,10],logical_tiles=10,physical_arrays=1,images=len(y),weight_levels=[12,19],read_slots_per_image=56,program_tau_per_tile=43.48))
    maps=[(f'paired_{hp}_{hn}',np.tile(np.array([hp,hn]*8),(16,1))) for hp in range(4) for hn in range(4)]
    maps += [(f'mixed_{seed}',np.random.default_rng(seed).integers(0,4,(16,16))) for seed in range(100)]
    def metrics(scores):
        pred=scores.argmax(1);return dict(correct=int((pred==y).sum()),accuracy=float((pred==y).mean()),changes_vs_ideal=int((pred!=ideal_pred).sum()))
    ideal_hidden=np.rint(np.clip((pixels/16)@(signs[0]*np.exp(p['a1']))+p['b1'],0,1)*255)/255
    ideal_scores=ideal_hidden@(signs[1]*np.exp(p['a2']))+p['b2'];ideal_pred=ideal_scores.argmax(1)
    allresults=[];predictions={};checks={};schedules={}
    for c in configs:
        t0=time.perf_counter();sch=schedule(c,ts,pixels,[np.exp(p['a1']),np.exp(p['a2'])],[p['b1'],p['b2']]);schedules[c['name']]=sch
        maxage=max(float(sch['starts'][:,j].max())+4.62+.12+.14*((5 if t['layer']==0 else 8)-1)+c['slot_overhead_s']/c['tau_s']*(5 if t['layer']==0 else 8) for j,t in enumerate(ts))
        valid=maxage<=10.12+1e-8
        result=dict(config=c,timing={k:v for k,v in sch.items() if k not in ['trace','starts','replayed_scores']},max_read_age_tau=maxage,within_marginal_table_envelope=valid,scenarios=[])
        if valid:
            reconstructed=infer('ideal',c,sch,ts,p,pixels,maps[0][1],table)
            checks[c['name']]=float(abs(reconstructed-ideal_scores).max());assert checks[c['name']]<1e-10
            for name,hm in maps:
                row=dict(history_map=name)
                for mode in ['static','wait','dynamic']:
                    scores=infer(mode,c,sch,ts,p,pixels,hm,table);row[mode]=metrics(scores)
                    if c['name']=='baseline':predictions[name+'_'+mode]=scores.argmax(1)
                result['scenarios'].append(row)
            result['summaries']={mode:{group:dict(mean=float(np.mean([v[mode]['accuracy'] for v in result['scenarios'] if select(v['history_map'])])),minimum=min(v[mode]['accuracy'] for v in result['scenarios'] if select(v['history_map'])),maximum=max(v[mode]['accuracy'] for v in result['scenarios'] if select(v['history_map']))) for group,select in [('mixed',lambda s:s.startswith('mixed')),('paired',lambda s:s.startswith('paired')),('coherent',lambda s:s.startswith('paired') and s.split('_')[1]==s.split('_')[2])]} for mode in ['static','wait','dynamic']}
        result['host_runtime_s']=time.perf_counter()-t0;allresults.append(result)
        print(c['name'],f"system={sch['total_s']:.6g}s age={maxage:.4f}tau",result.get('summaries',{}).get('dynamic'),flush=True)
        dump('results.partial.json',allresults)
    checks.update(programming_s_baseline=allresults[0]['timing']['breakdown_s']['Programming'],expected_programming_s=450*10*43.48e-6)
    assert abs(checks['programming_s_baseline']-checks['expected_programming_s'])<1e-10
    assert abs(allresults[0]['timing']['breakdown_s']['Array reads']-450*56*.14e-6)<1e-10
    # Native endpoint reproduction (within roundoff) checks extraction, hold interpolation and residual construction.
    _,tt,hh,rr,_=table
    for l in range(2):
        for h in range(4):
            native=load_path(PHY/'results'/f'valid_s32_w10_l{[12,19][l]}_h{h}_read8.csv')
            assert np.allclose(np.interp(.12+.14*np.arange(8),tt,hh[l,h])+rr[l,h],native[read_ids(native),1],rtol=1e-13,atol=0)
    checks['native_read8_endpoint_reconstruction']=True
    dump('trace_first_image.json',schedules['baseline']['trace']);np.savez_compressed(R/'predictions.npz',labels=y,ideal=ideal_pred,**predictions)
    result=dict(ideal=metrics(ideal_scores),configurations=allresults,scope=scope,host_runtime_s=time.perf_counter()-started,checks=checks)
    dump('results.json',result);dump('verification.json',dict(passed=True,**checks))
    files=[Path(__file__),PRE/'binary_model.npz',PHY/'level_s32_w10_l12.json',PHY/'level_s32_w10_l19.json']+table[-1]
    dump('provenance.json',[dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files])
    report(result,schedules['baseline']['trace'])

def report(r,trace):
    b=r['configurations'][0];m=b['summaries'];labels=['Static LUT','Wait + programming skew','Wait + read disturbance']
    fig,ax=plt.subplots(figsize=(9,4));means=[m[k]['mixed']['mean']*100 for k in ['static','wait','dynamic']]
    ax.bar(labels,means,color=['#557eb0','#ca8b37','#16a394']);ax.axhline(r['ideal']['accuracy']*100,color='#333',ls='--',label='Ideal binary network')
    for i,v in enumerate(means):ax.text(i,v+.15,f'{v:.2f}%',ha='center')
    ax.set_ylim(min(means)-3,100);ax.set_ylabel('Accuracy (%)');ax.legend();fig.tight_layout();fig.savefig(R/'accuracy.png',dpi=160);plt.close(fig)
    rows=''.join(f"<tr><td>{v['config']['name']}</td><td>{v['config']['tau_s']*1e6:g}</td><td>{v['config']['latency_s']*1e9:g}</td><td>{v['config']['bandwidth_Bps']/1e9:g}</td><td>{v['timing']['total_s']*1e3:.3f}</td><td>{v['max_read_age_tau']:.3f}</td><td>{(str(round(100*v['summaries']['dynamic']['mixed']['mean'],3))+'%') if 'summaries' in v else 'Outside table — no accuracy'}</td></tr>" for v in r['configurations'])
    effects=''.join(f"<tr><td>{mode}</td>"+''.join(f"<td>{100*m[mode][g]['mean']:.3f}% [{100*m[mode][g]['minimum']:.3f}, {100*m[mode][g]['maximum']:.3f}]</td>" for g in ['coherent','paired','mixed'])+'</tr>' for mode in ['static','wait','dynamic'])
    breakdown=''.join(f'<li>{k}: <b>{v*1e3:.3f} ms</b> ({v/b["timing"]["total_s"]*100:.2f}%)</li>' for k,v in b['timing']['breakdown_s'].items())
    scope=''.join(f'<li>{s}</li>' for s in r['scope'])
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Single-array time-multiplexed inference</title><style>
    body{background:#edf2f7;color:#213448;font:16px/1.6 system-ui;margin:0}main{max-width:1200px;margin:auto;padding:40px}h1{font-size:36px;line-height:1.2}h2{margin-top:0}section{background:white;border-radius:18px;padding:28px;margin:24px 0;box-shadow:0 6px 24px #18304a09}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:11px;border-bottom:1px solid #ddd;text-align:left}.tag{color:#137c78;text-transform:uppercase;letter-spacing:2px}.flow{display:flex;gap:20px;align-items:center;flex-wrap:wrap}.box{background:#eaf4f4;border:1px solid #88b9b5;border-radius:12px;padding:22px;flex:1}.array{display:grid;grid-template-columns:repeat(16,8px);gap:3px;background:#182e46;padding:20px}.cell{background:#d8a755;width:8px;height:8px;border-radius:2px}img{max-width:100%}details{margin:15px 0}pre{max-height:430px;overflow:auto;background:#142b40;color:#c9e9e7;padding:20px;font-size:12px}.notice{border-left:5px solid #cc9638;padding:16px;background:#fff6e4}select{padding:8px}</style><main><div class="tag">64 → 16 → 10 · two conductance levels</div><h1>One physical array.<br>Ten logical tiles, executed in time.</h1>
    <p>450 images · 4,500 array programming operations · 25,200 inference read slots · fixed +0.02 V bit-plane reads.</p>
    <div class="notice">Timing is conditional on an assumed τ-to-seconds conversion. Drift results use bounded empirical interpolation, not a new coupled COMSOL trajectory. Rewrite-cycle closure and mixed wait/read histories remain unvalidated.</div>
    <section><h2>Execution and memory interface</h2><div class="flow"><div class="box">128 KiB backing memory<br>Weights · pixels · hidden values · partial sums · logits<br><b>50 ns + bytes / 1 GB/s</b><br>512 B cache, 64 B lines</div><span>⇄</span><div class="box">Controller / buffers<br>One 16-value input buffer<br>Digital gains, bias and accumulation<br>SET32 → hold10 → RESET12/19</div><span>⇄</span><div class="array">CELLS</div></div><p>Per image: eight first-layer tiles → digital 8-bit hidden activation → two output tiles → argmax. Each tile is loaded and reprogrammed before inference reads. High-level devices wait 4.62τ for low-level devices to finish programming.</p></section>
    <section><h2>Accuracy: isolate the loss</h2><p>Ideal binary network: IDEAL. Coherent = one history; paired = one history per differential branch; mixed = 100 physical-cell maps. Brackets show scenario ranges, not confidence intervals.</p><img src="accuracy.png"><table><tr><th>Model</th><th>4 coherent maps</th><th>16 branch-pair maps</th><th>100 mixed maps</th></tr>EFFECTS</table></section>
    <section><h2>End-to-end simulated time</h2><p>Baseline 450 images: <b>TOTAL ms</b>; mean per-image latency <b>PER ms</b>. Host execution: HOST s.</p><ul>BREAKDOWN</ul><table><tr><th>Configuration</th><th>τ (µs)</th><th>Memory latency (ns)</th><th>Bandwidth (GB/s)</th><th>450 images (ms)</th><th>Max read age (τ)</th><th>Mixed-map accuracy</th></tr>ROWS</table><p>Programming includes fixed preparation, writing, gaps and verification. Array reads are inference only. Out-of-envelope runs retain valid architecture timing but do not extrapolate conductance.</p></section>
    <section><h2>First-image event trace</h2><p>Every pulse, verify read, hold and memory transaction is recorded. Select a logical tile.</p><select id="tile"><option value="all">Entire first image</option>OPTIONS</select><pre id="events"></pre></section>
    <section><h2>Interpretation and limits</h2><ul>SCOPE</ul><p><a href="results.json">Full results</a> · <a href="protocol.json">Frozen protocol</a> · <a href="trace_first_image.json">Event trace</a> · <a href="verification.json">Verification</a> · <a href="provenance.json">Source hashes</a></p></section></main><script>const trace=TRACE;function render(){const v=document.querySelector('#tile').value;document.querySelector('#events').textContent=trace.filter(e=>v==='all'||e.tile===+v).map(e=>`${(e.start_s*1e6).toFixed(3).padStart(11)} µs | ${(e.duration_s*1e6).toFixed(3).padStart(8)} µs | tile ${e.tile} | ${e.kind} | ${e.detail}${e.address===null?'':` @${e.address} (${e.bytes} B)`}`).join('\\n')}document.querySelector('#tile').onchange=render;render();</script></html>'''
    subs={'CELLS':'<i class="cell"></i>'*256,'IDEAL':f"{100*r['ideal']['accuracy']:.3f}%",'EFFECTS':effects,'TOTAL':f"{b['timing']['total_s']*1000:.3f}",'PER':f"{b['timing']['sample_latency_s']['mean']*1000:.3f}",'HOST':f"{r['host_runtime_s']:.2f}",'BREAKDOWN':breakdown,'ROWS':rows,'SCOPE':scope,'OPTIONS':''.join(f'<option value="{i}">Tile {i}: layer {1 if i<8 else 2}</option>' for i in range(10)),'TRACE':json.dumps(trace)}
    for k,v in subs.items():html=html.replace(k,v)
    (R/'report.html').write_text(html,encoding='utf-8')

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
