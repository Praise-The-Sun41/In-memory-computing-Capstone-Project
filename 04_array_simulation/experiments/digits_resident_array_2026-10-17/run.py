"""Ten persistent crossbars for the frozen 64-16-10 binary-weight network.

Inference executes through byte-addressable memory and fixed-amplitude read
bitplanes. Conductance snapshots do not assert unmeasured long-time retention.
"""
from pathlib import Path
from collections import OrderedDict,Counter
import json,hashlib,time
import numpy as np
from sklearn.datasets import load_digits
from threadpoolctl import threadpool_limits

R=Path(__file__).resolve().parent
PRE=R.parent/'digits_hidden_binary_2026-10-17'
PHY=R.parent/'set_reset_levels_2026-10-17'
def dump(n,x): (R/n).write_text(json.dumps(x,indent=2),encoding='utf-8')

class Memory:
    def __init__(self,latency=50e-9,bandwidth=1e9,cache_bytes=512):
        self.data=bytearray(131072);self.used=0;self.cache=OrderedDict();self.latency=latency;self.bandwidth=bandwidth;self.cache_lines=cache_bytes//64;self.count=Counter();self.allocations={}
    def alloc(self,name,value):
        a=np.ascontiguousarray(value);address=(self.used+63)//64*64;self.used=address+a.nbytes
        assert self.used<=len(self.data);self.data[address:self.used]=a.tobytes()
        self.allocations[name]=dict(address=address,bytes=a.nbytes,shape=list(a.shape),dtype=str(a.dtype));return address
    def access(self,address,n,payload=None):
        assert 0<=address<=address+n<=self.used and n>0
        if payload is not None:
            assert len(payload)==n;self.data[address:address+n]=payload
            self.count['write_requests']+=1;self.count['dram_write_bytes']+=n
            return self.latency+n/self.bandwidth,bytes(self.data[address:address+n])
        duration=0.;self.count['read_requests']+=1;self.count['requested_read_bytes']+=n
        for line in range(address//64,(address+n-1)//64+1):
            if line in self.cache:
                self.cache.move_to_end(line);self.count['cache_hits']+=1;duration+=2e-9
            else:
                self.count['cache_misses']+=1;self.count['dram_read_bytes']+=64;duration+=self.latency+64/self.bandwidth
                if self.cache_lines:
                    self.cache[line]=True
                    if len(self.cache)>self.cache_lines:self.cache.popitem(last=False);self.count['cache_evictions']+=1
        return duration,bytes(self.data[address:address+n])

def mapping(signs):
    result=[]
    for layer,s in enumerate(signs):
        for rb in range(s.shape[0]//16):
            for cb in range((s.shape[1]+7)//8):
                n=min(8,s.shape[1]-8*cb);high=np.zeros((16,16),bool)
                ss=s[rb*16:rb*16+16,cb*8:cb*8+n]
                high[:,:2*n:2]=ss>0;high[:,1:2*n:2]=ss<0
                result.append(dict(id=len(result),layer=layer,input_start=rb*16,output_start=cb*8,outputs=n,active_devices=32*n,high=high))
    assert len(result)==10 and sum(t['active_devices'] for t in result)==2368
    return result

def conductances(ts,levels,histories=None):
    gs=[]
    for t in ts:
        n=t['outputs'];g=np.zeros((16,16));s=t['high'][:,:2*n]
        if histories is None:g[:,:2*n]=np.where(s,levels[0]['nominal_G'],levels[1]['nominal_G'])
        else:
            h=histories[t['id']][:,:2*n]
            g[:,:2*n]=np.where(s,np.array(levels[0]['conductance_by_history'])[h,1],np.array(levels[1]['conductance_by_history'])[h,1])
        gs.append(g)
    return gs

def logical_maps(ts,signs,seed):
    # Preserve the earlier experiment's logical-weight history assignment policy.
    rng=np.random.default_rng(seed);lm=[(rng.integers(0,4,s.shape),rng.integers(0,4,s.shape)) for s in signs];out=[]
    for t in ts:
        h=np.zeros((16,16),int);rr=slice(t['input_start'],t['input_start']+16);cc=slice(t['output_start'],t['output_start']+t['outputs'])
        h[:,:2*t['outputs']:2]=lm[t['layer']][0][rr,cc];h[:,1:2*t['outputs']:2]=lm[t['layer']][1][rr,cc];out.append(h)
    return out

def reference(x,ts,gs,gains,biases,delta):
    cur=x/16
    for layer in range(2):
        out=np.zeros((len(x),16 if layer==0 else 10))
        for t,g in zip(ts,gs):
            if t['layer']!=layer:continue
            n=t['outputs'];rr=slice(t['input_start'],t['input_start']+16);cc=slice(t['output_start'],t['output_start']+n)
            out[:,cc]+=cur[:,rr]@((g[:,:2*n:2]-g[:,1:2*n:2])/delta)*gains[layer][cc]
        out+=biases[layer]
        if layer==0:cur=np.rint(np.clip(out,0,1)*255)/255
    return out

def execute(x,ts,gs,gains,biases,delta,latency=50e-9,bandwidth=1e9,cache_bytes=512):
    mem=Memory(latency,bandwidth,cache_bytes);wa=mem.alloc('packed_weights',np.array([np.packbits(t['high']) for t in ts],np.uint8))
    pa=mem.alloc('digital_gains_biases',np.concatenate(gains+biases));ia=mem.alloc('input_images',x.astype(np.uint8));oa=mem.alloc('output_logits',np.zeros((len(x),10),np.float64))
    events=[];totals=Counter();tau=0.;seconds=0.
    def event(kind,dtau=0.,ds=0.,sample=-1,detail='',address=None,n=None,arrays=None):
        nonlocal tau,seconds
        if sample<1:events.append(dict(kind=kind,start_tau=tau,start_seconds=seconds,duration_tau=dtau,duration_seconds=ds,sample=sample,detail=detail,address=address,bytes=n,arrays=arrays))
        tau+=dtau;seconds+=ds;totals[kind+'_tau']+=dtau;totals[kind+'_seconds']+=ds
    def access(address,n,payload=None,sample=-1,detail=''):
        elapsed,data=mem.access(address,n,payload);event('memory',ds=elapsed,sample=sample,detail=detail,address=address,n=n);return data
    # Single programming pass. Each array has an independent permanent matrix.
    persistent=[];program_calls=0
    for j,t in enumerate(ts):
        bits=np.unpackbits(np.frombuffer(access(wa+32*j,32,detail=f'Load array {j} once'),np.uint8)).reshape(16,16)
        assert np.array_equal(bits,t['high']);persistent.append(gs[j].copy());program_calls+=1
        event('initial_programming',dtau=43.48,detail=f'Array {j}: SET32, wait10, masked RESET12/19',arrays=[j])
    raw=np.frombuffer(access(pa,416,detail='Load digital parameters once'),np.float64)
    g1,g2,b1,b2=raw[:16],raw[16:26],raw[26:42],raw[42:52]
    assert np.array_equal(g1,gains[0]) and np.array_equal(b2,biases[1])
    startup_tau=tau;startup_seconds=seconds;memory_after_startup=dict(mem.count)
    fingerprints=[hashlib.sha256(g.tobytes()).hexdigest() for g in persistent]
    output=[];hidden_all=[];read_counts=np.zeros((10,16),int);latencies=[]
    for i in range(len(x)):
        initial_tau,initial_s=tau,seconds
        pixels=np.frombuffer(access(ia+i*64,64,sample=i,detail='Fetch image into 64-byte local input buffer'),np.uint8)
        current=pixels
        for layer,(nbits,den,gain,bias) in enumerate([(5,16,g1,b1),(8,255,g2,b2)]):
            ids=list(range(8)) if layer==0 else [8,9];out=np.zeros(16 if layer==0 else 10)
            for bit in range(nbits):
                event('readout_control',ds=20e-9,sample=i,detail=f'L{layer+1} bit {bit}: parallel readout',arrays=ids)
                event('inference_reads',dtau=.14,sample=i,detail=f'L{layer+1} bit {bit}: +0.02 V on active rows',arrays=ids)
                for j in ids:
                    t=ts[j];n=t['outputs'];rr=slice(t['input_start'],t['input_start']+16);cc=slice(t['output_start'],t['output_start']+n)
                    on=((current[rr]>>bit)&1).astype(float);read_counts[j]+=on.astype(int)
                    currents=(.02*on)@persistent[j]
                    out[cc]+=(currents[:2*n:2]-currents[1:2*n:2])/(.02*delta)*(2**bit/den)*gain[cc]
            out+=bias;event('digital_activation',ds=100e-9,sample=i,detail='8-bit hidden activation' if layer==0 else 'Final bias and argmax')
            if layer==0:
                current=np.rint(np.clip(out,0,1)*255).astype(np.uint8);hidden_all.append(current.copy())
        access(oa+i*80,80,out.astype(np.float64).tobytes(),i,'Write output logits');output.append(out)
        latencies.append([tau-initial_tau,seconds-initial_s])
    saved=np.frombuffer(mem.data[oa:oa+len(x)*80],np.float64).reshape(-1,10)
    assert np.array_equal(saved,np.array(output))
    assert fingerprints==[hashlib.sha256(g.tobytes()).hexdigest() for g in persistent]
    assert program_calls==10 and mem.count['read_requests']-memory_after_startup['read_requests']==len(x)
    return dict(scores=np.array(output),hidden=np.array(hidden_all),timing=dict(startup_tau=startup_tau,startup_seconds=startup_seconds,inference_tau=tau-startup_tau,inference_seconds=seconds-startup_seconds,total_tau=tau,total_seconds=seconds,mean_image_tau=float(np.mean(latencies,axis=0)[0]),mean_image_seconds=float(np.mean(latencies,axis=0)[1]),breakdown=dict(totals)),memory=dict(mem.count),allocations=mem.allocations,allocated_bytes=mem.used,events=events,program_calls=program_calls,read_counts=read_counts.tolist(),weight_unchanged=True)

def main():
    started=time.perf_counter();files=[PRE/'binary_model.npz',PRE/'results.json']+[PHY/f'level_s32_w10_l{k}.json' for k in [12,19]]
    before={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    z=np.load(files[0]);p={k:z[k] for k in ['w1','w2','a1','a2','b1','b2']};signs=[np.where(p[k]>=0,1.,-1.) for k in ['w1','w2']];gains=[np.exp(p['a1']),np.exp(p['a2'])];biases=[p['b1'],p['b2']]
    data=load_digits();x=data.data[z['test']].astype(int);y=data.target[z['test']];ts=mapping(signs);levels=[json.loads(f.read_text()) for f in files[2:]];delta=levels[0]['nominal_G']-levels[1]['nominal_G']
    limits=[
      'Ten permanently assigned16x16 arrays. No weight-tile time multiplexing, reprogramming between images, or hidden-layer spilling to external memory. Input bitplanes remain serial; arrays within a layer read concurrently.',
      'Same trained64-16-10 network and two conductance levels. No retraining or changes to the device baseline. Digital positive gains, biases and8-bit hidden activation are retained.',
      'Conductance snapshots are nominal or four-history ready-state tables. Accuracy does not validate retention over450 images or cumulative read disturbance. No automatic refresh policy is established.',
      'Separate timing ledger in tau and seconds. No physical tau-to-second calibration. A conversion selected in the viewer is an assumption only.',
      'The initial write ledger uses10 serialized array writes, each43.48tau, with ideal independently masked parallel programming inside an array. It is an optimistic addressing assumption, not a demonstrated standard1T1R write circuit. Row-serial full-protocol sensitivity multiplies initial writing by16.',
      'Resident-state assignment uses post-program snapshots without evolving during initialization. Real sequential loading creates different array ages; prior short retention data cannot validate this initialization or sustained operation.',
      'Eight simultaneous first-layer array read paths and two second-layer paths are provisioned. Each path senses its differential columns concurrently;20ns slot overhead and100ns digital activation per layer are design assumptions. No shared ADC serialization, wire resistance, ADC quantization/noise, transistor or power-supply constraints.',
      'Main memory128KiB,512B LRU read cache,64B lines, write-through/no-write-allocate. Baseline50ns transaction latency and1GB/s bandwidth; cache-hit2ns. Static dataset provisioning is excluded; all weight loading, parameter loading, inference inputs and output writes are counted.',
      'Digital sums/logits use float64 storage; inputs and hidden activations are uint8. On-chip input64B, hidden16B and sum128B buffers plus digital parameters are separate from the512B memory cache.',
      'Same450-image exploratory test cohort.100 random history maps retain the previous hidden-network logical mapping policy. Scenario averages are not confidence intervals or device yield estimates.'
    ]
    dump('protocol.json',dict(network=[64,16,10],physical_arrays=10,shape=[16,16],active_devices=2368,total_devices=2560,read_voltage=.02,parallel_slots_per_image=13,array_read_operations_per_image=56,main_memory_bytes=131072,cache_bytes=512,history_seeds=list(range(100)),limits=limits))
    metrics=lambda scores:dict(correct=int((scores.argmax(1)==y).sum()),accuracy=float((scores.argmax(1)==y).mean()))
    scenarios=[('nominal',None)]+[(f'history_{h}',[np.full((16,16),h) for t in ts]) for h in range(4)]+[(f'mixed_{seed}',logical_maps(ts,signs,seed)) for seed in range(100)]
    display=[];rows=[];checks={};runs={};predictions={}
    for name,hm in scenarios:
        gs=conductances(ts,levels,hm);ref=reference(x,ts,gs,gains,biases,delta);rows.append(dict(name=name,**metrics(ref)));predictions[name]=ref.argmax(1)
        if name in ['nominal','mixed_0']:
            run=execute(x,ts,gs,gains,biases,delta);err=float(abs(run['scores']-ref).max());assert err<1e-10
            assert np.array_equal(run['scores'].argmax(1),ref.argmax(1));checks[name]=err;runs[name]=run
        if name in ['nominal','history_0','history_1','history_2','history_3','mixed_0']:
            display.append(dict(name=name,accuracy=metrics(ref)['accuracy'],conductances=[g.tolist() for g in gs],histories=None if hm is None else [h.tolist() for h in hm],predictions=ref.argmax(1).tolist(),scores=ref.tolist()))
    previous=json.loads((PRE/'results.json').read_text())
    assert rows[0]['correct']==previous['ideal_hidden_binary']['correct']
    for v in previous['mixed']:
        if v['window']=='ready':assert next(q for q in rows if q['name']==f"mixed_{v['seed']}")['correct']==v['correct']
    run=runs['nominal'];timing=run['timing'];memory_sweep=[]
    for name,lat,bw,cache in [('baseline',50e-9,1e9,512),('slow',200e-9,1e8,512),('fast',10e-9,1e10,512),('uncached',50e-9,1e9,0)]:
        q=run if name=='baseline' else execute(x,ts,conductances(ts,levels),gains,biases,delta,lat,bw,cache)
        assert np.array_equal(q['scores'],run['scores']);memory_sweep.append(dict(name=name,latency_s=lat,bandwidth_Bps=bw,cache_bytes=cache,timing=q['timing'],memory=q['memory']))
    assert abs(timing['inference_tau']-450*13*.14)<1e-8
    assert abs(timing['startup_tau']-10*43.48)<1e-8
    assert before=={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    mixed=[q['accuracy'] for q in rows if q['name'].startswith('mixed')]
    result=dict(nominal=rows[0],mixed_ready=dict(mean=float(np.mean(mixed)),minimum=min(mixed),maximum=max(mixed)),scenarios=rows,timing=timing,memory=run['memory'],memory_sweep=memory_sweep,program_calls=10,weight_program_calls_during_inference=0,allocated_bytes=run['allocated_bytes'],checks=checks,previous_ready_scenarios_exactly_matched=True,host_runtime_s=time.perf_counter()-started,limits=limits)
    dump('results.json',result);dump('trace_first_image.json',run['events']);dump('memory_map.json',run['allocations']);dump('provenance.json',before)
    dump('verification.json',dict(passed=True,byte_memory_and_direct_logits_max_errors=checks,output_memory_roundtrip=True,weights_unchanged_after_450_images=True,programming_calls=10,inference_weight_reads=0,prior_ready_predictions_accuracy_matched=True,source_files_unchanged=True))
    np.savez_compressed(R/'predictions.npz',labels=y,**predictions)
    payload=dict(results=result,tiles=[{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in t.items()} for t in ts],scenarios=display,samples=x.tolist(),labels=y.tolist(),gains=[g.tolist() for g in gains],biases=[b.tolist() for b in biases],delta=delta,nominal_levels=[l['nominal_G'] for l in levels],trace=run['events'],memory=run['allocations'],read_counts=run['read_counts'])
    (R/'data.js').write_text('window.RESIDENT='+json.dumps(payload,separators=(',',':'))+';',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['nominal','mixed_ready','timing','host_runtime_s']},indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
