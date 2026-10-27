"""Scalar, byte-memory replay of every image, independently of batched inference."""
import run as s
import numpy as np
import json,time,hashlib
from sklearn.datasets import load_digits

def main():
    start=time.perf_counter();r=json.loads((s.R/'results.json').read_text());c=r['configurations'][0]['config']
    z=np.load(s.PRE/'binary_model.npz');p={k:z[k] for k in ['w1','w2','a1','a2','b1','b2']}
    signs=[np.where(p[k]>=0,1.,-1.) for k in ['w1','w2']];ts=s.tiles(signs)
    d=load_digits();pixels=d.data[z['test']].astype(int);levels,times,hold,residual,_=s.lut()
    delta=levels[0]['nominal_G']-levels[1]['nominal_G'];hm=np.random.default_rng(0).integers(0,4,(16,16))
    gains=[np.exp(p['a1']),np.exp(p['a2'])]
    def compute(j,x,delay):
        t=ts[j];o=t['outputs'];high=t['high'];counts=np.zeros(16,int);out=np.zeros(o);l=t['layer']
        for bit in range(5 if l==0 else 8):
            voltage=((x>>bit)&1)*.02;counts+=(voltage>0)
            age=delay+(bit+1)*c['slot_overhead_s']/c['tau_s']+.14*bit+.12
            g=np.zeros((16,2*o))
            for level in range(2):
                for h in range(4):
                    mask=(high[:,:2*o] if level==0 else ~high[:,:2*o])&(hm[:,:2*o]==h)
                    v=np.interp(age+(4.62 if level==0 else 0),times,hold[level,h])+residual[level,h,np.maximum(counts-1,0)]
                    g[mask]=np.broadcast_to(v[:,None],g.shape)[mask]
            currents=voltage@g
            out+=(currents[::2]-currents[1::2])/(.02*delta)*2**bit/(16 if l==0 else 255)
        return out*gains[l][t['col']*8:t['col']*8+o]
    replay=s.schedule(c,ts,pixels,gains,[p['b1'],p['b2']],compute)
    scores=np.array(replay['replayed_scores']);batched=s.infer('dynamic',c,replay,ts,p,pixels,hm,s.lut())
    error=float(abs(scores-batched).max());assert error<1e-10,error
    saved=np.load(s.R/'predictions.npz');assert np.array_equal(scores.argmax(1),saved['mixed_0_dynamic'])
    assert replay['memory']==r['configurations'][0]['timing']['memory']
    assert replay['total_s']==r['configurations'][0]['timing']['total_s']
    for item in json.loads((s.R/'provenance.json').read_text()):
        assert hashlib.sha256(s.Path(item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
    out=dict(passed=True,all_450_images_replayed_through_byte_memory=True,scalar_current_vs_batched_logits_max_error=error,predictions_identical=True,memory_counters_identical=True,timing_identical=True,source_hashes_verified=True,host_runtime_s=time.perf_counter()-start)
    s.dump('audit.json',out);print(json.dumps(out,indent=2))

if __name__=='__main__':main()
