"""Small binary-weight MLP; numpy Adam + straight-through training."""
from pathlib import Path
import json,time,copy
import numpy as np
from scipy.special import logsumexp
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

R=Path(__file__).resolve().parent;ROOT=R.parent.parent
def dump(name,x): (R/name).write_text(json.dumps(x,indent=2),encoding='utf-8')
def activate(z):return np.rint(np.clip(z,0,1)*255)/255
def sign(w):return np.where(w>=0,1.,-1.)
def effective(p,binary):
    if binary:return sign(p['w1'])*np.exp(p['a1']),sign(p['w2'])*np.exp(p['a2'])
    return p['w1'],p['w2']
def forward(X,p,binary=True):
    w1,w2=effective(p,binary);h=activate(X@w1+p['b1']);return h@w2+p['b2']
def stats(X,y,p,binary):
    s=forward(X,p,binary)
    return dict(correct=int((s.argmax(1)==y).sum()),cross_entropy=float((logsumexp(s,axis=1)-s[np.arange(len(y)),y]).mean()))
def initialize(seed):
    rng=np.random.default_rng(seed)
    return dict(w1=rng.normal(0,np.sqrt(2/64),(64,16)),b1=np.full(16,.1),w2=rng.normal(0,np.sqrt(2/16),(16,10)),b2=np.zeros(10))
def binarize(p):
    p=copy.deepcopy(p)
    for layer in [1,2]:
        a=np.maximum(abs(p[f'w{layer}']).mean(0),1e-5)
        p[f'w{layer}']=np.clip(.2*p[f'w{layer}']/a,-1,1)
        p[f'a{layer}']=np.log(a)
    return p
def train(X,y,p,binary,epochs,lr,seed,validation=None):
    p=copy.deepcopy(p);m={k:np.zeros_like(v) for k,v in p.items()};v=copy.deepcopy(m)
    rng=np.random.default_rng(seed);t=0;history=[];best=None;best_loss=float('inf');best_epoch=0
    def record(epoch):
        chosen=best if binary else p
        row=dict(epoch=epoch,training=stats(X,y,p,binary),selected_training_epoch=best_epoch if binary else epoch,selected_training=stats(X,y,chosen,binary))
        if validation is not None:row['validation']=stats(*validation,chosen,binary)
        history.append(row)
    best=copy.deepcopy(p);best_loss=stats(X,y,p,binary)['cross_entropy']
    record(0)
    for epoch in range(1,epochs+1):
        order=rng.permutation(len(y))
        for offset in range(0,len(y),128):
            ids=order[offset:offset+128];x=X[ids];target=y[ids]
            w1,w2=effective(p,binary);z=x@w1+p['b1'];h=activate(z);s=h@w2+p['b2']
            d=np.exp(s-logsumexp(s,axis=1,keepdims=True));d[np.arange(len(ids)),target]-=1;d/=len(ids)
            dw2=h.T@d;dh=(d@w2.T)*((z>0)&(z<1));dw1=x.T@dh
            g=dict(b1=dh.sum(0),b2=d.sum(0))
            for layer,dw,w in [(1,dw1,w1),(2,dw2,w2)]:
                if binary:
                    a=np.exp(p[f'a{layer}']);g[f'w{layer}']=dw*a
                    g[f'a{layer}']=(dw*w).sum(0)+1e-4*(w*w).sum(0)
                else:g[f'w{layer}']=dw+1e-4*w
            norm=np.sqrt(sum(np.sum(a*a) for a in g.values()));factor=min(1.,5./max(norm,1e-12));t+=1
            for k in p:
                grad=g[k]*factor;m[k]=.9*m[k]+.1*grad;v[k]=.999*v[k]+.001*grad*grad
                p[k]-=lr*(m[k]/(1-.9**t))/(np.sqrt(v[k]/(1-.999**t))+1e-8)
            if binary:
                for layer in [1,2]:
                    p[f'w{layer}']=np.clip(p[f'w{layer}'],-1,1);p[f'a{layer}']=np.clip(p[f'a{layer}'],-6,2)
        if binary:
            loss=stats(X,y,p,True)['cross_entropy']
            if loss<best_loss:
                best_loss=loss;best=copy.deepcopy(p);best_epoch=epoch
        if epoch%25==0 or epoch==epochs:record(epoch)
    return best if binary else p,history
def main():
    begin=time.perf_counter();z=np.load(ROOT/'vteam16/digits_checkpoint.npz');outer=z['train_indices'];test=z['test_indices']
    data=load_digits();X=data.data/16.;y=data.target
    inner,val=train_test_split(outer,test_size=.2,random_state=7301,stratify=y[outer])
    configs=[dict(seed=seed,lr=lr) for seed in [7,19] for lr in [.001,.003]]
    dump('protocol.json',dict(architecture=[64,16,10],binary_weights='Both layers use only +/-1 signs times per-output positive digital gains; differential GH/GL pair per sign.',hidden_activation='Digital clipped ReLU [0,1], rounded to8 bits; straight-through derivative for rounding during training. Inputs are unchanged5-bit pixels0..16.',selection='Four QAT configurations, each up to400 epochs. Choose checkpoint by inner-validation correct count, then cross entropy, then earlier epoch. Refit on all1347 training samples for the selected epoch count. Original450 test images are evaluated only after freezing.',configs=configs,continuous_initializer='200 epochs, Adam lr0.003, same architecture and8-bit activations, fit on inner training only during selection.',inner_training_count=len(inner),validation_count=len(val),test_count=len(test),split_seed=7301,read_voltage=.02,levels=[12,19],hardware_scope='Same four COMSOL histories and five short snapshots; same seeds0..99 assignment policy extended to both layer shapes. Frozen-state inference only; ideal periphery/interconnect.'))
    protocol=json.loads((R/'protocol.json').read_text())
    protocol['selection']='For each QAT configuration/budget, retain the checkpoint with minimum training cross entropy within that budget. Select configuration/budget by inner-validation correct count, then validation cross entropy, then earlier budget. Refit on all1347 examples with the same training-loss checkpoint policy and selected budget; do not select on test score.'
    protocol['adaptive_correction']='Initial last-checkpoint run is archived in initial_last_checkpoint. Its final full-training accuracy dropped to84% while earlier training checkpoints were around93%, exposing sign-flip instability. One bounded correction applies training-loss checkpoint retention consistently in inner selection and full refit. The earlier test score was already observed; final evaluation reuses that cohort and is exploratory, not a fresh independent test.'
    dump('protocol.json',protocol)
    histories=[];candidates=[]
    for seed in [7,19]:
        real,_=train(X[inner],y[inner],initialize(seed),False,200,.003,seed)
        for lr in [.001,.003]:
            _,history=train(X[inner],y[inner],binarize(real),True,400,lr,seed,validation=(X[val],y[val]))
            histories.append(dict(seed=seed,lr=lr,history=history))
            candidates.extend(dict(seed=seed,lr=lr,**h) for h in history)
            dump('validation_runs.json',histories)
            best=max(history,key=lambda h:(h['validation']['correct'],-h['validation']['cross_entropy'],-h['epoch']))
            print('VALIDATION',seed,lr,best,flush=True)
    selected=max(candidates,key=lambda h:(h['validation']['correct'],-h['validation']['cross_entropy'],-h['epoch']))
    dump('selected_before_test.json',selected)
    real,real_history=train(X[outer],y[outer],initialize(selected['seed']),False,200,.003,selected['seed'])
    p,final_history=train(X[outer],y[outer],binarize(real),True,selected['epoch'],selected['lr'],selected['seed'])
    np.savez_compressed(R/'binary_model.npz',**p,train=outer,test=test,inner=inner,validation=val)
    np.savez_compressed(R/'continuous_hidden_control.npz',**real)
    dump('training.json',dict(selected=selected,continuous_history=real_history,binary_history=final_history,training_seconds=time.perf_counter()-begin,binary_signs_per_layer=[1024,160],method='Binary forward pass in both layers; clipped shadow weights, learned digital gains/biases, Adam with straight-through sign and activation-rounding derivatives. Finite validation-selected search, not a global optimum claim.'))
    print('TRAINING_COMPLETE',selected,flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
