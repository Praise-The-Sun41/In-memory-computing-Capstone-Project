from pathlib import Path
from itertools import product
import json,hashlib
import numpy as np
from sklearn.datasets import load_digits
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train import R,ROOT,activate,sign,forward,dump
SOURCE=R.parent/'set_reset_levels_2026-10-17'

def main():
    z=np.load(R/'binary_model.npz');p={k:z[k] for k in ['w1','w2','a1','a2','b1','b2']}
    data=load_digits();X=data.data[z['test']]/16.;y=data.target[z['test']]
    signs=[sign(p['w1']),sign(p['w2'])];gains=[np.exp(p['a1']),np.exp(p['a2'])]
    weights=[s*a for s,a in zip(signs,gains)];ideal=forward(X,p);pred_ideal=ideal.argmax(1)
    previous=json.loads((R.parent/'digits_trained_binary_2026-10-17/results.json').read_text())
    real=dict(np.load(R/'continuous_hidden_control.npz'))
    levels=[json.loads((SOURCE/f'level_s32_w10_l{k}.json').read_text()) for k in [12,19]]
    GH,GL=[np.array(v['conductance_by_history']) for v in levels];delta=levels[0]['nominal_G']-levels[1]['nominal_G']
    windows=['program_verify','ready','hold1','read8','sham8'];columns=levels[0]['columns']
    def metrics(scores):
        pred=scores.argmax(1)
        return dict(correct=int((pred==y).sum()),accuracy=float((pred==y).mean()),prediction_changes_vs_ideal=int((pred!=pred_ideal).sum()),new_errors_vs_ideal=int(((pred!=y)&(pred_ideal==y)).sum()),corrected_errors_vs_ideal=int(((pred==y)&(pred_ideal!=y)).sum()))
    def conductances(layer,hp,hn,c):
        s=signs[layer]
        return np.where(s>0,GH[hp,c],GL[hp,c]),np.where(s>0,GL[hn,c],GH[hn,c])
    def physical_weight(layer,hp,hn,c):
        gp,gn=conductances(layer,hp,hn,c);return (gp-gn)/delta*gains[layer]
    def tiled(inputs,w):
        out=np.zeros((len(inputs),w.shape[1]))
        for rb in range((w.shape[0]+15)//16):
            rr=slice(16*rb,min(16*rb+16,w.shape[0]))
            for cb in range((w.shape[1]+7)//8):
                cc=slice(8*cb,min(8*cb+8,w.shape[1]));out[:,cc]+=inputs[:,rr]@w[rr,cc]
        return out
    def infer(w):return tiled(activate(tiled(X,w[0])+p['b1']),w[1])+p['b2']
    assert np.max(abs(infer(weights)-ideal))<1e-12
    coherent=[];paired=[];mixed=[];layer_only=[[],[]]
    for hp,hn in product(range(4),repeat=2):
        for window in windows:
            c=columns.index(window);w=[physical_weight(l,hp,hn,c) for l in range(2)]
            row=dict(plus_history=hp,minus_history=hn,window=window,**metrics(infer(w)))
            paired.append(row)
            if hp==hn:coherent.append(row)
    for seed in range(100):
        rng=np.random.default_rng(seed)
        maps=[(rng.integers(0,4,s.shape),rng.integers(0,4,s.shape)) for s in signs]
        for window in windows:
            c=columns.index(window);w=[physical_weight(l,*maps[l],c) for l in range(2)]
            mixed.append(dict(seed=seed,window=window,**metrics(infer(w))))
            for l in range(2):
                ablation=weights.copy();ablation[l]=w[l]
                layer_only[l].append(dict(seed=seed,window=window,**metrics(infer(ablation))))
    def summary(rows):
        a=[v['accuracy'] for v in rows]
        return dict(scenarios=len(rows),minimum=min(a),mean=float(np.mean(a)),maximum=max(a),maximum_prediction_changes=max(v['prediction_changes_vs_ideal'] for v in rows))
    ready={v['seed']:v['accuracy'] for v in mixed if v['window']=='ready'}
    drift={w:dict(mean_change_pp=float(np.mean([100*(v['accuracy']-ready[v['seed']]) for v in mixed if v['window']==w])),worst_change_pp=min(100*(v['accuracy']-ready[v['seed']]) for v in mixed if v['window']==w)) for w in windows if w!='ready'}
    # Independent current-domain bit-plane implementation, including both layers.
    def bit_read(integers,denominator,bits,gp,gn,gain):
        out=np.zeros((len(integers),gp.shape[1]))
        for rb in range((gp.shape[0]+15)//16):
            rr=slice(16*rb,min(16*rb+16,gp.shape[0]))
            for cb in range((gp.shape[1]+7)//8):
                cc=slice(8*cb,min(8*cb+8,gp.shape[1]))
                for bit in range(bits):
                    voltage=((integers[:,rr]>>bit)&1)*.02
                    current_difference=voltage@gp[rr,cc]-voltage@gn[rr,cc]
                    out[:,cc]+=current_difference/(.02*delta)*gain[cc]*(2**bit/denominator)
        return out
    checks={}
    for window in ['ready','read8']:
        rng=np.random.default_rng(0);maps=[(rng.integers(0,4,s.shape),rng.integers(0,4,s.shape)) for s in signs];c=columns.index(window)
        g=[conductances(l,*maps[l],c) for l in range(2)]
        h=activate(bit_read(data.data[z['test']].astype(int),16,5,*g[0],gains[0])+p['b1'])
        score=bit_read(np.rint(h*255).astype(int),255,8,*g[1],gains[1])+p['b2']
        direct=infer([physical_weight(l,*maps[l],c) for l in range(2)])
        error=float(abs(score-direct).max());assert error<1e-10,error;checks[window]=error
    # Audit selection against recorded validation metrics, independent of test scores.
    runs=json.loads((R/'validation_runs.json').read_text());selected=json.loads((R/'selected_before_test.json').read_text())
    candidates=[dict(seed=v['seed'],lr=v['lr'],**h) for v in runs for h in v['history']]
    assert selected==max(candidates,key=lambda h:(h['validation']['correct'],-h['validation']['cross_entropy'],-h['epoch']))
    assert not set(z['train'])&set(z['test']) and not set(z['inner'])&set(z['validation'])
    assert set(z['inner'])|set(z['validation'])==set(z['train'])
    for s in signs:assert set(np.unique(s))=={-1.,1.}
    mapping=dict(architecture=[64,16,10],layers=[dict(matrix=[64,16],input_blocks=4,output_blocks=2,tiles=8,active_devices=2048),dict(matrix=[16,10],input_blocks=1,output_blocks=2,tiles=2,active_devices=320)],physical_array_shape=[16,16],logical_tiles=10,active_devices=2368,allocated_devices_if_all_tiles_resident=2560,padding_devices=192,array_read_slots_per_sample=56,read_slot_formula='8 first-layer tiles x5 input bits +2 second-layer tiles x8 hidden bits; no zero-plane skipping.',hidden_bits=8,read_voltage=.02,note='Logical placement and bit-plane algebra. Tiles may share one physical array by reprogramming. Memory buffers, timing, read-induced state evolution and transistor/interconnect effects are not simulated.')
    result=dict(continuous_linear=previous['continuous'],trained_linear_binary=previous['ideal_binary'],continuous_hidden_control=metrics(forward(X,real,False)),ideal_hidden_binary=metrics(ideal),coherent_summary=summary(coherent),paired_summary=summary(paired),mixed_summary=summary(mixed),mixed_by_window={w:summary([v for v in mixed if v['window']==w]) for w in windows},drift_vs_same_map_ready=drift,layer_only_summary=[summary(v) for v in layer_only],coherent=coherent,paired=paired,mixed=mixed,layer_only=layer_only,nominal_high=levels[0]['nominal_G'],nominal_low=levels[1]['nominal_G'],normalization_delta=delta,mapping=mapping,selected=selected,train_count=len(z['train']),test_count=len(y),scope=['Both layers have binary weights; pixels remain multibit and hidden activations are8-bit digital values. No extra device conductance levels.', 'Same four histories, five short snapshots and seeds0..99 assignment policy; maps extend to the two new matrix shapes.', 'Digital positive output gains and biases are frozen across every hardware scenario; no history-specific recalibration.', 'All test scenarios are frozen-state snapshots;56 logical read slots do not establish retention for that full sequence or450 sequential images.', 'Readout and wires are ideal.8-bit hidden conversion is a chosen numerical operation, not measured peripheral hardware accuracy. No new COMSOL solve or memory-latency claim.', 'The reused450-image test cohort is for comparison, not a new independent cohort. No test-score-based model selection. Scenario means are not hardware yield probabilities.', 'Continuous hidden control is the200-epoch initializer refitted on all training data at the binary-selected seed; not a claim of optimal continuous-network accuracy.'])
    result['gain_vs_trained_linear_pp']=100*(result['ideal_hidden_binary']['accuracy']-result['trained_linear_binary']['accuracy'])
    result['gap_to_continuous_linear_pp']=100*(result['continuous_linear']['accuracy']-result['ideal_hidden_binary']['accuracy'])
    training=json.loads((R/'training.json').read_text())
    initial=json.loads((R/'initial_last_checkpoint/results.json').read_text())
    result['initial_last_checkpoint_accuracy']=initial['ideal_hidden_binary']['accuracy']
    result['final_selected_training_epoch']=training['binary_history'][-1]['selected_training_epoch']
    result['scope'].append('One adaptive checkpoint-policy correction followed the initial unstable run. Initial82.67% test result is archived, not discarded. Corrected configuration/budget selection uses inner validation and training losses only; the reused test set was already observed, so this is exploratory evaluation, not a fresh independent test.')
    dump('results.json',result);dump('mapping.json',mapping)
    dump('verification.json',dict(passed=True,binary_sign_counts=[s.size for s in signs],partitions_disjoint=True,validation_selection_verified=True,direct_vs_tiled_max_error=float(abs(infer(weights)-ideal).max()),two_layer_current_bit_plane_errors=checks))
    sources=[R/'train.py',R/'evaluate.py',R/'binary_model.npz',R/'continuous_hidden_control.npz',R/'protocol.json',SOURCE/'level_s32_w10_l12.json',SOURCE/'level_s32_w10_l19.json',R.parent/'digits_trained_binary_2026-10-17/results.json']
    dump('provenance.json',[dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sources])
    np.savez_compressed(R/'predictions.npz',test=z['test'],truth=y,ideal_prediction=pred_ideal,continuous_hidden_prediction=forward(X,real,False).argmax(1))
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    labels=['Continuous\nlinear','Trained binary\nlinear','Trained binary\nhidden16','History + drift\nscenario mean','Continuous\nhidden control']
    values=[result['continuous_linear']['accuracy'],result['trained_linear_binary']['accuracy'],result['ideal_hidden_binary']['accuracy'],result['mixed_summary']['mean'],result['continuous_hidden_control']['accuracy']]
    axes[0].bar(range(5),np.array(values)*100,color=['#4776aa','#91aaa8','#38a58c','#dea64f','#a1a6b1']);axes[0].set_xticks(range(5),labels,fontsize=8);axes[0].set(ylim=(0,102),ylabel='Test accuracy (%)')
    for i,v in enumerate(values):axes[0].text(i,v*100+1,f'{v*100:.2f}',ha='center')
    for i,w in enumerate(windows):
        s=result['mixed_by_window'][w];axes[1].plot([i,i],[s['minimum']*100,s['maximum']*100],lw=5,color='#38a58c');axes[1].plot(i,s['mean']*100,'o',color='#263c60')
    axes[1].axhline(result['ideal_hidden_binary']['accuracy']*100,ls='--',color='#d79b35',label='Ideal hidden binary');axes[1].set_xticks(range(5),windows,rotation=20);axes[1].set(ylabel='Accuracy (%)',title='100 mixed maps: min / mean / max');axes[1].legend()
    fig.tight_layout();fig.savefig(R/'accuracy.png',dpi=160);plt.close(fig)
    def table(rows,heads):return '<div class="scroll"><table><tr>'+''.join(f'<th>{h}</th>' for h in heads)+'</tr>'+''.join('<tr>'+''.join(f'<td>{v}</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'
    rows=[[name,f'{result[key]["accuracy"]*100:.2f}%',f'{result[key]["correct"]}/450'] for name,key in [('Continuous64→10','continuous_linear'),('Trained binary64→10','trained_linear_binary'),('Trained binary64→16→10','ideal_hidden_binary'),('Continuous64→16→10 control','continuous_hidden_control')]]
    s=result['mixed_summary'];rows.append(['Hidden binary + mixed histories / short drift',f'{s["mean"]*100:.2f}% mean',f'{s["minimum"]*100:.2f}–{s["maximum"]*100:.2f}% range'])
    for name,key in [('Same history across all devices','coherent_summary'),('Different coherent histories on plus/minus branches','paired_summary')]:
        s=result[key];rows.append([name,f'{s["mean"]*100:.2f}% mean',f'{s["minimum"]*100:.2f}–{s["maximum"]*100:.2f}% range'])
    body='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Small hidden-layer binary network</title><style>body{font:16px/1.65 system-ui;background:#101b2a;color:#e6edf4;max-width:1200px;margin:30px auto;padding:0 20px}h1,h2,a{color:#86d9c3}.box{padding:20px;background:#20374a;border-radius:10px}.scroll{overflow:auto}table{width:100%;border-collapse:collapse}th,td{padding:10px;text-align:left;border-bottom:1px solid #405468}img{width:100%;border-radius:8px;background:white}.flow{display:flex;flex-wrap:wrap;gap:12px}.stage{flex:1;min-width:160px;background:#20374a;border:1px solid #456278;border-radius:10px;padding:14px}</style><h1>64 →16 →10: two-level binary weights</h1>'''
    body+=f'<div class="box">Ideal binary accuracy: <b>{result["ideal_hidden_binary"]["accuracy"]*100:.2f}%</b>. Change from the trained binary linear model: <b>{result["gain_vs_trained_linear_pp"]:+.2f} percentage points</b>. Same RESET12/19 conductance levels.</div>'
    body+='<h2>Accuracy comparison</h2>'+table(rows,['Model','Accuracy','Count / range'])+'<img src="accuracy.png">'
    body+='<h2>16×16 array mapping</h2><div class="flow"><div class="stage">64 pixels<br>5-bit input planes</div><div class="stage">Layer1:64×16<br>8 differential tiles</div><div class="stage">Digital bias + clipped ReLU<br>16 hidden values,8 bits each</div><div class="stage">Layer2:16×10<br>2 differential tiles</div><div class="stage">Digital bias + argmax<br>10 digit classes</div></div><p>Each tile is16×16 and supports8 differential outputs. Total10 logical tiles,2368 active device positions, and192 padding positions if all tiles are resident. A single physical array can be reused sequentially with intermediate data held in memory; this report verifies placement and arithmetic, not that schedule’s latency or retention.</p><p>Fixed +0.02 V bit-plane input needs40 layer1 and16 layer2 read slots per image, before optimizations. Every device still uses GH or GL. Output gains, bias and hidden activation quantization are digital.</p>'
    body+='<h2>History and short-window sensitivity</h2>'+table([[w,f'{s["minimum"]*100:.2f}',f'{s["mean"]*100:.2f}',f'{s["maximum"]*100:.2f}'] for w,s in result['mixed_by_window'].items()],['Snapshot','Minimum %','Mean %','Maximum %'])
    body+=table([[f'Only layer{i+1} nonideal',f'{s["minimum"]*100:.2f}',f'{s["mean"]*100:.2f}',f'{s["maximum"]*100:.2f}'] for i,s in enumerate(result['layer_only_summary'])],['Layer sensitivity control','Minimum %','Mean %','Maximum %'])
    body+='<p>Mixed maps assign one of four measured histories to each physical device and retain that assignment across snapshots. Any accuracy increases are incidental corrections of existing errors, not evidence that drift is beneficial. Means/ranges are descriptive sensitivity results, not confidence intervals.</p>'
    body+=f'<h2>Training and selection</h2><p>Two binary-weight layers were trained with a straight-through forward/backward approximation, Adam, clipped shadow weights, and learned positive digital gains/biases. Hidden values are clipped and quantized to8 bits during training and evaluation. Four configurations were compared on1077 inner-training and270 validation images. Selected seed{selected["seed"]}, learning rate{selected["lr"]}, epoch{selected["epoch"]}; refitted on all1347 training images before the450-image test evaluation. Architecture capacity and training method both differ from the previous linear binary experiment, so the gain cannot be attributed to architecture alone.</p>'
    body+=f'<h2>Checkpoint instability and recorded correction</h2><p>The first last-checkpoint run reached only82.67% test accuracy, with a simultaneous fall in training accuracy to about84%. Its artifacts remain in <a href="initial_last_checkpoint/report.html">the initial report</a>. A single bounded correction retained the minimum-training-cross-entropy checkpoint within each candidate budget, then repeated internal-validation selection. The corrected full refit retained epoch{result["final_selected_training_epoch"]} within the selected{selected["epoch"]}-epoch budget. The checkpoint was not chosen by test accuracy.</p>'
    body+='<h2>Scope</h2><ul>'+''.join('<li>'+v+'</li>' for v in result['scope'])+'</ul><p><a href="results.json">Results</a> · <a href="mapping.json">Array mapping</a> · <a href="training.json">Training</a> · <a href="verification.json">Verification</a> · <a href="provenance.json">Source hashes</a> · <a href="REPRODUCE.txt">Reproduce</a></p></html>'
    (R/'report.html').write_text(body,encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['ideal_hidden_binary','continuous_hidden_control','gain_vs_trained_linear_pp','gap_to_continuous_linear_pp','mixed_summary','coherent_summary','paired_summary','drift_vs_same_map_ready','layer_only_summary']},indent=2))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
