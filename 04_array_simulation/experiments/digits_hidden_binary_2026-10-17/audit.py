import json,hashlib
from html.parser import HTMLParser
import numpy as np
from sklearn.datasets import load_digits
from train import R,ROOT,forward,stats
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
z=np.load(R/'binary_model.npz');p={k:z[k] for k in ['w1','w2','a1','a2','b1','b2']}
data=load_digits();t=read(R/'training.json');last=t['binary_history'][-1]
measured=stats(data.data[z['train']]/16.,data.target[z['train']],p,True)
assert measured['correct']==last['selected_training']['correct']
assert abs(measured['cross_entropy']-last['selected_training']['cross_entropy'])<1e-10
assert last['selected_training_epoch']<=t['selected']['epoch']
old=np.load(ROOT/'vteam16/digits_checkpoint.npz')
assert np.array_equal(z['train'],old['train_indices']) and np.array_equal(z['test'],old['test_indices'])
prior=read(R.parent/'digits_trained_binary_2026-10-17/results.json');r=read(R/'results.json')
for k in ['nominal_high','nominal_low','normalization_delta']:assert r[k]==prior[k]
assert [(x['seed'],x['window']) for x in r['mixed']]==[(x['seed'],x['window']) for x in prior['mixed']]
for source in read(R.parent/'digits_trained_binary_2026-10-17/provenance.json'):
    assert hashlib.sha256(__import__('pathlib').Path(source['path']).read_bytes()).hexdigest()==source['sha256'],source['path']
class Links(HTMLParser):
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ['src','href']:assert (R/v).is_file(),v
Links().feed((R/'report.html').read_text(encoding='utf-8'))
out=dict(passed=True,saved_model_matches_selected_training_checkpoint=True,same_train_test_indices=True,same_conductance_levels_and_scenario_identifiers=True,prior_sources_unchanged=True,report_links_resolve=True,forward_checks=read(R/'verification.json'))
(R/'audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
