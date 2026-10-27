"""Distinguish terminal-current accuracy from transient ionic-state accuracy."""
from coupled import *

reference_path=P/'Ti-thermal-C5-R2e-10.npz'
b=dict(np.load(reference_path));out={}
for tol in ('2e-08','2e-09'):
    path=P/f'Ti-thermal-C5-R{tol}.npz';a=dict(np.load(path))
    e=abs(expit(a['y'][:,:440])-expit(b['y'][:,:440]))
    i,j=np.unravel_index(e.argmax(),e.shape)
    out[tol]=dict(oxide_maximum_difference=float(e[i,j]),time=float(a['t'][i]),cell=int(j),
                  oxide_final_maximum_difference=float(e[-1].max()),
                  metal_maximum_difference=float(abs(a['y'][:,440:696]-b['y'][:,440:696]).max()),
                  current_peak_normalized_error=float(np.max(abs(a['I']-b['I']))/max(abs(b['I']))),
                  temperature_error_K=float(np.max(abs(a['T']-b['T']))),data_sha=sha(path))
result=dict(comparisons_to_rtol_2e10=out,reference_data_sha=sha(reference_path),
            reference_2e9_state_converged=out['2e-09']['oxide_maximum_difference']<5e-4,
            default_2e8_state_gate=out['2e-08']['oxide_maximum_difference']<5e-4)
(P/'python_state_accuracy.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2));assert result['reference_2e9_state_converged']
