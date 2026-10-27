"""Independent input-encoding check, not a COMSOL array inference result."""
from pathlib import Path
import numpy as np,json,time
from sklearn.datasets import load_digits
r=Path(__file__).resolve().parent;root=r.parents[1]
z=np.load(root/'vteam16/digits_checkpoint.npz');pixels=load_digits().data[z['test_indices']].astype(np.uint8)
W=z['weights'];b=z['bias'];direct=pixels/16@W+b;serial=np.zeros_like(direct)+b
for rb in range(4):
    for cb in range(2):
        end=min(10,cb*8+8)
        for bit in range(5):
            bits=((pixels[:,rb*16:(rb+1)*16]>>bit)&1).astype(float)
            serial[:,cb*8:end]+=bits@W[rb*16:(rb+1)*16,cb*8:end]*(2**bit/16)
error=float(abs(serial-direct).max());assert error<1e-10
result=dict(scope='Digital encoding identity only; no COMSOL-device or array accuracy claim',bit_planes=5,maximum_score_error=error,prediction_disagreements=int(np.sum(serial.argmax(1)!=direct.argmax(1))),samples=len(pixels),maximum_array_reads=len(pixels)*8*5,passed=True)
(r/'encoding_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
