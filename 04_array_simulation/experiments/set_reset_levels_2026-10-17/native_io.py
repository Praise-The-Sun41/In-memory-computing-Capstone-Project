from pathlib import Path
import json
import numpy as np

R=Path(__file__).resolve().parent
OLD=R.parent/'fixed_protocol_2026-10-17'

def load_path(path):
    lines=[s for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip() and not s.lstrip().startswith('%')]
    rows=[]
    for line in lines:
        try: rows.append([float(x) for x in line.split(',')])
        except ValueError: continue
    a=np.asarray(rows)
    assert a.ndim==2 and a.shape[1]==1141 and np.isfinite(a).all(), str(path)
    assert np.all((a[:,4:700]>0)&(a[:,4:700]<1)), str(path)
    assert np.max(abs(a[:,2]-a[0,2]))<=1e-6, str(path)
    return a

def load(tag): return load_path(R/'results'/(tag+'.csv'))

def read_ids(a): return np.array([i for i in range(len(a)-1) if abs(a[i,3]-.02)<1e-9 and a[i+1,3]<.02-1e-9],dtype=int)

def reads(a): return a[read_ids(a),1]

def snapshot(a,t):
    ids=np.flatnonzero(np.abs(a[:,0]-t)<1e-8)
    assert len(ids)==1, (t,ids)
    return a[ids[0],4:]

def original(h):
    filename=['polarity_positive','history_high','history_low','history_aged'][h]
    a=load_path(OLD/'results'/(filename+'.csv'))
    return a[0 if h==0 else -1,4:]

def dump(name,obj): (R/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
