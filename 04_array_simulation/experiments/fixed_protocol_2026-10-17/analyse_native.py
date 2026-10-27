"""Analyse native COMSOL exports only; no device integration or fitted substitution."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parent

def load(name):
    path=R/'results'/(name+'.csv')
    lines=[l for l in path.read_text(encoding='utf-8-sig').splitlines() if l.strip() and not l.lstrip().startswith('%')]
    rows=[]
    for l in lines:
        try: rows.append([float(v) for v in l.split(',')])
        except ValueError: continue
    a=np.array(rows)
    if a.ndim!=2 or a.shape[1]!=1141 or not np.isfinite(a).all():raise ValueError('Invalid native export '+name)
    if np.any(a[:,4:700]<=0) or np.any(a[:,4:700]>=1):raise ValueError('Native occupancy bounds')
    if np.max(abs(a[:,2]-a[0,2]))>1e-6:raise ValueError('Native inventory drift')
    return a

def read_levels(a):
    # End of each read plateau, before the falling edge; use exported exact endpoints.
    ids=[i for i in range(len(a)-1) if abs(a[i,3]-.02)<1e-9 and a[i+1,3]<.02-1e-9]
    return a[ids,1]

def stage1():
    pos,neg,ref=[load(n) for n in ['polarity_positive','polarity_negative','polarity_positive_refined']]
    p,n,r=map(read_levels,[pos,neg,ref]);error=float(np.max(abs(p-r)/abs(r)))
    result=dict(positive_read_G=p.tolist(),negative_read_G=n.tolist(),refined_relative_error=error,
                positive_is_SET=bool(p[-1]>p[0]),negative_is_RESET=bool(n[-1]<n[0]),passed=bool(error<=.001 and p[-1]>p[0] and n[-1]<n[0]))
    (R/'stage1_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
def history():
    rows=[]
    for count in [8,32,128]:
        files=[R/'results'/f'probe{count}_history{j}.csv' for j in range(4)]
        if not all(p.exists() for p in files):continue
        curves=np.array([read_levels(load(p.stem)) for p in files]);spread=np.ptp(curves,axis=0)/curves.mean(axis=0)
        if curves.shape!=(4,17):raise ValueError('Incomplete programming curves')
        rows.append(dict(reset_count=count,curves=curves.tolist(),spread_by_level=spread.tolist(),maximum_relative_spread=float(spread.max()),initial_relative_spread=float(spread[0]),passed=bool(spread.max()<=.01)))
    result=dict(candidates=rows,selected=next((r['reset_count'] for r in rows if r['passed']),None))
    (R/'history_summary.json').write_text(json.dumps(result,indent=2))
    for r in rows:print({k:v for k,v in r.items() if k not in ['curves','spread_by_level']})
    if rows:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,len(rows),figsize=(6*len(rows),4),squeeze=False)
        for a,r in zip(ax[0],rows):
            g=np.array(r['curves']);ref=g[0,0]
            for j,label in enumerate(['Fresh','Previously SET','Previously RESET','SET then held']):a.plot(range(17),g[j]/ref,'o-',ms=3,label=label)
            a.set(title=f"{r['reset_count']} RESET pulses; spread {r['maximum_relative_spread']*100:.2f}%",xlabel='SET pulse count',ylabel='Read G / reference G');a.grid(alpha=.25);a.legend(fontsize=8)
        fig.tight_layout();fig.savefig(R/'history_convergence.png',dpi=160);plt.close(fig)
    return result
if __name__=='__main__':
    import sys
    history() if 'history' in sys.argv else stage1()

