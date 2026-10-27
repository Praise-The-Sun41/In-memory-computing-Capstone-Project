"""Five complete cycles: tighter time tolerance and targeted Ti grid refinement.

This is not a new oxide-grid or full 3D convergence claim.
"""
from chemistry import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',choices=['time','metal','compare'],required=True);a=p.parse_args()
    if a.run!='compare':
        M=512 if a.run=='metal' else 256;rtol=2e-10 if a.run=='time' else 2e-9
        m=Device(CASES['binding_barrier'],M=M)
        solve(m,et.protocol(5),m.initial,f'binding_barrier-C5-L256-M{M}-R{rtol:g}',rtol)
        return
    stems=['binding_barrier-C10-L256-M256-R2e-09','binding_barrier-C5-L256-M256-R2e-10',
           'binding_barrier-C5-L256-M512-R2e-09']
    ds=[];hashes={}
    for st in stems:
        path=P/(st+'.npz');meta=json.loads(path.with_suffix('.json').read_text())
        assert meta['data_sha']==sha(path) and meta['config']['source_sha']==sha(P/'chemistry.py')
        ds.append(dict(np.load(path)));hashes[path.name]=sha(path)
    base,tight,fine=ds;n=len(tight['t']);out={}
    for label,d in [('time',tight),('Ti_grid',fine)]:
        assert np.allclose(base['t'][:n],d['t'],rtol=0,atol=1e-10)
        metrics=dict(current_peak_normalized=float(abs(base['I'][:n]-d['I']).max()/abs(d['I']).max()),
            bound_fraction_absolute=float(abs(base['bound_fraction'][:n]-d['bound_fraction']).max()),
            exchange_factor_relative=float(abs(base['exchange_factor'][:n]/d['exchange_factor']-1).max()),
            temperature_K=float(abs(base['T'][:n]-d['T']).max()))
        gates=dict(current=metrics['current_peak_normalized']<.005,
            bound_fraction=metrics['bound_fraction_absolute']<.002,
            exchange=metrics['exchange_factor_relative']<.01,temperature=metrics['temperature_K']<.05)
        if label=='time':
            error=float(max(abs(expit(base['y'][:n,:440])-expit(d['y'][:,:440])).max(),
                abs(base['y'][:n,440:696]-d['y'][:,440:696]).max(),
                abs(base['y'][:n,699:]-d['y'][:,699:]).max()))
            metrics['all_ionic_occupancy_absolute']=error;gates['ionic_states']=error<5e-4
        out[label]=dict(metrics=metrics,gates=gates,all_pass=all(gates.values()))
    result=dict(checks=out,all_pass=all(v['all_pass'] for v in out.values()),data_hashes=hashes,
        source_sha=sha(P/'chemistry.py'),scope='Own preparation plus five cycles; time and Ti grid only, oxide L256 fixed')
    (P/'resolution_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    assert result['all_pass']

if __name__=='__main__':main()
