"""Time-accuracy check at late cycling and long retention, no refitting."""
from run import *

def main():
    a=argparse.ArgumentParser();a.add_argument('--case',choices=['Ti_dynamic500','Pt_barrier']);args=a.parse_args()
    checks=[]
    for case in ([args.case] if args.case else ['Ti_dynamic500','Pt_barrier']):
        s=System(case);p=P/case
        c=dict(np.load(p/'cycles.npz'));start=int(np.flatnonzero(c['kind']=='C19-neg-fall')[-1])
        r,_=solve(s,cycles(20,20),c['y'][start],'check-C20-tight',2e-10)
        mask=c['t']>=152-1e-10
        expected={k:v[mask] for k,v in c.items()}
        assert len(r['t'])==len(expected['t'])
        def audit(a,b,label):
            occ=lambda x:np.c_[expit(x[:,:s.no]),x[:,s.no:s.n],x[:,s.n+3:]]
            e=dict(case=case,protocol=label,current_peak_relative=float(abs(a['I']-b['I']).max()/max(abs(b['I']).max(),1e-12)),
                G300_relative=float(abs(a['G300']/b['G300']-1).max()),
                state_max_error=float(abs(occ(a['y'])-occ(b['y'])).max()),temperature_error=float(abs(a['T']-b['T']).max()))
            e['pass']=e['current_peak_relative']<1e-3 and e['G300_relative']<1e-3 and e['state_max_error']<5e-4 and e['temperature_error']<.01
            assert e['pass'],e;checks.append(e)
        audit(r,expected,'C20 from same C19 endpoint')
        h=dict(np.load(p/'positive-hold.npz'))
        tight,_=solve(s,[(1,0,0,'hold1'),(9,0,0,'hold10'),(90,0,0,'hold100'),(900,0,0,'hold1000')],h['y'][0],'check-positive-hold-tight',2e-10)
        audit(tight,h,'1000 tau positive-state hold')
    name='time_checks_'+args.case+'.json' if args.case else 'time_checks.json'
    (P/name).write_text(json.dumps(dict(all_pass=True,checks=checks,sources=sources(),scope='Representative late-cycle and long-hold local time checks, not all-history spatial convergence.'),indent=2))
    print(json.dumps(checks,indent=2),flush=True)
if __name__=='__main__':main()
