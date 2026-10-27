"""Same-age zero-bias control and uninterrupted writes/holds/reads after cycling."""
from chemistry import *
from scenarios import CASES

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=CASES,required=True);a=p.parse_args()
    m=Device(CASES[a.case]);source=P/f'{a.case}-C10-L256-M256-R2e-09.npz'
    meta=json.loads(source.with_suffix('.json').read_text())
    assert meta['data_sha']==sha(source) and meta['config']['source_sha']==sha(P/'chemistry.py')
    old=dict(np.load(source));c0=np.flatnonzero(np.isclose(old['t'],21.4,rtol=0,atol=1e-10))
    assert len(c0)==1
    # Identical full preconditioned state and 80 tau age; remove the voltage
    # cycling only. Heating differs naturally and is not separately controlled.
    d,md=solve(m,[(80,0,0,'same-age-zero','linear')],old['y'][c0[0]],a.case+'-same-age-zero')
    md['parent_history_sha']=sha(source);md['parent_index']=int(c0[0]);md['time_origin_tau']=21.4
    (P/(a.case+'-same-age-zero.json')).write_text(json.dumps(md,indent=2))
    rows=[]
    for label,V in [('SET',1.),('RESET',-1.)]:
        rows += [(2,0,V,label+'-rise','linear'),(2,V,0,label+'-fall','linear'),
                 (100,0,0,label+'-hold','linear'),(.2,0,.02,label+'-read-rise','linear'),
                 (1,.02,.02,label+'-read','linear'),(.2,.02,0,label+'-read-fall','linear')]
    d,md=solve(m,rows,old['y'][-1],a.case+'-write-hold')
    md['parent_history_sha']=sha(source);md['parent_index']=len(old['t'])-1;md['time_origin_tau']=float(old['t'][-1])
    (P/(a.case+'-write-hold.json')).write_text(json.dumps(md,indent=2))

if __name__=='__main__':main()
