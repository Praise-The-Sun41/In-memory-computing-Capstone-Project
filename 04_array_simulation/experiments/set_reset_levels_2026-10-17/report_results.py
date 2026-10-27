from pathlib import Path
import json,hashlib,html,re,time
import numpy as np
from itertools import combinations
from native_io import R,OLD,load,load_path,reads,dump

def make_report():
    started=time.monotonic()
    while (R/'boundary.log').exists() and 'BOUNDARY_REFERENCE_COMPLETE' not in (R/'boundary.log').read_text(errors='replace'):
        if 'Traceback (most recent call last)' in (R/'boundary.log').read_text(errors='replace'):raise RuntimeError('Boundary reference failed; inspect before report release')
        if time.monotonic()-started>7200:raise TimeoutError('Boundary reference incomplete')
        time.sleep(3)
    result=json.loads((R/'campaign_result.json').read_text(encoding='utf-8'))
    protocol=json.loads((R/'protocol.json').read_text(encoding='utf-8'))
    screens=[json.loads((R/f'screen_{n}.json').read_text(encoding='utf-8')) for n in result['screened_set_counts']]
    expected={}
    for n in range(8,max(result['screened_set_counts'])+1,8):
        for h in range(4):expected[f'set_n{n}_h{h}']=(5.26,9)
    for n in result['screened_set_counts']:
        for h in range(4):
            for w in [0,1,10]:
                expected[f'reset_s{n}_w{w}_h{h}']=(10.38,17)
                if w:expected[f'wait_s{n}_w{w}_h{h}']=(w,0)
    validations=[]
    for p in sorted(R.glob('level_s*_w*_l*.json')):
        n,w,k=map(int,re.fullmatch(r'level_s(\d+)_w(\d+)_l(\d+)',p.stem).groups())
        validations.append(dict(set_count=n,wait_tau=w,**json.loads(p.read_text(encoding='utf-8'))))
        envelope=np.array(validations[-1]['conductance_by_history'])[:,[0,1,2,4,5]]
        intermediate=np.concatenate([reads(load(f'valid_s{n}_w{w}_l{k}_h{h}_read8')) for h in range(4)])
        assert intermediate.min()>=envelope.min() and intermediate.max()<=envelope.max(), ('Intermediate READ exceeds declared envelope',n,w,k)
        for h in range(4):
            for case,duration,num_reads in [('ready',.14,1),('hold1',1.14,1),('hold10',10.14,1),('read8',1.12,8),('sham8',1.12,1)]:
                expected[f'valid_s{n}_w{w}_l{k}_h{h}_{case}']=(duration,num_reads)
    for p in R.glob('results/ref_reset_*.csv'):
        n,w,h=map(int,re.fullmatch(r'ref_reset_s(\d+)_w(\d+)_h(\d+)',p.stem).groups())
        expected[p.stem]=(10.38,17);expected[f'ref_set_s{n}_h{h}']=(5.26,9);expected[f'ref_wait_s{n}_w{w}_h{h}']=(w,0)
    for pattern in ['results/tail_*.csv','results/ref_tail_*.csv']:
        for p in R.glob(pattern):expected[p.stem]=(10.38,17)
    for p in R.glob('results/boundary_read8_*.csv'):expected[p.stem]=(1.12,8)
    reference=load_path(OLD/'results/polarity_positive.csv')[0,2]
    audit=[]
    for tag,(duration,nreads) in expected.items():
        a=load(tag);assert abs(a[-1,0]-duration)<1e-7,(tag,a[-1,0]);assert len(reads(a))==nreads,tag
        error=float(np.max(abs(a[:,2]-reference)));assert error<=1e-6,(tag,error)
        audit.append(dict(case=tag,rows=len(a),end_tau=float(a[-1,0]),inventory_error=error,sha256=hashlib.sha256((R/'results'/(tag+'.csv')).read_bytes()).hexdigest()))
    for source in json.loads((R/'source_manifest.json').read_text(encoding='utf-8')):
        assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256'],source['path']
    dump('audit.json',dict(passed=True,complete_native_cases=len(audit),maximum_inventory_error=max(x['inventory_error'] for x in audit),cases=audit))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,len(screens),figsize=(6*len(screens),10),squeeze=False)
    labels=['Reference','Previously SET','Previously RESET','SET then held']
    for col,screen in enumerate(screens):
        for row,c in enumerate(screen['candidates']):
            ax=axes[row,col]
            for h,line in enumerate(c['curves']):ax.plot(range(17),line,'o-',ms=3,label=labels[h])
            ax.set(title=f"SET x{c['set_count']}; wait {c['wait_tau']} tau",xlabel='RESET pulse count',ylabel='Read G (model units)')
            ax.grid(alpha=.2);ax.legend(fontsize=7)
            for k in c['screening_passing_counts']:ax.axvline(k,color='#008476',lw=3,alpha=.12)
    fig.tight_layout();fig.savefig(R/'reset_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    last=max(result['screened_set_counts'])
    for h,label in enumerate(labels):
        values=[]
        for n in range(8,last+1,8):
            g=reads(load(f'set_n{n}_h{h}'))
            if n==8:values.append(g[0])
            values.extend(g[1:])
        axes[0].plot(range(last+1),values,label=label)
        axes[1].plot(result['screened_set_counts'],[100*s['candidates'][0]['set_last_four_max_step_change'][h] for s in screens],'o-',label=label)
    axes[0].set(xlabel='Cumulative SET pulse count',ylabel='Read G (model units)',title='Standardized SET preparation')
    axes[1].set(xlabel='Preparation pulse count',ylabel='Largest last-four step change (%)',title='Terminal response stationarity',yscale='log')
    axes[1].axhline(.1,ls='--',color='#ac6326',label='0.1% gate')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(R/'set_preparation.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,len(screens),figsize=(6*len(screens),4),squeeze=False)
    for col,s in enumerate(screens):
        for c in s['candidates']:axes[0,col].plot(range(17),100*np.array(c['relative_spread']),'o-',ms=3,label=f"Wait {c['wait_tau']} tau")
        axes[0,col].axhline(1,color='#ac6326',ls='--',label='1% gate')
        axes[0,col].set(title=f"SET x{s['set_count']}",xlabel='RESET pulse count',ylabel='Cross-history G range (%)')
        axes[0,col].grid(alpha=.2);axes[0,col].legend(fontsize=8)
    fig.tight_layout();fig.savefig(R/'history_ranges.png',dpi=150);plt.close(fig)
    def table(rows,cols):
        return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in cols)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x.get(k,'')))+'</td>' for k in cols)+'</tr>' for x in rows)+'</tbody></table></div>'
    screenrows=[]
    for s in screens:
        for c in s['candidates']:
            screenrows.append({'SET pulses':c['set_count'],'Wait (tau)':c['wait_tau'],'Stationary response':c['stationary_set_response'],'Counts with history range <=1%':c['screening_passing_counts'],'Separated candidate set':c['separated_groups'][0] if c['separated_groups'] else 'None'})
    extension=None
    if (R/'extension_screen.json').exists():
        extension=json.loads((R/'extension_screen.json').read_text(encoding='utf-8'))
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        for h,g in enumerate(extension['curves']):axes[0].plot(range(33),g,'o-',ms=3,label=labels[h])
        axes[0].set(xlabel='RESET pulse count',ylabel='G (model units)',title=f"Extended: SET x{extension['set_count']}, wait {extension['wait_tau']} tau")
        axes[1].plot(range(33),100*np.array(extension['relative_spread']),'o-',ms=3)
        axes[1].axhline(1,color='#ac6326',ls='--');axes[1].set(xlabel='RESET pulse count',ylabel='Cross-history G range (%)',title='Unchanged 1% gate')
        axes[0].legend(fontsize=7)
        for ax in axes:ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(R/'extended_reset.png',dpi=150);plt.close(fig)
    validationrows=[{'SET':v['set_count'],'Wait':v['wait_tau'],'RESET count':v['reset_pulses'],'Short envelope (%)':round(100*v['short_envelope_relative_range'],5),'Extended envelope (%)':round(100*v['extended_envelope_relative_range'],5),'Short pass':v['short_pass'],'Extended pass':v['extended_pass']} for v in validations]
    exploratory=[]
    for n,w in sorted({(v['set_count'],v['wait_tau']) for v in validations}):
        stable=sorted([v for v in validations if v['set_count']==n and v['wait_tau']==w and v['short_pass']],key=lambda x:x['reset_pulses'])
        for count in [2,3,4]:
            options=[]
            for group in combinations(stable,count):
                intervals=[np.array(v['conductance_by_history'])[:,[0,1,2,4,5]] for v in group]
                if not all(a.min()>b.max() for a,b in zip(intervals,intervals[1:])):continue
                gaps=[200*(a['nominal_G']-b['nominal_G'])/(a['nominal_G']+b['nominal_G']) for a,b in zip(group,group[1:])]
                options.append((min(gaps),group,gaps))
            if options:
                gap,group,gaps=max(options,key=lambda x:x[0])
                exploratory.append(dict(set_count=n,wait_tau=w,level_count=count,reset_counts=[v['reset_pulses'] for v in group],nominal_G=[v['nominal_G'] for v in group],short_ranges_percent=[100*v['short_envelope_relative_range'] for v in group],adjacent_separation_percent=gaps,minimum_separation_percent=gap,observed_short_envelopes_nonoverlapping=True,passes_original_10_percent_spacing=gap>=10,scope='Exploratory subset of tested nominal histories; no relaxed-gate acceptance or refined-reference claim.'))
    dump('stable_region_diagnostic.json',exploratory)
    if result['accepted']:
        selection=result['selection'];chosen=[v for v in selection['validated_levels'] if v['reset_pulses'] in selection['levels']]
        release=dict(scope='Only the four tested nominal histories, the declared fixed operation sequence and operating window. No arbitrary-history, device-variation, array-circuit or hardware-timing claim.',set_pulses=selection['set_count'],wait_tau=selection['wait_tau'],window=selection['window'],levels=chosen,numerical=selection['numerical'],pulse_parameters=protocol['pulse_parameters'],preparation_read_policy=protocol['preparation'],extra_read_checkpoint_after_reset16=bool(selection.get('extended_reset_max')))
        dump('accepted_levels.json',release)
        headline=f"{len(chosen)} repeatable conductance levels found in the tested scope"
        conclusion=f"SET x{selection['set_count']}, fixed wait {selection['wait_tau']} tau, then RESET counts {selection['levels']}. Accepted window: {selection['window']}."
        numeric=f"Maximum refined-reference error: {max(selection['numerical']['relative_errors_by_history']):.3e}. The last eight SET pulses, wait and RESET probe were repeated for all four histories at rtol=1e-9."
        levelshtml=table([{'RESET count':x['reset_pulses'],'Nominal G (model units)':f"{x['nominal_G']:.9e}",'Short range (%)':round(x['short_envelope_relative_range']*100,6),'Extended range (%)':round(x['extended_envelope_relative_range']*100,6)} for x in chosen],['RESET count','Nominal G (model units)','Short range (%)','Extended range (%)'])
    else:
        headline='No accepted 2–4-level protocol found among the tested candidates'
        conclusion='A failure is limited to these SET counts, waits, RESET counts and four initial histories. It does not establish that every initialization method is impossible.'
        numeric='No accepted level set: the conditional refined-reference stage was not required unless a stability candidate reached it.'
        levelshtml=''
    body='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SET Preparation / RESET Levels</title><style>body{font:15px/1.65 system-ui;background:#101a28;color:#e1eaf3;max-width:1240px;margin:36px auto;padding:0 22px}h1,h2{color:#8edbc8}h1{font-size:30px}.hero{padding:22px;border:1px solid #4a6579;background:#1b2b3b;border-radius:12px}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;margin:18px 0}th,td{padding:10px;text-align:left;border-bottom:1px solid #3a4d61}th{color:#9dc7dd}img{width:100%;border-radius:8px;background:white}a{color:#8edbc8}.note{border-left:3px solid #e6b875;padding:12px 18px;background:#29303b}</style></head><body>'''
    body+=f'<h1>SET preparation → fixed wait → RESET programming</h1><div class="hero"><h2>{html.escape(headline)}</h2><p>{html.escape(conclusion)}</p></div>'
    body+='<p>Same native isothermal COMSOL model and the original four histories. Ti 30 nm / TiOx 20 nm / AlOx 5 nm / Ni 30 nm, 300 K, 696 physical occupancy states and 1137 solver variables. Python only orchestrates native state transfer and analysis.</p><p class="note">Tau is an uncalibrated model time. Conductance remains in model units. A nearly flat terminal SET response is a screening criterion, not proof that all internal history has disappeared.</p>'
    body+='<h2>Chronological experiment sequence</h2>'+table([{'Step':1,'Experiment':'Freeze four native starting histories','Status':'Source hashes verified'},{'Step':2,'Experiment':'SET preparation in blocks of eight; verify after every pulse','Status':str(result['screened_set_counts'])},{'Step':3,'Experiment':'Wait 0 (control), 1, 10 tau, then 0–16 RESET pulses','Status':'Completed for screened counts'},{'Step':4,'Experiment':'Find common separated levels','Status':'<=1% history range; >=10% adjacent mean-G separation'},{'Step':5,'Experiment':'Validate selected levels across histories and operations','Status':f'{len(validations)} level candidates tested'},{'Step':6,'Experiment':'Refined native reference and release','Status':numeric}],['Step','Experiment','Status'])
    body+='<h2>Fixed protocol and acceptance</h2><p>SET +0.8 V, RESET −0.8 V; rise/plateau/fall 0.05/0.2/0.05 tau, zero-bias gap 0.2 tau. READ +0.02 V; rise/plateau/fall 0.02/0.1/0.02 tau. Each eight-SET block starts with a READ and each SET is followed by a READ. RESET programming starts with a READ and verifies every pulse. The identical read schedule is part of the protocol.</p><p>Preparation stationarity requires all four last-step relative G changes, in every history, to be <=0.1%. History range is (maximum−minimum)/mean across the four histories at the same RESET count. The operating envelope uses the same range metric across all histories and all included observations; it must be <=1%.</p><p>Short envelope: original programming verification, readiness READ, 1-tau hold then READ, eight consecutive READs, and an age-matched 0.98-tau zero-bias hold then READ. Extended envelope adds 10-tau hold then READ. These are tested operation windows, not a guarantee for arbitrary timings or histories. Total drift and extra read disturbance are not interchangeable.</p>'
    body+='<h2>SET preparation</h2><img src="set_preparation.png"><h2>RESET response after waiting</h2><p>Light vertical bands mark counts passing the cross-history screening gate only. Stability and separation must also pass.</p><img src="reset_curves.png">'+table(screenrows,['SET pulses','Wait (tau)','Stationary response','Counts with history range <=1%','Separated candidate set'])
    body+='<h2>Operating-window validation</h2>'+ (table(validationrows,['SET','Wait','RESET count','Short envelope (%)','Extended envelope (%)','Short pass','Extended pass']) if validations else '<p>No candidate reached the conditional operating-window stage.</p>')
    body+=levelshtml
    if exploratory:
        body+='<h2>Stable points with narrower spacing: exploratory diagnostic</h2><p>The 10% adjacent-G spacing is the engineering margin chosen for this experiment, not a physical law. The combinations below each pass the short-window 1% repeatability test and have nonoverlapping observed envelopes, but a spacing below 10% still fails the original acceptance gate. They are retrospective candidate subsets, not an accepted lookup-table release; device variation, readout noise and arbitrary histories remain untested.</p>'
        erows=[{'SET':x['set_count'],'Wait':x['wait_tau'],'Levels':x['level_count'],'RESET counts':x['reset_counts'],'Minimum adjacent gap (%)':round(x['minimum_separation_percent'],4),'Original spacing gate':x['passes_original_10_percent_spacing']} for x in exploratory]
        body+=table(erows,['SET','Wait','Levels','RESET counts','Minimum adjacent gap (%)','Original spacing gate'])+'<p><a href="stable_region_diagnostic.json">Exploratory candidate data and scope</a></p>'
    if protocol.get('adaptive_followup'):
        body+='<h2>Recorded adaptive follow-up</h2><p>The predeclared maximum-span extension selected SET128 / wait1 and did not yield an accepted level pair. A second, bounded extension then tested SET32 / wait10, whose RESET12-16 levels had already passed short-window stability but lacked sufficient separation. This second choice was made after inspecting the first results; it was not predeclared. Both extensions stop at RESET32 and retain every original acceptance gate.</p><img src="first_extension_extended_reset.png"><p><a href="first_extension_campaign_result.json">First extension result</a> · <a href="first_extension_extension_screen.json">First extension screening data</a></p>'
    if extension:
        body+='<h2>Conditional RESET-count extension</h2><p>The original 0–16 scan was extended to 32 for one fixed-wait protocol with a shared conductance region. Pulse parameters and all acceptance gates were unchanged. The native count-16 state was continued with one additional checkpoint READ and 16 further RESET pulses with verification; this extra READ is part of the extended protocol.</p><img src="extended_reset.png">'
        body+=f"<p>Extended screening-passing RESET counts: {extension['screening_passing_counts']}. Widest separated candidate set: {extension['separated_groups'][0] if extension['separated_groups'] else 'None'}.</p>"
    body+=f'<h2>Integrity and reproducibility</h2><p>{len(audit)} complete native cases validated. Maximum cross-case oxygen-inventory difference: {max(x["inventory_error"] for x in audit):.3e}. All exported physical occupancies remained strictly between zero and one. Source files were unchanged.</p>'
    body+='<p>The operating-envelope mean uses one endpoint observation for each declared branch. All eight intermediate READ observations were separately checked to lie within the reported short-envelope extrema; this check must pass before this report can be generated.</p>'
    if (R/'boundary_reference.json').exists():
        br=json.loads((R/'boundary_reference.json').read_text(encoding='utf-8'))
        body+=f'<h2>Numerical check near the acceptance boundary</h2><p>{html.escape(br["scope"])} Maximum relative G difference: {br["maximum_relative_G_error"]:.3e}.</p><p><a href="boundary_reference.json">Numerical boundary diagnostic</a></p>'
    body+='<p><a href="protocol.json">Predeclared protocol</a> · <a href="campaign_result.json">Campaign result</a> · <a href="audit.json">Native export audit</a> · <a href="source_manifest.json">Source provenance</a> · <a href="REPRODUCE.txt">Reproduction guide</a></p>'
    if result['accepted']:body+='<p><a href="accepted_levels.json">Accepted levels and restricted operating window</a></p>'
    body=body.replace('<h2>Operating-window validation</h2>','<h2>History consistency by RESET count</h2><img src="history_ranges.png"><h2>Operating-window validation</h2>')
    body+='</body></html>'
    (R/'report.html').write_text(body,encoding='utf-8')
    protocol['status']='completed';protocol['accepted']=result['accepted'];protocol['complete_native_cases']=len(audit)
    dump('protocol.json',protocol)
    dump('analysis_source_hashes.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in list(R.glob('*.py'))+list(R.glob('*.java'))})
    print(json.dumps(dict(accepted=result['accepted'],native_cases=len(audit),selection=result['selection']),indent=2))

if __name__=='__main__':make_report()
