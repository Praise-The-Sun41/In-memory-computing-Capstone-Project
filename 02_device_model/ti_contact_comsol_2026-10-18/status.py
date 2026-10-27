from pathlib import Path
import re,json,datetime
P=Path(__file__).resolve().parent
out=[]
for d in P.iterdir():
    if not d.is_dir() or not (d/'manifest.json').exists():continue
    man=json.loads((d/'manifest.json').read_text());f=d/'batch.log'
    if not f.exists():out.append(dict(case=d.name,status='not started'));continue
    text=f.read_text(encoding='utf-8',errors='replace');matches=re.findall(r'^\s*(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s',text,re.M)
    t=float(matches[-1][1]) if matches else 0.;clock=0.;label='assembly'
    for duration,a,b,name,shape in man['rows']:
        clock+=duration
        if t<=clock+1e-8:label=name;break
    result=json.loads((d/'comparison.json').read_text())['all_pass'] if (d/'comparison.json').exists() else None
    out.append(dict(case=d.name,tau=t,phase=label,accepted=result,solved=(d/'Ti-chemistry-contact.mph').exists()))
print(datetime.datetime.now().isoformat(timespec='seconds'));print(json.dumps(out,indent=2))
