"""Independent COMSOL solver workers; Python only transfers native checkpoints and analyses."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess,json,time,hashlib,sys
import numpy as np
from analyse_native import load,history
R=Path(__file__).resolve().parent;BIN=Path('E:/COMSOL/COMSOLMP64/COMSOL64/Multiphysics/bin/win64')

def worker(count,j):
    init=R/'results'/f'init{count}_history{j}.csv';probe=R/'results'/f'probe{count}_history{j}.csv'
    if probe.exists():
        a=load(probe.stem)
        if a.shape[0]==231:return
    skip=init.exists()
    if skip:source=init.stem;remaining=0
    elif count==8:source=['','history_high','history_low','history_aged'][j];remaining=8
    else:source=f'init{8 if count==32 else 32}_history{j}';remaining=count-(8 if count==32 else 32)
    data=load(source);y=data[-1,4:];assert len(y)==1137
    name=f'NativeC{count}H{j}';state=','.join(format(float(x),'.17g') for x in y)
    text='''import com.comsol.model.*;import com.comsol.model.util.*;
public class NAME extends FixedProtocol {public static Model run(){try{
m=ModelUtil.load("protocol","bounded-comsol-strict-full-r6-unsolved.mph");
names=m.component("c").physics("ge").feature("ge1").getStringArray("name");if(names.length!=1137)throw new Exception("State ordering changed");
expr=new String[names.length+4];expr[0]="t";expr[1]="c.conductance";expr[2]="c.inventory";expr[3]="c.bias";for(int i=0;i<names.length;i++)expr[i+4]="c."+names[i];
double[] y=new double[]{STATE};
INIT
solve("probeCOUNT_historyHISTORY",y,pulses(.8,16,true),1e-8);
System.out.println("WORKER_COMPLETE NAME");return m;}catch(Exception e){throw new RuntimeException(e);}}public static void main(String[] args){run();}}
'''.replace('NAME',name).replace('STATE',state).replace('COUNT',str(count)).replace('HISTORY',str(j)).replace('INIT','' if skip else f'y=solve("init{count}_history{j}",y,pulses(-.8,{remaining},false),1e-8);')
    (R/(name+'.java')).write_text(text,encoding='ascii')
    (R/(name+'-source.json')).write_text(json.dumps(dict(native_checkpoint=source,sha256=hashlib.sha256((R/'results'/(source+'.csv')).read_bytes()).hexdigest(),state_transfer='Exact exported COMSOL double values; no Python physical integration',remaining_reset_pulses=remaining),indent=2))
    for d in ['prefs-'+name,'temp-'+name]:(R/d).mkdir(exist_ok=True)
    with (R/(name+'-compile.log')).open('w') as out:subprocess.run([str(BIN/'comsolcompile.exe'),name+'.java'],cwd=R,stdout=out,stderr=subprocess.STDOUT,check=True)
    if not (R/(name+'.class')).exists():raise RuntimeError('Compilation failed '+name)
    print('START',name,flush=True)
    with (R/(name+'-console.log')).open('w') as out:subprocess.run([str(BIN/'comsolbatch.exe'),'-np','1','-alloc','native','-prefsdir','prefs-'+name,'-tmpdir','temp-'+name,'-autosave','off','-inputfile',name+'.class','-batchlog',name+'-batch.log'],cwd=R,stdout=out,stderr=subprocess.STDOUT,check=True)
    if 'WORKER_COMPLETE '+name not in (R/(name+'-console.log')).read_text(errors='replace'):raise RuntimeError('Native solve not confirmed '+name)
    load(f'probe{count}_history{j}');print('DONE',name,flush=True)

def main():
    for count in [8,32,128]:
        with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda j:worker(count,j),range(4)))
        result=history();print('CANDIDATE',count,result['candidates'][-1]['maximum_relative_spread'],flush=True)
        if result['selected'] is not None:break
    (R/'parallel_history_complete.json').write_text(json.dumps(result,indent=2));print('EXPERIMENT_HISTORY_COMPLETE',flush=True)
if __name__=='__main__':main()
