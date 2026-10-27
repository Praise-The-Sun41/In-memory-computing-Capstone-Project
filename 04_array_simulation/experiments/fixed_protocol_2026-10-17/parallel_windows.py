from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess,time,json,sys
R=Path(__file__).resolve().parent;BIN=Path('E:/COMSOL/COMSOLMP64/COMSOL64/Multiphysics/bin/win64')

def runclass(name,text,marker):
    (R/(name+'.java')).write_text(text,encoding='ascii')
    for d in ['prefs-'+name,'temp-'+name]:(R/d).mkdir(exist_ok=True)
    with (R/(name+'-compile.log')).open('w') as out:subprocess.run([str(BIN/'comsolcompile.exe'),name+'.java'],cwd=R,stdout=out,stderr=subprocess.STDOUT,check=True)
    if not (R/(name+'.class')).exists():raise RuntimeError('Compile failed '+name)
    print('START',name,flush=True)
    with (R/(name+'-console.log')).open('w') as out:subprocess.run([str(BIN/'comsolbatch.exe'),'-np','1','-alloc','native','-prefsdir','prefs-'+name,'-tmpdir','temp-'+name,'-autosave','off','-inputfile',name+'.class','-batchlog',name+'-batch.log'],cwd=R,stdout=out,stderr=subprocess.STDOUT,check=True)
    if marker not in (R/(name+'-console.log')).read_text(errors='replace'):raise RuntimeError('Incomplete native run '+name)
    print('DONE',name,flush=True)

def main():
    start=time.monotonic()
    while not (R/'parallel_history_complete.json').exists():
        log=(R/'parallel-history.log').read_text(errors='replace')
        if 'Traceback (most recent call last)' in log:raise RuntimeError('History solver failed')
        if time.monotonic()-start>7200:raise TimeoutError('History results not confirmed')
        time.sleep(5)
    h=json.loads((R/'parallel_history_complete.json').read_text(encoding='utf-8'));chosen=h['selected'] or h['candidates'][-1]['reset_count']
    print('History gate:',h['selected'],'diagnostic checkpoint:',chosen,flush=True)
    template=(R/'WindowProtocol.template').read_text(encoding='utf-8').replace('CHOSEN',str(chosen))
    a=template.index('  List<double[]> whole=');b=template.index('  for(int state=0;',a)
    only_windows=template[:a]+template[b:]
    jobs=[]
    for state in range(3):
        name=f'WindowState{state}';text=only_windows.replace('WindowProtocol',name).replace('for(int state=0;state<3;state++)',f'for(int state={state};state<{state+1};state++)').replace('window_last_native.mph',f'window_state{state}_last_native.mph')
        jobs.append((name,text,'EXPERIMENT_WINDOW_COMPLETE'))
    continuous='''import com.comsol.model.*;import com.comsol.model.util.*;import java.util.*;
public class ContinuousNative extends FixedProtocol {public static Model run(){try{
double[] fresh=seed();m=ModelUtil.load("protocol","bounded-comsol-strict-full-r6-unsolved.mph");
List<double[]> whole=new ArrayList<>();int previous=0;for(int n:new int[]{8,32,128}){if(n>CHOSEN)break;whole.addAll(pulses(-.8,n-previous,false));previous=n;}whole.addAll(pulses(.8,16,true));solve("continuous_CHOSEN",fresh,whole,1e-9);System.out.println("CONTINUOUS_NATIVE_COMPLETE");return m;
}catch(Exception e){throw new RuntimeException(e);}}public static void main(String[] args){run();}}
'''.replace('CHOSEN',str(chosen))
    jobs.append(('ContinuousNative',continuous,'CONTINUOUS_NATIVE_COMPLETE'))
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda args:runclass(*args),jobs))
    subprocess.run([sys.executable,'analyse_windows.py'],cwd=R,check=True)
    print('NATIVE_DIAGNOSTIC_SUITE_COMPLETE',flush=True)
if __name__=='__main__':main()
