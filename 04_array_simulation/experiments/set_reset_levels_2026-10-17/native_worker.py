from pathlib import Path
import subprocess,hashlib
from native_io import R,OLD,dump

BIN=Path('E:/COMSOL/COMSOLMP64/COMSOL64/Multiphysics/bin/win64')

def run(name,y,body):
    source='''import com.comsol.model.*;import com.comsol.model.util.*;import java.util.*;
public class NAME extends SetProtocol {public static Model run(){try{
m=ModelUtil.load("protocol","SOURCE");
names=m.component("c").physics("ge").feature("ge1").getStringArray("name");if(names.length!=1137)throw new Exception("Unexpected variable count");
expr=new String[names.length+4];expr[0]="t";expr[1]="c.conductance";expr[2]="c.inventory";expr[3]="c.bias";for(int j=0;j<names.length;j++)expr[j+4]="c."+names[j];
double[] y=new double[]{STATE};
BODY
System.out.println("WORKER_COMPLETE NAME");return m;
}catch(Exception e){throw new RuntimeException(e);}}public static void main(String[] args){run();}}
'''.replace('NAME',name).replace('SOURCE',(OLD/'bounded-comsol-strict-full-r6-unsolved.mph').as_posix()).replace('STATE',','.join(format(float(x),'.17g') for x in y)).replace('BODY',body)
    path=R/(name+'.java');path.write_text(source,encoding='ascii')
    path.with_suffix('.class').unlink(missing_ok=True)
    dump(name+'-state.json',dict(initial_state_sha256=hashlib.sha256(y.tobytes()).hexdigest(),state_transfer='Native COMSOL exported state; no Python integration'))
    for d in ['prefs-'+name,'temp-'+name]:(R/d).mkdir(exist_ok=True)
    with (R/(name+'-compile.log')).open('w') as out:subprocess.run([str(BIN/'comsolcompile.exe'),name+'.java'],cwd=R,stdout=out,stderr=subprocess.STDOUT,check=True)
    assert (R/(name+'.class')).exists(),name
    print('START',name,flush=True)
    with (R/(name+'-console.log')).open('w') as out:subprocess.run([str(BIN/'comsolbatch.exe'),'-np','1','-alloc','native','-prefsdir','prefs-'+name,'-tmpdir','temp-'+name,'-autosave','off','-inputfile',name+'.class','-batchlog',name+'-batch.log'],cwd=R,stdout=out,stderr=subprocess.STDOUT,check=True)
    assert 'WORKER_COMPLETE '+name in (R/(name+'-console.log')).read_text(errors='replace'), 'Incomplete native batch '+name
    print('DONE',name,flush=True)
