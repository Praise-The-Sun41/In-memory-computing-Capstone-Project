"""One-shot driver: accept the smoke result, then run the full 500-ohm case."""
from pathlib import Path
import subprocess,sys,time,os,json
P=Path(__file__).resolve().parent
smoke=P/'chemical_500-smoke';full=P/'chemical_500-full'
env=dict(os.environ,OPENBLAS_NUM_THREADS='1')
start=time.monotonic()
while True:
    log=(smoke/'console.log').read_text(encoding='utf-8',errors='replace') if (smoke/'console.log').exists() else ''
    if 'CHEMISTRY_NATIVE_COMPLETE' in log:break
    if 'Exception' in log or 'Error:' in log:raise RuntimeError('Smoke run reported an error; inspect logs')
    if time.monotonic()-start>10800:raise TimeoutError('Smoke completion was not confirmed within 3 hours')
    time.sleep(5)
subprocess.run([sys.executable,str(P/'compare.py'),'--case','chemical_500','--smoke'],check=True,env=env)
print('SMOKE_ACCEPTED_STARTING_FULL_500',flush=True)
assert not (full/'Ti-chemistry-contact.mph').exists(),'Preserve completed model'
exe=Path('E:/COMSOL/COMSOLMP64/COMSOL64/Multiphysics/bin/win64/comsolbatch.exe')
with (full/'console.log').open('w') as out:
    run=subprocess.run([str(exe),'-np','4','-alloc','native','-autosave','off','-inputfile','Chem500Full.class','-batchlog','batch.log'],cwd=full,env=env,stdout=out,stderr=subprocess.STDOUT)
assert run.returncode==0 and (full/'Ti-chemistry-contact.mph').exists(),'Native full run failed'
subprocess.run([sys.executable,str(P/'compare.py'),'--case','chemical_500'],check=True,env=env)
print('FULL_500_ACCEPTED',flush=True)
