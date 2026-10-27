from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess,sys,os
P=Path(__file__).resolve().parent
def job(name):
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    with (P/(name+'.log')).open('w') as out:
        r=subprocess.run([sys.executable,str(P/'run.py'),'--case',name],env=env,stdout=out,stderr=subprocess.STDOUT)
    print(name,r.returncode,flush=True)
    return name,r.returncode
if __name__=='__main__':
    # Processes isolate temporary changes to inherited thermal-module constants.
    names=['Ti_dynamic500','Pt_matched500','Ti_frozen500','Ti_static500','Ti_dynamic5000','Pt_matched5000','Ti_static5000','Pt_barrier']
    with ThreadPoolExecutor(max_workers=3) as pool:out=list(pool.map(job,names))
    assert all(code==0 for name,code in out),out
