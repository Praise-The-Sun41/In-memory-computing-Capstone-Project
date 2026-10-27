"""Independent single-threaded numerical processes, with per-case logs."""
from concurrent.futures import ThreadPoolExecutor
import subprocess, os, sys
from pathlib import Path
P=Path(__file__).resolve().parent

def job(name):
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    with (P/(name+'.log')).open('w') as out:
        result=subprocess.run([sys.executable,str(P/'run.py'),'--case',name,'--history'],stdout=out,stderr=subprocess.STDOUT,env=env)
    print(name,result.returncode,flush=True)
    if result.returncode:raise RuntimeError(name)

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(job,['no_contact','static_500','chemical_500','static_5000','chemical_5000']))
