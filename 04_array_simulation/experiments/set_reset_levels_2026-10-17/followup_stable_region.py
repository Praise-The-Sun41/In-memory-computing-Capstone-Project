import json,subprocess,sys,time,shutil
from datetime import datetime,timezone
from native_io import R,dump

start=time.monotonic()
while True:
    log=(R/'finish.log').read_text(errors='replace')
    if 'Traceback (most recent call last)' in log:raise RuntimeError('First extension/report failed; inspect before follow-up')
    if 'ALL_EXPERIMENTS_COMPLETE' in log:break
    if time.monotonic()-start>7200:raise TimeoutError('First extension incomplete')
    time.sleep(3)
result=json.loads((R/'campaign_result.json').read_text(encoding='utf-8'))
if not result['accepted']:
    for name in ['campaign_result.json','extension_screen.json','extension_decision.json','report.html','extended_reset.png']:
        shutil.copy2(R/name,R/('first_extension_'+name))
    protocol=json.loads((R/'protocol.json').read_text(encoding='utf-8'))
    protocol['status']='adaptive_followup_running'
    protocol['adaptive_followup']=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),set_count=32,wait_tau=10,maximum_reset_count=32,reason='Original maximum-span extension used SET128/wait1, whose tested levels all failed the 1% operating-window gate. Independently test extension of SET32/wait10, where RESET12-16 already passed short-window stability but lacked 10% separation. Bounded to one additional protocol, same pulses and gates. This was selected after inspecting earlier results, not predeclared.')
    dump('protocol.json',protocol)
    with (R/'stable_extension.log').open('w') as out:
        subprocess.run([sys.executable,'-c','from extend_reset import main; main((32,10))'],cwd=R,stdout=out,stderr=subprocess.STDOUT,check=True)
    subprocess.run([sys.executable,'report_results.py'],cwd=R,check=True)
print('STABLE_REGION_FOLLOWUP_COMPLETE',flush=True)
