import json,subprocess,sys,time
from native_io import R

start=time.monotonic()
while True:
    log=(R/'campaign.log').read_text(errors='replace')
    if 'Traceback (most recent call last)' in log:raise RuntimeError('Primary campaign failed; preserve logs and inspect before continuing')
    if 'EXPERIMENT_CAMPAIGN_COMPLETE' in log:break
    if time.monotonic()-start>10800:raise TimeoutError('Primary campaign did not complete')
    time.sleep(3)
result=json.loads((R/'campaign_result.json').read_text(encoding='utf-8'))
if not result['accepted']:
    with (R/'extension.log').open('w') as out:subprocess.run([sys.executable,'extend_reset.py'],cwd=R,stdout=out,stderr=subprocess.STDOUT,check=True)
subprocess.run([sys.executable,'report_results.py'],cwd=R,check=True)
print('ALL_EXPERIMENTS_COMPLETE',flush=True)
