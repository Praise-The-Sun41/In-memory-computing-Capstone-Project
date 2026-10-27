"""Continue the same authorized experiment run after native history extraction."""
from pathlib import Path
import subprocess,time,json,hashlib,sys
R=Path(__file__).resolve().parent
BIN=Path('E:/COMSOL/COMSOLMP64/COMSOL64/Multiphysics/bin/win64')
start=time.monotonic()
while True:
    console=(R/'parallel-history.log').read_text(errors='replace') if (R/'parallel-history.log').exists() else ''
    batch=(R/'parallel-history.log').read_text(errors='replace') if (R/'parallel-history.log').exists() else ''
    if 'EXPERIMENT_HISTORY_COMPLETE' in console:break
    if 'Traceback (most recent call last)' in batch or 'Error running java class' in batch:raise RuntimeError('Native history run failed; dependent run not started')
    if time.monotonic()-start>7200:raise TimeoutError('History run not confirmed within 2 hours')
    time.sleep(5)
subprocess.run([sys.executable,'analyse_native.py','history'],cwd=R,check=True)
h=json.loads((R/'history_summary.json').read_text(encoding='utf-8'));chosen=h['selected'] or h['candidates'][-1]['reset_count']
print('History results:',[(r['reset_count'],r['maximum_relative_spread']) for r in h['candidates']],flush=True)
print('Selected checkpoint for window/half-select diagnosis:',chosen,'gate passed:',h['selected'] is not None,flush=True)
(R/'WindowProtocol.java').write_text((R/'WindowProtocol.template').read_text(encoding='utf-8').replace('CHOSEN',str(chosen)),encoding='ascii')
with (R/'compile-window.log').open('w') as out:subprocess.run([str(BIN/'comsolcompile.exe'),'WindowProtocol.java'],cwd=R,stdout=out,stderr=subprocess.STDOUT,check=True)
if not (R/'WindowProtocol.class').exists():raise RuntimeError('No compiled window runner')
with (R/'window-console.log').open('w') as out:subprocess.run([str(BIN/'comsolbatch.exe'),'-np','1','-alloc','native','-prefsdir','prefs','-tmpdir','temp','-autosave','off','-inputfile','WindowProtocol.class','-batchlog','window-batch.log'],cwd=R,stdout=out,stderr=subprocess.STDOUT,check=True)
if 'EXPERIMENT_WINDOW_COMPLETE' not in (R/'window-console.log').read_text(errors='replace'):raise RuntimeError('Window/half-select experiment incomplete; inspect native logs')
subprocess.run([sys.executable,'analyse_windows.py'],cwd=R,check=True)
print('NATIVE_DIAGNOSTIC_SUITE_COMPLETE',flush=True)
