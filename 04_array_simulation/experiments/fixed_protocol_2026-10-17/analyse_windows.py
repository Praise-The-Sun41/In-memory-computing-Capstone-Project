from pathlib import Path
import json,numpy as np
from itertools import product
from analyse_native import load,read_levels,history
from functools import lru_cache
load=lru_cache(maxsize=96)(load)
R=Path(__file__).resolve().parent

def analyse():
 h=history();chosen=h['selected'] or h['candidates'][-1]['reset_count'];result=dict(chosen=chosen,history_gate_pass=h['selected'] is not None)
 # A failed continuous attempt is never counted as a completed reference.
 continuous=R/'results'/f'continuous_{chosen}.csv'
 if continuous.exists():
  ref_tag=f'continuous_{chosen}';result['single_run_continuous_completed']=True
  result['reference_scope']='Single continuous native initialization and programming reference.'
 else:
  assert 'INDEPENDENT_REFERENCE_COMPLETE' in (R/'recover-reference-console.log').read_text(encoding='utf-8',errors='replace')
  ref_tag=f'independent_probe{chosen}';partial=load('continuous_attempt_partial')
  result['single_run_continuous_completed']=False
  result['failed_continuous_attempt_last_tau']=float(partial[-1,0])
  result['reference_scope']='Single-run continuous attempt failed near tau=65.84. Reference uses its fully completed continuous initialization checkpoint at tau=64.42, then an independent native 16-SET probe at rtol=1e-9. This validates two-block reference versus the original checkpoint path; it is not a completed uninterrupted run.'
 result['reference_tag']=ref_tag
 a=read_levels(load(ref_tag))[-17:];b=read_levels(load(f'probe{chosen}_history0'))
 assert len(a)==len(b)==17
 result['continuation_validation_relative_error']=float(np.max(abs(a-b)/abs(a)))
 result['continuation_validation_passed']=bool(result['continuation_validation_relative_error']<=json.loads((R/'protocol.json').read_text(encoding='utf-8'))['gates']['numerical_relative_read_error'])
 result['hold']=[];result['read_disturb']=[];result['halfselect']=[]
 for state in [0,1,2]:
  reference=read_levels(load(f'reads1_state{state}'))[-1]
  for t in [1,10,100]:
   value=read_levels(load(f'hold{t}_state{state}'))[-1];error=float(abs(value-reference)/abs(reference));result['hold'].append(dict(state=state,wait_tau=t,relative_drift=error,passed=error<=.01))
  for n in [1,16,80]:
   value=read_levels(load(f'reads{n}_state{state}'))[-1];sham=read_levels(load(f'readsham{n}_state{state}'))[-1];error=float(abs(value-sham)/abs(sham));total=float(abs(value-reference)/abs(reference));result['read_disturb'].append(dict(state=state,reads=n,relative_sham_corrected_change=error,passed=error<=.01,relative_total_change=total,total_change_within_one_percent=total<=.01))
  for sign in ['set','reset']:
   for n in [1,16]:
    prefix=f'array2_{sign}_N{n}_state{state}_factor';g={f:read_levels(load(prefix+str(f)))[-1] for f in [1.0,.5,0.0]};error=float(abs(g[.5]-g[0.])/abs(g[0.]));result['halfselect'].append(dict(state=state,direction=sign,pulses=n,selected_G=float(g[1.]),halfselected_G=float(g[.5]),unselected_sham_G=float(g[0.]),relative_sham_corrected_change=error,passed=error<=.01))
 # Assemble actual ideal-source 2x2 column readout from native cell trajectories.
 array_rows=[]
 for sign in ['set','reset']:
  for n in [1,16]:
   for pattern in product(range(3),repeat=4):
    actual=[];isolated=[]
    for k,state in enumerate(pattern):
     prefix=f'array2_{sign}_N{n}_state{state}_factor'
     factor=1.0 if k==0 else (.5 if k in [1,2] else 0.0)
     actual.append(read_levels(load(prefix+str(factor)))[-1])
     isolated.append(read_levels(load(prefix+str(1.0 if k==0 else 0.0)))[-1])
    actual=np.array(actual).reshape(2,2);isolated=np.array(isolated).reshape(2,2)
    for bits in [[1,0],[0,1],[1,1]]:
     voltage=.02*np.array(bits);a=voltage@actual;b=voltage@isolated
     error=float(np.linalg.norm(a-b)/np.linalg.norm(b));array_rows.append(dict(direction=sign,pulses=n,initial_pattern=list(pattern),row_bits=bits,relative_read_vector_error=error))
 result['array2_readout']=array_rows
 result['array2_max_relative_read_vector_error']=max(x['relative_read_vector_error'] for x in array_rows)
 result['scope']='COMSOL native device extraction; 2x2 ideal voltage sources, zero line resistance; ideal off isolation uses zero-bias sham, not a transistor simulation. Array readout is a frozen-state snapshot after common terminal verification; 81 initial-state patterns and three binary row inputs are enumerated.'
 result['lookup_release_allowed']=bool(result['history_gate_pass'] and result['continuation_validation_passed'])
 result['next_gate']='A deployment also requires selecting a passing hold/read window and an addressing scheme. Absolute hardware timing remains uncalibrated.'
 (R/'window_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':analyse()
