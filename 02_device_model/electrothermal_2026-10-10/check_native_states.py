"""Check all sampled native occupancies, beyond terminal-current agreement."""
from coupled import *
p=argparse.ArgumentParser();p.add_argument('--adaptive',action='store_true');a=p.parse_args()
variant='adaptive-' if a.adaptive else ''
native=P/f'Ti-thermal-{variant}states.csv'
d=np.loadtxt(native,delimiter=',',comments='%')
py=dict(np.load(P/'Ti-thermal-C5-R2e-09.npz'))
assert d.shape==(3001,697) and np.allclose(d[:,0],py['t'],atol=1e-10,rtol=0)
reference=np.c_[expit(py['y'][:,:440]),py['y'][:,440:696]]
state=d[:,1:]
metrics=dict(minimum_occupancy=float(state.min()),maximum_occupancy=float(state.max()),
             oxide_maximum_absolute_difference=float(abs(state[:,:440]-reference[:,:440]).max()),
             metal_maximum_absolute_difference=float(abs(state[:,440:]-reference[:,440:]).max()))
gates=dict(physical_bounds=bool(np.all((state>0)&(state<1))),
           oxide_state=metrics['oxide_maximum_absolute_difference']<5e-4,
           metal_state=metrics['metal_maximum_absolute_difference']<5e-4)
result=dict(all_pass=all(gates.values()),metrics=metrics,gates=gates,
            native_states_csv=native.name,native_states_sha=sha(native),
            presentation_model=f'Ti-thermal-{variant}presentation.mph',
            presentation_model_sha=sha(P/f'Ti-thermal-{variant}presentation.mph'),
            python_npz_sha=sha(P/'Ti-thermal-C5-R2e-09.npz'),
            export_java_sha=sha(P/('ExportThermalAdaptiveStates.java' if a.adaptive else 'ExportThermalStates.java')),
            scope='All 696 ionic states at 3001 output times; no spatial-grid refinement claim')
(P/f'native_{"adaptive_" if a.adaptive else ""}states_check.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2));assert result['all_pass']
