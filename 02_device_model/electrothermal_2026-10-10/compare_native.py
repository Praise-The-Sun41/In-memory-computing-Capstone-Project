"""Compare independent native thermal Ti history to tightened Python trajectory."""
from coupled import *
parser=argparse.ArgumentParser();parser.add_argument('--adaptive',action='store_true');args=parser.parse_args()
csv='Ti-thermal-adaptive-native.csv' if args.adaptive else 'Ti-thermal-native.csv'
java='NativeThermalAdaptive.java' if args.adaptive else 'NativeThermalTi.java'
report='native_adaptive_comparison.json' if args.adaptive else 'native_comparison.json'
solved_mph='Ti-thermal-adaptive-geometry.mph' if args.adaptive else 'Ti-thermal-geometry.mph'
d=np.loadtxt(P/csv,delimiter=',',comments='%')
p=dict(np.load(P/'Ti-thermal-C5-R2e-09.npz'))
assert d.shape==(3001,10) and np.allclose(d[:,0],p['t'],atol=1e-10,rtol=0)
assert np.allclose(d[:,1],p['Vs'],atol=1e-10,rtol=0)
metrics=dict(current_peak_normalized_error=float(np.max(abs(d[:,3]-p['I']))/np.max(abs(p['I']))),
    temperature_error_K=float(np.max(abs(d[:,4]-p['T']))),
    device_voltage_error_V=float(np.max(abs(d[:,2]-p['Vdev']))),
    layer_voltage_error_V=float(max(np.max(abs(d[:,7]-p['TiOx_drop'])),np.max(abs(d[:,8]-p['AlOx_drop'])))),
    native_inventory_drift=float(np.max(abs(d[:,5]-d[0,5]))),
    native_heat_balance_Wtau=float(np.max(abs(d[:,6]))),
    native_current_excess_A=float(max(0,np.max(abs(d[:,3]))-Settings().compliance_A)))
gates=dict(current=metrics['current_peak_normalized_error']<.003,temperature=metrics['temperature_error_K']<.05,
    voltage=metrics['device_voltage_error_V']<.003,layer_voltage=metrics['layer_voltage_error_V']<.003,
    inventory=metrics['native_inventory_drift']<2e-6,heat_balance=metrics['native_heat_balance_Wtau']<1e-8,
    compliance=metrics['native_current_excess_A']<1e-9)
result=dict(all_pass=all(gates.values()),gates=gates,metrics=metrics,native_points=len(d),
    scope='Ti lumped temperature + circuit, own preparation + five continuous cycles; Pt thermal native not included',
    imported_Python_trajectory=False,spatial_heat_FEM=False,
    native_csv=csv,native_java=java,native_csv_sha=sha(P/csv),python_npz_sha=sha(P/'Ti-thermal-C5-R2e-09.npz'),
    java_sha=sha(P/java),base_java_sha=sha(P/'NativeThermalTi.java'),
    native_rtol=2e-6 if args.adaptive else 2e-8,adaptive_output_interpolation=args.adaptive,
    solved_model=solved_mph,solved_model_sha=sha(P/solved_mph),
    unsolved_model_sha=sha(P/'Ti-thermal-unsolved.mph'))
(P/report).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));assert result['all_pass']
