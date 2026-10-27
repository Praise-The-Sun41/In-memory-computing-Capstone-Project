"""Audit ports AND every native ionic/chemical state at all requested times."""
from pathlib import Path
import sys,json,argparse,hashlib
import numpy as np
from scipy.special import expit
P=Path(__file__).resolve().parent;SRC=P.parent/'ti_contact_oxidation_2026-10-18'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_python(stem):
    file=SRC/(stem+'.npz');meta=json.loads(file.with_suffix('.json').read_text())
    assert meta['data_sha']==sha(file) and meta['config']['source_sha']==sha(SRC/'interface.py')
    return dict(np.load(file)),meta

def compare(case,smoke=False):
    dest=P/(case+('-smoke' if smoke else '-full'));manifest=json.loads((dest/'manifest.json').read_text())
    assert manifest['interface_source_sha']==sha(SRC/'interface.py')
    assert manifest['java_sha']==sha(dest/(manifest['class_name']+'.java'))
    assert manifest['generator_sha']==sha(P/'build.py')
    assert manifest['chemistry_source_sha']==sha(P.parent/'ti_chemistry_2026-10-18/chemistry.py')
    assert manifest['thermal_source_sha']==sha(P.parent/'electrothermal_2026-10-10/coupled.py')
    assert manifest['template_sha']==sha(P.parent/'electrothermal_2026-10-10/Ti-thermal-unsolved.mph')
    py,pm=load_python(case+'-C5-R2e-09')
    hashes={case+'-C5-R2e-09.npz':pm['data_sha']}
    if smoke:py={k:v[:481] for k,v in py.items()}
    else:
        hold,hm=load_python(case+'-write-hold');assert hm['parent_sha']==pm['data_sha']
        hashes[case+'-write-hold.npz']=hm['data_sha'];hold['t']+=py['t'][-1]
        py={k:np.concatenate([v,hold[k][1:]]) for k,v in py.items()}
    ports=np.loadtxt(dest/'ports.csv',delimiter=',',comments='%')
    states=np.loadtxt(dest/'states.csv',delimiter=',',comments='%')
    assert np.isfinite(ports).all() and np.isfinite(states).all(), 'Non-finite native output'
    n=manifest['n_oxide'];nm=manifest['n_metal'];nb=manifest['n_bound'];points=manifest['points']
    assert ports.shape==(points,len(manifest['ports'])) and states.shape==(points,1+n+nm+nb)
    assert np.allclose(ports[:,0],py['t'],rtol=0,atol=1e-9) and np.allclose(states[:,0],py['t'],rtol=0,atol=1e-9)
    expected=np.c_[expit(py['y'][:,:n]),py['y'][:,n:n+nm],py['y'][:,n+nm+3:]]
    ns=states[:,1:];difference=abs(ns-expected)
    err=lambda i,k:float(max(abs(ports[:,i]-py[k])))
    peak=float(max(abs(py['I'])))
    metrics=dict(current_peak_relative=err(2,'I')/peak,current_absolute_A=err(2,'I'),
        source_error_V=err(1,'Vs'),oxide_voltage_error_V=err(3,'Voxide'),contact_voltage_error_V=err(4,'Vcontact'),
        device_voltage_error_V=err(5,'Vdev'),output_voltage_error_V=err(6,'Vout'),temperature_error_K=err(7,'T'),
        TiOx_drop_error_V=err(8,'TiOx_drop'),AlOx_drop_error_V=err(9,'AlOx_drop'),
        contact_resistance_relative=float(max(abs(ports[:,10]/py['Rcontact']-1))),
        bound_fraction_error=err(11,'bound_fraction'),bound_oxygen_error=err(12,'bound_oxygen'),
        mobile_oxygen_error=err(13,'mobile_oxygen'),
        total_oxygen_error=err(14,'total_Ti_oxygen'),region_oxygen_fraction_error=err(15,'region_oxygen_fraction'),
        device_conductance_relative=float(max(abs(ports[:,24]/py['G']-1))),
        oxide_occupancy_max=float(difference[:,:n].max()),metal_occupancy_max=float(difference[:,n:n+nm].max()),
        bound_occupancy_max=float(difference[:,n+nm:].max()),minimum_native_occupancy=float(ns.min()),maximum_native_occupancy=float(ns.max()),
        inventory_drift=float(max(abs(ports[:,16]-ports[0,16]))),heat_balance=float(max(abs(ports[:,17]))),
        power_balance=float(abs(ports[:,18:20]).max()),voltage_balance=float(abs(ports[:,20]).max()),
        exchange_flux_peak_relative=err(21,'exchange_flux')/max(float(max(abs(py['exchange_flux']))),1e-12),
        binding_flux_peak_relative=err(22,'binding_flux')/max(float(max(abs(py['binding_flux']))),1e-12),
        ionic_passivation_relative=float(max(abs(ports[:,23]/py['exchange_factor']-1))),
        compliance_excess_A=float(max(0,max(abs(ports[:,2]))-manifest['thermal']['compliance_A'])))
    gates=dict(current=metrics['current_peak_relative']<1e-3,source=metrics['source_error_V']<1e-9,
        all_voltages=max(metrics[k] for k in ['oxide_voltage_error_V','contact_voltage_error_V','device_voltage_error_V','output_voltage_error_V','TiOx_drop_error_V','AlOx_drop_error_V'])<1e-3,
        temperature=metrics['temperature_error_K']<.01,contact_resistance=metrics['contact_resistance_relative']<1e-3,
        oxide_states=metrics['oxide_occupancy_max']<5e-4,metal_states=metrics['metal_occupancy_max']<5e-4,
        bound_states=metrics['bound_occupancy_max']<5e-5,bound_fraction=metrics['bound_fraction_error']<5e-5,
        total_oxygen=metrics['total_oxygen_error']<1e-6,region_oxygen=metrics['region_oxygen_fraction_error']<1e-5,
        separate_oxygen_pools=max(metrics['bound_oxygen_error'],metrics['mobile_oxygen_error'])<1e-6,
        device_conductance=metrics['device_conductance_relative']<1e-3,
        physical_bounds=metrics['minimum_native_occupancy']>0 and metrics['maximum_native_occupancy']<1,
        inventory=metrics['inventory_drift']<2e-6,heat=metrics['heat_balance']<1e-8,
        power=metrics['power_balance']<1e-10,voltage_closure=metrics['voltage_balance']<1e-10,
        compliance=metrics['compliance_excess_A']<1e-9,
        exchange_flux=metrics['exchange_flux_peak_relative']<.01,
        binding_flux=metrics['binding_flux_peak_relative']<.01,
        ionic_passivation=metrics['ionic_passivation_relative']<1e-3)
    cycle=1 if smoke else 5
    ip=int(np.flatnonzero(py['kind']==f'C{cycle}-pos-fall')[-1]);im=int(np.flatnonzero(py['kind']==f'C{cycle}-neg-fall')[-1])
    observables=dict(cycle=cycle,native_endpoint_Gpositive_Gnegative=float(ports[ip,24]/ports[im,24]),
        python_endpoint_Gpositive_Gnegative=float(py['G'][ip]/py['G'][im]),
        endpoint_conductance_definition='Instantaneous native device G at each endpoint temperature, not the frozen-300K diagnostic.')
    if not smoke:
        irs=int(np.flatnonzero(py['kind']=='SET-read')[-1]);irr=int(np.flatnonzero(py['kind']=='RESET-read')[-1])
        observables.update(native_sequential_read_ratio=float(abs(ports[irs,2]/ports[irr,2])),
            python_sequential_read_ratio=float(abs(py['I'][irs]/py['I'][irr])))
    result=dict(all_pass=all(gates.values()),gates=gates,metrics=metrics,case=case,native_points=points,
        ionic_and_chemical_states=n+nm+nb,observables=observables,scope='One continuous cycle' if smoke else 'Five uninterrupted cycles followed by SET-hold-read-RESET-hold-read, 250.8 tau total',
        source_hashes=hashes,ports_sha=sha(dest/'ports.csv'),states_sha=sha(dest/'states.csv'),
        model_sha=sha(dest/'Ti-chemistry-contact.mph'),manifest_sha=sha(dest/'manifest.json'),
        comparator_sha=sha(__file__),imported_initial_state=True,imported_Python_trajectory=False,
        independent_discretization=False,spatial_3D_FEM=False,experimental_validation=False)
    (dest/'comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    assert result['all_pass'],gates
    return py,ports,states,result

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--case',required=True);a.add_argument('--smoke',action='store_true');v=a.parse_args();compare(v.case,v.smoke)
