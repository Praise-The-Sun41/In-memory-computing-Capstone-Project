"""Read completed runs, verify coupled histories and export figures/report data."""
from coupled import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

KINDS=('Ti','Pt','Pt_contact');MODES=('original','limited','thermal')
labels={'Ti':'Ti / Ni','Pt':'Pt / Pt, ideal contacts','Pt_contact':'Pt / Pt, contact barriers'}
colors={'original':'#8a99a6','limited':'#2979ac','thermal':'#dc7032'}

def main():
    data={};summary={};validation={};rng=np.random.default_rng(26)
    for k in KINDS:
        summary[k]={}
        for mode in MODES:
            name=f'{k}-{mode}-C5-R2e-08';d=dict(np.load(P/(name+'.npz')));meta=json.loads((P/(name+'.json')).read_text())
            assert meta['data_sha']==sha(P/(name+'.npz')) and meta['source_sha']==sha(P/'coupled.py')
            data[k,mode]=d;m=Device(k,mode);ratios=[];gpos=[];gneg=[]
            for cy in range(1,6):
                ip=np.flatnonzero(d['kind']==f'C{cy}-pos-fall')[-1];im=np.flatnonzero(d['kind']==f'C{cy}-neg-fall')[-1]
                # Freeze ionic state and cool electronically to 300 K for state-only diagnostic.
                vals=[]
                for ix in (ip,im):
                    z=d['y'][ix].copy();z[m.n]=300.;vals.append(m.diagnostic(z,0.)['G'])
                gpos.append(vals[0]);gneg.append(vals[1]);ratios.append(vals[0]/vals[1])
            cyc=np.char.startswith(d['kind'],'C');cc=d['limited']&cyc
            summary[k][mode]=dict(checks=meta['checks'],C1_ratio_300K=ratios[0],C5_ratio_300K=ratios[-1],
                ratios_300K=ratios,Gpos_300K_S=gpos,Gneg_300K_S=gneg,
                compliance_sample_fraction=float(cc.sum()/cyc.sum()),
                max_source_minus_device_V=float(np.max(abs(d['Vs'][cyc]-d['Vdev'][cyc]))))
            keys=['t','Vs','Vout','Vdev','I','T','G','TiOx_drop','AlOx_drop','Vcontact','Pdev','Pseries','limited']
            np.savetxt(P/(name+'-trace.csv'),np.column_stack([d[x] for x in keys]),delimiter=',',header=','.join(keys),comments='')
        # Independently solved old COMSOL trajectory checks the 300 K no-circuit limit.
        path=(R/'outputs/bounded-comsol-linear10-L256-sparse-raw.csv' if k=='Ti' else
              R/('pt_blocking_2026-10-06' if k=='Pt' else 'pt_contacts_2026-10-06')/'pt-native.csv')
        native=np.loadtxt(path,delimiter=',',comments='%');d=data[k,'original'];mask=np.char.startswith(d['kind'],'C')
        t=d['t'][mask]-21.4 if k=='Ti' else d['t'][mask]
        ref=np.interp(t,native[:,0],native[:,3])*m.scale
        err=float(np.max(abs(d['I'][mask]-ref))/np.max(abs(ref)))
        assert err<.003,(k,err)
        validation[k]=dict(old_native_COMSOL_peak_normalized_current_error=err)
        # Actual conducting states: verify CC branch and thermal-column Jacobian independently.
        d=data[k,'thermal'];m=Device(k);inds=np.flatnonzero(d['limited'])
        assert len(inds)>0
        i=inds[len(inds)//2];z=d['y'][i];v=d['Vs'][i];u=rng.normal(size=len(z));u[:m.no]*=.1
        if k=='Ti':u[m.no:m.n]*=1e-3
        u[m.n:]=[.3,0,0];u/=np.linalg.norm(u);h=1e-5
        fd=(m.rhs(0,z+h*u,v)-m.rhs(0,z-h*u,v))/(2*h);jj=m.jac(0,z,v)@u
        errj=float(np.linalg.norm(fd-jj)/np.linalg.norm(fd));assert errj<2e-5,(k,errj)
        validation[k]['active_CC_jacobian_relative_error']=errj
        # Oxygen conservation directly at RHS, including temperature-dependent exchange.
        rhs=m.rhs(0,z,v);th=expit(z[:m.no]);rate=float((m.ox.C*th*(1-th))@rhs[:m.no])
        if k=='Ti':rate-=float(m.core.metal.C@rhs[m.no:m.n])
        assert abs(rate)<1e-9;validation[k]['instantaneous_oxygen_balance']=rate
    plt.rcParams.update({'font.size':11,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(2,3,figsize=(15,8.5))
    for j,k in enumerate(KINDS):
        for mode in MODES:
            d=data[k,mode];mask=np.char.startswith(d['kind'],'C5-')
            ax[0,j].plot(d['Vdev'][mask],d['I'][mask]*1e6,color=colors[mode],label=mode,lw=1.8)
            ax[1,j].plot(range(1,6),summary[k][mode]['ratios_300K'],'o-',color=colors[mode],ms=4)
        ax[0,j].set(title=labels[k],xlabel='Actual device voltage (V)',ylabel='Scenario current (uA)')
        ax[0,j].axhline(300,color='#aaa',ls=':',lw=.8);ax[0,j].axhline(-300,color='#aaa',ls=':',lw=.8)
        ax[0,j].legend(frameon=False,fontsize=9)
        ax[1,j].set(xlabel='Cycle',ylabel='G after SET / G after RESET',xticks=range(1,6))
    for a in ax.flat:a.grid(alpha=.15)
    fig.suptitle('Continuous IV with circuit and self-heating | q = 2 tau | cycle 5 above',fontsize=16)
    fig.text(.5,.015,'Assumed current/thermal scales; 300 uA ideal compliance + 1 kohm. Ratios evaluated at 300 K with frozen ionic states. No lifetime prediction.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.045,1,.95])
    for ext in ('png','svg'):fig.savefig(P/f'IV_comparison.{ext}',dpi=190)
    fig,ax=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for k,color in zip(KINDS,['#234a70','#008c80','#dc7032']):
        d=data[k,'thermal'];mask=np.char.startswith(d['kind'],'C5-');t=d['t'][mask]-53.4
        ax[0].plot(t,d['T'][mask],color=color,label=labels[k])
        ax[1].plot(t,d['Vdev'][mask],color=color)
        ax[2].plot(t,d['I'][mask]*1e6,color=color)
    ax[1].plot(t,d['Vs'][mask],'k--',lw=1,label='Programmed source voltage')
    ax[0].set(ylabel='Lumped temperature (K)',title='Cycle 5 | assumed common thermal resistance and response time')
    ax[0].legend(frameon=False,ncol=3,fontsize=9);ax[1].set(ylabel='Voltage (V)');ax[1].legend(frameon=False)
    ax[2].set(xlabel='Time within cycle (tau)',ylabel='Scenario current (uA)')
    for a in ax:a.grid(alpha=.15)
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(P/f'thermal_circuit.{ext}',dpi=180)
    native=None
    for native_path in (P/'native_adaptive_comparison.json',P/'native_comparison.json'):
        if native_path.exists():
            candidate=json.loads(native_path.read_text())
            if candidate.get('all_pass'):native=candidate
    native_states=None
    if native:
        name='native_adaptive_states_check.json' if native['adaptive_output_interpolation'] else 'native_states_check.json'
        if (P/name).exists():native_states=json.loads((P/name).read_text())
    native_ti=bool(native and native.get('all_pass') and
                   native.get('native_csv_sha')==sha(P/native['native_csv']) and
                   native.get('python_npz_sha')==sha(P/'Ti-thermal-C5-R2e-09.npz') and
                   native.get('java_sha')==sha(P/native['native_java']) and
                   native.get('base_java_sha')==sha(P/'NativeThermalTi.java') and
                   native.get('unsolved_model_sha')==sha(P/'Ti-thermal-unsolved.mph') and
                   native.get('solved_model_sha')==sha(P/native['solved_model']) and
                   native_states and native_states.get('all_pass') and
                   native_states.get('native_states_sha')==sha(P/native_states['native_states_csv']) and
                   native_states.get('presentation_model_sha')==sha(P/native_states['presentation_model']) and
                   native_states.get('export_java_sha')==sha(P/('ExportThermalAdaptiveStates.java' if native['adaptive_output_interpolation'] else 'ExportThermalStates.java')) and
                   native_states.get('python_npz_sha')==sha(P/'Ti-thermal-C5-R2e-09.npz'))
    state_accuracy=json.loads((P/'python_state_accuracy.json').read_text())
    assert state_accuracy['reference_data_sha']==sha(P/'Ti-thermal-C5-R2e-10.npz')
    for tol,entry in state_accuracy['comparisons_to_rtol_2e10'].items():
        assert entry['data_sha']==sha(P/f'Ti-thermal-C5-R{tol}.npz')
    native_ti=native_ti and state_accuracy['reference_2e9_state_converged']
    result=dict(summary=summary,validation=validation,settings=asdict(Settings()),
                new_thermal_COMSOL_verified={'Ti':native_ti,'Pt':False,'Pt_contact':False},
                new_thermal_COMSOL_report=native,
                new_thermal_COMSOL_states=native_states,
                python_state_accuracy=state_accuracy,
                spatial_thermal_FEM=False,measured_temperature_prediction=False)
    (P/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
