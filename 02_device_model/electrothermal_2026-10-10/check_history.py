"""Temperature/current feedback checks and continuous SET-hold-RESET-hold."""
from coupled import *

def main():
    results={}
    for kind in ('Ti','Pt','Pt_contact'):
        m=Device(kind);source=P/f'{kind}-thermal-C5-R2e-08.npz';old=dict(np.load(source))
        rows=[]
        for label,amp in [('SET',1.),('RESET',-1.)]:
            rows += [(2,0,amp,label+'-rise','linear'),(2,amp,0,label+'-fall','linear'),
                     (10,0,0,label+'-hold','linear'),(.2,0,.02,label+'-read-rise','linear'),
                     (1,.02,.02,label+'-read','linear'),(.2,.02,0,label+'-read-fall','linear')]
        d,meta=solve(m,rows,old['y'][-1].copy(),kind+'-thermal-write-hold',2e-8)
        meta['source_history_sha']=sha(source);meta['source_history']=source.name
        (P/(kind+'-thermal-write-hold.json')).write_text(json.dumps(meta,indent=2))
        endtemps=[]
        for stage in ('SET','RESET'):
            ix=np.flatnonzero(d['kind']==stage+'-hold')[-1];endtemps.append(float(d['T'][ix]))
        assert max(abs(np.array(endtemps)-300))<1e-5
        results[kind]=dict(hold_end_temperature_K=endtemps,continuous_state=True)
    # Freeze an actual device's ionic state and remove electronic thermal activation.
    # This gives constant Joule power through the real circuit routine, allowing
    # a direct analytic comparison of the implemented thermal RHS (not a copy).
    m=Device('Ti');m.ox.Ea_e[:]=0.;m.pref=replace(m.pref,vacancy_transport=False)
    y=m.initial.copy();power=m.diagnostic(y,.8)['Pdev'];duration=.3
    def rhs(t,z,v):
        yy=y.copy();yy[m.n:]=z;return m.rhs(t,yy,v)[m.n:]
    sol=solve_ivp(lambda t,z:rhs(t,z,.8),(0,duration),y[m.n:],rtol=1e-10,atol=1e-12)
    exact=300+power*m.cfg.thermal_R_K_W*(1-np.exp(-duration/m.cfg.thermal_tau))
    heat_error=abs(sol.y[0,-1]-exact)
    cool=solve_ivp(lambda t,z:rhs(t,z,0.),(0,duration),sol.y[:,-1],rtol=1e-10,atol=1e-12)
    exactcool=300+(sol.y[0,-1]-300)*np.exp(-duration/m.cfg.thermal_tau)
    cool_error=abs(cool.y[0,-1]-exactcool)
    assert max(heat_error,cool_error)<1e-5
    results['frozen_device_analytic']=dict(heat_error_K=float(heat_error),cool_error_K=float(cool_error))
    a=dict(np.load(P/'Ti-thermal-C5-R2e-08.npz'));b=dict(np.load(P/'Ti-thermal-C5-R2e-09.npz'))
    assert np.array_equal(a['t'],b['t'])
    current=float(np.max(abs(a['I']-b['I']))/np.max(abs(b['I'])))
    temperature=float(np.max(abs(a['T']-b['T'])));voltage=float(np.max(abs(a['Vdev']-b['Vdev'])))
    assert current<.002 and temperature<.05 and voltage<.002
    results['Ti_time_tolerance']=dict(current_peak_normalized_error=current,temperature_error_K=temperature,device_voltage_error_V=voltage)
    results['all_pass']=True
    (P/'history_checks.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))

if __name__=='__main__':main()
