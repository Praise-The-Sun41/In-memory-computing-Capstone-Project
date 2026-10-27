"""Instantaneous counterfactual states; not trajectories or fitted phase data."""
from interface import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    y,_=reference_state();tmp=Device();B=float(tmp.weights@y[tmp.bi]);m=Device(CASES['chemical_500'],B)
    inside=m.base.core.metal.edges[m.active+1]<=m.chem.interface_depth_nm
    rows=[]
    for delta in np.linspace(-.08,.08,81):
        z=y.copy();z[m.bi[inside]]+=delta;z[m.mi[inside]]-=m.Cb[inside]*delta/m.Cm[m.active[inside]]
        assert np.all((z[m.mi]>0)&(z[m.mi]<1)&(z[m.bi]>0)&(z[m.bi]<1))
        d=m.diagnostic(z,.5);d['delta_bound_occupancy']=float(delta);rows.append(d)
    data={k:np.array([d[k] for d in rows]) for k in rows[0]}
    assert np.ptp(data['total_Ti_oxygen'])<1e-12 and np.ptp(data['region_oxygen'])<1e-12
    np.savez_compressed(P/'fixed_oxygen_probes.npz',**data)
    fig,ax=plt.subplots(1,3,figsize=(14,4.6),layout='constrained')
    ax[0].plot(data['bound_fraction'],data['Rcontact'],color='#007f82',lw=2)
    ax[0].set(xlabel='Bound-site occupancy B',ylabel='Interface resistance (ohm)',title='A  Bonding changes resistance\n(assumed constitutive law)')
    ax[1].plot(data['bound_fraction'],data['region_oxygen_fraction'],color='#bd5328',lw=2)
    ax[1].set(xlabel='Bound-site occupancy B',ylabel='O inventory / model capacity, first 1 nm',title='B  Total local oxygen stays fixed')
    ax[1].ticklabel_format(axis='y',style='plain',useOffset=False);mid=data['region_oxygen_fraction'][40];ax[1].set_ylim(mid-.02,mid+.02)
    eta=np.linspace(0,3,100)
    ax[2].plot(eta,eta*mid,lw=2,label='First 1 nm',color='#007f82')
    ax[2].plot(eta,eta*data['mean_Ti_oxygen_fraction'][40],lw=2,ls='--',label='Entire 30 nm Ti',color='#596f89')
    ax[2].set(xlabel='Unknown oxygen capacity per Ti atom, eta',ylabel='Conditional O/Ti = eta × occupancy',title='C  Absolute oxidation is not identified')
    ax[2].legend(frameon=False)
    for a in ax:a.grid(alpha=.15);a.spines[['top','right']].set_visible(False)
    fig.suptitle('Chemical state is not the same quantity as oxygen uptake',fontsize=16)
    fig.savefig(P/'chemical_state_vs_oxidation.png',dpi=190);fig.savefig(P/'chemical_state_vs_oxidation.svg');plt.close(fig)
    (P/'probe_summary.json').write_text(json.dumps(dict(R_min_ohm=float(min(data['Rcontact'])),R_max_ohm=float(max(data['Rcontact'])),
        region_fraction=float(mid),mean_fraction=float(data['mean_Ti_oxygen_fraction'][40]),
        total_oxygen_variation=float(np.ptp(data['total_Ti_oxygen'])),
        note='Instantaneous counterfactuals at fixed total oxygen. eta is unmeasured; x=eta*theta requires all oxygen to be represented. No phase fraction inferred.'),indent=2))

if __name__=='__main__':main()
