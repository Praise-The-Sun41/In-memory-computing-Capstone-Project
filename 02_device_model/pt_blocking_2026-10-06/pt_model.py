"""Pt/Pt blocking-electrode counterfactual; frozen oxide law, no electrode state."""
from pathlib import Path
import importlib.util,sys,json,argparse,time,hashlib
from dataclasses import replace,asdict
import numpy as np
from scipy.special import expit,logit
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix
P=Path(__file__).resolve().parent;R=P.parent
sp=importlib.util.spec_from_file_location('pt_frozen_base',R/'run-bounded-history.py')
b=importlib.util.module_from_spec(sp);sys.modules[sp.name]=b;sp.loader.exec_module(b)

class PtModel:
    def __init__(self,L=256):
        base=b.make_model('ref',L,256)
        self.ox=b.f.Oxide(replace(base.ox.p,Ti_exchange_nm_per_tau=0.),base.ox.edges,'span_shift')
        self.n=self.ox.N;self.initial=base.initial[:self.n].copy()
    def q(self,y):
        # The reused oxide evaluator has one legacy scalar argument. With k=0
        # it is an inert constant; it is NOT a Pt state or an oxygen reservoir.
        return np.r_[y,logit(.3)]
    def rhs(self,t,y,V):return self.ox.evaluate(self.q(y),V)[:-1]
    def jac(self,t,y,V):return csc_matrix(self.ox.jacobian(self.q(y),V)[:-1,:-1])
    def diag(self,y,V):
        d=self.ox.evaluate(self.q(y),V,True)
        assert d['exchange']==0
        return dict(G=d['G'],I=d['J'],N=float(self.ox.C@d['theta']),exchange=0.,
            TiOx_drop=float(d['drop'][self.ox.ti].sum()),AlOx_drop=float(d['drop'][~self.ox.ti].sum()),
            power=d['electronic_power'],joule=d['electronic_dissipation'],diss=d['diss'],
            energy_residual=d['thermodynamic_residual'])

def protocol():
    r=[(10,0,-1,'prep-rise','smooth'),(10,-1,0,'prep-fall','smooth'),(.2,0,.02,'read-rise','smooth'),(1,.02,.02,'read','smooth'),(.2,.02,0,'read-fall','smooth')]
    r += [(2,a,v,f'C{cy}-{label}','linear') for cy in range(1,11) for label,a,v in [('pos-rise',0,1),('pos-fall',1,0),('neg-rise',0,-1),('neg-fall',-1,0)]]
    return r

def verify(m):
    rng=np.random.default_rng(91);u=rng.normal(size=m.n);u/=np.linalg.norm(u);err=[]
    for v in [0,.37,-.63]:
        h=1e-6;fd=(m.rhs(0,m.initial+h*u,v)-m.rhs(0,m.initial-h*u,v))/(2*h);jj=m.jac(0,m.initial,v)@u
        err.append(float(np.linalg.norm(fd-jj)/max(np.linalg.norm(jj),1e-30)))
        assert abs((m.ox.C*expit(m.initial)*(1-expit(m.initial)))@m.rhs(0,m.initial,v))<1e-10
        d=m.ox.evaluate(m.q(m.initial),v,True)
        # Inert legacy argument cannot influence electrical or kinetic outputs.
        q=m.q(m.initial);q[-1]=logit(.8)
        assert np.array_equal(m.ox.evaluate(q,v)[:-1],m.rhs(0,m.initial,v))
        assert d['exchange']==0
    assert max(err)<1e-5
    return dict(jacobian_relative=max(err),electrode_state_count=0,zero_exchange=True)

def solve(m,y,rows,name,rtol=1e-9):
    file=P/(name+'.npz');meta=P/(name+'.json')
    if file.exists():
        old=json.loads(meta.read_text());assert old['script_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return dict(np.load(file))
    t0=0.;parts=[];audits=[];tic=time.perf_counter();initialN=m.ox.C@expit(y)
    for T,a,v,label,shape in rows:
        vf=lambda t,a=a,v=v,T=T,shape=shape:a+(v-a)*((t/T)**2*(3-2*t/T) if shape=='smooth' else t/T)
        sol=solve_ivp(lambda t,z:m.rhs(t,z,vf(t)),(0,T),y,method='BDF',jac=lambda t,z:m.jac(t,z,vf(t)),rtol=rtol,atol=.01*rtol,dense_output=True)
        if not sol.success:raise RuntimeError(sol.message)
        th=expit(sol.y);mass=float(abs(m.ox.C@th-initialN).max());assert mass<1e-6 and np.all((th>0)&(th<1))
        tt=np.r_[0,np.geomspace(.001,T,120)] if T>100 else np.linspace(0,T,121)
        yy=sol.sol(tt).T;sl=slice(None) if not parts else slice(1,None)
        parts.append(dict(t=tt[sl]+t0,y=yy[sl],V=vf(tt[sl]),kind=np.full(len(tt[sl]),label)))
        y=sol.y[:,-1];t0+=T;audits.append(dict(label=label,mass_error=mass,steps=len(sol.t)-1))
        print(name,label,round(time.perf_counter()-tic,1),flush=True)
    out={k:np.concatenate([x[k] for x in parts]) for k in parts[0]}
    ds=[m.diag(y,v) for y,v in zip(out['y'],out['V'])]
    for k in ds[0]:out[k]=np.asarray([d[k] for d in ds])
    out['edges']=m.ox.edges
    assert out['diss'].min()>-1e-9 and abs(out['energy_residual']).max()<1e-9
    np.savez_compressed(file,**out)
    meta.write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),oxide_parameters=asdict(m.ox.p),rows=rows,audits=audits,electrode_oxygen_states=0,physical_clock_calibrated=False,current_calibrated=False,contact_barriers_recalibrated=False),indent=2))
    return out

def run():
    m=PtModel();checks=verify(m);d=solve(m,m.initial,protocol(),'pt-history')
    for sign in ['pos','neg']:
        idx=np.flatnonzero(d['kind']==f'C5-{sign}-fall')[-1]
        solve(m,d['y'][idx].copy(),[(10,0,0,'hold10','linear'),(90,0,0,'hold100','linear'),(900,0,0,'hold1000','linear')],f'pt-hold-{sign}')
    (P/'unit-checks.json').write_text(json.dumps(checks,indent=2))

def java():
    q=json.dumps;fmt=lambda x:format(float(x),'.17g');arr=lambda v:'new String[]{'+','.join(q(str(x)) for x in v)+'}'
    times=[];terms=[];clock=0.
    for T,a,v,label,shape in protocol():
        arg=f'(tau-{fmt(clock)})/{fmt(T)}';fn=f'sstep({arg})' if shape=='smooth' else f'min(1,max(0,{arg}))'
        if a!=v:terms.append(f'({fmt(v-a)})*{fn}')
        t=np.linspace(0,T,121)+clock;times.extend(t if not times else t[1:]);clock+=T
    chunks=[' '.join(fmt(x) for x in times[i:i+400]) for i in range(0,len(times),400)]
    geom=R/'presentation_geometry_2026-10-04/Memristor_Geometry_Bound.mph';fresh=R/'outputs/bounded-comsol-strict-full-r6-unsolved.mph'
    lines=['import com.comsol.model.*;','import com.comsol.model.util.*;','public class PreparePt {public static Model run(){try{',f'Model src=ModelUtil.load("fresh",{q(fresh.as_posix())});',f'Model m=ModelUtil.load("pt",{q(geom.as_posix())});',
      'for(String tag:m.result().numerical().tags())m.result().numerical().remove(tag);',
      'for(String tag:m.result().table().tags())m.result().table().remove(tag);',
      'm.result().remove("oxygenview");',
      'for(String key:new String[]{"name","equation","initialValueU","initialValueUt"}){String[] old=src.component("c").physics("ge").feature("ge1").getStringArray(key); if(old.length!=1137)throw new IllegalStateException(key+" state count "+old.length);String[] val=new String[881];int j=0;for(int i=0;i<old.length;i++)if(i<440||i>=696)val[j++]=old[i];m.component("c").physics("ge").feature("ge1").set(key,val);}',
      'm.component("c").variable("v").set("exchange","0");m.component("c").variable("v").set("ra","0");m.component("c").variable("v").set("rb","0");',
      'm.component("c").variable("v").set("metalN","0");m.component("c").variable("v").set("inventory","oxideN");',
      'for(int i=0;i<255;i++)m.component("c").variable("v").remove("mflux"+i);',
      'm.component("c").variable("v").set("bias",'+q('+'.join(terms))+');',
      'm.component("c").label("Pt blocking boundaries - oxide states only");m.label("Pt TiOx AlOx Pt - blocking electrodes - uncalibrated");',
      'm.component("show").geom("gs").feature("be").label("Pt bottom - 30 nm");m.component("show").geom("gs").feature("te").label("Pt top - 30 nm");',
      'm.component("layout").geom("gl").feature("be").label("Pt bottom - 30 nm");m.component("layout").geom("gl").feature("te").label("Pt top - 30 nm");',
      'm.result("stackview").set("title","Pt / TiOx / AlOx / Pt | local 40 nm crop");',
      'm.result("stackview").feature("material0").set("text","Pt | 30 nm");m.result("stackview").feature("material3").set("text","Pt | 30 nm");',
      'm.result().param().set("show_tau","57.4","Absolute normalized time including 21.4 tau preparation; C5 positive/negative endpoints 57.4/61.4");',
      'm.study("std").label("Pt own preparation and ten continuous cycles");']
    for target in ['m.study("std").feature("time")','m.sol("sol1").feature("t1")']:lines.append(target+'.set("tlist",String.join(" ",'+arr(chunks)+'));')
    lines += ['m.sol("sol1").feature("t1").set("bwinitstepfrac",1e-6);','m.sol("sol1").feature("t1").set("maxstepconstraintbdf","const");m.sol("sol1").feature("t1").set("maxstepbdf","0.05[s]");',
      'for(String tag:m.sol("sol1").feature("t1").feature().tags())if(m.sol("sol1").feature("t1").feature(tag).getType().equals("Direct"))m.sol("sol1").feature("t1").feature(tag).set("linsolver","pardiso");',
      'm.sol("sol1").clearSolutionData();',f'm.save({q((P/"Pt-unsolved.mph").as_posix())});','System.out.println("PREPARED_PT_881_NO_OXYGEN_RESERVOIR");return m;}catch(Exception e){throw new RuntimeException(e);}}public static void main(String[] args){run();}}']
    (P/'PreparePt.java').write_text('\n'.join(lines))
    expr=['t','c.bias','c.conductance','c.current','c.TiDrop','c.AlDrop','c.epower','c.joule','c.inventory','c.exchange']+[f'c.th{i}' for i in range(440)]
    code=['import com.comsol.model.*;','import com.comsol.model.util.*;','public class ExportPt {public static Model run(){try{',f'Model m=ModelUtil.load("pt",{q((P/"Pt-solved.mph").as_posix())});',
      'm.result().numerical().create("pteval","EvalGlobal");m.result().numerical("pteval").set("data","protocoldata");',
      'm.result().numerical("pteval").set("expr",'+arr(expr)+');','double[][] d=m.result().numerical("pteval").getReal();if(d.length!=450||d[0].length!=5401)throw new IllegalStateException("Wrong export shape "+d.length+" "+d[0].length);',
      'double[][] rows=new double[d[0].length][d.length];for(int i=0;i<rows.length;i++)for(int j=0;j<d.length;j++){if(!Double.isFinite(d[j][i]))throw new IllegalStateException("Nonfinite");rows[i][j]=d[j][i];}',
      'm.result().table().create("pttable","Table");m.result().table("pttable").setTableData(rows);',f'm.result().table("pttable").save({q((P/"pt-native.csv").as_posix())});',
      'm.result("protocolplot").set("looplevelinput",new String[]{"manualindices"});m.result("protocolplot").set("looplevelindices",new String[]{"range(601,1,5401)"});',
      'm.result("protocolplot").feature("g").set("expr",new String[]{"c.current/"+Double.toString(rows[600][2])});',
      'm.result("protocolplot").set("title","Pt / TiOx / AlOx / Pt | ten cycles | q=2 tau | no oxygen exchange");',
      'm.result("protocolplot").label("01 | Pt native ten-cycle IV");',
      'for(String tag:new String[]{"stackview","vacancyview","potentialview","protocolplot"}){m.result(tag).run();m.result().export("showimage").set("plotgroup",tag);',f'm.result().export("showimage").set("pngfilename",{q(P.as_posix()+"/")}+tag+".png");m.result().export("showimage").run();}}',
      f'm.save({q((P/"Pt_TiOx_AlOx_Pt_Geometry_Bound.mph").as_posix())});','System.out.println("EXPORTED_PT_NATIVE_AND_GEOMETRY");return m;}catch(Exception e){throw new RuntimeException(e);}}public static void main(String[] args){run();}}']
    (P/'ExportPt.java').write_text('\n'.join(code))
    (P/'plan.json').write_text(json.dumps(dict(layers_bottom_to_top_nm={'Pt_bottom':30,'TiOx':20,'AlOx':5,'Pt_top':30},times=times,no_electrode_oxygen_states=True,base_files={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [geom,fresh,R/'bounded-span-field.py']},native_unknowns=881,python_states=440),indent=2))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['run','java']);args=a.parse_args();globals()[args.action]()
