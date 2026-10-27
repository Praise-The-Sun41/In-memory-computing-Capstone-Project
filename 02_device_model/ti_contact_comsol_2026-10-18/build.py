"""Generate native COMSOL chemistry/contact equations from frozen parameters.

Only one C0 state is transferred. No Python trajectory drives native dynamics.
Geometry is retained as an extrusion of the native finite-volume states.
"""
from pathlib import Path
import sys, json, hashlib, argparse
import numpy as np
from scipy.special import expit

P=Path(__file__).resolve().parent
SRC=P.parent/'ti_contact_oxidation_2026-10-18'
sys.path.insert(0,str(SRC))
import interface as py
q=json.dumps;f=lambda x:format(float(x),'.17g')
arr=lambda xs:'new String[]{'+','.join(q(str(x)) for x in xs)+'}'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def build(case,smoke=False):
    y,origin=py.reference_state();tmp=py.Device();B=float(tmp.weights@y[tmp.bi])
    m=py.Device(py.CASES[case],B);d=m.base;o=d.ox;n=m.no;nm=len(m.Cm);nb=m.nb
    tag=case+('-smoke' if smoke else '-full');cls='Chem'+case.split('_')[-1]+('Smoke' if smoke else 'Full')
    dest=P/tag;dest.mkdir(exist_ok=True)
    template=P.parent/'electrothermal_2026-10-10/Ti-thermal-unsolved.mph'
    rows=py.et.protocol(1 if smoke else 5)[5:]
    if not smoke:
        for name,V in [('SET',1.),('RESET',-1.)]:
            rows += [(2,0,V,name+'-rise','linear'),(2,V,0,name+'-fall','linear'),
                (100,0,0,name+'-hold','linear'),(.2,0,.02,name+'-read-rise','linear'),
                (1,.02,.02,name+'-read','linear'),(.2,.02,0,name+'-read-fall','linear')]
    setup=[]
    def add(name,expr):setup.append(f'm.component("c").variable("v").set({q(name)},{q(expr)});')
    setup += [f'm.label({q(tag+": native coupled chemistry/contact; uncalibrated")});',
        f'm.modelPath({q(dest.as_posix())});',
        'm.component("c").label("Native FV chemistry, electronic circuit and lumped heat");',
        'm.study("std").label("Independent native chemical/contact history");']
    # Retain relevant geometry plots; remove stale terminal plots/tables.
    setup += ['for(String s:m.result().tags())if(!java.util.Arrays.asList("stackview","vacancyview","oxygenview","potentialview","sigmaview","crossbarview").contains(s))m.result().remove(s);',
        'for(String s:m.result().numerical().tags())m.result().numerical().remove(s);',
        'for(String s:m.result().table().tags())m.result().table().remove(s);']
    add('Bbound','+'.join(f'{f(w)}*bnd{k}' for k,w in enumerate(m.weights)))
    add('ifaceR',f'{f(m.contact.reference_ohm)}*exp({f(m.contact.chemical_sensitivity_eV)}*(Bbound-{f(B)})/kt)')
    add('ifaceV','physicalI*ifaceR');add('devV','bias+ifaceV')
    add('outV',f'devV+physicalI*{f(d.Rs)}');add('heatP','physicalI*devV')
    add('bulkP','physicalI*bias');add('contactP','physicalI*ifaceV')
    add('seriesP',f'physicalI^2*{f(d.Rs)}');add('deviceG',f'1/(resistance/{f(d.scale)}+ifaceR)')
    add('powerResidual','physicalI*outV-heatP-seriesP')
    add('devicePowerResidual','heatP-bulkP-contactP')
    add('voltageResidual','outV-bias-ifaceV-physicalI*'+f(d.Rs))
    add('boundO','+'.join(f'{f(v)}*bnd{k}' for k,v in enumerate(m.Cb)))
    add('mobileO','+'.join(f'{f(v)}*s{i}' for i,v in enumerate(m.Cm)))
    add('totalTiO','mobileO+boundO')
    add('regionO','boundO+'+'+'.join(f'{f(m.region_mobile[i])}*s{i}' for i in np.flatnonzero(m.region_mobile)))
    add('regionFraction','regionO/'+f(m.region_capacity))
    add('inventory','+'.join(f'{f(v)}*th{i}' for i,v in enumerate(o.C))+'-totalTiO')
    add('bindingNu',f'{f(m.chem.capture_rate_per_tau)}*exp({f(m.chem.capture_Ea_eV/py.KB)}*(1/300-1/tempK))')
    add('bindingReverse',f'exp({f(m.chem.binding_delta_g_eV)}/kt)')
    add('ionicPass',f'exp(-{f(m.chem.passivation_barrier_eV)}*Bbound/kt)')
    for k,i in enumerate(m.active):
        add(f'bind{k}',f'{f(m.Cb[k])}*bindingNu*(s{i}*(1-bnd{k})-bindingReverse*bnd{k}*(1-s{i}))')
    add('bindingFlux','+'.join(f'bind{k}' for k in range(nb)))
    k=o.p.Ti_exchange_nm_per_tau*o.S[0];dg=o.G[0]+o.p.Ti_reaction_standard_eV
    add('ra',f'{f(k)}*exmult*ionicPass*exp(-({f(dg)}+2*ifaceV)/(2*kt))*(1-s0)')
    add('rb',f'{f(k)}*exmult*ionicPass*exp(({f(dg)}+2*ifaceV)/(2*kt))*s0')
    for i,K in enumerate(d.core.metal.K):add(f'mflux{i}',f'{f(K)}*metalmult*(s{i}-s{i+1})')
    for i in range(n):add(f'phiabs{i}',f'phi{i}+ifaceV')
    terms=[];times=[];clock=0.
    for dur,a,b,label,shape in rows:
        u=f'(tau-{f(clock)})/{f(dur)}';r=f'sstep({u})' if shape=='smooth' else f'min(1,max(0,{u}))'
        if a!=b:terms.append(f'({f(b-a)})*{r}')
        tt=np.linspace(0,dur,121)+clock;times.extend(tt if not times else tt[1:]);clock+=dur
    add('source','+'.join(terms))
    names=[f'th{i}' for i in range(n)]+[f's{i}' for i in range(nm)]+[f'av{i}' for i in range(n)]+['ci','tempK','qin','qout']+[f'bnd{k}' for k in range(nb)]
    eq=[]
    for i in range(n):
        left='exchange' if i==0 else f'flux{i-1}';right='0' if i==n-1 else f'flux{i}'
        eq.append(f'th{i}t-({left}-{right})/{f(o.C[i])}/1[s]')
    active={int(v):k for k,v in enumerate(m.active)}
    for i in range(nm):
        left='exchange' if i==0 else f'mflux{i-1}';right='0' if i==nm-1 else f'mflux{i}'
        bind=f'-bind{active[i]}' if i in active else ''
        eq.append(f's{i}t-({left}-{right}{bind})/{f(m.Cm[i])}/1[s]')
    for i in range(n):eq.append(f'av{i}-('+ '+'.join(f'{f(o.W[i,j])}*cv{j}' for j in np.flatnonzero(o.W[i]))+')')
    eq += [f'(ci*1e-8-min({f(d.Icc/d.scale)},max(-{f(d.Icc/d.scale)},source/(resistance+{f(d.scale)}*({f(d.Rs)}+ifaceR)))))/1e-8',
        f'tempKt-(heatP-lossP)*{f(d.cfg.thermal_R_K_W/d.cfg.thermal_tau)}/1[s]','qint-heatP/1[s]','qoutt-lossP/1[s]']
    eq += [f'bnd{k}t-bind{k}/{f(m.Cb[k])}/1[s]' for k in range(nb)]
    initial=np.r_[expit(y[:n]),y[n:m.n],o.W@(o.S*expit(y[:n])+o.bound),0.,y[m.n:m.n+3],y[m.bi]]
    assert len(initial)==len(names)==len(eq)
    # Split large arrays into separate assignments to stay below Java method limits.
    setup += [f'String[] names=new String[{len(names)}],eq=new String[{len(names)}],ini=new String[{len(names)}],dot=new String[{len(names)}];']
    # Arrays are stored in class fields, initialized once in the run method below.
    setup.pop()
    for i,(name,e,value) in enumerate(zip(names,eq,initial)):
        setup.append(f'names[{i}]={q(name)};eq[{i}]={q(e)};ini[{i}]={q(f(value))};dot[{i}]="0";')
    for prop,val in [('name','names'),('equation','eq'),('initialValueU','ini'),('initialValueUt','dot')]:
        setup.append(f'm.component("c").physics("ge").feature("ge1").set({q(prop)},{val});')
    chunks=[' '.join(map(f,times[i:i+300])) for i in range(0,len(times),300)]
    for target in ['m.study("std").feature("time")','m.sol("sol1").feature("t1")']:
        setup.append(target+'.set("tlist",String.join(" ",'+arr(chunks)+'));')
    setup += ['m.sol("sol1").feature("t1").set("rtol",2e-8);',
        'm.sol("sol1").feature("t1").set("atolglobalvaluemethod","manual");',
        'm.sol("sol1").feature("t1").set("atolglobalmethod","unscaled");',
        'm.sol("sol1").feature("t1").set("atolglobal",1e-13);',
        'm.sol("sol1").feature("t1").set("tstepsbdf","strict");',
        'm.sol("sol1").feature("t1").set("maxstepconstraintbdf","const");',
        'm.sol("sol1").feature("t1").set("maxstepbdf","1[s]");',
        'for(String s:m.sol("sol1").feature("t1").feature().tags()){String type=m.sol("sol1").feature("t1").feature(s).getType();if(type.equals("FullyCoupled")){m.sol("sol1").feature("t1").feature(s).set("ntolfact",0.01);m.sol("sol1").feature("t1").feature(s).set("maxiter",12);}if(type.equals("Direct"))m.sol("sol1").feature("t1").feature(s).set("linsolver","pardiso");}',
        'm.result().param().set("show_tau","36","DISPLAY ONLY, relative to shared C0; C5 positive end=36, negative end=40");',
        'm.result("oxygenview").label("Ti MOBILE oxygen occupancy - not total oxidation");',
        'String pv=m.result("potentialview").feature("potential").getString("expr");pv=pv.replace("c.bias,","c.devV,");for(int j=439;j>=0;j--)pv=pv.replace("c.phi"+j+",","c.phiabs"+j+",");m.result("potentialview").feature("potential").set("expr",pv);']
    ports=['t','c.source','c.physicalI','c.bias','c.ifaceV','c.devV','c.outV','c.tempK','c.TiDrop','c.AlDrop','c.ifaceR','c.Bbound','c.boundO','c.mobileO','c.totalTiO','c.regionFraction','c.inventory','c.heatResidual','c.powerResidual','c.devicePowerResidual','c.voltageResidual','c.exchange','c.bindingFlux','c.ionicPass','c.deviceG']
    states=['t']+[f'c.th{i}' for i in range(n)]+[f'c.s{i}' for i in range(nm)]+[f'c.bnd{k}' for k in range(nb)]
    head=['import com.comsol.model.*;','import com.comsol.model.util.*;',f'public class {cls} {{',
        f'static String[] names=new String[{len(names)}],eq=new String[{len(names)}],ini=new String[{len(names)}],dot=new String[{len(names)}];',
        'public static Model run(){try{',f'Model m=ModelUtil.load("chem",{q(template.as_posix())});']
    helpers=[]
    for j,start in enumerate(range(0,len(setup),70)):
        head.append(f'setup{j}(m);');helpers.append(f'static void setup{j}(Model m){{'+'\n'.join(setup[start:start+70])+'}')
    head += ['m.sol("sol1").clearSolutionData();',f'm.save({q((dest/"unsolved.mph").as_posix())});',
        'System.out.println("CHEMISTRY_NATIVE_START");','m.sol("sol1").runAll();',
        f'export(m,"chemports",{arr(ports)},{q((dest/"ports.csv").as_posix())},{len(times)});',
        f'export(m,"chemstates",{arr(states)},{q((dest/"states.csv").as_posix())},{len(times)});',
        'm.result().dataset().create("chemnative","Solution");m.result().dataset("chemnative").set("solution","sol1");',
        'm.result().create("chemiv","PlotGroup1D");m.result("chemiv").set("data","chemnative");m.result("chemiv").create("g","Global");m.result("chemiv").feature("g").set("expr",new String[]{"1e6*c.physicalI"});m.result("chemiv").feature("g").set("xdata","expr");m.result("chemiv").feature("g").set("xdataexpr","c.source");m.result("chemiv").label("Native chemical-contact IV, q=2 tau, uncalibrated uA");',
        'm.result("chemiv").set("looplevelinput",new String[]{"manualindices"});',f'm.result("chemiv").set("looplevelindices",new String[]{{"range(1,1,{481 if smoke else 2401})"}});',
        'm.result().create("chemhistory","PlotGroup1D");m.result("chemhistory").set("data","chemnative");m.result("chemhistory").create("g","Global");m.result("chemhistory").feature("g").set("expr",new String[]{"c.Bbound","c.regionFraction"});m.result("chemhistory").label("Bound occupancy and local total O loading - NOT O/Ti");',
        f'm.save({q((dest/"Ti-chemistry-contact.mph").as_posix())});',
        'System.out.println("CHEMISTRY_NATIVE_COMPLETE");return m;}catch(Exception e){throw new RuntimeException(e);}}',
        'static void export(Model m,String tag,String[] expressions,String path,int points){m.result().numerical().create(tag,"EvalGlobal");m.result().numerical(tag).set("expr",expressions);double[][] a=m.result().numerical(tag).getReal();if(a[0].length!=points)throw new IllegalStateException("Incomplete native history");double[][] rows=new double[points][a.length];for(int i=0;i<points;i++)for(int j=0;j<a.length;j++)rows[i][j]=a[j][i];m.result().table().create(tag,"Table");m.result().table(tag).setTableData(rows);m.result().table(tag).save(path);}',
        *helpers,'public static void main(String[] args){run();}}']
    java=dest/(cls+'.java');java.write_text('\n'.join(head),encoding='utf-8')
    meta=dict(case=case,smoke=smoke,class_name=cls,rows=rows,points=len(times),ports=ports,
        n_oxide=n,n_metal=nm,n_bound=nb,unknowns=len(names),origin=origin,contact=py.asdict(m.contact),
        chemistry=py.asdict(m.chem),thermal=py.asdict(d.cfg),reference_B=B,
        solver=dict(rtol=2e-8,atol=1e-13,newton_tolerance_factor=.01,strict_output_steps=True,max_step=1.),
        java_sha=sha(java),generator_sha=sha(__file__),template_sha=sha(template),
        interface_source_sha=sha(SRC/'interface.py'),chemistry_source_sha=sha(py.OLD/'chemistry.py'),
        thermal_source_sha=sha(py.ch.BASE/'coupled.py'),initial_sha=hashlib.sha256(y.tobytes()).hexdigest(),
        imported_initial_state=True,imported_trajectory=False,spatial_3D_FEM=False)
    (dest/'manifest.json').write_text(json.dumps(meta,indent=2));print(cls,len(names),len(times),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--case',choices=['chemical_500','chemical_5000'],required=True);a.add_argument('--smoke',action='store_true');v=a.parse_args();build(v.case,v.smoke)
