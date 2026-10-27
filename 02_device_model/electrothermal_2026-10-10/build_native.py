"""Independent native COMSOL Ti thermal/circuit equations; no trajectory import."""
from coupled import *
q=json.dumps;fmt=lambda x:format(float(x),'.17g')
arr=lambda xs:'new String[]{'+','.join(q(str(x)) for x in xs)+'}'
m=Device('Ti');o=m.ox;p=o.p;c=m.cfg;n=m.no;nm=m.core.nm
template=R/'presentation_geometry_2026-10-04/Memristor_Geometry_Bound.mph'
code=['import com.comsol.model.*;','import com.comsol.model.util.*;',
      'public class NativeThermalTi {public static Model run(){try{',
      f'Model m=ModelUtil.load("et",{q(template.as_posix())});',
      'm.label("Ti electrothermal circuit scenario; lumped temperature, uncalibrated");']
def add(k,v):code.append(f'm.component("c").variable("v").set({q(k)},{q(v)});')
def fun(tag,args,expr):
    code.extend([f'm.func().create("{tag}","Analytic");',f'm.func("{tag}").set("args",{q(args)});',f'm.func("{tag}").set("expr",{q(expr)});'])
fun('etboost','g,b','if(g<1e-5,1+g^2/6,exp(min(g,b))*(1-exp(-2*g))/(2*g))')
add('kt',f'{fmt(KB)}*tempK');add('cap',f'{fmt(p.hop_barrier_eV)}/kt')
for name,E in [('diffmult',c.oxide_migration_Ea_eV),('metalmult',c.metal_migration_Ea_eV),('exmult',c.exchange_Ea_eV)]:
    add(name,f'exp({fmt(E/KB)}*(1/300-1/tempK))')
terms=[];times=[];clock=0.
for duration,a,b,label,shape in protocol():
    arg=f'(tau-{fmt(clock)})/{fmt(duration)}';fn=f'sstep({arg})' if shape=='smooth' else f'min(1,max(0,{arg}))'
    if a!=b:terms.append(f'({fmt(b-a)})*{fn}')
    tt=np.linspace(0,duration,121)+clock;times.extend(tt if not times else tt[1:]);clock+=duration
add('source','+'.join(terms));add('bias','current*resistance')
add('physicalI',f'current*{fmt(m.scale)}');add('outV',f'bias+physicalI*{fmt(m.Rs)}')
add('heatP','physicalI*bias');add('lossP',f'(tempK-300)/{fmt(c.thermal_R_K_W)}')
add('heatResidual',f'{fmt(c.thermal_tau/c.thermal_R_K_W)}*(tempK-300)-qin+qout')
for i in range(n):add(f'sig{i}',f'1e-12+av{i}*exp(-2*(50*av{i})^(-1/3)/{fmt(o.xi[i])}-{fmt(o.Ea_e[i])}/kt)')
for i in range(n-1):
    dx=o.x[i+1]-o.x[i]
    add(f'gl{i}',f'abs(0.5*(dv{i}+dv{i+1}))*{fmt(p.lattice_hop_nm/dx)}/kt')
    add(f'gs{i}',f'abs(work{i})/kt')
    add(f'boost{i}',f'if(gl{i}<1e-20,etboost(gs{i},cap),min(etboost(gs{i},cap),exp(cap)/(2*gl{i})))')
    add(f'del{i}',f'({fmt(o.G[i+1]-o.G[i])}+dv{i}+dv{i+1})/kt')
    D=2*o.D[i]*o.D[i+1]/(o.D[i]+o.D[i+1])
    add(f'flux{i}',f'{fmt(D*np.sqrt(o.S[i]*o.S[i+1])/dx)}*diffmult*boost{i}*(bern(del{i})*th{i}*(1-th{i+1})-bern(-del{i})*th{i+1}*(1-th{i}))')
k=p.Ti_exchange_nm_per_tau*o.S[0];dg=o.G[0]+p.Ti_reaction_standard_eV
add('ra',f'{fmt(k)}*exmult*exp(-{fmt(dg)}/(2*kt))*(1-s0)')
add('rb',f'{fmt(k)}*exmult*exp({fmt(dg)}/(2*kt))*s0')
add('rc',f'{fmt(2*o.D[0]*o.S[0]/o.h[0])}*diffmult*bern(2*phi0/kt)*(1-th0)')
add('rd',f'{fmt(2*o.D[0]*o.S[0]/o.h[0])}*diffmult*bern(-2*phi0/kt)*th0')
for i in range(nm-1):add(f'mflux{i}',f'{fmt(m.core.metal.K[i])}*metalmult*(s{i}-s{i+1})')
# The geometry-bound strict native template uses direct th states and defines
# z=logit(th) as diagnostic aliases. Preserve those native state coordinates.
names=[f'th{i}' for i in range(n)]+[f's{i}' for i in range(nm)]+[f'av{i}' for i in range(n)]+['ci','tempK','qin','qout']
eq=[]
for i in range(n):
    left='exchange' if i==0 else f'flux{i-1}';right='0' if i==n-1 else f'flux{i}'
    eq.append(f'th{i}t-({left}-{right})/{fmt(o.C[i])}/1[s]')
for i in range(nm):
    left='exchange' if i==0 else f'mflux{i-1}';right='0' if i==nm-1 else f'mflux{i}'
    eq.append(f's{i}t-({left}-{right})/{fmt(m.core.metal.C[i])}/1[s]')
for i in range(n):eq.append(f'av{i}-('+ '+'.join(f'{fmt(o.W[i,j])}*cv{j}' for j in np.flatnonzero(o.W[i])) + ')')
eq += [f'(ci*1e-8-min({fmt(m.Icc/m.scale)},max(-{fmt(m.Icc/m.scale)},source/(resistance+{fmt(m.scale*m.Rs)}))))/1e-8',
       f'tempKt-(heatP-lossP)*{fmt(c.thermal_R_K_W/c.thermal_tau)}/1[s]','qint-heatP/1[s]','qoutt-lossP/1[s]']
initial=np.r_[expit(m.core.initial[:n]),m.core.initial[n:],o.W@o.initial_defects,0.,300.,0.,0.]
for key,value in [('name',names),('equation',eq),('initialValueU',map(fmt,initial)),('initialValueUt',['0']*len(names))]:
    code.append(f'm.component("c").physics("ge").feature("ge1").set({q(key)},{arr(value)});')
timestring=' '.join(map(fmt,times));chunks=[timestring[i:i+12000] for i in range(0,len(timestring),12000)]
# Split on tokens so joining chunks cannot corrupt a numeric time.
tokens=list(map(fmt,times));chunks=[' '.join(tokens[i:i+350]) for i in range(0,len(tokens),350)]
for target in ['m.study("std").feature("time")','m.sol("sol1").feature("t1")']:
    code.append(target+'.set("tlist",String.join(" ",'+arr(chunks)+'));')
code += ['m.sol("sol1").feature("t1").set("rtol",2e-8);',
         'm.sol("sol1").feature("t1").set("maxstepconstraintbdf","const");m.sol("sol1").feature("t1").set("maxstepbdf","0.05[s]");',
         'for(String tag:m.sol("sol1").feature("t1").feature().tags())if(m.sol("sol1").feature("t1").feature(tag).getType().equals("Direct"))m.sol("sol1").feature("t1").feature(tag).set("linsolver","pardiso");',
         'm.sol("sol1").clearSolutionData();',f'm.save({q((P/"Ti-thermal-unsolved.mph").as_posix())});',
         'm.sol("sol1").runAll();']
expressions=['t','c.source','c.bias','c.physicalI','c.tempK','c.inventory','c.heatResidual','c.TiDrop','c.AlDrop','c.outV']
code += ['m.result().numerical().create("etout","EvalGlobal");',f'm.result().numerical("etout").set("expr",{arr(expressions)});',
         'double[][] d=m.result().numerical("etout").getReal();',f'if(d[0].length!={len(times)})throw new IllegalStateException("Incomplete time history");',
         f'double[][] out=new double[{len(times)}][{len(expressions)}];for(int i=0;i<out.length;i++)for(int j=0;j<out[0].length;j++)out[i][j]=d[j][i];',
         'm.result().table().create("ettable","Table");m.result().table("ettable").setTableData(out);',
         f'm.result().table("ettable").save({q((P/"Ti-thermal-native.csv").as_posix())});',
         'm.result().create("etplot","PlotGroup1D");m.result("etplot").create("g","Global");m.result("etplot").feature("g").set("expr",new String[]{"c.physicalI"});m.result("etplot").feature("g").set("xdata","expr");m.result("etplot").feature("g").set("xdataexpr","c.bias");m.result("etplot").label("Thermal circuit IV - assumed scales");',
         'm.result().create("ettemp","PlotGroup1D");m.result("ettemp").create("g","Global");m.result("ettemp").feature("g").set("expr",new String[]{"c.tempK"});m.result("ettemp").label("Lumped temperature - not spatial heat FEM");',
         f'm.save({q((P/"Ti-thermal-geometry.mph").as_posix())});',
         'System.out.println("THERMAL_NATIVE_COMPLETE");return m;}catch(Exception e){throw new RuntimeException(e);}}public static void main(String[] args){run();}}']
cut=code.index('m.sol("sol1").clearSolutionData();')
setup=code[4:cut];helpers=[];calls=[]
for j,start in enumerate(range(0,len(setup),100)):
    helpers.append(f'static void setup{j}(Model m){{'+'\n'.join(setup[start:start+100])+'}')
    calls.append(f'setup{j}(m);')
# Keep each generated Java method below the bytecode-size limit.
body='\n'.join(code[:4]+calls+code[cut:])
body=body.replace('public static void main(String[] args){run();}}',
                  '\n'.join(helpers)+'public static void main(String[] args){run();}}')
(P/'NativeThermalTi.java').write_text(body)
print('Built independent native Ti equations',len(names),'unknowns',len(times),'time points')
