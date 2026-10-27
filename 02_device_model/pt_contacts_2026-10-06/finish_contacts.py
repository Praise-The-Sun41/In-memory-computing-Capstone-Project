"""Attach display geometry after the native Pt solve, keeping integration 1D."""
from pathlib import Path
import json
import numpy as np
from contact_model import Model as PtModel
P=Path(__file__).resolve().parent;m=PtModel();ox=m.ox;q=json.dumps
fmt=lambda x:format(float(x),'.17g')
arr=lambda v:'new String[]{'+','.join(q(str(x)) for x in v)+'}'
da=lambda v:'new double[]{'+','.join(fmt(x) for x in v)+'}'
def lit(s):return 'String.join("",'+arr([s[i:i+12000] for i in range(0,len(s),12000)])+')'
def lookup(bounds,leaves):
    if len(leaves)==1:return leaves[0]
    k=len(leaves)//2
    return f'if((z/1[nm]-30)<{fmt(bounds[k])},{lookup(bounds[:k+1],leaves[:k])},{lookup(bounds[k:],leaves[k:])})'
def stored(v):return f"withsol('sol1',c.{v},setval(t,show_tau*1[s]))"
theta=lookup(ox.edges,[stored(f'th{i}') for i in range(m.n)])
sig=lookup(ox.edges,[stored(f'sig{i}') for i in range(m.n)])
phi=lookup(ox.edges,[f'({stored(f"phi{i}")}+((z/1[nm]-30)-{fmt(ox.x[i])})/{fmt(ox.h[i])}*{stored(f"dv{i}")})' for i in range(m.n)])
phi=f'if(z<30[nm],0,if(z>55[nm],{stored("bias")},{stored("contact_vb")}+({phi})))'
expr=['t','c.bias','c.conductance','c.current','c.TiDrop','c.AlDrop','c.epower','c.joule','c.inventory','c.exchange']+[f'c.th{i}' for i in range(m.n)]+['c.contact_vb','c.contact_vt','c.oxide_bias','c.bulkG','c.contact_power','c.bulk_joule','c.contact_delta_phi','c.contact_Isb','c.contact_Ist']
code=['import com.comsol.model.*;','import com.comsol.model.util.*;','public class FinishContacts {public static Model run(){try{',
 f'Model m=ModelUtil.load("pt",{q((P/"Pt-contacts-solved.mph").as_posix())});',
 'm.result().numerical().create("pteval","EvalGlobal");m.result().numerical("pteval").set("data","protocoldata");',
 'm.result().numerical("pteval").set("expr",'+arr(expr)+');',
 'double[][] d=m.result().numerical("pteval").getReal();if(d.length!=459||d[0].length!=5401)throw new IllegalStateException("Native shape");',
 'double[][] rows=new double[5401][459];for(int i=0;i<5401;i++)for(int j=0;j<459;j++){if(!Double.isFinite(d[j][i]))throw new IllegalStateException("Nonfinite");rows[i][j]=d[j][i];}',
 'm.result().table().create("pttable","Table");m.result().table("pttable").setTableData(rows);',f'm.result().table("pttable").save({q((P/"pt-native.csv").as_posix())});',
 'm.result().create("protocolplot","PlotGroup1D");m.result("protocolplot").set("data","protocoldata");m.result("protocolplot").create("g","Global");',
 'm.result("protocolplot").set("looplevelinput",new String[]{"manualindices"});m.result("protocolplot").set("looplevelindices",new String[]{"range(601,1,5401)"});',
 'm.result("protocolplot").feature("g").set("expr",new String[]{"c.current/"+Double.toString(rows[600][2])});m.result("protocolplot").feature("g").set("xdata","expr");m.result("protocolplot").feature("g").set("xdataexpr","c.bias");',
 'm.result("protocolplot").label("01 | Pt contacts | native ten-cycle IV");m.result("protocolplot").set("xlabelactive",true);m.result("protocolplot").set("xlabel","Top bias (V)");m.result("protocolplot").set("ylabelactive",true);m.result("protocolplot").set("ylabel","I / (initial G x 1 V)");',
 'm.result().param().set("show_tau","57.4","Absolute normalized time; C5 endpoints 57.4 and 61.4");m.param().set("patch_w","40[nm]");',
 'm.component().create("show",true);m.component("show").label("Pt LOCAL STACK - displayed 1D state");m.component("show").geom().create("gs",3);m.component("show").geom("gs").lengthUnit("nm");']
layers=[('be','Pt bottom',0,30,[.62,.67,.73]),('tiox','TiOx',30,20,[.25,.68,.65]),('alox','AlOx',50,5,[.94,.72,.32]),('te','Pt top',55,30,[.62,.67,.73])]
for tag,label,z,h,c in layers:
    g='m.component("show").geom("gs")'
    code += [f'{g}.create("{tag}","Block");{g}.feature("{tag}").label("{label} {h} nm");',f'{g}.feature("{tag}").set("size",new String[]{{"patch_w","patch_w","{h}[nm]"}});',f'{g}.feature("{tag}").set("pos",new String[]{{"-patch_w/2","-patch_w/2","{z}[nm]"}});{g}.feature("{tag}").set("selresult","on");']
code+=['m.component("show").geom("gs").run();m.component("show").mesh().create("ms");m.component("show").mesh("ms").feature("size").set("custom","on");m.component("show").mesh("ms").feature("size").set("hmax","40[nm]");m.component("show").mesh("ms").feature("size").set("hmin","1e-5[nm]");']
dist=[np.linspace(0,1,5),np.unique(np.r_[ox.edges[ox.edges<=20],ox.x[ox.ti]])/20,(np.unique(np.r_[ox.edges[ox.edges>=20],ox.x[~ox.ti]])-20)/5,np.linspace(0,1,5)]
for (tag,label,z,h,c),dd in zip(layers,dist):
    sel=f'm.component("show").selection("src_{tag}")';sw=f'm.component("show").mesh("ms").feature("sw_{tag}")'
    code += [f'm.component("show").selection().create("src_{tag}","Box");{sel}.set("entitydim",2);{sel}.set("zmin","{z-1e-7}[nm]");{sel}.set("zmax","{z+1e-7}[nm]");{sel}.set("condition","inside");',f'm.component("show").mesh("ms").create("sw_{tag}","Sweep");{sw}.selection().named("gs_{tag}_dom");{sw}.selection("sourceface").named("src_{tag}");{sw}.create("dist","Distribution");{sw}.feature("dist").set("explicit",{da(dd)});']
code+=['m.component("show").mesh("ms").run();m.component("show").physics().create("viz","CoefficientFormPDE","gs");m.component("show").physics("viz").label("DISPLAY ONLY - u=0, no 3D physics");m.component("show").physics("viz").feature("cfeq1").set("c","0");m.component("show").physics("viz").feature("cfeq1").set("a","1");m.component("show").physics("viz").feature("cfeq1").set("f","0");',
 'm.study().create("vs");m.study("vs").create("stat","Stationary");m.study("vs").feature("stat").activate("ge",false);m.study("std").feature("time").activate("viz",false);m.study("vs").createAutoSequences("all");m.sol("sol2").runAll();',
 'm.result().dataset().create("boundgeom","Solution");m.result().dataset("boundgeom").set("solution","sol2");',
 'm.component("show").selection().create("oxides","Union");m.component("show").selection("oxides").set("entitydim",3);m.component("show").selection("oxides").set("input",new String[]{"gs_tiox_dom","gs_alox_dom"});']
for tag,label in [('stackview','02 | Pt TiOx AlOx Pt | local 40 nm crop'),('vacancyview','03 | Oxide vacancy occupancy | Pt contacts, ion blocking'),('potentialview','05 | Potential (V) | stored Pt history'),('sigmaview','06 | Log model conductivity')]:
    code += [f'm.result().create("{tag}","PlotGroup3D");m.result("{tag}").set("data","boundgeom");m.result("{tag}").label({q(label)});m.result("{tag}").set("titletype","manual");m.result("{tag}").set("title",{q(label)});']
for i,(tag,label,z,h,c) in enumerate(layers):
    code += [f'm.result("stackview").create("{tag}","Volume");m.result("stackview").feature("{tag}").create("sel","Selection");m.result("stackview").feature("{tag}").feature("sel").selection().named("gs_{tag}_dom");m.result("stackview").feature("{tag}").set("expr","1");m.result("stackview").feature("{tag}").set("coloring","uniform");m.result("stackview").feature("{tag}").set("color","custom");m.result("stackview").feature("{tag}").set("customcolor",{da(c)});',
      f'm.result("stackview").create("mat{i}","Annotation");m.result("stackview").feature("mat{i}").set("text","{label} | {h} nm");m.result("stackview").feature("mat{i}").set("posxexpr",new String[]{{"28[nm]"}});m.result("stackview").feature("mat{i}").set("posyexpr",new String[]{{"-20[nm]"}});m.result("stackview").feature("mat{i}").set("poszexpr",new String[]{{"{z+h/2}[nm]"}});m.result("stackview").feature("mat{i}").set("showpoint",false);m.result("stackview").feature("mat{i}").set("anchorpoint","middleleft");m.result("stackview").feature("mat{i}").set("level","global");']
for tag,v in [('vacancyview',theta),('potentialview',phi),('sigmaview','log10('+sig+')')]:
    code += [f'm.result("{tag}").create("field","Volume");m.result("{tag}").feature("field").set("expr",{lit(v)});m.result("{tag}").feature("field").set("smooth","none");m.result("{tag}").feature("field").set("resolution","norefine");m.result("{tag}").feature("field").set("colortable","Viridis");']
    if tag!='potentialview':code+=[f'm.result("{tag}").feature("field").create("sel","Selection");m.result("{tag}").feature("field").feature("sel").selection().named("oxides");']
    if tag=='vacancyview':code+=[f'm.result("{tag}").feature("field").set("rangecoloractive","on");m.result("{tag}").feature("field").set("rangecolormin",0);m.result("{tag}").feature("field").set("rangecolormax",1);']
# Independently evaluate all FV centres at both C5 endpoints after mapping.
z=30+ox.x
code += ['m.result().dataset().create("mapcheck","CutPoint3D");m.result().dataset("mapcheck").set("data","boundgeom");',
 'm.result().dataset("mapcheck").set("pointx",'+arr(['0']*440)+');m.result().dataset("mapcheck").set("pointy",'+arr(['0']*440)+');m.result().dataset("mapcheck").set("pointz",'+arr([fmt(v) for v in z])+');',
 'm.result().numerical().create("mapaudit","EvalPoint");m.result().numerical("mapaudit").set("data","mapcheck");m.result().numerical("mapaudit").set("expr",new String[]{'+lit(theta)+'});',
 'double[][] audit=new double[2][3];for(int k=0;k<2;k++){int index=k==0?2760:3000;double tt=k==0?57.4:61.4;m.result().param().set("show_tau",Double.toString(tt));double[][] val=m.result().numerical("mapaudit").getReal();double err=0;for(int i=0;i<440;i++)err=Math.max(err,Math.abs(val[i][0]-rows[index][10+i]));if(!Double.isFinite(err)||err>1e-8)throw new IllegalStateException("Pt mapping error "+err);audit[k]=new double[]{tt,440,err};}',
 'm.result().table().create("maptable","Table");m.result().table("maptable").setTableData(audit);',f'm.result().table("maptable").save({q((P/"pt-mapping-audit.csv").as_posix())});']

for tag,label,expressions in [('contactdrops','07 | Contact and oxide voltage partition',['c.contact_vb','c.TiDrop','c.AlDrop','c.contact_vt']),('barriershift','08 | Bottom effective barrier change (eV)',['c.contact_delta_phi'])]:
    code += [f'm.result().create("{tag}","PlotGroup1D");m.result("{tag}").set("data","protocoldata");m.result("{tag}").label({q(label)});m.result("{tag}").create("g","Global");m.result("{tag}").feature("g").set("expr",'+arr(expressions)+');']

# True-scale crossbar, display only; the 300 um lead length is a drawing assumption.
code += ['m.component().create("layout",true);m.component("layout").label("Pt crossbar 100 um lines - display only");m.component("layout").geom().create("gl",3);m.component("layout").geom("gl").lengthUnit("um");']
for tag,label,z,h,c in layers:
    size=['300','100'] if tag=='be' else ['100','300'] if tag=='te' else ['100','100']
    code += [f'm.component("layout").geom("gl").create("{tag}","Block");m.component("layout").geom("gl").feature("{tag}").set("size",{arr(size+[fmt(h/1000)])});m.component("layout").geom("gl").feature("{tag}").set("pos",{arr([str(-float(size[0])/2),str(-float(size[1])/2),fmt(z/1000)])});m.component("layout").geom("gl").feature("{tag}").label("{label}");']
code += ['m.component("layout").geom("gl").run();m.result().export().create("img","Image3D");m.result().export("img").set("imagetype","png");m.result().export("img").set("unit","px");m.result().export("img").set("width",1600);m.result().export("img").set("height",1200);',
 'm.result().param().set("show_tau","57.4");']
for tag in ['stackview','vacancyview','potentialview']:
    code += [f'm.result().param().set("show_tau","{55.4 if tag=="potentialview" else 57.4}");m.result("{tag}").run();m.result().export("img").set("plotgroup","{tag}");m.result().export("img").set("pngfilename",{q((P/(tag+".png")).as_posix())});m.result().export("img").run();']
code += ['m.result().param().set("show_tau","57.4");m.result("protocolplot").run();',f'm.save({q((P/"Pt_Contacts_Geometry_Bound.mph").as_posix())});','System.out.println("CONTACT_GEOMETRY_AND_NATIVE_EXPORT_PASS");return m;}catch(Exception e){throw new RuntimeException(e);}}public static void main(String[] args){run();}}']
(P/'FinishContacts.java').write_text('\n'.join(code))
