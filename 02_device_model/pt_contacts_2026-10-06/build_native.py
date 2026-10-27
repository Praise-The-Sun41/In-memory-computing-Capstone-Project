"""Native COMSOL thermionic pair, independent of Python time trajectories."""
from pathlib import Path
import json
from contact_model import Model,Contacts,KT,P
m=Model();o=m.ox;q=json.dumps;fmt=lambda v:format(float(v),'.17g')
arr=lambda v:'new String[]{'+','.join(q(str(x)) for x in v)+'}'
var='m.component("c").variable("v")'
code=['import com.comsol.model.*;','import com.comsol.model.util.*;','public class PrepareContacts {public static Model run(){try{',
f'Model m=ModelUtil.load("contacts",{q((P.parent/"pt_blocking_2026-10-06/Pt-plain-unsolved.mph").as_posix())});',
'm.label("Pt contacts - thermionic injection scenarios - not calibrated");m.component("c").label("Oxide states + contact injection, no electrode oxygen");']
for key,value,desc in [('contact_ratio',.01,'Reference total contact resistance / initial bulk oxide resistance'),('contact_bottom_share',.2,'Bottom fraction of initial contact resistance'),('contact_lambda',.1,'Bottom vacancy barrier lowering coefficient in eV; scenario'),('contact_KT',KT,'300 K thermal voltage; eV convention'),('contact_Rref',o.Rref,'Frozen virgin oxide reference resistance, model units'),('contact_theta_ref',o.theta_ref,'Initial averaged vacancy occupancy')]:
 code.append(f'm.param().set({q(key)},{q(fmt(value))},{q(desc)});')
def add(k,v):code.append(f'{var}.set({q(k)},{q(v)});')
add('contact_theta','+'.join(fmt(w)+f'*th{i}' for i,w in enumerate(o.wb) if w))
add('contact_delta_phi','-contact_lambda*(contact_theta-contact_theta_ref)')
add('contact_Isb','contact_KT/(contact_Rref*contact_ratio*contact_bottom_share)*exp(-contact_delta_phi/contact_KT)')
add('contact_Ist','contact_KT/(contact_Rref*contact_ratio*(1-contact_bottom_share))')
code+=['m.func().create("cexpm1","Analytic");m.func("cexpm1").set("funcname","cexpm1");m.func("cexpm1").set("args","x");m.func("cexpm1").set("expr","if(abs(x)<1e-4,x+x^2/2+x^3/6+x^4/24+x^5/120,exp(x)-1)");']
add('contact_pair_current','contact_Isb*contact_Ist*cexpm1(vc/contact_KT)/(contact_Isb*exp(vc/contact_KT)+contact_Ist)')
add('contact_vt','contact_KT*(log(contact_Isb*exp(vc/contact_KT)+contact_Ist)-log(contact_Isb+contact_Ist))')
add('contact_vb','vc-contact_vt')
add('oxide_bias','current*resistance');add('bulkG','1/resistance')
add('conductance','1/(resistance+contact_KT/contact_Isb+contact_KT/contact_Ist)')
add('bulk_joule','+'.join(f'sig{i}*(dv{i}/{fmt(o.h[i])})^2*{fmt(o.h[i])}' for i in range(m.n)))
add('contact_power','current*vc');add('joule','bulk_joule+contact_power')
code+=['for(String key:new String[]{"name","equation","initialValueU","initialValueUt"}){String[] old=m.component("c").physics("ge").feature("ge1").getStringArray(key);if(old.length!=881)throw new IllegalStateException("Old count");String[] val=java.util.Arrays.copyOf(old,882);if(key.equals("name"))val[881]="vc";else if(key.equals("equation")){val[880]="ci*1e-8*resistance+vc-bias";val[881]="(current-contact_pair_current)*1e8";}else val[881]="0";m.component("c").physics("ge").feature("ge1").set(key,val);}',
'm.study("std").label("Pt contacts | own preparation + ten q2 cycles");m.sol("sol1").clearSolutionData();',
f'm.save({q((P/"Pt-contacts-unsolved.mph").as_posix())});','System.out.println("PREPARED_CONTACTS_882");return m;}catch(Exception e){throw new RuntimeException(e);}}public static void main(String[] args){run();}}']
(P/'PrepareContacts.java').write_text('\n'.join(code))
# Reuse the already audited geometry builder with contact-specific output fields.
src=(P.parent/'pt_blocking_2026-10-06/finish_pt.py').read_text()
src=src.replace('from pt_model import PtModel','from contact_model import Model as PtModel')
src=src.replace('Pt-plain-solved.mph','Pt-contacts-solved.mph').replace('FinishPt','FinishContacts')
src=src.replace("+[f'c.th{i}' for i in range(m.n)]","+[f'c.th{i}' for i in range(m.n)]+['c.contact_vb','c.contact_vt','c.oxide_bias','c.bulkG','c.contact_power','c.bulk_joule','c.contact_delta_phi','c.contact_Isb','c.contact_Ist']")
src=src.replace('d.length!=450','d.length!=459').replace('[5401][450]','[5401][459]').replace('j<450','j<459')
src=src.replace("phi=f'if(z<30[nm],0,if(z>55[nm],{stored(\"bias\")},{phi}))'","phi=f'if(z<30[nm],0,if(z>55[nm],{stored(\"bias\")},{stored(\"contact_vb\")}+({phi})))'")
src=src.replace('Pt_TiOx_AlOx_Pt_Geometry_Bound.mph','Pt_Contacts_Geometry_Bound.mph')
src=src.replace('Pt native ten-cycle IV','Pt contacts | native ten-cycle IV').replace('Pt blocking','Pt contacts, ion blocking')
src=src.replace('PT_GEOMETRY_AND_NATIVE_EXPORT_PASS','CONTACT_GEOMETRY_AND_NATIVE_EXPORT_PASS')
extra='''
for tag,label,expressions in [('contactdrops','07 | Contact and oxide voltage partition',['c.contact_vb','c.TiDrop','c.AlDrop','c.contact_vt']),('barriershift','08 | Bottom effective barrier change (eV)',['c.contact_delta_phi'])]:
    code += [f'm.result().create("{tag}","PlotGroup1D");m.result("{tag}").set("data","protocoldata");m.result("{tag}").label({q(label)});m.result("{tag}").create("g","Global");m.result("{tag}").feature("g").set("expr",'+arr(expressions)+');']
'''
src=src.replace('# True-scale crossbar',extra+'\n# True-scale crossbar')
(P/'finish_contacts.py').write_text(src)
# Native holds seed the oxide and algebraic variables from native C5 endpoints.
src=(P.parent/'pt_blocking_2026-10-06/pt_hold_native.py').read_text()
src=src.replace('Pt-plain-solved.mph','Pt-contacts-solved.mph').replace('PtHolds','ContactHolds').replace('881','882')
src=src.replace('PT_HOLD_','CONTACT_HOLD_')
(P/'hold_contacts.py').write_text(src)
# Read-only saved-file audit.
src=(P.parent/'pt_blocking_2026-10-06/audit_saved.py').read_text()
src=src.replace('AuditPtSaved','AuditContactsSaved').replace('Pt_TiOx_AlOx_Pt_Geometry_Bound.mph','Pt_Contacts_Geometry_Bound.mph').replace('881','882').replace('450','459').replace('PT_SAVED_HISTORY_AND_GEOMETRY_PASS','CONTACT_SAVED_HISTORY_AND_GEOMETRY_PASS')
src=src.replace('new double[3][3]','new double[5][3]')
potcheck='''
m.result().numerical("mapaudit").set("expr",new String[]{m.result("potentialview").feature("field").getString("expr")});
m.result().numerical().create("potnative","EvalGlobal");m.result().numerical("potnative").set("data","protocoldata");
String[] pe=new String[440];for(int i=0;i<440;i++)pe[i]="c.contact_vb+c.phi"+i;
m.result().numerical("potnative").set("expr",pe);m.result().numerical("potnative").set("solrepresentation","solnum");m.result().numerical("potnative").set("innerinput","manual");
for(int k=0;k<2;k++){
double tt=k==0?55.4:59.4;m.result().param().set("show_tau",Double.toString(tt));
m.result().numerical("potnative").set("solnum",new int[]{k==0?2641:2881});
double[][] want=m.result().numerical("potnative").getReal();double[][] val=m.result().numerical("mapaudit").getReal();double e=0;
for(int i=0;i<440;i++)e=Math.max(e,Math.abs(want[i][0]-val[i][0]));
if(!Double.isFinite(e)||e>1e-8)throw new IllegalStateException("Contact potential mapping "+e);
audit[3+k]=new double[]{tt,440,e};}
'''
src=src.replace('m.result().table().create("reopen"',potcheck+'\nm.result().table().create("reopen"')
(P/'audit_saved.py').write_text(src)
print('Generated native 882-equation model and geometry/hold builders.')
