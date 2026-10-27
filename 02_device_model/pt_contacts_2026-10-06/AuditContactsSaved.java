import com.comsol.model.*;
import com.comsol.model.util.*;
public class AuditContactsSaved { public static Model run(){try{
Model m=ModelUtil.load("audit","F:/Cloud Storage/Onedrive/Academic Files/UoM/Casptone Project/Commercial Chip Development/md-python/comsol/pt_contacts_2026-10-06/Pt_Contacts_Geometry_Bound.mph");
if(m.component("c").physics("ge").feature("ge1").getStringArray("name").length!=882)throw new IllegalStateException("Unknown count");
double[][] d=m.result().numerical("pteval").getReal();
if(d.length!=459||d[0].length!=5401)throw new IllegalStateException("History shape");
double[][] ref=m.result().table("pttable").getReal();
if(ref.length!=5401||ref[0].length!=459)throw new IllegalStateException("Reference table shape");
double err=0;
for(int row=0;row<5401;row++)for(int j=0;j<459;j++)err=Math.max(err,Math.abs(d[j][row]-ref[row][j]));
if(!Double.isFinite(err)||err>1e-10)throw new IllegalStateException("Saved history error "+err);
double[][] audit=new double[5][3];audit[0]=new double[]{101.4,5401,err};
for(int k=0;k<2;k++){
double tt=k==0?57.4:61.4;int source=k==0?2760:3000;
m.result().param().set("show_tau",Double.toString(tt));
double[][] val=m.result().numerical("mapaudit").getReal();double me=0;
for(int i=0;i<440;i++)me=Math.max(me,Math.abs(val[i][0]-d[10+i][source]));
if(!Double.isFinite(me)||me>1e-8)throw new IllegalStateException("Saved mapping error "+me);
audit[k+1]=new double[]{tt,440,me};
}

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

m.result().table().create("reopen","Table");m.result().table("reopen").setTableData(audit);m.result().table("reopen").save("F:/Cloud Storage/Onedrive/Academic Files/UoM/Casptone Project/Commercial Chip Development/md-python/comsol/pt_contacts_2026-10-06/pt-reopen-audit.csv");
System.out.println("CONTACT_SAVED_HISTORY_AND_GEOMETRY_PASS");ModelUtil.remove("audit");return null;
}catch(Exception e){throw new RuntimeException(e);}}
public static void main(String[] args){run();}}
