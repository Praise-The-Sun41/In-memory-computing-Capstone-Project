"""Read-only check of the saved native potential-to-geometry binding."""
from pathlib import Path
import json
P=Path(__file__).resolve().parent;d=P/'chemical_500-smoke';q=json.dumps
code='''import com.comsol.model.*;
import com.comsol.model.util.*;
public class AuditChemGeometry {
public static Model run(){try{
Model m=ModelUtil.load("audit",MODEL);
m.result().numerical().create("cpcheck","EvalPoint");
m.result().numerical("cpcheck").set("data","mapcheck");
m.result().numerical("cpcheck").set("expr",new String[]{"("+m.result("potentialview").feature("potential").getString("expr")+")/1[V]"});
m.result().numerical().create("cnative","EvalGlobal");
String[] expr=new String[440];for(int i=0;i<440;i++)expr[i]="c.phiabs"+i;
m.result().numerical("cnative").set("expr",expr);
double[][] out=new double[2][3];int j=0;
for(int t:new int[]{2,6}){
m.result().param().set("show_tau",Integer.toString(t));
double[][] a=m.result().numerical("cpcheck").getReal();
m.result().numerical("cnative").set("solnum",new int[]{60*t+1});
double[][] b=m.result().numerical("cnative").getReal();
double error=0;for(int i=0;i<440;i++){double e=Math.abs(a[i][0]-b[i][0]);if(!Double.isFinite(e))throw new IllegalStateException("Nonfinite potential");error=Math.max(error,e);}
out[j++]=new double[]{t,440,error};
if(error>1e-10)throw new IllegalStateException("Contact potential mapping failed "+error);
}
m.result().table().create("cpaudit","Table");m.result().table("cpaudit").setTableData(out);m.result().table("cpaudit").save(OUTPUT);
System.out.println("CONTACT_GEOMETRY_MAPPING_PASS");return m;
}catch(Exception e){throw new RuntimeException(e);}}
public static void main(String[] args){run();}}
'''.replace('MODEL',q((d/'Ti-chemistry-contact.mph').as_posix())).replace('OUTPUT',q((P/'geometry_audit.csv').as_posix()))
(P/'AuditChemGeometry.java').write_text(code)
