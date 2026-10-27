import com.comsol.model.*;
import com.comsol.model.util.*;
public class RecoverReference extends FixedProtocol {
 public static Model run(){try{
  seed();
  Model partial=ModelUtil.load("partial","ContinuousNative_protocol.mph");
  partial.result().numerical().create("recoverReference","EvalGlobal");
  partial.result().numerical("recoverReference").set("expr",expr);
  double[][] v=partial.result().numerical("recoverReference").getReal();
  int n=v[0].length,index=-1;
  for(int k=0;k<n;k++)if(Math.abs(v[0][k]-64.42)<1e-8)index=k;
  if(index<0)throw new Exception("Completed continuous initialization checkpoint missing");
  double[][] table=new double[n][v.length];
  for(int k=0;k<n;k++)for(int j=0;j<v.length;j++){
   if(!Double.isFinite(v[j][k]))throw new Exception("Nonfinite partial data");
   table[k][j]=v[j][k];
  }
  partial.result().table().create("recoveredPartial","Table");
  partial.result().table("recoveredPartial").setColumnHeaders(expr);
  partial.result().table("recoveredPartial").setTableData(table);
  partial.result().table("recoveredPartial").save(ROOT+"/results/continuous_attempt_partial.csv");
  double[] y=new double[names.length];for(int j=0;j<y.length;j++)y[j]=v[j+4][index];
  System.out.println("RECOVERED_CONTINUOUS_INIT time="+v[0][index]+" partial_end="+v[0][n-1]);
  ModelUtil.remove("partial");
  m=ModelUtil.load("protocol","bounded-comsol-strict-full-r6-unsolved.mph");
  solve("independent_probe128",y,pulses(.8,16,true),1e-9);
  m.save(ROOT+"/results/independent_reference_completed.mph");
  System.out.println("INDEPENDENT_REFERENCE_COMPLETE");return m;
 }catch(Exception e){throw new RuntimeException(e);}}
 public static void main(String[] args){run();}
}
