import com.comsol.model.*;
import com.comsol.model.util.*;
import java.util.*;
public class HistoryResume extends FixedProtocol {
 public static Model run(){try{
  double[] fresh=seed();m=ModelUtil.load("protocol","bounded-comsol-strict-full-r6-unsolved.mph");
  double[][] starts=new double[4][];starts[0]=fresh;
  Model saved=ModelUtil.load("completed","HistoryProtocol_protocol.mph");
  for(int j=1;j<4;j++){double[][] table=saved.result().table("export"+j).getReal();double[] last=table[table.length-1];starts[j]=java.util.Arrays.copyOfRange(last,4,last.length);}
  double[][] completedProbe=saved.result().table("export5").getReal();
  double[][] completedInit=saved.result().table("export4").getReal();
  double[] continued=java.util.Arrays.copyOfRange(completedInit[completedInit.length-1],4,1141);
  for(double[] row:completedProbe){for(int k=4;k<700;k++)if(row[k]<=0||row[k]>=1)throw new Exception("Recovered state bounds");if(Math.abs(row[2]-completedProbe[0][2])>1e-6)throw new Exception("Recovered inventory drift");}
  saved.modelPath(ROOT);saved.result().table("export5").save("results/probe8_history0.csv");saved.save("results/probe8_history0.mph");
  ArrayList<Double> rec=new ArrayList<>();for(int k=0;k<completedProbe.length-1;k++)if(Math.abs(completedProbe[k][3]-.02)<1e-9&&completedProbe[k+1][3]<.02-1e-9)rec.add(completedProbe[k][1]);
  double[] recoveredCurve=new double[rec.size()];for(int k=0;k<rec.size();k++)recoveredCurve[k]=rec.get(k);ModelUtil.remove("completed");
  int previous=0;
  for(int count:new int[]{8,32,128}){
   double[][] curves=new double[4][];
   for(int j=0;j<4;j++){
    if(count==8 && j==0){starts[0]=continued;curves[0]=recoveredCurve;continue;}
    starts[j]=solve("init"+count+"_history"+j,starts[j],pulses(-.8,count-previous,false),1e-8);
    solve("probe"+count+"_history"+j,starts[j],pulses(.8,16,true),1e-8);curves[j]=lastReads.clone();
   }
   previous=count;double worst=0;for(int k=0;k<curves[0].length;k++){double lo=Double.POSITIVE_INFINITY,hi=0,sum=0;for(int j=0;j<4;j++){lo=Math.min(lo,curves[j][k]);hi=Math.max(hi,curves[j][k]);sum+=curves[j][k];}worst=Math.max(worst,(hi-lo)/(sum/4));}
   System.out.println("CANDIDATE_COMPLETE "+count+" max_relative_spread="+worst);if(worst<=.01){System.out.println("HISTORY_GATE_PASSED "+count);break;}
  }
  m.save("results/history_last_native.mph");System.out.println("EXPERIMENT_HISTORY_COMPLETE");return m;
 }catch(Exception e){throw new RuntimeException(e);}}
 public static void main(String[] args){run();}
}
