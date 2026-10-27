import com.comsol.model.*;
import com.comsol.model.util.*;
import java.util.*;
public class HistoryProtocol extends FixedProtocol {
 public static Model run(){try{
  double[] fresh=seed();m=ModelUtil.load("protocol","bounded-comsol-strict-full-r6-unsolved.mph");
  double[][] starts=new double[4][];starts[0]=fresh;
  starts[1]=solve("history_high",fresh,pulses(.8,4,true),1e-8);
  starts[2]=solve("history_low",fresh,pulses(-.8,4,true),1e-8);
  List<double[]> hold=new ArrayList<>();hold.add(new double[]{100,0,0});read(hold);
  starts[3]=solve("history_aged",starts[1],hold,1e-8);
  int previous=0;
  for(int count:new int[]{8,32,128}){
   double[][] curves=new double[4][];
   for(int j=0;j<4;j++){
    starts[j]=solve("init"+count+"_history"+j,starts[j],pulses(-.8,count-previous,false),1e-8);
    solve("probe"+count+"_history"+j,starts[j],pulses(.8,16,true),1e-8);curves[j]=lastReads.clone();
   }
   previous=count;double worst=0;for(int k=0;k<curves[0].length;k++){double lo=Double.POSITIVE_INFINITY,hi=0,sum=0;for(int j=0;j<4;j++){lo=Math.min(lo,curves[j][k]);hi=Math.max(hi,curves[j][k]);sum+=curves[j][k];}worst=Math.max(worst,(hi-lo)/(sum/4));}
   System.out.println("CANDIDATE_COMPLETE "+count+" max_relative_spread="+worst);if(worst<=.01){System.out.println("HISTORY_GATE_PASSED "+count);break;}
  }
  m.save(ROOT+"/results/history_last_native.mph");System.out.println("EXPERIMENT_HISTORY_COMPLETE");return m;
 }catch(Exception e){throw new RuntimeException(e);}}
 public static void main(String[] args){run();}
}
