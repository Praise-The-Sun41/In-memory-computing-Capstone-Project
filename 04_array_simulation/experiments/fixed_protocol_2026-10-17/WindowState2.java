import com.comsol.model.*;
import com.comsol.model.util.*;
import java.util.*;
public class WindowState2 extends FixedProtocol {
 static double[] snapshot(String path,double age) throws Exception {
  Model s=ModelUtil.load("snapshot",path);s.result().numerical().create("windowSeed","EvalGlobal");s.result().numerical("windowSeed").set("expr",expr);
  double[][] v=s.result().numerical("windowSeed").getReal();int last=v[0].length-1;if(age>=0){last=-1;for(int k=0;k<v[0].length;k++)if(Math.abs(v[0][k]-age)<1e-8)last=k;if(last<0)throw new Exception("Missing native snapshot time");}double[] out=new double[names.length];for(int i=0;i<out.length;i++)out[i]=v[i+4][last];ModelUtil.remove("snapshot");return out;
 }
 public static Model run(){try{
  double[] fresh=seed();m=ModelUtil.load("protocol","bounded-comsol-strict-full-r6-unsolved.mph");
  // Chosen checkpoint is explicitly substituted by orchestration after the history gate.
  double[] low=snapshot("results/init128_history0.mph",-1),high=snapshot("results/probe128_history0.mph",-1);
  double[] mid=snapshot("results/probe128_history0.mph",.78);
  for(int state=2;state<3;state++){
   double[] y=state==0?low:(state==1?mid:high);
   for(int duration:new int[]{1,10,100}){List<double[]> p=new ArrayList<>();p.add(new double[]{duration,0,0});read(p);solve("hold"+duration+"_state"+state,y,p,1e-8);}
   for(int count:new int[]{1,16,80}){
    List<double[]> p=new ArrayList<>();for(int k=0;k<count;k++)read(p);solve("reads"+count+"_state"+state,y,p,1e-8);
    List<double[]> sham=new ArrayList<>();if(count>1)sham.add(new double[]{(count-1)*.14,0,0});read(sham);solve("readsham"+count+"_state"+state,y,sham,1e-8);
   }
   // Ideal row/column voltage sources: 2x2 V/2 addresses (0,0).
   // Selected=V, both half-selected=V/2, unselected=0; finite-line study remains separate.
   for(double v:new double[]{.8,-.8})for(int count:new int[]{1,16}){
    String sign=v>0?"set":"reset";
    for(double factor:new double[]{1,.5,0}){
     List<double[]> p=pulses(v*factor,count,false);read(p);
     solve("array2_"+sign+"_N"+count+"_state"+state+"_factor"+factor,y,p,1e-8);
    }
   }
  }
  m.save(ROOT+"/results/window_state2_last_native.mph");System.out.println("EXPERIMENT_WINDOW_COMPLETE");return m;
 }catch(Exception e){throw new RuntimeException(e);}}
 public static void main(String[] args){run();}
}
