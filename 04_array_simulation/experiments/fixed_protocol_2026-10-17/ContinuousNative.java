import com.comsol.model.*;import com.comsol.model.util.*;import java.util.*;
public class ContinuousNative extends FixedProtocol {public static Model run(){try{
double[] fresh=seed();m=ModelUtil.load("protocol","bounded-comsol-strict-full-r6-unsolved.mph");
List<double[]> whole=new ArrayList<>();int previous=0;for(int n:new int[]{8,32,128}){if(n>128)break;whole.addAll(pulses(-.8,n-previous,false));previous=n;}whole.addAll(pulses(.8,16,true));solve("continuous_128",fresh,whole,1e-9);System.out.println("CONTINUOUS_NATIVE_COMPLETE");return m;
}catch(Exception e){throw new RuntimeException(e);}}public static void main(String[] args){run();}}
