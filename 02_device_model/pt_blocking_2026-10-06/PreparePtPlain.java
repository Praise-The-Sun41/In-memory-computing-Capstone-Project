import com.comsol.model.*;
import com.comsol.model.util.*;
public class PreparePtPlain {
 public static Model run(){try{
 Model m=ModelUtil.load("plain","Pt-unsolved.mph");
 for(String tag:m.result().tags())m.result().remove(tag);
 for(String tag:m.result().export().tags())m.result().export().remove(tag);
 for(String tag:m.result().numerical().tags())m.result().numerical().remove(tag);
 for(String tag:m.result().dataset().tags())m.result().dataset().remove(tag);
 m.study().remove("vs");m.sol().remove("sol2");
 m.component().remove("show");m.component().remove("layout");
 m.result().dataset().create("protocoldata","Solution");m.result().dataset("protocoldata").set("solution","sol1");
 m.save("Pt-plain-unsolved.mph");System.out.println("PT_PLAIN_READY");return m;
 }catch(Exception e){throw new RuntimeException(e);}}
 public static void main(String[] args){run();}
}
