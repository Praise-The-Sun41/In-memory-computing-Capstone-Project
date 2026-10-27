"""Generate COMSOL-only C5 endpoint continuation, no Python states imported."""
from pathlib import Path
import json,numpy as np
P=Path(__file__).resolve().parent;q=json.dumps
arr=lambda a:'new String[]{'+','.join(q(str(v)) for v in a)+'}'
times=np.r_[np.linspace(0,10,121),np.linspace(0,90,121)[1:]+10,np.geomspace(.001,900,120)+100]
assert len(times)==361
expr=['t','c.conductance','c.inventory']+[f'c.th{i}' for i in range(440)]
lines=['import com.comsol.model.*;','import com.comsol.model.util.*;','public class ContactHolds {public static Model run(){try{Model m=null;']
for sign,row in [('pos',2761),('neg',3001)]:
    lines += [f'm=ModelUtil.load("hold{sign}",{q((P/"Pt-contacts-solved.mph").as_posix())});',
      'String[] names=m.component("c").physics("ge").feature("ge1").getStringArray("name");if(names.length!=882)throw new IllegalStateException("State count");',
      'String[] expressions=new String[names.length];for(int i=0;i<names.length;i++)expressions[i]="c."+names[i];',
      'm.result().numerical().create("seed","EvalGlobal");m.result().numerical("seed").set("data","protocoldata");m.result().numerical("seed").set("expr",expressions);',
      'm.result().numerical("seed").set("solrepresentation","solnum");m.result().numerical("seed").set("innerinput","manual");',
      f'm.result().numerical("seed").set("solnum",new int[]{{{row}}});double[][] s=m.result().numerical("seed").getReal();String[] initial=new String[882];for(int i=0;i<882;i++)initial[i]=Double.toString(s[i][0]);',
      'm.component("c").physics("ge").feature("ge1").set("initialValueU",initial);m.component("c").variable("v").set("bias","0");',
      'm.study("std").feature("time").set("tlist",'+q(' '.join(format(v,'.17g') for v in times))+');',
      'm.sol("sol1").feature("t1").set("tlist",'+q(' '.join(format(v,'.17g') for v in times))+');',
      'm.sol("sol1").feature("t1").set("maxstepbdf","10[s]");m.sol("sol1").clearSolutionData();m.sol("sol1").runAll();',
      'm.result().numerical().create("holdout","EvalGlobal");m.result().numerical("holdout").set("data","protocoldata");m.result().numerical("holdout").set("expr",'+arr(expr)+');',
      'double[][] d=m.result().numerical("holdout").getReal();if(d.length!=443||d[0].length!=361)throw new IllegalStateException("Hold shape");double[][] out=new double[361][443];for(int i=0;i<361;i++)for(int j=0;j<443;j++)out[i][j]=d[j][i];',
      'm.result().table().create("holdtable","Table");m.result().table("holdtable").setTableData(out);',
      f'm.result().table("holdtable").save({q((P/f"pt-native-hold-{sign}.csv").as_posix())});',
      'm.result().create("holdplot","PlotGroup1D");m.result("holdplot").set("data","protocoldata");m.result("holdplot").create("g","Global");',
      'm.result("holdplot").feature("g").set("expr",new String[]{"c.conductance/"+Double.toString(out[0][1])});m.result("holdplot").label("Native Pt hold G / source G; age in tau");m.result("holdplot").run();',
      'm.result().param().set("show_tau","0");',f'm.save({q((P/f"Pt-hold-{sign}.mph").as_posix())});',f'System.out.println("CONTACT_HOLD_{sign}_DONE");ModelUtil.remove("hold{sign}");']
# Local blocks avoid repeated Java variable declarations.
body='\n'.join(lines)
body=body.replace('m=ModelUtil.load("holdpos"','{m=ModelUtil.load("holdpos"').replace('m=ModelUtil.load("holdneg"','}{m=ModelUtil.load("holdneg"')
body+='\n}return null;}catch(Exception e){throw new RuntimeException(e);}}public static void main(String[] args){run();}}'
(P/'ContactHolds.java').write_text(body)
