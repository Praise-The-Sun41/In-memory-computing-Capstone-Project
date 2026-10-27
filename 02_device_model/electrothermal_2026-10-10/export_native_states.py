"""Export solved states and save a labelled presentation copy; preserve source."""
from coupled import *
p=argparse.ArgumentParser();p.add_argument('--adaptive',action='store_true');a=p.parse_args()
variant='adaptive-' if a.adaptive else ''
class_name='ExportThermalAdaptiveStates' if a.adaptive else 'ExportThermalStates'
model=(P/f'Ti-thermal-{variant}geometry.mph').as_posix()
output=(P/f'Ti-thermal-{variant}states.csv').as_posix()
presentation=(P/f'Ti-thermal-{variant}presentation.mph').as_posix()
expressions=['t']+[f'c.th{i}' for i in range(440)]+[f'c.s{i}' for i in range(256)]
q=lambda x:json.dumps(x)
code='''import com.comsol.model.*;
import com.comsol.model.util.*;
public class CLASS_NAME {
public static Model run(){try{
Model m=ModelUtil.load("audit",MODEL);
m.result().numerical().create("etstates","EvalGlobal");
m.result().numerical("etstates").set("expr",new String[]{EXPRESSIONS});
double[][] d=m.result().numerical("etstates").getReal();
double[][] rows=new double[d[0].length][d.length];
for(int i=0;i<rows.length;i++)for(int j=0;j<d.length;j++)rows[i][j]=d[j][i];
m.result().table().create("statetable","Table");
m.result().table("statetable").setTableData(rows);
m.result().table("statetable").save(OUTPUT);
m.result().dataset().create("etnative","Solution");
m.result().dataset("etnative").set("solution","sol1");
m.result("etplot").set("data","etnative");
m.result("etplot").set("looplevelinput",new String[]{"manualindices"});
m.result("etplot").set("looplevelindices",new String[]{"range(601,1,3001)"});
m.result("etplot").label("Thermal circuit IV - five continuous cycles - assumed scales");
m.result("etplot").set("xlabelactive",true);m.result("etplot").set("xlabel","Actual device voltage (V)");
m.result("etplot").set("ylabelactive",true);m.result("etplot").set("ylabel","Scenario current (uA)");
m.result("etplot").feature("g").set("expr",new String[]{"1e6*c.physicalI"});
m.result().create("etiv5","PlotGroup1D");m.result("etiv5").set("data","etnative");
m.result("etiv5").label("Thermal circuit IV - cycle 5 - assumed scales");
m.result("etiv5").set("looplevelinput",new String[]{"manualindices"});
m.result("etiv5").set("looplevelindices",new String[]{"range(2521,1,3001)"});
m.result("etiv5").set("xlabelactive",true);m.result("etiv5").set("xlabel","Actual device voltage (V)");
m.result("etiv5").set("ylabelactive",true);m.result("etiv5").set("ylabel","Scenario current (uA)");
m.result("etiv5").create("g","Global");m.result("etiv5").feature("g").set("expr",new String[]{"1e6*c.physicalI"});
m.result("etiv5").feature("g").set("xdata","expr");m.result("etiv5").feature("g").set("xdataexpr","c.bias");
m.result("ettemp").set("data","etnative");
m.result("ettemp").set("xlabelactive",true);m.result("ettemp").set("xlabel","Model time (tau; not calibrated to seconds)");
m.result("ettemp").set("ylabelactive",true);m.result("ettemp").set("ylabel","Lumped temperature (K)");
m.result("ettemp").feature("g").set("xdata","expr");m.result("ettemp").feature("g").set("xdataexpr","c.tau");
m.result().param().set("show_tau","57.4","DISPLAY ONLY: model time / tau, 0 to 61.4; C5 after SET=57.4, after RESET=61.4; positive peak=55.4");
m.save(PRESENTATION);
System.out.println("STATE_EXPORT_COMPLETE");
return m;}catch(Exception e){throw new RuntimeException(e);}}
public static void main(String[] args){run();}}
'''.replace('CLASS_NAME',class_name).replace('MODEL',q(model)).replace('OUTPUT',q(output)).replace('PRESENTATION',q(presentation)).replace('EXPRESSIONS',','.join(map(q,expressions)))
(P/f'{class_name}.java').write_text(code)
