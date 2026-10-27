import com.comsol.model.*;
import com.comsol.model.util.*;
public class NativeThermalAdaptive {
  public static Model run(){try{
    String dir="F:/Cloud Storage/Onedrive/Academic Files/UoM/Casptone Project/Commercial Chip Development/md-python/comsol/electrothermal_2026-10-10/";
    Model m=ModelUtil.load("eta",dir+"Ti-thermal-unsolved.mph");
    m.sol("sol1").feature("t1").set("rtol",2e-6);
    m.sol("sol1").feature("t1").set("tstepsbdf","free");
    m.sol("sol1").clearSolutionData();
    m.sol("sol1").runAll();
    m.result().numerical().create("etout","EvalGlobal");
    m.result().numerical("etout").set("expr",new String[]{"t","c.source","c.bias","c.physicalI","c.tempK","c.inventory","c.heatResidual","c.TiDrop","c.AlDrop","c.outV"});
    double[][] d=m.result().numerical("etout").getReal();
    if(d[0].length!=3001)throw new IllegalStateException("Incomplete history");
    double[][] out=new double[3001][10];
    for(int i=0;i<3001;i++)for(int j=0;j<10;j++)out[i][j]=d[j][i];
    m.result().table().create("ettable","Table");m.result().table("ettable").setTableData(out);
    m.result().table("ettable").save(dir+"Ti-thermal-adaptive-native.csv");
    m.result().create("etplot","PlotGroup1D");m.result("etplot").create("g","Global");
    m.result("etplot").feature("g").set("expr",new String[]{"c.physicalI"});
    m.result("etplot").feature("g").set("xdata","expr");
    m.result("etplot").feature("g").set("xdataexpr","c.bias");
    m.result("etplot").label("Thermal circuit IV - assumed scales");
    m.result().create("ettemp","PlotGroup1D");m.result("ettemp").create("g","Global");
    m.result("ettemp").feature("g").set("expr",new String[]{"c.tempK"});
    m.result("ettemp").label("Lumped temperature - not spatial heat FEM");
    m.save(dir+"Ti-thermal-adaptive-geometry.mph");
    System.out.println("THERMAL_ADAPTIVE_COMPLETE");return m;
  }catch(Exception e){throw new RuntimeException(e);}}
  public static void main(String[] args){run();}
}
