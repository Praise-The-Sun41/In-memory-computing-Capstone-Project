import com.comsol.model.*;
import com.comsol.model.util.*;
import java.util.*;
import java.io.*;
public class SetProtocol {
 static final String ROOT="F:/Cloud Storage/Onedrive/Academic Files/UoM/Casptone Project/Array Simulation/experiments/set_reset_levels_2026-10-17";
 static Model m; static String[] names; static int serial=0;
 static String[] expr; static double[] lastReads;
 static void pulse(List<double[]> p,double v){p.add(new double[]{.05,0,v});p.add(new double[]{.2,v,v});p.add(new double[]{.05,v,0});p.add(new double[]{.2,0,0});}
 static void read(List<double[]> p){p.add(new double[]{.02,0,.02});p.add(new double[]{.1,.02,.02});p.add(new double[]{.02,.02,0});}
 static List<double[]> pulses(double v,int n,boolean reads){List<double[]> p=new ArrayList<>();read(p);for(int i=0;i<n;i++){pulse(p,v);if(reads)read(p);}return p;}
 static double[] solve(String tag,double[] initial,List<double[]> p,double rtol) throws Exception {
  m.modelPath(ROOT);
  String[] ini=new String[names.length];for(int i=0;i<ini.length;i++)ini[i]=Double.toString(initial[i]);
  m.sol("sol1").clearSolutionData();m.component("c").physics("ge").feature("ge1").set("initialValueU",ini);
  StringBuilder bias=new StringBuilder("0"),ts=new StringBuilder("0");double t=0;
  for(double[] a:p){if(a[1]!=a[2])bias.append("+("+(a[2]-a[1])+")*min(1,max(0,(tau-"+t+")/"+a[0]+"))");for(int k=1;k<=2;k++)ts.append(" "+(t+a[0]*k/2));t+=a[0];}
  m.component("c").variable("v").set("bias",bias.toString());m.study("std").feature("time").set("tlist",ts.toString());m.sol("sol1").feature("t1").set("tlist",ts.toString());
  m.sol("sol1").feature("t1").set("rtol",rtol);m.sol("sol1").feature("t1").set("tstepsbdf","strict");m.sol("sol1").feature("t1").set("maxstepbdf","0.02[s]");
  for(String k:m.sol("sol1").feature("t1").feature().tags())if(m.sol("sol1").feature("t1").feature(k).getType().equals("Direct"))m.sol("sol1").feature("t1").feature(k).set("linsolver","pardiso");
  long start=System.nanoTime();System.out.println("START_CASE "+tag);m.sol("sol1").runAll();
  String ev="out"+(serial++);m.result().numerical().create(ev,"EvalGlobal");m.result().numerical(ev).set("expr",expr);double[][] v=m.result().numerical(ev).getReal();int n=v[0].length;
  if(Math.abs(v[0][n-1]-t)>1e-7)throw new Exception("Incomplete time extent");
  double[][] table=new double[n][v.length];for(int j=0;j<n;j++)for(int i=0;i<v.length;i++){if(!Double.isFinite(v[i][j]))throw new Exception("Nonfinite result");table[j][i]=v[i][j];}
  String tb="export"+serial;m.result().table().create(tb,"Table");m.result().table(tb).setColumnHeaders(expr);m.result().table(tb).setTableData(table);m.result().table(tb).save(ROOT+"/results/"+tag+".csv");
  ArrayList<Double> rr=new ArrayList<>();for(int j=0;j<n-1;j++)if(Math.abs(v[3][j]-.02)<1e-9 && v[3][j+1]<.02-1e-9)rr.add(v[1][j]);lastReads=new double[rr.size()];for(int j=0;j<rr.size();j++)lastReads[j]=rr.get(j);
  double[] out=new double[names.length];for(int i=0;i<out.length;i++)out[i]=v[i+4][n-1];
  for(int j=0;j<n;j++){if(Math.abs(v[2][j]-v[2][0])>1e-6)throw new Exception("Inventory drift");for(int i=4;i<700;i++)if(v[i][j]<=0||v[i][j]>=1)throw new Exception("State bounds");}

  System.out.println("DONE_CASE "+tag+" rows="+n+" wall_s="+(System.nanoTime()-start)/1e9);return out;
 }
}
