"""Prepare labelled presentation copies only from accepted native models."""
from pathlib import Path
import json,argparse
P=Path(__file__).resolve().parent;q=json.dumps
a=argparse.ArgumentParser();a.add_argument('--case',choices=['chemical_500','chemical_5000']);args=a.parse_args()
cases=[args.case] if args.case else ['chemical_500','chemical_5000']
cls='PresentChemistry'+(args.case.split('_')[-1] if args.case else '')
code=['import com.comsol.model.*;','import com.comsol.model.util.*;',
      f'public class {cls} {{public static Model run(){{try{{Model m=null;']
for case in cases:
    d=P/(case+'-full');result=json.loads((d/'comparison.json').read_text());assert result['all_pass']
    code += ['if(m!=null)ModelUtil.remove("view");',f'm=ModelUtil.load("view",{q((d/"Ti-chemistry-contact.mph").as_posix())});',
        'm.result("chemiv").set("xlabelactive",true);m.result("chemiv").set("xlabel","Command voltage (V)");',
        'm.result("chemiv").set("ylabelactive",true);m.result("chemiv").set("ylabel","Electronic current (uA; assumed scale)");',
        'm.result("chemiv").label("01 | Five continuous IV cycles - command voltage");',
        'm.result().create("actualiv","PlotGroup1D");m.result("actualiv").set("data","chemnative");',
        'm.result("actualiv").label("02 | Cycle 5 IV - actual device voltage");',
        'm.result("actualiv").set("looplevelinput",new String[]{"manualindices"});m.result("actualiv").set("looplevelindices",new String[]{"range(1921,1,2401)"});',
        'm.result("actualiv").create("g","Global");m.result("actualiv").feature("g").set("expr",new String[]{"1e6*c.physicalI"});m.result("actualiv").feature("g").set("xdata","expr");m.result("actualiv").feature("g").set("xdataexpr","c.devV");',
        'm.result("actualiv").set("xlabelactive",true);m.result("actualiv").set("xlabel","Actual device voltage (V)");m.result("actualiv").set("ylabelactive",true);m.result("actualiv").set("ylabel","Electronic current (uA; assumed scale)");']
    for tag,label,expr,ylab in [('rh','03 | Chemical interface resistance','c.ifaceR','Interface resistance (ohm; scenario)'),
        ('gh','04 | Continuous write / hold / read history','1e6*c.deviceG','Instantaneous device conductance (uS; actual model T)'),
        ('th','05 | Lumped device temperature','c.tempK','Temperature (K; lumped model)')]:
        code += [f'm.result().create("{tag}","PlotGroup1D");m.result("{tag}").set("data","chemnative");',
            f'm.result("{tag}").label({q(label)});m.result("{tag}").create("g","Global");m.result("{tag}").feature("g").set("expr",new String[]{{{q(expr)}}});',
            f'm.result("{tag}").feature("g").set("xdata","expr");m.result("{tag}").feature("g").set("xdataexpr","c.tau");',
            f'm.result("{tag}").set("xlabelactive",true);m.result("{tag}").set("xlabel","Model time (tau; not calibrated seconds)");',
            f'm.result("{tag}").set("ylabelactive",true);m.result("{tag}").set("ylabel",{q(ylab)});']
    code += ['m.result("chemhistory").label("06 | Binding and local oxygen loading - not O/Ti");',
        'm.result().param().set("show_tau","36");',
        f'm.save({q((d/"Ti-chemistry-contact-presentation.mph").as_posix())});']
code += ['System.out.println("CHEMISTRY_PRESENTATION_COMPLETE");return m;}catch(Exception e){throw new RuntimeException(e);}}public static void main(String[] args){run();}}']
(P/(cls+'.java')).write_text('\n'.join(code))
