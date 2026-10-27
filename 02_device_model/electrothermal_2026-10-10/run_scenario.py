"""Custom scenarios with explicit settings and collision-safe output names."""
from coupled import *
import re
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--name',required=True)
p.add_argument('--kind',choices=['Ti','Pt','Pt_contact'],default='Ti');p.add_argument('--mode',choices=['original','limited','thermal'],default='thermal')
p.add_argument('--cycles',type=int,default=5);p.add_argument('--rtol',type=float,default=2e-9);a=p.parse_args()
if not re.fullmatch(r'[A-Za-z0-9_-]+',a.name):raise ValueError('Use a simple unique output name.')
if any((P/(a.name+ext)).exists() for ext in ('.npz','.json')):raise FileExistsError('Use a new name; no scenario result is overwritten.')
s=Settings(**json.loads(a.config.read_text()))
if not (s.reference_G_S>0 and s.series_ohm>=0 and s.compliance_A>0 and s.thermal_R_K_W>0 and s.thermal_tau>0):raise ValueError('Invalid scale or circuit/thermal parameter.')
if not a.cycles>0:raise ValueError('Positive cycle count required.')
if not 0<a.rtol<1:raise ValueError('Relative tolerance must be between zero and one.')
m=Device(a.kind,a.mode,s);solve(m,protocol(a.cycles),m.initial,a.name,a.rtol)
