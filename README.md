# In-memory computing with memristors — capstone code (E-12-ZJIN-003)

Code accompanying the final report, University of Melbourne. Release `v1.0-report`
is the version used for every number in the report.

| Report section | Folder | Content |
|---|---|---|
| 3.2, 4.2 | `02_device_model` | Python finite-volume device models (top-level `*.py`), dated run folders such as `ti_pt_history_2026-10-18/run.py` for the reported runs, and the COMSOL Java model scripts, including the 500/5000 Ω cross-validation (`ti_contact_comsol_2026-10-18`) |
| 3.3, 4.3 | `03_commercial_device` | Memristor Discovery V2 driver (`knowm_md.py`), safety gate (`safety.py`, `device_limits.py`), kT-RAM library (`ktram.py`), calibrated chip model (`sdc_model.py`), bar classifier and inverted-pendulum demonstrations |
| 4.3.1 | `03_commercial_device/data` | I–V and G–V sweeps of the 16 commercial devices |
| 3.4, 4.4 | `04_array_simulation/experiments` | SET/RESET level campaign, fixed-protocol checks, handwritten-digit network training and resident-array runs |

## Quick start (commercial device, simulated chip)

```
pip install numpy matplotlib
cd 03_commercial_device
python example_bars.py                    # bar classifier, sign readout
python example_bars.py --readout abs      # |y| readout
python demos/01_cartpole/demo_cartpole.py --seed 0
```
Add `--hw` to `example_bars.py` to run on the Memristor Discovery V2 board
(requires Digilent WaveForms and an Analog Discovery 2). Every hardware call
passes through the protection gate in `safety.py`.

The COMSOL scripts in `02_device_model` and `04_array_simulation` require
COMSOL Multiphysics 6.x with LiveLink for Java. Large binary files are not
included; see `NOT_INCLUDED.txt`.
