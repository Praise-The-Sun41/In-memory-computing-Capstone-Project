"""
sdc_model.py —— Knowm SDC device array model calibrated against measurements

There is only one goal: **make the simulated check_window.py result match the real-board figure**,
so that all later exploration (sparse coding, binary weights, on-chip learning, fault tolerance) can first run
in simulation, without trying a chip for every revision.

★ **This model is calibrated for "when the devices worked".** The 16 pairs (g_lo, g_hi) in the table
  were measured in the 2026-09-19 sweep, when all 16 devices switched back and forth normally -- every trace in the figure
  has two complete plateaus, which is evidence of **normal operation**, not of damage.
  We did later damage the devices by overcurrent (the irreversible damage of manual Note 1), but that happened **after
  this measurement** and has no causal relation to this table. Do not mix the two:
    · table = real characteristics of healthy devices (window 2.9×, per-device spread, 4 slow, 5 jittery)
    · damage = a later lesson, now prevented by the current limit in device_limits.py

Difference from the `SimBoard` in `md-tune/tune_resistance.py`: that is a **generic**
memristor model (log-domain relaxation, symmetric, one shared r_on/r_off), used to validate control algorithms;
this is a model of **this particular chip**, with each device's window, switching speed and bad-read rate calibrated from measurement.

The interface matches SimBoard (select / read_resistance / read_parallel / pulse),
so check_window.py and example_zero.py can both drive it directly.

=================== Source of the calibration data (must be stated clearly) ===================

`MEASURED_WINDOWS` was read off **the figure of the 10-pass hardware sweep of 2026-09-19**:
the right endpoint of each range bar + the span label on its right, from which (g_lo, g_hi) are solved.
It is self-consistent -- the common window labelled in the figure is 230.66 µS, and this table gives max(g_lo)=230.6,
with the lower edge set by device 1; the upper edge 668.0 µS is set by device 10. Accuracy about ±5%.

**The `window_devices.csv` on disk is not from this run.** In that file devices 9–16 are all
0.33–0.48 µS (≈2.5 MΩ, essentially open), bad-read rate 33%, clip_frac 0.5~0.94,
which does not match the figure at all. For an exact calibration, call `from_csv()` with the CSV from the same run as the figure.

=================== What is in the model ===================

1. **Per-device logarithmic window** (measured table). State variable x∈[0,1], g = g_lo·(g_hi/g_lo)^x.

2. **Abrupt switching**. On the real traces one direction reaches its end in 5~10 pulses and then goes completely flat -- this is the SDC
   filamentary switch, completely different from the slow relaxation of the generic model. So the step is made large
   (`set_rate`/`reset_rate` ≈ 0.26/pulse ⇒ 95% covered in 10 pulses, consistent with the transition in the figure
   taking about 25% of the phase),
   and **slightly asymmetric** (reset a bit faster than set, as seen in the figure).

   ★ Do not make it faster. Reads happen every `--every` (default 4) pulses; if switching completes within one read interval,
     the net-displacement estimator (`_seg_move` takes the median of the first and last 1/4) cannot see
     the transition at all, displacements in both directions read as ~0, and 8 of the 16 devices are misclassified as `one-way`.
     Measured: at 0.65/pulse 8 are misclassified, at 0.45, 5. Real device transitions span 10~20
     pulses (about 2.5 read intervals) and are sampled, so the model must be slow on that scale too.

3. **Threshold**. Below `v_th` (manual typical 0.26 V) the device does not move at all.
   This is the most important part of the model; the reason is at the end of the file.

4. **Cycle-to-cycle variation** (c2c): random jitter in where each push "ends", which is the source of the few `unstable`
   devices.

5. **Bad reads, state-dependent**. The spikes in the real figure appear almost entirely on the **high-conductance plateau**;
   the low-conductance plateau has only ordinary noise. Physically this makes sense: high conductance ⇒ small R ⇒ the read pulse's
   share across the device is only a few millivolts (R≈1 kΩ, Rs=10 kΩ: a 0.1 V read gives only 9 mV),
   where the RC inversion of `knowm_md.read_resistance` is most ill-conditioned and returns inf or negative values.
   So the bad-read probability is modelled as a function of R/Rs; the smaller R, the more likely a bad read.

=================== Calibration result: how well it matches the real-board figure ===================

    python check_window.py --model sdc --vdev 1.2 --i-limit 455 --repeats 3

(`--vdev 1.2 --i-limit 455` is required -- the figure was measured under these conditions:
the drive hit the AWG 5 V limit and actually pushed about 455 µA. With the default 100 µA,
**this set of resistances + 10 kΩ series resistor cannot switch a single device**, and the script reports exactly that.)

| | Hardware figure | Model |
|---|---|---|
| Common window | 230.66 – 668.0 µS | 223.40 – 681.0 µS |
| Span | 2.90× | 3.05× |
| Lower edge set by | device 1 | device 1 |
| Upper edge set by | device 10 | device 10 |
| Median span error over 16 devices | — | **2.9%** (max 8.3%) |

**The status labels are deliberately not matched.** That figure came from code **before bad-read immunity was added**; in its legend slow and
unstable are still the same yellow (exactly the bug I fixed later). At the time 16 devices were labelled
4 slow + 5 unstable; running this model with the **current** code leaves only 2 unstable.
Simulation showed earlier that 3% bad reads alone create 5 fake slow devices. So this is **a testable prediction** --
rerunning the real chip with the current code, most devices should return to ok.

=================== What current that measurement cost (this is the key point) ===================

**This is not "the chip is broken"; it is "the current we used was too large".** The difference decides
what to do next, so it must be worked out clearly.

The high-conductance end measured 690~1563 Ω. The voltage a device can get has a hard limit:

    V_dev ≤ I_limit · (R ∥ Rs)

R=1 kΩ, Rs=10 kΩ ⇒ R∥Rs = 909 Ω. To exceed the 0.26 V switching threshold requires

    I ≥ 0.26 / 909 Ω ≈ **286 µA**

That sweep only switched because the drive hit the AWG 5 V limit and actually pushed about **455 µA**. The devices
did all move at the time -- the 16 complete double-plateau traces are the evidence -- but 455 µA lies right in the irreversible damage range
of manual Note 1. **In other words: that nice figure was itself the price.**
The later damage was the cumulative result of this operating point, not an inherent defect of the devices.

The compliance current is now set at 100 µA (corresponding to V_dev ≤ 0.09 V), and `switching_headroom()`
honestly reports "0/16 switchable". This is not the model saying the chip is broken; it is saying
**"at a safe current, this Rs with this resistance range cannot switch at all; the hardware must change"** --
the manual's own endurance test uses a 50 kΩ series resistor, not 10 kΩ.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import device_limits as L                                  # noqa: E402

N_DEV = 16

# (g_lo, g_hi) read off the 2026-09-19 hardware figure, in µS. See the source notes in the file header.
MEASURED_WINDOWS = [
    (230.7, 1153), (124.3, 870), (73.3, 880), (52.5, 1050),
    (25.9, 700), (73.5, 1250), (46.8, 1030), (183.3, 1100),
    (76.7, 690), (66.8, 668), (37.3, 1120), (162.5, 1300),
    (207.1, 1450), (63.9, 1150), (67.5, 1080), (27.8, 1030),
]

# status of each device in the figure, used to check whether the model reproduces the same classification
MEASURED_STATUS = {
    1: "ok", 2: "ok", 3: "unstable", 4: "unstable", 5: "slow", 6: "ok",
    7: "ok", 8: "slow", 9: "slow", 10: "unstable", 11: "ok", 12: "slow",
    13: "ok", 14: "ok", 15: "unstable", 16: "unstable",
}


class SDCArray:
    """
    Model of this chip. The interface is a subset of `md-tune/tune_resistance.SimBoard`;
    check_window.py / example_zero.py can drive it directly.
    """

    def __init__(self, rs=10_000.0, windows=None, seed=0,
                 v_th=None, set_rate=0.26, reset_rate=0.30,
                 c2c=0.017, read_noise=0.02,
                 bad_read_at_1k=0.06, bad_read_knee=0.30,
                 unstable=(3, 4, 10, 15, 16), slow=(5, 8, 9, 12),
                 slow_factor=0.16, unstable_extra_c2c=0.045,
                 rail_margin=1.03,
                 write_jitter=0.0, write_miss=0.0, write_flip=0.0,
                 dead=()):
        self.rs = float(rs)
        self.rng = np.random.default_rng(seed)
        self.v_th = L.V_TH_FWD["W"] if v_th is None else float(v_th)
        self.set_rate, self.reset_rate = set_rate, reset_rate
        self.c2c, self.read_noise = c2c, read_noise
        # ---- write-control "inaccuracy" ----
        # these three parameters are the point of the demo: on the real board we **cannot do closed-loop verification at all**
        # (cannot switch at compliance current, see end of file), only "fire one pulse in one direction and move on".
        #   write_jitter random step multiplier per pulse (log-normal σ)
        #   write_miss   probability that this pulse has no effect at all
        #   write_flip   probability that this pulse goes the wrong direction (polarity misjudged / device reversed)
        self.write_jitter = write_jitter
        self.write_miss = write_miss
        self.write_flip = write_flip
        self.dead = set(int(i) for i in dead)
        self.bad_read_at_1k = bad_read_at_1k
        self.bad_read_knee = bad_read_knee

        # the measured window is **measured**, and plateau medians always sit slightly inside the true rails
        # (c2c + read noise). So the model's true rails must extend slightly beyond the table values,
        # so that the measured values return to the table. rail_margin is this compensation.
        w = MEASURED_WINDOWS if windows is None else windows
        m = float(rail_margin)
        self.g_lo = np.array([a * 1e-6 / m for a, _ in w])
        self.g_hi = np.array([b * 1e-6 * m for _, b in w])

        # per-device "temperament": whichever devices are slow / unstable in the figure get the corresponding fault.
        # slow  = each pulse moves less (still climbing after the pulses end)
        # unstable = large c2c jitter (the end position differs every pass)
        self.rate_scale = np.ones(N_DEV)
        self.c2c_i = np.full(N_DEV, c2c)
        for i in slow:
            self.rate_scale[i - 1] = slow_factor
        for i in unstable:
            self.c2c_i[i - 1] = c2c + unstable_extra_c2c

        self.x = self.rng.uniform(0.3, 0.7, N_DEV)      # state at power-up
        # c2c is a rail position jittered **once per half-cycle**, not per pulse. Applying it per pulse would
        # make x wander on the plateau: a 37× device has ln span 3.6, so 0.1 jitter in x is 43%
        # conductance jitter, the end-rate criterion reads absurd values like 400~600%, and the plateau median is
        # pulled inward. Real devices are random in "where this push ends", not jittering after they get there.
        self._last_dir = np.zeros(N_DEV)
        self._target = np.zeros(N_DEV)
        self.pol = np.ones(N_DEV)
        self.sel_all = (0,)
        self.sel = 0
        self.pulse_count = 0
        self.read_count = 0

    def kill(self, *idx, mode="short"):
        """
        Kill a few devices on the spot -- the "create a fault in public" action in the demo.

        The failure modes follow the manual's Critical Notes, not invented:

          mode="short" (default, = the real failure mode from the manual)
              "A forward applied voltage **without current limiting will cause
              devices to enter a very low-resistance state and consequently
              burn out**." -- the device is stuck at the **low-resistance / high-conductance** end and can no longer be written.
              This is what overcurrent burnout looks like, and exactly the risk this chip has always faced.

          mode="open"
              Another failure: stuck at the high-resistance end. A burnt contact or a fully dissolved filament looks like this.

          mode="stuck"
              Frozen in place, conductance kept at its current value. This **hardly affects accuracy**
              -- the learned weight is still there as it was -- so it is a control, not for the demo.

        ★ "short" is the one the demo uses because: one shorted device on a branch
          pulls the parallel resistance down by itself, the device voltage of the whole branch drops below threshold,
          and **every pattern containing it can no longer be written** (if written in bulk).
          Per-device writing avoids this. This is exactly what the `--bulk-write` control group is meant to show.
        """
        for i in idx:
            i = int(i)
            self.dead.add(i)
            if mode == "short":
                self.x[i] = 1.0
            elif mode == "open":
                self.x[i] = 0.0
            elif mode != "stuck":
                raise ValueError(f"unknown failure mode {mode!r}")

    def revive(self):
        self.dead = set()

    # ---------------------------------------------------------------- state
    def _g(self, i, x=None):
        x = self.x[i] if x is None else x
        return self.g_lo[i] * (self.g_hi[i] / self.g_lo[i]) ** np.clip(x, 0, 1)

    def conductances(self):
        return np.array([self._g(i) for i in range(N_DEV)])

    # ---------------------------------------------------------------- interface
    def select(self, *idx, settle=0.0):
        self.sel_all = tuple(idx) if idx else (0,)
        self.sel = self.sel_all[0]

    def all_off(self):
        pass

    def close(self):
        pass

    def _measure(self, r_true):
        """
        Turn a **real branch resistance** into one reading: multiplicative noise + resistance-dependent bad reads.

        ★ Noise and bad reads are added to the **branch**, not to each device. This is not a detail:
          `knowm_md.read_parallel` measures the branch current once and then inverts the RC; it cannot see
          individual devices at all. An earlier version injected bad reads per device in parallel reads; if just one of 16 devices
          drew a "negative resistance", `1/max(-r, 1e-9)` became 1e9 S -- the whole branch read as
          1e-9 Ω, and c shot up to 1e13 µS. A bug that killed the whole demo.
        """
        r = r_true * math.exp(self.rng.normal(0, self.read_noise))
        # bad-read probability rises as R/Rs falls -- at the high-conductance end the read pulse splits only a few millivolts,
        # where the RC inversion is most ill-conditioned. The spikes in the real figure are exactly all on the high-conductance plateau.
        p = self.bad_read_at_1k / (1.0 + (r / (self.bad_read_knee * self.rs)) ** 2)
        if self.rng.random() < p:
            u = self.rng.random()
            if u < 0.12:
                return float("inf")             # current reading ≈ 0
            if u < 0.22:
                return -abs(r) * 0.02           # negative after subtracting the parasitic
            return r * 10 ** self.rng.uniform(0.7, 2.0)   # single point jumps 1~2 orders of magnitude
        return r

    def _read_one(self, i):
        return self._measure(1.0 / self._g(i))

    def read_resistance(self, index=None, **kw):
        if index is not None:
            self.sel = index
        self.read_count += 1
        return self._read_one(self.sel)

    def read_parallel(self):
        self.read_count += 1
        g = sum(self._g(i) for i in self.sel_all)
        return self._measure(1.0 / max(g, 1e-12))

    WIDTH_REF = 50.0     # µs, pulse width used to calibrate set_rate/reset_rate

    def pulse(self, amplitude=1.0, width_us=50.0, n=1, **kw):
        """
        Write the whole connected batch of devices together. The device voltage is set by the **parallel** resistance, shared by all.

        Below threshold nothing moves -- the most important rule of this model, and the reason this chip currently
        cannot be switched: at the high-conductance end R∥Rs ≈ 909 Ω, and 100 µA gives only 0.09 V,
        far below the 0.26 V threshold.

        **Pulse width scales linearly** (`width_us / WIDTH_REF`). The SDC is a filamentary switch;
        above threshold filament growth is driven by ion transport, and to first order the state change of one pulse
        is proportional to the time the field acts. Calibration used 50 µs, so the default width changes no existing
        conclusion; `check_window.py` reproducing the hardware figure also goes through this path.

        ★ This is not optional decoration -- **write granularity is the real bottleneck of this chip at present.**
          One 50 µs pulse moves 15~26% of the log window, so weights cannot land on the optimum and can only wander
          around it (measured 5~7 percentage points worse in 02_online_digits). Cutting the width to
          5 µs makes the steps 10× finer, and the same learning rule immediately does much better.
        ★ The model is not calibrated below 1 µs. Real devices have an incubation time; pulses that are too narrow may not
          trigger at all, and the linear extrapolation here no longer holds.
        """
        wscale = max(float(width_us), 0.0) / self.WIDTH_REF
        idx = self.sel_all
        for _ in range(n):
            self.pulse_count += 1
            g_par = sum(self._g(i) for i in idx)
            r_par = 1.0 / max(g_par, 1e-12)
            v_dev = abs(amplitude) * r_par / (r_par + self.rs)
            if v_dev <= self.v_th:
                continue                         # below threshold: no movement
            drive = min(1.0, (v_dev - self.v_th) / self.v_th)
            up = (amplitude * self.pol[idx[0]]) > 0
            d = 1.0 if up else -1.0
            for i in idx:
                if self._last_dir[i] != d:
                    # direction change = new half-cycle: redraw where this push will end
                    self._last_dir[i] = d
                    base = 1.0 if up else 0.0
                    self._target[i] = float(np.clip(
                        base + self.rng.normal(0, self.c2c_i[i]) * (-d),
                        -0.25, 1.25))
                if i in self.dead:
                    continue                     # devices killed on purpose: cannot be written
                if self.write_miss and self.rng.random() < self.write_miss:
                    continue                     # this pulse was lost
                tgt = self._target[i]
                if self.write_flip and self.rng.random() < self.write_flip:
                    tgt = 1.0 - tgt              # this pulse went the wrong direction
                rate = (self.set_rate if up else self.reset_rate)
                step = min(1.0, rate * drive * self.rate_scale[i] * wscale)
                if self.write_jitter:
                    step = min(1.0, step * math.exp(
                        self.rng.normal(0, self.write_jitter)))
                self.x[i] += step * (tgt - self.x[i])
                self.x[i] = float(np.clip(self.x[i], 0.0, 1.0))


# ---------------------------------------------------------------- diagnostics
def switching_headroom(rs=10_000.0, i_limit=None, windows=None, v_th=None):
    """
    At a given compliance current, whether each device **can be switched**: compare V_dev = I·(R∥Rs) with the threshold.

    Returns (list of rows, number of devices still switchable). This is the only basis for judging "is this chip still usable".
    """
    i_limit = L.I_COMPLIANCE_DEFAULT if i_limit is None else i_limit
    v_th = L.V_TH_FWD["W"] if v_th is None else v_th
    w = MEASURED_WINDOWS if windows is None else windows
    rows, alive = [], 0
    for k, (lo, hi) in enumerate(w, 1):
        r_hi_g = 1.0 / (hi * 1e-6)          # high-conductance end = smallest resistance
        r_lo_g = 1.0 / (lo * 1e-6)
        v_at_hi = L.max_device_voltage(r_hi_g, rs, i_limit)
        v_at_lo = L.max_device_voltage(r_lo_g, rs, i_limit)
        ok = v_at_hi >= v_th
        alive += ok
        rows.append(dict(dev=k, r_min=r_hi_g, r_max=r_lo_g,
                         v_at_hi=v_at_hi, v_at_lo=v_at_lo, ok=ok,
                         i_needed=v_th / (r_hi_g * rs / (r_hi_g + rs))))
    return rows, alive


def report_headroom(rs=10_000.0, i_limits=(100e-6, 200e-6, 300e-6, 455e-6)):
    """Print out the account of 'how much current is needed to switch'."""
    v_th = L.V_TH_FWD["W"]
    out = [f"Switching threshold taken as {v_th:.2f} V (manual type W typical). V_dev ≤ I·(R∥Rs).", ""]
    out.append("  Compliance I   High-G end V_dev   Switchable devices   Manual fig. 4 HRS")
    for i in i_limits:
        rows, alive = switching_headroom(rs, i)
        v = np.median([r["v_at_hi"] for r in rows])
        out.append(f"  {i*1e6:6.0f} µA     {v*1000:7.0f} mV        "
                   f"{alive:2d}/16            {L.expected_hrs(i):.0e} Ω")
    need = max(r["i_needed"] for r in switching_headroom(rs, 100e-6)[0])
    out.append("")
    out.append(f"  ⇒ the hardest device needs **{need*1e6:.0f} µA** to reach the threshold.")
    out.append(f"    while manual Note 1 says a current of this order **irreversibly** lowers the high-resistance state --")
    out.append(f"    that is: each push narrows the window a little more. This is a dead loop.")
    return "\n".join(out)


def from_csv(path):
    """Recalibrate from the window_devices.csv produced by check_window.py (recommended)."""
    import csv as _csv
    w, status = [None] * N_DEV, {}
    with open(path, newline="") as f:
        for row in _csv.DictReader(f):
            k = int(row["device"])
            try:
                lo, hi = float(row["g_min_uS"]), float(row["g_max_uS"])
            except ValueError:
                continue
            w[k - 1] = (lo, hi)
            status[k] = row["status"]
    if any(x is None for x in w):
        raise ValueError(f"some devices in {path} are missing g_min/g_max")
    return w, status


if __name__ == "__main__":
    print("=" * 74)
    print("Within the safe current, can this set of resistances + a 10 kΩ series resistor be switched?")
    print("=" * 74)
    print(report_headroom())
