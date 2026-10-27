"""
tune_resistance.py —— closed-loop writing of the 16 memristors to specified resistance / conductance in Mode 1

    python tune_resistance.py --sim --ramp 20k,200k          # offline simulation, check first
    python tune_resistance.py --target 50k                   # real hardware, all 16 to 50 kΩ
    python tune_resistance.py --devices 0-3 --target 100k     # only the first 4

The board's Mode switch must be at **1** (one device connected at a time, series resistance A∥B = 10 kΩ).

=================== How the loop is closed ===================

    read R  →  above target? → SET pulse (lower resistance)
               below target? → RESET pulse (raise resistance)
               in dead band? → done

It sounds like a bang-bang controller, but written naively it oscillates around the target and never stops.
Four things are added here, each corresponding to a real device characteristic:

1. **Error is computed in the log domain.** Device updates are multiplicative (Δg ∝ remaining range),
   so 50 kΩ→45 kΩ and 500 kΩ→450 kΩ are steps of "the same size". Using the absolute difference of R as error
   overshoots at the high-resistance end and dawdles at the low-resistance end.

2. **The drive amplitude is inverted from the voltage division.** The device actually gets V_drive · R/(R+Rs);
   the lower R, the less it gets -- a positive-feedback trap: the lower the resistance, the harder to move
   (manual section 12.4; this is also what `check_ktram.py --hard-erase` deals with).
   So the control variable is not the drive voltage but the **device voltage** V_dev:

        V_drive = V_dev · (R + Rs) / R          ← recomputed each step from the currently measured R

   This lets the same parameters work at both 5 kΩ and 1 MΩ.

3. **Refine on overshoot (successive approximation).** Once the error changes sign the step went too far; multiply V_dev by 0.7
   and continue -- equivalent to bisection on amplitude. This is the standard program-and-verify approach in the RRAM literature.

4. **Raise voltage if it will not move; give up if it still will not at the top.** If R does not move for several consecutive steps,
   the switching threshold was not crossed, so V_dev is raised step by step; if it still does not move at the limit, the device is judged stuck
   and reported instead of looping forever.

Polarity (whether a positive pulse is SET or RESET) is **not hard-coded**; by default each device is measured once before starting --
on this chip half the devices write in the opposite direction to nominal, so hard-coding would get everything wrong.

=================== Output ===================

Terminal table + `tune_result.csv` (with R at every step) + `tune_result.png`
(4×4 small multiples, one device's convergence trajectory per cell + a final error bar).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time

# ---------------------------------------------------------------- locate knowm_md
_HERE = os.path.dirname(os.path.abspath(__file__))
_MDP = None          # the directory containing knowm_md.py / plotstyle.py
# this folder may be inside md-python/ or next to it; both must be recognised
for _cand in (os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "md-python"),
              _HERE, os.path.join(_HERE, "md-python")):
    if os.path.isfile(os.path.join(_cand, "knowm_md.py")):
        _MDP = os.path.abspath(_cand)
        sys.path.insert(0, _MDP)
        break


def mdp_path(name: str) -> str:
    """For terminal hints: how to write the path of that md-python script from the current working directory."""
    if not _MDP:
        return name
    try:
        return os.path.join(os.path.relpath(_MDP, os.getcwd()), name).replace("\\", "/")
    except ValueError:          # on Windows a relative path cannot be computed across drive letters
        return os.path.join(_MDP, name)


# ---------------------------------------------------------------- default parameters
SERIES_R = 10_000.0      # Mode 1: two 20 kΩ in parallel at A/B on the board. If one is removed, change to 20000
BOARD_VERSION = 2
N_DEVICES = 16

V_READ = 0.10            # read pulse amplitude (V). Small enough not to disturb the state
READ_WIDTH_US = 25.0
READ_AVERAGES = 3        # number of samples for the median per read -- the controller must not chase noise

PULSE_WIDTH_US = 50.0
V_DEV_START = 0.55       # starting **device voltage** (not drive voltage)
V_DEV_MIN = 0.20         # refinement lower limit: below this the switching threshold cannot be crossed
# ---- device protection: all numbers come from device_limits.py (Knowm datasheet Rev 3.2) ----
# the old values were V_DEV_MAX=1.60 / V_DRIVE_MAX=2.50, **both violating the manual**:
# the manual says devices should "normally operate under 1 V" and switch at 0.1~0.75 V; and a 2.5 V drive
# with a 10 kΩ series resistor gives 250 µA in the worst case, which manual Note 1 says
# **irreversibly** lowers the high-resistance state from 1 MΩ to below 100 kΩ.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import device_limits as L                                   # noqa: E402

V_DEV_MAX = L.V_DEV_MAX_DEFAULT      # 0.75 V, top of the manual's switching range
I_COMPLIANCE = L.I_COMPLIANCE_DEFAULT  # 100 µA, worst case R→0
V_DRIVE_ABS_MAX = L.AWG_ABS_MAX      # the AD2's W1 can only reach ±5 V
V_DRIVE_MAX = L.max_drive(SERIES_R, I_COMPLIANCE, V_DEV_MAX)
V_DRIVE_WARN = V_DRIVE_MAX           # should warn when exceeding the limit
SHRINK = 0.70            # amplitude shrink after overshoot
GROW = 1.20              # amplitude growth when it will not move

TOL = 0.05               # dead band: |R/R* − 1| < 5% counts as on target
MAX_ITERS = 60           # maximum iterations per device
PATIENCE = 6             # number of consecutive no-movement steps before judging stuck
MOVE_FLOOR = 0.015       # |Δln R| below this counts as "not moved" (must exceed read noise)

R_MIN_VALID = 200.0      # a reading below this is most likely a short / switch not fully open
R_MAX_VALID = 20e6       # above this the 1 MΩ scope input impedance dominates and the reading is unreliable


# ---------------------------------------------------------------- unit parsing
_SUFFIX = {"": 1.0, "k": 1e3, "K": 1e3, "M": 1e6, "meg": 1e6, "G": 1e9,
           "m": 1e-3, "u": 1e-6, "µ": 1e-6, "μ": 1e-6, "n": 1e-9, "p": 1e-12}


def parse_value(s: str) -> float:
    """'10k' → 10000.0, '4.7K' → 4700.0, '50u' → 5e-5, '1M' → 1e6."""
    s = s.strip().rstrip("ΩΩohmOHM ").strip()
    if not s:
        raise ValueError("empty value")
    for suf in ("meg", "µ", "μ", "k", "K", "M", "G", "m", "u", "n", "p"):
        if s.endswith(suf):
            return float(s[: -len(suf)]) * _SUFFIX[suf]
    return float(s)


def parse_devices(s: str) -> list[int]:
    """'0-3,7,10-11' → [0,1,2,3,7,10,11]"""
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    bad = [i for i in out if not 0 <= i < N_DEVICES]
    if bad:
        raise ValueError(f"device index outside 0..{N_DEVICES-1}: {bad}")
    return sorted(set(out))


def fmt_r(r: float) -> str:
    if r != r or r in (float("inf"), float("-inf")):
        return "   ---  "
    if r >= 1e6:
        return f"{r/1e6:7.3f}M"
    if r >= 1e3:
        return f"{r/1e3:7.2f}k"
    return f"{r:7.1f} "


# ---------------------------------------------------------------- simulation back end
class SimBoard:
    """
    Offline simulation; the interface is a subset of MemristorDiscovery.

    Model: resistance moves multiplicatively in the log domain within [r_on, r_off], with the step modulated by
    **the voltage the device actually gets**, and no movement at all below threshold. This is exactly the real device's
    "the lower the resistance, the harder to move" positive-feedback trap, which the controller must handle itself.

    faults can inject faults into some devices to test the controller's fallback paths:
        "flip"  reversed polarity      "stuck" cannot be moved      "weak" weak response
    """

    def __init__(self, rs=SERIES_R, seed=0, faults=None,
                 r_on=3e3, r_off=2e6, v_th=0.30, eta=0.16, noise=0.02):
        self.rs = rs
        self.board_version = BOARD_VERSION
        self.rng = random.Random(seed)
        self.r_on, self.r_off = r_on, r_off
        self.v_th, self.eta, self.noise = v_th, eta, noise
        self.faults = dict(faults or {})
        # power-up values: scattered randomly within the range, simulating an unprepared chip
        self.r = [math.exp(self.rng.uniform(math.log(8e3), math.log(8e5)))
                  for _ in range(N_DEVICES)]
        self.sel = 0
        self.pulse_count = 0
        self.read_count = 0

    # -- MemristorDiscovery interface
    def select(self, *idx, settle=0.0):
        self.sel = idx[0] if idx else 0

    def all_off(self):
        pass

    def close(self):
        pass

    def read_resistance(self, index=None, v_read=V_READ,
                        width_us=READ_WIDTH_US, use_rc_model=True):
        if index is not None:
            self.sel = index
        self.read_count += 1
        return self.r[self.sel] * math.exp(self.rng.gauss(0, self.noise))

    def pulse(self, amplitude=1.0, width_us=PULSE_WIDTH_US, n=1,
              shape="square", duty=0.5, samples_per_pulse=300, capture=False):
        i = self.sel
        fault = self.faults.get(i)
        for _ in range(n):
            self.pulse_count += 1
            r = self.r[i]
            v_dev = abs(amplitude) * r / (r + self.rs)      # ★ voltage division
            if v_dev <= self.v_th:
                continue
            drive = min(1.0, (v_dev - self.v_th) / self.v_th)
            step = self.eta * drive * (1 + self.rng.gauss(0, self.noise))
            step = max(0.0, min(1.0, step))
            if fault == "stuck":
                continue
            if fault == "weak":
                step *= 0.15
            setting = amplitude > 0
            if fault == "flip":
                setting = not setting
            lr = math.log(r)
            if setting:                                     # SET: lower resistance
                lr += step * (math.log(self.r_on) - lr)
            else:                                           # RESET: raise resistance
                lr += step * (math.log(self.r_off) - lr)
            self.r[i] = min(self.r_off, max(self.r_on, math.exp(lr)))
        return None


# ---------------------------------------------------------------- controller
class Tuner:
    def __init__(self, board, tol=TOL, v_dev_start=V_DEV_START,
                 width_us=PULSE_WIDTH_US, max_iters=MAX_ITERS,
                 v_drive_max=V_DRIVE_MAX, averages=READ_AVERAGES,
                 settle=0.0, verbose=True):
        self.b = board
        self.rs = getattr(board, "rs", SERIES_R)
        self.tol = tol
        # the starting value must also be clamped to [V_DEV_MIN, V_DEV_MAX]. The raising path has min(V_DEV_MAX, ...),
        # but that is only reached when "it will not move"; an oversized --vdev would bypass it and be fired directly.
        self.v_dev_start = min(V_DEV_MAX, max(V_DEV_MIN, v_dev_start))
        self.width_us = width_us
        self.max_iters = max_iters
        self.v_drive_max = min(v_drive_max, V_DRIVE_ABS_MAX)
        self.averages = averages
        self.settle = settle
        self.verbose = verbose
        self.polarity = {}          # device → +1 means a positive pulse lowers resistance (SET)

    # ---- basic operations
    def read(self, idx: int) -> float:
        """Read N times and take the median. Single-read noise would make the controller chase shadows."""
        vals = []
        for _ in range(self.averages):
            r = self.b.read_resistance(index=idx, v_read=V_READ,
                                       width_us=READ_WIDTH_US)
            if r == r and 0 < r < float("inf"):
                vals.append(r)
        if not vals:
            return float("nan")
        vals.sort()
        return vals[len(vals) // 2]

    def fire(self, idx: int, v_dev: float, r_now: float, setting: bool, n=1):
        """
        Fire a pulse after inverting the drive voltage from the target **device voltage**.

            V_drive = V_dev · (R + Rs) / R

        Returns the drive voltage actually used (possibly truncated by v_drive_max).
        """
        r = max(r_now, 1.0)
        v_drive = v_dev * (r + self.rs) / r
        v_drive = min(v_drive, self.v_drive_max)
        pol = self.polarity.get(idx, +1)
        amp = (pol if setting else -pol) * v_drive
        self.b.select(idx)
        if self.settle:
            time.sleep(self.settle)
        self.b.pulse(amplitude=amp, width_us=self.width_us, n=n, capture=False)
        return v_drive

    # ---- polarity measurement
    def detect_polarity(self, idx: int, r0: float | None = None) -> int:
        """
        Does a positive pulse lower or raise resistance? Fire two positive pulses and see which way R moves.

        A considerable fraction of devices on this chip go opposite to nominal; hard-coding polarity would drive the controller
        all the way in the wrong direction. Measuring once costs 2 pulses, well worth it.
        """
        r0 = self.read(idx) if r0 is None else r0
        v_dev = self.v_dev_start
        for _ in range(3):
            self.polarity[idx] = +1
            self.fire(idx, v_dev, r0, setting=True, n=2)
            r1 = self.read(idx)
            d = math.log(r1 / r0) if r0 > 0 and r1 > 0 else 0.0
            if d < -MOVE_FLOOR:              # positive pulse lowers resistance → positive = SET
                self.polarity[idx] = +1
                return +1
            if d > MOVE_FLOOR:               # positive pulse raises resistance → reversed polarity
                self.polarity[idx] = -1
                return -1
            v_dev = min(V_DEV_MAX, v_dev * 1.4)   # did not move; raise voltage and retry
            r0 = r1
        self.polarity[idx] = +1              # cannot be measured; use the nominal value
        return 0

    # ---- main loop
    def tune(self, idx: int, r_target: float, autopol=True):
        """Converge device idx to r_target. Returns a record."""
        trace_p, trace_r = [], []           # cumulative pulse count / measured resistance
        r = self.read(idx)
        pulses = 0
        trace_p.append(0)
        trace_r.append(r)

        rec = {"device": idx, "chip": idx + 1, "target": r_target,
               "r_start": r, "polarity": 0, "pulses": 0, "iters": 0,
               "final": r, "err": float("nan"), "status": "fail",
               "note": "", "trace_p": trace_p, "trace_r": trace_r}

        if not (R_MIN_VALID < r < R_MAX_VALID):
            rec["note"] = "initial reading out of range (short / switch not open / beyond range)"
            return rec

        if autopol:
            pol = self.detect_polarity(idx, r)
            rec["polarity"] = pol
            pulses += 6                      # detect fires at most 3×2
            r = self.read(idx)
            trace_p.append(pulses)
            trace_r.append(r)
            if pol == 0:
                rec["note"] = "polarity could not be measured (neither direction can be moved)"

        v_dev = self.v_dev_start
        prev_sign = 0
        stall = 0
        band = math.log1p(self.tol)

        for it in range(1, self.max_iters + 1):
            rec["iters"] = it
            if not (r == r and r > 0):
                rec["note"] = "invalid reading"
                break
            err = math.log(r / r_target)     # >0 → resistance too high → needs SET
            if abs(err) <= band:
                rec["status"] = "ok"
                break

            setting = err > 0
            sign = 1 if setting else -1
            if prev_sign and sign != prev_sign:
                v_dev = max(V_DEV_MIN, v_dev * SHRINK)     # overshoot → refine
            prev_sign = sign

            # far away: several pulses at once; close: one step at a time
            a = abs(err)
            n = 8 if a > 1.2 else (3 if a > 0.35 else 1)

            self.fire(idx, v_dev, r, setting=setting, n=n)
            pulses += n
            r_new = self.read(idx)

            moved = abs(math.log(r_new / r)) if (r_new == r_new and r_new > 0) else 0.0
            if moved < MOVE_FLOOR:
                stall += 1
                v_dev = min(V_DEV_MAX, v_dev * GROW)       # will not move → raise voltage
            else:
                stall = 0

            r = r_new
            trace_p.append(pulses)
            trace_r.append(r)

            if stall >= PATIENCE and v_dev >= V_DEV_MAX * 0.999:
                rec["status"] = "stuck"
                rec["note"] = "raised to the limit and still not moving"
                break
        else:
            rec["status"] = "maxiter"
            rec["note"] = f"not in the dead band after {self.max_iters} iterations"

        rec["final"] = r
        rec["pulses"] = pulses
        if r == r and r > 0:
            rec["err"] = r / r_target - 1.0
            if rec["status"] == "ok" or abs(rec["err"]) <= self.tol:
                rec["status"] = "ok"
        return rec


# ---------------------------------------------------------------- target generation
def build_targets(devices, args):
    """Returns {device index: target resistance Ω}."""
    n = len(devices)
    if args.targets:
        vals = [parse_value(v) for v in args.targets.split(",")]
        if len(vals) != n:
            raise SystemExit(f"--targets gives {len(vals)} values, but {n} devices are selected")
    elif args.ramp:
        a, b = (parse_value(v) for v in args.ramp.split(","))
        if n == 1:
            vals = [a]
        else:   # log-uniform spacing -- devices are linear only in the log domain
            vals = [math.exp(math.log(a) + (math.log(b) - math.log(a)) * i / (n - 1))
                    for i in range(n)]
    elif args.target:
        vals = [parse_value(args.target)] * n
    else:
        raise SystemExit("A target must be given: one of --target / --targets / --ramp")

    if args.conductance:            # input interpreted as conductance, converted to resistance
        vals = [1.0 / v for v in vals]
    return dict(zip(devices, vals))


# ---------------------------------------------------------------- figures
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"
LINE, TGT = "#2a78d6", "#8a8a84"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8a84"


def plot_results(recs, tol, path="tune_result.png"):
    """
    4×4 small multiples + one error summary.

    The 16 curves **cannot** go on one plot: a categorical palette has only 8 slots, and from the 9th on
    generated colours are needed, which readers cannot tell apart. Small multiples are the right answer for "several isomorphic series",
    and one device per cell is easier to debug anyway.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError as e:
        print(f"(skipping figures: {e})")
        return None
    try:
        from plotstyle import setup, T
        setup(verbose=False)
    except ImportError:
        def T(zh, en):
            return en

    recs = sorted(recs, key=lambda r: r["device"])
    fig = plt.figure(figsize=(13, 10.5))
    # row 5 is a near-zero-height placeholder row, only to separate the summary bar from the small multiples
    gs = GridSpec(6, 4, figure=fig, height_ratios=[1, 1, 1, 1, 0.02, 1.05],
                  hspace=0.62, wspace=0.30)

    for k, rec in enumerate(recs[:16]):
        ax = fig.add_subplot(gs[k // 4, k % 4])
        p, r = rec["trace_p"], rec["trace_r"]
        tgt = rec["target"]
        ax.axhspan(tgt * (1 - tol), tgt * (1 + tol), color=TGT, alpha=0.16, lw=0)
        ax.axhline(tgt, color=TGT, lw=1.2, ls="--", zorder=2)
        ax.plot(p, r, color=LINE, lw=2, marker="o", ms=3.5,
                mfc=LINE, mec="white", mew=0.6, zorder=3)
        ax.set_yscale("log")
        # the trajectory of a quickly converging device is almost a straight line, squeezing the y axis to a few tenths of a dB,
        # and the dead band fills the whole cell. Give each cell a minimum span so the context remains readable.
        pos = [v for v in r if v == v and v > 0] + [tgt]
        lo, hi = min(pos), max(pos)
        mid = math.sqrt(lo * hi)
        span = max(hi / lo, 1.8) ** 0.5
        ax.set_ylim(mid / span / 1.12, mid * span * 1.12)
        ax.grid(True, which="major", axis="y", color="#e6e5e1", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#d8d7d2")
        ax.tick_params(labelsize=7.5, colors=INK2, length=3)
        col = {"ok": GOOD, "stuck": CRIT, "maxiter": WARN}.get(rec["status"], CRIT)
        mark = {"ok": "", "stuck": "  STUCK", "maxiter": "  SLOW"}.get(rec["status"], "  FAIL")
        e = rec["err"]
        ax.set_title(f"D{rec['chip']}   {e*100:+.1f}%{mark}" if e == e
                     else f"D{rec['chip']}   --{mark}",
                     fontsize=9, color=col, pad=4, loc="left")
        if k % 4 == 0:
            ax.set_ylabel(T("Resistance (Ω)", "Resistance (Ω)"), fontsize=8, color=INK2)
        if k // 4 == 3:
            ax.set_xlabel(T("Cumulative pulses", "Pulses"), fontsize=8, color=INK2)

    # ---- summary bar
    ax = fig.add_subplot(gs[5, :])
    xs = list(range(len(recs)))
    errs = [(r["err"] * 100 if r["err"] == r["err"] else 0.0) for r in recs]
    cols = [GOOD if r["status"] == "ok" else (WARN if r["status"] == "maxiter" else CRIT)
            for r in recs]
    # a stuck device can have an error of −70%; setting the y axis by it would flatten the other 15 into a line.
    # so the axis is set by those "on the tolerance scale"; the out-of-range bar is truncated with its value still labelled.
    inl = [abs(e) for e in errs if abs(e) <= 25.0]
    lim = max(12.0, (max(inl) if inl else 10.0) * 1.45)
    ax.axhspan(-tol * 100, tol * 100, color=GOOD, alpha=0.10, lw=0)
    ax.axhline(0, color=INK3, lw=1)
    ax.bar(xs, [max(-lim, min(lim, e)) for e in errs], color=cols,
           width=0.62, zorder=3)
    for x, e, r in zip(xs, errs, recs):
        clipped = abs(e) > lim
        y = max(-lim, min(lim, e))
        if clipped:                       # truncated: value labelled inside the bar
            ax.text(x, y * 0.82, f"{e:+.0f}", ha="center", va="center",
                    fontsize=7.5, color="white", weight="bold")
        else:
            off = lim * 0.045
            txt = f"{e:+.1f}" if r["err"] == r["err"] else "--"
            ax.text(x, e + (off if e >= 0 else -off), txt, ha="center",
                    va="bottom" if e >= 0 else "top", fontsize=7.5, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"D{r['chip']}" for r in recs], fontsize=8, color=INK2)
    ax.set_ylabel(T("Final error (%)", "Final error (%)"), fontsize=9, color=INK2)
    ax.set_ylim(-lim, lim)
    ax.grid(True, axis="y", color="#e6e5e1", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, length=3)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=GOOD, label=T("on target", "within tolerance")),
                       Patch(facecolor=WARN, label=T("iterations exhausted", "hit iteration limit")),
                       Patch(facecolor=CRIT, label=T("stuck / failed", "stuck / failed"))],
              loc="upper right", fontsize=8, frameon=False, ncol=3)

    fig.text(0.5, 0.972, T("Closed-loop writing: convergence trajectories of 16 devices",
                           "Closed-loop resistance tuning - convergence per device"),
             ha="center", fontsize=14, color=INK, weight="bold")
    fig.text(0.5, 0.949,
             T(f"dashed = target, shaded band = ±{tol*100:.0f}% dead band; log y axis",
               f"Dashed = target, shaded band = ±{tol*100:.0f}% dead zone; log y-axis"),
             ha="center", fontsize=9.5, color=INK2)
    fig.subplots_adjust(top=0.925, bottom=0.05, left=0.07, right=0.975)
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    return path


def write_csv(recs, path="tune_result.csv"):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["device", "chip", "target_ohm", "r_start_ohm", "final_ohm",
                    "error_pct", "polarity", "pulses", "iters", "status", "note"])
        for r in sorted(recs, key=lambda x: x["device"]):
            w.writerow([r["device"], r["chip"], f"{r['target']:.1f}",
                        f"{r['r_start']:.1f}", f"{r['final']:.1f}",
                        f"{r['err']*100:.2f}" if r["err"] == r["err"] else "",
                        r["polarity"], r["pulses"], r["iters"],
                        r["status"], r["note"]])
        w.writerow([])
        w.writerow(["--- per-step trajectory ---"])
        w.writerow(["device", "chip", "cum_pulses", "resistance_ohm"])
        for r in sorted(recs, key=lambda x: x["device"]):
            for p, rr in zip(r["trace_p"], r["trace_r"]):
                w.writerow([r["device"], r["chip"], p, f"{rr:.1f}"])
    return path


# ---------------------------------------------------------------- main program
def main():
    ap = argparse.ArgumentParser(
        description="Closed-loop writing of 16 memristors to specified resistance/conductance in Mode 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Choose one target option:
  --target 50k              write all to 50 kΩ
  --targets 10k,20k,...     one value per device (count must match --devices)
  --ramp 20k,200k           log-uniform staircase over the selected devices
With --conductance the values above are interpreted as conductance (e.g. --target 20u = 20 µS = 50 kΩ)
""")
    ap.add_argument("--sim", action="store_true", help="Offline simulation; hardware not touched")
    ap.add_argument("--devices", default="0-15", help="Device indices, e.g. 0-15 or 0,3,5-7")
    ap.add_argument("--target", help="Same target for all devices, e.g. 50k")
    ap.add_argument("--targets", help="Per-device targets, comma separated")
    ap.add_argument("--ramp", help="Start and end values, log-uniform over the selected devices, e.g. 20k,200k")
    ap.add_argument("--conductance", action="store_true",
                    help="Interpret the values above as conductance (S) rather than resistance (Ω)")
    ap.add_argument("--tol", type=float, default=TOL * 100,
                    help=f"Dead band, percent (default {TOL*100:.0f})")
    ap.add_argument("--max-iters", type=int, default=MAX_ITERS)
    ap.add_argument("--vdev", type=float, default=V_DEV_START,
                    help=f"Starting device voltage V (default {V_DEV_START})")
    ap.add_argument("--vmax", type=float, default=V_DRIVE_MAX,
                    help=f"Drive voltage limit V (default {V_DRIVE_MAX}, "
                         f"AD2 hard limit {V_DRIVE_ABS_MAX:.0f})")
    ap.add_argument("--width", type=float, default=PULSE_WIDTH_US,
                    help=f"Pulse width µs (default {PULSE_WIDTH_US})")
    ap.add_argument("--averages", type=int, default=READ_AVERAGES,
                    help=f"Reads per measurement for the median (default {READ_AVERAGES})")
    ap.add_argument("--rs", type=float, default=SERIES_R,
                    help=f"Series resistance Ω (Mode 1 default {SERIES_R:.0f} = two 20k in parallel)")
    ap.add_argument("--no-autopol", action="store_true",
                    help="Do not measure polarity; assume positive pulses lower resistance")
    ap.add_argument("--seed", type=int, default=0, help="Simulation random seed")
    ap.add_argument("--sim-faults", default="",
                    help="Inject faults into the simulation, e.g. 3:stuck,7:flip,9:weak")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out", default="tune_result", help="Output file name prefix")
    args = ap.parse_args()

    devices = parse_devices(args.devices)
    targets = build_targets(devices, args)
    tol = args.tol / 100.0

    print("=" * 74)
    print("Mode 1 closed-loop writing -- write devices to specified resistance")
    print("=" * 74)

    if args.sim:
        faults = {}
        for item in args.sim_faults.split(","):
            if ":" in item:
                k, v = item.split(":", 1)
                faults[int(k)] = v.strip()
        board = SimBoard(rs=args.rs, seed=args.seed, faults=faults)
        print(f"Simulation mode  Rs={args.rs/1e3:.0f}kΩ  "
              f"{len(devices)} devices  dead band ±{args.tol:.0f}%"
              + (f"  injected faults {faults}" if faults else ""))
    else:
        import os as _os, sys as _sys
        _sys.path.insert(0, _os.path.dirname(_os.path.dirname(
            _os.path.abspath(__file__))))
        import scope_guard
        scope_guard.refuse_hardware(
            "Closed-loop resistance tuning (md-tune/tune_resistance.py)",
            why="It is a tool that **pushes resistance to a target in closed loop**: if the target is not reached, it keeps pushing. "
                "Within the safety envelope this cannot be done (at 50 µA the single-pulse step is limited, and there is a resistance floor), "
                "and the old version did it precisely by raising the drive to reach the target",
            suggest="Measure the window first: python characterize_safe.py --rs 20000 --chip <chip ID>",
            allow=("--i-know" in _sys.argv))
        from knowm_md import MemristorDiscovery
        board = MemristorDiscovery(series_resistance=args.rs,
                                   board_version=BOARD_VERSION)
        board.open()
        board.set_power(True)
        print(f"Hardware mode  Rs={args.rs/1e3:.0f}kΩ  "
              f"{len(devices)} devices  dead band ±{args.tol:.0f}%")
        print("★ The Mode switch must be at 1. One device is connected at a time.")
    print()

    tuner = Tuner(board, tol=tol, v_dev_start=args.vdev, width_us=args.width,
                  max_iters=args.max_iters, v_drive_max=args.vmax,
                  averages=args.averages, settle=0.0 if args.sim else 0.002)

    # ---- print the voltage/current limits explicitly, for checking against the datasheet
    vd, vdev = tuner.v_drive_max, V_DEV_MAX
    i_max = vd / args.rs                    # hard current limit set by the series resistor
    print(f"  Voltage limits: drive ≤ {vd:.2f} V, device ≤ {vdev:.2f} V, "
          f"read pulse {V_READ:.2f} V / {READ_WIDTH_US:.0f} µs")
    print(f"  Current limit: {i_max*1e6:.0f} µA -- set by the {args.rs/1e3:.0f}kΩ series resistor, "
          f"independent of device resistance (I = V_drive/(R+Rs) ≤ V_drive/Rs)")
    if args.vmax > V_DRIVE_ABS_MAX:
        print(f"  ★ --vmax {args.vmax} V exceeds the AD2's ±{V_DRIVE_ABS_MAX:.0f} V output range; "
              f"truncated to {vd:.2f} V")
    elif args.vmax > V_DRIVE_WARN:
        print(f"  ★ --vmax {args.vmax} V is already at hard-erase level; "
              f"confirm the devices can withstand it before running")
    if args.vdev != tuner.v_dev_start:
        print(f"  ★ --vdev {args.vdev} V is outside [{V_DEV_MIN}, {V_DEV_MAX}]; "
              f"clamped to {tuner.v_dev_start:.2f} V")
    print()

    print(f"  {'Dev':>4s}{'Chip':>5s}{'Target':>10s}{'Start':>10s}"
          f"{'Final':>10s}{'Error':>9s}{'Pol':>6s}{'Pulses':>7s}  Result")
    print("  " + "-" * 70)

    recs = []
    t0 = time.time()
    try:
        for i in devices:
            rec = tuner.tune(i, targets[i], autopol=not args.no_autopol)
            recs.append(rec)
            e = rec["err"]
            pol = {1: "pos", -1: "neg", 0: "?"}[rec["polarity"]]
            tag = {"ok": "on target", "stuck": "★stuck", "maxiter": "iterations exhausted",
                   "fail": "★failed"}[rec["status"]]
            print(f"  {i:>4d}{i+1:>5d}{fmt_r(rec['target']):>10s}"
                  f"{fmt_r(rec['r_start']):>10s}{fmt_r(rec['final']):>10s}"
                  f"{f'{e*100:+.1f}%' if e == e else '   --':>9s}"
                  f"{pol:>6s}{rec['pulses']:>7d}  {tag}"
                  + (f" — {rec['note']}" if rec["note"] else ""))
    finally:
        try:
            board.all_off()
            if not args.sim:
                board.set_power(False)
            board.close()
        except Exception:
            pass

    n_ok = sum(1 for r in recs if r["status"] == "ok")
    tot_p = sum(r["pulses"] for r in recs)
    print("  " + "-" * 70)
    print(f"  {n_ok}/{len(recs)} on target, {tot_p} pulses in total, "
          f"took {time.time()-t0:.1f} s")

    flipped = [r["chip"] for r in recs if r["polarity"] == -1]
    if flipped:
        print(f"\n  Devices with polarity opposite to nominal (chip numbers): {flipped}")
        print("  The controller reversed them automatically according to the measured polarity; the result is unaffected.")
    bad = [r for r in recs if r["status"] != "ok"]
    if bad:
        print(f"\n  The {len(bad)} not on target:")
        for r in bad:
            print(f"    chip {r['chip']}: {r['note'] or r['status']}"
                  f" (stopped at {fmt_r(r['final']).strip()}Ω)")
        print(f"  For stuck ones run python {mdp_path('check_ktram.py')} --hard-erase first,")
        print(f"  and if that fails, set Mode to 1 and erase it with one slow sweep of {mdp_path('example_dc.py')}.")

    print()
    print(f"Written {write_csv(recs, args.out + '.csv')}")
    if not args.no_plot:
        p = plot_results(recs, tol, args.out + ".png")
        if p:
            print(f"Written {p}")


if __name__ == "__main__":
    main()
