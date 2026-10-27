"""
demolib.py —— the layer shared by the four live demos

Every demo has the same three-act structure with a different task:

    Act 1  What this chip looks like    -- measured windows, per-device quirks, how coarse write control is
    Act 2  It learns the task           -- on-chip learning; the weights exist only in the devices throughout
    Act 3  Damage devices publicly      -- performance collapses, then recovers using the remaining devices

=================== Where the device characteristics come from (do not delete this section) ===================

`sdc_model.MEASURED_WINDOWS` comes from the hardware sweep of 2026-09-19, when
**all 16 devices switched back and forth normally** -- every trace had full high and low plateaus.
So these characteristics are "what the devices really look like when working", not when broken:

    Common conductance window   231 – 668 µS, only 2.9×
    Of the 16                   4 respond slowly, 5 vary strongly from cycle to cycle
    High-conductance end        690 ~ 1563 Ω

That sweep needed about 455 µA to move the devices -- inside the irreversible damage range of manual Note 1.
**The devices were later damaged at that operating point.** That is what act 3 demonstrates: not a hypothetical
fault model, but a pitfall we actually hit. `device_limits.py` now locks the compliance current at
100 µA, at the cost that these resistances cannot be driven with a 10 kΩ series resistor and the hardware must change
(the manual's own endurance test uses 50 kΩ).

=================== Why write control has to be coarse ===================

A closed loop "write once, read, top up if needed" is impossible on the real board: one read takes several hundred
microseconds, and the read itself returns inf or a negative value 6% of the time at the high-conductance end. So writing is
open loop, "fire one pulse in a direction and move on", and every pulse is imprecise:

    write_jitter 0.6   step size is log-normally random, differing by up to 3×
    write_miss   0.15  15% of pulses have no effect at all
    write_flip   0.05  5% of pulses go in the wrong direction

All four demos have a `--clean-write` control (setting these three to zero). **The conclusion is that it makes little difference** --
which is exactly the point worth making about on-chip learning: it does not need precise writes, only roughly correct direction.
The first demo (reinforcement learning) goes further: the "imprecision" itself is the exploration noise, which a digital chip
would have to spend area on a random number generator to obtain.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import device_limits as L                                  # noqa: E402
import sdc_model as M                                      # noqa: E402
import safety as _S                                        # noqa: E402

_SAFE_I = _S.I_SAFE          # 50 µA -- the real-hardware envelope. The simulation is not constrained by it.

N_DEV = 16

# default trio for coarse writing. Shared by the four demos; change it here to change all.
COARSE = dict(write_jitter=0.60, write_miss=0.15, write_flip=0.05)
CLEAN = dict(write_jitter=0.0, write_miss=0.0, write_flip=0.0)


# ================================================================ command line
def base_parser(desc: str) -> argparse.ArgumentParser:
    """Switches shared by the four demos. Each adds its own with add_argument."""
    ap = argparse.ArgumentParser(
        description=desc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rs", type=float, default=10_000.0,
                    help="Series resistance Ω. The default 10 kΩ is the condition of **the earlier chip sweep**, "
                         "not the operating point of the new chip (new chip ≥ 20 kΩ, see safety.py)")
    ap.add_argument("--i-limit", type=float, default=455.0,
                    help="Compliance current µA. Default 455 = **the operating point that damaged the previous chip**. "
                         "It is kept only so that the simulation reproduces the numbers of that time; "
                         "real hardware goes through cartpole_hw.py, which does not read this parameter at all.")
    ap.add_argument("--reads", type=int, default=3,
                    help="Number of repeats per read; the median is taken (immune to bad reads)")
    ap.add_argument("--width", type=float, default=5.0,
                    help="Write pulse width µs. The model is calibrated at 50 µs; smaller = finer steps")
    ap.add_argument("--clean-write", action="store_true",
                    help="Control: precise write control (jitter/miss/flip off)")
    ap.add_argument("--kill", type=int, default=4,
                    help="Number of devices damaged by over-current in act 3")
    ap.add_argument("--no-kill", dest="kill", action="store_const", const=0)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    return ap


def make_board(args, **kw):
    """
    Build a **simulated** chip according to the command-line switches.

    ★ What is built here is always `sdc_model.SDCArray` -- pure simulation, never touching real devices.
      It is calibrated to **the previous chip, which has already been damaged** (MEASURED_WINDOWS = 231–668 µS
      = 1.5–4.3 kΩ, the whole window below the new safety floor of 13 kΩ).

      So the numbers of this module are a "replay of the old chip", not expectations for the new chip.
      Real hardware goes through `demos/01_cartpole/cartpole_hw.py` → `hw_session.open_board()`,
      which has the protection gate and takes from the chip record the new window measured by
      `characterize_safe.py` within the 50 µA envelope.
    """
    p = dict(CLEAN if args.clean_write else COARSE)
    p.update(kw)
    return M.SDCArray(rs=args.rs, seed=args.seed, **p)


def drive_voltage(args) -> float:
    """
    Maximum drive amplitude allowed by this compliance current (worst case R→0, then clipped to the AWG range).

    ★ **For simulation only.** Default 455 µA × 10 kΩ = 4.55 V -- the operating point that pushed the previous chip
      into the irreversible region. On real hardware the amplitude is issued by safety.SafetyGate,
      not chosen by the caller, and this function is not used.
    """
    v = L.max_drive(args.rs, args.i_limit * 1e-6, L.V_DEV_MAX_DEFAULT)
    if args.i_limit * 1e-6 > _SAFE_I:
        import warnings
        warnings.warn(
            f"drive_voltage(): compliance current {args.i_limit:.0f} µA exceeds the safety envelope of "
            f"{_SAFE_I*1e6:.0f} µA. This is fine in simulation (the devices are not real), "
            f"but this number must **never** be used to drive a real board.", stacklevel=2)
    return v


# ================================================================ act 1
def act1(args, extra=()):
    """Lay out the measured characteristics of this chip. All four demos print this at the start."""
    lo = np.array([a for a, _ in M.MEASURED_WINDOWS])
    hi = np.array([b for _, b in M.MEASURED_WINDOWS])
    rows, alive = M.switching_headroom(args.rs, 100e-6)
    v100 = np.median([r["v_at_hi"] for r in rows]) * 1000
    need = max(r["i_needed"] for r in rows) * 1e6

    out = ["", "[Act 1] What this chip looks like when measured"
                " (sweep of 2026-09-19, when all 16 devices switched normally)"]
    out.append(f"  Common conductance window  {lo.max():.0f} – {hi.min():.0f} µS, "
               f"only **{hi.min()/lo.max():.1f}×** (5~37× per device)")
    out.append(f"  High-conductance resistance {1e6/hi.max():.0f} – {1e6/hi.min():.0f} Ω")
    out.append(f"  Quirks             4 respond slowly · 5 vary strongly between cycles · "
               f"about 6% of reads at the high-conductance end return inf or negative values")
    out.append(f"  That sweep needed {args.i_limit:.0f} µA to move the devices. "
               f"At the manual's safe value of 100 µA a device gets only {v100:.0f} mV, "
               f"while the threshold needs {L.V_TH_FWD['W']*1000:.0f} mV ⇒ **{alive}/16 can be moved**.")
    out.append(f"  → The hardest one needs {need:.0f} µA. 455 µA lies in manual Note 1's "
               f"irreversible damage range -- **act 3 shows exactly this cost.**")
    if args.clean_write:
        out.append("  Write control      **control: precise** (--clean-write)")
    else:
        out.append(f"  Write control      open loop, fire and forget: step size random ×3 · "
                   f"{COARSE['write_miss']*100:.0f}% of pulses lost · "
                   f"{COARSE['write_flip']*100:.0f}% in the wrong direction")
    out.extend(f"  {t}" for t in extra)
    out.append(f"  The demo runs on a simulation calibrated to measurements (sdc_model.py); the real board is not touched.")
    return "\n".join(out)


# ================================================================ readout
class Readout:
    """
    Turn "the parallel conductance of one branch" into "a weighted sum of signed weights".

        u(P) = G_P − c·|P| = Σ_{i∈P} (g_i − c)

    Conductance can only be positive, so a single device cannot represent a negative weight. Subtracting a **common mode** c
    fixes that: c is the measured mean conductance, so Σ_i w_i ≡ 0 and the weights are automatically zero-sum. This step also
    cancels common-mode errors such as temperature drift and contact resistance -- they affect G_P and c equally.

    ★ One parallel read returns the conductance of the whole branch, not 16 numbers. So this is
      **true analog multiply-accumulate**: the currents of |P| devices add on the wire, with no multiplication.
    """

    def __init__(self, board, n_read=3, n_dev=N_DEV, width_us=5.0):
        self.b = board
        self.n_read = n_read
        self.n = n_dev
        # Write pulse width -- this sets "how fine one pulse step is". The model is calibrated at 50 µs,
        # where one pulse moves 15~26% of the log window: too coarse for the weights to settle at the optimum.
        self.width_us = width_us
        self.c = 1e-4 / n_dev
        self.refresh_c()

    # ---- read ----
    def _par(self):
        """Conductance of the currently selected branch. Read n_read times and take the median."""
        v = []
        for _ in range(self.n_read):
            r = self.b.read_parallel()
            if math.isfinite(r) and r > 1e-3:
                v.append(1.0 / r)
        if not v:
            return self.c * self.n          # all bad: fall back to the common mode, better not to move
        return float(np.median(v))

    def refresh_c(self):
        """c = measured mean conductance. One parallel read of all devices, divided by 16."""
        self.b.select(*range(self.n))
        self.c = self._par() / self.n

    def u(self, on):
        """Weighted sum. Returns 0 if on is empty."""
        on = list(on)
        if not on:
            return 0.0
        self.b.select(*on)
        return self._par() - self.c * len(on)

    # ---- write ----
    def push(self, on, sign, v_drive, bulk=False):
        """
        Push the devices in on one step in one direction.

        ★ By default devices are written **one at a time**, not as a batch. The reason is hard: in a batch write the device
          voltage is set by the **parallel** resistance; |P| devices in parallel divide the resistance by |P|, and the device
          voltage is divided by |P| too. At the high-conductance end 4 devices in parallel are only ~400 Ω, and at 455 µA a
          device gets only 0.18 V, below the 0.26 V threshold -- the whole branch cannot be written.
          Writing one at a time, R∥Rs ≈ 909 Ω, and the same current gives 0.41 V, which is enough.

          `bulk=True` is kept as a control: it saves |P|× pulses and looks attractive,
          but once a device is shorted at the high-conductance end in act 3, the whole branch can no longer be written.
        """
        on = list(on)
        if not on:
            return 0
        if bulk:
            self.b.select(*on)
            self.b.pulse(amplitude=sign * v_drive, n=1,
                         width_us=self.width_us)
            return 1
        for i in on:
            self.b.select(i)
            self.b.pulse(amplitude=sign * v_drive, n=1,
                         width_us=self.width_us)
        return len(on)

    def weights(self):
        """Effective weight w = g − c. Used only for plotting; never called during learning."""
        return self.b.conductances() - self.c


class FaultTolerantReadout(Readout):
    """
    `Readout` plus one thing: **knowing which of its own devices are already dead**.

    After act 3 damages devices by over-current, they are stuck at the high-conductance end. Their conductances enter every
    parallel read and can no longer be written -- so the learning algorithm gets a read contaminated by a fixed bias,
    and keeps trying to push devices that cannot move.

    The fix is direct, and is what a real system would do: **check-up → record → subtract from the read**.

        u(P) = G_P − Σ_{i∈P and dead} ĝ_i − c·|alive members of P|

    ĝ_i is the stuck value measured in the check-up; after subtraction the remaining devices face a clean problem.
    Note this is **not** closed-loop write verification (which is impossible), just a one-off read + record.

    Measured: after 4 devices were damaged by over-current, with the check-up learning recovers 117 steps, without it only 47.
    """

    def __init__(self, *a, **kw):
        self.known = {}
        super().__init__(*a, **kw)

    def refresh_c(self):
        self.b.select(*range(self.n))
        tot = self._par()
        alive = self.n - len(self.known)
        self.c = (tot - sum(self.known.values())) / max(alive, 1)

    def u(self, on):
        on = list(on)
        if not on:
            return 0.0
        self.b.select(*on)
        tot = self._par()
        fixed = sum(self.known.get(i, 0.0) for i in on)
        alive = sum(1 for i in on if i not in self.known)
        return tot - fixed - self.c * alive

    def read_one(self, i, n=3):
        """Conductance of a single device, median. Bad reads are discarded."""
        v = []
        for _ in range(n):
            self.b.select(i)
            r = self.b.read_resistance()
            if math.isfinite(r) and r > 1e-3:
                v.append(1.0 / r)
        return float(np.median(v)) if v else None

    def diagnose(self, v_drive, thr=0.05, n_read=5):
        """
        Per-device check-up: push once in each direction, and judge dead only if **it moves in neither**.

        Why both directions are needed: a device that is alive but already at one end will of course not move when
        pushed towards that end -- trying only one direction would misjudge it as dead.
        Cost: 16 × (4 pulses + 4 reads), negligible among thousands of reads.
        """
        dead = {}
        for i in range(self.n):
            g0 = self.read_one(i, n_read)
            if g0 is None:
                continue
            self.b.select(i); self.b.pulse(amplitude=+v_drive, n=3,
                                           width_us=50.0)
            g1 = self.read_one(i, n_read)
            self.b.select(i); self.b.pulse(amplitude=-v_drive, n=4,
                                           width_us=50.0)
            g2 = self.read_one(i, n_read)
            self.b.select(i); self.b.pulse(amplitude=+v_drive, n=2,
                                           width_us=50.0)
            g3 = self.read_one(i, n_read)
            if None in (g1, g2, g3):
                continue
            move = max(abs(math.log(max(g1, 1e-12) / g0)),
                       abs(math.log(max(g2, 1e-12) / max(g1, 1e-12))))
            if move < thr:
                dead[i] = float(np.median([g0, g1, g2, g3]))
        self.known = dead
        self.refresh_c()
        return dead


def group_by_window(n_groups=4, windows=None):
    """
    Group the 16 devices **so that devices with similar windows go in the same group**. Returns an index table of length 16:
    which physical device is used at position j of group k.

    Why: after erasing, every device sits at its own low-conductance end, and these ends range from 26 µS to 231 µS,
    a factor of 9. If numbered arbitrarily, weights w = g − c within a group differ widely from the start --
    **device process variation masquerades as a "learned prior"**, and an untrained chip may solve the task straight away
    (measured: in 4 of 10 seeds an untrained chip held the full 200 steps, entirely an artefact).

    Grouping similar windows together makes weights within a group nearly equal after erasing, so the policy is a blank slate.
    Measured, this alone cuts "before learning" from an average of 115 steps to 15~20 -- only then is the demo honest.

    On the real board this table comes from window_devices.csv produced by `check_window.py`.
    """
    w = M.MEASURED_WINDOWS if windows is None else windows
    return np.argsort([a for a, _ in w])


# ================================================================ act 3
def burn(board, n_kill, rng, mode="short"):
    """
    Act 3: damage a few devices by over-current in front of the audience.

    The failure mode follows the manual's Critical Notes:
      "A forward applied voltage **without current limiting will cause devices
       to enter a very low-resistance state and consequently burn out**."
    -- stuck at the **low-resistance/high-conductance** end and no longer writable. `sdc_model.kill(mode="short")`
    implements exactly this. This is not an invented failure model; it is the one we actually ran into.

    Note why it is harsher than "just freezing": a shorted device has maximal conductance, c rises overall,
    **all** weights w = g − c shift together, and the whole learned decision boundary is skewed.
    """
    dead = sorted(rng.choice(N_DEV, n_kill, replace=False).tolist())
    board.kill(*dead, mode=mode)
    return dead


def act3_header(dead, n_left):
    return (f"\n[Act 3] Damage devices {[d+1 for d in dead]} by over-current in front of the audience"
            f" (manual Note 1: stuck at the low-resistance end, no longer writable)\n"
            f"  This is not a hypothetical fault -- 455 µA is the current we used ourselves to produce that plot.\n"
            f"  The weights have no software copy; they can only be recovered by the remaining {n_left} devices.")


# ================================================================ figures
PALETTE = dict(
    surf="#fcfcfb", ink="#0b0b0b", ink2="#52514e", ink3="#8a8a84",
    grid="#e6e5e1", good="#0ca30c", warn="#7b4fd1", crit="#d03b3b",
    cool="#2a6fd6",
)


def figure(w=12.5, h=4.3):
    """Common canvas + fonts. Returns (plt, fig, T, palette). Returns None if matplotlib is not installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed; skipping figures)")
        return None
    try:
        from plotstyle import setup, T
        setup(verbose=False)
    except ImportError:
        def T(zh, en):
            return en
    fig = plt.figure(figsize=(w, h), facecolor=PALETTE["surf"])
    return plt, fig, T, PALETTE


def tidy(ax, P=PALETTE, yaxis=True):
    ax.tick_params(labelsize=8.5, colors=P["ink2"])
    if yaxis:
        ax.grid(True, axis="y", color=P["grid"], lw=.8)
        ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(P["ink3"])


def title(ax, s, P=PALETTE):
    ax.set_title(s, loc="left", fontsize=10.5, fontweight="600",
                 color=P["ink"], pad=8)


def head(fig, h, sub, P=PALETTE, x=.065):
    fig.text(x, .93, h, fontsize=14, fontweight="bold", color=P["ink"])
    fig.text(x, .875, sub, fontsize=9.5, color=P["ink2"])


def weight_map(ax, w, dead=(), shape=(4, 4), fmt="{:+.0f}", P=PALETTE):
    """4×4 weight heat map; damaged devices are crossed out."""
    v = np.asarray(w, float).reshape(shape) * 1e6
    lim = float(np.nanmax(np.abs(v))) or 1.0
    ax.imshow(v, cmap="RdBu_r", vmin=-lim, vmax=lim)
    for i in range(shape[0]):
        for j in range(shape[1]):
            ax.text(j, i, fmt.format(v[i, j]), ha="center", va="center",
                    fontsize=8,
                    color="w" if abs(v[i, j]) > lim * .55 else P["ink2"])
    for d in sorted(dead):
        ax.plot(d % shape[1], d // shape[1], "x", color="#000", ms=13, mew=2.5)
    ax.set_xticks([])
    ax.set_yticks([])


def save(fig, path, P=PALETTE):
    fig.savefig(path, dpi=150, facecolor=P["surf"])
    print(f"\nWritten {os.path.abspath(path)}")
