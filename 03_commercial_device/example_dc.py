"""
example_dc.py — slow DC sweep (the GUI's DC experiment)

Difference from the hysteresis experiment
-----------------------------------------
`example_hysteresis.py` uses kHz sine/triangle waves and looks at **frequency-dependent** hysteresis loops --
how wide the loop opens depends on the sweep rate.

The DC experiment stretches the period to the ms range (the GUI allows 10–1000 ms), slow enough that
the device state is close to equilibrium at every point, so what you see is **quasi-static switching**:
set threshold, reset threshold, R_on / R_off window. These are the numbers for a datasheet.

Board setup: V2 board, Mode switch at "1".

    python example_dc.py
"""

import csv
import math

import sys
import scope_guard

from knowm_md import MemristorDiscovery

SERIES_R = 10_000.0     # Ω -- A∥B in Mode 1. Must match the board!
DEVICE = 0              # 0..15
AMPLITUDE = 0.5         # V peak. Start with 0.3~0.5, confirm there is no hard breakdown, then increase
PERIOD_MS = 100.0       # one ramp period. Slower is closer to quasi-static
RAMPS = 2               # number of cycles
SHAPE = "triangleupdown"   # 0 → +A → −A → 0
SAMPLES_PER_RAMP = 2000    # total points = 2000×2 = 4000, still within the 8192 single buffer


# ---------------------------------------------------------------- helpers
def moving_average(x, w=15):
    """Centred moving average; NaN is skipped (neither used nor contaminating neighbours)."""
    if w <= 1:
        return list(x)
    h, n, out = w // 2, len(x), []
    for k in range(n):
        acc, cnt = 0.0, 0
        for j in range(max(0, k - h), min(n, k + h + 1)):
            v = x[j]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            acc += v; cnt += 1
        out.append(acc / cnt if cnt else float("nan"))
    return out


def _isnan(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def split_branches(v):
    """Split one cycle into monotonic segments by the rising/falling direction of the applied voltage."""
    dv = moving_average([v[k + 1] - v[k] for k in range(len(v) - 1)] + [0.0], 21)
    branches, start, sign = [], 0, 0
    for i, d in enumerate(dv):
        s = 1 if d > 0 else (-1 if d < 0 else sign)
        if sign == 0:
            sign = s
        elif s != sign and s != 0:
            if i - start > 20:
                branches.append((start, i, sign))
            start, sign = i, s
    if len(v) - start > 20:
        branches.append((start, len(v), sign))
    return branches


def find_threshold(v, g, lo, hi, polarity, min_v=0.05):
    """
    Find the voltage at which the conductance changes most steeply within [lo,hi).
    polarity=+1 searches the positive side (set), −1 the negative side (reset).

    Both points must be valid conductances (boundaries of NaN-masked regions create huge false slopes,
    so the whole pair must be skipped rather than treating NaN as 0).
    """
    best_v, best_slope = None, 0.0
    for k in range(lo + 1, hi):
        if polarity * v[k] <= min_v:
            continue
        if _isnan(g[k]) or _isnan(g[k - 1]):
            continue
        dv = v[k] - v[k - 1]
        if abs(dv) < 1e-9:
            continue
        slope = (g[k] - g[k - 1]) / dv
        if abs(slope) > abs(best_slope):
            best_slope, best_v = slope, v[k]
    return best_v, best_slope


# ---------------------------------------------------------------- main flow
def main():
    # ---- scope gate: this script is no longer within current scope (inverted pendulum / H/V 3×3 bars) ----
    scope_guard.refuse_hardware(
        'Slow DC sweep example (example_dc.py)',
        why='The path of manual section 12.4 Method 1: large-amplitude long sweeps. It constructs MemristorDiscovery directly, with no preflight, no gate and no resistance floor',
        suggest='python characterize_safe.py --rs 20000 --chip <chip ID>',
        allow=('--i-know' in sys.argv))

    with MemristorDiscovery(series_resistance=SERIES_R, board_version=2) as md:
        md.select(DEVICE)

        r_before = md.read_resistance()
        print(f"Resistance before sweep: {r_before/1000:.1f} kΩ")

        t, vm, i = md.sweep_dc(
            amplitude=AMPLITUDE,
            period_ms=PERIOD_MS,
            ramps=RAMPS,
            shape=SHAPE,
            samples_per_ramp=SAMPLES_PER_RAMP,
        )

        r_after = md.read_resistance()
        md.all_off()

    print(f"Resistance after sweep: {r_after/1000:.1f} kΩ  "
          f"(change {(r_after/r_before - 1)*100:+.1f}%)")

    # ---- conductance. G = I/V amplifies noise beyond use at low voltage, so mask it as NaN.
    #      Do not fill the masked region with 0 -- that creates a false huge dG/dV at the mask boundary,
    #      and threshold extraction would consistently misplace it near 0 V.
    v_floor = 0.15 * AMPLITUDE
    g = [(cur / v if abs(v) > v_floor else float("nan")) for v, cur in zip(vm, i)]
    g_s = moving_average(g, 15)

    # ---- branches and thresholds
    branches = split_branches(vm)
    print(f"\nFound {len(branches)} monotonic branches:")
    v_set = v_reset = None
    for k, (lo, hi, sign) in enumerate(branches):
        seg = vm[lo:hi]
        direction = "rising" if sign > 0 else "falling"
        print(f"  segment {k}: {direction}  {min(seg):+.3f} → {max(seg):+.3f} V"
              f"  ({hi-lo} points)")
        vs, ss = find_threshold(vm, g_s, lo, hi, +1, v_floor)
        vr, sr = find_threshold(vm, g_s, lo, hi, -1, v_floor)
        if vs is not None and (v_set is None or abs(ss) > v_set[1]):
            v_set = (vs, abs(ss))
        if vr is not None and (v_reset is None or abs(sr) > v_reset[1]):
            v_reset = (vr, abs(sr))

    print()
    if v_set:
        print(f"set threshold   ≈ {v_set[0]:+.3f} V   (|dG/dV| = {v_set[1]*1e6:.1f} µS/V)")
    if v_reset:
        print(f"reset threshold ≈ {v_reset[0]:+.3f} V   (|dG/dV| = {v_reset[1]*1e6:.1f} µS/V)")

    valid = [x for x in g if not math.isnan(x) and x > 0]
    if valid:
        print(f"R window        : {1/max(valid)/1000:.1f} kΩ (R_on) "
              f"… {1/min(valid)/1000:.1f} kΩ (R_off)   "
              f"ratio {max(valid)/min(valid):.1f}×")

    # ---- CSV
    with open("dc_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "V_V", "I_A", "I_uA", "G_S", "R_ohm"])
        for tt, v, cur, gg in zip(t, vm, i, g):
            r = "" if (math.isnan(gg) or gg == 0) else f"{1/gg:.6g}"
            w.writerow([f"{tt:.9g}", f"{v:.6g}", f"{cur:.6g}",
                        f"{cur*1e6:.4g}", "" if math.isnan(gg) else f"{gg:.6g}", r])
    print("\nWritten dc_sweep.csv")

    # ---- figure
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib to get the figure directly: pip install matplotlib")
        return
    from plotstyle import setup, T
    setup()

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    ax[0].plot(t, vm, lw=1)
    ax[0].set_xlabel("t (s)"); ax[0].set_ylabel("V (V)")
    ax[0].set_title(f"{T('Drive waveform', 'Drive waveform')} {SHAPE} / {PERIOD_MS:g} ms")
    ax[0].grid(alpha=.3)

    seg_label = T("seg", "seg ")
    for k, (lo, hi, sign) in enumerate(branches):
        ax[1].plot(vm[lo:hi], [x * 1e6 for x in i[lo:hi]], lw=1,
                   label=f"{seg_label}{k} {'^' if sign > 0 else 'v'}")
    ax[1].set_xlabel("V (V)"); ax[1].set_ylabel("I (uA)")
    ax[1].set_title(f"I-V  device {DEVICE+1}  +/-{AMPLITUDE} V")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    ax[1].axhline(0, c="k", lw=.5); ax[1].axvline(0, c="k", lw=.5)

    ax[2].plot(vm, [x * 1e6 for x in g_s], lw=1)
    if v_set:
        ax[2].axvline(v_set[0], c="C3", ls="--", lw=1, label=f"set {v_set[0]:+.2f} V")
    if v_reset:
        ax[2].axvline(v_reset[0], c="C0", ls="--", lw=1, label=f"reset {v_reset[0]:+.2f} V")
    ax[2].set_xlabel("V (V)"); ax[2].set_ylabel("G (uS)")
    ax[2].set_title(T("G-V and thresholds", "G-V and thresholds"))
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig("dc_sweep.png", dpi=150)
    print("Written dc_sweep.png")
    plt.show()


if __name__ == "__main__":
    main()
