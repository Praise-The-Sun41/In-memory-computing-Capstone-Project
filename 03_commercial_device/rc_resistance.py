"""
rc_resistance.py — inversion of read-pulse voltage → memristor resistance.

Why it is needed
----------------
On the V2 board the top node of the memristor has about 180 pF of parasitic capacitance to ground, and the scope's 1X probe adds
1 MΩ input impedance in parallel. When the device resistance is high (e.g. 1 MΩ) and the read pulse is short (25 µs),
the node does not have time to charge to the steady-state divider value; the measured voltage is partway through RC charging.
Applying Ohm's law directly, R = V/I, badly underestimates the resistance.

The solution in Knowm's Java project: write the whole board (Rs, Rm, C, two 1 MΩ probes) as a
JSpice netlist MD_V2_Board, run transient simulations sweeping Rm from 10 Ω to 100 MΩ,
record the node voltage at the end of the read pulse, and build a voltage→resistance lookup table
(see RC_ResistanceComputer.java).

The same is done here, but with an analytic solution instead of SPICE -- the circuit is first-order RC, so a closed form suffices,
with no need to pull in a SPICE engine.

Circuit (V2 board, during the read pulse)
-----------------------------------------
        V_p ──[ Rs ]──┬── node ──[ Rm ]── GND
                      ├──[ R_scope=1MΩ ]── GND
                      └──[ C=180pF ]── GND

Thevenin equivalent at the node:
    Rp   = Rm ∥ R_scope
    V∞   = V_p · Rp / (Rs + Rp)
    R_th = Rs ∥ Rp
    τ    = R_th · C
    V(t) = V∞ + (V0 − V∞)·exp(−t/τ)

V(t_pulse) is monotonically increasing in Rm, so a table can be built and inverted by interpolation.
(Note the V0/V1 board topology differs: Rm on top, Rs below, and the node voltage is monotonically decreasing in Rm;
 board_version=2 is the default in this module, other versions use topology='v0v1'.)
"""

from __future__ import annotations

import math

__all__ = ["RCResistanceModel"]


class RCResistanceModel:
    def __init__(
        self,
        series_resistance: float = 10_000.0,
        pulse_amplitude: float = 0.1,
        pulse_width: float = 25e-6,
        capacitance: float = 180e-12,
        scope_resistance: float = 1e6,
        v_init: float = 0.0,
        topology: str = "v2",
        r_min: float = 10.0,
        r_max: float = 1e8,
        points_per_decade: int = 120,
    ):
        self.rs = series_resistance
        self.vp = pulse_amplitude
        self.tw = pulse_width
        self.c = capacitance
        self.rscope = scope_resistance
        self.v0 = v_init
        self.topology = topology

        # build the table: Rm log-uniform
        decades = math.log10(r_max / r_min)
        n = max(50, int(decades * points_per_decade))
        self.r_table = [r_min * (r_max / r_min) ** (k / (n - 1)) for k in range(n)]
        self.v_table = [self.voltage_from_resistance(r) for r in self.r_table]

    # ---------------------------------------------------------------- forward problem
    def voltage_from_resistance(self, rm: float) -> float:
        """Given Rm, compute the node voltage at the end of the read pulse."""
        rs, c, tw, vp, v0 = self.rs, self.c, self.tw, self.vp, self.v0

        if self.topology == "v2":
            # node between Rs and Rm, Rm to ground
            rp = _par(rm, self.rscope)
            v_inf = vp * rp / (rs + rp)
            r_th = _par(rs, rp)
        else:
            # V0/V1: W1 ─ Rm ─ node ─ Rs ─ GND, the node is the top of Rs
            rp = _par(rs, self.rscope)
            v_inf = vp * rp / (rm + rp)
            r_th = _par(rm, rp)

        tau = r_th * c
        if tau <= 0:
            return v_inf
        return v_inf + (v0 - v_inf) * math.exp(-tw / tau)

    # ---------------------------------------------------------------- inverse problem
    def resistance_from_voltage(self, v: float) -> float:
        """Given the measured node voltage, look up Rm (Ω). Outside the table the boundary value is returned."""
        vt, rt = self.v_table, self.r_table
        increasing = vt[-1] > vt[0]

        if increasing:
            if v <= vt[0]:
                return rt[0]
            if v >= vt[-1]:
                return float("inf")
            lo, hi = 0, len(vt) - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if vt[mid] <= v:
                    lo = mid
                else:
                    hi = mid
        else:
            if v >= vt[0]:
                return rt[0]
            if v <= vt[-1]:
                return float("inf")
            lo, hi = 0, len(vt) - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if vt[mid] >= v:
                    lo = mid
                else:
                    hi = mid

        # linear interpolation in the log domain, more accurate when resistance spans several decades
        dv = vt[hi] - vt[lo]
        f = 0.0 if dv == 0 else (v - vt[lo]) / dv
        return math.exp(math.log(rt[lo]) + f * (math.log(rt[hi]) - math.log(rt[lo])))

    # ---------------------------------------------------------------- diagnostics
    def steady_state_error(self, rm: float) -> float:
        """Returns (RC model voltage / ideal steady-state divider) − 1, to judge whether the pulse is long enough."""
        rp = _par(rm, self.rscope)
        ideal = self.vp * rp / (self.rs + rp)
        if ideal == 0:
            return 0.0
        return self.voltage_from_resistance(rm) / ideal - 1.0

    def suggest_pulse_width(self, rm: float, tolerance: float = 0.01) -> float:
        """Recommended minimum pulse width (s) for an Rm reading error below tolerance."""
        rp = _par(rm, self.rscope)
        tau = _par(self.rs, rp) * self.c
        return -tau * math.log(tolerance)


def _par(a: float, b: float) -> float:
    if a == float("inf"):
        return b
    if b == float("inf"):
        return a
    if a + b == 0:
        return 0.0
    return a * b / (a + b)


if __name__ == "__main__":
    m = RCResistanceModel(series_resistance=10_000, pulse_amplitude=0.1, pulse_width=25e-6)
    print(f"{'Rm (Ω)':>12} {'V_node (V)':>12} {'SS error':>10} {'Rm lookup':>12}")
    for rm in (1e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6):
        v = m.voltage_from_resistance(rm)
        print(f"{rm:12.0f} {v:12.6f} {m.steady_state_error(rm):9.1%} "
              f"{m.resistance_from_voltage(v):12.0f}")
    print()
    for rm in (1e5, 1e6, 1e7):
        print(f"For Rm={rm:.0e} Ω at 1% accuracy, the pulse width must be at least "
              f"{m.suggest_pulse_width(rm)*1e6:.1f} µs")
