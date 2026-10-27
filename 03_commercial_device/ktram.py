"""
ktram.py — Python reimplementation of Knowm's 8-synapse kT-RAM core (Classify12 / kTBitSatSolver).

Java counterparts:
    gui/mvc/experiments/classify12/KTRAM_Controller_12.java
    gui/mvc/experiments/classify12/Classify12Preferences.java   (Datasets)
    gui/mvc/experiments/classify12/Classify12Experiment.java    (three learning routines)
    gui/mvc/experiments/ktbitsatsolver/*                        (3SAT solver)

=================== Hardware structure ===================

V2 board, Mode 2; the 16 devices of the 1×16 chip are paired into 8 differential pairs:

    synapse k  =  (chip device k+1, chip device k+9)  =  (DIO k, DIO k+8)     k = 0..7

    Java basis: the two lines in KTRAM_Controller_12.execute()
        update2DigitalIOStatesAtOnce(spikePattern, true)      // side A, no offset
        update2DigitalIOStatesAtOnce(spikePattern, 8, true)   // side B, offset 8

Circuit (Mode 2):

    W1 ──┬──[ Rs ]── node A (scope 1+) ──┬─[Ma0]─┬─[Ma1]─ …  ── GND
         │                                 (only switches in the pattern are closed)
         └──[ Rs ]── node B (scope 2+) ──┬─[Mb0]─┬─[Mb1]─ …  ── GND

=================== Key point: a pattern is parallel, not a loop ===================

A spike pattern is a subset of {0..7}. When any instruction is executed,
the switches of **all** synapses in the pattern close at once, sharing one W1 and one pair of series resistors.
So the conductances of the active devices on side A add directly in parallel:

    G_A = Σ_{i∈pattern} ga_i          G_B = Σ_{i∈pattern} gb_i

Readout (FLV):

    V_A = V_W1 / (1 + Rs·G_A)         V_B = V_W1 / (1 + Rs·G_B)
    y   = −(V_A − V_B) / V_W1  =  1/(1+Rs·G_B) − 1/(1+Rs·G_A)

    y > 0  ⟺  G_A > G_B  ⟺  classified "True"

**The summation is done by Kirchhoff's current law, not a software loop.** Whether the pattern has
1 synapse or 8, only one read pulse is sent. This is the core advantage claimed for kT-RAM:
one read completes the whole dot product.

Writing is parallel too: under FA all side-A devices in the pattern receive the write pulse at once.
A side effect is that the voltage each device gets, V_A = V_W1/(1+Rs·G_A), falls as the number of active synapses grows:
**the more synapses are active, the weaker the update of each device** -- a built-in learning-rate normalisation
provided by physics. There is no line of code for it in software; it is in the circuit.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from safety import SafetyViolation as _Sv

__all__ = [
    "Instruction", "Pattern", "DATASETS", "ROUTINES",
    "KTRAM", "SimBackend", "HardwareBackend", "train", "solve_sat",
    "calibrate_offsets", "diagnose_read", "diagnose_write",
    "hard_reset", "hard_reset_all", "STUCK_G", "dio_masks", "mask_for",
]

N_SYNAPSES = 8
V_READ = 0.08          # FLV / RLV amplitude, hard-coded as ±0.08 V in Java


# ---------------------------------------------------------------- instructions
class Instruction:
    FLV = "FLV"          # forward low-voltage read (does not change state)
    RLV = "RLV"          # reverse low-voltage read
    FA = "FA"            # forward pulse on side A only → ga↑ → y↑
    FB = "FB"            # side B only          → gb↑ → y↓
    RA = "RA"            # reverse pulse on side A only → ga↓ → y↓
    RB = "RB"            # side B only          → gb↓ → y↑
    FAB = "FAB"          # A and B forward together (both rise, y nearly unchanged; used for initialisation)
    RAB = "RAB"          # A and B reverse together


@dataclass
class Pattern:
    """A spike pattern: which synapses are active + the supervision label."""
    spikes: tuple[int, ...]
    state: bool = False           # True = expect y>0

    def __repr__(self):
        return f"{list(self.spikes)}->{'T' if self.state else 'F'}"


def _P(state, *spikes):
    return Pattern(tuple(spikes), state)


# ---------------------------------------------------------------- data sets
# one-to-one with Classify12Preferences.Datasets. Indices are 0-based (switches in the GUI are 1-based).
DATASETS: dict[str, list[Pattern]] = {
    # two disjoint 4-tuples -- the easiest case
    "Ortho2": [_P(True, 0, 1, 2, 3), _P(False, 4, 5, 6, 7)],
    "AntiOrtho2": [_P(False, 0, 1, 2, 3), _P(True, 4, 5, 6, 7)],

    # four disjoint pairs
    "Ortho4": [_P(True, 0, 1), _P(True, 2, 3), _P(False, 4, 5), _P(False, 6, 7)],
    "AntiOrtho4": [_P(False, 0, 1), _P(False, 2, 3), _P(True, 4, 5), _P(True, 6, 7)],

    # eight single synapses -- each learns its own label independently
    "Ortho8": [_P(True, i) for i in range(4)] + [_P(False, i) for i in range(4, 8)],
    "AntiOrtho8": [_P(False, i) for i in range(4)] + [_P(True, i) for i in range(4, 8)],

    # sliding window; adjacent patterns share synapses → frustrated
    # note the last one [7,8]: 8 is outside 0..7, see the BUG note below
    "TwoEightPattern5Frustrated": [
        _P(True, 0, 1), _P(True, 1, 2), _P(True, 2, 3), _P(True, 3, 4),
        _P(False, 4, 5), _P(False, 5, 6), _P(False, 6, 7), _P(False, 7, 8),
    ],
    # ring version with the out-of-range index fixed (8 → 0), for comparison
    "TwoEightRingFixed": [
        _P(True, 0, 1), _P(True, 1, 2), _P(True, 2, 3), _P(True, 3, 4),
        _P(False, 4, 5), _P(False, 5, 6), _P(False, 6, 7), _P(False, 7, 0),
    ],

    # two 5-tuples sharing synapses 2 and 5
    "TwoPattern36Frustrated": [
        _P(True, 0, 1, 2, 3, 5), _P(False, 4, 5, 6, 7, 2),
    ],
}

DATASET_NOTES = {
    "TwoEightPattern5Frustrated":
        "Index 8 of the last pattern is outside 0..7. Java translates it into DIO 8 "
        "(= the B-side device of synapse 0, connected alone without its A-side partner) plus DIO 16 (the AD2 has only 16 lines, so it is dropped). "
        "As a result this F pattern carries an extra bare device that only adds conductance to branch B, naturally pulling y negative -- "
        "which happens to help the F label. At the same time the B side of synapse 0 receives FB/RB updates with this pattern.",
}


# ---------------------------------------------------------------- learning routines
# the three private methods of Classify12Experiment.
def learn_always(ktram, pattern, y):
    """Update unconditionally by label. Equivalent to supervised Hebbian with fixed step."""
    if pattern.state:
        ktram.execute(pattern, Instruction.FA)
        ktram.execute(pattern, Instruction.RB)
    else:
        ktram.execute(pattern, Instruction.RA)
        ktram.execute(pattern, Instruction.FB)


def learn_on_mistakes(ktram, pattern, y):
    """Update only on mistakes -- the perceptron learning rule."""
    if y < 0 and pattern.state:
        ktram.execute(pattern, Instruction.FA)
        ktram.execute(pattern, Instruction.RB)
    elif y > 0 and not pattern.state:
        ktram.execute(pattern, Instruction.RA)
        ktram.execute(pattern, Instruction.FB)


def learn_combo(ktram, pattern, y):
    """Reinforce the positive class unconditionally, correct the negative class only on mistakes. Asymmetric, favouring the positive class."""
    if pattern.state:
        ktram.execute(pattern, Instruction.FA)
        ktram.execute(pattern, Instruction.RB)
    elif y > 0:
        ktram.execute(pattern, Instruction.RA)
        ktram.execute(pattern, Instruction.FB)


ROUTINES = {
    "LearnAlways": learn_always,
    "LearnOnMistakes": learn_on_mistakes,
    "LearnCombo": learn_combo,
}


# ---------------------------------------------------------------- DIO masks
def dio_masks(spikes) -> tuple[int, int]:
    """
    pattern → (side-A call mask, side-B call mask), with exactly the same semantics as Java.

    Java makes two calls at the start of each instruction:
        update2DigitalIOStatesAtOnce(spikes, true)      → set (i)      ← side-A call
        update2DigitalIOStatesAtOnce(spikes, 8, true)   → set (i + 8)  ← side-B call
    then FA/RA undo the side-B call and FB/RB undo the side-A call.

    Out-of-range indices (such as the 8 in TwoEightPattern5Frustrated) show up here:
    the side-A call turns it into DIO 8 -- the B-side device of synapse 0;
    the side-B call turns it into DIO 16 -- the AD2 has only 16 lines, so it is dropped.
    This function reproduces that behaviour as is, without "fixing" it, so simulation and hardware agree.
    """
    ma = mb = 0
    for i in spikes:
        if 0 <= i < 16:
            ma |= 1 << i
        if 0 <= i + 8 < 16:
            mb |= 1 << (i + 8)
    return ma, mb


def mask_for(instruction: str, spikes) -> int:
    ma, mb = dio_masks(spikes)
    if instruction in (Instruction.FA, Instruction.RA):
        return ma
    if instruction in (Instruction.FB, Instruction.RB):
        return mb
    return ma | mb


# ---------------------------------------------------------------- back ends
class SimBackend:
    """
    Offline simulation back end -- for verifying algorithm logic, not a device model.

    Each device's conductance lies in [g_min, g_max]; forward pulses approach the upper bound multiplicatively
    and reverse pulses the lower bound, with the step modulated by the voltage the device actually gets (this is the key:
    it reproduces the physical normalisation "the more synapses active, the weaker each device's update").
    """

    def __init__(self, rs=20_000.0, g_min=1e-6, g_max=1e-4,
                 eta=0.25, v_th=0.25, noise=0.02, seed=0, g_init=1.2e-5):
        """
        Default parameters are of the order of Knowm SDC devices: R_on ≈ 10 kΩ (g_max), R_off ≈ 1 MΩ (g_min),
        write threshold ≈ 0.25 V, Rs = 20 kΩ, write amplitude 1 V.

        g_init is the power-up value (≈ 83 kΩ, like a freshly erased chip). **The initial value matters**:
        if all devices start at the low-resistance end, then with several devices of a pattern in parallel
        the voltage each gets after division falls below threshold, and no pulse can write anything again --
        the whole synapse array locks up. This is not a flaw of the simulation; real hardware behaves this way,
        and it is what Knowm's documentation means by "both devices go low resistance = the synapse is dead".
        """
        self.rs = rs
        self.g_min, self.g_max = g_min, g_max
        self.eta, self.v_th, self.noise = eta, v_th, noise
        self.rng = random.Random(seed)
        self.ga = [g_init * (1 + 0.15 * self.rng.uniform(-1, 1)) for _ in range(N_SYNAPSES)]
        self.gb = [g_init * (1 + 0.15 * self.rng.uniform(-1, 1)) for _ in range(N_SYNAPSES)]
        self.pulse_count = 0
        self.read_count = 0

    def device_write_voltage(self, n_parallel: int, g_each: float, v_drive: float = 1.0):
        """Write voltage each device actually gets when n devices of conductance g_each are in parallel."""
        return abs(v_drive) / (1 + self.rs * n_parallel * g_each)

    @staticmethod
    def _devices(mask: int):
        """DIO mask → (indices of connected side-A devices, indices of connected side-B devices)."""
        a = [b for b in range(8) if (mask >> b) & 1]
        b = [b - 8 for b in range(8, 16) if (mask >> b) & 1]
        return a, b

    def conductances(self, mask: int):
        """Parallel conductance of each branch."""
        a, b = self._devices(mask)
        return sum(self.ga[i] for i in a), sum(self.gb[i] for i in b)

    def read(self, mask: int, v=V_READ):
        self.read_count += 1
        GA, GB = self.conductances(mask)
        va = 1.0 / (1 + self.rs * GA)
        vb = 1.0 / (1 + self.rs * GB)
        return (vb - va) + self.rng.gauss(0, self.noise * 0.01), GA, GB

    def _program(self, idx, g, v_drive, forward):
        if not idx:
            return
        G = sum(g[i] for i in idx)
        v_dev = abs(v_drive) / (1 + self.rs * G)   # voltage each device actually gets
        if v_dev <= self.v_th:                     # no programming below threshold
            return
        drive = min(1.0, (v_dev - self.v_th) / max(self.v_th, 1e-9))
        for i in idx:
            step = self.eta * drive * (1 + self.rng.gauss(0, self.noise))
            step = max(0.0, min(1.0, step))
            if forward:
                g[i] += step * (self.g_max - g[i])
            else:
                g[i] -= step * (g[i] - self.g_min)
            g[i] = min(self.g_max, max(self.g_min, g[i]))

    def pulse(self, mask: int, forward: bool, amplitude: float):
        """Both branches are driven by W1 at the same time, each dividing according to its own parallel conductance."""
        self.pulse_count += 1
        a, b = self._devices(mask)
        self._program(a, self.ga, amplitude, forward)
        self._program(b, self.gb, amplitude, forward)


class HardwareBackend:
    """
    Real-hardware back end. Needs a V2 board + Mode 2 + two equal-value precision resistors at A/B.

    md : a MemristorDiscovery instance that has already been open()ed
    """

    def __init__(self, md, pulse_width_us=50.0,
                 scope1_offset=0.0, scope2_offset=0.0, w1_offset=0.0):
        self.md = md
        self.pw = pulse_width_us
        self.off1, self.off2, self.offw = scope1_offset, scope2_offset, w1_offset
        self.rs = md.rs
        self.pulse_count = 0
        self.read_count = 0

    N_SAMPLES = 300
    # noise floor of readings. Channel offset/noise measured by the Rs diagnosis is 0.3~1.7 mV (see SAFETY.md
    # section 5); 3 mV is used here as a conservative threshold.
    READ_NOISE_V = 3e-3
    READ_RANGE = 0.5      # the read pulse is only 80 mV, so the scope range must be tight
    WRITE_RANGE = 5.0
    SETTLE = 0.003        # the charge-injection transient after switching needs time to decay

    @staticmethod
    def _branches(mask):
        """DIO mask → (side-A device numbers, side-B device numbers), using the chip's 0..15 numbering.

        A/B are **two branches each with its own series resistor**, driven by the same W1.
        The gate needs them computed separately, otherwise it treats the two branches as one parallel branch (limiting too tightly).
        """
        a = [b for b in range(8) if (mask >> b) & 1]
        b = [b for b in range(8, 16) if (mask >> b) & 1]
        return a, b

    def _fire(self, mask, v, capture, rng=None, forward=None):
        """
        ★ This function used to call `md.set_dio()` + `md.start_custom()` directly,
        **completely bypassing the clipping in md.pulse()**. Every write pulse of the H/V 3×3 bar task went through here,
        so that clipping did not exist on this path.

        Now it passes the gate first: capture=True uses read authorisation (hard-clipped to 0.10 V),
        capture=False uses write authorisation (current limit + device voltage + floor + freshness + budget).
        The token issued by the gate is checked in `md.start_custom()`; if it does not match, no waveform is issued.
        """
        from knowm_md import make_pulse
        import dwfconst as C
        import time

        md = self.md
        gate = md._require_gate("ktram._fire")
        da, db = self._branches(mask)
        devs = da + db

        md.set_dio(mask)
        time.sleep(self.SETTLE)                 # let the switches settle before pulsing

        if capture:
            v = gate.authorize_read(devs, v)
        else:
            fwd = (v >= 0) if forward is None else bool(forward)
            v = gate.authorize_write(devs, v, width_us=self.pw,
                                     forward=fwd, n=1,
                                     branches=[da, db])
            v = abs(v) if fwd else -abs(v)

        wave = make_pulse("halfsine", v, duty=0.5, n=self.N_SAMPLES)
        freq = 0.5 / (self.pw * 1e-6)
        rng = rng or (self.READ_RANGE if capture else self.WRITE_RANGE)

        if capture:
            md.arm_capture(freq * self.N_SAMPLES, self.N_SAMPLES,
                           trigger_on=C.CH_W1, ranges=(rng, rng))
            md.wait_until_armed()
        md.start_custom(wave, freq, n_periods=1)
        raw = (None, None)
        if capture:
            md.wait_until_done(timeout=2.0)
            raw = md.fetch()
        else:
            time.sleep(1.0 / freq + 0.002)
        md.stop_wave()
        md.set_dio(0)          # Java calls turnOffAllSwitches(2) before and after each instruction
        time.sleep(0.001)
        return raw

    @staticmethod
    def _plateau(trace):
        """
        Take the mean near the top of the half-sine pulse.

        Java's `Util.maxAbs()` takes the **single point** with the largest absolute value in the whole segment --
        on a read signal of order 80 mV that point is most likely a noise spike rather than the pulse amplitude.
        Knowm itself uses the mean in PulseUtility, and this follows the mean.
        The pulse occupies the first half period with its peak at 25%; the 15%–35% segment is taken.
        """
        n = len(trace)
        seg = trace[int(n * 0.15):int(n * 0.35)]
        return sum(seg) / len(seg) if seg else 0.0

    def read(self, mask: int, v=V_READ, averages: int = 1):
        self.read_count += 1
        acc1 = acc2 = 0.0
        for _ in range(max(1, averages)):
            raw1, raw2 = self._fire(mask, v, capture=True)
            acc1 += self._plateau(raw1)
            acc2 += self._plateau(raw2)
        n = max(1, averages)
        p1 = acc1 / n - self.off1 - self.offw
        p2 = acc2 / n - self.off2 - self.offw
        w1 = self.md._polarity() * v + self.offw
        y = -(p1 - p2) / w1 if w1 else 0.0
        ia = (p1 - w1) / self.rs
        ib = (p2 - w1) / self.rs
        GA = max(-ia / p1, 0.0) if p1 else 0.0
        GB = max(-ib / p2, 0.0) if p2 else 0.0
        self.last_raw = (p1, p2, w1)

        # ★ feed the branch conductance back to the gate. This is the supply side of the "read before write" rule:
        # in a single-synapse FLV each branch connects only one device, so one run of ktram.read_all()
        # refreshes the freshness of all 16 devices.
        gate = self.md.gate
        if gate is not None:
            da, db = self._branches(mask)
            for devs, G, p in ((da, GA, p1), (db, GB, p2)):
                if not devs:
                    continue
                r = self._branch_r(G, p, w1)
                if r is not None:
                    gate.note_branch(devs, r)
        return y, GA, GB

    def _branch_r(self, G, p, w1):
        """
        Branch conductance → branch resistance (lower bound) fed back to the gate. Returns None if it cannot be read.

        ★ There is a pitfall here, hit twice on real hardware: `GA = max(-ia/p1, 0)` is **clamped to 0** on
          high-resistance devices, and the old code had `if GA > 0: note_branch(...)`
          -- 0 skipped everything and did not refresh freshness, so the second SET of 3×3 bar training reported
          "devices [8, 11, 13] ... reading stale".

        Why it is clamped to 0: the informative term is `p − w1 = w1·Rs/(R+Rs)`,

            R =  20 kΩ (refuse line) → 40.0 mV   ← 20× the noise, accurate
            R = 100 kΩ               → 13.3 mV
            R =   1 MΩ               →  1.6 mV   ← noise-floor level; even the sign may flip

          That is, **this reading is buried in noise exactly where it does not matter (high resistance),
          and has ample accuracy where it matters (near the refuse line)**. The asymmetry is favourable.

        So when the drop is below the noise threshold, the right conclusion is not "unknown" but a **lower bound**:

            drop < thr  ⇒  Rs/(R+Rs) < thr/|w1|  ⇒  R > Rs·(|w1|/thr − 1)

          Rs=20 kΩ, read level 80 mV, thr=3 mV ⇒ **R > 513 kΩ**,
          26× above the 20 kΩ refuse line -- a valid and generous basis for the floor.

        If the node voltage drops too much (|p| < 0.5|w1|) this inference is not made: that is not high resistance
        but a broken read (switch not closed, absurd offset); None is returned so the gate refuses SET.
        """
        import math as _m
        if G is not None and _m.isfinite(G) and G > 0:
            return 1.0 / G
        why = None
        if not (p and w1) or not _m.isfinite(p):
            why = "reading is not a finite number"
        elif abs(p) < 0.5 * abs(w1):
            why = (f"the node has only {abs(p)/abs(w1)*100:.0f}% of the source voltage -- "
                   f"this is not high resistance but a broken read (switch not closed? absurd offset?)")
        elif abs(p - w1) >= self.READ_NOISE_V:
            why = (f"drop {abs(p-w1)*1e3:.2f} mV is above the noise threshold "
                   f"{self.READ_NOISE_V*1e3:.0f} mV, yet a non-positive conductance was computed "
                   f"⇒ offset not compensated properly")
        if why is None:
            self._last_reject = None
            return self.rs * (abs(w1) / self.READ_NOISE_V - 1.0)
        self._last_reject = (why, p, w1, G)
        # print the raw voltages -- otherwise next time only the gate's "no resistance basis" message is seen,
        # and the faulty stage cannot be found. Print only the first few to avoid flooding the log.
        self._diag_n = getattr(self, "_diag_n", 0) + 1
        if self._diag_n <= 3:
            print(f"  [read anomaly] {why}")
            print(f"    raw values: node {p*1e3:+.3f} mV  source {w1*1e3:+.3f} mV  "
                  f"drop {(p-w1)*1e3:+.3f} mV  conductance {G!r}")
            print(f"    offset compensation: scope1 {self.off1*1e3:+.2f} / "
                  f"scope2 {self.off2*1e3:+.2f} / W1 {self.offw*1e3:+.2f} mV")
            print(f"    This will make the gate refuse the next SET (no resistance basis). "
                  f"Run calibrate_offsets or check_ktram.py first to check the read channel.")
        return None

    def _ensure_fresh(self, devs):
        """
        Refresh the **per-device** readings of these devices before SET.

        Why: the gate's floor check needs "**every** device ≥ refuse line".
        A branch reading can only conclude this when `R_branch ≥ refuse line`; N devices in parallel
        divide the resistance by N, so 3 devices of 50 kΩ in parallel are only 16.7 kΩ,
        in the undecidable range -- the members are fine, but the branch reading cannot prove it.

        The remedy is to read them one by one. In a single-synapse FLV each branch connects only one device,
        so one FLV refreshes the two devices A_i and B_i; at most 8 cover all 16.
        The read pulse is 0.08 V, 31% of threshold, and does not disturb the devices.

        (The inverted-pendulum path uses the same approach, see cartpole_hw.CartPoleHW._ensure_fresh.)
        """
        gate = self.md.gate
        if gate is None:
            return
        need = [d for d in devs if not gate.certified(d)]
        if not need:
            return
        for i in sorted({d % 8 for d in need}):
            self.read(mask_for(Instruction.FLV, (i,)), +V_READ)

        # ★ re-check after the extra reads. Still not fresh means **the read channel itself is faulty**,
        #   not "forgot to read" -- falling into the gate's generic "reading stale" message misleads
        #   (three rounds of debugging on real hardware). Report the real cause here directly.
        still = [d for d in devs if not gate.certified(d)]
        if still:
            side = ("side B" if all(d >= 8 for d in still) else
                    "side A" if all(d < 8 for d in still) else "both sides")
            rej = getattr(self, "_last_reject", None)
            lines = [
                f"After per-device re-reads, devices {still} are still not fresh -- "
                f"**this is not 'forgot to read'; the read channel of {side} returns nothing.**",
                "",
                f"In a single-synapse FLV each branch connects only one device and should give per-device resistance directly. "
                f"If it cannot be read, SET will never be allowed.",
            ]
            if rej:
                why, p, w1, G = rej
                lines += [
                    "", f"Last reading judged anomalous: {why}",
                    f"  node {p*1e3:+.3f} mV   source (assumed) {w1*1e3:+.3f} mV   "
                    f"drop {(p-w1)*1e3:+.3f} mV   conductance {G!r}",
                ]
            lines += [
                "",
                f"The most common cause of only one side failing: **Mode 2 needs both series resistors fitted at A/B**. "
                f"With one missing, the node on that side floats.",
                f"Run this first to see V(1+) / V(2+) per synapse:",
                f"  python check_ktram.py --chip <chip ID> --rs-nominal",
                f"It prints the raw voltages of both channels, showing at a glance which channel has no signal.",
            ]
            raise _Sv("\n  ".join(lines))

    def pulse(self, mask: int, forward: bool, amplitude: float):
        self.pulse_count += 1
        if forward and self.md.gate is not None:
            da, db = self._branches(mask)
            self._ensure_fresh(da + db)
        self._fire(mask, amplitude if forward else -amplitude,
                   capture=False, forward=forward)


# ---------------------------------------------------------------- calibration and diagnostics
def calibrate_offsets(md, v_probe=V_READ, n=8, verbose=True):
    """
    Measure the AD2's two scope offsets and the W1 offset automatically, **without taking the board apart or using a breadboard**.

    Principle: with all switches open, each branch reduces to "W1 ─ Rs ─ node ─ 1MΩ probe ─ ground".
      · With W1 at 0 V, the node is pulled to 0 V by Rs, so the scope reads its own offset.
      · With W1 at V, the node ≈ V·1MΩ/(1MΩ+Rs) ≈ 0.98·V,
        from which the actual W1 output follows; its difference from the set value is the W1 offset.

    Much simpler than the procedure in manual section 12.6 (disassemble board → ground the probes → breadboard 1+/2+ to W1),
    at the cost of a 2% divider error in the W1 offset -- entirely adequate for an 80 mV read signal.

    Returns (scope1_offset, scope2_offset, w1_offset) in V.
    """
    import time
    from knowm_md import make_pulse
    import dwfconst as C

    md.all_off()
    time.sleep(0.01)

    def _measure(v):
        a = b = 0.0
        for _ in range(n):
            freq = 0.5 / (50e-6)
            if abs(v) < 1e-9:
                # W1 held at 0 V (idle=Offset); free-running acquisition directly
                md.arm_capture(freq * 300, 300, trigger_on=None, ranges=(0.5, 0.5))
                time.sleep(0.01)
                md.wait_until_done(timeout=2.0)
                r1, r2 = md.fetch()
            else:
                # all switches open and the amplitude is the read level -- zero risk to the chip, but it still goes through the gate,
                # because start_custom now needs a token (see the "loophole" section of safety.py).
                v = md._require_gate("calibrate_offsets").authorize_read([], v)
                wave = make_pulse("halfsine", v, duty=0.5, n=300)
                md.arm_capture(freq * 300, 300, trigger_on=C.CH_W1, ranges=(0.5, 0.5))
                md.wait_until_armed()
                md.start_custom(wave, freq, n_periods=1)
                md.wait_until_done(timeout=2.0)
                r1, r2 = md.fetch()
                md.stop_wave()
            a += HardwareBackend._plateau(r1)
            b += HardwareBackend._plateau(r2)
        return a / n, b / n

    off1, off2 = _measure(0.0)                       # static: everything read is offset

    v_raw = md._polarity() * v_probe                 # voltage W1 should actually output
    d1, _ = _measure(v_probe)
    div = 1e6 / (1e6 + md.rs)                        # divider of the 1MΩ probe and Rs
    w1_actual = (d1 - off1) / div
    offw = w1_actual - v_raw

    md.all_off()
    if verbose:
        print("AD2 automatic offset calibration (all switches open)")
        print(f"  scope 1+ offset = {off1*1000:+7.2f} mV")
        print(f"  scope 2+ offset = {off2*1000:+7.2f} mV")
        print(f"  W1       offset = {offw*1000:+7.2f} mV")
        drift = -(off1 - off2) / v_raw
        print(f"\n  These offsets add a constant term of {drift:+.3f} to y.")
        if abs(drift) > 0.05:
            print(f"  A typical real y is only ±0.1~0.3 -- without compensation the sign is pinned by this constant,")
            print(f"  and classification accuracy locks at exactly 50% (always predicting one class).")
    return off1, off2, offw


def check_mode2_wiring(kt, verbose=True):
    """
    **Determine whether the board is wired for Mode 1 or Mode 2.** Read only, one read pulse.

    The ktram differential read assumes Mode 2:

        W1 ──┬──[ Rs ]── node A (scope 1+) ──[Ma]── GND
             └──[ Rs ]── node B (scope 2+) ──[Mb]── GND

    whereas Mode 1 / V0-V1 are wired as:

        W1 (scope 2+) ──[ Rs ]──┬── node (scope 1+) ──[Rm]── GND

    The difference is directly measurable: **in Mode 1 scope 2+ measures the W1 source voltage**, so it
    follows the source with no drop; in Mode 2 it is the node of branch B and must show a drop when a device is connected.

    This is how it was hit on real hardware (five rounds of debugging):
        V(1+) varies with the device between −65~−70 mV (drop present, normal)
        V(2+) fixed at −80.5 mV, while the assumed source is −75.2 mV
        p2 − w1 = −5.33 mV ≈ −offw (W1 offset +4.78 mV)
      ⇒ scope 2+ reads the source, and the difference is exactly the W1 offset subtracted once too often.

    Consequences: gb ≡ 0, y all of one sign, side B never refreshed, SET always refused by the gate.

    Returns {"mode2": bool, "p1","p2","w1", "why"}.
    """
    be = kt.backend
    if not hasattr(be, "md"):
        return {"mode2": True, "why": "simulation back end, no wiring involved"}
    kt.execute(Pattern((0,), True), Instruction.FLV)
    p1, p2, w1 = getattr(be, "last_raw", (float("nan"),) * 3)
    offw = getattr(be, "offw", 0.0)
    d1, d2 = p1 - w1, p2 - w1

    # scope2 follows the source (drop ≈ −offw, i.e. the offset subtracted once too often) ⇒ Mode 1
    looks_like_source = abs(d2 + offw) < max(2e-3, 0.3 * abs(offw) + 1e-3)
    mode2 = not (looks_like_source or abs(d2) < 1e-4)

    if verbose:
        print(f"  Wiring check: V(1+) = {p1*1e3:+.2f} mV (drop {d1*1e3:+.2f})  "
              f"V(2+) = {p2*1e3:+.2f} mV (drop {d2*1e3:+.2f})  "
              f"assumed source {w1*1e3:+.2f} mV")
        if mode2:
            print(f"    Both channels show a drop ⇒ **Mode 2** (two independent branches A/B) ✓")
        else:
            print(f"    ★ The drop on V(2+), {d2*1e3:+.2f} mV ≈ −offw "
                  f"({-offw*1e3:+.2f} mV) ⇒ **scope 2+ reads the W1 source**,")
            print(f"      not the node of branch B -- this is **Mode 1 wiring**.")
            print(f"      The ktram differential read does not work in Mode 1: gb is always 0, "
                  f"y is all of one sign, ")
            print(f"      side B is never refreshed, and SET will always be refused by the gate.")
            print(f"      **Set the Mode switch to 2 and confirm both 20 kΩ at A/B "
                  f"are fitted.**")
    return {"mode2": mode2, "p1": p1, "p2": p2, "w1": w1,
            "why": "scope2 tracks source" if not mode2 else "ok"}


STUCK_G = 15e-6      # > 15 µS (< 67 kΩ) counts as "stuck in the low-resistance state"


def _side_of(ga, gb):
    """Which side is stuck. Returns "a" / "b" / None."""
    worst = "a" if ga > gb else "b"
    return worst if max(ga, gb) > STUCK_G else None


def hard_reset(ktram, synapse, side="auto", target_r=100e3, v_start=None,
               v_max=None, v_step=0.1, width_us=2000, pulses=8,
               dry_run=False, verbose=True):
    """
    Erase devices stuck in the low-resistance state -- adaptive reset with stepped voltage.

    ★ Only **the stuck side** is biased; the switch of the other side stays open throughout.

    This is key. The two devices of a differential pair share one W1 but each has its own switch and series resistor.
    With RAB (both sides at once), the healthy side, having higher resistance, takes most of the voltage:
    for synapse 3 at 2 V drive, the stuck side B (5.1 kΩ) gets only 0.41 V,
    while the healthy side A (283 kΩ) takes 1.87 V -- the one needing rescue is not helped, and the one that should be left alone is hammered.
    So the default here is side="auto": read once, then send RA or RB only to the side whose conductance is over the limit.

    The 14 devices of the other 7 synapses have their switches open throughout and take no part.

    Why normal amplitudes cannot erase (manual section 12.4, verbatim):
    "The core problem of recovering a memristor that has achieved a very low
     resistance state is that the bulk of the voltage drop will occur across
     the series resistor instead of the memristor."

    The voltage the device gets is V_drive · Rm/(Rm+Rs): the more conductive, the less it gets -- a positive-feedback trap.

    ★★ **This used to be the highest-current path in the whole repository, and its purpose was to "rescue devices".**
    The old default was 2.5 V × 20 pulses of **2 ms**: with Rs=20 kΩ in Mode 2 that is 125 µA
    for 40 ms. Manual Note 1 says this order **irreversibly** lowers the high-resistance state from 1 MΩ to
    below 100 kΩ -- i.e. the "rescue" itself caused the very damage it was meant to repair.
    There was also an `allow_overcurrent=True` back door to break the limits.

    **That back door has been removed, and v_start / v_max are no longer decided by the caller.**
    The amplitude is now issued entirely by the protection gate: reverse pulses stay within the 50 µA envelope, with a drive limit of
    I_safe·Rs (1.0 V at Rs=20 kΩ). The only variable is time -- Method 1 of manual section 12.4
    uses −2 V for 2~5 s, relying on time rather than current.

    More importantly: **under the new envelope this function should hardly ever be used.** The floor rule keeps every
    device above 67 kΩ (the STUCK_G line), so "stuck" should not happen in the first place.
    It is kept to handle surprises on a new chip, not as part of the normal procedure.

    Parameters
    ----------
    side     : "auto" (default, fix only the stuck side) / "a" / "b" / "both"
    v_start  : None = start from half the gate's allowed limit (recommended)
    v_max    : None = the gate's allowed limit. Larger values are ignored.
    dry_run  : only print what would be applied; do not actually pulse
    """
    p = Pattern((synapse,), True)
    be = ktram.backend
    rs = getattr(be, "rs", 20e3)
    target_g = 1.0 / target_r
    INSTR = {"a": Instruction.RA, "b": Instruction.RB, "both": Instruction.RAB}
    CHIP = {"a": synapse + 1, "b": synapse + 9}

    ktram.execute(p, Instruction.FLV)
    ga, gb = ktram.last_ga, ktram.last_gb
    if side == "auto":
        side = _side_of(ga, gb)
        if side is None:
            if verbose:
                print(f"  Neither side of synapse {synapse} is stuck "
                      f"(ga={ga*1e6:.2f} gb={gb*1e6:.2f} µS); skipping.")
            return {"ok": True, "skipped": True, "ga": ga, "gb": gb}

    if verbose:
        mask = mask_for(INSTR[side], (synapse,))
        dio = [b for b in range(16) if mask >> b & 1]
        print(f"  Target: side {side.upper()} of synapse {synapse} = chip device "
              f"{CHIP.get(side, f'{synapse+1}+{synapse+9}')}")
        print(f"  Closed switches: DIO {dio} (the other {16-len(dio)} lines stay open throughout)")

    def _g(): 
        ktram.execute(p, Instruction.FLV)
        return ktram.last_ga if side == "a" else (
            ktram.last_gb if side == "b" else max(ktram.last_ga, ktram.last_gb))

    # ---- limits: all decided by the gate; the limit is copied here only for printing ----
    gate = getattr(getattr(be, "md", None), "gate", None)
    if gate is not None:
        v_safe = min(gate.env.i_safe * gate.env.rs, 5.0)
        if width_us > gate.env.width_max_rev_us:
            if verbose:
                print(f"  [clip] pulse width {width_us/1000:.1f} ms → "
                      f"{gate.env.width_max_rev_us/1000:.1f} ms (gate limit)")
            width_us = gate.env.width_max_rev_us
        pulses = min(int(pulses), gate.env.n_per_call_max)
    else:
        import device_limits as _L
        v_safe = _L.max_drive(rs, 50e-6)        # simulation back end: computed with the same envelope
    v_max = v_safe if v_max is None else min(float(v_max), v_safe)
    v_start = 0.5 * v_max if v_start is None else min(float(v_start), v_max)
    if verbose:
        print(f"  Amplitude set by the gate: {v_start:.2f} → {v_max:.2f} V "
              f"(= I_safe × Rs, worst-case current ≤ "
              f"{v_max/rs*1e6:.0f} µA); pulse width {width_us/1000:.1f} ms does the work.")

    if dry_run:
        g = _g(); R = 1.0 / max(g, 1e-12)
        print(f"  Currently {R/1000:.1f} kΩ. Would apply in turn (not executed):")
        v = v_start
        while v <= v_max + 1e-9:
            print(f"    {v:.1f} V × {pulses} pulses of {width_us/1000:.1f} ms"
                  f"  → device gets {v*R/(R+rs):.3f} V, current ≈ {v/(R+rs)*1e6:.0f} µA")
            v += v_step
        return {"ok": None, "dry_run": True, "ga": ga, "gb": gb}

    old_pw = getattr(be, "pw", None)
    old_vr = ktram.v_reverse
    try:
        if old_pw is not None:
            be.pw = width_us
        v = v_start
        while True:
            g = _g()
            if g <= target_g:
                if verbose:
                    print(f"  Recovered: side {side.upper()} = {g*1e6:.2f} µS "
                          f"= {1/max(g,1e-12)/1000:.0f} kΩ (target > {target_r/1000:.0f} kΩ)")
                return {"ok": True, "side": side, "g": g, "v_used": v}
            if v > v_max:
                if verbose:
                    print(f"  Still not recovered at {v_max} V ({g*1e6:.2f} µS).")
                    print("  Higher voltage risks burning the chip. Use Method 1 of manual section 12.4 instead:")
                    print(f"    set Mode to 1 → DEVICE = {CHIP.get(side,0)-1} in example_dc.py"
                          f" → run one slow large-amplitude sweep")
                    print("    (in Mode 1 the device is connected alone with no parallel load, so it is easiest to erase)")
                return {"ok": False, "side": side, "g": g, "v_used": v_max}
            R = 1.0 / max(g, 1e-12)
            if verbose:
                print(f"  {v:.1f} V × {pulses} reverse pulses of {width_us/1000:.1f} ms"
                      f"  → device gets {v*R/(R+rs):.3f} V, current ≈ {v/(R+rs)*1e6:.0f} µA"
                      f"  (currently {R/1000:.1f} kΩ)")
            ktram.v_reverse = v
            for _ in range(pulses):
                ktram.execute(p, INSTR[side])
            v += v_step
    finally:
        if old_pw is not None:
            be.pw = old_pw
        ktram.v_reverse = old_vr


def hard_reset_all(ktram, synapses=None, verbose=True, **kw):
    """
    Hard reset of the given synapses; synapses=None means scan all 8 and fix only the stuck ones.

    Within each synapse only the stuck side is biased and the other side is disconnected -- see hard_reset.
    """
    targets = range(N_SYNAPSES) if synapses is None else synapses
    fixed, failed, skipped = [], [], []
    for i in targets:
        ktram.execute(Pattern((i,), True), Instruction.FLV)
        ga, gb = ktram.last_ga, ktram.last_gb
        if _side_of(ga, gb) is None and synapses is None:
            skipped.append(i)
            continue
        if verbose:
            print(f"\nSynapse {i} (chip devices {i+1},{i+9})"
                  f"ga={ga*1e6:.2f} gb={gb*1e6:.2f} µS")
        r = hard_reset(ktram, i, verbose=verbose, **kw)
        if r.get("dry_run"):
            continue
        (fixed if r["ok"] else failed).append(i)
    if verbose and not kw.get("dry_run"):
        if skipped:
            print(f"\n  Not stuck (skipped): {skipped}")
        print(f"  Recovered: {fixed}")
        if failed:
            print(f"  Still not recovered: {failed}  ← use Mode 1 + example_dc.py slow sweep")
    return {"fixed": fixed, "failed": failed, "skipped": skipped}



def diagnose_write(ktram, synapse=0, n_pulses=10, side="a", verbose=True):
    """
    Write response test -- **checks both magnitude and direction**.

    On the given side of the given synapse: first hit N forward pulses (conductance should **rise**),
    then N reverse pulses (conductance should **fall**).

    Checking magnitude alone is not enough: a device may move, but opposite to the instruction.
    Then weights grow against the training signal and the final receptive field is disordered,
    while a number like "74% change" looks perfectly normal. This version judges the two directions separately.

    Returns fwd_ok / rev_ok / write_ok (ok only if both directions are correct).
    """
    p = Pattern((synapse,), True)
    F = Instruction.FA if side == "a" else Instruction.FB
    R = Instruction.RA if side == "a" else Instruction.RB
    chip = synapse + 1 if side == "a" else synapse + 9

    def g():
        ktram.execute(p, Instruction.FLV)
        return ktram.last_ga if side == "a" else ktram.last_gb

    g0 = g()
    for _ in range(n_pulses):
        ktram.execute(p, F)
    g1 = g()
    for _ in range(n_pulses):
        ktram.execute(p, R)
    g2 = g()

    tol = max(0.10 * max(g0, 1e-12), 2e-7)      # 10% or 0.2 µS, whichever is larger
    df, dr = g1 - g0, g2 - g1
    fwd = "ok" if df > tol else ("flip" if df < -tol else "stuck")
    rev = "ok" if dr < -tol else ("flip" if dr > tol else "stuck")
    fwd_ok, rev_ok = fwd == "ok", rev == "ok"
    swing = max(abs(df), abs(dr))
    rel = swing / max(g0, 1e-12)

    # three faults must be kept apart; lumping them into "wrong direction" misleads debugging
    if fwd_ok and rev_ok:
        kind = "ok"
    elif fwd == "flip" and rev == "flip":
        kind = "both_flip"        # genuinely reversed polarity
    elif "flip" in (fwd, rev):
        kind = "one_flip"         # only one direction is reversed
    else:
        kind = "stuck"            # one direction cannot be moved

    if verbose:
        print(f"Side {side.upper()} of synapse {synapse} (chip device {chip})"
              f"  {n_pulses} pulses per step, V_fwd={ktram.v_forward} / V_rev={ktram.v_reverse}")
        print(f"  initial            g = {g0*1e6:8.2f} µS")
        print(f"  after {n_pulses}× forward  g = {g1*1e6:8.2f} µS  ({(g1-g0)*1e6:+7.2f})  "
              f"{'up ok' if fwd_ok else ('down ★wrong direction' if (g1-g0) < -tol else 'barely moved')}")
        print(f"  after {n_pulses}× reverse  g = {g2*1e6:8.2f} µS  ({(g2-g1)*1e6:+7.2f})  "
              f"{'down ok' if rev_ok else ('up ★wrong direction' if (g2-g1) > tol else 'barely moved')}")
        if kind == "ok":
            print(f"\n  Magnitude {rel*100:.0f}%, both directions correct → write is normal")
        elif kind == "both_flip":
            print(f"\n  ★ Both directions reversed -- genuinely reversed polarity.")
            print("    If most devices behave like this, the 1×16 chip is probably inserted backwards (rotate 180° and reinsert).")
        elif kind == "one_flip":
            bad = "forward" if fwd == "flip" else "reverse"
            print(f"\n  ★ {bad} goes the wrong way (the other direction is normal).")
            print("    A one-direction anomaly usually means the drive amplitude is too large and pushes the device too far -- reduce it and retry.")
        else:
            bad = "forward" if fwd == "stuck" else "reverse"
            print(f"\n  ★ {bad} cannot be moved (the other direction responds).")
            print("    Amplitude too low, or the device is at its limit in that direction.")
    return {"g": (g0, g1, g2), "side": side, "chip": chip,
            "fwd": fwd, "rev": rev, "kind": kind,
            "fwd_ok": fwd_ok, "rev_ok": rev_ok,
            "write_ok": bool(fwd_ok and rev_ok), "swing_rel": rel}


def diagnose_read(ktram, verbose=True):
    """
    Read the 8 synapses one by one to see whether y is moving at all.

    When classification gives 50%, run this first. If all 8 signs of y are the same,
    or the range is as small as the noise, the problem is not the learning algorithm.
    """
    rows = []
    for i in range(N_SYNAPSES):
        y = ktram.execute(Pattern((i,), True), Instruction.FLV)
        raw = getattr(ktram.backend, "last_raw", (float("nan"),) * 3)
        rows.append((i, y, ktram.last_ga, ktram.last_gb, raw))

    ys = [r[1] for r in rows]
    spread = max(ys) - min(ys)
    same_sign = all(v >= 0 for v in ys) or all(v < 0 for v in ys)

    if verbose:
        import math as _m

        def _mv(x):
            return "        -" if x is None or _m.isnan(x) else f"{x*1000:>8.2f} "

        print(f"{'Syn':>4s}{'y':>10s}{'ga (µS)':>11s}{'gb (µS)':>11s}"
              f"{'V(1+) mV':>11s}{'V(2+) mV':>11s}")
        for i, y, ga, gb, raw in rows:
            print(f"{i:>4d}{y:>10.4f}{ga*1e6:>11.2f}{gb*1e6:>11.2f}"
                  f"{_mv(raw[0]):>11s}{_mv(raw[1]):>11s}")
        stuck = [(i, ga, gb) for i, _, ga, gb, _ in rows
                 if max(ga, gb) > STUCK_G]
        if stuck:
            print(f"\n  ★ {len(stuck)} synapses have devices stuck in the low-resistance state"
                  f" (> {STUCK_G*1e6:.0f} µS, i.e. < {1/STUCK_G/1000:.0f} kΩ):")
            for i, ga, gb in stuck:
                side = "side A" if ga > gb else "side B"
                chip = i + 1 if ga > gb else i + 9
                g = max(ga, gb)
                print(f"      synapse {i} {side} (chip device {chip})"
                      f"= {g*1e6:.1f} µS = {1/g/1000:.1f} kΩ")
            print("    The w of these synapses is pinned to one side and cannot be pulled back however training pushes.")
            print("    Normal RAB cannot erase them -- the more conductive the device, the more voltage the series resistor takes.")
            print("    Use the stepped hard reset: python check_ktram.py --hard-erase")

        print(f"\n  y range = {spread:.4f}   signs {'all the same ← problem' if same_sign else 'both positive and negative ← normal'}")
        if same_sign:
            print("  All 8 synapses cannot naturally have the same sign. Almost certainly the offsets are not compensated:")
            print("  run calibrate_offsets(md) and enter the three offsets into HardwareBackend.")
        if spread < 0.02:
            print("  y barely moves -- check that the Mode switch is at '2' and both resistors at A/B are fitted and equal.")
    return {"y": ys, "spread": spread, "same_sign": same_sign, "rows": rows}


# ---------------------------------------------------------------- controller
@dataclass
class KTRAM:
    """8-synapse kT-RAM core. The instruction set corresponds one-to-one with Java's Instruction12."""
    backend: object
    v_forward: float = 1.0
    v_reverse: float = 1.0
    last_y: float = 0.0
    last_ga: float = 0.0
    last_gb: float = 0.0
    log: list = field(default_factory=list)

    def execute(self, pattern: Pattern, instruction: str):
        mask = mask_for(instruction, pattern.spikes)
        if instruction == Instruction.FLV:
            self.last_y, self.last_ga, self.last_gb = self.backend.read(mask, +V_READ)
        elif instruction == Instruction.RLV:
            self.last_y, self.last_ga, self.last_gb = self.backend.read(mask, -V_READ)
        elif instruction in (Instruction.FA, Instruction.FB, Instruction.FAB):
            self.backend.pulse(mask, True, self.v_forward)
        elif instruction in (Instruction.RA, Instruction.RB, Instruction.RAB):
            self.backend.pulse(mask, False, self.v_reverse)
        else:
            raise ValueError(f"unknown instruction {instruction}")
        return self.last_y

    # ---- compound instructions (used in Synapse12 / the SAT solver)
    def hebbian(self, pattern):
        """Reinforce the current polarity."""
        self.execute(pattern, Instruction.FLV)
        if self.last_ga >= self.last_gb:
            self.execute(pattern, Instruction.FA); self.execute(pattern, Instruction.RB)
        else:
            self.execute(pattern, Instruction.RA); self.execute(pattern, Instruction.FB)

    def anti_hebbian(self, pattern):
        """Weaken the current polarity, pushing y towards 0."""
        self.execute(pattern, Instruction.FLV)
        if self.last_ga >= self.last_gb:
            self.execute(pattern, Instruction.RA); self.execute(pattern, Instruction.FB)
        else:
            self.execute(pattern, Instruction.FA); self.execute(pattern, Instruction.RB)

    def read_all(self):
        """Read the weights of the 8 synapses one by one (one read pulse per synapse)."""
        out = []
        for i in range(N_SYNAPSES):
            out.append(self.execute(Pattern((i,), True), Instruction.FLV))
        return out

    def scramble(self, epochs=20, rng=None):
        """
        The GUI's Scramble / ResetWorker:
        each epoch randomly picks 4 single synapses and applies anti-Hebbian to push them towards y≈0.
        """
        rng = rng or random.Random(0)
        for _ in range(epochs):
            for _ in range(4):
                i = rng.randrange(N_SYNAPSES)
                self.anti_hebbian(Pattern((i,), True))

    def initialize(self, rounds=100):
        """The SAT solver's Initialize: FLV → FAB → ANTI_HEBBIAN, raising both sides and then levelling them."""
        for _ in range(rounds):
            for i in range(N_SYNAPSES):
                p = Pattern((i,), True)
                self.execute(p, Instruction.FLV)
                self.execute(p, Instruction.FAB)
                self.anti_hebbian(p)


# ---------------------------------------------------------------- training
def train(ktram: KTRAM, dataset, routine="LearnAlways", epochs=50,
          k=0.05, shuffle=True, seed=0, record=True,
          stop_acc=None, stop_patience=2, stop_margin=0.05):
    """
    Reimplementation of Classify12Experiment.TrialWorker.

    Returns a dict: accuracy (exponential moving average after each presentation), weights (8 weights at the end of each epoch),
    final_acc (true accuracy on the whole data set in the last epoch), epochs_run (number of epochs actually run).

    stop_acc
        Accuracy threshold for early stopping; None = do not stop (default, original behaviour).
        At the end of each epoch the whole data set is evaluated **read-only**, and training stops after stop_patience consecutive epochs on target.
        The criterion is the true end-of-epoch accuracy, not the exponential moving average -- the EMA lags,
        and using it as the stopping condition stops a dozen epochs late.
        Extra cost: len(dataset) read pulses per epoch.

        Why two consecutive epochs: updates are unconditional, so a single epoch can reach the target by chance as weights swing.

    stop_margin
        Early stopping also requires the minimum |y| to reach this value (default 0.05). **Accuracy alone stops too early**:
        the sign decisions are all correct long before the weights have grown; measured, the target is reached in epoch 2 when |y| is only 0.03,
        which read noise eats directly on hardware. Decision reliability depends on margin, not accuracy, so both are required.
    """
    fn = ROUTINES[routine] if isinstance(routine, str) else routine
    rng = random.Random(seed)
    data = list(dataset)
    acc_ema, acc_trace, w_trace = 0.0, [], []
    hits, epochs_run = 0, 0

    for _ in range(epochs):
        epochs_run += 1
        if shuffle:
            rng.shuffle(data)
        for p in data:
            y = ktram.execute(p, Instruction.FLV)
            correct = (y > 0) == p.state
            acc_ema = (1 - k) * acc_ema + (k if correct else 0.0)
            fn(ktram, p, y)
            if record:
                acc_trace.append(acc_ema)
        if record:
            w_trace.append(ktram.read_all())
        if stop_acc is not None:
            ys = [ktram.execute(p, Instruction.FLV) for p in data]
            acc = sum((y > 0) == p.state for y, p in zip(ys, data)) / len(data)
            margin = min(abs(y) for y in ys)
            good = acc >= stop_acc and margin >= stop_margin
            hits = hits + 1 if good else 0
            if hits >= stop_patience:
                break

    # finally measure the true accuracy (read only, no update)
    n_ok = sum(1 for p in dataset
               if (ktram.execute(p, Instruction.FLV) > 0) == p.state)
    return {
        "accuracy": acc_trace,
        "weights": w_trace,
        "final_acc": n_ok / len(dataset),
        "final_y": [ktram.execute(p, Instruction.FLV) for p in dataset],
        "epochs_run": epochs_run,
        "stopped_early": stop_acc is not None and epochs_run < epochs,
    }


# ---------------------------------------------------------------- 3SAT
def solve_sat(ktram: KTRAM, constraints, max_iters=200):
    """
    Reimplementation of kTBitSatSolverExperiment.TrialWorker.

    constraints : [(a,b,c), ...], DIMACS-style signed literals, 1-based,
                  a minus sign means negation. Variables 1..8 correspond to the 8 synapses.

    Each round: read the 8 kTBits (sign(y)) → check each constraint →
    for each variable, Hebbian (reinforce the current value) if all constraints it takes part in are satisfied,
    otherwise Anti-Hebbian (push it towards flipping). Until all are satisfied.

    This is essentially moving WalkSAT-style stochastic local search onto analog hardware,
    with the "randomness" supplied naturally by device noise.
    """
    var2c = {}
    for c in constraints:
        for lit in c:
            var2c.setdefault(abs(lit), []).append(c)

    history = []
    for it in range(max_iters):
        y = ktram.read_all()
        bits = [1 if v >= 0 else -1 for v in y]

        def sat(c):
            return any(bits[abs(l) - 1] * l > 0 for l in c)

        n_sat = sum(1 for c in constraints if sat(c))
        history.append(n_sat)
        if n_sat == len(constraints):
            return {"solved": True, "iters": it, "bits": bits, "history": history}

        for x, cl in var2c.items():
            idx = x - 1
            p = Pattern((idx,), True)
            reward = all(sat(c) for c in cl)
            positive = y[idx] >= 0
            # reward → reinforce the current polarity; otherwise → weaken (encourage flipping)
            if reward == positive:
                ktram.execute(p, Instruction.FA); ktram.execute(p, Instruction.RB)
            else:
                ktram.execute(p, Instruction.RA); ktram.execute(p, Instruction.FB)

    return {"solved": False, "iters": max_iters, "bits": bits, "history": history}
