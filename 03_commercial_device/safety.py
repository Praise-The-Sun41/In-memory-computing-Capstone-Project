"""
safety.py —— hard protection gate for the new chip. **Every action that applies voltage must go through here.**

How the previous chip was damaged is written first, because every rule in this file follows from it:

    That sweep drew about 455 µA (the drive voltage was clipped by the AWG rail, and current limiting relied only on a 10 kΩ series resistor),
    right in the "irreversible damage" range of manual Note 1. After the devices dropped to 690 Ω ~ 1.5 kΩ,
    they entered a **positive feedback with no way back**:

        resistance drops → the voltage the device gets = V_drive·R/(R+Rs) drops with it
                         → at safe current the 0.26 V threshold can no longer be reached, so it cannot be erased
                         → erasing needs more current → deeper damage → resistance keeps dropping

    The entrance to this loop is called the **point of no return**. At I = 100 µA, V_th = 0.26 V,
    a device can still be erased only if I·(R∥Rs) ≥ V_th, i.e. R∥Rs ≥ 2.6 kΩ:

        Rs = 10 kΩ  →  R ≥ 3.51 kΩ
        Rs = 20 kΩ  →  R ≥ 2.99 kΩ
        Rs = 50 kΩ  →  R ≥ 2.75 kΩ

    The 690 Ω of the previous chip is **below** this line, so it did not "perform worse";
    it was physically locked. A fresh chip has HRS ≈ 1 MΩ, 300× away from this line,
    so the only task of this file is: **keep it above the line forever.**

=================== Five layers of protection, each against a different failure ===================

**Layer 1 · current limit (main defence)**   |V_drive| ≤ I_safe · Rs, computed for the worst case R→0.

  This layer alone is strong enough, and strong in a somewhat counter-intuitive way worth spelling out: the total branch current is
  V_drive/(Rs+R_branch) ≤ V_drive/Rs ≤ I_safe; and in a parallel branch **the current of any single device
  never exceeds the total branch current**. So as long as this holds, however low devices drop,
  however many are in parallel and however long the pulse, **every device's current at every instant is ≤ I_safe**.
  The previous chip was not damaged because this rule was wrong, but because I_limit was set to 455 µA at the time.

**Layer 2 · device voltage**   V_dev ≤ V_dev_max, keeping the operating point within the manual's switching range.

**Layer 3 · resistance floor (a trip, not the main defence)**
  Before any forward (SET) pulse, every connected device must have a **fresh** resistance reading
  ≥ R_FLOOR_GUARD.

  Since layer 1 already guarantees the current, what does this layer guard against? -- **against layer 1 not actually being in effect.**
  Layer 1 rests entirely on "the real value of Rs equals the number in the code". If the board actually has
  10 kΩ while the code assumes 20 kΩ, the real current is twice the setting, and the program
  never reports any error. Similar failures: the Mode switch in the wrong position, a shorted analog
  switch, a dry joint on the series resistor.

  Within the 50 µA envelope a device should have no chance of getting below 13 kΩ (manual figure 4:
  the high-resistance state for this compliance current is of order 1 MΩ). So **reading below 13 kΩ is itself
  evidence that "the envelope is not actually in effect"**. The right response then is not to keep pushing by the rules
  but to stop -- so this layer is refuse + quarantine + trip, not clipping.

  It also closes a real loophole of the past: the old clipping looked only at current, not at where the device
  was, so a script could push a device off the cliff step by step while every step was "compliant".

**Layer 4 · budget and records**   per-device/per-session pulse limits, cross-session chip_journal,
  per-pulse CSV audit. The purpose is to cap total exposure and to make **slow degradation visible across sessions** --
  last time nobody was watching the resistance trend.

**Layer 5 · start-up self-test**   preflight measures the real value of Rs directly (all switches open,
  using the divider of Rs and the scope's 1 MΩ input impedance), then reads the 16 devices one by one and compares with the record.
  This layer tests exactly the assumption that layer 1 depends on.

=================== The narrowed operating point (more conservative than the manual) ===================

The user asked for "type W, but with an even narrower range for safety", so neither the manual's absolute ratings
nor the 100 µA compliance of the previous version is used:

                        Manual absolute   Previous default   **Now**
    compliance current   1 mA              100 µA             **50 µA**
    device voltage       3 V               0.75 V             **0.60 V**
    series resistance    —                 10 kΩ              **≥ 20 kΩ (50 kΩ recommended)**
    resistance floor     —                 none               **13 kΩ (hard refuse)**

Why these numbers are self-consistent -- the key check is "can a device at the floor still be erased",
because failing to erase means falling into the loop above:

    maximum voltage the device can get  V_dev = min(I_safe·(R∥Rs), V_dev_max)

    R = 13 kΩ (floor)   Rs = 20 kΩ → 50 µA × 7.9 kΩ = 0.40 V  ✓ > 0.26 V
                        Rs = 50 kΩ → 50 µA × 10.3 kΩ = 0.52 V ✓
    R = 1 MΩ (new chip) both Rs limited by V_dev_max → 0.60 V   ✓

    Rs = 10 kΩ          → 50 µA × 5.7 kΩ = 0.28 V -- only 7% margin, not accepted.
                           **This is why Rs < 20 kΩ is refused.**

V_dev increases monotonically with R, so the check above is at the worst point: **any device in the whole
[13 kΩ, 1 MΩ] window can be moved within the safety envelope.** This is exactly the property the previous chip lost.

=================== Loopholes that used to allow bypassing ===================

  · `MemristorDiscovery.enforce_limits = False`        → removed; the switch no longer exists
  · `ktram.hard_reset(allow_overcurrent=True)`         → removed
  · calling `md.start_wave()` / `md.start_custom()` directly → now requires a token issued by the gate
  · `ktram.HardwareBackend._fire()` bypassing `md.pulse()`  → now obtains a token first

=================== Usage ===================

    import safety
    env = safety.Envelope.for_rs(20e3)          # Rs must be given explicitly, no default
    gate = safety.SafetyGate(env, chip_id="chip-2026-10", purpose="bars")

    v = gate.authorize_write([3], amplitude=1.0, width_us=50, forward=True)
    gate.note_resistance(3, 180e3)              # feed back after every read

The gate is **stateful**: it remembers each device's last reading, how many pulses since then,
and how many in total. There is no reset interface after a trip -- continuing requires restarting the process and redoing preflight,
deliberately.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict

import device_limits as _L

__all__ = [
    "SafetyViolation", "GateTripped", "Envelope", "SafetyGate",
    "I_SAFE", "V_DEV_MAX", "R_FLOOR", "R_FLOOR_GUARD", "R_CEILING",
    "RS_MIN", "RS_RECOMMENDED", "journal_path", "load_journal", "save_journal",
]


# ===========================================================================
# narrowed envelope constants -- before changing them, redo the self-consistency check above
# ===========================================================================

I_SAFE = 50e-6          # A   current allowed in the worst case (R→0). Manual type W rating 1 mA.
V_DEV_MAX = 0.60        # V   limit across the device. Manual switching range 0.1~0.75 V.
V_READ = 0.10           # V   read pulse amplitude, well below the 0.26 V threshold

R_FLOOR = 10e3          # Ω   physical floor: below this the device is regarded as in the damage region
R_FLOOR_GUARD = 20e3    # Ω   **refuse line**. Not hand-picked; set by the derivation below (see self_check)
R_CEILING = 5e6         # Ω   above this counts as "open/not connected", not a device state

RS_MIN = 20e3           # Ω   lower limit of series resistance (see the self-consistency check above)
RS_RECOMMENDED = 50e3   # Ω   the value used by the manual's own endurance test

WIDTH_MAX_FWD_US = 200.0    # µs  forward (SET) pulse width limit -- the dangerous direction, no time given
WIDTH_MAX_REV_US = 5000.0   # µs  reverse (RESET) pulse width limit -- the safe direction, which works through time
N_PER_CALL_MAX = 8          #     maximum pulses per call, against runaway loops

# by how much one 50 µs pulse can lower the resistance in the worst case.
# measured in demos/README: one pulse moves 15~26% of the log window; with the window 1 MΩ→50 kΩ (20×)
# 26% corresponds to ÷2.18; including 3× step jitter, 2.5 is taken as the worst single-pulse factor.
PULSE_WORST_DROP = 2.5

# after how many consecutive SET pulses a device must be read again.
# ★ **This value cannot be set arbitrarily.** The first version used 6, and 6 blind pulses can lower resistance 36× --
#   from the refuse line straight through the point of no return. It is now fixed at 1: **every SET must be preceded by
#   a fresh reading**, and the refuse line is set so that it withstands one blind pulse.
#   the cost is almost zero: ktram's train() already reads FLV once per pattern,
#   and after each FLV each device receives at most one SET.
STALE_LIMIT = 1
BUDGET_PER_DEVICE = 20_000   #   cumulative pulse budget per device (50 µs each ≈ 1 s of cumulative stress)
BUDGET_PER_SESSION = 120_000 #   cumulative pulse budget for the whole session
QUARANTINE_ABORT = 3    #     trip the whole gate when this many devices are quarantined

V_TH = _L.V_TH_FWD["W"]   # 0.26 V


class SafetyViolation(RuntimeError):
    """This pulse is refused. The caller should treat it as "the approach is wrong", not "try again"."""


class GateTripped(RuntimeError):
    """The gate has tripped. No further voltage may be applied in this process; there is no reset interface."""


# ===========================================================================
# envelope
# ===========================================================================
@dataclass(frozen=True)
class Envelope:
    """Hard envelope of one session. `rs` has no default -- it must be declared explicitly by the caller."""

    rs: float
    i_safe: float = I_SAFE
    v_dev_max: float = V_DEV_MAX
    r_floor: float = R_FLOOR
    r_floor_guard: float = R_FLOOR_GUARD
    r_ceiling: float = R_CEILING
    v_read: float = V_READ
    dopant: str = "W"
    width_max_fwd_us: float = WIDTH_MAX_FWD_US
    width_max_rev_us: float = WIDTH_MAX_REV_US
    n_per_call_max: int = N_PER_CALL_MAX
    stale_limit: int = STALE_LIMIT
    budget_per_device: int = BUDGET_PER_DEVICE
    budget_per_session: int = BUDGET_PER_SESSION

    # ------------------------------------------------------------------
    @classmethod
    def for_rs(cls, rs: float, **kw) -> "Envelope":
        """
        Build an envelope, and block "is Rs large enough" right here.

        Too small an Rs ruins both ends: the current limit V_drive ≤ I_safe·Rs pushes the drive very low,
        and the I_safe·(R∥Rs) a device can get falls with it -- so devices at the floor cannot be erased,
        and "cannot be erased" is the entrance to the point of no return.
        """
        rs = float(rs)
        if not math.isfinite(rs) or rs <= 0:
            raise SafetyViolation(f"series resistance must be positive, got {rs!r}")
        if rs < RS_MIN:
            v_dev_at_floor = kw.get("i_safe", I_SAFE) * _par(R_FLOOR_GUARD, rs)
            raise SafetyViolation(
                f"Series resistance Rs = {rs/1e3:.1f} kΩ is below the lower limit of {RS_MIN/1e3:.0f} kΩ; refusing to start.\n"
                f"  Reason: a device at the floor ({R_FLOOR_GUARD/1e3:.0f} kΩ) at "
                f"{kw.get('i_safe', I_SAFE)*1e6:.0f} µA gets only "
                f"{v_dev_at_floor:.3f} V, while the threshold is {V_TH:.2f} V.\n"
                f"  Cannot erase = point of no return. Set the Mode so that the two 20 kΩ series resistors at A/B\n"
                f"  act separately (20 kΩ), or replace them with the "
                f"{RS_RECOMMENDED/1e3:.0f} kΩ of the manual's endurance test.")
        env = cls(rs=rs, **kw)
        env.self_check()
        return env

    # ------------------------------------------------------------------
    def self_check(self):
        """
        Self-consistency check: a device at the floor must still be pushable past the threshold, with margin.

        This is the most important assertion in the whole file. It guarantees that **every device within the safe window is writable**,
        so the situation "having to exceed the current to make it move" can never arise.
        """
        v_at_floor = min(self.i_safe * _par(self.r_floor_guard, self.rs),
                         self.v_dev_max)
        margin = v_at_floor / V_TH
        if margin < 1.25:
            raise SafetyViolation(
                f"Envelope inconsistent: a floor device ({self.r_floor_guard/1e3:.0f} kΩ) gets at most "
                f"{v_at_floor:.3f} V, only {margin:.2f}× the threshold of {V_TH:.2f} V.\n"
                f"  Margin below 1.25× ⇒ a device near the floor may not be erasable ⇒ no return.\n"
                f"  Either increase Rs (now {self.rs/1e3:.0f} kΩ) or raise the floor.")
        # ★ the refuse line must withstand stale_limit **blind pulses**.
        #   blind pulses = SETs allowed after a reading and before the next one. During those the gate
        #   cannot see the resistance dropping, so the margin between the refuse line and the point of no return
        #   must exceed the factor those pulses can remove in the worst case.
        nr = self.no_return_point()
        worst = PULSE_WORST_DROP ** self.stale_limit
        if self.r_floor_guard < nr * worst:
            raise SafetyViolation(
                f"Envelope inconsistent: the refuse line {self.r_floor_guard/1e3:.1f} kΩ cannot withstand "
                f"{self.stale_limit} blind pulses.\n"
                f"  point of no return = {nr/1e3:.2f} kΩ"
                f" (at {self.i_safe*1e6:.0f} µA erasing requires R∥Rs ≥ "
                f"{V_TH/self.i_safe/1e3:.1f} kΩ)\n"
                f"  {self.stale_limit} blind pulses lower it by up to {worst:.1f}× ⇒ "
                f"from the refuse line down to {self.r_floor_guard/worst/1e3:.2f} kΩ"
                f" < point of no return.\n"
                f"  Either raise the refuse line to ≥ {nr*worst/1e3:.1f} kΩ, "
                f"or lower stale_limit.")

        # (3) single-device current: V_dev_max / r_floor_guard must be ≤ i_safe
        i_dev = self.v_dev_max / self.r_floor_guard
        if i_dev > self.i_safe * 1.001:
            raise SafetyViolation(
                f"Envelope inconsistent: a floor device at {self.v_dev_max:.2f} V alone draws "
                f"{i_dev*1e6:.1f} µA, above {self.i_safe*1e6:.0f} µA.\n"
                f"  In a parallel write the node voltage is the same for all devices, so this can really happen.\n"
                f"  Lower v_dev_max to ≤ {self.i_safe*self.r_floor_guard:.3f} V, "
                f"or raise the floor.")
        return margin

    # ------------------------------------------------------------------
    def no_return_point(self) -> float:
        """
        **Point of no return**: below this resistance the device can never be erased within this envelope.

        The condition is `I_safe·(R∥Rs) ≥ V_th`; solving for R:

            R ≥ 1 / (I_safe/V_th − 1/Rs)

        A denominator ≤ 0 means no resistance can be erased with this (I_safe, Rs) -- inf is returned.
        """
        k = self.i_safe / V_TH - 1.0 / self.rs
        return 1.0 / k if k > 0 else float("inf")

    def max_drive(self, r_branch: float | None, r_min_device: float | None) -> float:
        """
        Maximum drive amplitude (absolute) allowed for this pulse. The minimum of four constraints:

          (1) worst-case current limit   I_safe · Rs                     -- always in effect
          (2) device voltage limit       V_dev_max · (R_br+Rs)/R_br      -- only if the branch resistance is known
          (3) single-device current      I_safe·R_min · (R_br+Rs)/R_br   -- only if the smallest device is known
          (4) AWG range                  5 V

        Without resistance information only (1) and (4) remain -- the **safe side**: without knowing the devices, trust only the current limit.
        """
        v = min(self.i_safe * self.rs, _L.AWG_ABS_MAX)
        if r_branch is not None and r_branch > 0:
            div = (r_branch + self.rs) / r_branch          # V_drive / V_dev
            v = min(v, self.v_dev_max * div)
            if r_min_device is not None and r_min_device > 0:
                v = min(v, self.i_safe * r_min_device * div)
        return v

    def v_device(self, amplitude: float, r_branch: float) -> float:
        """Voltage at the device node for this drive amplitude."""
        if r_branch <= 0:
            return 0.0
        return abs(amplitude) * r_branch / (r_branch + self.rs)

    def describe(self) -> str:
        m = self.self_check()
        v_floor = min(self.i_safe * _par(self.r_floor_guard, self.rs), self.v_dev_max)
        v_hrs = min(self.i_safe * _par(1e6, self.rs), self.v_dev_max)
        return "\n".join([
            f"Safety envelope (more conservative than the manual)",
            f"  Series resistance Rs     {self.rs/1e3:.1f} kΩ"
            + ("" if self.rs >= RS_RECOMMENDED else
               f"   (recommended {RS_RECOMMENDED/1e3:.0f} kΩ, used by the manual's endurance test)"),
            f"  Compliance I_safe        {self.i_safe*1e6:.0f} µA"
            f"      (manual type W rating {_L.I_ABS_MAX['W']*1e3:.0f} mA, narrowed "
            f"{_L.I_ABS_MAX['W']/self.i_safe:.0f}×)",
            f"  Device voltage limit     {self.v_dev_max:.2f} V"
            f"      (manual switching range {_L.V_SWITCH_LO:.2f}~{_L.V_SWITCH_HI:.2f} V)",
            f"  Drive amplitude limit    {min(self.i_safe*self.rs, _L.AWG_ABS_MAX):.3f} V"
            f"   (= I_safe × Rs, for R→0)",
            f"  Safe resistance window   {self.r_floor_guard/1e3:.0f} kΩ ~ "
            f"{self.r_ceiling/1e6:.1f} MΩ"
            f"   (refuse line = point of no return × {self.r_floor_guard/self.no_return_point():.1f}, withstands {self.stale_limit} blind pulses)",
            f"  Floor device gets        {v_floor:.3f} V = {m:.2f}× threshold  ← must be > 1.25",
            f"  1 MΩ device gets         {v_hrs:.3f} V = {v_hrs/V_TH:.2f}× threshold",
            f"  Point of no return       {self.no_return_point()/1e3:.2f} kΩ"
            f"   refuse line leaves "
            f"{self.r_floor_guard/self.no_return_point():.1f}× margin",
            f"  Blind-pulse limit        {self.stale_limit} pulses"
            f" (fresh reading required before every SET; "
            f"{self.stale_limit} pulses lower by up to "
            f"{PULSE_WORST_DROP**self.stale_limit:.1f}×)",
            f"  Pulse width limit        forward {self.width_max_fwd_us:.0f} µs / "
            f"reverse {self.width_max_rev_us:.0f} µs",
            f"  Pulse budget             per device {self.budget_per_device:,} / "
            f"per session {self.budget_per_session:,}",
        ])


def _par(a: float, b: float) -> float:
    """Parallel resistance."""
    return a * b / (a + b)


# ===========================================================================
# cross-session chip records
# ===========================================================================
# record directory. Defaults to inside the repository; tests point it to a temporary directory,
# so regression tests do not pile __test_*.json into chip_journal/,
# nor become dependent on run order by inheriting the previous round's quarantine list.
JOURNAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "chip_journal")


def journal_path(chip_id: str) -> str:
    base = JOURNAL_DIR
    os.makedirs(base, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in chip_id)
    return os.path.join(base, f"{safe}.json")


def load_journal(chip_id: str) -> dict:
    p = journal_path(chip_id)
    if not os.path.exists(p):
        return {"chip_id": chip_id, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sessions": [], "quarantined": [], "r_best": {}, "r_last": {},
                "pulses": {}}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_journal(j: dict):
    p = journal_path(j["chip_id"])
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(j, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)


# ===========================================================================
# gate
# ===========================================================================
class SafetyGate:
    """
    Stateful protection gate. One per hardware session; every action applying voltage passes through it first.

    It has only four public methods:
        authorize_write(devices, amplitude, width_us, forward, n)  → clipped amplitude
        authorize_read(devices, amplitude)                        → clipped amplitude
        note_resistance(device, r)                                → feed back a reading
        note_branch(devices, r_branch)                            → feed back a branch reading
    """

    def __init__(self, env: Envelope, chip_id: str, purpose: str = "",
                 n_devices: int = 16, verbose: bool = True,
                 audit: bool = True):
        self.env = env
        self.chip_id = str(chip_id)
        self.purpose = str(purpose)
        self.n = int(n_devices)
        self.verbose = verbose

        self.r_last: dict[int, float] = {}
        # **lower bound** given by a branch reading: R_i ≥ r_bound[i] (the parallel resistance is smaller than any member).
        # like r_last it satisfies the floor check, only more conservatively.
        self.r_bound: dict[int, float] = {}
        # most recently measured parallel branch resistance of a device set, keyed by frozenset.
        # when writing the same set it is used directly for the divider -- the physically exact value.
        self.r_branch_last: dict[frozenset, float] = {}
        # last few feedback records, used for self-diagnosis on refusal: (time, device, resistance, accepted)
        self.read_log: list[tuple] = []
        self.r_min_seen: dict[int, float] = {}
        self.stale: dict[int, int] = {}
        self.pulses: dict[int, int] = {}
        self.pulses_total = 0
        self.reads_total = 0
        self.clamped = 0
        self.refused = 0
        self.quarantined: set[int] = set()
        self.tripped: str | None = None
        self.t0 = time.time()

        # the quarantine list of the previous session is inherited directly -- bad devices do not get better by restarting
        self.journal = load_journal(self.chip_id)
        for i in self.journal.get("quarantined", []):
            self.quarantined.add(int(i))
        for k, v in self.journal.get("r_last", {}).items():
            self.r_last[int(k)] = float(v)
        for k, v in self.journal.get("pulses", {}).items():
            self.pulses[int(k)] = int(v)
        # inherited readings all count as "not fresh"; they must be reread before SET
        for i in list(self.r_last):
            self.stale[i] = env.stale_limit

        self._audit = None
        if audit:
            self._open_audit()
        if verbose:
            print(env.describe())
            if self.quarantined:
                print(f"  ★ Quarantine list inherited from the record: devices "
                      f"{sorted(self.quarantined)} -- no SET will be sent to them in this session.")

    # ------------------------------------------------------------------ audit
    def _open_audit(self):
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit")
        os.makedirs(base, exist_ok=True)
        p = os.path.join(base, f"{time.strftime('%Y%m%d')}_{self.chip_id}.csv")
        new = not os.path.exists(p)
        self._audit_f = open(p, "a", newline="", encoding="utf-8")
        self._audit = csv.writer(self._audit_f)
        if new:
            self._audit.writerow([
                "t", "purpose", "op", "devices", "dir", "v_req", "v_out",
                "width_us", "n", "r_branch", "r_min_dev", "v_dev",
                "i_worst_uA", "i_dev_uA", "note"])
        self.audit_path = p

    def _log(self, op, devices, direction, v_req, v_out, width_us, n,
             r_branch, r_min_dev, note=""):
        if self._audit is None:
            return
        v_dev = self.env.v_device(v_out, r_branch) if r_branch else ""
        i_dev = (v_dev / r_min_dev * 1e6) if (v_dev and r_min_dev) else ""
        self._audit.writerow([
            f"{time.time()-self.t0:9.3f}", self.purpose, op,
            " ".join(str(d) for d in devices), direction,
            f"{v_req:+.4f}", f"{v_out:+.4f}", f"{width_us:.1f}", n,
            f"{r_branch:.0f}" if r_branch else "",
            f"{r_min_dev:.0f}" if r_min_dev else "",
            f"{v_dev:.4f}" if v_dev else "",
            f"{abs(v_out)/self.env.rs*1e6:.1f}",
            f"{i_dev:.1f}" if i_dev else "", note])
        self._audit_f.flush()

    # ------------------------------------------------------------------ trip
    def trip(self, why: str):
        """Trip. No further voltage may be applied in this process; **there is no reset interface**."""
        if self.tripped is None:
            self.tripped = why
            self._log("TRIP", [], "", 0, 0, 0, 0, None, None, why)
            self.flush_journal()
            if self.verbose:
                print(f"\n★★★ Protection gate tripped: {why}\n"
                      f"    No further voltage may be applied in this process. Power down, look at "
                      f"{getattr(self, 'audit_path', 'audit/')}, "
                      f"identify the cause, then restart and redo preflight.\n")
        raise GateTripped(self.tripped)

    def _check_live(self):
        if self.tripped is not None:
            raise GateTripped(self.tripped)

    def _refuse(self, msg: str, op="write", devices=(), v_req=0.0):
        self.refused += 1
        self._log(op, devices, "", v_req, 0.0, 0, 0, None, None, "REFUSED " + msg)
        if devices:
            msg = msg + "\n\n" + self.state_dump(devices)
        raise SafetyViolation(msg)

    def state_dump(self, devices=()) -> str:
        """
        On refusal, lay out the gate's own state -- otherwise one can only guess which stage did not refresh.

        Meaning of the three columns:
          r_last  direct per-device reading (preflight's read_all and single-synapse FLV use this)
          r_bound **lower bound** from a branch reading (multi-device FLV uses this)
          stale   SET pulses received since the last refresh (limit = stale_limit)
        """
        devs = sorted({int(d) for d in devices}) or sorted(
            set(self.r_last) | set(self.r_bound))
        out = ["Gate state (snapshot at refusal)",
               f"  refuse line {self.env.r_floor_guard/1e3:.0f} kΩ   "
               f"blind-pulse limit {self.env.stale_limit}",
               "  Device r_last      r_bound     stale  quarantined"]

        def f(x):
            if x is None:
                return "    —    "
            if x >= 1e6:
                return f"{x/1e6:7.2f} MΩ"
            return f"{x/1e3:7.1f} kΩ"

        for d in devs:
            out.append(f"  {d:4d}  {f(self.r_last.get(d))}  "
                       f"{f(self.r_bound.get(d))}  "
                       f"{self.stale.get(d, 0):5d}  "
                       f"{'yes' if d in self.quarantined else ''}")
        if self.read_log:
            out.append("  Recent feedback (time / devices / branch resistance / refreshed):")
            for t, dv, r, ok in self.read_log[-8:]:
                out.append(f"    {t:7.2f}s  {list(dv)}  {f(r)}  "
                           f"{'refreshed' if ok else '**below refuse line, not refreshed**'}")
        nrej = getattr(self, "_branch_reject", 0)
        if nrej:
            out.append(f"  ★ {nrej} branch readings were not refreshed because they were below the refuse line -- "
                       f"this is the source of 'reading stale'.")
        return "\n".join(out)

    # ------------------------------------------------------------------ feed back readings
    def note_resistance(self, device: int, r: float, quiet: bool = False):
        """
        Feed back a **single-device** reading. Should be called after every read.

        This is the only place that marks a device "fresh" -- so without reading there is no SET.
        """
        self.reads_total += 1
        if r is None or not math.isfinite(r) or r <= 0:
            return                      # garbage readings do not update state or refresh freshness
        d = int(device)
        self.r_last[d] = float(r)
        self.stale[d] = 0
        # also recorded in read_log: otherwise per-device refreshes are invisible in the state snapshot,
        # and half the table is missing when finding out "which stage did not refresh".
        self.read_log.append((time.time() - self.t0, (d,), float(r), True))
        del self.read_log[:-40]
        prev = self.r_min_seen.get(d)
        if prev is None or r < prev:
            self.r_min_seen[d] = float(r)

        if r < self.env.r_floor_guard:
            if d not in self.quarantined:
                self.quarantined.add(d)
                if not quiet and self.verbose:
                    print(f"  ★ Device {d} read {r/1e3:.2f} kΩ, below the refuse line "
                          f"{self.env.r_floor_guard/1e3:.0f} kΩ ⇒ **quarantined**; "
                          f"no SET will be sent to it now or later.")
                self._log("QUARANTINE", [d], "", 0, 0, 0, 0, r, r,
                          f"r={r:.0f} < guard")
                self.flush_journal()
            if len(self.quarantined) >= QUARANTINE_ABORT:
                self.trip(f"{len(self.quarantined)} devices dropped below the refuse line "
                          f"({sorted(self.quarantined)}); this is a systematic problem, "
                          f"not bad luck with a single device")

    def note_branch(self, devices, r_branch: float):
        """
        Feed back a **branch** reading (ktram's G_A / G_B are this).

        ★ This function at first only filed the reading **without refreshing freshness** -- so 3×3 bar
          training locked up after `stale_limit` epochs: `train()` did read before every write,
          but those were branch readings of 2~3 devices and were treated as "not read".
          The error on real hardware was: "devices [8, 11, 13] ... reading stale, refusing further SET".

        The correct reasoning (which I wrote in a comment at the time but did not implement):

            parallel resistance R_br = 1/(Σ 1/R_i) < min(R_i)   -- strictly smaller than any member
            ⇒ **R_br ≥ floor ⇒ every member ≥ floor**

        So a branch reading is already a **valid and conservative** floor check,
        and should refresh the freshness of all members. It gives a lower bound, not an estimate,
        so it is recorded in `r_bound` rather than `r_last`.

        If the branch is **below** the floor it does not refresh: then only "at least one member has a problem" is known,
        not which one, so they must be read one by one before writing can continue.
        """
        devs = sorted({int(d) for d in devices})
        if len(devs) == 1:
            return self.note_resistance(devs[0], r_branch)
        self.reads_total += 1
        if r_branch is None or not math.isfinite(r_branch) or r_branch <= 0:
            return
        r_branch = float(r_branch)
        self.r_branch_last[frozenset(devs)] = r_branch
        self._branch_last = (tuple(devs), r_branch)

        # ★★ three-part reasoning. The first version had only two parts, and the comment on the second was wrong:
        #    "R_branch < refuse line ⇒ at least one member has a problem" -- **does not hold**.
        #    the parallel resistance falls with the number of devices, regardless of member health: 3 of 50 kΩ in parallel are only
        #    16.7 kΩ, with every member fine. 3×3 bar patterns have 2~3 spikes, so
        #    devices below 60 kΩ fall into this trap, never refresh, and the second SET is always refused.
        #
        #      R_branch ≥ refuse line            ⇒ all members ≥ refuse line         (can refresh)
        #      R_branch < refuse line/N          ⇒ at least one member < refuse line (should alarm)
        #      refuse line/N ≤ R_branch < refuse line ⇒ **undecidable**, read per device
        n = len(devs)
        accepted = r_branch >= self.env.r_floor_guard
        definitely_bad = r_branch < self.env.r_floor_guard / n
        self.read_log.append((time.time() - self.t0, tuple(devs),
                              r_branch, accepted))
        del self.read_log[:-40]
        if accepted:
            for d in devs:
                prev = self.r_bound.get(d)
                if prev is None or r_branch > prev:
                    self.r_bound[d] = r_branch
                self.stale[d] = 0           # ← this is the line that was missing before
        elif definitely_bad:
            self._branch_bad = getattr(self, "_branch_bad", 0) + 1
            if self.verbose and self._branch_bad <= 5:
                print(f"  ★ Branch {devs} in parallel is only {r_branch/1e3:.2f} kΩ, "
                      f"below even refuse line/{n} = "
                      f"{self.env.r_floor_guard/n/1e3:.1f} kΩ ⇒ "
                      f"**at least one member is below the refuse line**.")
                print(f"    They must be read one by one to locate which, before deciding whether to quarantine.")
        else:
            # the undecidable middle range. Not bad news; this reading just lacks the resolution
            # to conclude "every member ≥ refuse line". The caller should add per-device reads.
            self._branch_undet = getattr(self, "_branch_undet", 0) + 1
            if self.verbose and self._branch_undet <= 2:
                print(f"  · Branch {devs} in parallel {r_branch/1e3:.1f} kΩ falls in "
                      f"the undecidable range "
                      f"[{self.env.r_floor_guard/n/1e3:.1f}, "
                      f"{self.env.r_floor_guard/1e3:.0f}) kΩ —— "
                      f"{n} devices in parallel naturally do this. ")
                print(f"    Per-device readings will be added automatically before SET ("
                      f"ktram.HardwareBackend._ensure_fresh).")

    # ------------------------------------------------------------------ core
    def _branch_resistance(self, devices) -> tuple[float | None, float | None]:
        """
        (parallel branch resistance, minimum device resistance used for clipping).

        Prefer **the most recently measured branch resistance of this device set** -- when writing the same set,
        that is the physically correct R for the divider. All members are ≥ it,
        so also using it as the "minimum device resistance" is conservative.

        Without a measured branch value, fall back to per device: the direct reading (r_last) first,
        otherwise the lower bound from a branch (r_bound). If any device has neither, return
        (None, None) -- the caller then gets only the conservative "trust only the current limit" amplitude.
        """
        devs = sorted({int(d) for d in devices})
        rb = self.r_branch_last.get(frozenset(devs))
        if rb and rb > 0:
            return rb, rb
        rs_dev = []
        for d in devs:
            r = self.r_last.get(d) or self.r_bound.get(d)
            if r is None:
                return None, None
            rs_dev.append(r)
        if not rs_dev:
            return None, None
        g = sum(1.0 / r for r in rs_dev)
        return (1.0 / g if g > 0 else None), min(rs_dev)

    def _floor_ok(self, d) -> tuple[bool, float | None]:
        """
        Whether device d has a basis that is **fresh and above the floor**.
        Returns (pass, the resistance used).

        Both direct readings and branch lower bounds count as valid -- the latter is a strict lower bound,
        so using it for the floor check is only stricter and never lets anything through wrongly.
        """
        if self.stale.get(d, 0) >= self.env.stale_limit:
            return False, None
        r_direct = self.r_last.get(d)
        r_bound = self.r_bound.get(d)
        if r_direct is None and r_bound is None:
            return False, None
        best = max(x for x in (r_direct, r_bound) if x is not None)
        return best >= self.env.r_floor_guard, best

    def certified(self, device: int) -> bool:
        """
        Whether this device currently has a basis that is **fresh and above the refuse line**.
        Callers (ktram / cartpole_hw) use it to decide whether to add a read before SET.
        """
        return self._floor_ok(int(device))[0]

    def authorize_read(self, devices, amplitude: float) -> float:
        """
        Read pulse. The amplitude is hard-clipped to v_read -- a read must never be able to rewrite.

        0.10 V is far below the 0.26 V threshold; even if a device drops to 1 kΩ,
        the current at 0.10 V is only 100/(1k+20k) ≈ 4.8 µA.
        """
        self._check_live()
        a = abs(float(amplitude))
        if a > self.env.v_read + 1e-9:
            self.clamped += 1
            if self.verbose and self.clamped <= 3:
                print(f"  [clip] read pulse {a:.3f} V → {self.env.v_read:.3f} V"
                      f" (reads may not exceed half the threshold)")
            a = self.env.v_read
        v = math.copysign(a, amplitude)
        self._authorized = (abs(v), time.time())
        self._log("read", devices, "r", amplitude, v, 0, 1, None, None)
        return v

    def authorize_write(self, devices, amplitude: float, width_us: float,
                        forward: bool, n: int = 1, branches=None) -> float:
        """
        Write pulse. Returns the amplitude only if all checks pass, otherwise raises SafetyViolation.

        forward=True means forward / SET / conductance rising -- **the direction that burns devices**;
        the manual's Critical Notes say verbatim "A forward applied voltage without
        current limiting will cause devices to enter a very low-resistance
        state and consequently burn out". So the floor applies only in this direction.

        branches : optional. ktram's Mode 2 has two branches A/B **each with its own series resistor**,
                   driven by the same W1. Passing [[side-A devices...], [side-B devices...]] lets the gate
                   compute the divider per branch; otherwise the 16 devices are treated as one branch (conservative but too tight).
        """
        self._check_live()
        devs = sorted({int(d) for d in devices})
        if not devs:
            self._refuse("Write pulse specifies no device -- not knowing whom it writes, it may not be sent.",
                         devices=devs, v_req=amplitude)
        for d in devs:
            if not 0 <= d < self.n:
                self._refuse(f"device index {d} out of range (0..{self.n-1})",
                             devices=devs, v_req=amplitude)

        direction = "SET" if forward else "RESET"
        w = float(width_us)
        n = int(n)

        # ---- pulse count and width limits: hard stop for runaway loops
        if n < 1 or n > self.env.n_per_call_max:
            self._refuse(
                f"{n} pulses in one call exceeds the limit of {self.env.n_per_call_max}."
                f"To send more, loop in the caller; every pulse passes the gate again.",
                devices=devs, v_req=amplitude)
        w_max = self.env.width_max_fwd_us if forward else self.env.width_max_rev_us
        if w <= 0 or w > w_max:
            self._refuse(
                f"{direction} pulse width {w:.0f} µs exceeds the limit of {w_max:.0f} µs."
                + ("Forward is the damaging direction; no long pulses." if forward else
                   "Reverse may rely on time, but not indefinitely."),
                devices=devs, v_req=amplitude)

        # ---- budget
        if self.pulses_total + n > self.env.budget_per_session:
            self.trip(f"Session pulse budget exhausted ({self.env.budget_per_session:,} pulses)."
                      f"A normal experiment should not need this many -- most likely a loop ran away.")
        for d in devs:
            if self.pulses.get(d, 0) + n > self.env.budget_per_device:
                self.trip(f"Cumulative pulse budget of device {d} exhausted"
                          f" ({self.env.budget_per_device:,} pulses, including previous sessions).")

        # ---- invariant (4): floor + freshness, SET only
        if forward:
            bad = [d for d in devs if d in self.quarantined]
            if bad:
                self._refuse(
                    f"Devices {bad} are on the quarantine list (once read below "
                    f"{self.env.r_floor_guard/1e3:.0f} kΩ); SET refused."
                    f"They can only be RESET, or bypassed as bad devices.",
                    devices=devs, v_req=amplitude)
            unknown = [d for d in devs
                       if d not in self.r_last and d not in self.r_bound]
            if unknown:
                self._refuse(
                    f"Devices {unknown} have no resistance basis; SET refused.\n"
                    f"  Read once (read_resistance / FLV) before writing -- "
                    f"pushing toward low resistance without knowing where the device is is not allowed; this is the premise of the floor rule.\n"
                    f"  Branch readings count as a basis too: the parallel resistance is a lower bound for every member, "
                    f"so one FLV is enough.",
                    devices=devs, v_req=amplitude)
            stale = [d for d in devs
                     if self.stale.get(d, 0) >= self.env.stale_limit]
            if stale:
                self._refuse(
                    f"Devices {stale} have received ≥ "
                    f"{self.env.stale_limit} SET pulses since the last reading; reading stale, refusing further SET.\n"
                    f"  Read again before writing. SET only goes down; blind pulses are walking a cliff in the dark.\n"
                    f"  (Branch readings refresh the freshness of all members -- the parallel resistance is a lower bound. "
                    f"If this still reports stale, check that the read mask and the write mask are the same device set.)",
                    devices=devs, v_req=amplitude)
            low = []
            for d in devs:
                ok, r = self._floor_ok(d)
                if not ok and r is not None:
                    low.append((d, r))
            if low:
                self._refuse(
                    "SET refused: " + ", ".join(
                        f"device {d} = {r/1e3:.2f} kΩ" for d, r in low)
                    + f" is already below the refuse line of {self.env.r_floor_guard/1e3:.0f} kΩ.\n"
                    f"  Going further approaches the point of no return"
                    f" (at {self.env.i_safe*1e6:.0f} µA erasing requires R∥Rs ≥ "
                    f"{V_TH/self.env.i_safe/1e3:.1f} kΩ).\n"
                    f"  The correct action is a RESET to push it back.",
                    devices=devs, v_req=amplitude)

        # ---- amplitude: computed per branch, take the tightest
        groups = [list(devs)] if not branches else [
            [int(d) for d in g] for g in branches if g]
        per = [self._branch_resistance(g) for g in groups]
        v_max = min(self.env.max_drive(rb, rm) for rb, rm in per)
        r_branch = min((rb for rb, _ in per if rb), default=None)
        r_min_dev = min((rm for _, rm in per if rm), default=None)
        a = abs(float(amplitude))
        note = ""
        if a > v_max + 1e-9:
            self.clamped += 1
            note = f"clamped from {a:.3f}"
            if self.verbose and self.clamped <= 5:
                detail = (f"R_branch={r_branch/1e3:.1f}kΩ R_min={r_min_dev/1e3:.1f}kΩ"
                          if r_branch else "resistance unknown ⇒ trust only the current limit")
                print(f"  [clip] {direction} {a:.3f} V → {v_max:.3f} V  ({detail})")
            a = v_max

        # what this pulse will really do -- record in the audit, refuse if necessary
        if r_branch:
            v_dev = self.env.v_device(a, r_branch)
            i_dev = v_dev / r_min_dev if r_min_dev else 0.0
            if i_dev > self.env.i_safe * 1.05:
                self._refuse(
                    f"Computed single-device current {i_dev*1e6:.1f} µA > "
                    f"{self.env.i_safe*1e6:.0f} µA -- envelope broken, not sending.",
                    devices=devs, v_req=amplitude)
            if forward and v_dev < V_TH * 0.6:
                # not being able to switch is not a safety issue, but firing many ineffective blind pulses wastes budget
                note += " below-threshold"

        # ---- bookkeeping
        self.pulses_total += n
        for d in devs:
            self.pulses[d] = self.pulses.get(d, 0) + n
            if forward:
                # after a SET the resistance goes down and the old reading is no longer accurate -- but it is **not cleared**,
                # only marked stale. Clearing it would make the next pulse "have no basis at all",
                # defeating the stale_limit mechanism; keeping the somewhat high old value is conservative for clipping
                # (the V_dev limit ∝ (R+Rs)/R loosens as R decreases).
                self.stale[d] = self.stale.get(d, 0) + n
        v = math.copysign(a, 1.0 if forward else -1.0)
        self._authorized = (abs(v), time.time())
        self._log(direction.lower(), devs, direction, amplitude, v, w, n,
                  r_branch, r_min_dev, note)
        return v

    # ------------------------------------------------------------------ open-circuit probe
    PROBE_MAX_V = 1.5           # V, amplitude limit allowed for the open-circuit probe

    def authorize_probe(self, dio_mask: int, amplitude: float) -> float:
        """
        **Open-circuit probe**: with all switches open, apply voltage to the series resistor itself.

        Why a voltage much higher than a read pulse (0.10 V) is allowed: with all switches open the circuit is

            W1 ──[ Rs ]── node ──[ scope input impedance ~1 MΩ ]── GND

        **No device is connected.** At 1.5 V the current is only 1.5 µA, and it does not flow through the chip.
        So this is not a hole in the envelope; it is a measurement in a topology where no device is present.

        The price is that the switches must really be confirmed open -- so the **DIO mask is checked** here,
        rather than trusting the caller saying "I opened them". Any non-zero mask is refused.

        Why this method exists: the divider difference of Rs against 1 MΩ is only 2% (with 20 kΩ, at 0.1 V
        only 1.96 mV), while the AD2 channel offset is tens of mV. Measuring Rs at 0.10 V
        measures the offset, not Rs. Raising the voltage to 1 V and doing a multi-point fit (the offset falls into the intercept)
        gives the signal-to-noise ratio needed.
        """
        self._check_live()
        if dio_mask & 0xFFFF:
            closed = [i for i in range(16) if (dio_mask >> i) & 1]
            self._refuse(
                f"The open-circuit probe requires all switches open, but devices {closed} are still connected in the DIO mask.\n"
                f"  The amplitude allowed by this authorisation (≤ {self.PROBE_MAX_V:.1f} V) is computed "
                f"for \"no device in the circuit\".\n"
                f"  Call all_off() before probing.",
                op="probe", v_req=amplitude)
        a = min(abs(float(amplitude)), self.PROBE_MAX_V)
        v = math.copysign(a, amplitude)
        self._authorized = (abs(v), time.time())
        self._log("probe", [], "open", amplitude, v, 0, 1, None, None,
                  "no device in circuit")
        return v

    # ------------------------------------------------------------------ sweep
    SWEEP_MAX_S = 10.0          # total duration limit of one sweep

    def authorize_sweep(self, devices, amplitude: float, duration_s: float):
        """
        Slow sweep / hysteresis loop. Handled separately from pulses, because **the gate cannot intervene during a sweep**.

        Rules:
          · single device only (a parallel sweep has no diagnostic value and divides the voltage by |P|)
          · like SET, requires a fresh reading, not quarantined, above the floor
          · amplitude authorised for the **most pessimistic resistance (= refuse line)**: assume the device drops all the way to the floor,
            so the device-voltage layer holds throughout the sweep
          · total duration capped, and counted against the pulse budget in 50 µs equivalents
        """
        self._check_live()
        devs = sorted({int(d) for d in devices})
        if len(devs) != 1:
            self._refuse(
                f"A sweep allows only a single connected device; connected now: {devs}.\n"
                f"  A parallel sweep has no diagnostic value (the voltage is split by |P|), "
                f"and the gate cannot tell which device is changing.",
                op="sweep", devices=devs, v_req=amplitude)
        d = devs[0]
        if duration_s <= 0 or duration_s > self.SWEEP_MAX_S:
            self._refuse(f"Sweep duration {duration_s:.2f} s exceeds the limit of "
                         f"{self.SWEEP_MAX_S:.0f} s.", op="sweep", devices=devs)
        if d in self.quarantined:
            self._refuse(f"Device {d} is on the quarantine list; sweep refused"
                         f" (a bipolar sweep always passes through the SET direction).",
                         op="sweep", devices=devs, v_req=amplitude)
        r = self.r_last.get(d)
        if r is None or self.stale.get(d, 0) >= self.env.stale_limit:
            self._refuse(f"Device {d} has no fresh resistance reading; sweep refused. Read once first.",
                         op="sweep", devices=devs, v_req=amplitude)
        if r < self.env.r_floor_guard:
            self._refuse(f"Device {d} = {r/1e3:.2f} kΩ is below the refuse line; sweep refused."
                         f"Push it back with a RESET first.",
                         op="sweep", devices=devs, v_req=amplitude)

        # pessimistic authorisation: assume the device is at the refuse line at the end of the sweep
        rp = self.env.r_floor_guard
        v_max = self.env.max_drive(rp, rp)
        a = min(abs(float(amplitude)), v_max)
        if a < abs(amplitude) - 1e-9:
            self.clamped += 1
            if self.verbose:
                print(f"  [clip] sweep {abs(amplitude):.3f} V → {a:.3f} V"
                      f" (computed for the most pessimistic resistance {rp/1e3:.0f} kΩ; "
                      f"current ≤ {self.env.i_safe*1e6:.0f} µA throughout)")
        equiv = max(1, int(duration_s / 50e-6))
        if self.pulses_total + equiv > self.env.budget_per_session:
            self.trip("Session budget exhausted (sweeps counted in 50 µs equivalents).")
        self.pulses_total += equiv
        self.pulses[d] = self.pulses.get(d, 0) + equiv
        self.stale[d] = self.env.stale_limit     # after a sweep the reading is necessarily stale
        self._authorized = (a, time.time())
        self._log("sweep", devs, "BIPOLAR", amplitude, a, duration_s * 1e6, 1,
                  r, r, f"pessimistic r={rp:.0f} equiv={equiv}")
        return a

    # ------------------------------------------------------------------ token
    def consume_token(self, peak: float, what: str = "start_custom"):
        """
        Called by the driver layer before actually issuing a waveform. Raises if there is no freshly issued authorisation --
        this closes the direct-call path of `md.start_wave()` / `md.start_custom()`.
        """
        self._check_live()
        tok = getattr(self, "_authorized", None)
        if tok is None:
            self._refuse(
                f"{what}() tried to issue a waveform without gate authorisation.\n"
                f"  All voltage application must first pass authorize_write/authorize_read.\n"
                f"  (ktram.HardwareBackend._fire used to bypass clipping from here.)",
                op=what)
        allowed, t = tok
        if time.time() - t > 10.0:
            self._authorized = None
            self._refuse(f"The authorisation used by {what}() has expired (> 2 s); request again.",
                         op=what)
        if abs(peak) > allowed + 1e-6:
            self._authorized = None
            self._refuse(
                f"{what}() would issue {abs(peak):.3f} V, but the gate only authorised "
                f"{allowed:.3f} V. The waveform was modified after authorisation -- not sending.", op=what)
        self._authorized = None

    # ------------------------------------------------------------------ wrap-up
    def flush_journal(self):
        j = self.journal
        j["quarantined"] = sorted(self.quarantined)
        j["r_last"] = {str(k): v for k, v in self.r_last.items()}
        j["r_best"] = {str(k): max(v, float(j.get("r_best", {}).get(str(k), 0)))
                       for k, v in self.r_last.items()}
        j["pulses"] = {str(k): v for k, v in self.pulses.items()}
        save_journal(j)

    def close(self, note: str = ""):
        self.journal.setdefault("sessions", []).append({
            "t": time.strftime("%Y-%m-%d %H:%M:%S"),
            "purpose": self.purpose,
            "rs": self.env.rs, "i_safe": self.env.i_safe,
            "v_dev_max": self.env.v_dev_max,
            "pulses": self.pulses_total, "reads": self.reads_total,
            "clamped": self.clamped, "refused": self.refused,
            "tripped": self.tripped, "note": note,
        })
        self.flush_journal()
        if self._audit is not None:
            self._audit_f.close()
            self._audit = None

    def report(self) -> str:
        live = [i for i in range(self.n) if i not in self.quarantined]
        rs = [self.r_last[i] for i in live if i in self.r_last]
        lines = [
            f"Session summary ({self.purpose or 'unnamed'})",
            f"  Pulses {self.pulses_total:,} (budget {self.env.budget_per_session:,})"
            f"  Reads {self.reads_total:,}",
            f"  Clipped {self.clamped}   Refused {self.refused}   "
            f"Tripped {'yes: ' + self.tripped if self.tripped else 'no'}",
        ]
        if rs:
            lines.append(f"  Live devices {len(live)}/{self.n}, resistance "
                         f"{min(rs)/1e3:.1f} ~ {max(rs)/1e3:.0f} kΩ"
                         f" (floor {self.env.r_floor_guard/1e3:.0f} kΩ)")
        if self.quarantined:
            lines.append(f"  ★ Quarantined: {sorted(self.quarantined)}")
        if self.refused == 0 and self.clamped == 0:
            lines.append("  The gate never had to act -- the script's own amplitudes were within the envelope, which is good.")
        return "\n".join(lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        if self.verbose:
            print(self.report())
        return False


# ===========================================================================
if __name__ == "__main__":
    print("=" * 72)
    for rs in (10e3, 20e3, 50e3):
        print()
        try:
            env = Envelope.for_rs(rs)
            print(env.describe())
        except SafetyViolation as e:
            print(f"Rs = {rs/1e3:.0f} kΩ refused:\n{e}")
    print()
    print("=" * 72)
    print("Point of no return (erasable only if I·(R∥Rs) ≥ V_th = 0.26 V):")
    for rs in (10e3, 20e3, 50e3):
        for i in (50e-6, 100e-6):
            # R∥Rs ≥ V_th/I  ⇒  R ≥ 1/(I/V_th − 1/Rs)
            k = i / V_TH - 1.0 / rs
            r = 1.0 / k if k > 0 else float("inf")
            print(f"  Rs={rs/1e3:4.0f} kΩ  I={i*1e6:3.0f} µA  →  "
                  f"R ≥ {r/1e3:6.2f} kΩ" if math.isfinite(r) else
                  f"  Rs={rs/1e3:4.0f} kΩ  I={i*1e6:3.0f} µA  →  "
                  f"**no resistance can be erased**")
    k = I_SAFE / V_TH - 1.0 / RS_MIN
    worst = 1.0 / k if k > 0 else float("inf")
    print(f"\nThe worst line inside the envelope is Rs={RS_MIN/1e3:.0f} kΩ / "
          f"I={I_SAFE*1e6:.0f} µA → {worst/1e3:.2f} kΩ.")
    print(f"The refuse line is set at {R_FLOOR_GUARD/1e3:.0f} kΩ, still "
          f"{R_FLOOR_GUARD/worst:.2f}× margin; the floor {R_FLOOR/1e3:.0f} kΩ has "
          f"{R_FLOOR/worst:.2f}×.")
