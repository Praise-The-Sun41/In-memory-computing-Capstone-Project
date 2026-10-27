"""
device_limits.py —— voltage/current limits of Knowm SDC memristors (single source of truth for the whole repository)

All numbers come from **Knowm Self Directed Channel Memristors, Rev 3.2, 2019-10-06**.
Before changing any of them, go back to the manual; do not write a separate set at the call site.

=================== What the manual says ===================

Critical Notes / Warnings (page 4):

  · **Limit Device Current** —— "Set a compliance current or use series resistance.
    A forward applied voltage **without current limiting will cause devices to
    enter a very low-resistance state and consequently burn out**."
  · **Limit Applied Voltage** —— "Formed devices typically change resistance
    between **0.1 and 0.75 Volts** and are intended to be **normally operated
    under 1V**. High voltages may induce damage, especially if current is not
    limited."

Maximum Ratings (page 5) + Note 1:

  | Max forward voltage | 3 V |
  | Max reverse voltage | −5 V |
  | Max current W/Sn/Cr | 1 mA |
  | Max current C | 0.05 mA |

  Note 1: "**High Resistant State (HRS) will be irreversibly reduced from 1 MΩ
  to 100 kΩ or less if operated at these currents**".

**This is irreversible.** Figure 4 (HRS/LRS vs write compliance current) gives roughly:

    Compliance  100 nA    1 µA    10 µA    100 µA    1 mA
    HRS(W)       ~1e7     ~1e7    ~1e6     ~3e5      ~1e5
    HRS(Sn/Cr)   ~1e12    ~1e11   ~1e11    ~1e10     ~1e5

    → **The compliance current directly sets how high a high-resistance state you can get**, and so how wide the conductance window is.
      A narrow window is not necessarily a poor device; it may have been narrowed by your own write current, irreversibly.

=================== How to limit ===================

The board has no active current limiting; **the series resistor is the current limiter**. So:

    I = V_drive / (Rs + R_device)

**The limit must assume the worst case R_device → 0**, not the currently measured R:

    V_drive ≤ I_limit × Rs

The reason is the manual's sentence -- the burn-out mechanism is precisely "the device suddenly dropping into the low-resistance
state during the pulse". Calculating with the currently measured R fails exactly when protection is most needed.

Rs = 10 kΩ, I_limit = 100 µA ⇒ drive at most 1.0 V. Then:

    R = 10 kΩ → device gets 0.50 V   (threshold 0.15~0.35 V, enough)
    R = 100 kΩ → device gets 0.91 V   (still within the manual's "under 1 V")

=================== An identity that makes this clear ===================

Combining the two constraints and eliminating V_drive:

    V_dev = V_drive · R/(R+Rs)   and   V_drive ≤ I_limit · Rs

    ⇒  **V_dev ≤ I_limit · (R ∥ Rs)**

This line determines everything, and the conclusion is **counter-intuitive**:

  · **High-resistance device** (R ≫ Rs): R∥Rs ≈ Rs ⇒ V_dev ≤ I_limit·Rs.
    The larger the series resistor, the higher the device voltage available -- that is why the manual uses 50 kΩ.
  · **Low-resistance device** (R ≪ Rs): R∥Rs ≈ R ⇒ **V_dev ≤ I_limit·R, independent of Rs**.
    Changing the series resistor does nothing. A 3 kΩ device at 100 µA gets at most 0.3 V,
    barely above the typical 0.26 V threshold -- this is the quantitative version of manual section 12.4's
    "the lower the resistance, the harder to erase".

So when "a low-resistance device cannot be moved", **raising the current is the only immediately effective remedy, and it is
exactly the source of irreversible damage**. The right approach is a longer pulse width or a slow DC sweep, accumulating charge
at low current, rather than raising the current.
"""
from __future__ import annotations

# ---------------------------------------------------------------- absolute maximum ratings
V_FWD_ABS_MAX = 3.0            # V, forward (manual Maximum Ratings)
V_REV_ABS_MAX = 5.0            # V, absolute value in reverse
I_ABS_MAX = {                  # A, by dopant type
    "W": 1.0e-3, "Sn": 1.0e-3, "Cr": 1.0e-3, "C": 50e-6,
}

# ---------------------------------------------------------------- normal operating range
V_OPERATE_MAX = 1.0            # "normally operated under 1V"
V_SWITCH_LO = 0.10             # "typically change resistance between
V_SWITCH_HI = 0.75             #  0.1 and 0.75 Volts"

# typical thresholds (page 6, DC/quasi-static)
V_TH_FWD = {"W": 0.26, "Sn": 0.26, "C": 0.26, "Cr": 0.33}
V_TH_REV = {"W": -0.11, "Sn": -0.11, "C": -0.11, "Cr": -0.19}

# ---------------------------------------------------------------- default operating point
# conservative but still reliably writable: device voltage at the top of the manual's switching range, current at 100 µA.
V_DEV_MAX_DEFAULT = 0.75       # V, limit applied to the **device**
I_COMPLIANCE_DEFAULT = 100e-6  # A, current limit in the worst case (R→0)
V_READ_DEFAULT = 0.10          # V, read pulse; well below threshold, non-disturbing

AWG_ABS_MAX = 5.0              # AD2 W1 output range, hard limit

DOPANT_DEFAULT = "W"           # chip type on this board; type C has a 20× lower current limit


# ---------------------------------------------------------------- approximation of figure 4
_HRS_TABLE = [(1e-7, 1e7), (1e-6, 1e7), (1e-5, 1e6), (1e-4, 3e5), (1e-3, 1e5)]


def expected_hrs(i_limit: float) -> float:
    """Estimate the high-resistance state obtainable at this compliance current from manual figure 4 (type W, order of magnitude)."""
    if i_limit <= _HRS_TABLE[0][0]:
        return _HRS_TABLE[0][1]
    for (i0, h0), (i1, h1) in zip(_HRS_TABLE, _HRS_TABLE[1:]):
        if i_limit <= i1:
            import math
            f = (math.log(i_limit) - math.log(i0)) / (math.log(i1) - math.log(i0))
            return math.exp(math.log(h0) + f * (math.log(h1) - math.log(h0)))
    return _HRS_TABLE[-1][1]


# ---------------------------------------------------------------- core
def max_drive(rs: float, i_limit: float = I_COMPLIANCE_DEFAULT,
              v_dev_max: float = V_DEV_MAX_DEFAULT,
              r_dev: float | None = None,
              awg_max: float = AWG_ABS_MAX) -> float:
    """
    **Maximum drive voltage** allowed with this series resistor. The minimum of three constraints:

      1. current (worst case R→0)                V ≤ i_limit · Rs
      2. device voltage (meaningful only if R known) V ≤ v_dev_max · (R+Rs)/R
      3. AWG range                                V ≤ awg_max

    With r_dev=None only 1 and 3 are used -- this is the **safe side**: if the device resistance is unknown, trust only the current limit.
    """
    v = min(i_limit * rs, awg_max)
    if r_dev is not None and r_dev > 0:
        v = min(v, v_dev_max * (r_dev + rs) / r_dev)
    return v


def clamp_drive(amplitude: float, rs: float,
                i_limit: float = I_COMPLIANCE_DEFAULT,
                v_dev_max: float = V_DEV_MAX_DEFAULT,
                r_dev: float | None = None,
                dopant: str = DOPANT_DEFAULT) -> tuple[float, str | None]:
    """
    Clip a drive amplitude into the safe range. Returns (clipped amplitude, description of the limit triggered or None).

    The sign is kept: absolute maximum ratings differ for forward and reverse (+3 V / −5 V).
    """
    sign = 1.0 if amplitude >= 0 else -1.0
    a = abs(float(amplitude))
    why = None

    lim_abs = i_limit * rs
    if a > lim_abs:
        a, why = lim_abs, (f"current compliance {i_limit*1e6:.0f} µA × Rs {rs/1e3:.0f}kΩ "
                           f"⇒ drive ≤ {lim_abs:.2f} V")

    if r_dev is not None and r_dev > 0:
        lim_dev = v_dev_max * (r_dev + rs) / r_dev
        if a > lim_dev:
            a, why = lim_dev, (f"device voltage limit {v_dev_max:.2f} V "
                               f"(R={r_dev/1e3:.1f}kΩ) ⇒ drive ≤ {lim_dev:.2f} V")

    # absolute maximum ratings: even if everything above allows it, the manual's Maximum Ratings must not be exceeded
    v_abs = V_FWD_ABS_MAX if sign > 0 else V_REV_ABS_MAX
    if r_dev is not None and r_dev > 0:
        v_abs = v_abs * (r_dev + rs) / r_dev
    if a > min(v_abs, AWG_ABS_MAX):
        a = min(v_abs, AWG_ABS_MAX)
        why = f"absolute maximum rating {'+3' if sign > 0 else '−5'} V (manual Maximum Ratings)"

    return sign * a, why


def max_device_voltage(r_dev: float, rs: float,
                       i_limit: float = I_COMPLIANCE_DEFAULT) -> float:
    """
    The maximum voltage a device can get at this compliance current: **V_dev ≤ I_limit · (R ∥ Rs)**.

    For a low-resistance device (R ≪ Rs) it reduces to I_limit·R -- independent of the series resistor; changing it does nothing.
    """
    if r_dev <= 0:
        return 0.0
    return i_limit * (r_dev * rs) / (r_dev + rs)


def worst_case_current(amplitude: float, rs: float) -> float:
    """The current this drive amplitude would inject in the worst case (device dropping to 0 Ω)."""
    return abs(amplitude) / max(rs, 1e-9)


def describe(rs: float, i_limit: float = I_COMPLIANCE_DEFAULT,
             v_dev_max: float = V_DEV_MAX_DEFAULT,
             dopant: str = DOPANT_DEFAULT) -> str:
    """Print one line before starting, stating the limits and costs of this run."""
    vmax = max_drive(rs, i_limit, v_dev_max)
    hrs = expected_hrs(i_limit)
    lines = [
        f"Limits: device voltage ≤ {v_dev_max:.2f} V, compliance current ≤ {i_limit*1e6:.0f} µA"
        f" (worst case R→0) ⇒ drive ≤ {vmax:.2f} V @ Rs={rs/1e3:.0f}kΩ",
        f"      manual absolute maximum: +{V_FWD_ABS_MAX:.0f} V / −{V_REV_ABS_MAX:.0f} V, "
        f"type {dopant} current {I_ABS_MAX.get(dopant, 1e-3)*1e3:.2f} mA",
        f"      expected high-resistance state at this compliance current about {hrs:.0e} Ω (manual figure 4)",
    ]
    if i_limit > 200e-6:
        lines.append(f"      ★ compliance current > 200 µA: manual Note 1 says this **irreversibly** "
                     f"lowers HRS from 1 MΩ to below 100 kΩ.")
    if rs < 20e3 and i_limit >= 100e-6:
        lines.append(f"      ★ Rs is only {rs/1e3:.0f} kΩ. To have both "
                     f"'low current' and 'high enough device voltage', use a larger series resistor -- "
                     f"the manual's own endurance test uses 50 kΩ.")
    return "\n".join(lines)
