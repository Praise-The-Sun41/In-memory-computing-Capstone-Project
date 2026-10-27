"""
hw_session.py —— **the only entry point for getting a real board.**

Previously a dozen scripts each called `MemristorDiscovery(series_resistance=...)`,
each with its own defaults and clipping habits, and some bypassed clipping and called start_custom directly.
A wrong series resistance, the wrong Mode, an excessive amplitude -- nothing anywhere would stop them.

Now there is only this path:

    import hw_session
    with hw_session.open_board(rs=20_000, chip_id="chip-2026-10",
                               purpose="bars", mode=2) as s:
        r = s.md.read_all()
        ...

`open_board` does the following in order, and hands over the board only if every step passes:

    1. Set up the envelope (safety.Envelope) -- Rs must be given explicitly; below 20 kΩ it is refused
    2. Open the AD2 and confirm ±5 V is up (otherwise the analog switches do not conduct and every read is open circuit)
    3. **Measure Rs** and compare with the declared value (all switches open, zero risk)
    4. Read the 16 devices one by one and compare with the history in chip_journal
    5. Put devices below the refuse line into the quarantine list; if any are below it, **do not start**
    6. Require the operator to type a confirmation once (unless the caller has already confirmed)

On exit, normal or not: open all switches → stop waveform → turn off ±5 V → close device
→ write back the chip record → print the session summary.

=================== Why Rs has no default ===================

Because it is **the whole of the current limiting**. There is no active current limiting on the board; current is V/(Rs+R).
If Rs is entered as 20 kΩ but the board has 10 kΩ, the real current is twice the setting,
and the program never reports any error -- exactly the failure that most easily and silently drags a chip down.

The Mode of the V2 board determines Rs:

    Mode 1     two 20 kΩ at A/B in parallel = 10 kΩ    → **below the lower limit; refused by this repository**
    Mode 2     each of the A/B branches 20 kΩ          → 20 kΩ   ✓
    modified   replaced by the 50 kΩ of the manual's endurance test → 50 kΩ   ✓ recommended

So: **the inverted pendulum uses single devices, but must also run under Mode 2's 20 kΩ (or the modified 50 kΩ)**,
connecting one path at a time. A single device + 20 kΩ has exactly the same envelope as a differential pair + 20 kΩ;
the only difference is how many paths are connected. The ktram differential pairs of the H/V 3×3 bar task are naturally in Mode 2.
"""

from __future__ import annotations

import sys
import time

import safety
from safety import Envelope, SafetyGate, SafetyViolation

__all__ = ["open_board", "HwSession", "preflight", "PreflightFailed"]


class PreflightFailed(RuntimeError):
    """Start-up self-test failed. Do not bypass it -- it stops exactly what nobody stopped last time."""


# ===========================================================================
class HwSession:
    """A hardware session. Obtain with `with`; powers down automatically on exit."""

    def __init__(self, md, gate, baseline, env):
        self.md = md
        self.gate = gate
        self.baseline = baseline        # resistances of the 16 devices at start-up (Ω)
        self.env = env

    @property
    def usable(self):
        """Indices of devices usable normally (including SET)."""
        return [i for i, r in enumerate(self.baseline)
                if i not in self.gate.quarantined
                and r is not None and self.env.r_floor_guard <= r <= self.env.r_ceiling]

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        note = "" if et is None else f"{et.__name__}: {ev}"
        try:
            self.md.all_off()
        except Exception:
            pass
        try:
            self.md.close()
        finally:
            self.gate.close(note)
            print()
            print(self.gate.report())
            print(f"  Audit log: {getattr(self.gate, 'audit_path', '(none)')}")
            print(f"  Chip record: {safety.journal_path(self.gate.chip_id)}")
        return False


# ===========================================================================
def preflight(md, gate, verbose=True, check_rs=True, rs_tol=0.25,
              min_usable=16, rs_source=None):
    """
    Start-up self-test. Returns the list of resistances of the 16 devices. Raises PreflightFailed on failure.

    The self-test itself uses only 0.10 V read pulses (38% of threshold), which do not disturb the chip.
    """
    env = gate.env
    if verbose:
        print("\nStart-up self-test")
        print("-" * 72)

    # ---- 1. power
    vp, ip, vn, inn = md.power_status()
    if vp < 4.0 or vn > -4.0:
        raise PreflightFailed(
            f"±5 V supply not established (V+={vp:+.2f}, V−={vn:+.2f}). "
            f"The analog switches will not conduct and every read is open circuit -- no judgement can be trusted now.")
    if verbose:
        print(f"  Power: V+ = {vp:+.2f} V, V− = {vn:+.2f} V  ✓")

    # ---- 2. actual Rs
    #
    # three outcomes must be handled separately. Previously only "pass/fail" existed, so "instrument not sensitive enough"
    # was reported as "board miswired" -- a false alarm that also locked people out.
    if check_rs:
        # ★★ I downgraded this section once; that was a wrong decision, and it had a cost:
        #   after the on-board measurement was changed to "not part of the decision", the user started with a declared 20 kΩ,
        #   while the real value was 9.5 kΩ -- the actual worst-case current was twice the envelope. The hard block is now restored.
        #
        #   The reason for the earlier downgrade was "the board cannot determine Rs in principle (with no current it cannot be observed)".
        #   **That argument was wrong**: the scope input impedance (1 MΩ) and node capacitance (180 pF)
        #   are themselves a load; the current is small but fully measurable -- at 1 V the drop is 10 mV, residual
        #   0.05 mV, R² = 0.985. Failing to measure it earlier was my own bug:
        #   the scope range followed the amplitude, and the AD2 offset changes with range.
        r = md.verify_rs_full(tol=rs_tol, verbose=verbose)
        st = r["status"]
        rs_m = r.get("rs_best")

        if st == "no_series_resistor":
            # ★★ The conclusion here is **valid only if the topology assumption of the no-load probe holds**,
            #    and measurements show that assumption does not hold in Mode 2. Recorded because it
            #    is the conclusion of this whole round of debugging:
            #
            #    Mode 1 probe: DC 9.3 kΩ (via R_scope) + transient 9.76 kΩ
            #      (via C_par) agree, and 20k∥20k = 10.0 kΩ --
            #      **exactly what two 20 kΩ in parallel look like**, i.e. both resistors are
            #      in the loop with correct values. This conclusion rests on two independently calibrated quantities.
            #    Mode 2 probe: measures only parasitic-level values, **directly contradicting** the above.
            #      It rests on a single topology assumption (scope 1+ is after the series resistor),
            #      and the Mode switch does change the wiring -- in Mode 1 scope 2+ measures the W1 source,
            #      in Mode 2 node B; if it can change 2+ it may also change 1+.
            #
            #    Agreement of two independent methods > one unverifiable topology assumption.
            #    So **this probe is not used for decisions in Mode 2**, only printed as a diagnostic;
            #    Rs must come from a **valid** measurement: the Mode 1 probe or a multimeter.
            #
            #    This does not relax any limit -- the real defences (current limit from the declared Rs, resistance floor,
            #    a fresh read before every SET, per-pulse audit) are unchanged throughout. Only a measurement shown to be
            #    invalid was replaced.
            rs_m = r["rs_best"]
            r_par = r.get("r_parasitic") or 310.0
            if verbose:
                print(f"  · The no-load probe measured {rs_m/1e3:.2f} kΩ (parasitic level, "
                      f"parasitic itself about {r_par:.0f} Ω).")
                print(f"    **This probe is not used for decisions in Mode 2** -- its topology assumption"
                      f" (scope 1+ after the series resistor)")
                print(f"    is refuted by the Mode 1 result: in Mode 1 two independent methods agree on "
                      f"9.3~9.8 kΩ ≈ 20k∥20k,")
                print(f"    showing that both 20 kΩ are in the loop. Rs is instead determined by the declared source.")
            if rs_source not in ("dmm", "nominal"):
                raise PreflightFailed("\n  ".join([
                    f"The no-load probe is unreliable in Mode 2 (measured {rs_m/1e3:.2f} kΩ = "
                    f"parasitic level), so the source of Rs must be declared explicitly.", "",
                    f"★ The most reliable way to verify the resistors is to **switch to Mode 1 and measure once**: then the two "
                    f"resistors are in parallel,",
                    f"  and the no-load probe is reliable in Mode 1 (two independently calibrated quantities agree).",
                    f"  A reading ≈ {env.rs/2/1e3:.0f} kΩ shows that both "
                    f"{env.rs/1e3:.0f} kΩ are present; Rs in Mode 2 is twice that.", "",
                    f"  python check_rs.py --rs {env.rs/2:.0f} "
                    f"--chip {gate.chip_id}      # Mode switch at 1",
                    "",
                    f"After confirming, continue with --rs-dmm (verified physically/in Mode 1).",
                ]))
            st = "probe_invalid_mode2"
            r["status"] = st

        if st == "mismatch" or st == "mismatch_rc":
            extra = []
            if rs_m and abs(rs_m - 10e3) / 10e3 < 0.25:
                extra = [
                    "",
                    "★ Measured about 10 kΩ -- that is exactly **Mode 1** (two 20 kΩ at A/B "
                    "**in parallel**).",
                    "  Set the Mode switch to **2**; only then is each branch an independent 20 kΩ.",
                    "  The ktram differential read also works only in Mode 2: "
                    "in Mode 1 scope 2+ measures",
                    "  the source voltage of W1 rather than the B branch node, so gb ≡ 0 and y all has one sign.",
                ]
            raise PreflightFailed("\n  ".join([
                f"Measured Rs ≈ {rs_m/1e3:.2f} kΩ, declared {env.rs/1e3:.1f} kΩ"
                f" (deviation {r['err']*100:.0f}%).",
                f"The whole premise of current limiting is this number; starting with it in doubt is not allowed.",
                f"A wrong Rs scales **both** the worst-case current and the resistance floor:",
                f"  worst-case current = {env.i_safe*1e6:.0f} µA × "
                f"{env.rs/rs_m:.2f} = {env.i_safe*env.rs/rs_m*1e6:.0f} µA",
                f"  a device reported at {env.r_floor_guard/1e3:.0f} kΩ is really only "
                f"{env.r_floor_guard*rs_m/env.rs/1e3:.1f} kΩ",
            ] + extra + [
                "",
                f"Change rs= to the measured value, or fix the hardware."
                f" (Below {safety.RS_MIN/1e3:.0f} kΩ the envelope refuses -- "
                f"deliberately, see SAFETY.md.)",
            ]))

        if st in ("determined", "ok", "ok_rc"):
            src = "board-crossvalidated" if r.get("cross_validated") else "board"
            if rs_source == "dmm":
                src = "multimeter+board"
            gate.journal["rs_source"] = {
                "t": time.strftime("%Y-%m-%d %H:%M:%S"),
                "rs": env.rs, "rs_measured": rs_m, "method": src}
            gate.flush_journal()
            if verbose:
                print(f"  Rs = {env.rs/1e3:.1f} kΩ ✓ agrees with the on-board measurement "
                      f"{rs_m/1e3:.2f} kΩ"
                      + (" (two methods cross-checked)" if r.get("cross_validated") else ""))
        elif rs_source == "dmm":
            gate.journal["rs_source"] = {
                "t": time.strftime("%Y-%m-%d %H:%M:%S"),
                "rs": env.rs, "method": "multimeter",
                "board_selftest": st}
            gate.flush_journal()
            if verbose:
                print(f"  ★ The on-board measurement did not determine Rs ({st}), "
                      f"continuing with the **multimeter-measured** {env.rs/1e3:.1f} kΩ.")
        elif rs_source == "nominal":
            gate.journal["rs_source"] = {
                "t": time.strftime("%Y-%m-%d %H:%M:%S"),
                "rs": env.rs, "method": "nominal", "board_selftest": st}
            gate.flush_journal()
            if verbose:
                print(f"  ★ The on-board measurement did not determine Rs ({st}), "
                      f"continuing with the **nominal value** {env.rs/1e3:.1f} kΩ.")
                print(f"    Risk: worst-case current and resistance floor both scale by "
                      f"Rs_declared/Rs_actual. Under-reporting is the safe direction.")
        else:
            raise PreflightFailed(
                f"The on-board measurement could not determine Rs ({st}), and no source was declared.\n"
                f"  Use --rs-dmm (multimeter-measured) or --rs-nominal (accept the nominal value).\n"
                f"  Diagnose: python check_rs.py --rs {env.rs:.0f} "
                f"--chip {gate.chip_id}")

    # ---- 3. read one by one
    # ★ In Mode 2 read_all_mode2 must be used: devices 8..15 are on the B branch (scope 2+),
    #   while read_all() looks only at scope 1+ (the single-node topology of Mode 1).
    #   On real hardware this reported 8 good devices as "open circuit".
    #   The Mode 2 formula **assumes** the source voltage, so the offsets must be measured first.
    if verbose:
        print(f"  Measuring DC offsets first (the Mode 2 resistance formula assumes the source voltage, "
              f"and uncompensated offsets cause errors of tens of percent)...")
    import ktram as _K
    offs = _K.calibrate_offsets(md, verbose=verbose)
    if verbose:
        print(f"  Reading the 16 devices one by one ({env.v_read:.2f} V read pulse, "
              f"{env.v_read/safety.V_TH*100:.0f}% of threshold; "
              f"side A via scope 1+, side B via scope 2+)...")
    r = md.read_all_mode2(v_read=env.v_read, width_us=25.0, offsets=offs)
    gate.journal["offsets"] = list(offs)

    j = gate.journal
    prev = {int(k): v for k, v in j.get("r_last", {}).items()}
    best = {int(k): v for k, v in j.get("r_best", {}).items()}

    # ★ Degradation warnings cannot be compared across Rs. Every resistance the board reports is
    #   Rs_declared × (voltage ratio), so if the previous session used a different Rs,
    #   the two sets differ by a common factor and look like "dropped by half".
    #   This produced a false alarm on real hardware: the previous session was recorded at 20 kΩ while the real Rs was 9.5 kΩ;
    #   after switching to a real 20 kΩ all readings halved, and 7 devices were judged degraded.
    rs_prev = (j.get("rs_source") or {}).get("rs")
    rs_comparable = rs_prev is None or abs(rs_prev - env.rs) / env.rs < 0.05
    if not rs_comparable and verbose:
        print(f"  · The previous session's Rs was {rs_prev/1e3:.1f} kΩ, this one is "
              f"{env.rs/1e3:.1f} kΩ ⇒ the two sets of resistances differ by a common factor, "
              f"**not comparable**; skipping degradation warnings.")
        print(f"    (All resistances are Rs_declared × voltage ratio, so changing Rs scales them all.)")

    bounded = set(getattr(md, "read_bounded", ()) or ())
    low, open_, ok_ = [], [], []
    drift = []
    for i, ri in enumerate(r):
        if i in bounded:
            # drop below the noise threshold ⇒ only a **lower bound** is known. The lower bound is far above the floor,
            # so it is acceptable from a safety point of view and counts as usable -- judging high resistance as a bad device is wrong.
            ok_.append(i)
            continue
        if ri is None or ri != ri or ri <= 0 or ri > env.r_ceiling:
            open_.append(i)
        elif ri < env.r_floor_guard:
            low.append(i)
        else:
            ok_.append(i)
        b = best.get(i)
        if (b and ri and ri == ri and 0 < ri <= env.r_ceiling
                and ri < 0.5 * b and rs_comparable and i not in bounded):
            drift.append((i, b, ri))

    if verbose:
        print()
        print("    Device Resistance   Previous    Hist. max   Status")
        for i, ri in enumerate(r):
            s = ("open/not connected" if i in open_ else
                 "★ below refuse line" if i in low else
                 f"high resistance ≥{r[i]/1e6:.1f}MΩ (beyond read resolution)" if i in bounded
                 else "normal")
            if i in gate.quarantined:
                s = "★ quarantined"
            def f(x):
                if x is None or x != x:
                    return "    —   "
                if x > env.r_ceiling:
                    return "   open "
                return f"{x/1e3:7.1f}k"
            print(f"    {i:4d}  {f(ri)}  {f(prev.get(i))}  {f(best.get(i))}"
                  f"    {s}")
        print()

    # ---- 4. decision
    for i in low:
        gate.note_resistance(i, r[i])      # this puts them in the quarantine list

    if low:
        raise PreflightFailed(
            f"Devices {low} read below the refuse line of {env.r_floor_guard/1e3:.0f} kΩ.\n"
            f"  Within the {env.i_safe*1e6:.0f} µA envelope a device **should have no chance of getting here**"
            f" (manual figure 4: this compliance current corresponds to a high-resistance state of order 1 MΩ).\n"
            f"  So this reading indicates not a 'poor device state' but that **the envelope is not actually in effect** -- "
            f"most likely a wrong Rs, the wrong Mode, or a shorted switch.\n"
            f"  They have been written to the quarantine list. Check the hardware first; do not suspect the devices first.")

    if drift:
        msg = ", ".join(f"device {i}: {b/1e3:.0f}k → {now/1e3:.1f}k"
                        for i, b, now in drift)
        raise PreflightFailed(
            f"Dropped by more than half compared with the historical maximum: {msg}\n"
            f"  This is what slow degradation looks like -- the previous chip went down step by step like this, "
            f"with nobody watching these numbers at the time.\n"
            f"  Find out why before continuing. If confirmed acceptable, pass allow_drift=True.")

    if len(ok_) < min_usable:
        raise PreflightFailed(
            f"Only {len(ok_)}/16 devices usable (need ≥ {min_usable}): "
            f"open {open_}.\n"
            f"  Open circuit is usually not a device fault: check the Mode switch, whether the chip is fully inserted, "
            f"and whether the ±5 V supply is sufficient (common with passive hubs).")

    if verbose:
        rr = [r[i] for i in ok_]
        print(f"  {len(ok_)}/16 usable, resistance {min(rr)/1e3:.1f} ~ "
              f"{max(rr)/1e3:.0f} kΩ  ✓")
        if bounded:
            print(f"  · The readings of devices {sorted(bounded)} are only **lower bounds**"
                  f" (drop below the noise threshold "
                  f"{md.READ_NOISE_V*1e3:.1f} mV, resolution limit about "
                  f"{r[sorted(bounded)[0]]/1e6:.1f} MΩ).")
            print(f"    Acceptable from a safety point of view (far above the "
                  f"{env.r_floor_guard/1e3:.0f} kΩ floor), "
                  f"but do not use that number for fine inferences.")
        print(f"  Safe window {env.r_floor_guard/1e3:.0f} kΩ ~ "
              f"{env.r_ceiling/1e6:.1f} MΩ, "
              f"closest to the refuse line is device {ok_[rr.index(min(rr))]}"
              f" ({min(rr)/env.r_floor_guard:.1f}×)")
        print("-" * 72)
    return r


# ===========================================================================
def open_board(rs: float, *, chip_id: str, purpose: str, mode: int = 2,
               confirm: bool = False, verbose: bool = True,
               i_safe: float | None = None, v_dev_max: float | None = None,
               check_rs: bool = True, allow_drift: bool = False,
               min_usable: int = 16, dwf_path: str | None = None,
               rs_source: str | None = None) -> HwSession:
    """
    Get a board with the protection gate installed and the self-test passed.

    Parameters
    ----------
    rs       : measured series resistance, **must be given explicitly**. See the module notes.
    chip_id  : chip ID. Records and audit logs are filed under it; use a new ID for a new chip.
    purpose  : what this run is for (written to the audit log, e.g. "bars-train", "cartpole").
    mode     : position of the V2 board's Mode switch. Only 2 is accepted -- Rs in Mode 1 is 10 kΩ,
               below the lower limit of the envelope.
    confirm  : True means the caller has already confirmed (e.g. a GUI dialog),
               skipping the typed confirmation on the command line. Default False, which requires typing chip_id.
    i_safe / v_dev_max : only values **more conservative** than the defaults are allowed; relaxing is refused.
    """
    if mode != 2:
        raise SafetyViolation(
            f"mode={mode} not accepted. In Mode 1 of the V2 board the two 20 kΩ at A/B are in parallel = 10 kΩ, "
            f"and at 10 kΩ a device at the floor gets only 0.28 V (threshold 0.26 V), "
            f"a 7% margin -- once it cannot be erased there is no way back.\n"
            f"  If the inverted pendulum uses single devices, it also runs in Mode 2, connecting one path at a time.")

    kw = {}
    if i_safe is not None:
        if i_safe > safety.I_SAFE:
            raise SafetyViolation(
                f"i_safe={i_safe*1e6:.0f} µA is wider than the default {safety.I_SAFE*1e6:.0f} µA. "
                f"This entry point accepts only more conservative values.")
        kw["i_safe"] = float(i_safe)
    if v_dev_max is not None:
        if v_dev_max > safety.V_DEV_MAX:
            raise SafetyViolation(
                f"v_dev_max={v_dev_max:.2f} V is wider than the default {safety.V_DEV_MAX:.2f} V. "
                f"This entry point accepts only more conservative values.")
        kw["v_dev_max"] = float(v_dev_max)

    env = Envelope.for_rs(rs, **kw)     # Rs < 20 kΩ is blocked here

    if verbose:
        print()
        print("=" * 72)
        print(f"Real hardware session: {purpose}    chip {chip_id}    Mode {mode}")
        print("=" * 72)

    gate = SafetyGate(env, chip_id=chip_id, purpose=purpose, verbose=verbose)

    # ---- typed confirmation
    if not confirm:
        print()
        print(f"  This applies voltage to a real chip. Before confirming, check two things:")
        print(f"    · the Mode switch is at 2 and both {rs/1e3:.0f} kΩ series resistors are fitted at A/B")
        print(f"    · the chip is {chip_id} (records and budget are kept under this ID)")
        try:
            ans = input(f"  Type the chip ID again to continue (or press Enter to abort): ").strip()
        except EOFError:
            ans = ""
        if ans != chip_id:
            gate.close("aborted by user")
            raise PreflightFailed(
                f"Not confirmed (entered {ans!r}, expected {chip_id!r}); the board will not be opened.")

    from knowm_md import MemristorDiscovery
    md = MemristorDiscovery(series_resistance=rs, board_version=2,
                            dwf_path=dwf_path, verbose=verbose)
    md.open()
    md.attach_gate(gate)

    try:
        base = preflight(md, gate, verbose=verbose, check_rs=check_rs,
                         min_usable=min_usable, rs_source=rs_source) \
               if not allow_drift else \
               _preflight_allow_drift(md, gate, verbose, check_rs,
                                      min_usable, rs_source)
    except Exception:
        try:
            md.all_off(); md.close()
        finally:
            gate.close("preflight failed")
        raise

    if verbose:
        print(f"Self-test passed; board handed to {purpose}.\n")
    return HwSession(md, gate, base, env)


def _preflight_allow_drift(md, gate, verbose, check_rs, min_usable,
                           rs_source=None):
    """allow_drift=True: only downgrades degradation to a warning; every other check stays."""
    try:
        return preflight(md, gate, verbose=verbose, check_rs=check_rs,
                         min_usable=min_usable, rs_source=rs_source)
    except PreflightFailed as e:
        if "dropped by more than half" not in str(e):
            raise
        print(f"\n  ★ The following was allowed with allow_drift=True; please keep it in mind:\n{e}\n")
        return md.read_all(v_read=gate.env.v_read, width_us=25.0)


# ===========================================================================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Start-up self-test only: read the 16 devices, measure Rs and compare with the record, without writing anything.")
    p.add_argument("--rs", type=float, required=True,
                   help="Measured series resistance, Ω. Required; no default.")
    p.add_argument("--chip", required=True, help="Chip ID")
    p.add_argument("--yes", action="store_true", help="Skip the typed confirmation")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--rs-dmm", action="store_true",
                   help="The --rs value was **measured with a multimeter**. The board cannot determine Rs in principle "
                        "(with no current the series resistance cannot be observed), so the source must be declared explicitly.")
    g.add_argument("--rs-nominal", action="store_true",
                   help="--rs is only the board's nominal value, not measured. The risk will be printed and recorded."
                        "Note: a wrong Rs scales both the worst-case current **and the resistance floor**. "
                        "Under-reporting Rs is the safe direction.")
    p.add_argument("--allow-drift", action="store_true")
    p.add_argument("--min-usable", type=int, default=16)
    a = p.parse_args()
    try:
        with open_board(a.rs, chip_id=a.chip, purpose="preflight-only",
                        confirm=a.yes, allow_drift=a.allow_drift,
                        min_usable=a.min_usable,
                        rs_source="dmm" if a.rs_dmm else "nominal") as s:
            print("  (Read-only session; no write pulses were sent.)")
    except (PreflightFailed, SafetyViolation) as e:
        print(f"\nSelf-test failed:\n{e}\n")
        sys.exit(1)
