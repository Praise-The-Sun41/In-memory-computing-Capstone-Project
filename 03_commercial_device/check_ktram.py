"""
check_ktram.py — health check of the kT-RAM differential synapse array. Run this when classification gives 50%.

    python check_ktram.py                 # full suite
    python check_ktram.py --synapse 3     # use another synapse for the write test
    python check_ktram.py --no-calib      # use the hand-entered offsets in example_classify.py

Board requirements: V2 board, Mode switch set to **2**, two **equal-value** precision resistors at A/B.

It answers three questions in order; if the first fails there is no point looking further:

    1. How large are the offsets?        -- the read voltage is only 80 mV; tens of mV of offset will pin the sign of y
    2. Is the read channel alive?        -- the y of the 8 synapses must include both signs and be distinguishable
    3. Do write pulses change devices?   -- hit FA hard, then RA hard, and see whether ga moves

Why "exactly 50%" is a signature: every data set has balanced positive and negative samples
(Ortho4 = 2T/2F, Ortho8 = 4T/4F). If the sign of y never changes and only one class is ever predicted,
accuracy is mathematically **equal** to 50.0% -- that is not "poor learning", it means read or write is broken.
"""

from __future__ import annotations

import argparse
import sys

import ktram as K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synapse", type=int, default=0, help="Synapse 0..7 used for the write test")
    ap.add_argument("--pulses", type=int, default=10, help="Number of pulses per step in the write test")
    ap.add_argument("--no-calib", action="store_true", help="Do not auto-calibrate; use the offsets entered in the script")
    ap.add_argument("--all", action="store_true",
                    help="Run the write test on all 8 synapses one by one and tabulate."
                         "Use this if some devices are suspected faulty or responses differ systematically")
    ap.add_argument("--erase", type=int, metavar="N", default=0,
                    help="Before the check, apply N rounds of RAB to all 8 synapses (normal amplitude, normal width)."
                         "Use this for mildly low resistance")
    ap.add_argument("--hard-erase", nargs="?", const="auto", metavar="synapse index",
                    help="Hard reset with stepped voltage + long pulse width. No argument = scan all 8 and fix only stuck ones; "
                         "with argument = fix only the given synapse, e.g. --hard-erase 3."
                         "Bias only the stuck side; the switch of the other side stays open")
    ap.add_argument("--side", choices=("auto", "a", "b", "both"), default="auto",
                    help="Which side the hard reset biases (default auto = only the side whose conductance is over the limit)")
    ap.add_argument("--chip", required=True,
                    help="Chip ID. Budget, quarantine list and degradation trend are filed under it.")
    ap.add_argument("--allow-drift", action="store_true",
                    help="If the start-up self-test finds resistance more than halved from its historical maximum, downgrade to a warning and continue")
    _g = ap.add_mutually_exclusive_group(required=True)
    _g.add_argument("--rs-dmm", action="store_true",
                    help="Rs was measured with a multimeter. The board cannot determine Rs in principle.")
    _g.add_argument("--rs-nominal", action="store_true",
                    help="Rs is only the nominal value; accept the scaling risk")
    # --vmax / --allow-overcurrent removed: the amplitude is now issued by safety.SafetyGate,
    # not chosen by the caller. Reverse limit = I_safe × Rs; to push harder, only the pulse width can be increased.
    ap.add_argument("--vfwd", type=float, default=None,
                    help="Override the write amplitude (default from example_classify.py)")
    ap.add_argument("--vrev", type=float, default=None,
                    help="Override the reset amplitude. If directions are abnormal, reduce this first and retest")
    ap.add_argument("--sweep-v", metavar="V1,V2,...", default=None,
                    help="Measure the direction of all 16 devices at several amplitudes to find a working window in which all devices "
                         "behave normally. Example: --sweep-v 0.8,1.0,1.3,1.6,2.0")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only print what the hard reset would apply (voltage/current/closed DIO); do not actually pulse")
    args = ap.parse_args()

    from example_classify import make_hw, V_FORWARD, V_REVERSE, SERIES_R
    V_FORWARD = args.vfwd if args.vfwd is not None else V_FORWARD
    V_REVERSE = args.vrev if args.vrev is not None else V_REVERSE

    print("=" * 66)
    print("kT-RAM differential synapse array health check")
    print(f"  Mode 2 · series resistor per branch {SERIES_R/1000:.0f} kΩ · "
          f"V_fwd {V_FORWARD} V · V_rev {V_REVERSE} V")
    print("=" * 66)

    # ---------------------------------------------- 1. open + offset calibration
    print("\n── 1. Open device and measure DC offsets " + "─" * 34)
    try:
        s, backend = make_hw(skip_calibration=args.no_calib,
                             chip_id=args.chip, purpose='ktram-check',
                             allow_drift=args.allow_drift,
                             rs_source="dmm" if args.rs_dmm else "nominal")
        md = s.md
    except Exception as e:
        print(f"  Open failed: {e}")
        print("  If refused because of Rs: python check_rs.py --rs 20000 "
              "--chip " + str(args.chip))
        print("  (check_device.py is no longer in scope; it was replaced by the start-up self-test"
              " in hw_session.)")
        sys.exit(1)

    verdict = []
    n_partial = False
    try:
        kt = K.KTRAM(backend, v_forward=V_FORWARD, v_reverse=V_REVERSE)

        if args.erase:
            print(f"\n── Erase: {args.erase} rounds of RAB on each of the 8 synapses " + "─" * 24)
            for r in range(args.erase):
                for i in range(K.N_SYNAPSES):
                    kt.execute(K.Pattern((i,), True), K.Instruction.RAB)
            print(f"  Issued {args.erase * K.N_SYNAPSES} RAB instructions; devices should be back in the high-resistance state.")

        if args.hard_erase:
            only = None if args.hard_erase == "auto" else [int(args.hard_erase)]
            title = "all stuck synapses" if only is None else f"synapse {only[0]} only"
            print(f"\n── Hard reset ({title})" + "─" * 32)
            print("  Per section 12.4 of the manual: the more conductive the device, the more voltage the series resistor takes,")
            print("  so normal amplitudes cannot move it.")
            print("  ★ The amplitude limit is set by the protection gate (= I_safe × Rs); **voltage is no longer stepped up**;")
            print("    the only variable is pulse width. Method 1 of the manual uses −2 V for 2~5 s,")
            print("    relying on time, not current.")
            print("  ★ Under the new envelope this path should hardly ever be used: the resistance floor keeps every device")
            print("    above 67 kΩ, so 'stuck' should not happen in the first place.")
            print("  Only the stuck side is biased; the other side of the same synapse is switched off,")
            print("  and the switches of all other synapses stay open throughout.")
            if args.dry_run:
                print("  [Dry run] Only list the plan; no pulses are sent.")
            K.hard_reset_all(kt, synapses=only, side=args.side,
                             dry_run=args.dry_run)
            if args.dry_run:
                print("\n  Remove --dry-run to execute.")
                return

        # ---------------------------------------------- 2. read channel
        print("\n── 2. Read channel: FLV on each synapse " + "─" * 32)
        r = K.diagnose_read(kt)
        read_ok = not r["same_sign"] and r["spread"] >= 0.02
        verdict.append(("Read channel", read_ok))

        # ---------------------------------------------- amplitude sweep
        if args.sweep_v:
            vs = [float(x) for x in args.sweep_v.split(",")]
            print("\n── Amplitude sweep: find a working window in which all 16 devices behave normally " + "─" * 15)
            print("  At each amplitude the direction of each of the 16 devices is measured (forward should rise, reverse should fall).")
            print("  Knowm's own app uses ±1 V -- if low amplitudes are all correct and high amplitudes fail,")
            print("  the drive is too strong and pushes devices too far; the devices are not faulty.\n")
            print(f"  {'Ampl.':>7s}{'All OK':>7s}{'Both rev':>10s}{'One rev':>8s}"
                  f"{'Stuck':>8s}   Problem chip devices")
            results = []
            for v in vs:
                kt.v_forward = kt.v_reverse = v
                tally = {"ok": 0, "both_flip": 0, "one_flip": 0, "stuck": 0}
                bad = []
                for i in range(K.N_SYNAPSES):
                    for sd in ("a", "b"):
                        r = K.diagnose_write(kt, synapse=i, n_pulses=args.pulses,
                                             side=sd, verbose=False)
                        tally[r["kind"]] += 1
                        if r["kind"] != "ok":
                            bad.append(r["chip"])
                print(f"  {v:>6.1f}V{tally['ok']:>6d}/16{tally['both_flip']:>10d}"
                      f"{tally['one_flip']:>8d}{tally['stuck']:>8d}   {sorted(bad)}")
                results.append((v, tally["ok"]))
            kt.v_forward, kt.v_reverse = V_FORWARD, V_REVERSE

            full = [v for v, ok in results if ok == 2 * K.N_SYNAPSES]
            if full:
                # among the all-OK amplitudes pick the largest: stronger writes give larger decision margins after training
                pick = max(full)
                print(f"\n  Amplitudes at which all 16 devices are normal: {[f'{v:.1f}' for v in full]} V")
                print(f"  → Devices are fine; it is a drive amplitude problem. Take the largest in the window, {pick:.1f} V")
                print(f"     (stronger writes give larger decision margins after training):")
                print(f"     set both V_FORWARD / V_REVERSE in example_classify.py to {pick:.1f},")
                print(f"     or run python example_bars.py --hw --vfwd {pick:.1f} --vrev {pick:.1f}")
            else:
                best = max(results, key=lambda r: r[1])
                print(f"\n  No amplitude makes all 16 devices normal.")
                print(f"  The best is {best[0]:.1f} V ({best[1]}/16) -- the remaining devices are genuinely abnormal.")
            print("\n  Note: the sweep itself repeatedly reads and writes the devices, so their state afterwards differs from before.")

        # ---------------------------------------------- 3. write channel
        if args.all:
            print("\n── 3. Write channel: direction test of each of the 16 devices " + "─" * 25)
            print("  Conductance should **rise** after a forward pulse and **fall** after a reverse pulse. Looking only at magnitude misses")
            print("  the case 'moving but in the wrong direction' -- which makes training push the wrong way.\n")
            print(f"  {'Syn':>4s}{'Chip dev':>9s}{'Side':>4s}{'Init µS':>10s}"
                  f"{'After fwd':>10s}{'After rev':>10s}  Verdict")
            n_ok = n_flip = n_dead = 0
            for i in range(K.N_SYNAPSES):
                for sd in ("a", "b"):
                    r = K.diagnose_write(kt, synapse=i, n_pulses=args.pulses,
                                         side=sd, verbose=False)
                    g0, g1, g2 = (v * 1e6 for v in r["g"])
                    LABEL = {"ok": "normal", "both_flip": "★ both reversed",
                             "one_flip": "★ one reversed", "stuck": "★ stuck"}
                    tag = LABEL[r["kind"]]
                    if r["kind"] == "ok":
                        n_ok += 1
                    elif r["kind"] == "both_flip":
                        n_flip += 1
                    else:
                        n_dead += 1
                    print(f"  {i:>4d}{r['chip']:>9d}{sd.upper():>4s}"
                          f"{g0:>10.2f}{g1:>10.2f}{g2:>10.2f}  {tag}")
            print(f"\n  normal {n_ok}/16   both reversed {n_flip}   one-direction abnormal {n_dead}")
            if n_ok < 2 * K.N_SYNAPSES:
                print("  Do not conclude the devices are faulty yet -- the most common cause of one-direction anomalies is too large a drive amplitude.")
                print("  Run an amplitude sweep to find the working window:")
                print("    python check_ktram.py --sweep-v 0.8,1.0,1.3,1.6,2.0")
            if n_flip >= 12:
                print("  ★ Most devices are reversed -- the 1×16 chip is very likely inserted backwards.")
                print("    SDC devices are polar; inserting the chip rotated 180° swaps set/reset for all devices.")
                print("    Remove the chip, rotate it 180°, reinsert it and run this command again.")
            elif n_flip:
                print("  ★ Only some devices are reversed -- those devices themselves are abnormal; the chip is not reversed.")
                print("    During training they push weights the wrong way, so the receptive field will be disordered.")
                print("    Experiments can use only the synapses with correct direction (reduced pattern set).")
            write_ok = (n_ok == 2 * K.N_SYNAPSES)
            n_partial = 0 < n_ok < 2 * K.N_SYNAPSES
        else:
            print("\n── 3. Write channel: hit FA hard, then RA hard " + "─" * 29)
            r = K.diagnose_write(kt, synapse=args.synapse, n_pulses=args.pulses,
                                 side=args.side if args.side in ("a", "b") else "a")
            write_ok = r["write_ok"]
        verdict.append(("Write channel", write_ok))

        # ---------------------------------------------- 4. conclusion
        print("\n" + "=" * 66)
        for name, ok in verdict:
            print(f"  {name}  {'normal' if ok else 'abnormal'}")
        print("=" * 66)

        if all(ok for _, ok in verdict):
            print("\nRead and write are both normal; classification can be run:")
            print("    python example_classify.py --hw --dataset Ortho4")
        elif args.all and n_partial:
            print("\nOnly some synapses can be written. First identify the faulty devices using the hints above,")
            print("then decide whether to replace the chip or run experiments with only the working synapses")
            print(" (reduce the data set to patterns that fall on working synapses).")
        elif not verdict[1][1]:
            print("\nThe write channel is the main problem -- however accurate the read, the weights will not change.")
            print("Check in order of likelihood:")
            print("  1. The Mode switch is not at '2'. The board is still in single-device mode and the A/B resistors are in parallel,")
            print("     so no differential pair exists. This is the easiest to miss; check it first.")
            print("  2. Only one resistor is inserted at A/B, or the two values differ. In Mode 2 each branch needs one.")
            print("  3. Write voltage too low. If the change in ga above is only slightly small, raise the amplitude in example_classify.py")
            print("     after first confirming it is not limited by compliance: the compliance current in device_limits.py"
                  " sets the drive limit,")
            print("     Rs=20kΩ + 100 µA ⇒ at most 2.0 V. For higher device voltage use a larger series resistor,")
            print("     not a higher current -- raising the current irreversibly lowers the high-resistance state (manual Note 1).")
            print("  4. The synapse is dead (both sides in low resistance). Erase back to high resistance and recheck:")
            print("       python check_ktram.py --erase 15")
        else:
            print("\nWrite works but read is wrong -- the devices are moving, but y does not reflect it faithfully.")
            print("  · If step 1 above measured offsets of tens of mV, that is the culprit,")
            print("    adding a constant term to y of the same order as the signal.")
            print("  · Alternatively measure manually per section 12.6 of the manual and enter the values at the top of example_classify.py")
            print("    in SCOPE1_OFFSET / SCOPE2_OFFSET / W1_OFFSET, then run with --no-calib.")
            print("  · If the range of y is very small (<0.02), check that the two series resistors are equal.")

    finally:
        # session.__exit__ powers down, writes the chip record and prints the session summary
        s.__exit__(None, None, None)


if __name__ == "__main__":
    main()
