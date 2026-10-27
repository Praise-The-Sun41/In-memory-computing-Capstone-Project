"""
example_classify.py — 8-synapse kT-RAM online supervised classification (the GUI's Classify 1-2 experiment)

    python example_classify.py                    # simulation, all data sets × all routines
    python example_classify.py --dataset Ortho4   # simulation, single data set
    python example_classify.py --hw               # real hardware (V2 board + Mode 2)

Hardware
--------
V2 board, Mode switch at "2", two **equal-value** precision resistors at A/B (20 kΩ as shipped).
In Mode 2 SERIES_R is the value of a single resistor, not the parallel value of Mode 1.

It is strongly recommended to measure the three DC offsets of the AD2 per manual section 12.6 first and enter them here --
the read voltage is only 0.08 V, and a few mV of offset is enough to make the computed conductance negative.
"""

from __future__ import annotations

import argparse
import random
import statistics as st

import ktram as K
import scope_guard

SERIES_R = 20_000.0     # Mode 2: single series resistor
PULSE_W_US = 50.0

# write / reset amplitudes. **These values are device dependent and must be measured on your own board.**
#
# Knowm's Classify12Preferences defaults both to 1.0 V. Measured on this board
# (check_ktram.py --sweep-v 0.8,1.0,1.3,1.6,2.0, checking at each amplitude whether the write direction of the 16 devices
# is correct), the result is:
#
#     0.8 V -> 10/16 normal     1.0 V -> 12/16     1.3 V -> 14/16
#     1.6 V -> 15/16          2.0 V -> 16/16
#
# failures at low amplitude are almost all "cannot move in one direction", i.e. the device voltage does not reach the SDC
# switching threshold -- not over-driving. The series resistor takes a large share (V_device = V·Rm/(Rm+Rs)),
# so the drive must be well above the device threshold.
#
# too little amplitude does more than "learn slowly": weights of devices that cannot move get stuck while movable ones keep updating,
# so the final receptive field is disordered and even the training set fails. After changing board or chip, rerun --sweep-v to find the window.
# ---- device protection ----
# a fixed-amplitude script cannot invert the measured resistance as a closed loop can, so it uses values **safe in the worst case**:
# with drive ≤ 1.0 V a device with R→∞ gets at most 1.0 V (manual: "normally operated
# under 1V"), and with Rs=20kΩ the worst-case current is only 50 µA.
# the old value of 2.0 V gives 1.67 V on a 100 kΩ device -- beyond the manual's operating range.
V_FORWARD = 1.0
V_REVERSE = 1.0

# AD2 DC offsets. Leave None to measure automatically before each hardware run (recommended);
# or measure manually per manual section 12.6 and hard-code them here.
#
# Why these three numbers are critical: the read pulse is only 80 mV, and in the manual's own example
# the offset of scope 2+ is −41 mV. Without compensation a constant term of +0.5 enters y,
# larger than the real signal -- the sign of y is pinned, the model always predicts one class,
# so accuracy on any balanced data set is **exactly** 50%.
SCOPE1_OFFSET = None
SCOPE2_OFFSET = None
W1_OFFSET = None


def make_sim(seed=0):
    return K.SimBackend(rs=SERIES_R, seed=seed, eta=0.12)


def make_hw(skip_calibration=False, *, chip_id=None, purpose="bars",
            rs=None, confirm=False, allow_drift=False,
            rs_source=None):
    """
    Real-hardware back end. **Returns (session, backend)** -- it used to return (md, backend).

    It returns a session because the board must now be obtained from `hw_session.open_board()`:
    it sets up the safety envelope, measures Rs, runs the start-up self-test, attaches the protection gate and powers down on exit.
    Without a board there is no backend; this is deliberate.

    `chip_id` is required -- pulse budget and degradation trend are filed per chip,
    and reusing an old ID for a new chip would mix the histories of two chips.
    Usage:

        with make_hw(chip_id="chip-2026-10")[0] as s: ...

    or more clearly:

        s, be = make_hw(chip_id="chip-2026-10")
        with s:
            ...
    """
    import hw_session
    if not chip_id:
        raise SystemExit(
            "make_hw() requires chip_id. Pulse budget, quarantine list and degradation trend are filed per chip;\n"
            "use a new ID for a new chip (e.g. --chip chip-2026-10).")
    s = hw_session.open_board(
        rs=(SERIES_R if rs is None else rs),
        chip_id=chip_id, purpose=purpose, mode=2,
        confirm=confirm, allow_drift=allow_drift,
        rs_source=rs_source)

    o1, o2, ow = SCOPE1_OFFSET, SCOPE2_OFFSET, W1_OFFSET
    if None in (o1, o2, ow) and not skip_calibration:
        o1, o2, ow = K.calibrate_offsets(s.md)
        print()
    o1, o2, ow = (o1 or 0.0), (o2 or 0.0), (ow or 0.0)

    be = K.HardwareBackend(s.md, pulse_width_us=PULSE_W_US,
                           scope1_offset=o1, scope2_offset=o2, w1_offset=ow)

    # ★ Determine the wiring before training. In Mode 1 the differential read does not exist at all,
    #   and the symptoms (gb≡0 / y all of one sign / SET refused) look like a dozen other problems --
    #   five rounds of debugging were spent on this on real hardware.
    w = K.check_mode2_wiring(K.KTRAM(be), verbose=True)
    if not w["mode2"]:
        s.__exit__(None, None, None)
        raise SystemExit(
            "\nThe board is wired for Mode 1, but the ktram differential read needs Mode 2.\n"
            "  Set the Mode switch to 2, check that both 20 kΩ series resistors are fitted at A/B, "
            "and try again.\n"
            "  (In Mode 1 Rs is two in parallel = 10 kΩ, which is also below the lower limit of the safety envelope.)")
    return s, be


def run_one(backend, dataset_name, routine, epochs, seed, scramble,
            v_forward=None, v_reverse=None):
    kt = K.KTRAM(backend,
                 v_forward=V_FORWARD if v_forward is None else v_forward,
                 v_reverse=V_REVERSE if v_reverse is None else v_reverse)
    if scramble:
        kt.scramble(epochs=8, rng=random.Random(seed))
    ds = K.DATASETS[dataset_name]
    res = K.train(kt, ds, routine, epochs=epochs, seed=seed)
    return kt, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw", action="store_true", help="Use real hardware instead of simulation")
    ap.add_argument("--chip", default=None, help="Chip ID (required with --hw)")
    ap.add_argument("--i-know", action="store_true",
                    help="Allow this script to drive real hardware even though its data sets are known to be out of current scope")
    ap.add_argument("--dataset", default=None, help="Run only one data set")
    ap.add_argument("--routine", default=None, help="Run only one learning routine")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=10, help="Number of simulation repeats")
    ap.add_argument("--no-scramble", action="store_true")
    ap.add_argument("--plot", action="store_true",
                    help="Make figures (always in hardware mode). In simulation, one representative figure per data set.")
    ap.add_argument("--no-calib", action="store_true", help="Skip automatic offset calibration")
    ap.add_argument("--force", action="store_true", help="Train even if the read-channel self-test fails")
    ap.add_argument("--vfwd", type=float, default=None,
                    help=f"Write amplitude, overrides the default {V_FORWARD} V")
    ap.add_argument("--repeat", type=int, default=10,
                    help="After training, read each pattern N times to measure decision margin and sign-flip rate (default 10)")
    ap.add_argument("--vrev", type=float, default=None,
                    help=f"Reset amplitude, overrides the default {V_REVERSE} V. "
                         "Too low pushes devices into the low-resistance state in one direction")
    args = ap.parse_args()

    v_fwd = args.vfwd if args.vfwd is not None else V_FORWARD
    v_rev = args.vrev if args.vrev is not None else V_REVERSE

    datasets = [args.dataset] if args.dataset else list(K.DATASETS)
    routines = [args.routine] if args.routine else list(K.ROUTINES)
    scramble = not args.no_scramble

    # ---------------------------------------------------------- hardware mode
    if args.hw:
        # ★ Note the scope: `make_hw` itself **is** in scope -- the H/V 3×3 bar task
        #   (bars_gui / example_bars) gets its hardware back end from here.
        #   What is out of scope is **this script's own data sets** (Ortho4 etc. digit classification).
        #   So what is blocked here is `python example_classify.py --hw`, not make_hw.
        scope_guard.refuse_hardware(
            'kT-RAM data-set classification (example_classify.py --hw)',
            why="This script's own data sets (Ortho4 etc.) are out of current scope."
                'Its make_hw() is still used; the 3×3 bar task gets its board from there',
            suggest='H/V 3×3 bars: python example_bars.py --hw --chip <chip ID>\n'
                    '           or python bars_gui.py',
            allow=getattr(args, 'i_know', False))
        s, backend = make_hw(skip_calibration=args.no_calib,
                             chip_id=args.chip, purpose='classify-cli')
        md = s.md
        try:
            name = datasets[0]
            routine = routines[0]

            # confirm the read channel is alive before training. Skipping this and looking only at the final accuracy
            # misdiagnoses "read broken" as "cannot learn".
            kt0 = K.KTRAM(backend, v_forward=v_fwd, v_reverse=v_rev)
            print(f"write {v_fwd} V / reset {v_rev} V")
            print("Pre-training read-channel self-test (FLV on each synapse)")
            d = K.diagnose_read(kt0)
            if d["same_sign"] or d["spread"] < 0.02:
                print("\nThe read channel is abnormal; fix this before training -- training now would only give 50%.")
                if not args.force:
                    return
                print("(--force given, continuing)")
            print()

            print(f"data set {name} / routine {routine} / {args.epochs} epochs")
            print(f"pattern: {K.DATASETS[name]}")
            if name in K.DATASET_NOTES:
                print(f"Note: {K.DATASET_NOTES[name]}")

            stuck_before = {i for i, _, ga, gb, _ in d["rows"]
                            if max(ga, gb) > K.STUCK_G}

            kt, res = run_one(backend, name, routine, args.epochs, 0, scramble,
                              v_forward=v_fwd, v_reverse=v_rev)
            print(f"\nFinal accuracy {res['final_acc']*100:.0f}%")
            print("y of each pattern:", [f"{v:+.4f}" for v in res["final_y"]])
            print("Weights of the 8 synapses:", [f"{v:+.4f}" for v in kt.read_all()])
            print(f"Used {backend.read_count} read pulses and {backend.pulse_count} write pulses")

            if args.repeat > 1:
                print(f"\nDecision margin and repeatability (each pattern read {args.repeat} times)")
                print("  With 4 patterns accuracy has only 25% resolution; the margin is more meaningful.")
                print(f"  {'pattern':>12s}{'label':>5s}{'mean y':>10s}{'σ':>9s}"
                      f"{'margin/σ':>9s}{'correct':>9s}")
                worst_margin, worst_snr = 9e9, 9e9
                total_ok = total_n = 0
                for pat in K.DATASETS[name]:
                    ys = [kt.execute(pat, K.Instruction.FLV)
                          for _ in range(args.repeat)]
                    mu = sum(ys) / len(ys)
                    sd = (sum((v - mu) ** 2 for v in ys) / len(ys)) ** .5
                    ok = sum(1 for v in ys if (v > 0) == pat.state)
                    total_ok += ok; total_n += len(ys)
                    snr = abs(mu) / sd if sd > 1e-9 else 9e9
                    worst_margin = min(worst_margin, abs(mu))
                    worst_snr = min(worst_snr, snr)
                    print(f"  {str(list(pat.spikes)):>12s}"
                          f"{'T' if pat.state else 'F':>5s}"
                          f"{mu:>+10.4f}{sd:>9.4f}"
                          f"{snr:>9.1f}{f'{ok}/{args.repeat}':>9s}")
                print(f"\n  Robust accuracy {total_ok/total_n*100:.1f}%"
                      f" ({total_ok}/{total_n} reads)")
                print(f"  Thinnest margin {worst_margin:.4f}, worst SNR {worst_snr:.1f}×")
                if worst_snr < 5:
                    print("  ★ The margin is less than 5× the noise -- accuracy will jump between two values.")
                    print("     Ways to widen the margin, in order of priority:")
                    print(f"       1. Run more epochs: --epochs {args.epochs * 3}")
                    print(f"       2. Increase the write amplitude: --vfwd "
                          f"{min(v_fwd + 0.3, 3.0):.1f} (currently {v_fwd} V). "
                          f"Run check_ktram.py --sweep-v first to confirm this chip's window")
                    print("       3. Average more per read (the averages parameter of HardwareBackend.read)")
                elif worst_snr < 15:
                    print("  The margin is acceptable but not generous. Add --epochs for more stability.")
                else:
                    print("  The margin is ample; this result is stable.")

            print("\nPost-training check: which synapses learned the wrong way")
            # an absolute conductance threshold gives false alarms after training (one side of a device is supposed to go low-resistance),
            # so a more direct criterion is used: the direction of updates this synapse received vs the direction of its final weight.
            expect = {}
            for pat in K.DATASETS[name]:
                for i in pat.spikes:
                    expect.setdefault(i, set()).add(pat.state)
            wfin = res["weights"][-1]
            W_FLOOR = 0.05      # if |w| is below this it has not differentiated yet and does not count as "wrong way"
            wrong, flat = [], []
            for i, labels in sorted(expect.items()):
                if len(labels) != 1:
                    continue                     # frustrated synapse; its direction is undetermined anyway
                want = 1 if next(iter(labels)) else -1
                if abs(wfin[i]) < W_FLOOR:
                    flat.append((i, wfin[i]))
                elif (1 if wfin[i] >= 0 else -1) != want:
                    wrong.append((i, want, wfin[i]))
            if wrong:
                print("  ★ The following synapses received updates in the same direction every epoch, but their final weights are reversed:")
                for i, want, w in wrong:
                    d_ = "push positive (FA+RB)" if want > 0 else "push negative (RA+FB)"
                    side = "B" if want > 0 else "A"
                    chip = i + 9 if want > 0 else i + 1
                    print(f"      synapse {i} (chip devices {i+1},{i+9}) each epoch {d_}, "
                          f"measured w = {w:+.3f}")
                    print(f"        → the {side}-side device (chip {chip}) cannot be pushed back, "
                          f"use python check_ktram.py --hard-erase {i}")
                print(f"\n  Reset amplitude is currently {v_rev} V. If training reverses again a few epochs after a hard reset,")
                print(f"  it is creeping back during training -- raise the reset amplitude: --vrev "
                      f"{min(v_rev + 0.5, 3.0):.1f}")
            elif not flat:
                print("  The weight directions of all 8 synapses agree with the training signal.")
            if flat:
                print(f"  The following synapse weights are still small (|w| < {W_FLOOR}) and have not differentiated; "
                      f"they do not affect decisions but do not contribute either:")
                print("      " + "  ".join(f"syn{i}({w:+.3f})" for i, w in flat))

            with open("classify_weights.csv", "w") as f:
                f.write("epoch," + ",".join(f"syn{i}" for i in range(8)) + "\n")
                for e, w in enumerate(res["weights"]):
                    f.write(f"{e}," + ",".join(f"{v:.6g}" for v in w) + "\n")
            with open("classify_accuracy.csv", "w") as f:
                f.write("step,accuracy_ema\n")
                for i, a in enumerate(res["accuracy"]):
                    f.write(f"{i},{a:.6g}\n")
            print("Written classify_weights.csv / classify_accuracy.csv")

            from classify_plot import plot_run
            p = plot_run(res, K.DATASETS[name],
                         title=f"{name} · {routine} · measured on hardware",
                         path="classify_result.png")
            print(f"Written {p}")
        finally:
            md.all_off()
            md.close()
        return

    # ---------------------------------------------------------- simulation mode
    print(f"Simulation mode  Rs={SERIES_R/1000:.0f}kΩ  {args.epochs} epochs  "
          f"×{args.seeds} repeats  scramble={'yes' if scramble else 'no'}\n")
    head = f"{'Data set':<30s}{'Width':>4s}{'Shared':>7s}"
    print(head + "".join(f"{r:>17s}" for r in routines))
    print("-" * (len(head) + 17 * len(routines) + 8))

    for name in datasets:
        ds = K.DATASETS[name]
        width = max(len(p.spikes) for p in ds)
        seen = {}
        for p in ds:
            for i in p.spikes:
                seen.setdefault(i, set()).add(p.state)
        shared = sorted(i for i, s in seen.items() if len(s) > 1)

        cells = []
        for r in routines:
            accs = []
            for s in range(args.seeds):
                _, res = run_one(make_sim(s), name, r, args.epochs, s, scramble)
                accs.append(res["final_acc"])
            cells.append(f"{st.mean(accs)*100:5.1f}% ±{st.pstdev(accs)*100:4.1f}")
        print(f"{name:<30s}{width:>4d}{str(shared) if shared else '-':>7s}"
              + "".join(f"{c:>17s}" for c in cells))

    if args.plot:
        from classify_plot import plot_run
        print()
        for name in datasets:
            r0 = routines[0]
            kt, res = run_one(make_sim(0), name, r0, args.epochs, 0, scramble)
            out = f"classify_{name}.png"
            plot_run(res, K.DATASETS[name],
                     title=f"{name} · {r0} · simulation"
                           + (" (Scramble first)" if scramble else " (chip starts from the high-resistance state)"),
                     path=out)
            print(f"Written {out}   (final-epoch accuracy {res['final_acc']*100:.0f}%)")

    for name in datasets:
        if name in K.DATASET_NOTES:
            print(f"\n[{name}] {K.DATASET_NOTES[name]}")

    # trajectories of frustrated vs exclusive synapses
    if not args.dataset or args.dataset == "TwoPattern36Frustrated":
        print("\nWhere the frustrated synapses go (TwoPattern36Frustrated; synapses 2 and 5 appear in both T and F):")
        kt, res = run_one(make_sim(3), "TwoPattern36Frustrated", "LearnAlways",
                          args.epochs, 3, scramble)
        w = res["weights"]
        for i in range(8):
            tag = "shared" if i in (2, 5) else "exclusive"
            print(f"  synapse{i} [{tag}]  final weight {w[-1][i]:+.4f}   "
                  f"jitter over last 20 epochs σ={st.pstdev([x[i] for x in w[-20:]]):.4f}")


if __name__ == "__main__":
    main()
