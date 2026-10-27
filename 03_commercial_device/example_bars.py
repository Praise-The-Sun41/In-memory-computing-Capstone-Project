"""
example_bars.py — learn a "receptive field" with 8 memristor synapses: horizontal vs vertical bars

    python example_bars.py                # simulation
    python example_bars.py --hw           # real hardware (V2 board + Mode 2)

=================== Task ===================

A 3×3 pixel grid without its centre has exactly 8 pixels -- one pixel per synapse:

        0  1  2
        3  ·  4
        5  6  7

Training set (4 patterns):      Held-out set (2 patterns, not trained):
        ███  H                  · █ ·  V
        · · ·                   · █ ·

        · · ·                   · · ·  H
        ███  H                  █ · █

        █ · ·  V
        █ · ·

        · · █  V
        · · █

Why this task is worth showing
------------------------------
1. **What is learned is a picture, not a bar.** The 8 weights arranged as 3×3 form a
   "receptive field", showing directly which pixels the hardware learned to be discriminative.
2. **The result is counter-intuitive but explainable.** The discriminative information is not in the corners -- each corner
   appears once in a horizontal and once in a vertical bar, so they do not help and their weights stay at 0. The
   midpoints of the four edges do the work.
3. **The held-out set must fail, and this can be computed.** The pixel union of all horizontal patterns
   = the union of all vertical ones = 8 pixels, so Σ(horizontal sums) = Σ(vertical sums) = total weight sum,
   and it is impossible for one group to be all > 0 and the other all < 0. This is the capacity limit of a single-layer
   node, of the same origin as XOR -- an honest, provable point to show, far more interesting than "100% accuracy".

Hardware: V2 board, Mode switch at **2**, two equal-value precision resistors at A/B.
"""

from __future__ import annotations

import argparse
import random

import ktram as K

# pixel numbering of the 3×3 grid without centre
GRID = [[0, 1, 2], [3, None, 4], [5, 6, 7]]

# discriminative pixels (midpoints of the four edges) and corner pixels.
# a corner pixel appears once in a horizontal and once in a vertical bar, net drift 0, so its weight should stay at 0;
# the weights that actually grow are the four pixels on the cross.
CROSS = (1, 3, 4, 6)
CORNERS = (0, 2, 5, 7)


def build_patterns():
    """Generate the training and held-out sets. Pixel i maps directly to synapse i."""

    def P(pixels, state):
        return K.Pattern(tuple(sorted(pixels)), state)

    train = [(P((0, 1, 2), True), "Top row"),
             (P((5, 6, 7), True), "Bottom row"),
             (P((0, 3, 5), False), "Left col"),
             (P((2, 4, 7), False), "Right col")]
    held = [(P((3, 4), True), "Mid row"),
            (P((1, 6), False), "Mid col")]
    return train, held


SERIES_R = 20_000.0
PULSE_W_US = 50.0
# ---- device protection ----
# a fixed-amplitude script cannot invert the measured resistance as a closed loop can, so it uses values **safe in the worst case**:
# with drive ≤ 1.0 V a device with R→∞ gets at most 1.0 V (manual: "normally operated
# under 1V"), and with Rs=20kΩ the worst-case current is only 50 µA.
# the old value of 2.0 V gives 1.67 V on a 100 kΩ device -- beyond the manual's operating range.
V_FORWARD = 1.0        # = I_safe × Rs = 50 µA × 20 kΩ, the drive limit of the gate
V_REVERSE = 1.0
# ★ A set of measured operating points used to be recorded here:
#     0.8V 10/16 devices normal · 1.0V 12/16 · 1.3V 14/16 · 1.6V 15/16 · 2.0V 16/16
#   **These were measured on a chip that had already been damaged; do not use them as a basis.** Back then "cannot move"
#   was because devices had dropped to 690 Ω~1.5 kΩ; the voltage a low-resistance device gets, I·(R∥Rs), falls with it,
#   so the threshold could only be reached by raising amplitude (= raising current) -- exactly the path that damaged it.
#
#   The new chip is around 1 MΩ, the opposite situation: a 1.0 V drive is clipped by the gate to about 0.61 V by the device
#   voltage limit, and the device still gets the full 0.60 V, 2.3× the 0.26 V threshold.
#   **High-resistance devices do not need high voltage; they only need the current not to be let loose.**
#   So seeing [clip] is not a problem; it is the envelope working normally.


def render(spikes, on="█", off="·"):
    """Draw a pattern as 3 lines of text."""
    s = set(spikes)
    out = []
    for row in GRID:
        out.append(" ".join(" " if c is None else (on if c in s else off)
                            for c in row))
    return out


PIXEL_PATTERNS = [((0, 1, 2), True, "Top row"), ((5, 6, 7), True, "Bottom row"),
                  ((0, 3, 5), False, "Left col"), ((2, 4, 7), False, "Right col")]
PIXEL_HELD = [((3, 4), True, "Mid row"), ((1, 6), False, "Mid col")]


def print_patterns():
    print("Training set (H=T, V=F)                       Held-out set (not trained)")
    tr = [(render(sp), name, "H" if st else "V") for sp, st, name in PIXEL_PATTERNS]
    hd = [(render(sp), name, "H" if st else "V") for sp, st, name in PIXEL_HELD]
    for r in range(3):
        line = "   ".join(f"{g[r]}" for g, _, _ in tr)
        line2 = "   ".join(f"{g[r]}" for g, _, _ in hd)
        print(f"  {line}        {line2}")
    print("  " + "   ".join(f"{n}({c})" for _, n, c in tr)
          + "        " + "   ".join(f"{n}({c})" for _, n, c in hd))


def evaluate(kt, patterns):
    """Returns [(pattern, name, y, correct)]"""
    out = []
    for p, name in patterns:
        y = kt.execute(p, K.Instruction.FLV)
        out.append((p, name, y, (y > 0) == p.state))
    return out


# ================================================================ abs readout
# The default readout is sign(y), which has a **provable** capacity limit: the pixel union of the three horizontal patterns
# = the union of the three vertical ones = each pixel exactly once, so for any weight vector m,
#     Σ_H (m·P) = Σ_V (m·P) = Σ m
# to have all three H > t and all three V < t requires both S > 3t and S < 3t.
# switching to tanh / sigmoid does not help -- a monotonic function does not change sign(y), so the decision boundary is identical;
# the same holds for the device's own saturating nonlinearity (y = f(G_B) − f(G_A), f monotonically decreasing).
#
# a **non-monotonic** readout avoids it:
#     |y| > θ → H        |y| < θ → V
# targets Top=+1, Bottom=+1, MidRow=−2, three verticals=0, all six correct.
# MidRow=−2 is not arbitrary: Σ_H = Σ_V = 0 forces 1+1+x = 0.
#
# three costs that must be stated:
#   1. |·| is equivalent to a second layer with fixed weights -- it is no longer single-layer.
#   2. "which horizontal is positive and which negative" is a hand-assigned code; the hidden representation is hand-designed.
#   3. **the middle row/column must be trained**. With only the original 4 constraints the solution is underdetermined,
#      and even the ideal minimum-norm solution misclassifies the middle column (see research/study1_activation.py).
ABS_SPEC = [((0, 1, 2), +1, "Top row"),
            ((5, 6, 7), +1, "Bottom row"),
            ((3, 4),    -1, "Mid row"),
            ((0, 3, 5),  0, "Left col"),
            ((2, 4, 7),  0, "Right col"),
            ((1, 6),     0, "Mid col")]

ABS_IS_H = {"Top row": True, "Bottom row": True, "Mid row": True,
            "Left col": False, "Right col": False, "Mid col": False}


def abs_patterns():
    """The 6 patterns of abs mode (all trained, no held-out set)."""
    return [(K.Pattern(tuple(sp), tgt > 0), name) for sp, tgt, name in ABS_SPEC]


def calibrate_theta(kt, ys=None):
    """
    Calibrate the decision threshold θ on the training set: the midpoint of |y| between the horizontal and vertical groups.

    Returns (theta, margin). margin = min|y|_H − max|y|_V; negative means the two groups are not yet separated.
    """
    if ys is None:
        ys = {nm: kt.execute(K.Pattern(tuple(sp), True), K.Instruction.FLV)
              for sp, _, nm in ABS_SPEC}
    mh = [abs(ys[nm]) for nm in ys if ABS_IS_H[nm]]
    mv = [abs(ys[nm]) for nm in ys if not ABS_IS_H[nm]]
    return (min(mh) + max(mv)) / 2, min(mh) - max(mv)


def train_abs(kt, epochs=60, seed=0, record=True, k=0.05,
              stop_acc=None, stop_patience=2, stop_margin=0.05):
    """
    Three-valued target training. Uses only the three existing primitives, no new instructions:

        target +1 → FA + RB          (push y positive to saturation)
        target -1 → RA + FB          (push negative to saturation)
        target  0 → anti_hebbian     (push y towards 0; already in the library)

    Returns a dict of the same shape as K.train(), so bars_plot can be reused.

    **Accuracy is computed once at the end of each epoch, not at each presentation.** The sign-mode approach of
    "accounting at presentation + exponential moving average" gives a badly biased-low curve in abs mode
    (measured: true 6/6 at epoch end, but the curve only 0.5~0.86), for two reasons:

      1. **θ lags by a whole epoch.** The criterion compares |y| with θ, and θ can only be recalibrated after the epoch
         ends and all 6 have been read. Measured θ jumps between 0.047~0.117 from one epoch to the next.
      2. **Within-epoch swings of |y| are of the same order as θ.** Updates are unconditional; the |y| of the same pattern
         when read within the epoch can differ by 0.065 from its value at epoch end -- while θ itself is only about 0.05.
         Sign mode is unaffected because the sign decision saturates and a 0.065 swing does not matter.

    So here all 6 are read at the end of each epoch, and θ and accuracy are computed from **the same set of reads**.
    The extra cost is 0 -- calibrating θ requires reading these 6 anyway.

    stop_acc is the early-stopping threshold (None = do not stop). Training stops after stop_patience consecutive epochs on target,
    using the end-of-epoch accuracy above directly, with no extra read pulse.
    The gap between the two groups must also be ≥ stop_margin -- accuracy alone stops too early,
    when the decisions are all correct but the margin is still thin and cannot withstand read noise on hardware.
    """
    rng = random.Random(seed)
    data = list(ABS_SPEC)
    acc_trace, w_trace = [], []
    theta = 0.0
    hits, epochs_run = 0, 0
    for _ in range(epochs):
        epochs_run += 1
        rng.shuffle(data)
        for spikes, tgt, name in data:
            p = K.Pattern(tuple(spikes), tgt > 0)
            if tgt > 0:
                kt.execute(p, K.Instruction.FA)
                kt.execute(p, K.Instruction.RB)
            elif tgt < 0:
                kt.execute(p, K.Instruction.RA)
                kt.execute(p, K.Instruction.FB)
            else:
                kt.anti_hebbian(p)
        if record:
            w_trace.append(kt.read_all())
            # read all 6 once at epoch end; θ and accuracy both come from **this same set** of reads.
            # do not account at presentation time -- see the two reasons in the docstring above.
            ys = {nm: kt.execute(K.Pattern(tuple(sp), True), K.Instruction.FLV)
                  for sp, _, nm in ABS_SPEC}
            theta = calibrate_theta(kt, ys)[0]
            acc = sum((abs(ys[nm]) > theta) == ABS_IS_H[nm] for nm in ys) / 6.0
            acc_trace.append(acc)
            if stop_acc is not None:
                gap = calibrate_theta(kt, ys)[1]
                good = acc >= stop_acc and gap >= stop_margin
                hits = hits + 1 if good else 0
                if hits >= stop_patience:
                    break
    return {"accuracy": acc_trace, "weights": w_trace,
            "final_acc": acc_trace[-1] if acc_trace else 0.0, "theta": theta,
            "epochs_run": epochs_run,
            "stopped_early": stop_acc is not None and epochs_run < epochs}


def evaluate_abs(kt, patterns, theta):
    """Returns [(pattern, name, y, correct)], criterion |y| > theta."""
    out = []
    for p, name in patterns:
        y = kt.execute(p, K.Instruction.FLV)
        out.append((p, name, y, (abs(y) > theta) == ABS_IS_H[name]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw", action="store_true", help="Use real hardware instead of simulation")
    ap.add_argument("--chip", default=None,
                    help="Chip ID (required with --hw). Budget and degradation trend are filed under it.")
    ap.add_argument("--allow-drift", action="store_true",
                    help="If the start-up self-test finds resistance more than halved from its historical maximum, downgrade to a warning and continue")
    ap.add_argument("--rs-dmm", action="store_true",
                    help="With --hw one must be chosen: Rs was measured with a multimeter")
    ap.add_argument("--rs-nominal", action="store_true",
                    help="With --hw one must be chosen: Rs is only nominal; accept the scaling risk")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--routine", default="LearnAlways")
    ap.add_argument("--repeat", type=int, default=10,
                    help="After training, number of repeated reads of each pattern to measure margin")
    ap.add_argument("--vfwd", type=float, default=V_FORWARD)
    ap.add_argument("--vrev", type=float, default=V_REVERSE)
    ap.add_argument("--no-scramble", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stop-acc", type=float, default=0.9,
                    help="Stop early when end-of-epoch accuracy reaches this value for 2 consecutive epochs (default 0.9). "
                         "0 means no early stopping; run all --epochs")
    ap.add_argument("--stop-margin", type=float, default=0.05,
                    help="Early stopping also requires the margin (min |y|) to reach this value (default 0.05). "
                         "Accuracy alone stops too early -- the signs are right long before the weights have grown")
    ap.add_argument("--readout", choices=("sign", "abs"), default="sign",
                    help="sign = default, sign(y) decides H/V; middle row/column are held out and necessarily wrong; "
                         "abs = |y|>θ decides H/V, all 6 correct, but all 6 must be trained")
    args = ap.parse_args()

    print("=" * 68)
    print("Horizontal vs vertical bars -- 8 memristor synapses learn a receptive field")
    print("=" * 68 + "\n")
    print_patterns()
    print()

    hw = None
    if args.hw:
        if not args.chip:
            raise SystemExit(
                "--hw requires --chip <chip ID> as well.\n"
                "  Pulse budget, quarantine list and degradation trend are filed per chip; "
                "use a new ID for a new chip.")
        from example_classify import make_hw
        if args.rs_dmm == args.rs_nominal:
            raise SystemExit(
                "--hw requires one of --rs-dmm or --rs-nominal.\n"
                "  The board cannot determine Rs in principle (with no current the series resistance cannot be observed),\n"
                "  and a wrong Rs scales both the worst-case current and the resistance floor, so the source must be declared.")
        hw, backend = make_hw(chip_id=args.chip, purpose="bars-cli",
                              allow_drift=args.allow_drift,
                              rs_source="dmm" if args.rs_dmm else "nominal")
        print(f"write {args.vfwd} V / reset {args.vrev} V"
              f" (the gate clips again according to device resistance)\n")
    else:
        backend = K.SimBackend(rs=SERIES_R, seed=args.seed, eta=0.12)

    try:
        kt = K.KTRAM(backend, v_forward=args.vfwd, v_reverse=args.vrev)

        TRAIN, HELD = build_patterns()
        abs_mode = args.readout == "abs"
        theta = 0.0

        stop_acc = args.stop_acc if args.stop_acc > 0 else None

        if not args.no_scramble:
            kt.scramble(epochs=8, rng=random.Random(args.seed))

        if abs_mode:
            TRAIN, HELD = abs_patterns()[:4], abs_patterns()[4:]
            res = train_abs(kt, epochs=args.epochs, seed=args.seed, record=True,
                            stop_acc=stop_acc, stop_margin=args.stop_margin)
            theta = res["theta"]
        else:
            dataset = [p for p, _ in TRAIN]
            res = K.train(kt, dataset, args.routine, epochs=args.epochs,
                          seed=args.seed, record=True, stop_acc=stop_acc,
                          stop_margin=args.stop_margin)

        w = res["weights"][-1]
        res_px = res
        if abs_mode:
            how = f"three-valued targets + |y|>θ readout, θ={theta:.4f}"
        else:
            how = args.routine
        ran = res.get("epochs_run", args.epochs)
        if res.get("stopped_early"):
            print(f"Training complete (stopped early at epoch {ran}: accuracy ≥ "
                  f"{stop_acc*100:.0f}% and margin ≥ {args.stop_margin} for 2 consecutive epochs, "
                  f"limit {args.epochs}; {how})\n")
        else:
            print(f"Training complete ({ran} epochs, {how})\n")
        print("Learned receptive field (weights arranged 3×3):")
        for row in GRID:
            print("   " + " ".join("   ·   " if c is None else f"{w[c]:+7.3f}"
                                   for c in row))
        print()

        if abs_mode:
            theta, margin = calibrate_theta(kt)
            tr = evaluate_abs(kt, TRAIN, theta)
            hd = evaluate_abs(kt, HELD, theta)
        else:
            tr = evaluate(kt, TRAIN)
            hd = evaluate(kt, HELD)
        n_tr = sum(1 for *_, ok in tr if ok)
        n_hd = sum(1 for *_, ok in hd if ok)

        if abs_mode:
            print(f"Decision threshold θ = {theta:.4f} (calibrated on all 6), "
                  f"gap between the two groups = {margin:+.4f}\n")
            for tag, group in (("", tr), ("", hd)):
                for _, n, y, ok in group:
                    print(f"  {n:<11s} y={y:+.4f}  |y|={abs(y):.4f}  "
                          f"→ {'horizontal' if abs(y) > theta else 'vertical'}"
                          f" (truth {'horizontal' if ABS_IS_H[n] else 'vertical'})"
                          f" {'✓' if ok else '✗'}")
            print(f"\n  {n_tr + n_hd}/6 all correct" if n_tr + n_hd == 6
                  else f"\n  {n_tr + n_hd}/6")
            print("\nWith the **non-monotonic** readout |y|>θ, the original capacity limit is avoided.")
            print("  Three costs must be stated:")
            print("  1. |·| is equivalent to a second layer with fixed weights -- it is no longer single-layer.")
            print("  2. 'Which horizontal is positive and which negative' is a hand-assigned code; the hidden representation is hand-designed.")
            print("  3. The middle row/column **must be trained**: with only the original 4 the solution is underdetermined,")
            print("     and even the ideal minimum-norm solution misclassifies the middle column; it is not that the learning rule is not good enough.")
        else:
            print(f"training set {n_tr}/4 ", end="")
            print("  ".join(f"{n}:{y:+.3f}{'✓' if ok else '✗'}"
                            for _, n, y, ok in tr))
            print(f"held-out set {n_hd}/2 ", end="")
            print("  ".join(f"{n}:{y:+.3f}{'✓' if ok else '✗'}"
                            for _, n, y, ok in hd))

            print("\nThe held-out errors are **expected**, not a sign of poor training:")
            print("  the pixel union of all horizontal patterns = the union of all vertical ones = 8 pixels,")
            print("  so Σ(horizontal sums) = Σ(vertical sums) = total weight sum, and one group cannot be all>0 while the other is all<0.")
            print("  This is the capacity of a single-layer node on this task, of the same origin as XOR.")
            print("  To go beyond it: --readout abs (non-monotonic readout, all 6 correct).")

        # ---- margin and repeatability
        margins = None
        if args.repeat > 1:
            print(f"\nDecision margin (each pattern read {args.repeat} times)")
            margins = []
            for p, name in (TRAIN + HELD if abs_mode else TRAIN):
                ys = [kt.execute(p, K.Instruction.FLV) for _ in range(args.repeat)]
                mu = sum(ys) / len(ys)
                sd = (sum((v - mu) ** 2 for v in ys) / len(ys)) ** .5
                if abs_mode:
                    ok = sum(1 for v in ys if (abs(v) > theta) == ABS_IS_H[name])
                else:
                    ok = sum(1 for v in ys if (v > 0) == p.state)
                margins.append((name, mu, sd, ok, args.repeat))
                snr = abs(mu) / sd if sd > 1e-9 else 9e9
                print(f"  {name}  y = {mu:+.4f} ± {sd:.4f}   margin/σ = {snr:6.1f}"
                      f"   correct {ok}/{args.repeat}")

        # ---- figures
        try:
            from bars_plot import plot_bars
            path = plot_bars(res_px, w, tr, hd, margins,
                             title="Bar orientation - " +
                                   ("hardware" if args.hw else "Hardware"),
                             path="bars_result.png",
                             readout=args.readout, theta=theta)
            print(f"\nWritten {path}")
        except ImportError as e:
            print(f"\n(skipping figures: {e})")

        with open("bars_weights.csv", "w") as f:
            f.write("epoch," + ",".join(f"px{i}" for i in range(8)) + "\n")
            for e, ww in enumerate(res_px["weights"]):
                f.write(f"{e}," + ",".join(f"{v:.6g}" for v in ww) + "\n")
        print("Written bars_weights.csv")

    finally:
        if hw is not None:
            # session.__exit__ handles: open switches → stop waveform → turn off ±5 V → close device
            # → write back chip record → print session summary (clip/refuse/trip counts)
            hw.__exit__(None, None, None)


if __name__ == "__main__":
    main()
