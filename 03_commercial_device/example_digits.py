"""
example_digits.py —— 10-class handwritten digits with 16 memristors (Mode 1)

    python example_digits.py                  # simulation, sklearn's built-in 8×8 digits
    python example_digits.py --n-test 200     # test only 200 images, faster
    python example_digits.py --spread 10      # device conductance spread (real chip measured about 4×)
    python example_digits.py --faults 3       # inject 3 devices with faulty write direction
    python example_digits.py --mnist DIR      # use real MNIST instead (see load_data)
    python example_digits.py --hw             # real hardware (Mode switch at 1)

Needs numpy + scikit-learn. The simulated board and closed-loop write controller are reused directly from md-tune/tune_resistance.py.

=================== Architecture ===================

Mode 1, 16 devices each independent (**no differential pairs**), each device = one positive weight.
One select can connect several devices at once, and parallel conductances add -- KCL does the summation.

    input    28×28 → PCA 16 dims → binarise → 16-bit mask (= which devices to connect)
    weights  16 conductances, holding the template of only **one class** at a time
    score    connect the mask, one read pulse, measure the parallel resistance R
    decision load each of the 10 classes in turn, argmin R = argmax G = highest score

Time-multiplexing is necessary: 16 devices hold only 16 weights, while a 10-class linear classifier needs 160.
Fortunately programming happens only 10 times (once per class), after which the whole test set is scanned with the same weights,
so the cost is 10 programmings + N×10 reads, not N×10 programmings.

**Here the chip is an inference accelerator, not a learner** -- templates are trained in software and then written in.
This is different from the online learning in example_bars.py; do not mix them up when presenting.

=================== Three non-obvious points ===================

1. **The saturating nonlinearity is completely harmless under an argmax decision.** The measured quantity is the parallel resistance R,
   monotonically decreasing in G, and all ten classes share the same function, so argmin R ≡ argmax G with zero distortion.
   But it is harmless only within **a single read** -- weighted sums across multiple reads must convert R back to G first.

2. **Only positive weights can be used, at zero cost -- but the shift must be global.**
   score_k = Σ_{i∈P}(w_ki + c) = orig_k + |P|·c; the |P|·c term is the same for all ten classes,
   cancels in the comparison → argmax is strictly unchanged. **Shifting each class separately changes the argmax directly**
   (measured 77.6% → 37.4%). The same applies when scaling into the conductance window: min/max must also be taken globally.

3. **The bottleneck is binarisation, not the number of devices.** 16 continuous features reach 93% in software, 78% after binarisation;
   increasing PCA dimensions from 16 to 32 adds only 0.6 points. Savings are better made in input coding than in devices.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys

import numpy as np

import scope_guard

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "md-tune"))
import tune_resistance as T          # noqa: E402  simulated board + closed-loop write controller

N_DEV = 16
G_LO, G_HI = 5e-6, 2e-4              # programmable conductance window (200 kΩ .. 5 kΩ)
R_LO, R_HI = 1 / G_HI, 1 / G_LO


# ---------------------------------------------------------------- data
def load_data(mnist_dir=None, seed=0, test_frac=0.3):
    """
    By default use sklearn's built-in 8×8 digits (1797 images, bundled, no download needed).

    To run real MNIST: download the idx files into a directory yourself, then --mnist /path/to/mnist;
    the directory must contain train-images-idx3-ubyte and train-labels-idx1-ubyte.
    """
    from sklearn.model_selection import train_test_split
    if mnist_dir and mnist_dir.endswith(".npz"):
        d = np.load(mnist_dir)
        X = np.vstack([d["Xtr"], d["Xte"]]).astype(float)
        y = np.concatenate([d["ytr"], d["yte"]]).astype(int)
        name = f"MNIST 28x28 ({len(y)} images, from {os.path.basename(mnist_dir)})"
    elif mnist_dir:
        import gzip
        def rd(stem, img):
            for ext in (".gz", ""):
                path = os.path.join(mnist_dir, stem + ext)
                if os.path.exists(path):
                    op = gzip.open if ext == ".gz" else open
                    with op(path, "rb") as f:
                        f.read(4)
                        n = int.from_bytes(f.read(4), "big")
                        if img:
                            r = int.from_bytes(f.read(4), "big")
                            c = int.from_bytes(f.read(4), "big")
                            return np.frombuffer(f.read(), np.uint8).reshape(n, r * c)
                        return np.frombuffer(f.read(), np.uint8)
            # just reporting a path is useless -- most likely the data were never downloaded, so spell out the remedy
            hint = os.path.join(mnist_dir, "fetch_mnist.py")
            raise FileNotFoundError(
                f"{os.path.join(mnist_dir, stem)}[.gz] does not exist.\n"
                f"  MNIST data must be downloaded yourself:\n"
                f"    python {hint}\n"
                f"  afterwards {mnist_dir}/ should contain 4 idx.gz files (about 11 MB in total).\n"
                f"  If MNIST already exists elsewhere, point --mnist to that directory, or to an .npz\n"
                f"  (keys Xtr/ytr/Xte/yte).")
        X = rd("train-images-idx3-ubyte", True).astype(float)
        y = rd("train-labels-idx1-ubyte", False).astype(int)
        name = f"MNIST 28x28 ({len(y)} images)"
    else:
        from sklearn.datasets import load_digits
        d = load_digits()
        X, y, name = d.data, d.target, "sklearn built-in 8x8 digits"
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_frac,
                                          random_state=seed, stratify=y)
    return Xtr, Xte, ytr, yte, name


def quantize(Z, lo, hi, nbit):
    """Quantise features uniformly to 0..2^nbit−1. nbit=1 is binarisation."""
    L = 2 ** nbit - 1
    return np.round(np.clip((Z - lo) / (hi - lo), 0, 1) * L)


def bit_planes(Q, nbit):
    """Integer features → nbit 0/1 masks. Plane b is one select+read on hardware."""
    Qi = Q.astype(int)
    return [((Qi >> b) & 1).astype(float) for b in range(nbit)]


def calibrate_window(board, n_pulse=25, v=None, margin=0.15,
                     min_span=1.5, verbose=True):
    """
    Measure the resistance range each device can reach and take the **intersection** as the programmable window.

    This step cannot be skipped. If the target window is chosen arbitrarily (e.g. 5k..200k), with large device spread
    some devices cannot reach it at all -- measured write errors reach 12000%, and that class's template is useless.

    Why the intersection costs no accuracy: the argmax decision is completely immune to a **common** proportional scaling and shift
    (score_k = Σ(a·w_ki + b) = a·orig_k + |P|·b, all ten classes change together).
    So a narrower window only compresses the noise margin; it does not change decisions.
    """
    # amplitude computed from the manual limits (device_limits.py). The old hard-coded 2.0 V = worst case 200 µA,
    # which manual Note 1 says **irreversibly** lowers the high-resistance state, i.e. narrows this very window.
    import device_limits as _L
    v = _L.max_drive(board.rs) if v is None else v
    lo, hi, dead = [], [], []
    for i in range(N_DEV):
        board.select(i)
        for _ in range(n_pulse):
            board.pulse(amplitude=+v, n=1)
        a = board.read_resistance(i)
        for _ in range(n_pulse):
            board.pulse(amplitude=-v, n=1)
        b = board.read_resistance(i)
        # **no polarity assumed**: for a reversed device the two reads are swapped, so take min/max directly.
        # assuming polarity computes the intersection backwards; measured, 3 bad devices can collapse accuracy to 1%.
        r1, r2 = min(a, b), max(a, b)
        if r2 / max(r1, 1.0) < min_span:            # cannot move → dead device
            dead.append(i)
        else:
            lo.append(r1)
            hi.append(r2)
    if not lo:
        raise RuntimeError("None of the 16 devices can be moved; run check_ktram.py --hard-erase first")
    r_lo, r_hi = max(lo), min(hi)                   # intersection of live devices
    span = r_hi / r_lo if r_lo > 0 else 0
    r_lo, r_hi = r_lo * (1 + margin), r_hi * (1 - margin)
    if verbose:
        print(f"\nMeasured programmable window: {r_lo/1e3:.1f}k .. {r_hi/1e3:.0f}k Ω"
              f" (intersection of {len(lo)} live devices, {margin*100:.0f}% margin, span {span:.1f}×)")
        if dead:
            print(f"  ★ Devices that cannot be moved (chip numbers): {[i+1 for i in dead]} -- excluded.")
            print("    Their conductance is **the same constant** for all ten classes and cancels in the argmax,")
            print("    so excluding them does not affect decisions; there are just fewer effective weights.")
        if span < 3:
            print("  ★ Span below 3×; weight resolution will be poor -- first run "
                  "check_ktram.py --hard-erase to recover stuck devices")
    return r_lo, r_hi, dead


def make_templates(Btr, ytr, r_lo=None, r_hi=None):
    """
    Train 10 class templates in software, then map them linearly to target resistances.

    Both transforms must be done **globally**, not per class:
      · shift to non-negative: score_k += |P|·c, the same for all ten classes, argmax unchanged.
      · scale into the conductance window: likewise.
    Doing each class separately changes the argmax.
    """
    from sklearn.linear_model import LogisticRegression
    g_lo = 1.0 / (r_hi if r_hi else R_HI)
    g_hi = 1.0 / (r_lo if r_lo else R_LO)
    # **fit_intercept=False is required.** The hardware score is just Σ_{i∈P} g_ki, with nowhere
    # to put a per-class constant; a model trained with an intercept cannot be implemented on hardware, and evaluation is inflated
    # (measured on real MNIST, 4 bit was misreported 83.4% vs 20.7% ... and in the wrong direction, because
    # the argmax without the intercept is meaningless). For an intercept, use --bias to dedicate a device to it.
    W = LogisticRegression(max_iter=2000, fit_intercept=False).fit(Btr, ytr).coef_
    Wp = W - W.min()                                               # global shift
    G = g_lo + (Wp - Wp.min()) / (Wp.max() - Wp.min()) * (g_hi - g_lo)
    return W, 1.0 / G                                              # target resistance


# ---------------------------------------------------------------- simulated board
class DigitsSim(T.SimBoard):
    """
    Adds two things only a real chip has to md-tune's SimBoard:

      · **per-device resistance limits**. The original shares one r_on/r_off pair across all devices,
        and writing evens them out, so device spread cannot be simulated -- yet on this chip the measured
        single-device conductance differs by 3.8× between synapses, the main error source.
      · **parallel read**. Classification connects a batch of devices and reads the parallel resistance; the original reads only one.
    """

    def __init__(self, spread=1.0, **kw):
        super().__init__(**kw)
        r = random.Random(kw.get("seed", 0) * 977 + 13)
        s = max(1.0, spread)
        # each device's own low-resistance end: spread log-uniformly over [r_on, r_on·spread]
        self.r_on_i = [math.exp(r.uniform(math.log(self.r_on),
                                          math.log(self.r_on * s)))
                       for _ in range(N_DEV)]
        self.r_off_i = [math.exp(r.uniform(math.log(self.r_off / s),
                                           math.log(self.r_off)))
                        for _ in range(N_DEV)]
        self.sel_all = (0,)

    def select(self, *idx, settle=0.0):
        self.sel_all = tuple(idx) if idx else (0,)
        self.sel = self.sel_all[0]

    def read_parallel(self):
        """Parallel resistance of the connected batch (this is what classification scores use)."""
        self.read_count += 1
        g = sum(1.0 / self.r[i] for i in self.sel_all)
        return (1.0 / g) * math.exp(self.rng.gauss(0, self.noise))

    def pulse(self, amplitude=1.0, width_us=T.PULSE_WIDTH_US, n=1, **kw):
        i = self.sel
        fault = self.faults.get(i)
        for _ in range(n):
            self.pulse_count += 1
            r = self.r[i]
            v_dev = abs(amplitude) * r / (r + self.rs)     # voltage division
            if v_dev <= self.v_th:
                continue
            drive = min(1.0, (v_dev - self.v_th) / self.v_th)
            step = max(0.0, min(1.0, self.eta * drive *
                                (1 + self.rng.gauss(0, self.noise))))
            if fault == "stuck":
                continue
            if fault == "weak":
                step *= 0.15
            setting = amplitude > 0
            if fault == "flip":
                setting = not setting
            lo, hi = self.r_on_i[i], self.r_off_i[i]       # per-device limits
            lr = math.log(r)
            lr += step * ((math.log(lo) if setting else math.log(hi)) - lr)
            self.r[i] = min(hi, max(lo, math.exp(lr)))


class HwBoard:
    """Real hardware. A wrapper that adds the parallel read."""

    def __init__(self, rs, board_version=2):
        from knowm_md import MemristorDiscovery
        self.md = MemristorDiscovery(series_resistance=rs,
                                     board_version=board_version)
        self.md.open()
        self.md.set_power(True)
        self.rs = rs
        self.board_version = board_version
        self.sel_all = (0,)
        self.pulse_count = self.read_count = 0

    def select(self, *idx, settle=0.005):
        self.sel_all = tuple(idx) if idx else (0,)
        self.md.select(*self.sel_all, settle=settle)

    def read_resistance(self, index=None, **kw):
        self.read_count += 1
        return self.md.read_resistance(index, **kw)

    def read_parallel(self):
        self.read_count += 1
        return self.md.read_resistance(None, v_read=T.V_READ,
                                       width_us=T.READ_WIDTH_US)

    def pulse(self, **kw):
        self.pulse_count += 1
        return self.md.pulse(**kw)

    def all_off(self):
        self.md.all_off()

    def close(self):
        try:
            self.md.all_off()
            self.md.set_power(False)
        finally:
            self.md.close()


# ---------------------------------------------------------------- main flow
def program_class(tuner, targets, skip=(), verbose=False):
    """Write the target resistances of one class in closed loop (skipping dead devices in skip)."""
    n0 = tuner.b.pulse_count
    errs = []
    for i, r_t in enumerate(targets):
        if i in skip:
            continue
        rec = tuner.tune(i, float(r_t), autopol=True)
        errs.append(rec["err"] if rec["err"] == rec["err"] else float("nan"))
        if verbose and rec["status"] != "ok":
            print(f"      device {i+1}: {rec['status']} {rec['note']}")
    return tuner.b.pulse_count - n0, errs


def main():
    ap = argparse.ArgumentParser(
        description="10-class handwritten digits with 16 memristors (Mode 1, time-multiplexed)")
    ap.add_argument("--hw", action="store_true", help="Real hardware (Mode switch at 1)")
    ap.add_argument("--i-know", action="store_true",
                    help="Allow this script to drive real hardware even though it is known to be out of current scope")
    ap.add_argument("--mnist", help="Directory of real MNIST idx files")
    ap.add_argument("--n-test", type=int, default=300, help="Number of test images (default 300)")
    ap.add_argument("--bits", type=int, default=1,
                    help="Input quantisation bits (default 1 = binary). Each extra bit adds one read pulse per image per class "
                         "without rewriting weights. On real MNIST 1/2/3/4 bit ≈ "
                         "68/77/83/84%%")
    ap.add_argument("--bias", action="store_true",
                    help="Dedicate one device to a constant term (the other 15 hold features). "
                         "The hardware score is a pure Σg with no intercept; this is the only way to add one")
    ap.add_argument("--spread", type=float, default=4.0,
                    help="Device spread factor in simulation (this chip measured about 4)")
    ap.add_argument("--faults", type=int, default=0,
                    help="Number of devices with faulty write direction injected in simulation")
    ap.add_argument("--tol", type=float, default=5.0,
                    help="Write dead band %% (default 5). Tighten for more bits -- see --bits")
    ap.add_argument("--read-noise", type=float, default=0.5,
                    help="Simulated read noise %% (default 0.5, close to measured)")
    ap.add_argument("--rs", type=float, default=10_000.0,
                    help="Mode 1 series resistance Ω (two 20k in parallel = 10k)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    print("=" * 74)
    print("16 memristors · 10-class handwritten digits · Mode 1 time-multiplexed")
    print("=" * 74)

    # ---- 1. data and encoding
    Xtr, Xte, ytr, yte = load_data(args.mnist, args.seed)[:4]
    name = load_data(args.mnist, args.seed)[4]
    from sklearn.decomposition import PCA
    n_feat = N_DEV - 1 if args.bias else N_DEV
    pca = PCA(n_components=n_feat, random_state=args.seed).fit(Xtr)
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
    lo, hi = np.percentile(Ztr, 1, 0), np.percentile(Ztr, 99, 0)
    L = 2 ** args.bits - 1
    Qtr, Qte = quantize(Ztr, lo, hi, args.bits), quantize(Zte, lo, hi, args.bits)
    if args.bias:                      # constant column = a device that is always connected
        Qtr = np.hstack([Qtr, np.full((len(Qtr), 1), L)])
        Qte = np.hstack([Qte, np.full((len(Qte), 1), L)])
    n_test = min(args.n_test, len(yte))
    Qte, yte = Qte[:n_test], yte[:n_test]
    planes = bit_planes(Qte, args.bits)
    print(f"Data {name}   train {len(ytr)}  test {n_test}")
    print(f"PCA {Xtr.shape[1]} → {n_feat} dims"
          + (" (+1 bias device)" if args.bias else "")
          + f", retained variance {pca.explained_variance_ratio_.sum()*100:.1f}%")
    print(f"Input quantised to {args.bits} bit → {args.bits} read pulses per image per class, "
          f"on average {np.mean([pl.sum(1).mean() for pl in planes]):.1f}/16 devices connected")

    # ---- 3. build board
    if args.hw:
        scope_guard.refuse_hardware(
            'Handwritten digits with 16 devices (example_digits.py)',
            why='It uses Mode 1 / Rs = 10 kΩ, and at 10 kΩ a device at the floor gets only 0.28 V, a margin of only 7%',
            suggest='H/V 3×3 bars: python example_bars.py --hw --chip <chip ID>',
            allow=getattr(args, 'i_know', False))
        board = HwBoard(args.rs)
        print(f"\nHardware mode  Rs={args.rs/1e3:.0f}kΩ   ★ Mode switch must be at 1")
    else:
        faults = {}
        rng = random.Random(args.seed + 99)
        for i in rng.sample(range(N_DEV), min(args.faults, N_DEV)):
            faults[i] = rng.choice(["flip", "stuck", "weak"])
        board = DigitsSim(spread=args.spread, rs=args.rs, seed=args.seed,
                          faults=faults, noise=args.read_noise / 100.0)
        print(f"\nSimulation mode  Rs={args.rs/1e3:.0f}kΩ  device spread ×{args.spread:g}  "
              f"write dead band ±{args.tol:g}%  read noise {args.read_noise:g}%"
              + (f"  injected faults {faults}" if faults else ""))
    tuner = T.Tuner(board, tol=args.tol / 100.0, averages=3,
                    settle=0.0 if not args.hw else 0.002, verbose=False)

    # ---- 2. measure the programmable window, then train templates in software and map them in
    r_lo, r_hi, dead = calibrate_window(board)
    W, R_target = make_templates(Qtr, ytr, r_lo, r_hi)
    acc_sw = (np.argmax(Qte @ W.T, axis=1) == yte).mean() * 100
    print(f"\nSoftware upper bound (same binary input, ideal weights): {acc_sw:.1f}%")
    print(f"Target resistance range {R_target.min()/1e3:.1f}k .. {R_target.max()/1e3:.0f}k Ω")

    # ---- 4. program each class + read the whole test set
    print(f"\n{'Cls':>3s}{'Writes':>9s}{'Write error median/max':>22s}{'Reads':>9s}")
    scores = np.zeros((n_test, 10))
    total_pulse = 0
    for k in range(10):
        npul, errs = program_class(tuner, R_target[k], skip=dead)
        total_pulse += npul
        e = np.abs([v for v in errs if v == v]) * 100
        r0 = board.read_count
        for j in range(n_test):
            tot = 0.0
            for b, pl in enumerate(planes):
                on = np.nonzero(pl[j])[0].tolist()
                if not on:
                    continue
                board.select(*on)
                # ★ must convert back to G before summing with bit weights. R is nonlinear in G,
                #   and adding R (or −R) directly across bit planes destroys accuracy.
                tot += (2 ** b) / board.read_parallel()
            scores[j, k] = tot
        print(f"{k:>3d}{npul:>9d}{f'{np.median(e):.1f}% / {e.max():.1f}%':>22s}"
              f"{board.read_count - r0:>9d}")

    # ---- 5. results
    pred = scores.argmax(1)
    acc = (pred == yte).mean() * 100
    print("\n" + "=" * 74)
    print(f"Hardware accuracy {acc:.1f}%    (software upper bound {acc_sw:.1f}%, "
          f"loss {acc_sw-acc:.1f} points)")
    print("=" * 74)
    cm = np.zeros((10, 10), int)
    for t, p in zip(yte, pred):
        cm[t, p] += 1
    per = np.array([cm[i, i] / max(1, cm[i].sum()) * 100 for i in range(10)])
    print(f"  Per-class accuracy  " + "  ".join(f"{i}:{per[i]:.0f}%" for i in range(10)))
    worst = int(np.argmin(per))
    conf = int(np.argmax(cm[worst] - np.eye(10, dtype=int)[worst] * cm[worst, worst]))
    print(f"  Worst is {worst} ({per[worst]:.0f}%), most often confused with {conf}")
    print(f"\n  Total cost: {total_pulse} write pulses (one-off) + "
          f"{n_test*10*args.bits} read pulses ({args.bits} bit × 10 classes)")
    if args.hw:
        print(f"  Measured read pulses {board.read_count}")

    try:
        board.all_off()
        board.close()
    except Exception:
        pass

    # ---- 6. figures
    if not args.no_plot:
        try:
            path = plot(cm, per, acc, acc_sw, args)
            print(f"\nWritten {path}")
        except ImportError as e:
            print(f"\n(skipping figures: {e})")
    np.savetxt("digits_confusion.csv", cm, fmt="%d", delimiter=",")
    print("Written digits_confusion.csv")


def plot(cm, per, acc, acc_sw, args, path="digits_result.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    try:
        from plotstyle import setup, T as TR
        setup(verbose=False)
    except ImportError:
        def TR(zh, en):
            return en

    SURF, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8a84"
    GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"
    # single-hue sequential scale for the confusion matrix; status colours for per-class accuracy, both with value labels
    BLUES = ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
             "#256abf", "#184f95", "#0d366b"]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", BLUES)

    fig = plt.figure(figsize=(11.5, 5.4), facecolor=SURF)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.15, 1],
                  wspace=.28, left=.07, right=.97, top=.80, bottom=.13)

    ax = fig.add_subplot(gs[0])
    frac = cm / np.maximum(1, cm.sum(1, keepdims=True))
    ax.imshow(frac, cmap=cmap, vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7.5,
                        color="white" if frac[i, j] > .5 else INK2)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.tick_params(labelsize=8.5, colors=INK2, length=0)
    ax.set_xlabel(TR("Predicted", "predicted"), fontsize=9.5, color=INK2)
    ax.set_ylabel(TR("True", "true"), fontsize=9.5, color=INK2)
    ax.set_title(TR("Confusion matrix (counts in cells)", "Confusion matrix (counts)"),
                 loc="left", fontsize=10.5, fontweight="600", color=INK, pad=8)
    for s in ax.spines.values():
        s.set_visible(False)

    ax = fig.add_subplot(gs[1])
    col = [GOOD if v >= 80 else (WARN if v >= 60 else CRIT) for v in per]
    ax.bar(range(10), per, color=col, width=.62, zorder=3)
    for i, v in enumerate(per):
        ax.text(i, v + 1.5, f"{v:.0f}", ha="center", va="bottom",
                fontsize=8, color=INK2)
    ax.axhline(acc, color=INK3, lw=1.4, ls=(0, (4, 3)))
    ax.text(-0.45, acc + 1.5, TR(f"Overall {acc:.1f}%", f"overall {acc:.1f}%"),
            ha="left", va="bottom", fontsize=8.5, color=INK3)
    ax.set_ylim(0, 108)
    ax.set_xticks(range(10)); ax.tick_params(labelsize=8.5, colors=INK2, length=3)
    ax.set_xlabel(TR("Digit", "digit"), fontsize=9.5, color=INK2)
    ax.set_ylabel(TR("Accuracy (%)", "accuracy (%)"), fontsize=9.5, color=INK2)
    ax.grid(True, axis="y", color="#e6e5e1", lw=.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(TR("Per-class accuracy", "Per-digit accuracy"), loc="left",
                 fontsize=10.5, fontweight="600", color=INK, pad=8)

    mode = "hardware" if args.hw else f"simulation (spread x{args.spread:g}"\
                                      f"{f', {args.faults} faulty' if args.faults else ''})"
    fig.text(.07, .93, TR("10-class handwritten digits with 16 memristors",
                          "Ten-class handwritten digits on 16 memristors"),
             fontsize=14, fontweight="bold", color=INK)
    fig.text(.07, .875, TR(f"Mode 1 time-multiplexed · {mode} · hardware {acc:.1f}% / "
                           f"software upper bound {acc_sw:.1f}%",
                           f"Mode 1, time-multiplexed - {mode} - "
                           f"{acc:.1f}% vs {acc_sw:.1f}% software ceiling"),
             fontsize=9.5, color=INK2)
    fig.savefig(path, dpi=150, facecolor=SURF)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
