"""
bars_plot.py — visualisation of the horizontal/vertical bar experiment.

Four panels:
  1. pixel maps of the 6 patterns (4 training + 2 held out)
  2. formation of the receptive field — snapshots across epochs showing the cross emerging from noise
  3. final receptive field (large, with values)
  4. cumulative accuracy + decision margin of each pattern

The colour scheme reuses the CVD-checked diverging bipolar palette from classify_plot (blue <-> red, grey midpoint).
"""

from __future__ import annotations

from classify_plot import COL, _cmap

__all__ = ["plot_bars"]

GRID = [[0, 1, 2], [3, None, 4], [5, 6, 7]]


def _draw_grid(ax, values, vmax, cmap, show_values=False, title=None,
               title_size=9, hole=COL["surface"]):
    """Draw 8 values as a 3x3 grid (centre empty). values may be a dict/list, or a set for binary values."""
    import matplotlib.patches as mp
    binary = isinstance(values, (set, frozenset, tuple, list)) and \
        not isinstance(values, list) or isinstance(values, (set, frozenset))
    for r, row in enumerate(GRID):
        for c, idx in enumerate(row):
            x, y = c, 2 - r
            if idx is None:
                ax.add_patch(mp.Rectangle((x, y), .92, .92, facecolor=hole,
                                          edgecolor=COL["grid"], lw=1,
                                          linestyle=(0, (2, 2))))
                continue
            if binary:
                on = idx in values
                fc = COL["ink"] if on else COL["surface"]
                ec = COL["ink"] if on else COL["grid"]
            else:
                v = values[idx]
                fc = cmap(0.5 + 0.5 * (v / vmax if vmax else 0))
                ec = COL["surface"]
            ax.add_patch(mp.Rectangle((x, y), .92, .92, facecolor=fc,
                                      edgecolor=ec, lw=1.6))
            if show_values and not binary:
                v = values[idx]
                ax.text(x + .46, y + .46, f"{v:+.2f}", ha="center", va="center",
                        fontsize=8.5, color=COL["ink"] if abs(v) < vmax * .55
                        else COL["surface"], fontweight="600")
    ax.set_xlim(-.08, 3.0); ax.set_ylim(-.08, 3.0)
    ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=title_size, color=COL["ink2"], pad=4)


def _cls_of(name, p, is_abs, T):
    """In abs mode the ground truth comes from the name (middle row is horizontal, middle column vertical), not p.state."""
    if is_abs:
        from example_bars import ABS_IS_H
        isH = ABS_IS_H.get(name, p.state)
    else:
        isH = p.state
    return T("H", "H") if isH else T("V", "V")


def plot_bars(res, w_final, train_eval, held_eval, margins=None,
              title="", path="bars_result.png", show=False,
              readout="sign", theta=0.0):
    """
    readout="sign": default. The first 4 patterns are the training set and the last 2 are held out; criterion sign(y).
    readout="abs" : non-monotonic readout, criterion |y| > theta; **all** 6 patterns are used for training
                    (so nothing is marked "held out"); the margin panel plots |y| with θ as a vertical line.
    """
    is_abs = readout == "abs"
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from plotstyle import setup, T
    setup(verbose=False)

    W = res["weights"]
    acc = res["accuracy"]
    n_ep = len(W)
    vmax = max(1e-9, max(abs(v) for row in W for v in row))
    cmap = _cmap()

    fig = plt.figure(figsize=(13, 8.6), facecolor=COL["surface"])
    gs = GridSpec(3, 6, figure=fig, height_ratios=[1, 1, 1.05],
                  hspace=.62, wspace=.30,
                  left=.055, right=.965, top=.845, bottom=.075)

    row_head = {}          # row index -> first axes in that row, used to place section titles

    # ---------------------------------------------------- 1. pattern display
    for k, (p, name, y, ok) in enumerate(train_eval):
        ax = fig.add_subplot(gs[0, k])
        row_head.setdefault(0, ax)
        cls = _cls_of(name, p, is_abs, T)
        head = (f"{name} → {cls}   |y|={abs(y):.3f}" if is_abs
                else f"{name} → {cls}   y={y:+.3f}")
        _draw_grid(ax, set(p.spikes), vmax, cmap,
                   title=f"{head} {T('ok','ok') if ok else T('X','X')}")
    for k, (p, name, y, ok) in enumerate(held_eval):
        ax = fig.add_subplot(gs[0, 4 + k])
        cls = _cls_of(name, p, is_abs, T)
        tag = "" if is_abs else f"[{T('held','held')}] "
        val = f"|y|={abs(y):.3f}" if is_abs else f"y={y:+.3f}"
        _draw_grid(ax, set(p.spikes), vmax, cmap,
                   title=f"{tag}{name} → {cls}\n"
                         f"{val} {T('ok','ok') if ok else T('X','X')}")

    # ---------------------------------------------------- 2. receptive-field formation
    n_snap = 5
    idxs = [round(i * (n_ep - 1) / (n_snap - 1)) for i in range(n_snap)]
    for k, e in enumerate(idxs):
        ax = fig.add_subplot(gs[1, k])
        row_head.setdefault(1, ax)
        _draw_grid(ax, W[e], vmax, cmap, title=f"epoch {e}")

    # ---------------------------------------------------- 3. final receptive field
    ax = fig.add_subplot(gs[1, 5])
    _draw_grid(ax, w_final, vmax, cmap, show_values=True,
               title=T("Final (w = ga - gb)", "final (w = ga - gb)"))

    # ---------------------------------------------------- 4. accuracy
    ax = fig.add_subplot(gs[2, :4])
    ax.set_facecolor(COL["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(COL["grid"]); ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=COL["ink2"], labelsize=9, length=3)
    ax.grid(True, color=COL["grid"], lw=.8); ax.set_axisbelow(True)
    ax.plot(range(len(acc)), acc, lw=2, color=COL["accent"], solid_capstyle="round",
            marker="o" if is_abs and len(acc) <= 60 else None, ms=3.5,
            mfc=COL["accent"], mec=COL["surface"], mew=.8)
    ax.axhline(.5, lw=1.2, ls=(0, (4, 3)), color=COL["ink3"])
    ax.text(len(acc) * .004, .5, T(" chance 0.5", " chance 0.5"), va="bottom",
            fontsize=8.5, color=COL["ink3"])
    ax.set_ylim(-.03, 1.06); ax.set_xlim(0, max(1, len(acc) - 1))
    # In abs mode accuracy is computed once at the **end of each epoch** (all 6 patterns evaluated together), not as a running average during presentation --
    # accounting during presentation is badly biased low because θ lags by one epoch and |y| swings within an epoch by the same order as θ.
    ax.set_xlabel(T("epoch", "epochs") if is_abs
                  else T("pattern presentations", "presentations"),
                  fontsize=9.5, color=COL["ink2"])
    ax.set_ylabel(T("Accuracy", "accuracy") if is_abs
                  else T("Accuracy (EMA)", "accuracy (EMA)"),
                  fontsize=9.5, color=COL["ink2"])
    ax.set_title(T("3. Training-set accuracy (6 patterns evaluated at end of each epoch)",
                   "3  Training accuracy (all six, end of each epoch)") if is_abs
                 else T("3. Training-set accuracy", "3  Training accuracy"), loc="left",
                 fontsize=10.5, fontweight="600", color=COL["ink"], pad=6)

    # ---------------------------------------------------- 5. margin
    ax = fig.add_subplot(gs[2, 4:])
    ax.set_facecolor(COL["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(COL["grid"]); ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=COL["ink2"], labelsize=9, length=3)
    ax.grid(True, axis="x", color=COL["grid"], lw=.8); ax.set_axisbelow(True)

    if margins:
        names = [m[0] for m in margins]
        mus = [m[1] for m in margins]
        sds = [m[2] for m in margins]
        lab = T("4. Decision margin (±σ)", "4  Decision margin (±σ)")
    else:
        names = [n for _, n, _, _ in train_eval]
        mus = [y for _, _, y, _ in train_eval]
        sds = None
        lab = T("4. y of each pattern", "4  y per pattern")
    if is_abs:
        # the criterion compares |y| with θ, so plotting signed y would hide right/wrong
        mus = [abs(v) for v in mus]
        lab = T("4. |y| and threshold θ", "4  |y| against threshold")
    if is_abs:
        # |y| is always positive, so colouring by sign gives one colour -- colour by predicted class instead,
        # so it is obvious on which side of θ each group falls
        bar_col = [COL["pos"] if v > theta else COL["neg"] for v in mus]
    else:
        bar_col = [COL["pos"] if v >= 0 else COL["neg"] for v in mus]
    ax.barh(range(len(mus)), mus, xerr=sds, height=.6, color=bar_col,
            edgecolor=COL["surface"], linewidth=1.4,
            error_kw=dict(ecolor=COL["ink3"], lw=1.2, capsize=3))
    ax.axvline(0, color=COL["ink3"], lw=1)
    if is_abs:
        ax.axvline(theta, color=COL["ink3"], lw=1.4, ls=(0, (4, 3)))
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9, color=COL["ink2"])
    ax.invert_yaxis()
    lim = max(abs(v) for v in mus) * 1.7 or 1
    ax.set_xlim(0 if is_abs else -lim, lim)
    # θ goes in the axis label; do not look for space inside the plot -- it is already full of per-bar values
    ax.set_xlabel(f"|y|   (θ = {theta:.3f})" if is_abs else "y",
                  fontsize=9.5, color=COL["ink2"])
    ax.set_title(lab, loc="left", fontsize=10.5, fontweight="600",
                 color=COL["ink"], pad=6)
    for i, v in enumerate(mus):
        off = lim * .05
        ax.text(v + (off if v >= 0 else -off), i, f"{v:+.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8.5, color=COL["ink2"])

    # section titles: placed just above each row of axes so they do not clash with content
    heads = [
        (0, T("1. All 6 patterns are used for training (abs readout has no held-out set)",
              "1  All six patterns are trained (abs readout has no held-out set)")
            if is_abs else
            T("1. The 4 training patterns (left) and the 2 held-out patterns (right)",
              "1  Four training patterns (left), two held out (right)")),
        (1, T("2. Formation of the receptive field -- corner pixels grow, top and bottom midpoints pushed to 0",
              "2  The receptive field emerging; final state at right")
            if is_abs else
            T("2. Formation of the receptive field -- the cross emerges from noise; final result at right",
              "2  The receptive field emerging; final state at right")),
    ]
    for r, txt in heads:
        bb = row_head[r].get_position()
        fig.text(.055, bb.y1 + .028, txt, fontsize=10.5,
                 fontweight="600", color=COL["ink"], va="bottom")

    fig.suptitle(title or T("Receptive field learned by 8 memristor synapses",
                            "A receptive field learned on 8 memristor synapses"),
                 x=.055, ha="left", y=.965, fontsize=14,
                 fontweight="600", color=COL["ink"])

    fig.savefig(path, dpi=160, facecolor=COL["surface"])
    plt.show() if show else plt.close(fig)
    return path
