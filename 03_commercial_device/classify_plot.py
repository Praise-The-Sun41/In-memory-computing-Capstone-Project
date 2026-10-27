"""
classify_plot.py — plot the results of the kT-RAM classification experiment.

Three panels, each answering one question:

  1. Cumulative accuracy    Did it learn, and how fast?          -- single-series line
  2. Weight evolution map   Which way did each of 8 synapses go? -- diverging map (grey=0, red=positive, blue=negative)
  3. Final weights          What is the final division of labour? -- diverging bars

Why a heat map rather than 8 lines for the weights: 8 overlapping lines are a tangle, while a weight is essentially
a **polar quantity** (positive = favours True, negative = favours False, 0 = abstains). A diverging map shows polarity
directly, and "shared synapses stay at 0" is visible at a glance -- in a line plot those are the lines that get buried.

Colours come from a CVD-checked palette:
  diverging poles  blue #2a78d6 <-> red #e34948 (protan ΔE 21.6, normal ΔE 32.3, both pass)
  midpoint         light grey #f0efec (only a grey midpoint reads as "nothing")
"""

from __future__ import annotations

__all__ = ["plot_run", "COL"]

COL = {
    "accent": "#2a78d6",     # single-series line
    "neg": "#2a78d6",        # diverging cold pole
    "pos": "#e34948",        # diverging warm pole
    "mid": "#f0efec",        # diverging midpoint (grey, not a third hue)
    "ink": "#1a1a19",
    "ink2": "#52514e",
    "ink3": "#8a8a86",
    "grid": "#e4e4e0",
    "surface": "#fcfcfb",
}


def _cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "weights", [COL["neg"], COL["mid"], COL["pos"]], N=256)


def plot_run(res, dataset, title="", path="classify_result.png", show=False):
    """
    res     : return value of train() (requires record=True)
    dataset : the training pattern list, used to mark shared synapses
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from plotstyle import setup, T
    setup(verbose=False)

    acc = res["accuracy"]
    W = res["weights"]                      # [epoch][synapse]
    if not W:
        raise ValueError("res has no weights -- train() must be called with record=True")
    n_ep, n_syn = len(W), len(W[0])
    per_epoch = max(1, len(acc) // n_ep)    # patterns presented per epoch

    # synapses that appear in both T and F
    seen = {}
    for p in dataset:
        for i in p.spikes:
            seen.setdefault(i, set()).add(p.state)
    shared = {i for i, s in seen.items() if len(s) > 1}

    vmax = max(1e-9, max(abs(v) for row in W for v in row))

    fig = plt.figure(figsize=(11.5, 7.2), facecolor=COL["surface"])
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.35],
                  width_ratios=[2.6, 1], hspace=.42, wspace=.30,
                  left=.075, right=.955, top=.90, bottom=.085)

    def style(ax):
        ax.set_facecolor(COL["surface"])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(COL["grid"])
            ax.spines[s].set_linewidth(1)
        ax.tick_params(colors=COL["ink2"], labelsize=9, length=3, width=1)
        ax.grid(True, color=COL["grid"], lw=.8, alpha=.9)
        ax.set_axisbelow(True)

    # ------------------------------------------------ 1. cumulative accuracy
    ax = fig.add_subplot(gs[0, :]); style(ax)
    ax.plot(range(len(acc)), acc, lw=2, color=COL["accent"], solid_capstyle="round")
    ax.axhline(.5, lw=1.2, ls=(0, (4, 3)), color=COL["ink3"])
    ax.text(len(acc) * .004, .5, T(" chance 0.5", " chance 0.5"),
            va="bottom", ha="left", fontsize=8.5, color=COL["ink3"])
    # epoch boundaries: with many epochs drawing all becomes noise, so thin to at most ~20 lines
    step = max(1, round(n_ep / 20))
    for e in range(step, n_ep, step):
        ax.axvline(e * per_epoch, color=COL["grid"], lw=.8, zorder=0)
    ax.set_ylim(-.03, 1.06)
    ax.set_xlim(0, max(1, len(acc) - 1))
    ax.set_xlabel(T("pattern presentations (thin vertical lines = epoch boundaries)",
                    "presentations (thin rules = epoch boundary)"),
                  fontsize=9.5, color=COL["ink2"])
    ax.set_ylabel(T("Accuracy (EMA, k=0.05)", "accuracy (EMA, k=0.05)"),
                  fontsize=9.5, color=COL["ink2"])
    ax.set_title(T("1. Cumulative training accuracy", "1  Running training accuracy"),
                 loc="left", fontsize=11, fontweight="600", color=COL["ink"], pad=8)
    if acc:
        ax.plot([len(acc) - 1], [acc[-1]], "o", ms=6, color=COL["accent"],
                mec=COL["surface"], mew=1.6, zorder=5)
        ax.annotate(f"{acc[-1]*100:.0f}%", (len(acc) - 1, acc[-1]),
                    textcoords="offset points", xytext=(-6, 9),
                    ha="right", fontsize=10, fontweight="600", color=COL["accent"])
    ax.annotate(T(f"final full-set measurement {res['final_acc']*100:.0f}%",
                  f"final pass {res['final_acc']*100:.0f}%"),
                xy=(.995, .06), xycoords="axes fraction", ha="right",
                fontsize=9.5, color=COL["ink2"])

    # ------------------------------------------------ 2. weight evolution map
    ax = fig.add_subplot(gs[1, 0]); style(ax)
    ax.grid(False)
    im = ax.imshow([[W[e][i] for e in range(n_ep)] for i in range(n_syn)],
                   aspect="auto", cmap=_cmap(), vmin=-vmax, vmax=vmax,
                   interpolation="nearest", origin="upper",
                   extent=(-.5, n_ep - .5, n_syn - .5, -.5))
    ax.set_yticks(range(n_syn))
    ax.set_yticklabels(
        [f"syn {i}" + (" ●" if i in shared else "") for i in range(n_syn)],
        fontsize=9)
    for lbl, i in zip(ax.get_yticklabels(), range(n_syn)):
        if i in shared:
            lbl.set_color(COL["ink"]); lbl.set_fontweight("600")
        else:
            lbl.set_color(COL["ink2"])
    ax.set_xlabel(T("epoch", "epoch"), fontsize=9.5, color=COL["ink2"])
    ax.set_title(T("2. Weight evolution   w = ga - gb", "2  Weight evolution   w = ga - gb"),
                 loc="left", fontsize=11, fontweight="600", color=COL["ink"], pad=8)
    if shared:
        ax.annotate(T("● = shared synapse appearing in both T and F",
                      "● = shared between T and F patterns"),
                    xy=(0, -.185), xycoords="axes fraction",
                    fontsize=8.8, color=COL["ink3"])
    # leave a background gap between rows so the 8 rows read as 8 separate objects rather than one gradient
    for i in range(n_syn - 1):
        ax.axhline(i + .5, color=COL["surface"], lw=2.2, zorder=3)

    cb = fig.colorbar(im, ax=ax, pad=.015, fraction=.045)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=COL["ink2"], labelsize=8.5, length=2)
    # a short label at each end reads better than a sentence set vertically
    cb.ax.text(.5, 1.035, T("favours T", "toward T"), transform=cb.ax.transAxes,
               ha="center", va="bottom", fontsize=8.5, color=COL["pos"], fontweight="600")
    cb.ax.text(.5, -.035, T("favours F", "toward F"), transform=cb.ax.transAxes,
               ha="center", va="top", fontsize=8.5, color=COL["neg"], fontweight="600")

    # ------------------------------------------------ 3. final weights
    ax = fig.add_subplot(gs[1, 1]); style(ax)
    final = W[-1]
    ax.barh(range(n_syn), final, height=.62,
            color=[COL["pos"] if v >= 0 else COL["neg"] for v in final],
            edgecolor=COL["surface"], linewidth=1.4)
    ax.axvline(0, color=COL["ink3"], lw=1)
    ax.set_yticks(range(n_syn))
    ax.set_yticklabels([f"{i}" for i in range(n_syn)], fontsize=9, color=COL["ink2"])
    ax.invert_yaxis()
    ax.set_xlim(-vmax * 1.75, vmax * 1.75)
    ax.set_xlabel(T("Final weights", "final weight"), fontsize=9.5, color=COL["ink2"])
    ax.set_title(T("3. Final division of labour", "3  Final assignment"),
                 loc="left", fontsize=11, fontweight="600", color=COL["ink"], pad=8)
    for i, v in enumerate(final):                       # label values directly rather than relying on colour
        off = vmax * .06
        ax.text(v + (off if v >= 0 else -off), i, f"{v:+.3f}",
                va="center", ha="left" if v >= 0 else "right",
                fontsize=8.5, color=COL["ink2"])

    fig.suptitle(title or T("kT-RAM 8-synapse online supervised classification",
                            "kT-RAM 8-synapse online supervised classification"),
                 x=.075, ha="left", y=.972, fontsize=13.5,
                 fontweight="600", color=COL["ink"])

    fig.savefig(path, dpi=160, facecolor=COL["surface"])
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path
