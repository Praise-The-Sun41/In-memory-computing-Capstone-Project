"""
scope_guard.py —— keeps scripts that are no longer in scope away from real hardware.

The current application scope has only two items: the **inverted pendulum** and the **3×3 horizontal/vertical bars**.
The other scripts (handwritten digits, channel equalisation, associative memory, the various example_* and check_*,
md-tune, research) remain in the repository because their analysis and conclusions are still useful;
but they are **no longer allowed to touch real devices**.

Why add a separate layer instead of relying on the gate alone:

  The gate already blocks them -- `MemristorDiscovery` refuses to apply voltage without a gate,
  and a gate can only be obtained from `hw_session.open_board()`. So these scripts
  are bound to fail on the real board.

  The problem is that **they fail badly**: the error reads "no protection gate attached", which looks like a bug,
  and the next step is easily someone trying to "get hold of a gate". This module says it up front:
  it was not forgotten, it is **deliberately withheld**.

  It also provides an explicit override (`--i-know` / allow=True), so that when an old script is really needed
  temporarily, it is **an explicit decision** that stays in the command-line history,
  rather than a silently changed default.
"""

from __future__ import annotations

import sys

IN_SCOPE = {
    "bars": "3×3 bars (bars_gui.py / example_bars.py)",
    "cartpole": "Inverted pendulum (demos/01_cartpole/)",
}


class OutOfScope(SystemExit):
    pass


def refuse_hardware(script: str, why: str = "", allow: bool = False,
                    suggest: str = ""):
    """
    Call before a script enters the hardware path. Only allow=True lets it through.

        if args.hw:
            scope_guard.refuse_hardware("example_digits.py",
                                        allow=args.i_know_what_im_doing)
    """
    if allow:
        print()
        print("=" * 72)
        print(f"★ {script} has been explicitly allowed to touch real hardware.")
        print("  It is outside the current scope (inverted pendulum / 3×3 bars); nobody maintains its safety path.")
        print("  The gate remains in effect throughout -- but this script's own procedure has not been reviewed against the new envelope.")
        print("=" * 72)
        print()
        return

    lines = [
        "",
        "=" * 72,
        f"{script} is not allowed to drive real hardware.",
        "=" * 72,
        "",
        "The current scope has only two applications:",
    ]
    for v in IN_SCOPE.values():
        lines.append(f"  · {v}")
    lines += [
        "",
        "This script stays in the repository for its analysis and conclusions, not to go on the board again.",
        "Its hardware path has not been reviewed against the new safety envelope (50 µA / 0.60 V / 20 kΩ floor).",
    ]
    if why:
        lines += ["", f"Specific reason: {why}"]
    if suggest:
        lines += ["", f"To do this, use: {suggest}"]
    lines += [
        "",
        "Simulation is unaffected: drop --hw and it runs.",
        "To really allow it, pass --i-know explicitly (so the decision stays in the command-line history).",
        "",
    ]
    raise OutOfScope("\n".join(lines))


def add_arg(ap):
    """Add the unified override switch to argparse."""
    ap.add_argument("--i-know", action="store_true",
                    help="Knowing this script is outside the current scope, still allow it to drive real hardware")
    return ap
