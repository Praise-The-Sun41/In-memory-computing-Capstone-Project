"""
demo_cartpole.py —— 16 memristors learn to balance a pole

    python demo_cartpole.py                  # full three acts
    python demo_cartpole.py --clean-write    # control: precise write control
    python demo_cartpole.py --no-diagnose    # control: no self-test in act 3
    python demo_cartpole.py --flip           # extra in act 3: reverse the motor in front of the audience
    python demo_cartpole.py --gif            # export a pole animation (loop this during the demo)

=================== Why reinforcement learning ===================

Other on-chip learning demos (classification, regression) share one weakness: **they need labels**. When asked
"where do the labels come from", the answer is "computed off-chip", and half the story falls apart.

Reinforcement learning does not have this problem -- the only external signal is "how many steps this episode lasted", one number.

More importantly, it turns the chip's biggest **weakness** into a **necessity**:

    Reinforcement learning needs exploration noise. In a digital implementation this costs real money -- an LFSR or a true
    random number generator, taking area and power and needing verification.

    This chip: write step size varies log-normally by a factor of 3, 15% of pulses are lost, 5% go the wrong way,
    and reads at the high-conductance end return garbage 6% of the time. **That noise is already there, for free.**

The `--clean-write` control measured (8 seeds):

    | | Coarse write | Precise write |
    |---|---|---|
    | Learned (act 2) | 182 steps | 179 steps |
    | Relearned after burn-out (act 3) | 139 steps | 177 steps |

**The first row is the selling point of this demo; the second is not.** For learning a new task, write precision makes
almost no difference -- learning averages out the error. But to reconverge on an already damaged array, precision
is still valuable: coarse writing only recovers 139 steps. I first guessed that "coarse writing might even be better (noise as
exploration)"; the data did not support that, so it was changed.

=================== How the 16 devices are laid out ===================

Tile coding: 4 tilings × 2×2 tiles each = 16 tiles = 16 devices,
all four tilings over the (θ, θ̇) plane, offset from each other by 1/4 tile:

    any state activates **exactly** 1 tile in each tiling ⇒ |P| = 4

    u = G_P − c·|P| = Σ_{i∈P} w_i     one parallel read; the currents sum on the wire
    u > 0 → push right, otherwise push left

The four offset tilings cut each axis into 5 segments instead of 2, buying much finer resolution than 2×2 with 16
weights, plus built-in generalisation (neighbouring states share tiles). That is the entire forward computation:
**one read, zero multiplications.**

★ The policy does not include cart position -- 16 weights cannot hold a second plane (tried: adding it dropped the density
  of solutions from 9% to 3.6%, and learning failed). The cost is that the cart slowly drifts, so an episode is capped at
  200 steps (4 seconds). This is not arbitrary: an exhaustive software search for the best weights shows this feature set
  can hold at most 250~280 steps. **Capacity can be computed, and stating it is better than glossing over it.**

=================== Device numbering: an easily missed pitfall ===================

After erasing, every device sits at its own low-conductance end, and these ends range from 26 µS to 231 µS,
a factor of 9. If devices are mapped to tiles in arbitrary order, **process variation masquerades as a "learned
prior"** -- in 4 of 10 measured seeds, an untrained chip held the full 200 steps straight away. That ruins the demo.

So devices are grouped by window size (`demolib.group_by_window`), with similar windows in the same tiling.
After erasing, weights within a group are nearly equal and the policy is a blank slate. "Before learning" drops from an average
of 115 steps to 15~20 steps.

=================== Learning rule: three factors, feedback is one number ===================

One update after each complete episode:

    eligibility  e_i = Σ_t 1[i ∈ P_t] · (+1 push right / −1 push left)   16 local counters
    baseline     b ← b + 0.2·(R − b)                                      one scalar
    probability  p_i = p_max·[p_floor + (1−p_floor)·min(1, |R−b| / ⟨|R−b|⟩)]
                    · |e_i| / max|e|
    direction    sign( (R − b) · e_i )

Read as: **if this episode was better than usual, reinforce what you just did; if worse, pull it back;
the bigger the surprise, the more you move.** No backpropagation, no gradients, no Q-table.

Two details were found by tuning, both for a reason:

  · **The p_floor = 0.15 floor cannot be omitted.** Without it, once the policy is so bad that every episode is equally bad,
    R−b → 0 ⇒ update probability → 0 ⇒ nothing moves again. After devices are burned in act 3 this deadlock always
    occurs (measured: 7 of 10 seeds stuck at 11~20 steps).
  · **Each device has a net pulse budget (±8, leaking 3% per episode).** Without it, learning pushes devices all the way
    to a rail; at the rail they can only move one way, and the reverse correction needed in act 3 is impossible.

=================== Act 3: self-test after burn-out ===================

Over-current writing = stuck at the **low-resistance/high-conductance** end (the "burn out" in the manual's Critical Notes).
The stuck conductance still enters every parallel read as a fixed bias that cannot be moved.

    182 steps  →(burn 4)→  51 steps  →(self-test + relearn)→  139 steps
                                      without self-test only 87 steps

**How much it drops depends a lot on luck: 8 seeds range from 10 to 140 steps.** The reason is worth explaining: the largest
current flows through devices already in the low-resistance state, so the ones burned are often exactly those learning has
just pushed to the high-conductance end -- their stuck value is close to the learned value, so the damage is nearly invisible at first.
**Over-current damage is sometimes invisible at the moment it happens; what it takes first is future plasticity.**
`--flip` reverses the motor in front of the audience to expose this hidden loss: the correct policy is now to
flip the sign of every weight, and whether the damaged array can still do it is obvious at once.

How the self-test works: each device is pushed once in each direction, and is judged dead only if **it moves in neither** (trying only one
direction would misjudge devices that are alive but sitting at one end). Their measured conductances are then subtracted from the
reads, and the remaining devices face a clean problem. This is not closed-loop write verification (which is impossible here),
just a one-off read plus record, costing 16×(4 pulses + 4 reads).

`--no-diagnose` is the control.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import demolib as D                                        # noqa: E402

N_TILING, N_TILE = 4, 4
N_DEV = N_TILING * N_TILE


# ================================================================ environment
class CartPole:
    """Standard inverted pendulum with Gym CartPole-v1 parameters, so it is instantly recognisable."""
    g, m_c, m_p, ell, force, tau = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
    th_lim, x_lim = 15 * math.pi / 180, 2.4

    def __init__(self, rng, max_steps=200):
        self.rng, self.max_steps = rng, max_steps
        self.reset()

    def reset(self):
        self.s = self.rng.uniform(-0.05, 0.05, 4)      # x, ẋ, θ, θ̇
        self.t = 0
        return self.s

    def step(self, right):
        x, xd, th, thd = self.s
        f = self.force if right else -self.force
        ct, st = math.cos(th), math.sin(th)
        m = self.m_c + self.m_p
        tmp = (f + self.m_p * self.ell * thd * thd * st) / m
        thdd = (self.g * st - ct * tmp) / (
            self.ell * (4.0 / 3.0 - self.m_p * ct * ct / m))
        xdd = tmp - self.m_p * self.ell * thdd * ct / m
        self.s = np.array([x + self.tau * xd, xd + self.tau * xdd,
                           th + self.tau * thd, thd + self.tau * thdd])
        self.t += 1
        done = (abs(self.s[0]) > self.x_lim or abs(self.s[2]) > self.th_lim
                or self.t >= self.max_steps)
        return self.s, done


# ================================================================ tile coding
class Tiles:
    """4 tilings of 2×2 tiles over (θ, θ̇), offset from each other by 1/4 tile."""
    TH, TD = 0.26, 2.4
    OFFSETS = (-0.375, -0.125, 0.125, 0.375)

    def __init__(self, mapping=None):
        self.map = np.arange(N_DEV) if mapping is None else np.asarray(mapping)

    def __call__(self, s):
        th, td = s[2], s[3]
        on = []
        for k, off in enumerate(self.OFFSETS):
            a = int(np.clip(th / self.TH + off, -0.999, 0.999) >= 0)
            b = int(np.clip(td / self.TD + off, -0.999, 0.999) >= 0)
            on.append(int(self.map[k * N_TILE + a * 2 + b]))
        return on


# ================================================================ agent
class Agent:
    """The policy lives in 16 devices. The controller holds only: c, one baseline, 16 counters."""

    def __init__(self, board, v_drive, rng, n_read=3, top_k=16,
                 beta=0.20, p_max=0.50, p_floor=0.15, cap=8, leak=0.03,
                 width_us=50.0):
        self.io = D.FaultTolerantReadout(board, n_read=n_read, n_dev=N_DEV,
                                         width_us=width_us)
        self.tiles = Tiles(D.group_by_window(N_TILING))
        self.v, self.rng = v_drive, rng
        self.top_k, self.beta = top_k, beta
        self.p_max, self.p_floor = p_max, p_floor
        self.cap, self.leak = cap, leak
        self.net = np.zeros(N_DEV)
        self.baseline = None
        self.mad = 1.0                    # ⟨|R − b|⟩, used to normalise surprise
        self.e = np.zeros(N_DEV)          # eligibility trace (accumulated within an episode)
        self.best = 1.0                   # best score seen, used to switch exploration off

    # ---- power-on: erase the whole chip ----
    def erase(self, n=12):
        """
        Full-chip RESET. This is the starting point of the demo, and an honest one: **first show publicly that it knows nothing.**
        Each device receives n reverse pulses in a row, without verification (verification is not possible anyway).
        """
        for i in range(N_DEV):
            if i in self.io.known:
                continue
            self.io.b.select(i)
            # Erasing deliberately uses the calibration pulse width (50 µs): the goal here is to slam devices to the rail,
            # no fine steps needed. Narrow pulses are used only during learning.
            self.io.b.pulse(amplitude=-self.v, n=n, width_us=50.0)
        self.io.refresh_c()
        self.net[:] = 0.0
        self.e[:] = 0.0
        self.baseline, self.mad, self.best = None, 1.0, 1.0

    # ---- one episode ----
    def act(self, s):
        on = self.tiles(s)
        return on, (self.io.u(on) > 0.0)

    # ---- an episode split into three steps so the GUI can drive it frame by frame ----
    #
    # `episode()` is the command-line "run a whole episode"; the GUI renders frame by frame, so the same logic
    # is split into begin / record / finish. **Both paths run the same code**,
    # so the GUI and script versions cannot learn differently.
    def begin_episode(self):
        self.e = np.zeros(N_DEV)

    def record(self, on, right):
        self.e[on] += 1.0 if right else -1.0

    def finish_episode(self, R, learn=True):
        """Three-factor update at the end of an episode. Returns the number of pulses fired in this episode."""
        R = float(R)
        if not learn:
            return 0
        if self.baseline is None:
            self.baseline = R
            return 0

        adv = R - self.baseline
        self.baseline += self.beta * adv
        self.mad = 0.9 * self.mad + 0.1 * abs(adv)
        self.best = max(getattr(self, "best", R), R)
        e = self.e

        # keep the two mechanisms separate, not mixed in one probability -- mixing them self-destructs at the optimum:
        #   exploitation: reinforce/withdraw the recent actions in proportion to how much this episode differs from usual
        #   exploration: fire a random pulse in proportion to how far this episode is from the best seen
        # at the cap R ≡ b ⇒ adv = 0 ⇒ exploitation term is 0, and R/best = 1 ⇒ exploration term
        # is also 0, so it **stops by itself**. The first version treated adv=0 as "no direction information,
        # flip a coin"; after learning 200 steps it kept firing random pulses every episode and collapsed 200 → 104 → 12.
        em = float(np.abs(e).max()) or 1.0
        p_use = self.p_max * min(1.0, abs(adv) / max(self.mad, 1.0))
        p_rnd = self.p_max * self.p_floor * max(0.0, 1.0 - R / max(self.best, 1.0))

        fired = 0
        for i in np.argsort(-np.abs(e))[:self.top_k]:
            i = int(i)
            if i in self.io.known:
                continue              # known dead device: pushing it wastes pulses
            w = abs(e[i]) / em
            sg = 0.0
            if e[i] != 0.0 and adv != 0.0 and self.rng.random() < p_use * w:
                sg = math.copysign(1.0, adv * e[i])
            elif self.rng.random() < p_rnd:
                sg = 1.0 if self.rng.random() < 0.5 else -1.0
            if sg == 0.0:
                continue
            if self.cap and abs(self.net[i] + sg) > self.cap:
                continue              # net pulse budget used up: do not push it to the rail
            self.net[i] += sg
            fired += self.io.push([i], sg, self.v)
        self.net *= (1.0 - self.leak)
        self.io.refresh_c()
        return fired

    def episode(self, env, learn=True):
        s = env.reset()
        self.begin_episode()
        done = False
        while not done:
            on, right = self.act(s)
            self.record(on, right)
            s, done = env.step(right)
        R = float(env.t)
        return R, self.finish_episode(R, learn=learn)

    # ---- for plotting (never called during learning) ----
    def policy_map(self, n=61):
        w = self.io.weights()
        for i, g in self.io.known.items():
            w[i] = g - self.io.c       # dead devices: use the value measured in the self-test
        A = np.linspace(-Tiles.TH, Tiles.TH, n)
        B = np.linspace(-Tiles.TD, Tiles.TD, n)
        Z = np.zeros((n, n))
        s = np.zeros(4)
        for p, b in enumerate(B):
            for q, a in enumerate(A):
                s[2], s[3] = a, b
                on = self.tiles(s)
                Z[p, q] = sum(w[i] for i in on)
        return A, B, Z


# ================================================================ main flow
def main():
    ap = D.base_parser("16 memristors learn to balance a pole (on-chip reinforcement learning)")
    ap.add_argument("--episodes", type=int, default=150, help="Number of episodes in act 2")
    ap.add_argument("--episodes2", type=int, default=180, help="Number of episodes in act 3")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--p-max", type=float, default=0.50)
    ap.add_argument("--p-floor", type=float, default=0.15)
    ap.add_argument("--cap", type=float, default=8.0, help="Net pulse budget per device")
    ap.add_argument("--leak", type=float, default=0.03)
    ap.add_argument("--beta", type=float, default=0.20)
    ap.add_argument("--no-diagnose", dest="diagnose", action="store_false",
                    help="Control: no self-test in act 3, continue learning directly")
    ap.add_argument("--flip", action="store_true",
                    help="Extra in act 3: reverse the motor in front of the audience (correct policy flips sign everywhere)")
    ap.add_argument("--gif", action="store_true", help="Export pole animation")
    ap.set_defaults(width=50.0)   # for reinforcement learning, coarser steps actually converge faster
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    board = D.make_board(args)
    env = CartPole(rng, max_steps=args.max_steps)
    v = D.drive_voltage(args)

    print("=" * 74)
    print("16 memristors learn to balance a pole   ·   on-chip reinforcement learning · feedback is a single number")
    print("=" * 74)
    print(D.act1(args, extra=[
        "Task             inverted pendulum. 4 tilings × 4 = 16 devices; 4 connected per step, "
        "one parallel read, zero multiplications",
        "Feedback         only 'how many steps this episode lasted'; no labels, no gradients, no Q-table",
        "Exploration noise **is the write and read error itself** -- "
        "a digital chip needs a dedicated random number generator for this",
    ]))

    ag = Agent(board, v, rng, n_read=args.reads, beta=args.beta,
               p_max=args.p_max, p_floor=args.p_floor,
               cap=args.cap, leak=args.leak, width_us=args.width)

    print("\n  At power-on, erase the whole chip first (full-chip RESET, 16×12 reverse pulses, no verification)")
    ag.erase()
    base = float(np.mean([ag.episode(env, learn=False)[0] for _ in range(15)]))
    print(f"  Before learning: holds **{base:.0f} steps** on average"
          f" (cap {args.max_steps} steps = 4 s; the pole falls in 0.3 s)")

    # ---- act 2 ----
    print(f"\n[Act 2] On-chip reinforcement learning -- at the end of each episode, roll dice on the eligibility trace and fire a few pulses")
    print(f"  Episode  Mean steps (last 20)  Pulses this episode")
    curve, marks = [], {}
    for k in range(args.episodes):
        R, f = ag.episode(env)
        curve.append(R)
        if (k + 1) % 25 == 0:
            print(f"  {k+1:5d}     {np.mean(curve[-20:]):10.0f}          {f:6d}")
    learned = float(np.mean(curve[-20:]))
    w_before = ag.io.weights()
    map_before = ag.policy_map()
    print(f"  → mean **{learned:.0f} steps**"
          f" ({base:.0f} → {learned:.0f}, {learned/max(base,1):.0f}×), "
          f"{board.pulse_count} write pulses used so far")

    # ---- act 3 ----
    dead, after_burn, after_flip, found = [], None, None, {}
    if args.kill:
        dead = D.burn(board, args.kill, rng)
        ag.io.refresh_c()
        marks["burn"] = len(curve)
        after_burn = float(np.mean(
            [ag.episode(env, learn=False)[0] for _ in range(15)]))
        print(D.act3_header(dead, N_DEV - args.kill))
        print(f"  immediately drops to **{after_burn:.0f} steps**"
              f" (the stuck conductance still enters every parallel read as an immovable fixed bias)")
        if after_burn > 0.5 * learned:
            print(f"  The drop is small -- the largest current flows through devices already in the low-resistance state, "
                  f"i.e. the ones learning has just pushed up, ")
            print(f"  so the stuck value is close to the learned value. **In such cases the damage is invisible: "
                  f"what it takes first is future plasticity, not present performance.**")
        if args.flip:
            print(f"\n  One more step: reverse the motor in front of the audience (push right becomes push left):")
            env.force = -env.force
            marks["flip"] = len(curve)
            after_flip = float(np.mean(
                [ag.episode(env, learn=False)[0] for _ in range(15)]))
            print(f"  the correct policy becomes **every weight with flipped sign** -- "
                  f"immediately drops to **{after_flip:.0f} steps**")

        if args.diagnose:
            found = ag.io.diagnose(v)
            hit = sorted(set(found) & set(dead))
            print(f"  Self-test: push each device once in each direction (trying only one side would "
                  f"misjudge 'alive but at its limit' as dead) -- "
                  f"found {len(found)} dead; of the {len(dead)} truly dead, caught "
                  f"**{len(hit)}/{len(dead)}**")
            print(f"        Their measured conductances are subtracted from the reads, "
                  f"so the remaining {N_DEV - len(found)} devices face a clean problem")
        else:
            print(f"  (--no-diagnose: no check-up, continue learning directly. This is the control.)")

        # The old policy is now exactly reversed; erasing and starting over beats modifying it -- and learning has already
        # pushed devices to the rail, where they can only move one way, so a reverse correction cannot be made.
        print(f"  Erase the remaining devices and learn again from scratch:")
        ag.erase()
        print(f"  Episode  Mean steps (last 20)  Pulses this episode")
        for k in range(args.episodes2):
            R, f = ag.episode(env)
            curve.append(R)
            if (k + 1) % 25 == 0:
                print(f"  {k+1:5d}     {np.mean(curve[-20:]):10.0f}"
                      f"          {f:6d}")
        print(f"  → recovers **{np.mean(curve[-20:]):.0f} steps** by itself")

    # ---- wrap-up ----
    final = float(np.mean(curve[-20:]))
    print("\n" + "=" * 74)
    line = f"erased {base:.0f} steps  →  learned {learned:.0f} steps"
    if args.kill:
        line += f"  →  {args.kill} burned {after_burn:.0f} steps"
        if after_flip is not None:
            line += f"  →  motor reversed {after_flip:.0f} steps"
        line += (f"  →  {'self-test + ' if args.diagnose else ''}"
                 f"relearned {final:.0f} steps")
    print(line)
    print("=" * 74)
    print("  No backpropagation, no software copy of the weights, no closed-loop write verification; "
          "the only external signal is 'how many steps it lasted'.")
    print("  Exploration noise cost nothing -- it is the write and read error.")
    print(f"  Total cost: {board.pulse_count} write pulses + {board.read_count} reads")

    if not args.no_plot:
        plot(ag, curve, marks, map_before, args, base, learned,
             after_burn, after_flip, final, dead, found)
    if args.gif:
        make_gif(ag, CartPole(np.random.default_rng(args.seed + 99),
                              max_steps=args.max_steps))


# ================================================================ figures
def plot(ag, curve, marks, map_before, args, base, learned, after_burn,
         after_flip, final, dead, found, path="cartpole_result.png"):
    got = D.figure(13.4, 4.5)
    if got is None:
        return
    plt, fig, T, P = got
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.85, 1, 1],
                  wspace=.26, left=.055, right=.975, top=.73, bottom=.145)

    ax = fig.add_subplot(gs[0])
    x = np.arange(1, len(curve) + 1)
    ax.plot(x, curve, lw=.7, color=P["ink3"], alpha=.5)
    k = 15
    ax.plot(np.arange(k, len(curve) + 1),
            np.convolve(curve, np.ones(k) / k, mode="valid"),
            lw=2.2, color=P["good"])
    ax.axhline(base, color=P["ink3"], lw=1, ls=(0, (4, 3)))
    ax.text(len(curve), base + args.max_steps * .02,
            T("Erased chip", "blank chip"), ha="right", fontsize=8,
            color=P["ink3"])
    if "burn" in marks:
        b = marks["burn"]
        ax.axvline(b + .5, color=P["crit"], lw=1.8)
        ax.text(b + len(curve) * .012, args.max_steps * .05,
                T(f"{args.kill} burned by over-current → self-test → relearn",
                  f"burn {args.kill} devices -> self-test -> relearn"),
                fontsize=9, color=P["crit"], rotation=90, va="bottom")
    ax.set_xlim(0, len(curve))
    ax.set_ylim(0, args.max_steps * 1.06)
    ax.set_xlabel(T("Episode", "episode"), fontsize=9.5, color=P["ink2"])
    ax.set_ylabel(T("Steps held", "steps balanced"), fontsize=9.5,
                  color=P["ink2"])
    D.tidy(ax)
    D.title(ax, T("1. On-chip reinforcement learning curve", "1. On-chip RL"))

    for col, (mp, ttl) in enumerate([
            (map_before, T("2. Learned policy", "2. Learned policy")),
            (ag.policy_map(), T("3. After burn-out + self-test + relearning",
                                "3. After burn-out + self-test + relearn"))]):
        a = fig.add_subplot(gs[col + 1])
        A, B, Z = mp
        lim = float(np.abs(Z).max()) or 1.0
        a.pcolormesh(A, B, Z * 1e6, cmap="RdBu_r", shading="auto",
                     vmin=-lim * 1e6, vmax=lim * 1e6)
        a.contour(A, B, Z, levels=[0.0], colors=[P["ink"]], linewidths=1.8)
        a.axhline(0, color=P["ink3"], lw=.6)
        a.axvline(0, color=P["ink3"], lw=.6)
        a.set_xlabel(T("Pole angle θ (rad)", "pole angle θ (rad)"),
                     fontsize=9, color=P["ink2"])
        if col == 0:
            a.set_ylabel(T("Angular velocity ω (rad/s)", "angular rate ω (rad/s)"),
                         fontsize=9, color=P["ink2"])
        D.tidy(a, yaxis=False)
        D.title(a, ttl)
    fig.text(.70, .055, T("Red = push right, blue = push left, black line = decision boundary",
                          "red = push right, blue = push left"),
             fontsize=8, color=P["ink3"])

    sub = T(f"16 devices · window 2.9× · open-loop writes with 3× step jitter · "
            f"feedback is only 'steps held' | "
            f"{base:.0f} → {learned:.0f} steps"
            + (f" → {args.kill} burned {after_burn:.0f}"
               + (f" → motor reversed {after_flip:.0f}" if after_flip else "")
               + f" → self-test and relearn {final:.0f} steps" if marks else ""),
            f"16 devices | 2.9x window | open-loop coarse writes | "
            f"scalar reward | {base:.0f} -> {learned:.0f} steps"
            + (f" -> burn {args.kill}: {after_burn:.0f}"
               + (f" -> reversed: {after_flip:.0f}" if after_flip else "")
               + f" -> self-test + relearn {final:.0f}" if marks else ""))
    D.head(fig, T("Teaching 16 memristors to balance a pole",
                  "Teaching 16 memristors to balance a pole"), sub, x=.055)
    D.save(fig, path)
    plt.close(fig)


def make_gif(ag, env, path="cartpole.gif", n=200):
    """Export an animation of the pole after learning -- looping this during a demo works better than reading out numbers."""
    got = D.figure(5.2, 2.6)
    if got is None:
        return
    plt, fig, T, P = got
    import matplotlib.animation as anim
    ax = fig.add_axes([.04, .10, .92, .82])
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-.35, 1.25)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
    ax.plot([-2.6, 2.6], [0, 0], color=P["grid"], lw=2)
    cart, = ax.plot([], [], lw=11, color=P["ink2"], solid_capstyle="butt")
    pole, = ax.plot([], [], lw=4.5, color=P["good"], solid_capstyle="round")
    txt = ax.text(-2.5, 1.05, "", fontsize=9, color=P["ink2"])

    s, frames = env.reset(), []
    for _ in range(n):
        _, right = ag.act(s)
        s, done = env.step(right)
        frames.append(s.copy())
        if done:
            break

    def draw(i):
        x, _, th, _ = frames[i]
        cart.set_data([x - .22, x + .22], [.05, .05])
        pole.set_data([x, x + math.sin(th)], [.08, .08 + math.cos(th)])
        txt.set_text(f"step {i+1}")
        return cart, pole, txt

    anim.FuncAnimation(fig, draw, frames=len(frames), interval=20,
                       blit=True).save(path, writer=anim.PillowWriter(fps=50))
    plt.close(fig)
    print(f"Written {os.path.abspath(path)} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
