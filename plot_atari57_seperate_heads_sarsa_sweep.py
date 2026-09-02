"""Per-game summary of the Atari-57 separate-value-head SARSA sweep.

Layout it expects (as produced by
job_scripts/atari/atari57_seperate_heads_sarsa_sweep):

    <root>/<game>/beta_<b>/intrinsic_gamma_<gI>/epsilon_<e>/seed_<s>/metrics.npz

57 games x 33 (beta, gamma_I, epsilon) combinations x 5 seeds. gamma_E is pinned at
0.99 sweep-wide and so never appears in the path. beta = 0 is the extrinsic-only
control (intrinsic_loss_coef = 0.0 as well) and takes a single gamma_I, since a frozen
intrinsic head makes gamma_I inert there.

This is the sibling of plot_atari4_seperate_heads_gamma_sweep.py for a differently
shaped sweep: that one varies two discounts at a fixed epsilon, this one varies the
intrinsic discount and the exploration floor. Everything except the layout, the metric
set, the legend text and the CSV columns is imported from plot_count_layer_sweep --
smoothing, seed aggregation, the Student-t band, the palette and the axis styling stay
defined in one place.

What this sweep exists to measure
---------------------------------
`intrinsic_action_divergence` is the fraction of visited states where
argmax(Q_e + beta*Q_i) differs from argmax(Q_e) -- how often the count bonus changes
the greedy *rule*. `intrinsic_action_override` is the subset of those where epsilon did
not fire, so the bonus changed the action actually *taken*.

Both answer something the exploration share beta*Q_i / (Q_e + beta*Q_i) cannot: that
ratio compares the two heads' levels, while behaviour turns on their spread across
actions. A share of 1% is compatible with the argmax flipping constantly, and a share
near 1 with it never flipping.

    Pin --epsilon before comparing `override` across cells. In expectation
    override = (1 - epsilon) * divergence, so an unpinned epsilon axis makes that
    metric report the exploration schedule rather than the intrinsic head.
    `divergence` is epsilon-free and needs no such care.

For each ranking metric it writes, per game, one figure per selection
(top10 / best_per_beta) x per metric in RANKED_METRICS, plus per-beta head comparisons
for each best-per-beta configuration:

    <score>/summary.csv                         top 10 + best-per-beta, all games
    <score>/<game>/top10_<metric>.png
    <score>/<game>/best_per_beta_<metric>.png
    <score>/<game>/beta_<b>_values.png          Q_e, Q_i and beta*Q_i together
    <score>/<game>/beta_<b>_losses.png          both TD losses together
    <score>/<game>/beta_<b>_exploration.png     divergence and override together

There is deliberately no exploration-share figure (beta*Q_i / (Q_e + beta*Q_i)). That is a
ratio of *levels*, and is only a meaningful share when Q_e > 0 -- but Q_e goes negative on
real games here (fishing_derby reaches -5.4, double_dunk hovers near 0), where the "share"
inverts or exceeds 1. `divergence` answers the same question by comparing argmaxes rather
than levels, and is well-defined on every game.

The per-beta figures hold beta fixed and overlay related series, which is the readable
view for the Q values and losses: their magnitudes move with beta, so putting six
betas on one such axis compares quantities of different scale. divergence and override
are exempt from that problem -- both are probabilities -- but they share a figure
because the gap between them *is* epsilon, and that is the thing worth seeing.

Scores are not comparable with the other sweeps' scripts
-------------------------------------------------------
This script scores on the final 2% of a 200-update rolling mean;
plot_atari4_seperate_heads_gamma_sweep.py and plot_atari57_count_layer_sweep.py use the
final 10% of a 750-update one. `--smooth 750 --final-frac 0.1` restores the old behaviour
if a like-for-like comparison with those sweeps is needed.

Caveats, all of which matter more here than in the 4-game sweep
--------------------------------------------------------------
Combinations are always ranked by the *extrinsic* metric; every other figure shows
that same selection, so one set of runs is described by the whole family.

Both selections rank on the same seed means the confidence bands are computed from, so
the winners are optimistically biased and their bands under-cover. With 33
combinations and 5 seeds per game that bias is not small; treat the ranking as a
shortlist, not a measurement.

best_per_beta picks the best (gamma_I, epsilon) at each beta, so both vary underneath
the beta axis unless --gamma-i and --epsilon pin them. Any narrowing is reported in
every subtitle so a filtered figure cannot be mistaken for the whole sweep.

`divergence` and `override` are identically 0 in the beta=0 arm by construction, since
the fused decision variable is then literally Q_e. Those series are a null line, not a
measurement.

While the sweep is still running some cells have fewer than five finished seeds. Those
are excluded from ranking by default (--min-seeds) and listed in incomplete.csv,
because a two-seed cell can otherwise take a top-10 slot on luck.
"""

from __future__ import annotations

import csv
import dataclasses
import math
import re
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import tyro

from plot_count_layer_sweep import (
    Combo,
    Layout,
    Metric,
    load_combos,
    plot_combo_metrics,
    plot_curves,
    score_combo,
)

# Every series this sweep can draw. Passed explicitly to load_combos rather than
# added to the shared DEFAULT_METRICS: that function records a run as unreadable if
# any requested key is missing, so asking the older sweeps for
# intrinsic_action_divergence would void every one of them.
#
# "extrinsic" must stay in this dict under that exact name -- score_combo reads it by
# hardcoded key, so ranking breaks silently if it is renamed.
METRICS = {
    "extrinsic": Metric("extrinsic_return_per_game_ema", "Extrinsic return (EMA)"),
    "intrinsic": Metric("intrinsic_return_per_game_ema", "Intrinsic return (EMA)"),
    # Per-step bonus magnitude, as against the episode sum above. The UCB bonus decays
    # as states are revisited, which only this one shows.
    "intrinsic_reward": Metric("intrinsic_reward_mean", "Mean per-step count bonus"),
    "divergence": Metric(
        "intrinsic_action_divergence", "P(intrinsic changes greedy action)"
    ),
    "override": Metric(
        "intrinsic_action_override", "P(intrinsic changes taken action)"
    ),
    "q": Metric("q_values", "Mean selected $Q_e$"),
    "intrinsic_q": Metric("intrinsic_q_values", "Mean selected $Q_i$"),
    "loss_q": Metric("loss_q", "Extrinsic TD loss", log=True),
    "loss_intrinsic_q": Metric("loss_intrinsic_q", "Intrinsic TD loss", log=True),
}

# The subset that gets the top-k / best-per-beta treatment. The Q values and losses are
# left out because they are only readable with beta held fixed, which is what the
# per-beta figures do.
RANKED_METRICS = (
    "extrinsic",
    "intrinsic",
    "intrinsic_reward",
    "divergence",
    "override",
)

SARSA_LAYOUT = Layout(
    glob="*/beta_*/intrinsic_gamma_*/epsilon_*/seed_*/metrics.npz",
    # No shared-prefix hazard here, unlike the two-gamma sweep: gamma_E is pinned and
    # absent from the path, so intrinsic_gamma_ is the only gamma-ish level.
    pattern=re.compile(
        r"(?P<game>[^/]+)/beta_(?P<beta>[^/]+)/intrinsic_gamma_(?P<gamma_i>[^/]+)"
        r"/epsilon_(?P<epsilon>[^/]+)/seed_(?P<seed>[^/]+)$"
    ),
    dims=("gamma_i", "epsilon"),
)


@dataclasses.dataclass(frozen=True)
class Args:
    root_dir: Path = Path("data/atari57_seperate_heads_sarsa_sweep")
    """Sweep dir containing <game>/beta_*/intrinsic_gamma_*/epsilon_*/seed_*/."""
    output_dir: Path = Path("graphs/atari57_sarsa_figures/individual")
    """Written to <output_dir>/<score>/ -- summary.csv plus one dir per game."""
    scores: tuple[Literal["final", "auc"], ...] = ("final", "auc")
    """Ranking metrics; each gets its own output tree. One score halves the figures."""
    games: tuple[str, ...] = ()
    """Restrict to these games. Empty means all 57."""
    gamma_i: tuple[str, ...] = ()
    """Restrict to these intrinsic discounts, as they appear in the path. Empty means
    all. Pin this and --epsilon to compare betas without the best-per-beta selection
    silently varying them underneath."""
    epsilon: tuple[str, ...] = ()
    """Restrict to these exploration floors. Empty means all. Pin this before reading
    `override` across cells -- it scales with (1 - epsilon)."""
    top_k: int = 10
    """How many combinations in the top-N figures and rows. Above 10 the palette
    repeats hues."""
    min_seeds: int = 5
    """Combinations with fewer finished seeds are excluded and reported instead."""
    final_frac: float = 0.02
    """Fraction of the run averaged for the "final" score.

    2%, not the 10% the other sweeps' scripts use. Most runs are still improving at 100M,
    so a 10% tail under-reports final performance by ~1% (median over 726 cells). The raw
    endpoint was rejected instead of a narrower window: it moves the best beta on 7 of 23
    games, against 4 at 2%.

    Note the interaction with --smooth: the tail is ceil(num_updates * final_frac) = 489
    updates at 2%, so a smoothing window wider than that would decide the score instead of
    this setting. That is why --smooth dropped to 200 alongside."""
    smooth: int = 200
    """Rolling-mean window in updates. 1 disables smoothing.

    200, not the 750 the other sweeps' scripts use. 750 was doing nearly all the smoothing
    and far more than intended: the in-training EMA is over *episodes* (ema_alpha = 2/31,
    span 30) while this is over *updates*, and 2-8.5 episodes finish per update -- so 750
    updates spans 1,500-6,400 episodes, 50-200x the EMA. At w=200, 93-99% of the
    point-to-point variation is already gone (freeway 0.070 of raw, alien 0.013); 750 gets
    to 97-99.8% for a 3.75x wider window.

    Long-episode games get rougher curves under this: double_dunk finishes 0.42 episodes
    per update, so its raw series is the noisiest relative to any window."""
    band_max_series: int = 10
    """Cap above which CI bands are dropped."""
    ci: Literal["normal", "t"] = "normal"
    """Band multiplier on the SEM. "normal" is 1.96; "t" is the Student-t value that
    actually attains 95% coverage at these seed counts (2.776 at n=5)."""
    theme: Literal["light", "dark"] = "light"
    dpi: int = 300
    figsize: tuple[float, float] = (10.0, 6.0)


def series_label(combo: Combo) -> str:
    """beta=0 is the control arm, which also zeroes intrinsic_loss_coef, so its
    gamma_I is pinned and worth flagging rather than reading as a swept value."""
    gamma_i = combo.params["gamma_i"]
    epsilon = combo.params["epsilon"]
    if float(combo.beta) == 0.0:
        return f"β={combo.beta}  ε={epsilon}  (control)"
    return f"β={combo.beta}  γI={gamma_i}  ε={epsilon}"


def write_rows(rows: list[dict], fieldnames: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


SUMMARY_FIELDS = (
    "game",
    "selection",
    "rank",
    "beta",
    "gamma_i",
    "epsilon",
    "n_seeds",
    "score",
    "final_extrinsic",
    "final_intrinsic",
    "final_divergence",
    "final_override",
)


def main(args: Args) -> None:
    combos, failures = load_combos(args.root_dir, SARSA_LAYOUT, METRICS)
    if not combos:
        raise SystemExit(f"no runs found under {args.root_dir}")
    if args.games:
        wanted = set(args.games)
        combos = [c for c in combos if c.game in wanted]
        if not combos:
            raise SystemExit(f"no runs for games {sorted(wanted)}")

    # Narrowing these makes best_per_beta a clean beta comparison: with one (gamma_I,
    # epsilon) pair exactly one config survives per beta, so nothing is selected by
    # score and neither dimension can vary underneath the beta axis.
    for field in ("gamma_i", "epsilon"):
        wanted = set(getattr(args, field))
        if wanted:
            combos = [c for c in combos if c.params[field] in wanted]
            if not combos:
                raise SystemExit(
                    f"no runs for --{field.replace('_', '-')} {sorted(wanted)}"
                )

    # Carried into every subtitle so a narrowed figure is never mistaken for the whole
    # sweep.
    filter_note = ""
    if args.gamma_i:
        filter_note += f" · γI∈{{{','.join(args.gamma_i)}}}"
    if args.epsilon:
        filter_note += f" · ε∈{{{','.join(args.epsilon)}}}"

    complete = [c for c in combos if c.n_seeds >= args.min_seeds]
    incomplete = [c for c in combos if c.n_seeds < args.min_seeds]
    games = sorted({c.game for c in complete})
    print(
        f"{len(combos)} combinations over {len({c.game for c in combos})} games; "
        f"{len(complete)} complete (>= {args.min_seeds} seeds), {len(incomplete)} not"
    )

    if failures:
        write_rows(
            failures,
            ("game", "beta", "gamma_i", "epsilon", "seed", "error"),
            args.output_dir / "unreadable_runs.csv",
        )
        print(f"  {len(failures)} unreadable runs -> unreadable_runs.csv")

    if incomplete:
        write_rows(
            [
                {
                    "game": c.game,
                    "beta": c.beta,
                    "gamma_i": c.params["gamma_i"],
                    "epsilon": c.params["epsilon"],
                    "n_seeds": c.n_seeds,
                }
                for c in sorted(
                    incomplete,
                    key=lambda c: (
                        c.game,
                        float(c.beta),
                        float(c.params["gamma_i"]),
                        float(c.params["epsilon"]),
                    ),
                )
            ],
            ("game", "beta", "gamma_i", "epsilon", "n_seeds"),
            args.output_dir / "incomplete.csv",
        )
        print(f"  {len(incomplete)} incomplete combinations -> incomplete.csv")

    if not games:
        raise SystemExit(
            f"every combination has fewer than {args.min_seeds} seeds; "
            "lower --min-seeds to plot the sweep as it stands"
        )

    # Say up front how much is about to be written: full per-game trees over 57 games
    # run to thousands of files, and --scores / --games are the levers.
    n_betas = len({c.beta for c in complete})
    per_game = 2 * len(RANKED_METRICS) + 3 * n_betas
    print(
        f"  writing up to {per_game} figures per game x {len(games)} games "
        f"x {len(args.scores)} score(s) = ~{per_game * len(games) * len(args.scores)} "
        "files"
    )

    def tail_mean(combo: Combo, metric: str) -> float:
        mean, _ = combo.aggregated(metric, args.smooth)
        tail = math.ceil(mean.shape[0] * args.final_frac)
        return float(np.nanmean(mean[-tail:]))

    for score in args.scores:
        score_label = {"final": "final performance", "auc": "area under the curve"}[
            score
        ]
        out = args.output_dir / score
        rows: list[dict] = []

        def row(combo: Combo, selection: str, rank: object) -> dict:
            return {
                "game": combo.game,
                "selection": selection,
                "rank": rank,
                "beta": combo.beta,
                "gamma_i": combo.params["gamma_i"],
                "epsilon": combo.params["epsilon"],
                "n_seeds": combo.n_seeds,
                "score": round(
                    score_combo(combo, score, args.final_frac, args.smooth), 3
                ),
                "final_extrinsic": round(tail_mean(combo, "extrinsic"), 3),
                "final_intrinsic": round(tail_mean(combo, "intrinsic"), 3),
                "final_divergence": round(tail_mean(combo, "divergence"), 4),
                "final_override": round(tail_mean(combo, "override"), 4),
            }

        for game in games:
            ranked = sorted(
                (c for c in complete if c.game == game),
                key=lambda c: score_combo(c, score, args.final_frac, args.smooth),
                reverse=True,
            )
            top = ranked[: args.top_k]
            # Best combination at each beta: `ranked` is already sorted, so the first
            # match for a beta is its best. Ordered by beta, not by rank.
            best_per_beta = [
                next(c for c in ranked if c.beta == beta)
                for beta in sorted({c.beta for c in ranked}, key=float)
            ]

            # The seed count is deliberately absent from the subtitles: cells can hold
            # more seeds than --min-seeds, and the per-series (n=...) in the legend
            # already reports each one correctly.
            band = (
                f"seed mean with 95% CI{'' if args.ci == 't' else ' (normal)'}"
                f"{filter_note}"
            )

            for metric in RANKED_METRICS:
                spec = METRICS[metric]
                plot_curves(
                    top,
                    metric=metric,
                    title=f"{game} — top {len(top)} combinations by {score_label}",
                    subtitle=f"{spec.label}, {band}",
                    path=out / game / f"top{args.top_k}_{metric}.png",
                    args=args,
                    label_fn=series_label,
                    metrics=METRICS,
                )
                plot_curves(
                    best_per_beta,
                    metric=metric,
                    title=f"{game} — best combination per β by {score_label}",
                    subtitle=f"{spec.label}, {band} · series ordered by β",
                    path=out / game / f"best_per_beta_{metric}.png",
                    args=args,
                    label_fn=series_label,
                    metrics=METRICS,
                )

            # Per-beta comparisons. The figures above hold a metric fixed and vary
            # beta; these hold beta fixed and put related series together, which is the
            # readable view for the Q values and losses because their magnitudes move
            # with beta.
            for combo in best_per_beta:
                beta = float(combo.beta)
                # beta=0 also sets intrinsic_loss_coef=0 in this sweep, so the
                # intrinsic head never trains -- say so rather than let a flat line at
                # its initialisation read as a result.
                note = " · intrinsic head untrained at β=0" if beta == 0.0 else ""
                cell = f"γI={combo.params['gamma_i']} ε={combo.params['epsilon']}"
                plot_combo_metrics(
                    combo,
                    series=[
                        ("q", 1.0, "$Q_e$"),
                        ("intrinsic_q", 1.0, "$Q_i$"),
                        ("intrinsic_q", beta, f"β·$Q_i$  (β={combo.beta})"),
                    ],
                    title=f"{game} — value heads at β={combo.beta}",
                    subtitle=f"{cell}, {band}{note}",
                    ylabel="Mean selected Q",
                    path=out / game / f"beta_{combo.beta}_values.png",
                    args=args,
                    metrics=METRICS,
                )
                plot_combo_metrics(
                    combo,
                    series=[
                        ("loss_q", 1.0, "Extrinsic TD loss"),
                        ("loss_intrinsic_q", 1.0, "Intrinsic TD loss"),
                    ],
                    title=f"{game} — TD losses at β={combo.beta}",
                    subtitle=f"{cell}, {band}{note}",
                    ylabel="TD loss",
                    path=out / game / f"beta_{combo.beta}_losses.png",
                    args=args,
                    metrics=METRICS,
                    log=True,
                )
                # Both series are probabilities, so they share an axis honestly, and
                # the gap between them is epsilon: divergence counts states where the
                # greedy rule would change, override only those where it reached the
                # executed action. At β=0 both are identically 0 by construction.
                null = (
                    " · both identically 0 at β=0 by construction"
                    if beta == 0.0
                    else ""
                )
                plot_combo_metrics(
                    combo,
                    series=[
                        ("divergence", 1.0, "greedy action changed"),
                        ("override", 1.0, "taken action changed"),
                    ],
                    title=f"{game} — intrinsic influence on actions at β={combo.beta}",
                    subtitle=f"{cell}, {band} · gap between them is ε{null}",
                    ylabel="Fraction of visited states",
                    path=out / game / f"beta_{combo.beta}_exploration.png",
                    args=args,
                    metrics=METRICS,
                )

            rows += [row(c, f"top{args.top_k}", i + 1) for i, c in enumerate(top)]
            rows += [row(c, "best_per_beta", "") for c in best_per_beta]

        write_rows(rows, SUMMARY_FIELDS, out / "summary.csv")
        print(
            f"  {score}: {len(rows)} rows, "
            f"{len(list(out.glob('*/*.png')))} figures -> {out}"
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
