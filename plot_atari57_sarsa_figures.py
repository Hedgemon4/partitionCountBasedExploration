"""Cross-game figures for the Atari-57 separate-value-head SARSA sweep.

Self-contained by design: this imports nothing from plot_atari57_sarsa_summary.py, which
it is intended to replace and which can be deleted once its figures are no longer wanted.
The only imports are the sweep's layout and metric set from the per-game script, and the
shared plotting machinery -- neither of which is going away.

Layout it reads:

    <root>/<game>/beta_<b>/intrinsic_gamma_<gI>/epsilon_<e>/seed_<s>/metrics.npz

Figures, per <score>/<selection> tree
-------------------------------------
    per_game_grid.png                    raw score against beta, one panel per game
    curve_grid_<metric>.png              learning curves, one panel per game, one line per beta
    improvement_pct_beta_<b>.png         100*(beta - control)/|control|, one bar per game
    improvement_pct_best_beta.png        each game's best beta, bars coloured by which won
    improvement_hns_beta_<b>.png         human-normalised improvement, in percentage points
    improvement_hns_best_beta.png

Config selection
----------------
Which (gamma_I, epsilon) cell represents a beta is a real choice, and both answers are
produced as parallel trees:

    global      one config per beta, chosen once across all games. "Which configuration
                would I ship." One choice amortised over every game, so the selection
                barely biases anything.
    per_game    per game, per beta, that game's best cell. "What is beta worth once tuned
                per game." Optimistically biased, and asymmetrically so.

`global` uses a **mean rank**, not a ratio: within each game the available configs are
ranked 1..k by score, and the config with the lowest mean rank across games wins. Ranks
are bounded, so no game can dominate the average and no median is needed on top -- a
median only coarsens the input, and ties constantly at these sizes. Comparison is therefore only ever *within* a game, so nothing is normalised and no
game is excluded from voting. The alternative -- ranking on score/control -- has to drop
every game whose control is negative or zero, which here means montezuma_revenge, pitfall,
skiing, double_dunk and tennis: precisely the games a count bonus is aimed at.

Two consequences of that rule, both recorded per row in config_selection.csv:

  * Ballots differ in length. A game missing a config still votes on the ones it has, so a
    config appearing mainly in games with few candidates gets systematically better ranks.
    `n_ballots` and `n_candidates` make a winner from a thin field visible.
  * Cells vote with however many seeds finished. A 2-seed cell sits beside a 5-seed one,
    so a thin cell can win a game's rank on luck. Defensible while the sweep is partial --
    the alternative discards most of it -- but `n_seeds` is recorded, and this is worth
    revisiting once every cell has 5.

Two improvement measures, and why both
--------------------------------------
Neither works everywhere, and they fail on *different* games:

  pct = 100*(beta - control)/|control|
        Dividing by |control| rather than control is what keeps the sign honest: the
        numerator already carries the direction, and a negative denominator would invert
        it. Well-defined for negative and mixed-sign scores; undefined only at
        control == 0, and explosive when |control| is small (montezuma at +2,155%).

  hns = 100*(score - random)/(human - random),  reported as hns_beta - hns_control
        The denominator is a positive range by construction, so zero and negative scores
        are fine. But it amplifies where human-random is narrow -- double_dunk's range is
        3.1, so it reads +357pp where pct reads a sane +170% -- and atari_hns.csv covers
        only 49 of the 57 games, missing berzerk, defender, phoenix, pitfall, skiing,
        solaris, surround and yars_revenge.

So pct rescues pitfall and skiing, hns rescues montezuma, and between them all 57 games
are covered. Games a measure cannot express are named in that figure's subtitle and given
a reason in improvement.csv -- never silently dropped.

Both families use a symlog y-axis: the range runs to several hundred percent while the
interesting region is a few tens, and a linear axis lets one bar flatten the rest.

Game ordering
-------------
One order shared by every improvement figure, taken from the `per_game` best-beta pct
improvement, ascending as in the reference figure. pct rather than hns because it is
defined for all 57 games; the hns figures draw their 49 in the same relative order, so x
positions differ between families but the ordering does not.

That order is *doubly* selected -- best config per game, then best beta per game -- so a
game's position near the right edge is partly selection noise. It affects position only;
bar heights are unaffected.

Caveats
-------
`best_beta` maximises over 5 betas per game, so under a null its median improvement would
still be *positive*. Read it as "what per-game tuning could buy", not as an effect size.

Per-bar and per-band intervals are percentile bootstraps over that cell's seeds. At 5
seeds a percentile bootstrap is about 38% narrower than the Student-t interval and
under-covers; they are labelled as bootstraps rather than as 95% CIs for that reason.
"""

from __future__ import annotations

import csv
import dataclasses
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Literal, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro
from matplotlib.ticker import FuncFormatter

from plot_atari57_seperate_heads_sarsa_sweep import METRICS, SARSA_LAYOUT
from plot_count_layer_sweep import (
    SERIES_COLOURS,
    THEMES,
    Combo,
    load_combos,
    rolling_mean,
)

CONTROL_BETA = "0.0"
STEP_KEY = "env_step"

# Only these three are loaded. Each metric is ~918 MB of float32 at 57 games x 5 seeds,
# so asking for the Q values and losses as well would triple a figure that is already
# ~2.75 GB resident.
FIGURE_METRICS = ("extrinsic", "divergence", "override")
SUMMARY_METRICS = {name: METRICS[name] for name in FIGURE_METRICS}


# ----------------------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------------------


def _score_rows(
    smoothed: np.ndarray, steps: np.ndarray, score: str, final_frac: float
) -> np.ndarray:
    """Score each already-smoothed seed row, returning (n_seeds,)."""
    out = np.full(smoothed.shape[0], np.nan)
    for i, row in enumerate(smoothed):
        finite = np.isfinite(row)
        if not finite.any():
            continue
        if score == "final":
            tail = math.ceil(row.shape[0] * final_frac)
            out[i] = np.nanmean(row[-tail:])
        else:
            x, y = steps[finite], row[finite]
            out[i] = y.mean() if x[-1] == x[0] else np.trapezoid(y, x) / (x[-1] - x[0])
    return out


def seed_scores(combo: Combo, final_frac: float, window: int) -> dict[str, np.ndarray]:
    """Per-seed "final" and "auc" scores.

    score_combo() in plot_count_layer_sweep scores the *seed-mean* curve, which leaves no
    per-seed spread to bootstrap over. This is its body applied row-wise instead, so the
    mean of what it returns matches score_combo for "final".

    Both modes come out of one smoothing pass: rolling_mean is the expensive part of this
    script, and computing them separately would double it.
    """
    curves = np.asarray(combo.curves("extrinsic"), dtype=np.float64)
    smoothed = np.stack([rolling_mean(row, window) for row in curves])
    return {
        "final": _score_rows(smoothed, combo.steps, "final", final_frac),
        "auc": _score_rows(smoothed, combo.steps, "auc", final_frac),
    }


@dataclasses.dataclass
class Cell:
    """One (game, beta, gamma_I, epsilon) cell and its per-seed scores."""

    game: str
    beta: str
    gamma_i: str
    epsilon: str
    combo: Combo
    scores: dict[str, np.ndarray]

    @property
    def config(self) -> tuple[str, str]:
        return (self.gamma_i, self.epsilon)

    @property
    def n_seeds(self) -> int:
        return int(np.isfinite(next(iter(self.scores.values()))).sum())

    def mean(self, score: str) -> float:
        values = self.scores[score]
        return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")

    def label(self) -> str:
        return f"γI={self.gamma_i} ε={self.epsilon}"


def build_cells(
    root: Path, args: Args
) -> tuple[dict[tuple[str, str], dict[tuple[str, str], Cell]], list[dict]]:
    """Load the sweep into cells keyed by (game, beta) -> (gamma_I, epsilon) -> Cell."""
    combos, failures = load_combos(root, SARSA_LAYOUT, SUMMARY_METRICS)
    cells: dict[tuple[str, str], dict[tuple[str, str], Cell]] = defaultdict(dict)
    for combo in combos:
        gamma_i, epsilon = combo.params["gamma_i"], combo.params["epsilon"]
        cells[(combo.game, combo.beta)][(gamma_i, epsilon)] = Cell(
            game=combo.game,
            beta=combo.beta,
            gamma_i=gamma_i,
            epsilon=epsilon,
            combo=combo,
            scores=seed_scores(combo, args.final_frac, args.smooth),
        )
    return cells, failures


# ----------------------------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------------------------


@dataclasses.dataclass
class Selection:
    """One selection rule's result: a cell per (game, beta), plus its audit trail."""

    chosen: dict[tuple[str, str], Cell]
    audit: list[dict]
    fields: Sequence[str]
    note: str


def select_global(
    cells: dict[tuple[str, str], dict[tuple[str, str], Cell]],
    games: Sequence[str],
    betas: Sequence[str],
    score: str,
) -> Selection:
    """One (gamma_I, epsilon) per beta, by lowest mean rank across games.

    Within a game the configs present are ranked 1..k on score, ties averaged. Ranking is
    ordinal and within-game, so no normalisation is involved and every game votes -- unlike
    a score/control rule, which cannot use a game whose control is negative or zero.

    The mean, not the median. A median would be the right choice for an unbounded quantity
    -- it is what a score/anchor rule needs, so one game running to 22x cannot appoint the
    config for all 57 -- but ranks are bounded at 1..k by construction, so no single game
    can drag the mean. Ranking already supplies the robustness, and a median on top of it
    only discards information: it cannot tell a config placing 6th from one placing 3rd.
    It also ties constantly at these sizes; median rank produced a three-way tie at beta=0.1
    over 23 games, which then had to be broken on the mean anyway.
    """
    chosen: dict[tuple[str, str], Cell] = {}
    audit: list[dict] = []
    for beta in betas:
        ranks: dict[tuple[str, str], list[float]] = defaultdict(list)
        candidates: dict[tuple[str, str], list[int]] = defaultdict(list)
        for game in games:
            available = cells.get((game, beta), {})
            scored = [(c, cell.mean(score)) for c, cell in available.items()]
            scored = [(c, v) for c, v in scored if np.isfinite(v)]
            if not scored:
                continue
            # rank 1 = best; ties share the average rank
            order = sorted(scored, key=lambda cv: -cv[1])
            values = [v for _, v in order]
            for position, (config, value) in enumerate(order):
                tied = [i for i, v in enumerate(values) if v == value]
                ranks[config].append(1.0 + sum(tied) / len(tied))
                candidates[config].append(len(order))
        if not ranks:
            continue
        summary = {c: float(np.mean(rs)) for c, rs in ranks.items()}
        medians = {c: float(np.median(rs)) for c, rs in ranks.items()}

        # Ballot count breaks the (vanishingly unlikely) exact tie, preferring the config
        # more games actually voted on.
        def key(config: tuple[str, str]) -> tuple[float, int]:
            return (summary[config], -len(ranks[config]))

        winner = min(summary, key=key)
        for config in sorted(summary, key=key):
            audit.append(
                {
                    "beta": beta,
                    "gamma_i": config[0],
                    "epsilon": config[1],
                    "mean_rank": round(summary[config], 3),
                    "median_rank": round(medians[config], 3),
                    "n_ballots": len(ranks[config]),
                    "mean_n_candidates": round(float(np.mean(candidates[config])), 2),
                    "chosen": int(config == winner),
                }
            )
        for game in games:
            cell = cells.get((game, beta), {}).get(winner)
            if cell is not None:
                chosen[(game, beta)] = cell
    return Selection(
        chosen=chosen,
        audit=audit,
        fields=(
            "beta",
            "gamma_i",
            "epsilon",
            "mean_rank",
            "median_rank",
            "n_ballots",
            "mean_n_candidates",
            "chosen",
        ),
        note="one config per β, lowest mean rank across games",
    )


def select_per_game(
    cells: dict[tuple[str, str], dict[tuple[str, str], Cell]],
    games: Sequence[str],
    betas: Sequence[str],
    score: str,
) -> Selection:
    """Per game, per beta, that game's best-scoring cell."""
    chosen: dict[tuple[str, str], Cell] = {}
    audit: list[dict] = []
    for game in games:
        for beta in betas:
            available = cells.get((game, beta), {})
            scored = [
                (c, cell) for c, cell in available.items() if np.isfinite(cell.mean(score))
            ]
            if not scored:
                continue
            config, cell = max(scored, key=lambda cc: cc[1].mean(score))
            chosen[(game, beta)] = cell
            audit.append(
                {
                    "game": game,
                    "beta": beta,
                    "gamma_i": config[0],
                    "epsilon": config[1],
                    "n_candidates": len(available),
                    "n_seeds": cell.n_seeds,
                    "score": round(cell.mean(score), 3),
                }
            )
    return Selection(
        chosen=chosen,
        audit=audit,
        fields=("game", "beta", "gamma_i", "epsilon", "n_candidates", "n_seeds", "score"),
        note="per game, that game's best config at each β",
    )


SELECTORS: dict[str, Callable[..., Selection]] = {
    "global": select_global,
    "per_game": select_per_game,
}


# ----------------------------------------------------------------------------------------
# Improvement measures
# ----------------------------------------------------------------------------------------


def load_hns(path: Path) -> dict[str, tuple[float, float]]:
    """game -> (random, human) from atari_hns.csv. 49 of the 57 games are present."""
    out: dict[str, tuple[float, float]] = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            random_, human = float(row["random"]), float(row["human"])
            if human <= random_:
                raise ValueError(f"{row['game']}: human <= random, HNS undefined")
            out[row["game"]] = (random_, human)
    return out


def pct_improvement(treatment: float, control: float) -> float:
    """100*(t - c)/|c|.

    |c| and not c: the numerator already carries the direction, and dividing by a negative
    control would inverte it -- pitfall improving from -24.4 to -16.9 would read as -31%.
    Undefined at c == 0, which is the one case this cannot express.
    """
    if not np.isfinite(control) or control == 0.0 or not np.isfinite(treatment):
        return float("nan")
    return 100.0 * (treatment - control) / abs(control)


def hns_improvement(
    treatment: float, control: float, reference: tuple[float, float] | None
) -> float:
    """Difference of human-normalised scores, in percentage points."""
    if reference is None or not np.isfinite(control) or not np.isfinite(treatment):
        return float("nan")
    random_, human = reference
    span = human - random_
    return 100.0 * (treatment - control) / span


MEASURES = {
    "pct": ("100·(β−control)/|control|  [%]", "control = 0"),
    "hns": ("human-normalised improvement  [pp]", "absent from atari_hns.csv"),
}


def bootstrap_ci(
    treatment: np.ndarray,
    control: np.ndarray,
    measure: Callable[[float, float], float],
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile bootstrap over seeds, resampling treatment and control together.

    The control mean is itself estimated from 5 seeds, so holding it fixed would understate
    the interval. At these seed counts the interval under-covers regardless -- it is
    reported as a bootstrap, not as a 95% CI.
    """
    t = treatment[np.isfinite(treatment)]
    c = control[np.isfinite(control)]
    if t.size == 0 or c.size == 0:
        return (float("nan"), float("nan"))
    draws = np.empty(n_boot)
    for i in range(n_boot):
        draws[i] = measure(
            float(t[rng.integers(0, t.size, t.size)].mean()),
            float(c[rng.integers(0, c.size, c.size)].mean()),
        )
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5)))


# ----------------------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------------------


def _style_panel(ax, theme: dict, xlabel: str = "", ylabel: str = "") -> None:
    """The repo's axis styling, minus plot_count_layer_sweep._style_axes' assumptions.

    That helper hardcodes an "Environment steps" label and a 1e6 tick formatter, which suit
    neither a categorical beta axis, nor a game axis, nor a 57-panel grid where only the
    edges should carry labels.
    """
    ax.set_facecolor(theme["surface"])
    ax.grid(True, color=theme["grid"], linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=theme["muted"], labelsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=theme["ink_secondary"], fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=theme["ink_secondary"], fontsize=10)


def _titles(fig, theme: dict, title: str, subtitle: str) -> float:
    """Place the heading block and return the tight_layout rect top it needs.

    Positioned in *inches* from the top rather than figure fractions: these figures range
    from 5 inches tall to over 20, and a fixed fraction that clears the title on a tall
    grid overlaps it on a short bar chart. Returns the fraction below which axes may go.
    """
    height = fig.get_size_inches()[1]
    lines = subtitle.count("\n") + 1
    fig.suptitle(
        title, color=theme["ink"], fontsize=14, x=0.01, ha="left", y=1 - 0.26 / height
    )
    fig.text(
        0.01, 1 - 0.50 / height, subtitle, color=theme["muted"], fontsize=9,
        ha="left", va="top",
    )
    return 1 - (0.54 + 0.17 * lines) / height


def _save(fig, path: Path, args: Args, theme: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor=theme["surface"])
    plt.close(fig)


def write_rows(rows: list[dict], fieldnames: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def beta_colours(betas: Sequence[str]) -> dict[str, str]:
    """beta -> colour, fixed by position so one legend is valid across every panel.

    plot_curves colours by *rank*, which would make the same colour mean a different beta
    in each panel of a grid.
    """
    return {b: SERIES_COLOURS[i % len(SERIES_COLOURS)] for i, b in enumerate(betas)}


def _grid_shape(n: int, cols: int) -> tuple[int, int]:
    """Never stretch a handful of games across the full width."""
    cols = max(1, min(cols, n))
    return math.ceil(n / cols), cols


# ----------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------


def plot_per_game_grid(
    panels: list[dict],
    *,
    betas: Sequence[str],
    title: str,
    subtitle: str,
    path: Path,
    args: Args,
) -> None:
    """One panel per game: raw score against beta, each on its own y-axis.

    Deliberately un-normalised. Every other figure here divides or subtracts something, so
    if a normalisation is doing something strange this is the figure that shows it -- the
    only assumption it makes is that a game's own scores are comparable to each other.
    """
    theme = THEMES[args.theme]
    rows, cols = _grid_shape(len(panels), args.grid_cols)
    colours = beta_colours(betas)
    fig, axes = plt.subplots(
        rows, cols, figsize=(2.6 * cols, 2.3 * rows + 0.75), dpi=args.dpi, squeeze=False
    )
    fig.patch.set_facecolor(theme["surface"])
    x = np.arange(len(betas))
    for index, ax in enumerate(axes.flat):
        if index >= len(panels):
            ax.axis("off")
            continue
        panel = panels[index]
        means = np.array([panel["mean"].get(b, np.nan) for b in betas])
        errs = np.array([panel["err"].get(b, np.nan) for b in betas])
        if np.isfinite(panel["control"]):
            ax.axhline(
                panel["control"], color=theme["ink_secondary"], linewidth=0.9,
                linestyle="--", zorder=1,
            )
        for i, b in enumerate(betas):
            if not np.isfinite(means[i]):
                continue
            ax.errorbar(
                x[i], means[i], yerr=(0.0 if not np.isfinite(errs[i]) else errs[i]),
                fmt="o", markersize=4, color=colours[b],
                ecolor=theme["ink_secondary"], elinewidth=1.0, capsize=2, zorder=3,
            )
        _style_panel(ax, theme)
        ax.set_title(panel["game"], color=theme["ink"], fontsize=9, loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels(betas, fontsize=7, rotation=45, ha="right")
        if index // cols != rows - 1:
            ax.set_xticklabels([])
    top = _titles(fig, theme, title, subtitle)
    fig.tight_layout(rect=(0, 0, 1, top))
    _save(fig, path, args, theme)


def plot_curve_grid(
    panels: list[dict],
    *,
    metric: str,
    betas: Sequence[str],
    title: str,
    subtitle: str,
    path: Path,
    args: Args,
    rng: np.random.Generator,
) -> None:
    """One panel per game: the metric against env steps, one line per beta.

    Bands are a percentile bootstrap over that cell's seeds, evaluated at --curve-points
    positions with one set of draws reused across them: drawing fresh seeds at every x
    would make neighbouring points independent and the band jagged rather than a band
    around a curve.
    """
    theme = THEMES[args.theme]
    rows, cols = _grid_shape(len(panels), args.grid_cols)
    colours = beta_colours(betas)
    spec = METRICS[metric]
    fig, axes = plt.subplots(
        rows, cols, figsize=(2.9 * cols, 2.3 * rows + 0.75), dpi=args.dpi, squeeze=False
    )
    fig.patch.set_facecolor(theme["surface"])
    million = FuncFormatter(lambda v, _: f"{v / 1e6:.0f}M")

    for index, ax in enumerate(axes.flat):
        if index >= len(panels):
            ax.axis("off")
            continue
        panel = panels[index]
        lowest = np.inf
        for beta in betas:
            cell = panel["cells"].get(beta)
            if cell is None:
                continue
            curves = np.asarray(cell.combo.curves(metric), dtype=np.float64)
            smoothed = np.stack([rolling_mean(row, args.smooth) for row in curves])
            steps = cell.combo.steps
            positions = np.unique(
                np.linspace(0, steps.shape[0] - 1, args.curve_points).astype(int)
            )
            sub = smoothed[:, positions]
            mean = np.nanmean(sub, axis=0)
            finite = np.isfinite(mean)
            if not finite.any():
                continue
            n = sub.shape[0]
            draws = rng.integers(0, n, size=(args.curve_boot, n))
            boots = np.nanmean(sub[draws], axis=1)
            # A position where every seed is NaN yields an all-NaN bootstrap column;
            # percentile would warn and return NaN. Mask those out explicitly.
            usable = np.isfinite(boots).any(axis=0)
            lo = np.full(boots.shape[1], np.nan)
            hi = np.full(boots.shape[1], np.nan)
            if usable.any():
                lo[usable] = np.nanpercentile(boots[:, usable], 2.5, axis=0)
                hi[usable] = np.nanpercentile(boots[:, usable], 97.5, axis=0)
            finite = finite & usable
            xs = steps[positions][finite]
            ax.fill_between(
                xs, lo[finite], hi[finite], color=colours[beta], alpha=0.15,
                linewidth=0, zorder=2,
            )
            ax.plot(xs, mean[finite], color=colours[beta], linewidth=1.2, zorder=3)
            lowest = min(lowest, float(np.nanmin(lo[finite])))
        _style_panel(ax, theme)
        ax.set_title(panel["game"], color=theme["ink"], fontsize=9, loc="left")
        ax.xaxis.set_major_formatter(million)
        # Anchor at zero only when nothing plotted goes below it: several games score
        # negative, where a blanket floor would hide most of the curve.
        if np.isfinite(lowest) and lowest >= 0:
            ax.set_ylim(bottom=0)
        if index // cols != rows - 1:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Environment steps", color=theme["ink_secondary"], fontsize=8)
        if index % cols == 0:
            ax.set_ylabel(spec.label, color=theme["ink_secondary"], fontsize=8)

    handles = [
        plt.Line2D([], [], color=colours[b], linewidth=2.0, label=f"β={b}") for b in betas
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=len(betas), frameon=False,
        fontsize=9, labelcolor=theme["ink_secondary"], bbox_to_anchor=(0.5, -0.012),
    )
    top = _titles(fig, theme, title, subtitle)
    fig.tight_layout(rect=(0, 0.015, 1, top))
    _save(fig, path, args, theme)


def plot_improvement_bars(
    bars: list[dict],
    *,
    title: str,
    subtitle: str,
    ylabel: str,
    path: Path,
    args: Args,
    colour_by_beta: dict[str, str] | None = None,
) -> None:
    """One bar per game, sorted, in the style of the PQN-over-Rainbow figure.

    Symlog y-axis: the range runs to several hundred percent while the interesting region
    is a few tens, so a linear axis lets one bar flatten the rest. `linthresh` keeps the
    +-10 band linear so small effects stay readable.
    """
    theme = THEMES[args.theme]
    drawn = [b for b in bars if np.isfinite(b["value"])]
    fig, ax = plt.subplots(figsize=(max(8.0, 0.22 * len(drawn) + 2.0), 5.2), dpi=args.dpi)
    fig.patch.set_facecolor(theme["surface"])
    x = np.arange(len(drawn))
    values = np.array([b["value"] for b in drawn])
    colours = (
        [colour_by_beta[b["beta"]] for b in drawn]
        if colour_by_beta
        else [SERIES_COLOURS[0]] * len(drawn)
    )
    ax.axhline(0.0, color=theme["ink_secondary"], linewidth=1.0, zorder=2)
    ax.bar(x, values, width=0.72, color=colours, zorder=3, linewidth=0)
    lo = np.array([b["lo"] for b in drawn])
    hi = np.array([b["hi"] for b in drawn])
    ok = np.isfinite(lo) & np.isfinite(hi)
    if ok.any():
        ax.errorbar(
            x[ok], values[ok],
            yerr=np.vstack([values[ok] - lo[ok], hi[ok] - values[ok]]).clip(min=0),
            fmt="none", ecolor=theme["ink"], elinewidth=0.8, capsize=2, zorder=4,
        )
    ax.set_yscale("symlog", linthresh=10.0)
    _style_panel(ax, theme, xlabel="Game", ylabel=ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([b["game"] for b in drawn], rotation=90, fontsize=7)
    ax.set_xlim(-0.8, len(drawn) - 0.2)
    if colour_by_beta:
        # Only betas that actually won a game. beta=0 is the control and can never be a
        # bar, so listing it would imply a category that cannot appear.
        present = [b for b in colour_by_beta if any(bar["beta"] == b for bar in drawn)]
        handles = [
            plt.Line2D([], [], marker="s", linestyle="", color=colour_by_beta[b],
                       label=f"β={b}")
            for b in present
        ]
        ax.legend(
            handles=handles, loc="upper left", frameon=False, fontsize=8,
            labelcolor=theme["ink_secondary"], ncol=2,
        )
    top = _titles(fig, theme, title, subtitle)
    fig.tight_layout(rect=(0, 0, 1, top))
    _save(fig, path, args, theme)


# ----------------------------------------------------------------------------------------
# CLI and driver
# ----------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Args:
    root_dir: Path = Path("data/atari57_seperate_heads_sarsa_sweep_partial")
    """Sweep dir containing <game>/beta_*/intrinsic_gamma_*/epsilon_*/seed_*/."""
    output_dir: Path = Path("graphs/atari57_sarsa_figures")
    """Figures and CSVs are written to <output_dir>/<score>/<selection>/."""
    hns_csv: Path = Path("atari_hns.csv")
    """Random and human reference scores. Covers 49 of the 57 games."""
    games_file: Path = Path("atari57_games.txt")
    """Newline-separated game list, used only to report games with no data at all."""
    scores: tuple[Literal["final", "auc"], ...] = ("final", "auc")
    """Scoring rules; each gets its own output tree."""
    selection: tuple[Literal["global", "per_game"], ...] = ("global", "per_game")
    """Cell-selection rules; each gets its own tree under the score."""
    games: tuple[str, ...] = ()
    """Restrict to these games. Empty means every game on disk."""
    betas: tuple[str, ...] = ()
    """Restrict to these betas, as they appear in the path. beta=0.0 is always kept, since
    it is the control everything is compared against."""
    final_frac: float = 0.02
    """Fraction of the run averaged for the "final" score. Matches the per-game script."""
    smooth: int = 200
    """Rolling-mean window in updates. Matches the per-game script."""
    n_boot: int = 4000
    """Bootstrap resamples for the improvement bars' intervals."""
    curve_boot: int = 1000
    """Bootstrap resamples for the curve-grid bands."""
    curve_points: int = 300
    """x positions the curve bands are evaluated at. The curves are already smoothed, so
    bootstrapping all 24,414 updates buys nothing and costs a great deal of memory."""
    boot_seed: int = 0
    """Seeds the bootstrap RNG so figures are reproducible."""
    theme: Literal["light", "dark"] = "light"
    dpi: int = 300
    grid_cols: int = 8
    """Columns in the per-game panel grids."""


def coverage_rows(
    cells: dict[tuple[str, str], dict[tuple[str, str], Cell]],
    games: Sequence[str],
    betas: Sequence[str],
    expected_games: Sequence[str],
    min_seeds: int = 5,
) -> list[dict]:
    """What is missing or thin, so a figure is never read as complete when it is not."""
    rows: list[dict] = []
    for game in sorted(set(expected_games) - set(games)):
        rows.append({"kind": "absent_game", "game": game, "beta": "", "detail": "no runs"})
    for game in games:
        for beta in betas:
            available = cells.get((game, beta), {})
            if not available:
                rows.append(
                    {"kind": "absent_cell", "game": game, "beta": beta, "detail": "no runs"}
                )
                continue
            thin = [c.label() for c in available.values() if c.n_seeds < min_seeds]
            if thin:
                rows.append(
                    {
                        "kind": "thin_cell",
                        "game": game,
                        "beta": beta,
                        "detail": f"<{min_seeds} seeds: " + ", ".join(sorted(thin)),
                    }
                )
    return rows


def _best_beta_per_game(
    selection: Selection, games: Sequence[str], betas: Sequence[str], score: str
) -> dict[str, str]:
    """Per game, the beta>0 whose cell scores highest. Optimistic by construction."""
    out: dict[str, str] = {}
    for game in games:
        candidates = [
            (selection.chosen[(game, b)].mean(score), b)
            for b in betas
            if b != CONTROL_BETA and (game, b) in selection.chosen
        ]
        candidates = [(v, b) for v, b in candidates if np.isfinite(v)]
        if candidates:
            out[game] = max(candidates)[1]
    return out


def main(args: Args) -> None:
    rng = np.random.default_rng(args.boot_seed)
    hns = load_hns(args.hns_csv)
    expected = [g.strip() for g in open(args.games_file) if g.strip()]
    cells, failures = build_cells(args.root_dir, args)
    if not cells:
        raise SystemExit(f"no runs found under {args.root_dir}")

    games = sorted({g for g, _ in cells})
    if args.games:
        games = [g for g in games if g in set(args.games)]
        if not games:
            raise SystemExit(f"no runs for games {sorted(args.games)}")
    betas = sorted({b for _, b in cells}, key=float)
    if args.betas:
        keep = set(args.betas) | {CONTROL_BETA}
        betas = [b for b in betas if b in keep]
    print(f"{len(games)} games, betas {betas}, {sum(len(v) for v in cells.values())} cells")

    if failures:
        write_rows(
            failures,
            ("game", "beta", "gamma_i", "epsilon", "seed", "error"),
            args.output_dir / "unreadable_runs.csv",
        )
        print(f"  {len(failures)} unreadable runs -> unreadable_runs.csv")
    cover = coverage_rows(cells, games, betas, expected)
    if cover:
        write_rows(cover, ("kind", "game", "beta", "detail"), args.output_dir / "coverage.csv")
        print(f"  {len(cover)} coverage notes -> coverage.csv")

    for score in args.scores:
        # The shared game ordering comes from the per_game best-beta pct improvement:
        # pct because it is defined for all 57 games where hns covers only 49.
        order_sel = select_per_game(cells, games, betas, score)
        order_best = _best_beta_per_game(order_sel, games, betas, score)
        order_key: dict[str, float] = {}
        for game in games:
            beta = order_best.get(game)
            control = order_sel.chosen.get((game, CONTROL_BETA))
            cell = order_sel.chosen.get((game, beta)) if beta else None
            order_key[game] = (
                pct_improvement(cell.mean(score), control.mean(score))
                if cell is not None and control is not None
                else float("-inf")
            )
        ordered = sorted(games, key=lambda g: (np.isfinite(order_key[g]), order_key[g]))

        for mode in args.selection:
            sel = SELECTORS[mode](cells, games, betas, score)
            out = args.output_dir / score / mode
            base = f"{len(games)} games, {score} · {sel.note}"
            write_rows(sel.audit, sel.fields, out / "config_selection.csv")

            # --- per-game raw scores by beta -------------------------------------------
            panels = []
            for game in ordered:
                means, errs = {}, {}
                for beta in betas:
                    cell = sel.chosen.get((game, beta))
                    if cell is None:
                        continue
                    values = cell.scores[score]
                    means[beta] = cell.mean(score)
                    finite = values[np.isfinite(values)]
                    errs[beta] = (
                        float(finite.std(ddof=1) / math.sqrt(finite.size))
                        if finite.size > 1
                        else np.nan
                    )
                if means:
                    control = sel.chosen.get((game, CONTROL_BETA))
                    panels.append(
                        {
                            "game": game,
                            "mean": means,
                            "err": errs,
                            "control": control.mean(score) if control else np.nan,
                            "cells": {
                                b: sel.chosen[(game, b)]
                                for b in betas
                                if (game, b) in sel.chosen
                            },
                        }
                    )
            plot_per_game_grid(
                panels, betas=betas,
                title="Per-game raw scores by β",
                subtitle=f"{base}\nOwn y-axis per panel, no normalisation · dashed line is that game's β=0 control · ±1 SEM",
                path=out / "per_game_grid.png", args=args,
            )

            # --- learning-curve grids ---------------------------------------------------
            for metric in FIGURE_METRICS:
                null = (
                    " · β=0 is identically 0 by construction"
                    if metric in ("divergence", "override")
                    else ""
                )
                plot_curve_grid(
                    panels, metric=metric, betas=betas,
                    title=f"Per-game {METRICS[metric].label} by β",
                    subtitle=f"{base}\nBands are a percentile bootstrap over seeds, smoothed over {args.smooth} updates{null}",
                    path=out / f"curve_grid_{metric}.png", args=args, rng=rng,
                )

            # --- improvement bars -------------------------------------------------------
            rows: list[dict] = []
            best = _best_beta_per_game(sel, games, betas, score)
            colours = beta_colours(betas)
            for measure in ("pct", "hns"):
                ylabel, blank_reason = MEASURES[measure]
                for target in [b for b in betas if b != CONTROL_BETA] + ["best"]:
                    bars, skipped = [], []
                    for game in ordered:
                        beta = best.get(game) if target == "best" else target
                        control = sel.chosen.get((game, CONTROL_BETA))
                        cell = sel.chosen.get((game, beta)) if beta else None
                        if cell is None or control is None:
                            continue
                        ref = hns.get(game)
                        if measure == "pct":
                            fn = pct_improvement
                        else:
                            fn = lambda t, c, _r=ref: hns_improvement(t, c, _r)
                        value = fn(cell.mean(score), control.mean(score))
                        if not np.isfinite(value):
                            skipped.append(game)
                            continue
                        lo, hi = bootstrap_ci(
                            cell.scores[score], control.scores[score], fn, args.n_boot, rng
                        )
                        bars.append(
                            {"game": game, "beta": beta, "value": value, "lo": lo, "hi": hi}
                        )
                        if target == "best":
                            rows.append(
                                {
                                    "game": game, "beta": beta, "measure": measure,
                                    "gamma_i": cell.gamma_i, "epsilon": cell.epsilon,
                                    "control": round(control.mean(score), 4),
                                    "treatment": round(cell.mean(score), 4),
                                    "improvement": round(value, 4),
                                    "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                                    "n_seeds": cell.n_seeds, "blank_reason": "",
                                }
                            )
                    for game in skipped:
                        if target == "best":
                            rows.append(
                                {
                                    "game": game, "beta": best.get(game, ""),
                                    "measure": measure, "gamma_i": "", "epsilon": "",
                                    "control": "", "treatment": "", "improvement": "",
                                    "ci_lo": "", "ci_hi": "", "n_seeds": "",
                                    "blank_reason": blank_reason,
                                }
                            )
                    note = f" · omitted ({blank_reason}): {', '.join(skipped)}" if skipped else ""
                    label = "best β per game" if target == "best" else f"β={target}"
                    plot_improvement_bars(
                        bars,
                        title=f"Improvement over β=0 — {label}",
                        subtitle=f"{base}\nSymlog axis, linear within ±10 · error bars are a percentile bootstrap over seeds{note}",
                        ylabel=ylabel,
                        path=out / f"improvement_{measure}_{'best_beta' if target == 'best' else f'beta_{target}'}.png",
                        args=args,
                        colour_by_beta=colours if target == "best" else None,
                    )
            write_rows(
                rows,
                ("game", "beta", "measure", "gamma_i", "epsilon", "control", "treatment",
                 "improvement", "ci_lo", "ci_hi", "n_seeds", "blank_reason"),
                out / "improvement.csv",
            )
            n_png = len(list(out.glob("*.png")))
            print(f"  {score}/{mode}: {n_png} figures -> {out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
