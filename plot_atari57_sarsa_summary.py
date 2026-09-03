"""Cross-game summary of the Atari-57 separate-value-head SARSA sweep.

The per-game script (plot_atari57_seperate_heads_sarsa_sweep.py) writes thousands of
figures, every one of them about a single game. This is its counterpart: four figures
that answer the question the sweep exists to ask -- does the count bonus help, and at
which beta -- by aggregating across games.

Layout it reads is the same one that script reads, via the same SARSA_LAYOUT:

    <root>/<game>/beta_<b>/intrinsic_gamma_<gI>/epsilon_<e>/seed_<s>/metrics.npz

Normalisation
-------------
57 games score on wildly different scales, so nothing can be averaged across them raw.
There is no human/random reference table in this repo (atari_scores.csv holds only
Rainbow and PQN final scores), but the sweep carries its own control: beta=0 is the
extrinsic-only arm at matched budget, code and seeds. Every game is therefore normalised
by its own beta=0 score, and 1.0 means "no better than not having the bonus at all".

That choice also fixes what the aggregate means: this reports what the bonus *bought*,
not where the agent sits against published numbers.

Selection
---------
Which (gamma_I, epsilon) cell represents a beta is a real choice, and both answers are
produced, as parallel output trees:

    global    one config per beta, picked once across all games. "Which beta and
              configuration would I ship." One choice out of six amortised over every
              game, so the selection barely biases anything.
    per_game  the rule the existing best_per_beta figures use: per game, per beta, take
              the best cell. "What is beta worth once tuned per game." Optimistically
              biased -- and asymmetrically, because beta=0 maximises over 3 cells while
              beta>0 maximise over 6, handing the treatment arms a small free advantage.
              Measured at 1-2% on this sweep: real, always in the same direction, and
              small. Quote `global` as the headline; read `per_game` beside it.

Both rank cells against a *selection-free* anchor -- the mean over all three beta=0
cells -- because the control is itself one of the choices and cannot be used to pick
itself.

Statistic
---------
IQM (interquartile mean) over the pooled (game, seed) runs, with a stratified percentile
bootstrap that resamples games and then seeds within each sampled game, for the treatment
cell and the control cell alike, so control noise sits inside the interval. Resampling
seeds alone would produce intervals far too narrow at 57 games.

The mean-aggregated twin of the bar is written beside it on purpose: a mean well above
its IQM means the aggregate is being carried by a handful of games, which is the failure
mode IQM exists to expose. Read both.

Written per <score>/<selection> tree:

    beta_bar_iqm.png       IQM of the control-normalised score, one bar per beta
    beta_bar_mean.png      the same with the mean, for the comparison above
    beta_curve_iqm.png     that IQM against env step -- catches a bonus that helps
                           early and washes out by 100M, which no final-score bar can
    per_game_grid.png      one panel per game, raw scores, own y-axis: the backstop
                           that assumes no normalisation at all
    summary_beta.csv       per (game, beta): chosen cell, raw score, control, ratio
    config_selection.csv   which cell won, and what the runners-up scored
    aggregate.csv          the numbers behind the two bar figures

and once per run, at the top of the output directory:

    coverage.csv           every cell short of seeds, absent, or otherwise dropped
    unreadable_runs.csv    runs whose metrics.npz would not load

Caveats
-------
No --min-seeds gate. Every cell that has data is used and every shortfall is reported
instead, because a sweep is usually read while it is still finishing.

Games whose control does not finish positive are dropped from the normalised aggregates
and listed, because a ratio needs a denominator with a meaningful sign and scale: at a
negative control, a beta that makes the game *worse* scores above 1. double_dunk is the
one such game in this sweep. They still appear in the per-game grid, which plots raw
scores and so needs no denominator.

beta=5.0 is in this sweep as a reference point for the divergence metrics rather than a
performance candidate (see the sweep's generate_config.sh); --betas drops it.
"""

from __future__ import annotations

import csv
import dataclasses
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable, Literal, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tyro
from matplotlib.ticker import FuncFormatter
from scipy import stats

from plot_atari57_seperate_heads_sarsa_sweep import METRICS, SARSA_LAYOUT
from plot_count_layer_sweep import (
    SERIES_COLOURS,
    THEMES,
    Combo,
    load_combos,
    rolling_mean,
)

CONTROL_BETA = "0.0"

# Only the extrinsic return is ever scored here, and asking load_combos for the other
# eight series would multiply the resident set by nine -- 9405 runs x 24414 points is
# already ~1 GB at one metric.
SUMMARY_METRICS = {"extrinsic": METRICS["extrinsic"]}


@dataclasses.dataclass(frozen=True)
class Args:
    root_dir: Path = Path("data/atari57_seperate_heads_sarsa_sweep_partial")
    """Sweep dir containing <game>/beta_*/intrinsic_gamma_*/epsilon_*/seed_*/."""
    output_dir: Path = Path("graphs/atari57_seperate_heads_sarsa_sweep_partial/summary")
    """Figures and CSVs are written to <output_dir>/<score>/<selection>/."""
    games_file: Path = Path("atari57_games.txt")
    """Newline-separated game list, used only to report games with no data at all."""
    scores: tuple[Literal["final", "auc"], ...] = ("final", "auc")
    """Scoring rules; each gets its own output tree."""
    selection: tuple[Literal["global", "per_game"], ...] = ("global", "per_game")
    """Cell-selection rules; each gets its own tree under the score. See the module
    docstring -- `global` is the one to quote, `per_game` matches best_per_beta."""
    games: tuple[str, ...] = ()
    """Restrict to these games. Empty means every game on disk."""
    betas: tuple[str, ...] = ()
    """Restrict to these betas, as they appear in the path. Empty means all. beta=0.0 is
    always kept regardless, since it is the control everything is normalised by."""
    final_frac: float = 0.1
    """Fraction of the run averaged for the "final" score."""
    smooth: int = 750
    """Rolling-mean window in updates, matching the per-game script's default."""
    n_boot: int = 10000
    """Bootstrap resamples for the bar CIs."""
    curve_boot: int = 2000
    """Bootstrap resamples for the curve band. Lower than n_boot because the band is
    computed at every one of --curve-points positions."""
    curve_points: int = 200
    """x positions the curve band is evaluated at. The curves are already smoothed over
    750 updates, so bootstrapping all 24,414 of them buys nothing."""
    anchor_floor: float = 1e-6
    """Games whose β=0 control does not reach this are dropped from the normalised
    aggregates and reported instead. The bar is a *positive* one, not |x|: a negative
    control inverts the ratio, so a β that makes the game worse would score above 1."""
    boot_seed: int = 0
    """Seeds the bootstrap RNG so figures are reproducible."""
    theme: Literal["light", "dark"] = "light"
    dpi: int = 300
    figsize: tuple[float, float] = (10.0, 6.0)
    grid_cols: int = 8
    """Columns in the per-game panel grid."""


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------


def seed_scores(combo: Combo, final_frac: float, window: int) -> dict[str, np.ndarray]:
    """Per-seed "final" and "auc" scores, as (n_seeds,) arrays.

    score_combo() in plot_count_layer_sweep scores the *seed-mean* curve, which leaves
    no per-seed spread to bootstrap over. This is its body applied row-wise instead, so
    the mean of what it returns matches score_combo for "final" and the new CSVs
    reconcile with that script's summary.csv.

    Both scores come out of one smoothing pass: rolling_mean over a 750-wide window is
    the expensive part of this script (~9,400 curves at 57 games), and computing the two
    modes separately would double it.
    """
    curves = np.asarray(combo.curves("extrinsic"), dtype=np.float64)
    smoothed = np.stack([rolling_mean(row, window) for row in curves])
    return {
        "final": _score_rows(smoothed, combo.steps, "final", final_frac),
        "auc": _score_rows(smoothed, combo.steps, "auc", final_frac),
    }


def _score_rows(
    smoothed: np.ndarray, steps: np.ndarray, score: str, final_frac: float
) -> np.ndarray:
    out = np.full(smoothed.shape[0], np.nan)
    for i, row in enumerate(smoothed):
        finite = np.isfinite(row)
        if not finite.any():
            continue
        if score == "final":
            tail = math.ceil(row.shape[0] * final_frac)
            out[i] = float(np.nanmean(row[-tail:]))
        else:
            # Normalising by the x-range keeps runs of unequal length comparable.
            x, y = steps[finite], row[finite]
            out[i] = (
                float(y.mean())
                if x[-1] == x[0]
                else float(np.trapezoid(y, x) / (x[-1] - x[0]))
            )
    return out


@dataclasses.dataclass
class Cell:
    """One (game, beta, gamma_I, epsilon) cell and its per-seed scores."""

    game: str
    beta: str
    gamma_i: str
    epsilon: str
    combo: Combo
    scores: dict[str, np.ndarray]  # score mode -> (n_seeds,)

    @property
    def config(self) -> tuple[str, str]:
        return (self.gamma_i, self.epsilon)

    @property
    def n_seeds(self) -> int:
        return next(iter(self.scores.values())).shape[0]

    def mean(self, score: str) -> float:
        return float(np.nanmean(self.scores[score]))

    def label(self) -> str:
        return f"γI={self.gamma_i} ε={self.epsilon}"


# --------------------------------------------------------------------------------------
# Aggregation and the bootstrap
# --------------------------------------------------------------------------------------


def trimmed_mean_rows(values: np.ndarray, lo_q: float, hi_q: float) -> np.ndarray:
    """Row-wise trimmed mean of a 2-D array, NaN-aware.

    (0.25, 0.75) is the IQM -- sort, drop the bottom and top quarter, average what is
    left. (0.0, 1.0) is the plain mean, so the bar figures and their bootstrap share one
    code path and cannot drift apart.

    Rows may hold different numbers of real samples (a game short a seed pads with NaN),
    so the trim points are computed per row from that row's finite count. np.sort puts
    NaN last, which is what makes taking the first `n` entries of a row correct.
    """
    values = np.atleast_2d(np.asarray(values, dtype=np.float64))
    ordered = np.sort(values, axis=1)
    counts = np.sum(np.isfinite(values), axis=1)
    lo = np.floor(counts * lo_q).astype(int)
    hi = np.ceil(counts * hi_q).astype(int)
    # A row with 1-3 samples can trim to nothing; keep the whole row rather than
    # returning NaN for it.
    empty = hi <= lo
    lo = np.where(empty, 0, lo)
    hi = np.where(empty, counts, hi)

    ranks = np.arange(ordered.shape[1])[None, :]
    mask = (ranks >= lo[:, None]) & (ranks < hi[:, None])
    kept = mask.sum(axis=1)
    total = np.sum(np.where(mask, np.nan_to_num(ordered), 0.0), axis=1)
    return np.divide(total, kept, out=np.full(total.shape, np.nan), where=kept > 0)


def iqm(values: np.ndarray) -> float:
    """Interquartile mean of a flat sample."""
    return float(trimmed_mean_rows(np.asarray(values).reshape(1, -1), 0.25, 0.75)[0])


STATISTICS: dict[str, tuple[float, float]] = {"iqm": (0.25, 0.75), "mean": (0.0, 1.0)}


def _pad(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Stack ragged per-game seed arrays into (n_games, max_seeds), NaN-padded."""
    width = max(a.shape[0] for a in arrays)
    out = np.full((len(arrays), width), np.nan)
    for i, a in enumerate(arrays):
        out[i, : a.shape[0]] = a
    return out


def bootstrap_ratio_ci(
    treatment: Sequence[np.ndarray],
    control: Sequence[np.ndarray],
    quantiles: tuple[float, float],
    n_boot: int,
    rng: np.random.Generator,
    anchor_floor: float,
) -> tuple[float, float]:
    """Stratified percentile CI for a statistic of control-normalised run scores.

    `treatment[i]` and `control[i]` are the raw per-seed scores of one game's chosen beta
    cell and its control cell. Each resample draws games with replacement, then seeds
    with replacement *within* each drawn game -- for the control as well as the
    treatment, so the interval carries the control's noise too. Resampling only seeds
    would ignore game-to-game spread, which at 57 games is the larger of the two and
    would leave the bands far too narrow.

    Fully vectorised: the naive triple loop is 10,000 x 57 x 5 in Python and dominates
    the whole script.
    """
    t_raw, c_raw = _pad(treatment), _pad(control)
    n_games = t_raw.shape[0]
    t_counts = np.sum(np.isfinite(t_raw), axis=1)
    c_counts = np.sum(np.isfinite(c_raw), axis=1)

    games = rng.integers(0, n_games, size=(n_boot, n_games))
    # Draw within each sampled game's own seed count; NaN padding is never indexed.
    t_idx = rng.integers(
        0,
        np.maximum(t_counts[games], 1)[..., None],
        size=(n_boot, n_games, t_raw.shape[1]),
    )
    c_idx = rng.integers(
        0,
        np.maximum(c_counts[games], 1)[..., None],
        size=(n_boot, n_games, c_raw.shape[1]),
    )

    t_boot = np.take_along_axis(t_raw[games], t_idx, axis=2)
    c_boot = np.take_along_axis(c_raw[games], c_idx, axis=2)

    denominator = np.nanmean(c_boot, axis=2)
    denominator = np.where(denominator < anchor_floor, np.nan, denominator)
    ratios = t_boot / denominator[..., None]

    stats_per_boot = trimmed_mean_rows(ratios.reshape(n_boot, -1), *quantiles)
    return (
        float(np.nanpercentile(stats_per_boot, 2.5)),
        float(np.nanpercentile(stats_per_boot, 97.5)),
    )


# --------------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class Selection:
    """One selection rule's result: a cell per (game, beta), plus its audit trail."""

    chosen: dict[tuple[str, str], Cell]
    audit: list[dict]  # rows for config_selection.csv
    fields: Sequence[str]  # that file's columns, which differ by mode
    note: str  # carried into every subtitle


def anchors(
    cells: dict[tuple[str, str], dict[tuple[str, str], Cell]],
    games: Sequence[str],
    score: str,
) -> dict[str, float]:
    """Per game, the mean over *all* beta=0 cells -- a denominator no selection touches.

    The control is one of the choices being ranked, so ranking cells against the chosen
    control would let the control's own selection set the scale it is judged on. Averaging
    over all three epsilon cells sidesteps that: the anchor is fixed before anything is
    picked, and is used only to rank. The figures still normalise by the *chosen* control.
    """
    out: dict[str, float] = {}
    for game in games:
        control_cells = cells.get((game, CONTROL_BETA), {})
        if not control_cells:
            continue
        pooled = np.concatenate([c.scores[score] for c in control_cells.values()])
        value = float(np.nanmean(pooled))
        if np.isfinite(value):
            out[game] = value
    return out


def normalisable(value: float, floor: float) -> bool:
    """Can a game's control score legitimately act as a denominator?

    It must be positive, not merely non-zero. double_dunk's control finishes at -6.5, and
    dividing by a negative flips the whole scale: at beta=5.0 the score falls to -19.6 --
    much worse -- and scores a ratio of 3.0, the largest "win" in the sweep, while
    beta=0.01 lifting it to +4.6 scores -0.70. Under `abs()` that single game set the
    upper bound of every mean CI. Games that fail this are dropped from the normalised
    aggregates and reported; they keep their panel in the raw per-game grid.
    """
    return bool(np.isfinite(value)) and value >= floor


def select_global(
    cells: dict[tuple[str, str], dict[tuple[str, str], Cell]],
    games: Sequence[str],
    betas: Sequence[str],
    anchor: dict[str, float],
    score: str,
    anchor_floor: float,
) -> Selection:
    """One (gamma_I, epsilon) per beta, chosen once across every game.

    Ranked on the median across games of score/anchor rather than the mean, so one game
    whose ratio runs to 40x cannot appoint the config for all 57.
    """
    chosen: dict[tuple[str, str], Cell] = {}
    audit: list[dict] = []
    for beta in betas:
        ratios: dict[tuple[str, str], list[float]] = defaultdict(list)
        for game in games:
            scale = anchor.get(game)
            if scale is None or not normalisable(scale, anchor_floor):
                continue
            for config, cell in cells.get((game, beta), {}).items():
                ratios[config].append(cell.mean(score) / scale)
        if not ratios:
            continue
        summary = {
            config: float(np.nanmedian(values)) for config, values in ratios.items()
        }
        winner = max(summary, key=lambda c: summary[c])
        for config in sorted(summary, key=lambda c: summary[c], reverse=True):
            audit.append(
                {
                    "beta": beta,
                    "gamma_i": config[0],
                    "epsilon": config[1],
                    "median_ratio_to_anchor": round(summary[config], 4),
                    "n_games": len(ratios[config]),
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
            "median_ratio_to_anchor",
            "n_games",
            "chosen",
        ),
        note="one config per β, chosen across all games",
    )


def select_per_game(
    cells: dict[tuple[str, str], dict[tuple[str, str], Cell]],
    games: Sequence[str],
    betas: Sequence[str],
    anchor: dict[str, float],
    score: str,
    anchor_floor: float,
) -> Selection:
    """The best cell for each (game, beta) -- what best_per_beta already does.

    Optimistically biased: the cell is picked on the same seeds that are then reported,
    so the winner keeps whatever luck selected it. Worse, the bias is asymmetric, because
    beta=0 maximises over 3 cells and beta>0 over 6. Measured at 1-2% here. The `global`
    tree is the unbiased comparison; this one exists to be read beside it.
    """
    chosen: dict[tuple[str, str], Cell] = {}
    audit: list[dict] = []
    for game in games:
        scale = anchor.get(game)
        for beta in betas:
            candidates = cells.get((game, beta), {})
            if not candidates:
                continue
            cell = max(candidates.values(), key=lambda c: c.mean(score))
            chosen[(game, beta)] = cell
            audit.append(
                {
                    "game": game,
                    "beta": beta,
                    "gamma_i": cell.gamma_i,
                    "epsilon": cell.epsilon,
                    "n_candidates": len(candidates),
                    "score": round(cell.mean(score), 3),
                    "ratio_to_anchor": (
                        round(cell.mean(score) / scale, 4)
                        if scale is not None and normalisable(scale, anchor_floor)
                        else ""
                    ),
                }
            )
    return Selection(
        chosen=chosen,
        audit=audit,
        fields=(
            "game",
            "beta",
            "gamma_i",
            "epsilon",
            "n_candidates",
            "score",
            "ratio_to_anchor",
        ),
        note="best config per game per β · optimistically biased, see docstring",
    )


SELECTORS: dict[str, Callable[..., Selection]] = {
    "global": select_global,
    "per_game": select_per_game,
}


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def _style_panel(ax, theme: dict, xlabel: str = "", ylabel: str = "") -> None:
    """The repo's axis styling, minus plot_count_layer_sweep._style_axes' assumptions.

    That helper hardcodes an "Environment steps" label and a 1e6 tick formatter, neither
    of which suits a categorical beta axis or a 57-panel grid. The palette is shared so
    these figures still sit beside the per-game ones.
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


def _titles(ax, theme: dict, title: str, subtitle: str) -> None:
    ax.set_title(title, color=theme["ink"], fontsize=13, loc="left", pad=18)
    ax.text(
        0.0,
        1.02,
        subtitle,
        transform=ax.transAxes,
        color=theme["muted"],
        fontsize=9,
        va="bottom",
    )


def _save(fig, path: Path, args: Args, theme: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor=theme["surface"])
    plt.close(fig)


def plot_beta_bar(
    rows: list[dict],
    *,
    statistic: str,
    title: str,
    subtitle: str,
    path: Path,
    args: Args,
    ylim: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """One bar per beta>0, control-normalised, with its bootstrap CI.

    beta=0 is a dashed reference line at 1.0 rather than a bar: it is the denominator, so
    a bar for it would be exactly 1.0 with no interval and would read as a measurement.

    Bars are coloured by beta rank, matching the per-game figures' convention that a beta
    keeps its colour across a family of plots. Returns the y-limits so the IQM and mean
    twins can be drawn on the same scale.
    """
    theme = THEMES[args.theme]
    fig, ax = plt.subplots(figsize=args.figsize, dpi=args.dpi)
    fig.patch.set_facecolor(theme["surface"])

    x = np.arange(len(rows))
    values = np.array([r[statistic] for r in rows])
    lo = np.array([r[f"{statistic}_lo"] for r in rows])
    hi = np.array([r[f"{statistic}_hi"] for r in rows])
    colours = [SERIES_COLOURS[(i + 1) % len(SERIES_COLOURS)] for i in range(len(rows))]

    ax.axhline(
        1.0,
        color=theme["ink_secondary"],
        linewidth=1.0,
        linestyle="--",
        zorder=1,
        label="β=0 control",
    )
    ax.bar(x, values, width=0.62, color=colours, zorder=2, linewidth=0)
    ax.errorbar(
        x,
        values,
        yerr=np.vstack([values - lo, hi - values]),
        fmt="none",
        ecolor=theme["ink"],
        elinewidth=1.2,
        capsize=4,
        zorder=3,
    )
    for xi, value, top in zip(x, values, hi):
        ax.annotate(
            f"{value:.2f}",
            xy=(xi, top),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            color=theme["ink_secondary"],
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"β={r['beta']}" for r in rows])
    _style_panel(ax, theme, ylabel="Score / β=0 control")
    ax.set_ylim(*(ylim if ylim is not None else (0.0, float(hi.max()) * 1.12)))
    _titles(ax, theme, title, subtitle)
    ax.legend(
        loc="upper right", fontsize=8, frameon=False, labelcolor=theme["ink_secondary"]
    )
    _save(fig, path, args, theme)
    return ax.get_ylim()


def shared_bar_ylim(rows: list[dict]) -> tuple[float, float]:
    """A y-limit that fits the IQM figure and its mean twin.

    They must share a scale to be comparable at a glance, but sizing it to the IQM alone
    clips the mean's wider intervals, and a truncated error bar reads as a short one.
    Headroom above the tallest bound leaves room for the value labels.
    """
    tops = [
        row[f"{name}_hi"] for row in rows for name in STATISTICS if f"{name}_hi" in row
    ]
    return (0.0, max(tops + [1.0]) * 1.12)


def plot_beta_curve(
    series: list[dict],
    *,
    title: str,
    subtitle: str,
    path: Path,
    args: Args,
) -> None:
    """Control-normalised IQM against env step, one line per beta.

    The cross-game analogue of the per-game best_per_beta_extrinsic figures, and the only
    one of these that can show a bonus helping early and washing out by 100M -- a final
    score cannot distinguish that from never having helped.
    """
    theme = THEMES[args.theme]
    fig, ax = plt.subplots(figsize=args.figsize, dpi=args.dpi)
    fig.patch.set_facecolor(theme["surface"])

    # The control's own curve, not a flat line at 1.0. Every series is divided by the
    # control's *final* score, so all of them climb from 0 -- without the control drawn
    # on the same scale a reader cannot tell "still learning" from "behind the control",
    # and every β would look catastrophic for the first 40M steps. The flat line is kept
    # as the endpoint marker it actually is.
    ax.axhline(1.0, color=theme["grid"], linewidth=1.0, linestyle="--", zorder=1)
    for rank, entry in enumerate(series):
        control = entry["beta"] == CONTROL_BETA
        # The control is rank 0 and drawn in ink, so treatment betas start at colour 1 --
        # which is where plot_beta_bar starts them too, so a beta keeps its colour between
        # the bar and the curve the way the per-game figures keep one across a family.
        colour = (
            theme["ink_secondary"]
            if control
            else SERIES_COLOURS[rank % len(SERIES_COLOURS)]
        )
        ax.plot(
            entry["steps"],
            entry["values"],
            color=colour,
            linewidth=2.4 if control else 2.0,
            linestyle="--" if control else "-",
            solid_capstyle="round",
            label=(
                f"β={entry['beta']} control  (n={entry['n_runs']})"
                if control
                else f"β={entry['beta']}  (n={entry['n_runs']})"
            ),
            zorder=4 if control else 3,
        )
        ax.fill_between(
            entry["steps"],
            entry["lo"],
            entry["hi"],
            color=colour,
            alpha=0.10 if control else 0.12,
            linewidth=0,
            zorder=2,
        )

    _style_panel(ax, theme, ylabel="IQM score / β=0 control")
    ax.set_xlabel("Environment steps", color=theme["ink_secondary"], fontsize=10)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _: "0" if v == 0 else f"{v / 1e6:,.0f}M")
    )
    ax.set_ylim(bottom=0)
    _titles(ax, theme, title, subtitle)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
        frameon=False,
        labelcolor=theme["ink_secondary"],
    )
    _save(fig, path, args, theme)


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

    Deliberately un-normalised. Every other figure here divides by the control, so if the
    normalisation is doing something strange this is the figure that shows it -- the only
    assumption it makes is that a game's own scores are comparable to each other.

    Panels are ordered by how much the best beta>0 beat the control, so the games where
    the bonus helps sit together at the front instead of scattered alphabetically.
    """
    theme = THEMES[args.theme]
    # Never stretch a handful of games across the full width -- at --games boxing alien
    # an 8-column grid is two panels and six blanks.
    cols = max(1, min(args.grid_cols, len(panels)))
    rows = math.ceil(len(panels) / cols)
    panel_h = 2.3
    # The heading block is a fixed number of inches, so it neither collides with the top
    # row on a one-row grid nor floats away from it on an eight-row one.
    head_h = 0.75
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(2.6 * cols, panel_h * rows + head_h),
        dpi=args.dpi,
        squeeze=False,
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
        colours = [SERIES_COLOURS[i % len(SERIES_COLOURS)] for i in range(len(betas))]

        if np.isfinite(panel["control"]):
            ax.axhline(
                panel["control"],
                color=theme["ink_secondary"],
                linewidth=0.9,
                linestyle="--",
                zorder=1,
            )
        ax.errorbar(
            x,
            means,
            yerr=errs,
            fmt="o",
            markersize=4,
            color=theme["ink"],
            ecolor=theme["muted"],
            elinewidth=1.0,
            capsize=2,
            zorder=3,
            linestyle="none",
        )
        ax.scatter(x, means, c=colours, s=26, zorder=4, linewidths=0)

        _style_panel(ax, theme)
        ax.set_xticks(x)
        ax.set_xticklabels([b for b in betas], fontsize=6, rotation=45)
        ax.tick_params(labelsize=6)
        ax.set_title(panel["game"], color=theme["ink"], fontsize=8, loc="left", pad=4)
        if np.isfinite(means).any() and np.nanmin(means - np.nan_to_num(errs)) >= 0:
            ax.set_ylim(bottom=0)

    height = panel_h * rows + head_h
    fig.text(
        0.006,
        1 - 0.28 / height,
        title,
        color=theme["ink"],
        fontsize=14,
        ha="left",
        va="top",
    )
    fig.text(
        0.006,
        1 - 0.52 / height,
        subtitle,
        color=theme["muted"],
        fontsize=9,
        ha="left",
        va="top",
    )
    fig.tight_layout(rect=(0, 0, 1, 1 - head_h / height))
    _save(fig, path, args, theme)


def bootstrap_curve_band(
    raw: np.ndarray,
    control: np.ndarray,
    positions: np.ndarray,
    quantiles: tuple[float, float],
    n_boot: int,
    rng: np.random.Generator,
    anchor_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap band for the IQM curve, sharing one set of resamples across positions.

    `raw` is (games, seeds, steps) of un-normalised curves and `control` is (games,
    seeds) of raw control scores. Drawing fresh games and seeds at every x would make
    neighbouring points independent and the band jagged; reusing the draws keeps it a
    band around a curve. Control denominators are resampled once for the same reason and
    because they do not vary with x.
    """
    n_games, n_seeds, _ = raw.shape
    present = np.any(np.isfinite(raw), axis=2).sum(axis=1)
    control_counts = np.sum(np.isfinite(control), axis=1)

    games = rng.integers(0, n_games, size=(n_boot, n_games))
    seed_idx = rng.integers(
        0, np.maximum(present[games], 1)[..., None], size=(n_boot, n_games, n_seeds)
    )
    control_idx = rng.integers(
        0,
        np.maximum(control_counts[games], 1)[..., None],
        size=(n_boot, n_games, control.shape[1]),
    )
    denominator = np.nanmean(
        np.take_along_axis(control[games], control_idx, axis=2), axis=2
    )
    denominator = np.where(denominator < anchor_floor, np.nan, denominator)

    lo = np.empty(positions.shape[0])
    hi = np.empty(positions.shape[0])
    for j, position in enumerate(positions):
        column = raw[:, :, position]
        drawn = np.take_along_axis(column[games], seed_idx, axis=2)
        per_boot = trimmed_mean_rows(
            (drawn / denominator[..., None]).reshape(n_boot, -1), *quantiles
        )
        lo[j] = np.nanpercentile(per_boot, 2.5)
        hi[j] = np.nanpercentile(per_boot, 97.5)
    return lo, hi


# --------------------------------------------------------------------------------------
# CSV helpers
# --------------------------------------------------------------------------------------


def write_rows(rows: list[dict], fieldnames: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


COVERAGE_FIELDS = ("kind", "game", "beta", "gamma_i", "epsilon", "n_seeds", "detail")
SUMMARY_FIELDS = (
    "game",
    "beta",
    "gamma_i",
    "epsilon",
    "n_seeds",
    "score",
    "control_score",
    "ratio",
)
AGGREGATE_FIELDS = (
    "beta",
    "gamma_i",
    "epsilon",
    "n_games",
    "n_runs",
    "iqm",
    "iqm_lo",
    "iqm_hi",
    "mean",
    "mean_lo",
    "mean_hi",
    "median",
)


def coverage_rows(
    cells: dict[tuple[str, str], dict[tuple[str, str], Cell]],
    games: Sequence[str],
    betas: Sequence[str],
    listed_games: Sequence[str],
) -> list[dict]:
    """Everything the sweep is missing, rather than a filter that hides it.

    The expected grid is the union of configs actually seen for each beta, not a
    hardcoded 33: a sweep whose shape changes should not need this file edited, and a
    config missing from *every* game is a fact about the sweep script, not about a run.
    """
    expected: dict[str, set[tuple[str, str]]] = defaultdict(set)
    counts: list[int] = []
    for (game, beta), configs in cells.items():
        expected[beta].update(configs)
        counts.extend(cell.n_seeds for cell in configs.values())
    modal = max(set(counts), key=counts.count) if counts else 0

    rows: list[dict] = []
    for game in listed_games:
        if game not in games:
            rows.append(
                {
                    "kind": "missing_game",
                    "game": game,
                    "beta": "",
                    "gamma_i": "",
                    "epsilon": "",
                    "n_seeds": 0,
                    "detail": "no runs on disk",
                }
            )
    for game in games:
        for beta in betas:
            present = cells.get((game, beta), {})
            for config in sorted(expected[beta]):
                cell = present.get(config)
                if cell is None:
                    rows.append(
                        {
                            "kind": "missing_cell",
                            "game": game,
                            "beta": beta,
                            "gamma_i": config[0],
                            "epsilon": config[1],
                            "n_seeds": 0,
                            "detail": "cell absent",
                        }
                    )
                elif cell.n_seeds < modal:
                    rows.append(
                        {
                            "kind": "short_seeds",
                            "game": game,
                            "beta": beta,
                            "gamma_i": config[0],
                            "epsilon": config[1],
                            "n_seeds": cell.n_seeds,
                            "detail": f"expected {modal}",
                        }
                    )
    return rows


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def build_cells(
    root: Path, final_frac: float, smooth: int, games: Sequence[str] = ()
) -> tuple[dict[tuple[str, str], dict[tuple[str, str], Cell]], list[dict]]:
    """Load every run and score it per seed. The smoothing pass dominates the runtime.

    `games` is applied before scoring rather than after: at 57 games the rolling means
    are minutes of work, and --games exists precisely to skip them while iterating.
    """
    combos, failures = load_combos(root, SARSA_LAYOUT, SUMMARY_METRICS)
    if not combos:
        raise SystemExit(f"no runs found under {root}")
    if games:
        wanted = set(games)
        combos = [c for c in combos if c.game in wanted]
        if not combos:
            raise SystemExit(f"no runs for games {sorted(wanted)}")

    cells: dict[tuple[str, str], dict[tuple[str, str], Cell]] = defaultdict(dict)
    for index, combo in enumerate(combos, start=1):
        if index % 200 == 0 or index == len(combos):
            print(f"  scoring {index}/{len(combos)} cells", flush=True)
        cell = Cell(
            game=combo.game,
            beta=combo.beta,
            gamma_i=combo.params["gamma_i"],
            epsilon=combo.params["epsilon"],
            combo=combo,
            scores=seed_scores(combo, final_frac, smooth),
        )
        cells[(cell.game, cell.beta)][cell.config] = cell
    return cells, failures


def main(args: Args) -> None:
    print(f"loading {args.root_dir}")
    cells, failures = build_cells(
        args.root_dir, args.final_frac, args.smooth, args.games
    )

    games = sorted({game for game, _ in cells})
    betas = sorted({beta for _, beta in cells}, key=float)
    if args.betas:
        wanted = set(args.betas) | {CONTROL_BETA}
        betas = [b for b in betas if b in wanted]
    treatment_betas = [b for b in betas if b != CONTROL_BETA]
    if CONTROL_BETA not in betas:
        raise SystemExit(
            f"no β={CONTROL_BETA} arm under {args.root_dir}; every figure here is "
            "normalised by that control"
        )
    print(f"{len(games)} games, betas {betas}")

    listed = []
    if args.games_file.exists():
        listed = [
            line.strip()
            for line in args.games_file.read_text().splitlines()
            if line.strip()
        ]
    write_rows(
        coverage_rows(cells, games, betas, listed),
        COVERAGE_FIELDS,
        args.output_dir / "coverage.csv",
    )
    if failures:
        write_rows(
            failures,
            ("game", "beta", "gamma_i", "epsilon", "seed", "error"),
            args.output_dir / "unreadable_runs.csv",
        )
    print(f"  coverage.csv, {len(failures)} unreadable runs")

    for score in args.scores:
        anchor = anchors(cells, games, score)
        for mode in args.selection:
            rng = np.random.default_rng(args.boot_seed)
            out = args.output_dir / score / mode
            selection = SELECTORS[mode](
                cells, games, betas, anchor, score, args.anchor_floor
            )

            # The control every figure divides by: whichever beta=0 cell this mode chose.
            controls: dict[str, float] = {}
            dropped: list[dict] = []
            for game in games:
                cell = selection.chosen.get((game, CONTROL_BETA))
                value = cell.mean(score) if cell is not None else np.nan
                if cell is None or not normalisable(value, args.anchor_floor):
                    dropped.append(
                        {
                            "kind": "unnormalisable_game",
                            "game": game,
                            "beta": CONTROL_BETA,
                            "gamma_i": "",
                            "epsilon": "",
                            "n_seeds": cell.n_seeds if cell else 0,
                            "detail": f"{score}/{mode}: control score {value:.3g}",
                        }
                    )
                    continue
                controls[game] = value

            summary: list[dict] = []
            aggregate: list[dict] = []
            bar_rows: list[dict] = []
            for beta in treatment_betas:
                usable = [
                    game
                    for game in games
                    if game in controls and (game, beta) in selection.chosen
                ]
                if not usable:
                    continue
                treatment = [
                    selection.chosen[(game, beta)].scores[score] for game in usable
                ]
                control = [
                    selection.chosen[(game, CONTROL_BETA)].scores[score]
                    for game in usable
                ]
                ratios = np.concatenate(
                    [t / controls[game] for t, game in zip(treatment, usable)]
                )

                row = {
                    "beta": beta,
                    "n_games": len(usable),
                    "n_runs": int(np.sum(np.isfinite(ratios))),
                    "median": round(float(np.nanmedian(ratios)), 4),
                }
                configs = {selection.chosen[(game, beta)].config for game in usable}
                only = (
                    next(iter(configs)) if len(configs) == 1 else ("varies", "varies")
                )
                row["gamma_i"], row["epsilon"] = only
                for name, quantiles in STATISTICS.items():
                    point = float(
                        trimmed_mean_rows(ratios.reshape(1, -1), *quantiles)[0]
                    )
                    lo, hi = bootstrap_ratio_ci(
                        treatment,
                        control,
                        quantiles,
                        args.n_boot,
                        rng,
                        args.anchor_floor,
                    )
                    row[name] = round(point, 4)
                    row[f"{name}_lo"] = round(lo, 4)
                    row[f"{name}_hi"] = round(hi, 4)
                aggregate.append({k: row.get(k, "") for k in AGGREGATE_FIELDS})
                bar_rows.append(row)

            for game in games:
                for beta in betas:
                    cell = selection.chosen.get((game, beta))
                    if cell is None:
                        continue
                    control_score = controls.get(game)
                    summary.append(
                        {
                            "game": game,
                            "beta": beta,
                            "gamma_i": cell.gamma_i,
                            "epsilon": cell.epsilon,
                            "n_seeds": cell.n_seeds,
                            "score": round(cell.mean(score), 3),
                            "control_score": (
                                round(control_score, 3) if control_score else ""
                            ),
                            "ratio": (
                                round(cell.mean(score) / control_score, 4)
                                if control_score
                                else ""
                            ),
                        }
                    )

            write_rows(summary, SUMMARY_FIELDS, out / "summary_beta.csv")
            write_rows(selection.audit, selection.fields, out / "config_selection.csv")
            write_rows(aggregate, AGGREGATE_FIELDS, out / "aggregate.csv")
            if dropped:
                write_rows(dropped, COVERAGE_FIELDS, out / "unnormalisable_games.csv")

            label = {"final": "final performance", "auc": "area under the curve"}[score]
            configs_note = " · ".join(
                f"β={r['beta']}: γI={r['gamma_i']} ε={r['epsilon']}" for r in bar_rows
            )
            base = f"{len(controls)} games, {label} · {selection.note}" + (
                f" · {len(dropped)} game(s) unnormalisable" if dropped else ""
            )

            if bar_rows:
                ylim = shared_bar_ylim(bar_rows)
                plot_beta_bar(
                    bar_rows,
                    statistic="iqm",
                    title="Count bonus versus its own control, by β",
                    subtitle=f"IQM over (game, seed) runs · {base}",
                    path=out / "beta_bar_iqm.png",
                    args=args,
                    ylim=ylim,
                )
                plot_beta_bar(
                    bar_rows,
                    statistic="mean",
                    title="Count bonus versus its own control, by β",
                    subtitle=(
                        f"Mean over (game, seed) runs · {base} · "
                        "compare against the IQM twin"
                    ),
                    path=out / "beta_bar_mean.png",
                    args=args,
                    ylim=ylim,
                )
                print(f"  {score}/{mode}: bars -> {out}")
                _write_curve(
                    selection, controls, treatment_betas, score, out, base, args, rng
                )
                _write_grid(selection, controls, betas, score, out, base, args)

            if configs_note:
                (out / "chosen_configs.txt").write_text(configs_note + "\n")


def _write_curve(
    selection: Selection,
    controls: dict[str, float],
    treatment_betas: Sequence[str],
    score: str,
    out: Path,
    base: str,
    args: Args,
    rng: np.random.Generator,
) -> None:
    """Build and draw the IQM-against-steps figure.

    Curves are normalised by the control's *scalar* score, not by its curve. The control
    curve is ~0 for the first few million steps (the EMA is undefined until a seed's
    first episode ends), so a pointwise ratio would explode exactly where this figure is
    meant to be read.
    """
    series: list[dict] = []
    for beta in [CONTROL_BETA, *treatment_betas]:
        usable = [
            game
            for game in controls
            if (game, beta) in selection.chosen
            and (game, CONTROL_BETA) in selection.chosen
        ]
        if not usable:
            continue
        curves, control_scores = [], []
        for game in usable:
            cell = selection.chosen[(game, beta)]
            raw = np.asarray(cell.combo.curves("extrinsic"), dtype=np.float64)
            curves.append(np.stack([rolling_mean(row, args.smooth) for row in raw]))
            control_scores.append(selection.chosen[(game, CONTROL_BETA)].scores[score])

        length = min(c.shape[1] for c in curves)
        width = max(c.shape[0] for c in curves)
        padded = np.full((len(curves), width, length), np.nan)
        for i, curve in enumerate(curves):
            padded[i, : curve.shape[0]] = curve[:, :length]

        steps = selection.chosen[(usable[0], beta)].combo.steps[:length]
        positions = np.unique(
            np.linspace(0, length - 1, min(args.curve_points, length)).astype(int)
        )
        normalised = padded / np.array([controls[g] for g in usable])[:, None, None]
        point = trimmed_mean_rows(
            normalised[:, :, positions]
            .transpose(2, 0, 1)
            .reshape(positions.shape[0], -1),
            *STATISTICS["iqm"],
        )
        lo, hi = bootstrap_curve_band(
            padded,
            _pad(control_scores),
            positions,
            STATISTICS["iqm"],
            args.curve_boot,
            rng,
            args.anchor_floor,
        )
        series.append(
            {
                "beta": beta,
                "steps": steps[positions],
                "values": point,
                "lo": lo,
                "hi": hi,
                "n_runs": int(np.sum(np.any(np.isfinite(padded), axis=2))),
            }
        )

    if series:
        plot_beta_curve(
            series,
            title="Count bonus versus its own control, over training",
            subtitle=(
                "IQM over (game, seed) runs, each divided by its game's control "
                f"{'final' if score == 'final' else 'AUC'} score · {base}"
            ),
            path=out / "beta_curve_iqm.png",
            args=args,
        )


def _write_grid(
    selection: Selection,
    controls: dict[str, float],
    betas: Sequence[str],
    score: str,
    out: Path,
    base: str,
    args: Args,
) -> None:
    """Build and draw the per-game panel grid, on raw scores."""
    panels: list[dict] = []
    for game in sorted({g for g, _ in selection.chosen}):
        means, errs = {}, {}
        for beta in betas:
            cell = selection.chosen.get((game, beta))
            if cell is None:
                continue
            values = cell.scores[score]
            finite = np.isfinite(values)
            means[beta] = float(np.nanmean(values))
            if finite.sum() > 1:
                sem = float(np.nanstd(values, ddof=1) / math.sqrt(finite.sum()))
                errs[beta] = sem * float(stats.t.ppf(0.975, finite.sum() - 1))
            else:
                errs[beta] = np.nan
        if not means:
            continue
        control = means.get(CONTROL_BETA, np.nan)
        treatment = [means[b] for b in betas if b != CONTROL_BETA and b in means]
        effect = (
            max(treatment) / control
            if treatment and normalisable(control, args.anchor_floor)
            else -np.inf
        )
        panels.append(
            {
                "game": game,
                "mean": means,
                "err": errs,
                "control": control,
                "effect": effect,
            }
        )

    panels.sort(key=lambda p: p["effect"], reverse=True)
    plot_per_game_grid(
        panels,
        betas=betas,
        title="Per-game raw scores by β",
        subtitle=(
            f"{base}\nOwn y-axis per panel, no normalisation · dashed line is that "
            "game's β=0 control · panels ordered by best β / control"
        ),
        path=out / "per_game_grid.png",
        args=args,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
