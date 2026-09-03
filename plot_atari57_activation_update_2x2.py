"""Atari-57 2x2: activation (all-relu vs FTA) x bootstrap rule (Q-learning vs SARSA).

Every other plotter in this repo reads one sweep root and recovers hyperparameters from
directory names. Activation and `sarsa_returns` appear in no path level -- they are
encoded in *which sweep* you are looking at -- so this is the one script that crosses a
sweep boundary, and the four cells come from three sweeps written by three training
scripts:

    relu x Q-learning  data/atari57_baseline/<game>/qlearning/seed_*          pqn_atari.py
    relu x SARSA       data/atari57_baseline/<game>/sarsa/seed_*              pqn_atari.py
    FTA  x Q-learning  data/atari57_count_layer_sweep/<game>/conv2/beta_0.0/next_0.0/seed_*
                                                              pqn_atari_with_counts.py
    FTA  x SARSA       data/atari57_seperate_heads_sarsa_sweep/<game>/beta_0.0
                           /intrinsic_gamma_0.9/epsilon_0.001/seed_*
                              pqn_atari_counts_with_seperate_value_head.py

Why these four are comparable
-----------------------------
Checked rather than assumed, because a cross-sweep comparison is exactly where a silent
mismatch hides:

  * `diff` of the four config.yaml files differs only in `network.cnn_activation_2`
    (relu vs fta, bound 1.0 / eta 0.25 / static centres), `network.type`, `sarsa_returns`,
    `num_env_threads` (throughput only), and the counts scripts' inert extras. Learning
    rates, gamma=0.99, lam=0.65, epsilon_end=0.001, 128 envs x 32 steps, 2 epochs,
    32 minibatches, reward_scale and total_time_steps=1e8 are identical.

  * Both FTA arms are genuinely null-bonus. pqn_atari_with_counts.py adds
    `beta * intrinsic_reward` to the reward, which is zero at beta=0. The separate-heads
    script weights the intrinsic head's loss by `intrinsic_loss_coef`, which the sweep
    sets to 0.0 in every beta=0 run, so that head gets no gradient and never reaches the
    shared trunk. Both reduce to plain PQN on an FTA-at-conv2 trunk.

  * The return metric is one quantity under two names. The baseline logs
    `extrinsic_return_ema`; the counts scripts log `extrinsic_return_per_game_ema`. The
    call sites are line-for-line identical -- same helper_functions.update_ema, same
    is_done-weighted mean of `returned_episode_returns`, same
    ema_alpha = 2/(num_episodes_for_average+1) at num_episodes_for_average=30. Only the
    field name differs, and it is the one thing that would break this quietly:
    load_combos raises KeyError on a missing key and files the run as unreadable, so
    pointing any other script in this repo at the baseline marks all 570 runs bad.

Figures, per <score> tree
-------------------------
    hns_aggregate.png          IQM human-normalised score, one bar per arm, plus the
                               two main effects and the interaction read off them
    hns_per_game.png           four grouped bars per game, ordered by relu x Q-learning
    curve_grid_extrinsic.png   episode return against env steps, one panel per game

and once at the top level, because no scoring rule enters a curve figure:

    games/<game>/extrinsic.png the same four curves full size, one file per game

Where the uncertainty is shown, and where it is not
---------------------------------------------------
Bands on the curve figures are 95% percentile bootstraps over that arm's seeds. At 5
seeds a percentile interval is about 0.62x the width of the Student-t one and
under-covers; it is labelled "95% bootstrap CI" because that is what it is, not because
it attains 95% coverage.

The HNS figures carry no intervals at all, following plot_atari57_sarsa_figures.py's
`plot_improvement_bars`. An IQM over games has two nested sources of noise (games, then
seeds) and a bar chart of 57 games x 4 arms with error bars on every bar is unreadable;
seed spread belongs on -- and is shown on -- the curve figures.

Caveats
-------
The two FTA arms come from different training scripts. Both are provably null-bonus, but
the separate-heads one still *constructs* an untrained intrinsic head, which consumes
network-init RNG differently. Seed k is therefore not the same initialisation across
arms: read this as four independent 5-seed samples, never as a paired comparison.

pqn_atari.py received the minibatch-shuffle key fix only in the commit accompanying
atari57_baseline (the counts scripts got it earlier). All four arms' data postdate the
fix; earlier pure-PQN data on disk does not, and must not be substituted for the relu
arms.

Coverage is derived at run time and never hardcoded, because atari57_baseline is still
filling in. Anything absent or thin lands in coverage.csv rather than being silently
averaged over fewer seeds.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from pathlib import Path
from typing import Literal, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro
from matplotlib.ticker import FuncFormatter

from plot_atari57_seperate_heads_sarsa_sweep import SARSA_LAYOUT
from plot_atari57_sarsa_figures import (
    _grid_shape,
    _save,
    _style_panel,
    _titles,
    load_hns,
    seed_scores,
    write_rows,
)
from plot_count_layer_sweep import (
    COUNT_LAYER_LAYOUT,
    THEMES,
    Combo,
    Layout,
    Metric,
    rolling_mean,
    load_combos,
)

import re

# ----------------------------------------------------------------------------------------
# Arms
# ----------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Arm:
    """One cell of the 2x2."""

    key: str
    activation: Literal["relu", "fta"]
    update: Literal["qlearning", "sarsa"]
    label: str


ARMS = (
    Arm("relu_qlearning", "relu", "qlearning", "relu · Q-learning"),
    Arm("relu_sarsa", "relu", "sarsa", "relu · SARSA"),
    Arm("fta_qlearning", "fta", "qlearning", "FTA · Q-learning"),
    Arm("fta_sarsa", "fta", "sarsa", "FTA · SARSA"),
)
ARM_BY_KEY = {a.key: a for a in ARMS}

# Hue carries the activation, lightness step within the hue carries the bootstrap rule,
# so the factorial structure is legible in the colours themselves.
#
# NOT SERIES_COLOURS. That palette is matplotlib's tab10, and for these four arms it
# fails: #ff7f0e vs #2ca02c is Delta E 0.7 in OKLab under both protanopia and
# deutanopia -- indistinguishable -- against a floor of 6 and a target of 8, and #ff7f0e
# also sits outside the dark-mode lightness band. These steps were found by searching
# OKLCH on the Okabe-Ito hues (244, 236, 48, 77 degrees) under the same six checks, per
# mode, because a dark palette is selected rather than flipped. Both modes pass with no
# warnings: worst all-pairs CVD Delta E 19.3 light / 13.5 dark, worst normal-vision pair
# 19.7 / 15.2 against a floor of 15, contrast >= 3.02 / 3.04 against the surface. Re-run
# those checks if any value here changes.
ARM_COLOURS = {
    "light": {
        "relu_qlearning": "#005d93",
        "relu_sarsa": "#079cda",
        "fta_qlearning": "#823500",
        "fta_sarsa": "#b37b04",
    },
    "dark": {
        "relu_qlearning": "#036ba8",
        "relu_sarsa": "#0d9ddc",
        "fta_qlearning": "#aa4900",
        "fta_sarsa": "#c0850f",
    },
}

# Redundant with hue on the curve figures, where a line can carry it. Bars cannot, which
# is why the colours above have to stand on their own.
ARM_LINESTYLE = {"qlearning": "-", "sarsa": "--"}

# The baseline's tree is <game>/<bootstrap rule>/seed_*: no beta level, which is why
# load_combos treats its `beta` group as optional.
BASELINE_LAYOUT = Layout(
    glob="*/*/seed_*/metrics.npz",
    pattern=re.compile(
        r"(?P<game>[^/]+)/(?P<update>qlearning|sarsa)/seed_(?P<seed>[^/]+)$"
    ),
    dims=("update",),
)

# One metric, under each sweep's own name for it. See the module docstring: these are the
# same quantity, and asking for the wrong one marks every run in that sweep unreadable.
BASELINE_METRICS = {"extrinsic": Metric("extrinsic_return_ema", "Episode return (EMA)")}
COUNTS_METRICS = {
    "extrinsic": Metric("extrinsic_return_per_game_ema", "Episode return (EMA)")
}


# ----------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------


def _normalise_failures(failures: list[dict], source: str, arm: str) -> list[dict]:
    """Fold each layout's own dim columns into one schema.

    load_combos builds a failure row as {game, beta, **dims, seed, error}, and the three
    sources have different dims -- `update`, `gamma_i`/`epsilon`, `position`/`next`. Handing
    the concatenation to write_rows raises "dict contains fields not in fieldnames", so the
    dims are collapsed into a `cell` string here, where the layout is still known.

    `arm` is a literal for the two single-arm sources; the baseline carries both of its arms
    in the `update` dim, so it passes "" and the arm is read off the row.
    """
    rows = []
    for failure in failures:
        dims = {k: v for k, v in failure.items() if k not in ("game", "seed", "error")}
        rows.append(
            {
                "source": source,
                "game": failure["game"],
                "arm": f"relu_{failure['update']}" if "update" in failure else arm,
                # Empty values dropped: the baseline has no beta level, and "beta=" adds
                # nothing but noise to every one of its rows.
                "cell": " ".join(f"{k}={v}" for k, v in sorted(dims.items()) if v),
                "seed": failure["seed"],
                "error": failure["error"],
            }
        )
    return rows


def load_arms(args: "Args") -> tuple[dict[tuple[str, str], Combo], list[dict]]:
    """(game, arm_key) -> Combo, plus every run that could not be read.

    Each source is filtered down to the single cell that represents its arm; a sweep root
    that is missing entirely is reported rather than raising, since the baseline is still
    filling in and a partial local copy has to be plottable.
    """
    combos: dict[tuple[str, str], Combo] = {}
    failures: list[dict] = []

    def _load(
        root: Path,
        layout: Layout,
        metrics: dict[str, Metric],
        source: str,
        arm: str = "",
    ):
        if not root.exists():
            print(f"  ! {source}: {root} does not exist, skipping")
            return []
        loaded, failed = load_combos(root, layout, metrics)
        failures.extend(_normalise_failures(failed, source, arm))
        return loaded

    for combo in _load(
        args.baseline_root, BASELINE_LAYOUT, BASELINE_METRICS, "atari57_baseline"
    ):
        arm = f"relu_{combo.params['update']}"
        combos[(combo.game, arm)] = combo

    for combo in _load(
        args.count_layer_root,
        COUNT_LAYER_LAYOUT,
        COUNTS_METRICS,
        "atari57_count_layer_sweep",
        "fta_qlearning",
    ):
        if (
            combo.params.get("position") != args.count_layer_position
            or combo.beta != args.count_layer_beta
            or combo.params.get("next") != args.count_layer_next
        ):
            continue
        combos[(combo.game, "fta_qlearning")] = combo

    for combo in _load(
        args.sarsa_root,
        SARSA_LAYOUT,
        COUNTS_METRICS,
        "atari57_seperate_heads_sarsa_sweep",
        "fta_sarsa",
    ):
        if (
            combo.beta != args.sarsa_beta
            or combo.params.get("gamma_i") != args.sarsa_gamma_i
            or combo.params.get("epsilon") != args.sarsa_epsilon
        ):
            continue
        combos[(combo.game, "fta_sarsa")] = combo

    return combos, failures


# ----------------------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------------------


def trimmed_mean(values: np.ndarray, lo: float = 0.25, hi: float = 0.75) -> float:
    """Mean of the values whose rank falls in [lo, hi). NaN-aware.

    A local copy rather than an import from plot_atari57_sarsa_summary.py, whose own
    docstring says it is the predecessor of plot_atari57_sarsa_figures.py and can be
    deleted -- the same call that script makes about _score_rows.
    """
    finite = np.sort(np.asarray(values, dtype=float)[np.isfinite(values)])
    if finite.size == 0:
        return float("nan")
    ranks = np.arange(finite.size) / finite.size
    kept = finite[(ranks >= lo) & (ranks < hi)]
    return float(kept.mean()) if kept.size else float(finite.mean())


def hns_score(score: float, reference: tuple[float, float] | None) -> float:
    """100*(score - random)/(human - random). The denominator is positive by construction,
    so unlike a ratio to a control this is well defined at zero and negative scores."""
    if reference is None or not np.isfinite(score):
        return float("nan")
    random_, human = reference
    return 100.0 * (score - random_) / (human - random_)


def effects(iqm_by_arm: dict[str, float]) -> list[tuple[str, float]]:
    """The two main effects and the interaction, on the aggregate HNS scale.

    A main effect is averaged over the other factor's two levels, which is what makes it a
    main effect rather than one slice of the table; the interaction is the difference of
    the two SARSA-minus-Q differences, i.e. how much the bootstrap rule is worth
    *differently* under FTA than under relu.
    """
    g = iqm_by_arm.get
    rq, rs = g("relu_qlearning", np.nan), g("relu_sarsa", np.nan)
    fq, fs = g("fta_qlearning", np.nan), g("fta_sarsa", np.nan)
    return [
        ("FTA − relu", 0.5 * ((fq - rq) + (fs - rs))),
        ("SARSA − Q", 0.5 * ((rs - rq) + (fs - fq))),
        ("interaction", (fs - fq) - (rs - rq)),
    ]


def coverage_rows(
    combos: dict[tuple[str, str], Combo],
    games: Sequence[str],
    expected_games: Sequence[str],
    min_seeds: int,
) -> list[dict]:
    """What is missing or thin, so a figure is never read as complete when it is not.

    Modelled on plot_atari57_sarsa_figures.coverage_rows, which cannot be imported: it is
    keyed on (game, beta) cells and calls Cell.label().
    """
    rows: list[dict] = []
    present = {g for g, _ in combos}
    for game in sorted(set(expected_games) - present):
        rows.append({"kind": "absent_game", "game": game, "arm": "", "detail": "no runs"})
    for game in games:
        for arm in ARMS:
            combo = combos.get((game, arm.key))
            if combo is None:
                rows.append(
                    {"kind": "absent_arm", "game": game, "arm": arm.key, "detail": "no runs"}
                )
            elif combo.n_seeds < min_seeds:
                rows.append(
                    {
                        "kind": "thin_arm",
                        "game": game,
                        "arm": arm.key,
                        "detail": f"{combo.n_seeds} seeds, expected {min_seeds}",
                    }
                )
    return rows


# ----------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------

MILLIONS = FuncFormatter(lambda v, _: f"{v / 1e6:.0f}M")


def _headroom(ax, values, fraction: float = 0.18) -> None:
    """Leave room above the tallest bar for its value label, and below for a negative one.

    Without this the annotations collide with the panel title, and on the per-game figure
    the legend sits on top of the highest group.
    """
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not finite.size:
        return
    lo, hi = float(finite.min()), float(finite.max())
    span = max(hi, 0.0) - min(lo, 0.0)
    if span <= 0:
        return
    if hi > 0:
        ax.set_ylim(top=hi + fraction * span)
    if lo < 0:
        ax.set_ylim(bottom=lo - fraction * span)


def curve_band(
    combo: Combo, args: "Args", rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """(x, mean, lo, hi) for one arm's smoothed curve, bands a 95% percentile bootstrap.

    The band is evaluated at --curve-points subsampled positions with one set of draws
    reused across all of them: drawing fresh seeds at each x would make neighbouring points
    independent and give a jagged ribbon rather than a band around a curve. Taken from
    plot_atari57_sarsa_figures.plot_curve_grid, which does the same thing per beta.
    """
    curves = np.asarray(combo.curves("extrinsic"), dtype=np.float64)
    smoothed = np.stack([rolling_mean(row, args.smooth) for row in curves])
    positions = np.unique(
        np.linspace(0, combo.steps.shape[0] - 1, args.curve_points).astype(int)
    )
    sub = smoothed[:, positions]
    # Count-normalised rather than nanmean: the EMA is NaN until a game's first episode
    # ends, so the earliest subsampled position is an all-NaN column and nanmean would warn
    # about it. The value is NaN either way and is dropped below.
    valid = np.isfinite(sub)
    counts = valid.sum(axis=0)
    mean = np.divide(
        np.where(valid, sub, 0.0).sum(axis=0),
        counts,
        out=np.full(counts.shape, np.nan, dtype=float),
        where=counts > 0,
    )
    n = sub.shape[0]
    draws = rng.integers(0, n, size=(args.n_boot, n))
    # Counts are per (draw, position): a resample can pick only the NaN seeds at a position
    # where other seeds are finite, so a per-column mask is not enough.
    drawn_counts = valid[draws].sum(axis=1)
    boots = np.divide(
        np.where(valid, sub, 0.0)[draws].sum(axis=1),
        drawn_counts,
        out=np.full(drawn_counts.shape, np.nan, dtype=float),
        where=drawn_counts > 0,
    )
    usable = np.isfinite(boots).any(axis=0)
    lo = np.full(boots.shape[1], np.nan)
    hi = np.full(boots.shape[1], np.nan)
    if usable.any():
        lo[usable] = np.nanpercentile(boots[:, usable], 2.5, axis=0)
        hi[usable] = np.nanpercentile(boots[:, usable], 97.5, axis=0)
    finite = np.isfinite(mean) & usable
    if not finite.any():
        return None
    return combo.steps[positions][finite], mean[finite], lo[finite], hi[finite]


def _draw_arms(ax, per_arm: dict[str, tuple], colours: dict[str, str], linewidth: float):
    """Plot the arms present on one axes, in ARMS order so z-order and legend agree."""
    lowest = np.inf
    for arm in ARMS:
        band = per_arm.get(arm.key)
        if band is None:
            continue
        xs, mean, lo, hi = band
        ax.fill_between(
            xs, lo, hi, color=colours[arm.key], alpha=0.15, linewidth=0, zorder=2
        )
        ax.plot(
            xs,
            mean,
            color=colours[arm.key],
            linewidth=linewidth,
            linestyle=ARM_LINESTYLE[arm.update],
            zorder=3,
        )
        lowest = min(lowest, float(np.nanmin(lo)))
    return lowest


def _arm_patch_handles(colours: dict[str, str]):
    """Filled swatches, for the bar figures. A line handle with a dash pattern would
    advertise a linestyle the bars do not have."""
    return [
        plt.Rectangle((0, 0), 1, 1, color=colours[a.key], label=a.label) for a in ARMS
    ]


def _arm_handles(colours: dict[str, str], present: Sequence[str]):
    return [
        plt.Line2D(
            [],
            [],
            color=colours[a.key],
            linewidth=2.0,
            linestyle=ARM_LINESTYLE[a.update],
            label=a.label,
        )
        for a in ARMS
        if a.key in present
    ]


def plot_curve_grid(
    panels: list[dict], *, title: str, subtitle: str, path: Path, args: "Args"
) -> None:
    """One panel per game, four arms, each panel on its own y-axis."""
    theme = THEMES[args.theme]
    colours = ARM_COLOURS[args.theme]
    rows, cols = _grid_shape(len(panels), args.grid_cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(2.9 * cols, 2.3 * rows + 0.75), dpi=args.dpi, squeeze=False
    )
    fig.patch.set_facecolor(theme["surface"])
    present: set[str] = set()
    for index, ax in enumerate(axes.flat):
        if index >= len(panels):
            ax.axis("off")
            continue
        panel = panels[index]
        present |= set(panel["bands"])
        lowest = _draw_arms(ax, panel["bands"], colours, 1.2)
        _style_panel(ax, theme)
        ax.set_title(panel["game"], color=theme["ink"], fontsize=9, loc="left")
        ax.xaxis.set_major_formatter(MILLIONS)
        # Anchor at zero only when nothing drawn goes below it: several games score
        # negative, where a blanket floor would hide most of the curve.
        if np.isfinite(lowest) and lowest >= 0:
            ax.set_ylim(bottom=0)
        # The last *populated* panel of each column carries the x label, not merely the
        # last row: a short final row would otherwise leave those columns unlabelled.
        if index + cols < len(panels):
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Environment steps", color=theme["ink_secondary"], fontsize=8)
        if index % cols == 0:
            ax.set_ylabel("Episode return (EMA)", color=theme["ink_secondary"], fontsize=8)
    handles = _arm_handles(colours, sorted(present))
    # An inch-based strip: these grids run from one row to eight, and a fixed figure
    # fraction that clears the axis labels on a tall grid sits on top of them on a short one.
    strip = 0.42 / fig.get_size_inches()[1]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=9,
        labelcolor=theme["ink_secondary"],
        bbox_to_anchor=(0.5, 0.0),
    )
    top = _titles(fig, theme, title, subtitle)
    fig.tight_layout(rect=(0, strip, 1, top))
    _save(fig, path, args, theme)


def plot_game_curves(
    bands: dict[str, tuple], *, game: str, subtitle: str, path: Path, args: "Args"
) -> None:
    """The same four curves full size, for reading one game closely."""
    theme = THEMES[args.theme]
    colours = ARM_COLOURS[args.theme]
    fig, ax = plt.subplots(figsize=args.figsize, dpi=args.dpi)
    fig.patch.set_facecolor(theme["surface"])
    lowest = _draw_arms(ax, bands, colours, 1.8)
    _style_panel(ax, theme, xlabel="Environment steps", ylabel="Episode return (EMA)")
    ax.xaxis.set_major_formatter(MILLIONS)
    if np.isfinite(lowest) and lowest >= 0:
        ax.set_ylim(bottom=0)
    handles = _arm_handles(colours, sorted(bands))
    # "best" rather than a fixed corner: these are 57 different games and the free corner
    # is wherever that game's curves are not.
    ax.legend(
        handles=handles, loc="best", frameon=False, fontsize=10,
        labelcolor=theme["ink_secondary"],
    )
    top = _titles(
        fig, theme, f"{game} — episode return by activation and bootstrap rule", subtitle
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    _save(fig, path, args, theme)


def plot_hns_aggregate(
    iqm_by_arm: dict[str, float],
    mean_by_arm: dict[str, float],
    *,
    n_games: str,
    subtitle: str,
    path: Path,
    args: "Args",
) -> None:
    """Left: IQM human-normalised score per arm. Right: the 2x2 read off those four.

    No intervals, deliberately -- see the module docstring. The dashed rule on the right
    panel is zero, which is the only reference an effect needs.
    """
    theme = THEMES[args.theme]
    colours = ARM_COLOURS[args.theme]
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11.0, 5.0), dpi=args.dpi, gridspec_kw={"width_ratios": (4, 3)}
    )
    fig.patch.set_facecolor(theme["surface"])

    values = [iqm_by_arm.get(a.key, np.nan) for a in ARMS]
    left.bar(
        np.arange(len(ARMS)),
        values,
        width=0.68,
        color=[colours[a.key] for a in ARMS],
        zorder=3,
        linewidth=0,
    )
    left.axhline(0.0, color=theme["ink_secondary"], linewidth=1.0, zorder=2)
    _style_panel(left, theme, ylabel="IQM human-normalised score [%]")
    left.set_xticks(np.arange(len(ARMS)))
    left.set_xticklabels([a.label for a in ARMS], fontsize=9, rotation=20, ha="right")
    for i, v in enumerate(values):
        if np.isfinite(v):
            left.annotate(
                f"{v:.0f}",
                (i, v),
                textcoords="offset points",
                xytext=(0, 4 if v >= 0 else -12),
                ha="center",
                fontsize=9,
                color=theme["ink_secondary"],
            )
    _headroom(left, values)
    left.set_title(
        f"IQM over {n_games} games", color=theme["ink_secondary"], fontsize=10, loc="left"
    )

    names, effect_values = zip(*effects(iqm_by_arm))
    # One neutral ink for all three: these are contrasts, not arms, and painting them in
    # arm colours would imply an arm each.
    right.bar(
        np.arange(len(names)),
        effect_values,
        width=0.58,
        color=theme["ink_secondary"],
        zorder=3,
        linewidth=0,
    )
    right.axhline(0.0, color=theme["ink_secondary"], linewidth=1.0, zorder=2)
    _style_panel(right, theme, ylabel="effect on IQM HNS [pp]")
    right.set_xticks(np.arange(len(names)))
    right.set_xticklabels(names, fontsize=9, rotation=20, ha="right")
    for i, v in enumerate(effect_values):
        if np.isfinite(v):
            right.annotate(
                f"{v:+.0f}",
                (i, v),
                textcoords="offset points",
                xytext=(0, 4 if v >= 0 else -12),
                ha="center",
                fontsize=9,
                color=theme["ink_secondary"],
            )
    _headroom(right, effect_values)
    right.set_title(
        "main effects and interaction",
        color=theme["ink_secondary"],
        fontsize=10,
        loc="left",
    )

    top = _titles(
        fig, theme, "Human-normalised score across Atari-57", subtitle
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    _save(fig, path, args, theme)


def plot_hns_per_game(
    ordered: Sequence[str],
    hns: dict[tuple[str, str], float],
    *,
    subtitle: str,
    path: Path,
    args: "Args",
) -> None:
    """Four grouped bars per game, ordered by relu x Q-learning.

    Symlog y with a linear +-10 band, as in plot_atari57_sarsa_figures.plot_improvement_bars:
    the range runs to several hundred percent while the interesting region is a few tens, and
    a linear axis lets one game flatten every other.
    """
    theme = THEMES[args.theme]
    colours = ARM_COLOURS[args.theme]
    bar_width = 0.8 / len(ARMS)
    # Figure width is set by the game count; height follows it so 57 games do not come out
    # as a letterbox strip. Capped, or the 4-bar groups swim in vertical space.
    fig_width = max(10.0, 0.42 * len(ordered) + 2.0)
    fig, ax = plt.subplots(
        figsize=(fig_width, min(9.0, max(5.6, 0.30 * fig_width))), dpi=args.dpi
    )
    fig.patch.set_facecolor(theme["surface"])
    x = np.arange(len(ordered))
    for i, arm in enumerate(ARMS):
        values = np.array([hns.get((g, arm.key), np.nan) for g in ordered])
        ax.bar(
            x + (i - (len(ARMS) - 1) / 2) * bar_width,
            values,
            width=bar_width * 0.92,  # a surface gap between adjacent bars
            color=colours[arm.key],
            label=arm.label,
            zorder=3,
            linewidth=0,
        )
    ax.axhline(0.0, color=theme["ink_secondary"], linewidth=1.0, zorder=2)
    ax.set_yscale("symlog", linthresh=10.0)
    _style_panel(ax, theme, xlabel="Game", ylabel="Human-normalised score [%]")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered, rotation=90, fontsize=7)
    ax.set_xlim(-0.8, len(ordered) - 0.2)
    _headroom(ax, list(hns.values()))
    top = _titles(fig, theme, "Human-normalised score by game", subtitle)
    # In the header band rather than inside the axes: at 57 games the tallest group reaches
    # the top of the plot, and an in-axes legend either covers it or costs a decade of
    # symlog headroom to clear.
    fig.legend(
        handles=_arm_patch_handles(colours),
        loc="upper right",
        bbox_to_anchor=(0.995, 1 - 0.18 / fig.get_size_inches()[1]),
        ncol=len(ARMS),
        frameon=False,
        fontsize=9,
        labelcolor=theme["ink_secondary"],
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    _save(fig, path, args, theme)


# ----------------------------------------------------------------------------------------
# CLI and driver
# ----------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Args:
    baseline_root: Path = Path("data/atari57_baseline")
    """Pure PQN, all-relu. <game>/{qlearning,sarsa}/seed_*/."""
    sarsa_root: Path = Path("data/atari57_seperate_heads_sarsa_sweep")
    """Separate-heads SARSA sweep. Locally this tree is the `_partial` copy; pass it."""
    count_layer_root: Path = Path("data/atari57_count_layer_sweep")
    """Count-layer sweep, the FTA x Q-learning arm at beta=0."""
    output_dir: Path = Path("graphs/atari57_activation_update_2x2")
    """Summaries go to <output_dir>/<score>/, per-game curves to <output_dir>/games/."""
    hns_csv: Path = Path("atari_hns.csv")
    """Random and human reference scores, one row per game."""
    games_file: Path = Path("atari57_games.txt")
    """Newline-separated game list, used only to report games with no data at all."""

    sarsa_beta: str = "0.0"
    """Which cell of the SARSA sweep is the null-bonus control."""
    sarsa_gamma_i: str = "0.9"
    """At beta=0 the intrinsic head is frozen, so the sweep only ran this one gamma_I."""
    sarsa_epsilon: str = "0.001"
    """AtariConfig's default epsilon_end, which is what the other three arms ran at."""
    count_layer_position: str = "conv2"
    """FTA position. conv2 is the one the SARSA sweep also uses."""
    count_layer_beta: str = "0.0"
    count_layer_next: str = "0.0"

    scores: tuple[Literal["final", "auc"], ...] = ("final", "auc")
    """Scoring rules; each gets its own output tree."""
    game_set: Literal["common", "any"] = "common"
    """common: only games where all four arms have at least one seed, so the arms are
    averaged over one game set. any: every game with any data, which widens coverage at the
    cost of comparing arms averaged over different games."""
    games: tuple[str, ...] = ()
    """Restrict to these games. Empty means every game on disk."""
    min_seeds: int = 5
    """Below this an arm is reported as thin in coverage.csv."""
    final_frac: float = 0.02
    """Fraction of the run averaged for the "final" score. Matches the sibling scripts."""
    smooth: int = 200
    """Rolling-mean window in updates. Matches the sibling scripts."""
    n_boot: int = 2000
    """Bootstrap resamples for the curve bands."""
    curve_points: int = 300
    """x positions the bands are evaluated at. The curves are already smoothed, so
    bootstrapping all 24,414 updates buys nothing and costs a great deal of memory."""
    boot_seed: int = 0
    """Seeds the bootstrap RNG so figures are reproducible."""
    theme: Literal["light", "dark"] = "light"
    dpi: int = 300
    grid_cols: int = 8
    """Columns in the per-game panel grid."""
    figsize: tuple[float, float] = (10.0, 6.0)
    """Single-game figure size."""
    per_game_figures: bool = True
    """The one-file-per-game curves, written to <output_dir>/games/<game>/. Outside the
    score trees because no scoring rule enters a curve figure. Turn off for a quick pass."""


def main(args: Args) -> None:
    rng = np.random.default_rng(args.boot_seed)
    hns_reference = load_hns(args.hns_csv)
    expected = [g.strip() for g in open(args.games_file) if g.strip()]

    combos, failures = load_arms(args)
    if not combos:
        raise SystemExit(
            "no runs found under any of "
            f"{args.baseline_root}, {args.count_layer_root}, {args.sarsa_root}"
        )

    games = sorted({g for g, _ in combos})
    if args.games:
        games = [g for g in games if g in set(args.games)]
        if not games:
            raise SystemExit(f"no runs for games {sorted(args.games)}")
    # Coverage is reported over every game with data, *before* the common-set filter:
    # a game dropped for a missing arm is exactly the thing worth knowing about, and
    # reporting only the survivors would leave it invisible.
    with_data = list(games)
    if args.game_set == "common":
        games = [g for g in games if all((g, a.key) in combos for a in ARMS)]
        if not games:
            raise SystemExit(
                "no game has all four arms; re-run with --game-set any to plot what is "
                "on disk, and see coverage.csv"
            )
    dropped = [g for g in with_data if g not in set(games)]
    print(
        f"{len(games)} games ({args.game_set}), "
        f"{sum(1 for g, _ in combos if g in games)} arm-cells"
    )
    if dropped:
        print(
            f"  {len(dropped)} games have data but not all four arms, excluded by "
            f"--game-set common: {', '.join(dropped)}"
        )

    if failures:
        write_rows(
            failures,
            ("source", "game", "arm", "cell", "seed", "error"),
            args.output_dir / "unreadable_runs.csv",
        )
        print(f"  {len(failures)} unreadable runs -> unreadable_runs.csv")
    cover = coverage_rows(combos, with_data, expected, args.min_seeds)
    if cover:
        write_rows(
            cover, ("kind", "game", "arm", "detail"), args.output_dir / "coverage.csv"
        )
        print(f"  {len(cover)} coverage notes -> coverage.csv")

    # Scored once per (game, arm); seed_scores returns both rules from one smoothing pass,
    # which is the expensive part.
    scores = {
        key: seed_scores(combo, args.final_frac, args.smooth)
        for key, combo in combos.items()
        if key[0] in games
    }

    # Bands are independent of the score rule, so they are computed once rather than per
    # tree. This is the dominant cost of the whole script.
    bands: dict[tuple[str, str], tuple] = {}
    for game in games:
        for arm in ARMS:
            combo = combos.get((game, arm.key))
            if combo is None:
                continue
            band = curve_band(combo, args, rng)
            if band is not None:
                bands[(game, arm.key)] = band

    missing_hns = sorted({g for g in games if g not in hns_reference})
    if missing_hns:
        print(f"  ! no HNS reference for {len(missing_hns)} games: {', '.join(missing_hns)}")

    for score in args.scores:
        out = args.output_dir / score
        raw = {
            key: float(np.nanmean(v[score])) if np.isfinite(v[score]).any() else float("nan")
            for key, v in scores.items()
        }
        hns = {
            key: hns_score(value, hns_reference.get(key[0]))
            for key, value in raw.items()
        }

        summary = [
            {
                "game": game,
                "arm": arm.key,
                "activation": arm.activation,
                "update": arm.update,
                "n_seeds": combos[(game, arm.key)].n_seeds,
                "score": raw.get((game, arm.key), float("nan")),
                "hns": hns.get((game, arm.key), float("nan")),
            }
            for game in games
            for arm in ARMS
            if (game, arm.key) in combos
        ]
        write_rows(
            summary,
            ("game", "arm", "activation", "update", "n_seeds", "score", "hns"),
            out / "summary.csv",
        )

        iqm_by_arm, mean_by_arm, n_by_arm = {}, {}, {}
        for arm in ARMS:
            values = np.array(
                [hns[(g, arm.key)] for g in games if (g, arm.key) in hns], dtype=float
            )
            values = values[np.isfinite(values)]
            iqm_by_arm[arm.key] = trimmed_mean(values)
            mean_by_arm[arm.key] = float(values.mean()) if values.size else float("nan")
            n_by_arm[arm.key] = int(values.size)

        aggregate = [
            {
                "kind": "arm",
                "name": arm.key,
                "iqm": iqm_by_arm[arm.key],
                "mean": mean_by_arm[arm.key],
                "n_games": n_by_arm[arm.key],
            }
            for arm in ARMS
        ] + [
            {"kind": "effect", "name": name, "iqm": value, "mean": "", "n_games": ""}
            for name, value in effects(iqm_by_arm)
        ]
        write_rows(aggregate, ("kind", "name", "iqm", "mean", "n_games"), out / "aggregate.csv")

        counts = sorted(set(n_by_arm.values()))
        n_common = (
            str(counts[0]) if len(counts) == 1 else f"{counts[0]}–{counts[-1]}"
        )
        # One line, because _titles reserves vertical space per subtitle line and a
        # three-line block leaves a visible gap above the axes on the shorter figures.
        # "up to 5 seeds" rather than "5 seeds": thin arms are real, and named in
        # coverage.csv rather than papered over here.
        stem = (
            f"{score} score · {len(games)} game{'' if len(games) == 1 else 's'} "
            f"({args.game_set} set) · up to 5 seeds per "
            "arm · FTA at conv2 vs all-relu trunk · SARSA vs Q-learning λ-return"
        )
        plot_hns_aggregate(
            iqm_by_arm,
            mean_by_arm,
            n_games=n_common,
            subtitle=stem + "\nPoint estimates, no intervals — seed spread is on the curve figures",
            path=out / "hns_aggregate.png",
            args=args,
        )

        # One order for the figure, taken from relu x Q-learning ascending. Games without
        # that arm sort first rather than being dropped, so nothing disappears silently.
        def order_key(game: str) -> tuple[bool, float]:
            value = hns.get((game, "relu_qlearning"), float("nan"))
            return (np.isfinite(value), value if np.isfinite(value) else 0.0)

        ordered = sorted(games, key=order_key)
        plot_hns_per_game(
            ordered,
            hns,
            subtitle=stem + "\nOrdered by relu · Q-learning ascending · symlog y, linear within ±10 · no intervals",
            path=out / "hns_per_game.png",
            args=args,
        )

        panels = [
            {"game": g, "bands": {a.key: bands[(g, a.key)] for a in ARMS if (g, a.key) in bands}}
            for g in ordered
        ]
        plot_curve_grid(
            panels,
            title="Episode return by activation and bootstrap rule",
            subtitle=stem + "\nBands are 95% bootstrap CIs over seeds",
            path=out / "curve_grid_extrinsic.png",
            args=args,
        )

        print(f"  {score}: {out}")

    # Outside the score loop: a curve figure shows curves, and no scoring rule enters it.
    # Under <score>/ these would be 57 pairs of files differing only in a subtitle naming a
    # score that played no part in drawing them.
    if args.per_game_figures:
        note = (
            f"{len(games)} games ({args.game_set} set) · up to 5 seeds per arm · FTA at "
            "conv2 vs all-relu trunk · SARSA vs Q-learning λ-return"
            "\nBands are 95% bootstrap CIs over seeds"
        )
        for game in games:
            per_arm = {a.key: bands[(game, a.key)] for a in ARMS if (game, a.key) in bands}
            if not per_arm:
                continue
            plot_game_curves(
                per_arm,
                game=game,
                subtitle=note,
                path=args.output_dir / "games" / game / "extrinsic.png",
                args=args,
            )
        print(f"  per-game curves: {args.output_dir / 'games'}")


if __name__ == "__main__":
    main(tyro.cli(Args))
