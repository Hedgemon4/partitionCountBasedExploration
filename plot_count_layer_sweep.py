"""Learning-curve plots for the count-layer sweep.

Layout it expects (as produced by
job_scripts/atari/venture_spaceinvaders_count_layer_sweep):

    <root>/<game>/<position>/beta_<b>/next_<n>/seed_<s>/metrics.npz

A *combination* is one (position, beta, next_state_coef) triple, aggregated over
its seeds. Per game there are 45 combinations (3 positions x 5 betas x 3 coefs).

For each game and each ranking metric it writes:

    top10_extrinsic.png / top10_intrinsic.png      top 10 of 45
    <position>_top5_extrinsic.png / _intrinsic.png top 5 of 15, per position
    best_per_beta_extrinsic.png / _intrinsic.png   best of 9 at each beta
    ranking.csv                                    the table twin of the plots

The best-per-beta figures hold beta fixed and take the best (position,
next_state_coef) pair for it, so they isolate what beta is worth once the other
two hyperparameters are tuned for it. Their series are ordered by beta rather
than by rank, so a given beta keeps its colour across the two metrics.

Combinations are always ranked by the *extrinsic* metric; the intrinsic figures
show the same selection, so a pair of figures describes one set of runs.

One colour per legend entry, assigned in rank order. Note that past ~8 series a
categorical palette can no longer guarantee every pair is distinguishable (and
colourblind-safe), so on the 10-series figures the legend order -- not the hue
alone -- is what pins identity down.

Confidence bands are drawn on every figure by default, at a lower opacity once a
figure carries more than six lines so ten overlapping bands stay readable. Set
`--band-max-series 6` to drop them from the ten-line figures.

Note on intrinsic curves at beta=0: the count bonus is computed regardless of
beta and only multiplied by it in the TD target, so beta=0 runs still report a
non-zero intrinsic return. It is the bonus the agent would have received, not a
reward it acted on.
"""

from __future__ import annotations

import csv
import dataclasses
import math
import re
from pathlib import Path
from typing import Callable, Literal, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tyro
from matplotlib.ticker import FuncFormatter
from scipy import stats

EXTRINSIC_METRIC = "extrinsic_return_per_game_ema"
INTRINSIC_METRIC = "intrinsic_return_per_game_ema"
STEP_KEY = "env_step"


@dataclasses.dataclass(frozen=True)
class Metric:
    """One plottable series: where to read it and how to draw it."""

    key: str  # the metrics.npz key
    label: str  # y-axis label
    log: bool = False  # losses span orders of magnitude


# The two series every sweep in this repo has. A caller wanting more (the
# separate-value-head runs also log per-head Q values and losses) passes its own
# mapping to load_combos -- asking for a key that a sweep's runs predate would
# otherwise mark every one of them unreadable.
DEFAULT_METRICS = {
    "extrinsic": Metric(EXTRINSIC_METRIC, "Extrinsic return (EMA)"),
    "intrinsic": Metric(INTRINSIC_METRIC, "Intrinsic return (EMA)"),
}

# Count positions in network order (conv2 -> conv3 -> first MLP block).
POSITION_ORDER = ("conv2", "conv3", "mlp1")

# Ten-colour categorical series palette, listed explicitly rather than pulled
# from a colormap so the order is stable and obvious.
SERIES_COLOURS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
    },
}


@dataclasses.dataclass(frozen=True)
class Args:
    root_dir: Path = Path("data/venture_spaceinvaders_count_layer_sweep")
    """Sweep directory containing <game>/<position>/beta_*/next_*/seed_*/."""
    output_dir: Path = Path("graphs/venture_spaceinvaders_count_layer_sweep")
    """Figures are written to <output_dir>/<score>/<game>/."""
    scores: tuple[Literal["final", "auc"], ...] = ("final", "auc")
    """Ranking metrics to produce a full set of figures for."""
    top_k: int = 10
    """How many combinations in the per-game figures."""
    top_k_per_position: int = 5
    """How many combinations in the per-position figures."""
    final_frac: float = 0.1
    """Fraction of the run averaged for the "final" score."""
    smooth: int = 750
    """Rolling-mean window in updates. 1 disables smoothing."""
    band_max_series: int = 10
    """Cap above which 95% CI bands are dropped. Set to 6 to bare the top-10 figures."""
    theme: Literal["light", "dark"] = "light"
    dpi: int = 300
    figsize: tuple[float, float] = (10.0, 6.0)


@dataclasses.dataclass(frozen=True)
class Layout:
    """How a sweep's directory tree maps onto Combo fields.

    Sweeps of different shapes (count positions vs. a two-gamma grid) live under
    different path layouts, so the glob and its parse travel together rather than
    being hardcoded in load_combos. `dims` names the grouping levels other than
    game/beta/seed, in the order their columns should be reported.
    """

    glob: str
    pattern: re.Pattern
    dims: tuple[str, ...]


COUNT_LAYER_LAYOUT = Layout(
    glob="*/*/beta_*/next_*/seed_*/metrics.npz",
    pattern=re.compile(
        r"(?P<game>[^/]+)/(?P<position>[^/]+)/beta_(?P<beta>[^/]+)"
        r"/next_(?P<next>[^/]+)/seed_(?P<seed>[^/]+)$"
    ),
    dims=("position", "next"),
)


@dataclasses.dataclass
class Combo:
    """One (beta, *layout dims) cell, stacked over its seeds."""

    game: str
    beta: str
    params: dict[str, str]  # the layout's dims, e.g. {"position": "conv2"}
    steps: np.ndarray  # (T,)
    series: dict[str, np.ndarray]  # metric name -> (n_seeds, T), float32
    # aggregate() results, keyed by (metric, window, ci). Ranking, each figure the
    # combination appears in, and the CSV columns all ask for the same curves, so
    # without this the smoothing runs ~6x per combination.
    _cache: dict = dataclasses.field(default_factory=dict, repr=False)

    @property
    def n_seeds(self) -> int:
        return next(iter(self.series.values())).shape[0]

    # Named accessors for the count-layer layout's dims and the two universal
    # metrics, so callers written against them keep working now that both live
    # in dicts.
    @property
    def position(self) -> str:
        return self.params.get("position", "")

    @property
    def next_coef(self) -> str:
        return self.params.get("next", "")

    @property
    def extrinsic(self) -> np.ndarray:
        return self.series["extrinsic"]

    @property
    def intrinsic(self) -> np.ndarray:
        return self.series["intrinsic"]

    def label(self) -> str:
        return f"{self.position}  β={self.beta}  next={self.next_coef}"

    def curves(self, metric: str) -> np.ndarray:
        return self.series[metric]

    def aggregated(
        self, metric: str, window: int, ci: str = "t"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Memoised aggregate() for one of this combination's metrics."""
        key = (metric, window, ci)
        if key not in self._cache:
            self._cache[key] = aggregate(self.curves(metric), window, ci)
        return self._cache[key]


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """NaN-aware centred rolling mean that shrinks at the edges.

    Dividing by the count of real samples rather than the window width matters
    here: a plain convolution drags the first and last points toward zero, and
    the tail of the curve is exactly what the "final" score reads.
    """
    if window <= 1:
        return values
    kernel = np.ones(window)
    finite = np.isfinite(values)
    total = np.convolve(np.where(finite, values, 0.0), kernel, mode="same")
    count = np.convolve(finite.astype(float), kernel, mode="same")
    return np.divide(
        total, count, out=np.full(total.shape, np.nan, dtype=float), where=count > 0
    )


def load_combos(
    root: Path,
    layout: Layout = COUNT_LAYER_LAYOUT,
    metrics: dict[str, Metric] = DEFAULT_METRICS,
) -> tuple[list[Combo], list[dict[str, str]]]:
    """Read every run under `root`, grouped into combinations.

    Also returns the runs that could not be read, so a partial sweep is reported
    rather than silently averaged over fewer seeds. `layout` selects the on-disk
    path shape and `metrics` the series to load; both default to the count-layer
    sweeps' choices.
    """
    # key is (game, beta, *dim values) -- a tuple so the grouping stays hashable
    # and sorts in a stable order.
    grouped: dict[tuple[str, ...], list[tuple[str, Path]]] = {}
    for metrics_path in sorted(root.glob(layout.glob)):
        match = layout.pattern.search(
            metrics_path.parent.relative_to(root).as_posix()
        )
        if match is None:
            continue
        key = (match["game"], match["beta"], *(match[d] for d in layout.dims))
        grouped.setdefault(key, []).append((match["seed"], metrics_path))

    combos: list[Combo] = []
    failures: list[dict[str, str]] = []
    for key, entries in sorted(grouped.items()):
        params = dict(zip(layout.dims, key[2:]))
        steps: list[np.ndarray] = []
        series: dict[str, list[np.ndarray]] = {name: [] for name in metrics}
        for seed, path in sorted(entries):
            try:
                with np.load(path) as data:
                    # Read every metric before appending any, so a run missing one
                    # key is recorded as a failure rather than left half-loaded
                    # with its series out of step with each other.
                    loaded = {
                        name: np.asarray(data[metric.key], dtype=np.float32)
                        for name, metric in metrics.items()
                    }
                    # Keep the on-disk float32; aggregate() upcasts for the maths.
                    steps.append(np.asarray(data[STEP_KEY], dtype=np.float64))
                    for name, values in loaded.items():
                        series[name].append(values)
            except (OSError, KeyError, ValueError) as error:
                failures.append(
                    {
                        "game": key[0],
                        "beta": key[1],
                        **params,
                        "seed": seed,
                        "error": type(error).__name__,
                    }
                )
        if not steps:
            continue
        # Seeds can differ in length if a run was cut short; use the common prefix.
        length = min(arr.shape[0] for arrs in series.values() for arr in arrs)
        combos.append(
            Combo(
                game=key[0],
                beta=key[1],
                params=params,
                steps=steps[0][:length],
                series={
                    name: np.stack([a[:length] for a in arrs])
                    for name, arrs in series.items()
                },
            )
        )
    return combos, failures


def aggregate(
    curves: np.ndarray, window: int, ci: str = "t"
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth each seed, then return (mean, half-width of a 95% CI).

    The EMA metrics are NaN until a seed's first episode ends, so everything is
    NaN-aware.

    `ci` picks the multiplier on the standard error. "t" is the Student-t
    interval, which is the one that actually attains 95% coverage at these seed
    counts (2.776 at n=5). "normal" is the familiar 1.96, i.e. the large-sample
    limit; at n=5 it is ~30% narrower and covers about 91%, not 95%. It is
    offered because it is the convention elsewhere, not because it is better
    here.

    Curves are held as float32 (their on-disk dtype) but the arithmetic runs in
    float64, so halving the stored size does not move any result.
    """
    curves = np.asarray(curves, dtype=np.float64)
    smoothed = np.stack([rolling_mean(row, window) for row in curves])
    n_valid = np.sum(np.isfinite(smoothed), axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(smoothed, axis=0)
        std = np.nanstd(smoothed, axis=0, ddof=1)
    sem = np.divide(
        std,
        np.sqrt(n_valid),
        out=np.full(std.shape, np.nan, dtype=float),
        where=n_valid > 1,
    )
    if ci == "normal":
        critical = 1.959963984540054
    else:
        critical = stats.t.ppf(0.975, max(smoothed.shape[0] - 1, 1))
    return mean, sem * critical


def score_combo(combo: Combo, score: str, final_frac: float, window: int) -> float:
    """Rank a combination by its extrinsic curve. Higher always wins.

    The band choice cannot change a ranking -- only the mean is read here -- so
    this always asks for the default aggregation and shares its cache entry.
    """
    mean, _ = combo.aggregated("extrinsic", window)
    finite = np.isfinite(mean)
    if not finite.any():
        return float("-inf")
    if score == "final":
        tail = math.ceil(mean.shape[0] * final_frac)
        return float(np.nanmean(mean[-tail:]))
    # Normalising by the x-range keeps runs of unequal length comparable.
    steps, values = combo.steps[finite], mean[finite]
    if steps[-1] == steps[0]:
        return float(values.mean())
    return float(np.trapezoid(values, steps) / (steps[-1] - steps[0]))


def _style_axes(ax, theme: dict, ylabel: str) -> None:
    ax.set_facecolor(theme["surface"])
    ax.grid(True, color=theme["grid"], linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=theme["muted"], labelsize=9)
    ax.set_xlabel("Environment steps", color=theme["ink_secondary"], fontsize=10)
    ax.set_ylabel(ylabel, color=theme["ink_secondary"], fontsize=10)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _: "0" if v == 0 else f"{v / 1e6:,.0f}M")
    )


def _format_endpoint(value: float) -> str:
    """Enough significant figures to be informative at any magnitude.

    Returns run to thousands and TD losses to hundredths of a unit, so the plain
    thousands-separated integer that suits a score of 12,480 renders a loss of
    0.044 as a bare "0".
    """
    magnitude = abs(value)
    if magnitude >= 100 or value == 0:
        return f"{value:,.0f}"
    if magnitude >= 1:
        return f"{value:,.1f}"
    if magnitude >= 0.01:
        return f"{value:.3f}"
    return f"{value:.1e}"


def plot_curves(
    combos: Sequence[Combo],
    *,
    metric: str,
    title: str,
    subtitle: str,
    path: Path,
    args: Args,
    label_fn: Callable[[Combo], str] = Combo.label,
    metrics: dict[str, Metric] = DEFAULT_METRICS,
    hline: tuple[float, str] | None = None,
) -> None:
    """One axes, one file, one colour per series in rank order.

    label_fn lets a caller drop hyperparameters that are constant across its sweep
    (e.g. next_state_coef, which never varies in the Atari-57 sweep) from the legend.
    `metrics` supplies the axis label and scale for `metric`. `hline` is an optional
    (y, label) reference line, for metrics with a meaningful threshold.
    """
    spec = metrics[metric]
    ci = getattr(args, "ci", "t")
    theme = THEMES[args.theme]
    show_band = len(combos) <= args.band_max_series
    # Ten overlapping bands saturate at the opacity that suits five, so fade them.
    band_alpha = 0.12 if len(combos) <= 6 else 0.07
    fig, ax = plt.subplots(figsize=args.figsize, dpi=args.dpi)
    fig.patch.set_facecolor(theme["surface"])

    if hline is not None:
        y, hlabel = hline
        ax.axhline(
            y,
            color=theme["ink_secondary"],
            linewidth=1.0,
            linestyle="--",
            label=hlabel,
            zorder=1,
        )

    endpoints: list[tuple[float, float]] = []
    lowest_mean = np.inf
    for rank, combo in enumerate(combos):
        mean, band = combo.aggregated(metric, args.smooth, ci)
        finite = np.isfinite(mean)
        if not finite.any():
            continue
        steps = combo.steps[finite]
        colour = SERIES_COLOURS[rank % len(SERIES_COLOURS)]
        endpoints.append((float(mean[finite][-1]), float(steps[-1])))
        lowest_mean = min(lowest_mean, float(np.min(mean[finite])))

        ax.plot(
            steps,
            mean[finite],
            color=colour,
            linewidth=2.0,
            solid_capstyle="round",
            solid_joinstyle="round",
            label=f"{label_fn(combo)}  (n={combo.n_seeds})",
            zorder=3 + (len(combos) - rank),
        )
        if show_band:
            ax.fill_between(
                steps,
                (mean - band)[finite],
                (mean + band)[finite],
                color=colour,
                alpha=band_alpha,
                linewidth=0,
                zorder=2,
            )

    # Label one endpoint only; a value on every line reads as noise. It is the
    # highest-finishing line rather than the top-ranked one, so the number is
    # unambiguously attached to the topmost curve at the right edge -- on the
    # intrinsic figures (ranked by extrinsic) those are not the same series.
    if endpoints:
        value, x = max(endpoints)
        ax.annotate(
            _format_endpoint(value),
            xy=(x, value),
            xytext=(6, 0),
            textcoords="offset points",
            color=theme["ink_secondary"],
            fontsize=9,
            va="center",
        )

    _style_axes(ax, theme, spec.label)
    if spec.log:
        # Losses fall across orders of magnitude, so a linear axis flattens the
        # whole tail against zero. A log axis has no floor to anchor.
        ax.set_yscale("log")
    elif lowest_mean >= 0:
        # Anchor the axis at zero only when nothing plotted goes below it. Several
        # Atari games score negative (pong, ice_hockey, tennis, double_dunk, skiing),
        # so a blanket floor of 0 would hide most of their curve. Where the metric
        # genuinely cannot go negative -- every intrinsic curve, and the many games
        # with non-negative scores -- the floor stays, and a CI band dipping below it
        # is clipped as the interval artefact it is.
        ax.set_ylim(bottom=0)
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
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
        frameon=False,
        labelcolor=theme["ink_secondary"],
        handlelength=2.2,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor=theme["surface"])
    plt.close(fig)


def plot_combo_metrics(
    combo: Combo,
    *,
    series: Sequence[tuple[str, float, str]],
    title: str,
    subtitle: str,
    ylabel: str,
    path: Path,
    args: Args,
    metrics: dict[str, Metric] = DEFAULT_METRICS,
    log: bool = False,
) -> None:
    """Several of one combination's metrics on one axes.

    The transpose of plot_curves: that draws one metric across many combinations,
    this draws many metrics of a single combination. Comparing the two value
    heads (or the two TD losses) of a *single* configuration is the readable
    view, because their magnitudes shift with beta -- overlaying five betas puts
    quantities of different scale on one axis.

    `series` is (metric name, scale, legend label). `scale` multiplies both the
    mean and its band, so a caller can draw e.g. beta * Q_i -- the quantity that
    actually competes with Q_e in the argmax -- beside the raw curves without
    that being a separately stored metric.
    """
    ci = getattr(args, "ci", "t")
    theme = THEMES[args.theme]
    fig, ax = plt.subplots(figsize=args.figsize, dpi=args.dpi)
    fig.patch.set_facecolor(theme["surface"])

    endpoints: list[tuple[float, float]] = []
    lowest_mean = np.inf
    for rank, (metric, scale, label) in enumerate(series):
        mean, band = combo.aggregated(metric, args.smooth, ci)
        mean, band = mean * scale, band * scale
        finite = np.isfinite(mean)
        if not finite.any():
            continue
        steps = combo.steps[finite]
        colour = SERIES_COLOURS[rank % len(SERIES_COLOURS)]
        endpoints.append((float(mean[finite][-1]), float(steps[-1])))
        lowest_mean = min(lowest_mean, float(np.min(mean[finite])))

        ax.plot(
            steps,
            mean[finite],
            color=colour,
            linewidth=2.0,
            solid_capstyle="round",
            solid_joinstyle="round",
            label=label,
            zorder=3 + (len(series) - rank),
        )
        ax.fill_between(
            steps,
            (mean - band)[finite],
            (mean + band)[finite],
            color=colour,
            alpha=0.12,
            linewidth=0,
            zorder=2,
        )

    # Unlike plot_curves, every series here is annotated: there are only two or
    # three, and the whole point of the figure is the gap between their levels.
    for value, x in endpoints:
        ax.annotate(
            _format_endpoint(value),
            xy=(x, value),
            xytext=(6, 0),
            textcoords="offset points",
            color=theme["ink_secondary"],
            fontsize=9,
            va="center",
        )

    _style_axes(ax, theme, ylabel)
    if log:
        ax.set_yscale("log")
    elif lowest_mean >= 0:
        ax.set_ylim(bottom=0)
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
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
        frameon=False,
        labelcolor=theme["ink_secondary"],
        handlelength=2.2,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor=theme["surface"])
    plt.close(fig)


def write_ranking_csv(rows: list[dict], path: Path) -> None:
    """The table twin: every number the figures encode, in text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "game",
                "rank",
                "position",
                "beta",
                "next_state_coef",
                "n_seeds",
                "score",
                "final_extrinsic",
                "final_intrinsic",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main(args: Args) -> None:
    combos, failures = load_combos(args.root_dir)
    if not combos:
        raise SystemExit(f"no runs found under {args.root_dir}")

    games = sorted({c.game for c in combos})
    print(f"{len(combos)} combinations across {len(games)} games: {', '.join(games)}")
    if failures:
        failure_path = args.output_dir / "failed_runs.csv"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        fields = ("game", "position", "beta", "next", "seed", "error")
        with open(failure_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields))
            writer.writeheader()
            writer.writerows(failures)
        print(f"  {len(failures)} unreadable runs -> {failure_path}")

    def tail_mean(combo: Combo, metric: str) -> float:
        mean, _ = combo.aggregated(metric, args.smooth)
        tail = math.ceil(mean.shape[0] * args.final_frac)
        return float(np.nanmean(mean[-tail:]))

    for score in args.scores:
        score_label = {"final": "final performance", "auc": "area under the curve"}[
            score
        ]
        for game in games:
            ranked = sorted(
                (c for c in combos if c.game == game),
                key=lambda c: score_combo(c, score, args.final_frac, args.smooth),
                reverse=True,
            )
            out = args.output_dir / score / game

            top = ranked[: args.top_k]
            for metric in ("extrinsic", "intrinsic"):
                plot_curves(
                    top,
                    metric=metric,
                    title=f"{game} — top {len(top)} combinations by {score_label}",
                    subtitle=(
                        f"{metric} return, mean of {top[0].n_seeds} seeds "
                        "with 95% CI"
                    ),
                    path=out / f"top{args.top_k}_{metric}.png",
                    args=args,
                )

            for position in POSITION_ORDER:
                best = [c for c in ranked if c.position == position][
                    : args.top_k_per_position
                ]
                if not best:
                    continue
                for metric in ("extrinsic", "intrinsic"):
                    plot_curves(
                        best,
                        metric=metric,
                        title=f"{game} / {position} — top {len(best)} by {score_label}",
                        subtitle=(
                            f"{metric} return, mean of {best[0].n_seeds} seeds "
                            "with 95% CI"
                        ),
                        path=out
                        / f"{position}_top{args.top_k_per_position}_{metric}.png",
                        args=args,
                    )

            # Best (position, next_state_coef) for each beta. Ordered by beta,
            # not by rank, so a beta keeps its colour between the two metrics.
            best_per_beta = [
                next(c for c in ranked if c.beta == beta)
                for beta in sorted({c.beta for c in ranked}, key=float)
            ]
            for metric in ("extrinsic", "intrinsic"):
                plot_curves(
                    best_per_beta,
                    metric=metric,
                    title=f"{game} — best combination per β by {score_label}",
                    subtitle=(
                        f"{metric} return, mean of {best_per_beta[0].n_seeds} "
                        "seeds with 95% CI · series ordered by β"
                    ),
                    path=out / f"best_per_beta_{metric}.png",
                    args=args,
                )

            write_ranking_csv(
                [
                    {
                        "game": game,
                        "rank": i + 1,
                        "position": c.position,
                        "beta": c.beta,
                        "next_state_coef": c.next_coef,
                        "n_seeds": c.n_seeds,
                        "score": round(
                            score_combo(c, score, args.final_frac, args.smooth), 3
                        ),
                        "final_extrinsic": round(tail_mean(c, "extrinsic"), 3),
                        "final_intrinsic": round(tail_mean(c, "intrinsic"), 3),
                    }
                    for i, c in enumerate(ranked)
                ],
                out / "ranking.csv",
            )
            print(f"  {score}/{game}: {len(list(out.glob('*.png')))} figures -> {out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
