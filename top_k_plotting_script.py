import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tyro
import yaml
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Dict
from pathlib import Path
import re
import warnings
import scipy.stats as stats

# The shared histogram helpers (count_histogram.py) are imported lazily — see
# _count_histogram_helpers() below. Keeping the import lazy means the reward
# curve plots, and in particular --group-seeds mode, still run on checkouts
# where count_histogram.py is not present.


def _count_histogram_helpers():
    """Lazily import the shared bin-usage histogram helpers.

    These keep every bin-usage plot across the project in the exact same
    format (stacked by action, seed std error bars, per-seed dots, percentage
    labels, outlier annotation). They are only needed by the optional count /
    evolution plots, so importing them lazily lets the reward-curve plots run
    without count_histogram.py on the path.
    """
    from count_histogram import (
        plot_histogram_with_actions,
        plot_neuron_summary,
        find_outlier_seeds,
        aggregate_counts,
    )

    return (
        plot_histogram_with_actions,
        plot_neuron_summary,
        find_outlier_seeds,
        aggregate_counts,
    )


# Snapshot subdirectories written by pqn_with_counts.py when
# count_save_timestep_interval > 0. Each directory contains one .npy file
# per saved boundary timestep, named {prefix}_timestep_{N}.npy.
SNAPSHOT_DIRS = {
    "counts": ("counts", "counts"),  # (dirname, filename prefix)
    "observation_counts": ("observation_counts", "observation_counts"),
    "grid_counts": ("grid_counts", "grid_discrete"),
}

_TIMESTEP_RE = re.compile(r"_timestep_(\d+)\.npy$")


@dataclass
class Args:
    """Analyze and plot specific runs from a large hyperparameter sweep."""

    root_dir: Path = Path("data/mountaincar_bounds_sweep")
    metric: str = "length_ema"
    intrinsic_metric: str = "intrinsic_return_ema"
    top_k: int = 10
    smooth: int = 1
    output_dir: Path = Path("graphs/mountaincar_bounds_sweep/top_10/last_1pct/")

    # --- SCORING PARAMETERS ---
    score_metric: str = "last_10pct"
    """How to rank runs for top-k selection. Options:
      last_10pct  - mean over the final 10%% of timesteps (original behaviour)
      auc         - area under the curve (trapezoidal, normalised by x-range)
      max         - peak mean value across all timesteps
    """

    # If you want to look at shorter episode length you need to have the data from largest to smallest value
    reverse: bool = False
    plot_extra: bool = False

    group_seeds: bool = False
    """Seed-grouping mode for sweeps that write ONE folder per
    (hyperparameter, seed) combination — e.g. data/freeway_sweep, where each
    metrics.npz holds 1-D arrays for a single seed.

    When set, run folders that share the same `beta` and
    `network.next_state_coef` are merged into a single multi-seed run (one row
    per seed), the merged runs are ranked by `score_metric`, and ONLY the
    extrinsic and intrinsic reward curves are produced (each with its own
    legend). The per-folder multi-seed behaviour is used when this is off."""

    next_state_coefs: Optional[Tuple[float, ...]] = None
    """--group-seeds only: restrict plotting to these `network.next_state_coef`
    values. None = all of them. Pass e.g. `--next-state-coefs 0.25 0.5 1.0` to
    drop the 0.0 (no next-state-prediction) baseline and compare only the runs
    that actually use the next-state-prediction loss."""

    # --- LEGEND PARAMETERS ---
    legend_vars: Optional[List[str]] = field(
        default_factory=lambda: [
            "beta",
            "epsilon_end",
            "epsilon_decay",
            "initial_learning_rate",
            "network.next_state_coef",
        ]
    )

    # --- FILTER PARAMETERS ---
    beta: Optional[float] = None
    activation: Optional[str] = None
    max_grad_norm: Optional[float] = None
    epsilon_end: Optional[float] = None
    hidden_size: Optional[int] = None
    learnable_norm: Optional[bool] = None
    total_time_steps: Optional[float] = None
    next_state_coef: Optional[float] = None

    # --- PLOT PARAMETERS ---
    y_lim: Optional[Tuple[float, float]] = (100, 200)
    """Y-axis limits for the extrinsic learning-curves plot, as (ymin, ymax).
    Leave unset to use matplotlib's autoscaling. Example: --y-lim -200 0
    """
    intrinsic_y_lim: Optional[Tuple[float, float]] = (0, 80)
    """Y-axis limits for the intrinsic-reward curves plot, as (ymin, ymax).
    Leave unset to use matplotlib's autoscaling.
    """


# network.next-state-coef


def moving_average(x: np.ndarray, w: int):
    if w <= 1:
        return x
    return np.apply_along_axis(
        lambda m: np.convolve(m, np.ones(w), "valid") / w, axis=1, arr=x
    )


def matches_filters(folder_path: Path, args: Args) -> bool:
    config_file = folder_path / "config.yaml"
    if not config_file.exists():
        return False
    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    except Exception:
        return False

    if args.beta is not None and config.get("beta") != args.beta:
        return False
    if (
        args.max_grad_norm is not None
        and config.get("max_grad_norm") != args.max_grad_norm
    ):
        return False
    if args.epsilon_end is not None and config.get("epsilon_end") != args.epsilon_end:
        return False

    network_config = config.get("network", {})
    if (
        args.hidden_size is not None
        and network_config.get("hidden_size") != args.hidden_size
    ):
        return False
    if (
        args.learnable_norm is not None
        and network_config.get("learnable_norm_params") != args.learnable_norm
    ):
        return False
    if (
        args.total_time_steps is not None
        and config.get("total_time_steps") != args.total_time_steps
    ):
        return False
    if (
        args.activation is not None
        and network_config.get("activation1", {}).get("type") != args.activation
    ):
        return False
    if (
        args.next_state_coef is not None
        and network_config.get("next_state_coef") != args.next_state_coef
    ):
        return False
    return True


def compute_score(
    values: np.ndarray, steps: np.ndarray, score_metric: str
) -> Tuple[float, np.ndarray]:
    """Return (scalar score for ranking, per-seed final values for the box plot).

    Per-seed final values are always the last-10% mean so the variance box
    plot remains consistent regardless of which ranking metric is chosen.

    score_metric options
    --------------------
    last_10pct  – mean over the final 10% of timesteps (original behaviour)
    auc         – area under the mean curve (trapezoidal), normalised by x-range
    max         – peak of the mean curve across all timesteps
    """
    last_10_percent = max(1, int(values.shape[1] * 0.1))
    final_values_per_seed = np.mean(values[:, -last_10_percent:], axis=1)

    if score_metric == "last_10pct":
        score = float(np.mean(final_values_per_seed))
    elif (
        score_metric == "auc"
    ):  # nanmean + NaN-strip handles seeds with missing early logs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_curve = np.nanmean(values, axis=0)
        valid = ~np.isnan(mean_curve)
        steps_v, curve_v = steps[valid], mean_curve[valid]
        x_range = steps_v[-1] - steps_v[0] if valid.sum() > 1 else 0
        score = (
            float(np.trapezoid(curve_v, steps_v) / x_range)
            if x_range > 0
            else float(np.nanmean(mean_curve))
        )
    elif score_metric == "max":
        mean_curve = np.nanmean(values, axis=0)
        score = float(np.nanmax(mean_curve))
    else:
        raise ValueError(
            f"Unknown score_metric '{score_metric}'. "
            "Choose from: last_10pct, auc, max"
        )
    return score, final_values_per_seed


def load_run_data(
    folder_path: Path, metric_name: str, score_metric: str = "last_10pct"
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    metrics_file = folder_path / "metrics.npz"
    if not metrics_file.exists():
        return None, None, None, None
    try:
        data = np.load(metrics_file)
        steps = data["env_step"][0]
        if metric_name not in data:
            return None, None, None, None
        values = data[metric_name]
        score, final_values_per_seed = compute_score(values, steps, score_metric)
        return steps, values, score, final_values_per_seed
    except Exception:
        return None, None, None, None


def get_config_value(config: dict, key_path: str) -> Any:
    val = config
    try:
        for part in key_path.split("."):
            if isinstance(val, list):
                val = val[int(part)]
            else:
                val = val.get(part)
        return val
    except (AttributeError, TypeError, KeyError, IndexError, ValueError):
        return None


def format_legend_label(folder_path: Path, legend_vars: Optional[List[str]]) -> str:
    if not legend_vars:
        return folder_path.name
    config_file = folder_path / "config.yaml"
    if not config_file.exists():
        return folder_path.name
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    label_parts = [
        f"{var.split('.')[-1]}: {get_config_value(config, var)}" for var in legend_vars
    ]
    return " | ".join(label_parts)


def plot_curves(
    results,
    metric_name,
    output_path,
    title,
    smooth_win,
    y_lim: Optional[Tuple[float, float]] = None,
    show_legend: bool = False,
    nan_aware: bool = False,
):
    """Plot one mean +/- 95% CI curve per result.

    Parameters
    ----------
    show_legend
        Draw the legend directly on the axes. Used by --group-seeds mode so
        each reward plot is self-contained (the per-folder path instead saves
        a separate legend image).
    nan_aware
        Aggregate across seeds with NaN-aware reductions. The freeway reward
        metrics are NaN until each seed finishes its first episode, so the
        leading timesteps must be skipped per-seed rather than poisoning the
        whole column.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.get_cmap("tab20", max(len(results), 1))
    for i, res in enumerate(results):
        if res["values"] is None:
            continue
        smoothed = moving_average(res["values"], smooth_win)
        steps = res["steps"][: smoothed.shape[1]]
        n_seeds = len(res["values"])
        if nan_aware:
            with warnings.catch_warnings():
                # All-NaN leading columns are expected (pre-first-episode).
                warnings.simplefilter("ignore", RuntimeWarning)
                mean = np.nanmean(smoothed, axis=0)
                std = np.nanstd(smoothed, axis=0, ddof=1)
            n_valid = np.sum(~np.isnan(smoothed), axis=0)
            std_err = std / np.sqrt(np.maximum(n_valid, 1))
        else:
            mean, std_err = np.mean(smoothed, axis=0), stats.sem(smoothed, axis=0)
        ci = std_err * stats.t.ppf((1 + 0.95) / 2.0, max(n_seeds - 1, 1))
        label = f"Rank {i + 1} | {res['name']}"
        ax.plot(steps, mean, label=label, color=colors(i), linewidth=2)
        ax.fill_between(steps, mean - ci, mean + ci, color=colors(i), alpha=0.15)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.7)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if show_legend:
        ax.legend(fontsize=8, loc="best")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return ax.get_legend_handles_labels()


def _resolve_reward_metrics(args: Args) -> Tuple[str, Optional[str]]:
    """Pick the extrinsic / intrinsic metric names for --group-seeds mode.

    `metric` defaults to "length_ema" (a MountainCar metric). For a reward
    sweep like freeway that default is meaningless, so fall back to
    "extrinsic_return_ema" unless the user passed something explicit.
    """
    ext_metric = args.metric
    if ext_metric in (None, "length_ema"):
        ext_metric = "extrinsic_return_ema"
    return ext_metric, args.intrinsic_metric


def _main_grouped(args: Args):
    """--group-seeds mode.

    Merges single-seed run folders by (beta, network.next_state_coef), ranks
    the merged multi-seed runs by `score_metric` (best reward first), and emits
    ONLY the extrinsic and intrinsic reward curves, each with its own legend.
    """
    from sweep_grouping import build_grouped_runs, filter_by_config, format_group_label

    ext_metric, int_metric = _resolve_reward_metrics(args)
    metric_names = [ext_metric] + ([int_metric] if int_metric else [])

    groups = build_grouped_runs(
        args.root_dir,
        group_keys=("beta", "network.next_state_coef"),
        metric_names=tuple(metric_names),
    )
    if not groups:
        print("No grouped runs found — check --root-dir.")
        return

    groups = filter_by_config(groups, "network.next_state_coef", args.next_state_coefs)
    if not groups:
        print(f"No groups left after --next-state-coefs {args.next_state_coefs}.")
        return

    results = []
    for g in groups:
        ext = g.metrics.get(ext_metric)
        if ext is None:
            continue
        score, final_vals = compute_score(ext, g.steps, args.score_metric)
        results.append(
            {
                "name": format_group_label(g),
                "steps": g.steps,
                "values": ext,
                "int_values": g.metrics.get(int_metric) if int_metric else None,
                "score": score,
                "final_seed_vals": final_vals,
            }
        )

    # Reward metrics: higher is better, so rank descending (best first).
    results.sort(key=lambda x: x["score"], reverse=True)
    top_results = results[: args.top_k]
    if not top_results:
        print("No matching groups.")
        return

    print(
        f"\n--- Top {len(top_results)} groups by {args.score_metric} "
        f"on {ext_metric} ---"
    )
    for i, res in enumerate(top_results, start=1):
        print(f"  Rank {i}: score={res['score']:.3f} | {res['name']}")

    # The dataclass y-limit defaults (100, 200)/(0, 80) are MountainCar
    # length-scale ranges; for a reward sweep let matplotlib autoscale unless
    # the user passed explicit limits.
    ext_ylim = None if args.y_lim == (100, 200) else args.y_lim
    int_ylim = None if args.intrinsic_y_lim == (0, 80) else args.intrinsic_y_lim

    # 1. Extrinsic reward curves
    plot_curves(
        top_results,
        ext_metric,
        args.output_dir / "filtered_learning_curves.png",
        f"Extrinsic Reward — Top {len(top_results)} groups by {args.score_metric}",
        args.smooth,
        y_lim=ext_ylim,
        show_legend=True,
        nan_aware=True,
    )
    print(f"  saved {args.output_dir / 'filtered_learning_curves.png'}")

    # 2. Intrinsic reward curves
    if int_metric:
        intrinsic_res = [
            {"name": r["name"], "steps": r["steps"], "values": r["int_values"]}
            for r in top_results
        ]
        plot_curves(
            intrinsic_res,
            int_metric,
            args.output_dir / "intrinsic_reward_curves.png",
            f"Intrinsic Reward — Top {len(top_results)} groups",
            args.smooth,
            y_lim=int_ylim,
            show_legend=True,
            nan_aware=True,
        )
        print(f"  saved {args.output_dir / 'intrinsic_reward_curves.png'}")


def main(args: Args):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Seed-grouping mode (one folder per (hyperparam, seed)) takes its own path
    # and only produces the extrinsic + intrinsic reward curves.
    if args.group_seeds:
        _main_grouped(args)
        return

    run_folders = [d for d in args.root_dir.iterdir() if d.is_dir()]

    all_filtered = []
    for folder in run_folders:
        if matches_filters(folder, args):
            steps, values, score, final_vals = load_run_data(
                folder, args.metric, args.score_metric
            )
            if steps is not None:
                all_filtered.append(
                    {
                        "name": format_legend_label(folder, args.legend_vars),
                        "folder": folder,
                        "steps": steps,
                        "values": values,
                        "score": score,
                        "final_seed_vals": final_vals,
                    }
                )

    all_filtered.sort(key=lambda x: x["score"], reverse=args.reverse)
    top_results = all_filtered[: args.top_k]
    if not top_results:
        print("No matching folders.")
        return

    # 1. Extrinsic Curves
    handles, labels = plot_curves(
        top_results,
        args.metric,
        args.output_dir / "filtered_learning_curves.png",
        f"Learning Curves (Top {len(top_results)} by {args.score_metric})",
        args.smooth,
        y_lim=args.y_lim,
    )

    # 2. Intrinsic Curves
    if args.intrinsic_metric is not None:
        intrinsic_res = []
        for res in top_results:
            s, v, _, _ = load_run_data(res["folder"], args.intrinsic_metric)
            intrinsic_res.append({"name": res["name"], "steps": s, "values": v})

        plot_curves(
            intrinsic_res,
            args.intrinsic_metric,
            args.output_dir / "intrinsic_reward_curves.png",
            f"Intrinsic Reward Curves (Top {len(top_results)} Runs)",
            args.smooth,
            y_lim=args.intrinsic_y_lim,
        )

    # 3. Box Plot for Variance
    fig_box, ax_box = plt.subplots(figsize=(12, 6))
    data_to_plot = [res["final_seed_vals"] for res in top_results]
    box_labels = [f"Rank {i + 1}" for i in range(len(top_results))]

    ax_box.boxplot(
        data_to_plot,
        labels=box_labels,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", color="blue"),
        medianprops=dict(color="red", linewidth=2),
    )
    ax_box.set_ylabel(f"Final {args.metric.replace('_', ' ').title()} (Last 10%)")
    ax_box.set_title("Variance Across Seeds")
    plt.xticks(rotation=45, ha="right")
    fig_box.savefig(
        args.output_dir / "filtered_seed_variance.png", dpi=300, bbox_inches="tight"
    )

    # 4. Legend
    fig_leg = plt.figure(figsize=(10, len(top_results) * 0.3))
    fig_leg.legend(
        handles,
        labels,
        loc="center",
        ncol=1,
        frameon=False,
        prop={"family": "monospace", "size": 9},
    )
    fig_leg.savefig(
        args.output_dir / "legend_filtered_learning_curves.png",
        dpi=300,
        bbox_inches="tight",
    )

    # 5. Count histograms — one per top-k run and a combined comparison
    _plot_count_histograms(top_results, args)

    if args.plot_extra:
        # 6. Evolution plots — per-timestep snapshots of FTA counts,
        #    observation counts, and grid-discrete representation.
        #    Each run that has `counts/`, `observation_counts/`, and/or
        #    `grid_counts/` subdirs contributes its own set of figures.
        _plot_snapshot_evolution(top_results, args)

        # 7. Goal-reach plot — for MountainCar-style envs, count visits to the
        #    goal region (position bins at or above goal_position) per top-k run.
        _plot_goal_reach_counts(top_results, args)


def _plot_count_histograms(top_results, args: Args):
    """For each top-k run, produce a bin-usage histogram in the same format
    as graphing_scripts/count_histogram.py (stacked by action, seed std error
    bars, per-seed dots, percentage labels, outlier annotation).

    A combined side-by-side plot with a shared y-axis is also produced so
    runs can be compared directly.
    """
    plot_histogram_with_actions, plot_neuron_summary, _, _ = _count_histogram_helpers()

    hist_dir = args.output_dir / "count_histograms"
    hist_dir.mkdir(parents=True, exist_ok=True)

    # Locate counts files up-front so we can skip runs that didn't log counts.
    # Newer runs (pqn_with_counts.py) save the run-aggregate counts as
    # `final_counts.npy` alongside a `counts/` directory with timestep snapshots.
    # Older runs used `counts.npy`/`counts.npz.npy`. Accept all of them.
    runs_with_counts = []
    for rank, res in enumerate(top_results, start=1):
        folder = res["folder"]
        candidates = [
            folder / "counts.npy",
            folder / "counts.npz.npy",
            folder / "final_counts.npy",
        ]
        counts_file = next((c for c in candidates if c.exists()), None)
        if counts_file is None:
            print(f"[hist] rank {rank} ({folder.name}): no counts file, skipping")
            continue
        runs_with_counts.append((rank, res, counts_file))

    if not runs_with_counts:
        print("[hist] No top-k runs have counts files; skipping histograms.")
        return

    # --- Per-run histograms ---------------------------------------------------
    outlier_report_lines = []
    for rank, res, counts_file in runs_with_counts:
        folder = res["folder"]
        counts = np.load(counts_file)
        title = (
            f"FTA bin usage  |  Rank {rank}  |  {folder.name}\n"
            f"{res['name']}  (score={res['score']:.2f})  "
            f"(error bars: seed std; stacks: per-action)"
        )
        out = hist_dir / f"hist_rank_{rank:02d}_{folder.name}.png"
        _, outliers, _ = plot_histogram_with_actions(
            counts,
            title=title,
            out_path=out,
            highlight=(rank == 1),
            show_outlier_box=False,
        )
        outlier_report_lines.append(
            f"Rank {rank} | {folder.name} | score={res['score']:.2f} | "
            f"outlier seeds: {[o['seed'] for o in outliers] if outliers else 'none'}"
        )
        # Per-neuron bin-activation summary for this run
        neuron_out = hist_dir / f"neuron_summary_rank_{rank:02d}_{folder.name}.png"
        plot_neuron_summary(
            counts,
            title_prefix=f"Rank {rank}  |  {folder.name}  (score={res['score']:.2f})",
            out_path=neuron_out,
        )

    # --- Combined comparison plot with shared y-axis --------------------------
    n = len(runs_with_counts)
    ncols = min(n, 5)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 6 * nrows), sharey=True, squeeze=False
    )
    for idx, (rank, res, counts_file) in enumerate(runs_with_counts):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        counts = np.load(counts_file)
        plot_histogram_with_actions(
            counts,
            title=f"Rank {rank}  |  {res['folder'].name}  (score={res['score']:.2f})",
            out_path=hist_dir,  # unused because ax is provided
            highlight=(rank == 1),
            ax=ax,
            show_legend=(idx == 0),
            show_outlier_box=False,
        )
        if c != 0:
            ax.set_ylabel("")

    # Hide any empty subplots (when n < nrows*ncols)
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    fig.suptitle(
        f"FTA bin usage across top {n} runs  "
        f"(error bars = seed std, dots = individual seeds, stacks = actions)",
        fontsize=12,
    )
    fig.tight_layout()
    combined_out = hist_dir / "hist_top_k_comparison.png"
    fig.savefig(combined_out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {combined_out}")

    # --- Outlier summary text file -------------------------------------------
    report_path = hist_dir / "outlier_seeds.txt"
    with open(report_path, "w") as f:
        f.write("Outlier seeds per top-k run\n")
        f.write(
            "(Flagged when a bin is 0, an (action,bin) cell is 0, or |z|>2 for any bin)\n\n"
        )
        f.write("\n".join(outlier_report_lines))
    print(f"  saved {report_path}")


# ---------------------------------------------------------------------------
# Evolution plots (timestep snapshots written by pqn_with_counts.py)
# ---------------------------------------------------------------------------


def _load_obs_space_bounds(
    folder: Path,
) -> Tuple[Optional[List[float]], Optional[List[float]], Optional[List[str]]]:
    """Best-effort lookup of the observation-space extents for a run.

    Used purely for nicer axis labels on the 2D heatmaps. Returns
    (low, high, axis_names) or (None, None, None) if we can't tell.
    """
    cfg_file = folder / "config.yaml"
    try:
        with open(cfg_file) as f:
            cfg = yaml.safe_load(f)
    except Exception:
        return None, None, None
    env = cfg.get("environment")
    if env == "MountainCar-v0":
        return [-1.2, -0.07], [0.6, 0.07], ["position", "velocity"]
    if env == "CartPole-v1":
        # Cartpole has 4 obs dims; we only plot 2D obs spaces below, so this
        # is just a heuristic for axis names if it ever applies.
        return None, None, ["obs0", "obs1"]
    return None, None, None


def _load_snapshot_history(folder: Path) -> Dict[str, Any]:
    """Collect per-timestep snapshots for a single run.

    Returns a dict with keys:
      timesteps        : sorted list[int] (the timestep boundaries saved)
      counts           : np.ndarray of shape (T, seeds, actions, neurons, bins), or None
      observation_counts: np.ndarray of shape (T, seeds, actions, num_bins, num_bins), or None
      grid_discrete    : np.ndarray of shape (T, seeds, num_grid_points, neurons, bins), or None

    Any entry is None if that directory is missing. Timesteps are taken from
    the `counts/` directory if present, otherwise from whichever of the three
    exists. All arrays are aligned to the same timestep list.
    """
    dir_info = {k: folder / dirname for k, (dirname, _) in SNAPSHOT_DIRS.items()}

    def _list_timesteps(d: Path, prefix: str) -> List[int]:
        if not d.is_dir():
            return []
        ts = []
        for p in d.iterdir():
            m = _TIMESTEP_RE.search(p.name)
            if m and p.name.startswith(prefix):
                ts.append(int(m.group(1)))
        return sorted(ts)

    counts_ts = _list_timesteps(dir_info["counts"], SNAPSHOT_DIRS["counts"][1])
    obs_ts = _list_timesteps(
        dir_info["observation_counts"], SNAPSHOT_DIRS["observation_counts"][1]
    )
    grid_ts = _list_timesteps(dir_info["grid_counts"], SNAPSHOT_DIRS["grid_counts"][1])

    # Use the counts timesteps as the canonical list when available.
    timesteps = counts_ts or obs_ts or grid_ts
    if not timesteps:
        return {
            "timesteps": [],
            "counts": None,
            "observation_counts": None,
            "grid_discrete": None,
        }

    def _stack(d: Path, prefix: str, ts_available: List[int]) -> Optional[np.ndarray]:
        if not d.is_dir() or not ts_available:
            return None
        # Only keep timesteps that exist for this particular directory.
        avail = set(ts_available)
        keep = [t for t in timesteps if t in avail]
        if not keep:
            return None
        arrs = [np.load(d / f"{prefix}_timestep_{t}.npy") for t in keep]
        return np.stack(arrs, axis=0)

    return {
        "timesteps": timesteps,
        "counts": _stack(dir_info["counts"], SNAPSHOT_DIRS["counts"][1], counts_ts),
        "observation_counts": _stack(
            dir_info["observation_counts"],
            SNAPSHOT_DIRS["observation_counts"][1],
            obs_ts,
        ),
        "grid_discrete": _stack(
            dir_info["grid_counts"], SNAPSHOT_DIRS["grid_counts"][1], grid_ts
        ),
    }


def _fmt_step(t: int) -> str:
    """Short human-friendly timestep label: 49152 -> '49k', 450560 -> '451k'."""
    if t >= 1_000_000:
        return (
            f"{t/1_000_000:.2f}M".rstrip("0").rstrip(".") + "M"
            if False
            else f"{t/1_000_000:.2f}M"
        )
    if t >= 1_000:
        return f"{round(t/1000)}k"
    return str(t)


def _plot_fta_evolution(
    counts_hist: np.ndarray,
    timesteps: List[int],
    rank: int,
    res: dict,
    out_dir: Path,
) -> None:
    """Per-timestep snapshots of FTA bin usage for one run.

    Two figures are produced:
      * hist_grid     : small-multiples grid of the standard per-timestep
                        stacked-by-action histogram (one column per timestep,
                        shared y-axis so magnitudes are comparable).
      * bin_heatmap   : a timestep x bin heatmap of the mean (across seeds)
                        per-seed bin total, summed over neurons and actions.
    """
    plot_histogram_with_actions, *_ = _count_histogram_helpers()

    T = counts_hist.shape[0]
    # --- (a) Grid of per-timestep histograms -----------------------------
    # Wrap the timesteps into a roughly-square 2D grid so each subplot is
    # wide enough for its percentage labels and stacked bars to be legible.
    ncols = int(np.ceil(np.sqrt(T)))
    nrows = int(np.ceil(T / ncols))
    subplot_w, subplot_h = 4.0, 3.6
    fig_w = subplot_w * ncols
    fig_h = subplot_h * nrows + 0.8  # extra room for suptitle + legend

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(fig_w, fig_h), sharey=True, squeeze=False
    )
    legend_handles = None
    legend_labels = None
    for i, t in enumerate(timesteps):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        plot_histogram_with_actions(
            counts_hist[i],
            title=f"t = {_fmt_step(t)}",
            out_path=out_dir,  # unused — ax provided
            highlight=(rank == 1 and i == T - 1),
            ax=ax,
            show_legend=False,  # single shared legend below
            show_outlier_box=False,
            compact=True,
        )
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        # Only label y-axis on the leftmost subplot of each row.
        if c != 0:
            ax.set_ylabel("")
        # Only label x-axis on the bottom row of the grid to save space.
        if r != nrows - 1:
            ax.set_xlabel("")

    # Hide any leftover empty axes (e.g., T=7 in a 3x3 grid).
    for j in range(T, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    # Figure-level legend above the grid.
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=len(legend_labels),
            fontsize=9,
            frameon=True,
            bbox_to_anchor=(0.5, 0.965),
        )

    fig.suptitle(
        f"FTA bin usage over training  |  Rank {rank}  |  {res['folder'].name}  "
        f"(score={res['score']:.2f})",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = out_dir / f"fta_hist_grid_rank_{rank:02d}_{res['folder'].name}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")

    # --- (b) Timestep x bin heatmap --------------------------------------
    # counts_hist: (T, seeds, actions, neurons, bins)
    # per-seed per-bin total (sum over neurons, actions) then mean over seeds
    per_seed_bin = counts_hist.sum(axis=(2, 3))  # (T, seeds, bins)
    mean_per_bin = per_seed_bin.mean(axis=1)  # (T, bins)

    n_bins = mean_per_bin.shape[1]
    fig2, ax2 = plt.subplots(figsize=(max(6, 0.45 * n_bins + 4), max(3, 0.4 * T + 1.5)))
    im = ax2.imshow(
        mean_per_bin,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
    )
    ax2.set_xticks(np.arange(n_bins))
    ax2.set_yticks(np.arange(T))
    ax2.set_yticklabels([_fmt_step(t) for t in timesteps])
    ax2.set_xlabel("FTA Bin Index")
    ax2.set_ylabel("Timestep")
    ax2.set_title(
        f"Mean per-seed bin count over training  |  Rank {rank}  |  "
        f"{res['folder'].name}  (score={res['score']:.2f})\n"
        "(summed over neurons & actions, averaged across seeds)",
        fontsize=10,
    )
    fig2.colorbar(im, ax=ax2, label="mean per-seed count")
    fig2.tight_layout()
    out_path2 = out_dir / f"fta_bin_heatmap_rank_{rank:02d}_{res['folder'].name}.png"
    fig2.savefig(out_path2, dpi=180, bbox_inches="tight")
    plt.close(fig2)
    print(f"  saved {out_path2}")


def _plot_obs_counts_evolution(
    obs_hist: np.ndarray,
    timesteps: List[int],
    rank: int,
    res: dict,
    out_dir: Path,
    folder: Path,
) -> None:
    """Per-timestep 2D state-space visitation maps for one run.

    Assumes a 2D observation space (the saved shape is
    (T, seeds, actions, num_bins_dim0, num_bins_dim1)). Produces:
      * overall     : one row of 2D heatmaps (summed over actions, averaged
                      over seeds) per timestep, shared colour scale.
      * per_action  : one row per action, one column per timestep, shared
                      colour scale per action (so intra-action trends are
                      visible without one hot action saturating others).
    """
    if obs_hist.ndim != 5:
        print(f"  [obs] unexpected shape {obs_hist.shape} — skipping 2D plots")
        return

    T, S, A, B0, B1 = obs_hist.shape
    low, high, axis_names = _load_obs_space_bounds(folder)
    xlabel = f"{axis_names[0]} bin" if axis_names else "obs dim 0 bin"
    ylabel = f"{axis_names[1]} bin" if axis_names else "obs dim 1 bin"
    extent_kwargs: Dict[str, Any] = {}
    if low is not None and high is not None and len(low) >= 2 and len(high) >= 2:
        extent_kwargs = dict(extent=[low[0], high[0], low[1], high[1]])
        xlabel = axis_names[0] if axis_names else xlabel
        ylabel = axis_names[1] if axis_names else ylabel

    # ---- (a) Overall visitation (summed over actions) -------------------
    overall = obs_hist.sum(axis=2).mean(axis=1)  # (T, B0, B1)
    vmax = float(overall.max()) if overall.max() > 0 else 1.0

    fig_w = max(4, 3.0 * T)
    fig, axes = plt.subplots(1, T, figsize=(fig_w, 3.6), squeeze=False, sharey=True)
    for i, t in enumerate(timesteps):
        ax = axes[0][i]
        # imshow with origin=lower. Our array is indexed [B0, B1] where B0 is
        # the first obs dim. Transpose so x=dim0, y=dim1.
        im = ax.imshow(
            overall[i].T,
            aspect="auto",
            origin="lower",
            cmap="magma",
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
            **extent_kwargs,
        )
        ax.set_title(f"t = {_fmt_step(t)}", fontsize=9)
        ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_ylabel(ylabel)
    fig.suptitle(
        f"State-space visitation over training  |  Rank {rank}  |  "
        f"{res['folder'].name}  (score={res['score']:.2f})\n"
        "(summed over actions, averaged across seeds)",
        fontsize=11,
    )
    # Leave headroom for the two-line suptitle before adding the colorbar
    # (colorbar reuses the axes layout, so adjust BEFORE calling it).
    fig.subplots_adjust(top=0.78, bottom=0.15)
    cbar = fig.colorbar(
        im, ax=axes.ravel().tolist(), shrink=0.85, label="mean visits per seed"
    )
    out_path = out_dir / f"obs_visitation_rank_{rank:02d}_{res['folder'].name}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")

    # ---- (b) Per-action visitation --------------------------------------
    per_action = obs_hist.mean(axis=1)  # (T, A, B0, B1)
    fig2, axes2 = plt.subplots(
        A, T, figsize=(fig_w, 3.2 * A), squeeze=False, sharex=True, sharey=True
    )
    # IMPORTANT: reserve gutter space *before* creating colorbars. Calling
    # `subplots_adjust` after `fig.colorbar` (or invoking colorbar with the
    # full row of axes via `ax=axes2[a].tolist()`) caused matplotlib to
    # rebuild the gridspec per row and place one of the per-row colorbars on
    # the seam *between* the second-to-last and last subplot — showing up as
    # a thin vertical bright streak inside the t≈399k panel.
    fig2.subplots_adjust(top=0.90, bottom=0.08, right=0.93)
    # Shared colour scale per action (each row) so the action-specific
    # distribution is visible even when one action dominates the counts.
    for a in range(A):
        row_vmax = float(per_action[:, a].max()) if per_action[:, a].max() > 0 else 1.0
        for i, t in enumerate(timesteps):
            ax = axes2[a][i]
            im = ax.imshow(
                per_action[i, a].T,
                aspect="auto",
                origin="lower",
                cmap="magma",
                vmin=0,
                vmax=row_vmax,
                interpolation="nearest",
                **extent_kwargs,
            )
            if a == 0:
                ax.set_title(f"t = {_fmt_step(t)}", fontsize=9)
            if i == 0:
                ax.set_ylabel(f"action {a}\n{ylabel}")
            if a == A - 1:
                ax.set_xlabel(xlabel)
        # Per-row colorbar — anchor to the rightmost subplot only so
        # matplotlib doesn't try to insert a colorbar column into the row's
        # gridspec.
        fig2.colorbar(
            im,
            ax=axes2[a, -1],
            shrink=0.85,
            pad=0.02,
            label=f"action {a}: mean visits per seed",
        )

    fig2.suptitle(
        f"State-space visitation per action over training  |  Rank {rank}  |  "
        f"{res['folder'].name}  (score={res['score']:.2f})\n"
        "(colour scale shared within each action row, averaged across seeds)",
        fontsize=11,
    )
    out_path2 = (
        out_dir / f"obs_visitation_per_action_rank_{rank:02d}_{res['folder'].name}.png"
    )
    fig2.savefig(out_path2, dpi=180, bbox_inches="tight")
    plt.close(fig2)
    print(f"  saved {out_path2}")


def _plot_grid_dominant_bin_evolution(
    grid_hist: np.ndarray,
    timesteps: List[int],
    rank: int,
    res: dict,
    out_dir: Path,
    folder: Path,
) -> None:
    """Dominant-bin maps across a grid of observation-space points.

    grid_hist shape: (T, seeds, num_grid_points, neurons, bins). Each grid
    point contributes a distribution over bins (how many neurons pick each
    bin). We average that distribution across seeds, argmax over bins, and
    plot the resulting dominant-bin index as a 2D map per snapshot.

    Assumes num_grid_points = G*G for a 2D observation space (which is how
    pqn_with_counts.py builds it via meshgrid on num_bins).
    """
    if grid_hist.ndim != 5:
        print(f"  [grid] unexpected shape {grid_hist.shape} — skipping")
        return

    T, S, G_total, N, B = grid_hist.shape
    side = int(round(np.sqrt(G_total)))
    if side * side != G_total:
        print(
            f"  [grid] {G_total} grid points is not a perfect square — "
            "can only plot 2D observation spaces; skipping"
        )
        return

    # Average the per-grid-point bin distribution across seeds, then argmax.
    mean_over_seeds = grid_hist.mean(axis=1)  # (T, G, N, B)
    bin_dist = mean_over_seeds.sum(axis=2)  # (T, G, B)
    dominant = bin_dist.argmax(axis=-1)  # (T, G)
    # Reshape to the 2D grid. meshgrid(indexing='ij') → first obs dim is the
    # outer axis, so reshape gives (dim0, dim1).
    dominant_2d = dominant.reshape(T, side, side)

    low, high, axis_names = _load_obs_space_bounds(folder)
    xlabel = axis_names[0] if axis_names else "obs dim 0"
    ylabel = axis_names[1] if axis_names else "obs dim 1"
    extent_kwargs: Dict[str, Any] = {}
    if low is not None and high is not None and len(low) >= 2 and len(high) >= 2:
        extent_kwargs = dict(extent=[low[0], high[0], low[1], high[1]])

    # Use a discrete colormap with B categories.
    cmap = plt.get_cmap("tab10" if B <= 10 else "tab20", B)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, B + 0.5, 1), cmap.N)

    fig_w = max(4, 3.0 * T)
    fig, axes = plt.subplots(1, T, figsize=(fig_w, 3.6), squeeze=False, sharey=True)
    for i, t in enumerate(timesteps):
        ax = axes[0][i]
        im = ax.imshow(
            dominant_2d[i].T,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            **extent_kwargs,
        )
        ax.set_title(f"t = {_fmt_step(t)}", fontsize=9)
        ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_ylabel(ylabel)

    fig.suptitle(
        f"FTA dominant-bin map over state space  |  Rank {rank}  |  "
        f"{res['folder'].name}  (score={res['score']:.2f})\n"
        "(per grid point, argmax over bins of the seed-mean bin distribution)",
        fontsize=11,
    )
    # Leave headroom for the two-line suptitle before adding the colorbar
    fig.subplots_adjust(top=0.78, bottom=0.15)
    cbar = fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        shrink=0.85,
        ticks=np.arange(B),
        label="dominant FTA bin",
    )
    cbar.ax.set_yticklabels([str(b) for b in range(B)])
    out_path = out_dir / f"grid_dominant_bin_rank_{rank:02d}_{res['folder'].name}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def _plot_grid_dominant_bin_per_seed(
    grid_hist: np.ndarray,
    timesteps: List[int],
    rank: int,
    res: dict,
    out_dir: Path,
    folder: Path,
) -> None:
    """Per-seed dominant-bin maps across the observation-space grid.

    Same computation as the aggregated version, but we skip the cross-seed
    average. For each (timestep, seed, grid_point), we take the argmax over
    bins of the sum-over-neurons counts, giving the modal FTA bin for that
    seed's representation of that state. One figure per timestep; within a
    figure, one small-multiples panel per seed.
    """
    if grid_hist.ndim != 5:
        print(f"  [grid-per-seed] unexpected shape {grid_hist.shape} — skipping")
        return

    T, S, G_total, N, B = grid_hist.shape
    side = int(round(np.sqrt(G_total)))
    if side * side != G_total:
        print(
            f"  [grid-per-seed] {G_total} grid points is not a perfect square — "
            "skipping"
        )
        return

    # Per-seed per-grid-point bin distribution (sum over neurons), then argmax.
    bin_dist = grid_hist.sum(axis=3)  # (T, S, G, B)
    dominant = bin_dist.argmax(axis=-1)  # (T, S, G)
    dominant_2d = dominant.reshape(T, S, side, side)  # (T, S, side, side)

    low, high, axis_names = _load_obs_space_bounds(folder)
    xlabel = axis_names[0] if axis_names else "obs dim 0"
    ylabel = axis_names[1] if axis_names else "obs dim 1"
    extent_kwargs: Dict[str, Any] = {}
    if low is not None and high is not None and len(low) >= 2 and len(high) >= 2:
        extent_kwargs = dict(extent=[low[0], high[0], low[1], high[1]])

    cmap = plt.get_cmap("tab10" if B <= 10 else "tab20", B)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, B + 0.5, 1), cmap.N)

    # Choose a near-square seed layout (favour more columns for wide screens).
    ncols = int(np.ceil(np.sqrt(S)))
    nrows = int(np.ceil(S / ncols))

    run_dir = out_dir / f"grid_dominant_bin_per_seed_rank_{rank:02d}_{folder.name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for ti, t in enumerate(timesteps):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(1.8 * ncols, 1.8 * nrows + 1.5),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        im = None
        for s in range(S):
            r, c = divmod(s, ncols)
            ax = axes[r][c]
            im = ax.imshow(
                dominant_2d[ti, s].T,
                aspect="auto",
                origin="lower",
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                **extent_kwargs,
            )
            ax.set_title(f"seed {s}", fontsize=8)
            ax.tick_params(labelsize=7)
            if r == nrows - 1:
                ax.set_xlabel(xlabel, fontsize=8)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=8)
        # Blank any leftover panels
        for s in range(S, nrows * ncols):
            r, c = divmod(s, ncols)
            axes[r][c].axis("off")

        fig.suptitle(
            f"FTA dominant-bin map per seed  |  Rank {rank}  |  "
            f"{folder.name}  (score={res['score']:.2f})  |  t = {_fmt_step(t)}",
            fontsize=11,
        )
        fig.subplots_adjust(top=0.90, bottom=0.08, right=0.88, hspace=0.45, wspace=0.15)
        cbar_ax = fig.add_axes([0.91, 0.10, 0.015, 0.78])
        cbar = fig.colorbar(
            im, cax=cbar_ax, ticks=np.arange(B), label="dominant FTA bin"
        )
        cbar.ax.set_yticklabels([str(b) for b in range(B)])

        out_path = run_dir / f"t_{t}.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
    print(f"  saved {T} per-seed figures under {run_dir}/")


def _plot_grid_full_distribution(
    grid_hist: np.ndarray,
    timesteps: List[int],
    rank: int,
    res: dict,
    out_dir: Path,
    folder: Path,
) -> None:
    """Full per-grid-point bin distribution at every snapshot.

    For each grid point on the observation-space grid, plot a small bar
    chart showing how the 64 neurons split their FTA activations across the
    10 bins (seed-averaged). This reveals the interior bins that the
    argmax-based dominant-bin view hides.

    One figure per snapshot timestep, saved under a per-run subdir.
    """
    if grid_hist.ndim != 5:
        print(f"  [grid-dist] unexpected shape {grid_hist.shape} — skipping")
        return

    T, S, G_total, N, B = grid_hist.shape
    side = int(round(np.sqrt(G_total)))
    if side * side != G_total:
        print(f"  [grid-dist] {G_total} grid points is not a perfect square — skipping")
        return

    # Mean across seeds, sum across neurons → per-grid-point bin count.
    bin_counts = grid_hist.mean(axis=1).sum(axis=2)  # (T, G, B)
    # Normalize each grid point's distribution to a probability so bars are
    # comparable across locations even when totals differ.
    totals = bin_counts.sum(axis=-1, keepdims=True).clip(min=1e-12)
    bin_probs = bin_counts / totals  # (T, G, B)
    # Reshape using the meshgrid(indexing="ij") convention: first axis is
    # obs dim 0 (position), second is obs dim 1 (velocity).
    bin_probs_2d = bin_probs.reshape(T, side, side, B)  # (T, pos, vel, bin)

    low, high, axis_names = _load_obs_space_bounds(folder)
    xlabel = axis_names[0] if axis_names else "obs dim 0"
    ylabel = axis_names[1] if axis_names else "obs dim 1"

    # Discrete bin colours (match the dominant-bin plot's palette).
    bin_cmap = plt.get_cmap("tab10" if B <= 10 else "tab20", B)
    bin_colors = [bin_cmap(b) for b in range(B)]

    # Global y-limit so heights are comparable across grid points and across
    # timesteps in the same run.
    y_max = float(bin_probs.max()) * 1.05 or 1.0

    out_subdir = out_dir / f"grid_full_distribution_rank_{rank:02d}_{folder.name}"
    out_subdir.mkdir(parents=True, exist_ok=True)

    for ti, t in enumerate(timesteps):
        fig, axes = plt.subplots(
            side,
            side,
            figsize=(1.1 * side + 1.5, 1.1 * side + 1.5),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        for vel_bin in range(side):
            for pos_bin in range(side):
                # Subplot row/col: position sweeps left-to-right (columns),
                # velocity sweeps bottom-to-top (rows) — so invert the row
                # index so "row 0" on the display is the highest velocity.
                ax = axes[side - 1 - vel_bin][pos_bin]
                probs = bin_probs_2d[ti, pos_bin, vel_bin]
                ax.bar(
                    np.arange(B), probs, color=bin_colors, edgecolor="none", width=0.9
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_ylim(0, y_max)
                # Thin grid at each bin boundary so the x-axis is readable
                for spine in ax.spines.values():
                    spine.set_color("#cccccc")
                    spine.set_linewidth(0.5)

        # Edge labels: bin indices on the bottom row and the left column.
        for pos_bin in range(side):
            axes[side - 1][pos_bin].set_xlabel(str(pos_bin), fontsize=7)
        for vel_bin in range(side):
            axes[side - 1 - vel_bin][0].set_ylabel(
                str(vel_bin),
                fontsize=7,
                rotation=0,
                ha="right",
                va="center",
                labelpad=6,
            )

        # Legend mapping bar colour -> bin index. One compact legend for the
        # whole figure, placed to the right of the subplot grid.
        handles = [plt.Rectangle((0, 0), 1, 1, color=bin_colors[b]) for b in range(B)]
        fig.legend(
            handles,
            [f"bin {b}" for b in range(B)],
            loc="center right",
            bbox_to_anchor=(1.0, 0.5),
            fontsize=8,
            frameon=False,
            title="FTA bin",
        )

        fig.supxlabel(f"{xlabel} bin", fontsize=10)
        fig.supylabel(f"{ylabel} bin", fontsize=10)
        fig.suptitle(
            f"Per-grid-point FTA bin distribution  |  Rank {rank}  |  "
            f"{folder.name}  (score={res['score']:.2f})  |  t = {_fmt_step(t)}\n"
            f"(seed-averaged; each bar chart = P(bin | neuron) at one "
            f"(position, velocity) grid point; y-axis shared: 0..{y_max:.2f})",
            fontsize=10,
        )
        fig.tight_layout(rect=[0.03, 0.03, 0.88, 0.92])
        out_path = out_subdir / f"t_{t}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  saved {T} full-distribution figures under {out_subdir}/")


def _plot_snapshot_evolution(top_results, args: Args) -> None:
    """For each top-k run that has saved timestep snapshots, emit evolution
    plots for counts, observation counts, and grid-discrete representation.

    Runs without snapshot directories are silently skipped (those runs simply
    predate the count_save_timestep_interval logging).
    """
    evo_dir = args.output_dir / "count_evolution"
    evo_dir.mkdir(parents=True, exist_ok=True)

    any_plotted = False
    for rank, res in enumerate(top_results, start=1):
        folder: Path = res["folder"]
        snaps = _load_snapshot_history(folder)
        timesteps = snaps["timesteps"]
        if not timesteps:
            print(f"[evo] rank {rank} ({folder.name}): no snapshot dirs, skipping")
            continue

        print(
            f"[evo] rank {rank} ({folder.name}): {len(timesteps)} snapshots "
            f"(t={timesteps[0]}..{timesteps[-1]})"
        )
        any_plotted = True

        if snaps["counts"] is not None:
            _plot_fta_evolution(snaps["counts"], timesteps, rank, res, evo_dir)
        if snaps["observation_counts"] is not None:
            _plot_obs_counts_evolution(
                snaps["observation_counts"], timesteps, rank, res, evo_dir, folder
            )
        if snaps["grid_discrete"] is not None:
            _plot_grid_dominant_bin_evolution(
                snaps["grid_discrete"], timesteps, rank, res, evo_dir, folder
            )
            _plot_grid_dominant_bin_per_seed(
                snaps["grid_discrete"], timesteps, rank, res, evo_dir, folder
            )
            _plot_grid_full_distribution(
                snaps["grid_discrete"], timesteps, rank, res, evo_dir, folder
            )

    if not any_plotted:
        print("[evo] no top-k runs had snapshot directories; skipping evolution plots")


# ---------------------------------------------------------------------------
# Goal-reach counts (MountainCar-style envs)
# ---------------------------------------------------------------------------

# Goal position used by Gymnasium MountainCar-v0. Exposed at module scope so
# downstream callers (or future envs) can override if needed.
MOUNTAINCAR_GOAL_POSITION = 0.5


def _goal_reach_per_seed(
    obs_hist: np.ndarray,
    folder: Path,
    goal_position: float = MOUNTAINCAR_GOAL_POSITION,
) -> Optional[np.ndarray]:
    """Sum observation counts in position bins at or above `goal_position`.

    Parameters
    ----------
    obs_hist : (T, S, A, B0, B1) cumulative observation counts.
        B0 indexes the position dim, B1 the velocity dim — matching the
        binning used by ObservationCounts.update_counts (one indexer per obs
        dim, in obs-vector order, which for MountainCar is [position,
        velocity]).
    folder : run folder, used to read the obs-space bounds from config.
    goal_position : position threshold (inclusive) defining "reached the goal".

    Returns
    -------
    np.ndarray of shape (T, S) giving the cumulative goal-region visit count
    per (snapshot, seed), summed over actions, velocity bins, and qualifying
    position bins. Returns None if the obs-space bounds can't be determined
    or if `obs_hist` doesn't have the expected 5D layout.
    """
    if obs_hist is None or obs_hist.ndim != 5:
        return None
    low, high, _ = _load_obs_space_bounds(folder)
    if low is None or high is None or len(low) < 1:
        return None

    T, S, A, B0, B1 = obs_hist.shape
    pos_low, pos_high = float(low[0]), float(high[0])
    if pos_high <= pos_low:
        return None

    # Replicate ObservationCounts.update_counts binning so we know which bin
    # indexes correspond to position >= goal_position. The bin holding
    # `goal_position` is the lowest bin we count (it spans [pos_low + bw*k,
    # pos_low + bw*(k+1))) — values at and above the threshold all land in
    # bin k or higher).
    bin_width = (pos_high - pos_low) / B0
    if bin_width <= 0:
        return None
    goal_bin_lo = int(np.floor((goal_position - pos_low) / bin_width))
    goal_bin_lo = max(0, min(B0 - 1, goal_bin_lo))

    # Sum over actions (axis=2), velocity (axis=4), and position bins >=
    # goal_bin_lo (axis=3).
    summed = obs_hist[:, :, :, goal_bin_lo:, :].sum(axis=(2, 3, 4))  # (T, S)
    return summed


def _plot_goal_reach_counts(top_results, args: Args) -> None:
    """For each top-k run, plot cumulative goal-region visits over training.

    Counts each visit to a position bin at or above MOUNTAINCAR_GOAL_POSITION
    (summed over actions and velocity bins). Two figures are written:
      * goal_reach_curves.png  – mean ± SEM curve per run (snapshot-level)
      * goal_reach_final.png   – bar chart of the final-snapshot mean per run
                                 with per-seed dots and SEM error bars

    Runs whose env doesn't have a recognised observation-space bound (e.g.
    non-MountainCar) or that lack `observation_counts` snapshots are skipped.
    """
    out_dir = args.output_dir / "goal_reach"
    out_dir.mkdir(parents=True, exist_ok=True)

    series: List[Dict[str, Any]] = []
    for rank, res in enumerate(top_results, start=1):
        folder: Path = res["folder"]
        snaps = _load_snapshot_history(folder)
        timesteps = snaps["timesteps"]
        obs_hist = snaps["observation_counts"]
        if not timesteps or obs_hist is None:
            continue
        per_seed = _goal_reach_per_seed(obs_hist, folder)
        if per_seed is None:
            continue
        series.append(
            {
                "rank": rank,
                "name": res["name"],
                "folder_name": folder.name,
                "score": res["score"],
                "timesteps": np.asarray(timesteps),
                "per_seed": per_seed,  # (T, S)
            }
        )

    if not series:
        print(
            "[goal] no top-k runs had observation_counts snapshots; skipping goal-reach plot"
        )
        return

    cmap = plt.cm.get_cmap("tab10", max(10, len(series)))

    # --- (a) Curves over training ----------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, s in enumerate(series):
        per_seed = s["per_seed"]  # (T, S)
        mean = per_seed.mean(axis=1)
        sem = per_seed.std(axis=1, ddof=1) / np.sqrt(per_seed.shape[1])
        label = f"Rank {s['rank']} | {s['folder_name']} " f"(score={s['score']:.2f})"
        ax.plot(
            s["timesteps"], mean, marker="o", color=cmap(i), label=label, linewidth=2
        )
        ax.fill_between(
            s["timesteps"], mean - sem, mean + sem, color=cmap(i), alpha=0.15
        )
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel(
        f"Cumulative goal-region visits per seed\n"
        f"(position ≥ {MOUNTAINCAR_GOAL_POSITION})"
    )
    ax.set_title(
        f"Goal-reach counts over training (Top {len(series)} by {args.score_metric})"
    )
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=8, loc="best", frameon=True)
    fig.tight_layout()
    out_curves = out_dir / "goal_reach_curves.png"
    fig.savefig(out_curves, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_curves}")

    # --- (b) Bar of final-snapshot mean ----------------------------------
    fig2, ax2 = plt.subplots(figsize=(max(6, 0.9 * len(series) + 3), 5))
    xs = np.arange(len(series))
    finals = np.array([s["per_seed"][-1] for s in series])  # (n_runs, S)
    means = finals.mean(axis=1)
    sems = finals.std(axis=1, ddof=1) / np.sqrt(finals.shape[1])
    bar_colors = [cmap(i) for i in range(len(series))]
    ax2.bar(
        xs,
        means,
        yerr=sems,
        capsize=4,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
    )
    # Per-seed dots (jittered) on top of each bar so individual seed
    # behaviour is visible — some seeds may reach the goal many times while
    # others never do.
    rng = np.random.default_rng(0)
    for i, s in enumerate(series):
        seeds_final = s["per_seed"][-1]
        jitter = rng.uniform(-0.18, 0.18, size=seeds_final.shape[0])
        ax2.scatter(
            np.full_like(seeds_final, i, dtype=float) + jitter,
            seeds_final,
            s=14,
            color="black",
            alpha=0.5,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )
    ax2.set_xticks(xs)
    ax2.set_xticklabels(
        [f"Rank {s['rank']}\n{s['folder_name']}" for s in series],
        rotation=30,
        ha="right",
        fontsize=8,
    )
    ax2.set_ylabel(
        f"Final cumulative goal-region visits per seed\n"
        f"(position ≥ {MOUNTAINCAR_GOAL_POSITION})"
    )
    ax2.set_title(f"Final goal-reach counts (Top {len(series)} by {args.score_metric})")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.6)
    fig2.tight_layout()
    out_final = out_dir / "goal_reach_final.png"
    fig2.savefig(out_final, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"  saved {out_final}")


if __name__ == "__main__":
    main(tyro.cli(Args))
