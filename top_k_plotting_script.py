import numpy as np
import matplotlib.pyplot as plt
import tyro
import yaml
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any
from pathlib import Path
import warnings
import scipy.stats as stats
# Import shared histogram helpers so every bin-usage plot across the project
# uses the exact same format (stacked by action, seed std error bars, per-seed
# dots, percentage labels, outlier annotation).
from count_histogram import (
    plot_histogram_with_actions,
    plot_neuron_summary,
    find_outlier_seeds,
    aggregate_counts,
)


@dataclass
class Args:
    """Analyze and plot specific runs from a large hyperparameter sweep."""

    root_dir: Path = Path("data/mountaincar_pqn_baseline_sweep")
    metric: str = "extrinsic_return_ema"
    intrinsic_metric: str = None
    top_k: int = 10
    smooth: int = 1
    output_dir: Path = Path("graphs/mountaincar_pqn_baseline_sweep/top_10/auc/")

    # --- SCORING PARAMETERS ---
    score_metric: str = "auc"
    """How to rank runs for top-k selection. Options:
      last_10pct  - mean over the final 10%% of timesteps (original behaviour)
      auc         - area under the curve (trapezoidal, normalised by x-range)
      max         - peak mean value across all timesteps
    """

    # --- LEGEND PARAMETERS ---
    legend_vars: Optional[List[str]] = None

    # --- FILTER PARAMETERS ---
    beta: Optional[float] = None
    activation: Optional[str] = None
    max_grad_norm: Optional[float] = None
    epsilon_end: Optional[float] = None
    hidden_size: Optional[int] = None
    learnable_norm: Optional[bool] = None
    total_time_steps: Optional[float] = None


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
            val = val.get(part)
        return val
    except (AttributeError, TypeError):
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


def plot_curves(results, metric_name, output_path, title, smooth_win):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.get_cmap("tab20", len(results))
    for i, res in enumerate(results):
        if res["values"] is None:
            continue
        smoothed = moving_average(res["values"], smooth_win)
        steps = res["steps"][: smoothed.shape[1]]
        mean, std_err = np.mean(smoothed, axis=0), stats.sem(smoothed, axis=0)
        ci = std_err * stats.t.ppf((1 + 0.95) / 2.0, len(res["values"]) - 1)
        label = f"Rank {i + 1} | {res['name']}"
        ax.plot(steps, mean, label=label, color=colors(i), linewidth=2)
        ax.fill_between(steps, mean - ci, mean + ci, color=colors(i), alpha=0.15)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return ax.get_legend_handles_labels()


def main(args: Args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
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

    all_filtered.sort(key=lambda x: x["score"], reverse=True)
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


def _plot_count_histograms(top_results, args: Args):
    """For each top-k run, produce a bin-usage histogram in the same format
    as graphing_scripts/count_histogram.py (stacked by action, seed std error
    bars, per-seed dots, percentage labels, outlier annotation).

    A combined side-by-side plot with a shared y-axis is also produced so
    runs can be compared directly.
    """
    hist_dir = args.output_dir / "count_histograms"
    hist_dir.mkdir(parents=True, exist_ok=True)

    # Locate counts files up-front so we can skip runs that didn't log counts
    runs_with_counts = []
    for rank, res in enumerate(top_results, start=1):
        folder = res["folder"]
        candidates = [folder / "counts.npy", folder / "counts.npz.npy"]
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
            counts, title=title, out_path=out, highlight=(rank == 1)
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
            show_outlier_box=True,
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
        f.write("(Flagged when a bin is 0, an (action,bin) cell is 0, or |z|>2 for any bin)\n\n")
        f.write("\n".join(outlier_report_lines))
    print(f"  saved {report_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
