import numpy as np
import matplotlib.pyplot as plt
import tyro
import yaml
import warnings
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, List, Optional
from pathlib import Path
import scipy.stats as stats


@dataclass
class Args:
    """Find and plot the best performing run for each unique beta value."""

    # The root directory containing all the run subfolders
    root_dir: Path = Path("data/venture_counts_sweep")
    # Metrics to analyze
    metric: str = "extrinsic_return_per_game_ema"
    intrinsic_metric: str = "intrinsic_return_per_game_ema"
    # Window size for smoothing the line plot
    smooth: int = 20
    # Output directory for graphs
    output_dir: Path = Path("graphs/venture_counts_sweep/less_smooth/best_beta")

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

    # --- SCORING PARAMETERS ---
    score_metric: str = "last_10pct"
    """How to rank runs when picking the best one per beta. Options:
      last_10pct  - mean over the final 10%% of timesteps (original behaviour)
      auc         - area under the curve (trapezoidal, normalised by x-range)
      max         - peak mean value across all timesteps
    """

    reverse: bool = False
    """Controls how runs are ranked within each beta when picking the
    "best" config. Mirrors the top_k_plotting_script semantics:
      reverse = False  → smaller scores win  (e.g. MountainCar length_ema —
                         shorter episodes are better)
      reverse = True   → larger  scores win  (e.g. extrinsic_return_ema —
                         higher return is better)  [default]
    """

    group_seeds: bool = True
    """Seed-grouping mode for sweeps that write ONE folder per
    (hyperparameter, seed) combination — e.g. data/freeway_sweep, where each
    metrics.npz holds 1-D arrays for a single seed.

    When set, run folders that share the same `beta` and
    `network.next_state_coef` are merged into a single multi-seed run (one row
    per seed). For each beta the best `next_state_coef` group (highest reward)
    is selected, and ONLY the extrinsic and intrinsic reward curves are
    produced (each with its own legend). The per-folder multi-seed behaviour is
    used when this is off."""

    next_state_coefs: Optional[Tuple[float, ...]] = None
    """--group-seeds only: restrict the per-beta search to these
    `network.next_state_coef` values. None = all of them. Pass e.g.
    `--next-state-coefs 0.25 0.5 1.0` to drop the 0.0 (no next-state-prediction)
    baseline so each beta's champion is the best run that actually uses the
    next-state-prediction loss."""

    # --- FILTER PARAMETERS ---
    beta_filter: Optional[float] = None  # Used if you only want to plot specific betas
    activation: Optional[str] = None
    max_grad_norm: Optional[float] = None
    epsilon_end: Optional[float] = None
    hidden_size: Optional[int] = None
    learnable_norm: Optional[bool] = None
    total_time_steps: Optional[float] = None
    next_state_coef: Optional[float] = 0.0

    # --- PLOT PARAMETERS ---
    y_lim: Optional[Tuple[float, float]] = (100, 200)
    """Y-axis limits for the extrinsic best-by-beta curves plot, as
    (ymin, ymax). Leave unset to use matplotlib's autoscaling.
    Example: --y-lim -200 0
    """
    intrinsic_y_lim: Optional[Tuple[float, float]] = (0, 80)
    """Y-axis limits for the intrinsic best-by-beta curves plot, as
    (ymin, ymax). Leave unset to use matplotlib's autoscaling.
    """


def moving_average(x: np.ndarray, w: int):
    if w <= 1:
        return x
    return np.apply_along_axis(
        lambda m: np.convolve(m, np.ones(w), "valid") / w, axis=1, arr=x
    )


def get_config_value(config: dict, key_path: str) -> Any:
    val = config
    try:
        for part in key_path.split("."):
            val = val.get(part)
        return val
    except (AttributeError, TypeError):
        return None


def format_legend_label(folder_path: Path, legend_vars: Optional[List[str]]) -> str:
    """Formats the legend label based on requested config variables."""
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


def matches_filters(folder_path: Path, args: Args) -> bool:
    """Checks if a run directory matches the specified CLI filters."""
    config_file = folder_path / "config.yaml"
    if not config_file.exists():
        return False

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    except Exception:
        return False

    if args.beta_filter is not None and config.get("beta") != args.beta_filter:
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
        args.total_time_steps is not None
        and config.get("total_time_steps") != args.total_time_steps
    ):
        return False
    if (
        args.learnable_norm is not None
        and network_config.get("learnable_norm_params") != args.learnable_norm
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


def get_beta_from_config(folder_path: Path) -> float:
    """Reads the config.yaml and extracts the beta value."""
    config_file = folder_path / "config.yaml"
    if not config_file.exists():
        return None

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        return config.get("beta")
    except Exception as e:
        print(f"Error reading {config_file}: {e}")
        return None


def compute_score(
    values: np.ndarray, steps: np.ndarray, score_metric: str
) -> Tuple[float, np.ndarray]:
    """Return (scalar score for ranking, per-seed final values for the box plot).

    Mirrors top_k_plotting_script.compute_score so both scripts agree on
    what e.g. "auc" or "max" means.

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
    elif score_metric == "auc":
        # nanmean + NaN-strip handles seeds with missing early logs
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
    """Loads metrics and calculates the score for ranking + per-seed finals."""
    metrics_file = folder_path / "metrics.npz"
    if not metrics_file.exists():
        return None, None, None, None

    try:
        data = np.load(metrics_file)
        if metric_name not in data:
            return None, None, None, None

        steps = data["env_step"][0]
        values = data[metric_name]

        score, final_values_per_seed = compute_score(values, steps, score_metric)

        return steps, values, score, final_values_per_seed
    except Exception as e:
        return None, None, None, None


def plot_beta_curves(
    best_runs: Dict[float, Dict[str, Any]],
    sorted_betas: List[float],
    metric_key: str,
    metric_title: str,
    output_path: Path,
    smooth: int,
    y_lim: Optional[Tuple[float, float]] = None,
    show_legend: bool = False,
    nan_aware: bool = False,
):
    """Helper function to plot learning curves for a specific metric.

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
    colors = plt.cm.get_cmap("tab10", max(len(sorted_betas), 1))

    for i, b in enumerate(sorted_betas):
        res = best_runs[b]

        # Some metrics (like intrinsic) might not be present in all runs
        if res.get(metric_key) is None:
            continue

        smoothed_vals = moving_average(res[metric_key], smooth)
        plot_steps = res["steps"][: smoothed_vals.shape[1]]

        n_seeds = len(res[metric_key])
        if nan_aware:
            with warnings.catch_warnings():
                # All-NaN leading columns are expected (pre-first-episode).
                warnings.simplefilter("ignore", RuntimeWarning)
                mean = np.nanmean(smoothed_vals, axis=0)
                std = np.nanstd(smoothed_vals, axis=0, ddof=1)
            n_valid = np.sum(~np.isnan(smoothed_vals), axis=0)
            std_err = std / np.sqrt(np.maximum(n_valid, 1))
        else:
            mean = np.mean(smoothed_vals, axis=0)
            std_err = stats.sem(smoothed_vals, axis=0)
        ci = std_err * stats.t.ppf((1 + 0.95) / 2.0, max(n_seeds - 1, 1))

        label = f"Beta {b} | {res['legend_name']}"

        ax.plot(plot_steps, mean, label=label, color=colors(i), linewidth=2)
        ax.fill_between(plot_steps, mean - ci, mean + ci, color=colors(i), alpha=0.15)

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel(metric_title.replace("_", " ").title())
    ax.set_title(
        f"Best Configurations per Beta: {metric_title.replace('_', ' ').title()}"
    )
    ax.grid(True, linestyle="--", alpha=0.7)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if show_legend:
        ax.legend(fontsize=8, loc="best")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved {metric_title} curves to {output_path}")

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

    Merges single-seed run folders by (beta, network.next_state_coef). For
    each beta the best-scoring next_state_coef group is selected (highest
    reward), and ONLY the extrinsic and intrinsic reward curves are produced,
    each with its own legend.
    """
    from sweep_grouping import build_grouped_runs, filter_by_config

    ext_metric, int_metric = _resolve_reward_metrics(args)

    # Extrinsic is required; intrinsic is optional so baseline runs (no
    # intrinsic-reward logging) survive grouping. `sarsa_returns` is part of
    # the key so SARSA and Q-learning runs don't get merged into one group.
    groups = build_grouped_runs(
        args.root_dir,
        group_keys=("beta", "network.next_state_coef", "sarsa_returns"),
        metric_names=(ext_metric,),
        optional_metric_names=(int_metric,) if int_metric else (),
    )
    if not groups:
        print("No grouped runs found — check --root-dir.")
        return

    groups = filter_by_config(groups, "network.next_state_coef", args.next_state_coefs)
    if not groups:
        print(f"No groups left after --next-state-coefs {args.next_state_coefs}.")
        return

    # For each beta, keep the next_state_coef group with the best (highest)
    # reward score.
    best_runs_by_beta: Dict[float, Dict[str, Any]] = {}
    for g in groups:
        beta = g.config.get("beta")
        coef = g.config.get("network.next_state_coef")
        ext = g.metrics.get(ext_metric)
        if beta is None or ext is None:
            continue

        score, final_seed_vals = compute_score(ext, g.steps, args.score_metric)

        # Reward: higher is better.
        if beta not in best_runs_by_beta or score > best_runs_by_beta[beta]["score"]:
            best_runs_by_beta[beta] = {
                "legend_name": f"next_state_coef={coef} | n={g.n_seeds} seeds",
                "steps": g.steps,
                "ext_values": ext,
                "int_values": g.metrics.get(int_metric) if int_metric else None,
                "score": score,
                "final_seed_vals": final_seed_vals,
                "beta": beta,
                "next_state_coef": coef,
            }

    if not best_runs_by_beta:
        print("No valid grouped runs found. Exiting.")
        return

    sorted_betas = sorted(best_runs_by_beta.keys())

    print("\n--- Champions per Beta (best next_state_coef) ---")
    for b in sorted_betas:
        res = best_runs_by_beta[b]
        print(
            f"Beta {b}: score {res['score']:.3f} | "
            f"next_state_coef={res['next_state_coef']}"
        )

    # The dataclass y-limit defaults (100, 200)/(0, 80) are MountainCar
    # length-scale ranges; for a reward sweep let matplotlib autoscale unless
    # the user passed explicit limits.
    ext_ylim = None if args.y_lim == (100, 200) else args.y_lim
    int_ylim = None if args.intrinsic_y_lim == (0, 80) else args.intrinsic_y_lim

    # 1. Extrinsic reward curves
    plot_beta_curves(
        best_runs_by_beta,
        sorted_betas,
        "ext_values",
        ext_metric,
        args.output_dir / "best_by_beta_extrinsic_curves.png",
        args.smooth,
        y_lim=ext_ylim,
        show_legend=True,
        nan_aware=True,
    )

    # 2. Intrinsic reward curves
    if int_metric:
        plot_beta_curves(
            best_runs_by_beta,
            sorted_betas,
            "int_values",
            int_metric,
            args.output_dir / "best_by_beta_intrinsic_curves.png",
            args.smooth,
            y_lim=int_ylim,
            show_legend=True,
            nan_aware=True,
        )


def main(args: Args):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Seed-grouping mode (one folder per (hyperparam, seed)) takes its own path
    # and only produces the extrinsic + intrinsic reward curves.
    if args.group_seeds:
        _main_grouped(args)
        return

    run_folders = [d for d in args.root_dir.iterdir() if d.is_dir()]

    # Dictionary to keep track of the absolute best run for each beta value
    best_runs_by_beta: Dict[float, Dict[str, Any]] = {}

    print(f"Scanning directories to find the best run per beta...")

    for folder in run_folders:
        if not matches_filters(folder, args):
            continue

        beta = get_beta_from_config(folder)
        if beta is None:
            continue

        steps, values, score, final_seed_vals = load_run_data(
            folder, args.metric, args.score_metric
        )
        if steps is None:
            continue

        # Update dictionary if it's the first time seeing this beta or if score is better.
        # `reverse` mirrors top_k_plotting_script.py: reverse=True → larger is better,
        # reverse=False → smaller is better (e.g. shorter MountainCar episodes).
        def _is_better(new: float, old: float) -> bool:
            return new > old if args.reverse else new < old

        if beta not in best_runs_by_beta or _is_better(
            score, best_runs_by_beta[beta]["score"]
        ):
            legend_name = format_legend_label(folder, args.legend_vars)

            # Load intrinsic data simultaneously for the winning config
            _, int_values, _, _ = load_run_data(folder, args.intrinsic_metric)

            best_runs_by_beta[beta] = {
                "folder_name": folder.name,
                "legend_name": legend_name,
                "steps": steps,
                "ext_values": values,
                "int_values": int_values,
                "score": score,
                "final_seed_vals": final_seed_vals,
                "beta": beta,
                "folder_path": folder,
            }

    if not best_runs_by_beta:
        print("No valid runs found matching filters. Exiting.")
        return

    sorted_betas = sorted(best_runs_by_beta.keys())

    print("\n--- Champions per Beta ---")
    for b in sorted_betas:
        res = best_runs_by_beta[b]
        print(f"Beta {b}: Score {res['score']:.2f} | Config: {res['folder_name']}")

    # ==========================================
    # PLOT 1: Extrinsic Learning Curves
    # ==========================================
    ext_out = args.output_dir / "best_by_beta_extrinsic_curves.png"
    handles, labels = plot_beta_curves(
        best_runs_by_beta,
        sorted_betas,
        "ext_values",
        args.metric,
        ext_out,
        args.smooth,
        y_lim=args.y_lim,
    )

    # Save a detached legend based on the extrinsic handles
    fig_leg = plt.figure(figsize=(10, len(sorted_betas) * 0.4))
    fig_leg.legend(
        handles,
        labels,
        loc="center",
        ncol=1,
        frameon=False,
        prop={"family": "monospace", "size": 9},
    )
    leg_out = args.output_dir / "legend_best_by_beta.png"
    fig_leg.savefig(leg_out, dpi=300, bbox_inches="tight")
    print(f"Saved separated legend to {leg_out}")

    # ==========================================
    # PLOT 2: Intrinsic Learning Curves
    # ==========================================
    int_out = args.output_dir / "best_by_beta_intrinsic_curves.png"
    plot_beta_curves(
        best_runs_by_beta,
        sorted_betas,
        "int_values",
        args.intrinsic_metric,
        int_out,
        args.smooth,
        y_lim=args.intrinsic_y_lim,
    )

    # ==========================================
    # PLOT 3: Boxplot for Final Performance
    # ==========================================
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    data_to_plot = [best_runs_by_beta[b]["final_seed_vals"] for b in sorted_betas]
    box_labels = [f"Beta {b}" for b in sorted_betas]

    ax2.boxplot(
        data_to_plot,
        labels=box_labels,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", color="blue"),
        medianprops=dict(color="red", linewidth=2),
    )

    ax2.set_ylabel(f"Final {args.metric.replace('_', ' ').title()} (Last 10%)")
    ax2.set_title("Variance Across Seeds (Best Configs per Beta)")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    box_out = args.output_dir / "best_by_beta_seed_variance.png"
    fig2.savefig(box_out, dpi=300, bbox_inches="tight")
    print(f"\nSaved seed variance boxplot to {box_out}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
