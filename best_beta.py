import numpy as np
import matplotlib.pyplot as plt
import tyro
import yaml
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, List, Optional
from pathlib import Path
import scipy.stats as stats


@dataclass
class Args:
    """Find and plot the best performing run for each unique beta value."""

    # The root directory containing all the run subfolders
    root_dir: Path = Path("data/mountaincar_static_epsilon")
    # Metrics to analyze
    metric: str = "length_ema"
    intrinsic_metric: str = "intrinsic_return_ema"
    # Window size for smoothing the line plot
    smooth: int = 1
    # Output directory for graphs
    output_dir: Path = Path("graphs/mountaincar_static_epsilon/best_beta/")

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
    reverse: bool = False
    """Controls how runs are ranked within each beta when picking the
    "best" config. Mirrors the top_k_plotting_script semantics:
      reverse = False  → smaller scores win  (e.g. MountainCar length_ema —
                         shorter episodes are better)
      reverse = True   → larger  scores win  (e.g. extrinsic_return_ema —
                         higher return is better)  [default]
    """

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


def load_run_data(
    folder_path: Path, metric_name: str
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Loads metrics and calculates the final score."""
    metrics_file = folder_path / "metrics.npz"
    if not metrics_file.exists():
        return None, None, None, None

    try:
        data = np.load(metrics_file)
        if metric_name not in data:
            return None, None, None, None

        steps = data["env_step"][0]
        values = data[metric_name]

        last_10_percent = max(1, int(values.shape[1] * 0.1))
        final_values_per_seed = np.mean(values[:, -last_10_percent:], axis=1)
        mean_score = np.mean(final_values_per_seed)

        return steps, values, mean_score, final_values_per_seed
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
):
    """Helper function to plot learning curves for a specific metric."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.get_cmap("tab10", len(sorted_betas))

    for i, b in enumerate(sorted_betas):
        res = best_runs[b]

        # Some metrics (like intrinsic) might not be present in all runs
        if res.get(metric_key) is None:
            continue

        smoothed_vals = moving_average(res[metric_key], smooth)
        plot_steps = res["steps"][: smoothed_vals.shape[1]]

        mean = np.mean(smoothed_vals, axis=0)
        std_err = stats.sem(smoothed_vals, axis=0)
        ci = std_err * stats.t.ppf((1 + 0.95) / 2.0, len(res[metric_key]) - 1)

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

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved {metric_title} curves to {output_path}")

    return ax.get_legend_handles_labels()


def main(args: Args):
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

        steps, values, score, final_seed_vals = load_run_data(folder, args.metric)
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
