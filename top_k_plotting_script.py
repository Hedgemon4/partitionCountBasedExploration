import numpy as np
import matplotlib.pyplot as plt
import tyro
import yaml
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any
from pathlib import Path
import scipy.stats as stats


"""
Plots the top K runs based on having the best final performance (mean of last 10% of the metric) after applying user-specified filters.
"""

@dataclass
class Args:
    """Analyze and plot specific runs from a large hyperparameter sweep."""
    root_dir: Path = Path("data/mountaincar_count_first_layer_combined")
    metric: str = "extrinsic_return_ema"
    top_k: int = 20
    smooth: int = 1
    output_dir: Path = Path("graphs/updated_top_k_plots")

    # --- LEGEND PARAMETERS ---
    # Specify which config keys to show in the legend (e.g., ["beta", "network.hidden_size"])
    legend_vars: Optional[List[str]] = None

    # --- FILTER PARAMETERS ---
    beta: Optional[float] = None
    activation: Optional[str] = "fta"
    max_grad_norm: Optional[float] = None
    epsilon_end: Optional[float] = None
    hidden_size: Optional[int] = None
    learnable_norm: Optional[bool] = None


def moving_average(x: np.ndarray, w: int):
    if w <= 1:
        return x
    return np.apply_along_axis(lambda m: np.convolve(m, np.ones(w), "valid") / w, axis=1, arr=x)


def matches_filters(folder_path: Path, args: Args) -> bool:
    """Reads the config.yaml and checks if the parameters match the requested filters."""
    config_file = folder_path / "config.yaml"

    if not config_file.exists():
        return False

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading {config_file}: {e}")
        return False

    # Check top-level config variables
    if args.beta is not None and config.get("beta") != args.beta:
        return False
    if args.max_grad_norm is not None and config.get("max_grad_norm") != args.max_grad_norm:
        return False
    if args.epsilon_end is not None and config.get("epsilon_end") != args.epsilon_end:
        return False

    # Check nested 'network' config variables
    network_config = config.get("network", {})

    if args.hidden_size is not None and network_config.get("hidden_size") != args.hidden_size:
        return False

    if args.learnable_norm is not None and network_config.get("learnable_norm_params") != args.learnable_norm:
        return False

    # Check double-nested 'activation1' config variable
    if args.activation is not None:
        act_type = network_config.get("activation1", {}).get("type")
        if act_type != args.activation:
            return False

    return True


def load_run_data(folder_path: Path, metric_name: str) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    metrics_file = folder_path / "metrics.npz"
    if not metrics_file.exists():
        return None, None, None, None

    try:
        data = np.load(metrics_file)
        steps = data["env_step"][0]
        values = data[metric_name]

        last_10_percent = max(1, int(values.shape[1] * 0.1))
        final_values_per_seed = np.mean(values[:, -last_10_percent:], axis=1)
        mean_score = np.mean(final_values_per_seed)

        return steps, values, mean_score, final_values_per_seed
    except Exception as e:
        return None, None, None, None


def get_config_value(config: dict, key_path: str) -> Any:
    """Helper to get nested values using dot notation (e.g., 'network.hidden_size')"""
    parts = key_path.split(".")
    val = config
    try:
        for part in parts:
            val = val.get(part)
        return val
    except (AttributeError, TypeError):
        return None


def format_legend_label(folder_path: Path, legend_vars: Optional[List[str]]) -> str:
    """Creates a human-readable label based on selected config variables."""
    if not legend_vars:
        return folder_path.name  # Fallback to folder name

    config_file = folder_path / "config.yaml"
    if not config_file.exists():
        return folder_path.name

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    label_parts = []
    for var in legend_vars:
        # Special case for activation1 mapping if you prefer shorter names
        if var == "network.activation1.type":
            val = get_config_value(config, var)
            label_parts.append(f"act: {val}")
        else:
            val = get_config_value(config, var)
            # Use the last part of the dot notation as the label key
            display_name = var.split(".")[-1]
            label_parts.append(f"{display_name}: {val}")

    return " | ".join(label_parts)


# ... [Keep moving_average, matches_filters, and load_run_data as they were] ...

def main(args: Args):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_folders = [d for d in args.root_dir.iterdir() if d.is_dir()]
    results = []

    filtered_folders = [f for f in run_folders if matches_filters(f, args)]
    print(f"Found {len(filtered_folders)} folders matching your filters.")

    if not filtered_folders:
        print("No folders matched your criteria. Exiting.")
        return

    for folder in filtered_folders:
        steps, values, score, final_seed_vals = load_run_data(folder, args.metric)
        if steps is not None:
            # Generate the nice legend name here
            nice_name = format_legend_label(folder, args.legend_vars)
            results.append({
                "name": nice_name,
                "folder_name": folder.name,
                "steps": steps,
                "values": values,
                "score": score,
                "final_seed_vals": final_seed_vals
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    top_results = results[:args.top_k]

    # --- PLOT 1: Learning Curves ---
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.get_cmap("tab20", len(top_results))

    for i, res in enumerate(top_results):
        smoothed_vals = moving_average(res["values"], args.smooth)
        plot_steps = res["steps"][:smoothed_vals.shape[1]]

        mean = np.mean(smoothed_vals, axis=0)
        std_err = stats.sem(smoothed_vals, axis=0)
        ci = std_err * stats.t.ppf((1 + 0.95) / 2., len(res["values"]) - 1)

        ax.plot(plot_steps, mean, label=res['name'], color=colors(i), linewidth=2)
        ax.fill_between(plot_steps, mean - ci, mean + ci, color=colors(i), alpha=0.15)

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel(args.metric.replace("_", " ").title())
    ax.set_title(f"Learning Curves (Top {len(top_results)} Filtered)")
    ax.grid(True, linestyle="--", alpha=0.7)

    # Move legend outside or to a separate file as you had it
    handles, labels = ax.get_legend_handles_labels()

    curve_out = args.output_dir / "filtered_learning_curves.png"
    fig.savefig(curve_out, dpi=300, bbox_inches="tight")

    # Legend only figure - updated for readability
    fig_leg = plt.figure(figsize=(10, len(top_results) * 0.3))  # Dynamic height
    fig_leg.legend(handles, labels, loc='center', ncol=1, frameon=False, prop={'family': 'monospace', 'size': 9})
    leg_out = args.output_dir / "legend_filtered_learning_curves.png"
    fig_leg.savefig(leg_out, dpi=300, bbox_inches="tight")

    # --- PLOT 2: Boxplot ---
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    data_to_plot = [res["final_seed_vals"] for res in top_results]

    # Just use rank
    box_labels = [f"Rank {i + 1}"for i, res in enumerate(top_results)]

    ax2.boxplot(data_to_plot, labels=box_labels, patch_artist=True,
                boxprops=dict(facecolor="lightblue", color="blue"),
                medianprops=dict(color="red", linewidth=2))

    ax2.set_ylabel(f"Final {args.metric.replace('_', ' ').title()} (Last 10%)")
    ax2.set_title("Variance Across Seeds")
    plt.xticks(rotation=45, ha="right")

    box_out = args.output_dir / "filtered_seed_variance.png"
    fig2.savefig(box_out, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
