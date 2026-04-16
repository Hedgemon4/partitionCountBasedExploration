import numpy as np
import matplotlib.pyplot as plt
import tyro
import yaml
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any
from pathlib import Path
import scipy.stats as stats


@dataclass
class Args:
    """Analyze and plot specific runs from a large hyperparameter sweep."""
    root_dir: Path = Path("data/mountaincar_longer_runs")
    metric: str = "extrinsic_return_ema"
    intrinsic_metric: str = "intrinsic_return_ema"
    top_k: int = 10
    smooth: int = 1
    output_dir: Path = Path("graphs/mountaincar_longer_runs/timesteps_2e6/top_10/")

    # --- LEGEND PARAMETERS ---
    legend_vars: Optional[List[str]] = None

    # --- FILTER PARAMETERS ---
    beta: Optional[float] = None
    activation: Optional[str] = "fta"
    max_grad_norm: Optional[float] = None
    epsilon_end: Optional[float] = None
    hidden_size: Optional[int] = None
    learnable_norm: Optional[bool] = None
    total_time_steps: Optional[float] = None


def moving_average(x: np.ndarray, w: int):
    if w <= 1:
        return x
    return np.apply_along_axis(lambda m: np.convolve(m, np.ones(w), "valid") / w, axis=1, arr=x)


def matches_filters(folder_path: Path, args: Args) -> bool:
    config_file = folder_path / "config.yaml"
    if not config_file.exists(): return False
    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    except Exception:
        return False

    if args.beta is not None and config.get("beta") != args.beta: return False
    if args.max_grad_norm is not None and config.get("max_grad_norm") != args.max_grad_norm: return False
    if args.epsilon_end is not None and config.get("epsilon_end") != args.epsilon_end: return False

    network_config = config.get("network", {})
    if args.hidden_size is not None and network_config.get("hidden_size") != args.hidden_size: return False
    if args.learnable_norm is not None and network_config.get(
        "learnable_norm_params") != args.learnable_norm: return False
    if args.total_time_steps is not None and config.get("total_time_steps") != args.total_time_steps: return False
    if args.activation is not None and network_config.get("activation1", {}).get(
        "type") != args.activation: return False
    return True


def load_run_data(folder_path: Path, metric_name: str) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    metrics_file = folder_path / "metrics.npz"
    if not metrics_file.exists(): return None, None, None, None
    try:
        data = np.load(metrics_file)
        steps = data["env_step"][0]
        if metric_name not in data: return None, None, None, None
        values = data[metric_name]
        last_10_percent = max(1, int(values.shape[1] * 0.1))
        final_values_per_seed = np.mean(values[:, -last_10_percent:], axis=1)
        return steps, values, np.mean(final_values_per_seed), final_values_per_seed
    except Exception:
        return None, None, None, None


def get_config_value(config: dict, key_path: str) -> Any:
    val = config
    try:
        for part in key_path.split("."): val = val.get(part)
        return val
    except (AttributeError, TypeError):
        return None


def format_legend_label(folder_path: Path, legend_vars: Optional[List[str]]) -> str:
    if not legend_vars: return folder_path.name
    config_file = folder_path / "config.yaml"
    if not config_file.exists(): return folder_path.name
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    label_parts = [f"{var.split('.')[-1]}: {get_config_value(config, var)}" for var in legend_vars]
    return " | ".join(label_parts)


def plot_curves(results, metric_name, output_path, title, smooth_win):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.get_cmap("tab20", len(results))
    for i, res in enumerate(results):
        if res["values"] is None: continue
        smoothed = moving_average(res["values"], smooth_win)
        steps = res["steps"][:smoothed.shape[1]]
        mean, std_err = np.mean(smoothed, axis=0), stats.sem(smoothed, axis=0)
        ci = std_err * stats.t.ppf((1 + 0.95) / 2., len(res["values"]) - 1)
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
            steps, values, score, final_vals = load_run_data(folder, args.metric)
            if steps is not None:
                all_filtered.append({
                    "name": format_legend_label(folder, args.legend_vars),
                    "folder": folder, "steps": steps, "values": values,
                    "score": score, "final_seed_vals": final_vals
                })

    all_filtered.sort(key=lambda x: x["score"], reverse=True)
    top_results = all_filtered[:args.top_k]
    if not top_results: print("No matching folders."); return

    # 1. Extrinsic Curves
    handles, labels = plot_curves(top_results, args.metric, args.output_dir / "filtered_learning_curves.png",
                                  f"Learning Curves (Top {len(top_results)} by {args.metric})", args.smooth)

    # 2. Intrinsic Curves
    intrinsic_res = []
    for res in top_results:
        s, v, _, _ = load_run_data(res["folder"], args.intrinsic_metric)
        intrinsic_res.append({"name": res["name"], "steps": s, "values": v})

    plot_curves(intrinsic_res, args.intrinsic_metric, args.output_dir / "intrinsic_reward_curves.png",
                f"Intrinsic Reward Curves (Top {len(top_results)} Runs)", args.smooth)

    # 3. Box Plot for Variance (Restored)
    fig_box, ax_box = plt.subplots(figsize=(12, 6))
    data_to_plot = [res["final_seed_vals"] for res in top_results]
    box_labels = [f"Rank {i + 1}" for i in range(len(top_results))]

    ax_box.boxplot(data_to_plot, labels=box_labels, patch_artist=True,
                   boxprops=dict(facecolor="lightblue", color="blue"),
                   medianprops=dict(color="red", linewidth=2))
    ax_box.set_ylabel(f"Final {args.metric.replace('_', ' ').title()} (Last 10%)")
    ax_box.set_title("Variance Across Seeds")
    plt.xticks(rotation=45, ha="right")
    fig_box.savefig(args.output_dir / "filtered_seed_variance.png", dpi=300, bbox_inches="tight")

    # 4. Legend
    fig_leg = plt.figure(figsize=(10, len(top_results) * 0.3))
    fig_leg.legend(handles, labels, loc='center', ncol=1, frameon=False, prop={'family': 'monospace', 'size': 9})
    fig_leg.savefig(args.output_dir / "legend_filtered_learning_curves.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main(tyro.cli(Args))
