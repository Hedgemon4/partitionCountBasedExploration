import numpy as np
import matplotlib.pyplot as plt
import tyro
import yaml
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
from collections import defaultdict


@dataclass
class Args:
    """Plotting script to group and compare PQN runs based on config.yaml."""

    # The root directory to search for metrics.npz files
    root_dir: Path = Path("data/pqn_mountaincar")

    # The key in the .npz file to visualize
    metric: str = "extrinsic_return_ema"

    # Window size for simple moving average smoothing (1 = no smoothing)
    smooth: int = 1

    # Directory to save the resulting plots
    output_dir: Path = Path("graphs/pqn_mountaincar/")

    # Config keys to group by. A separate plot will be generated for each combination.
    # Note: Use dot notation for nested yaml keys, e.g., 'network.hidden_size'
    group_by: Tuple[str, ...] = ("max_grad_norm", "epsilon_end")

    # These keys are ignored when trying to figure out what changed for the legend labels,
    # because they usually differ for every single run or are irrelevant.
    ignore_in_labels: Tuple[str, ...] = (
        "output_folder_name",
        "seed",
        "total_time_steps",
        "num_seeds",
    )


def moving_average(x: np.ndarray, w: int):
    if w <= 1:
        return x
    return np.convolve(x, np.ones(w), "valid") / w


def process_metrics(filename: Path, metric_name: str, smooth_window: int):
    """Loads data, applies smoothing, and calculates mean and 95% CI."""
    try:
        data = np.load(filename)
        steps = data["env_step"]
        values = data[metric_name]

        # Handle shapes properly (assume n_seeds, n_timesteps)
        if values.ndim == 2:
            mean = np.mean(values, axis=0)
            std = np.std(values, axis=0)
            ci = 1.96 * std / np.sqrt(values.shape[0])
            steps = steps[0] if steps.ndim == 2 else steps
        else:
            mean = values
            ci = np.zeros_like(values)

        mean = moving_average(mean, smooth_window)
        ci = moving_average(ci, smooth_window)
        steps = steps[: len(mean)]

        return steps, mean, ci
    except Exception as e:
        print(f"Could not process {filename}: {e}")
        return None, None, None


def flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """Flattens a nested dictionary using dot notation for keys."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def main(args: Args):
    # Find all config.yaml files
    config_paths = list(args.root_dir.rglob("config.yaml"))
    if not config_paths:
        print(f"No config.yaml files found in {args.root_dir}")
        return

    # 1. Group the files based on the requested config hyperparameters
    groups = defaultdict(list)
    for config_path in config_paths:
        metrics_path = config_path.parent / "metrics.npz"
        if not metrics_path.exists():
            continue

        with open(config_path, "r") as f:
            try:
                config_data = yaml.safe_load(f)
                flat_config = flatten_dict(config_data)
            except yaml.YAMLError as e:
                print(f"Error parsing {config_path}: {e}")
                continue

        # Create a unique key for the group based on the specified group_by keys
        group_key = tuple(f"{k}={flat_config.get(k, 'N/A')}" for k in args.group_by)
        groups[group_key].append((metrics_path, flat_config))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Create a separate plot for each group
    for group_key, items in groups.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.get_cmap("tab20", len(items))

        # Determine which configuration values ACTUALLY vary within this specific group
        # This prevents our legends from having 30 parameters listed
        all_keys = set(
            k
            for _, config in items
            for k in config.keys()
            if k not in args.group_by and k not in args.ignore_in_labels
        )

        varying_keys = []
        for key in all_keys:
            # Check if there is more than 1 unique value for this key in the current group
            unique_values = set(str(config.get(key, "N/A")) for _, config in items)
            if len(unique_values) > 1:
                varying_keys.append(key)

        for i, (metrics_path, config) in enumerate(items):
            steps, mean, ci = process_metrics(metrics_path, args.metric, args.smooth)
            if steps is not None:
                # Build a label using ONLY the parameters that vary within this group
                label_parts = [f"{k}={config.get(k, 'N/A')}" for k in varying_keys]

                if label_parts:
                    label = ", ".join(label_parts)
                else:
                    # Fallback if nothing varies except seed/output_folder
                    label = metrics_path.parent.name

                ax.plot(steps, mean, label=label, color=colors(i), linewidth=2)
                ax.fill_between(steps, mean - ci, mean + ci, color=colors(i), alpha=0.2)

        # Format the plot
        ax.set_xlabel("Timesteps")
        ax.set_ylabel(args.metric.replace("_", " ").title())

        group_title = " | ".join(group_key)
        ax.set_title(f"Comparison: {args.metric}\nFixed: {group_title}")
        ax.grid(True, linestyle="--", alpha=0.7)

        # Move legend outside the plot area so it doesn't overlap the data
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Create a safe filename using the group key
        safe_key = "_".join(
            [str(k).replace("=", "-").replace("/", "-") for k in group_key]
        )
        filename = f"{args.metric}_{safe_key}.png"

        fig.savefig(args.output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved {filename} with {len(items)} lines.")


if __name__ == "__main__":
    tyro.cli(main)
