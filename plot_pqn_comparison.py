import numpy as np
import matplotlib.pyplot as plt
import tyro
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class Args:
    """Refactored plotting script to find and compare PQN runs automatically."""

    # The root directory to search for metrics.npz files
    root_dir: Path = Path("data/cartpole_beta_sweep")

    # The filename to look for in subdirectories
    filename_pattern: str = "metrics.npz"

    # The key in the .npz file to visualize
    metric: str = "extrinsic_return_ema"

    # Window size for simple moving average smoothing (1 = no smoothing)
    smooth: int = 1

    # Path to save the resulting plot
    output: Path = Path("beta_comparison_plot.png")

    # Title of the plot
    title: Optional[str] = None


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

        # Assume shape (n_seeds, n_timesteps)
        n_seeds = values.shape[0]

        # Apply smoothing to each seed independently
        if smooth_window > 1:
            # Adjust steps to match the 'valid' convolution length
            steps = steps[:, smooth_window - 1 :]
            smoothed_values = np.array(
                [moving_average(seed, smooth_window) for seed in values]
            )
            values = smoothed_values

        avg_steps = np.mean(steps, axis=0)
        avg_values = np.mean(values, axis=0)
        std_values = np.std(values, axis=0)

        # 95% Confidence Interval
        ci_95 = 1.96 * (std_values / np.sqrt(n_seeds))

        return avg_steps, avg_values, ci_95
    except KeyError:
        print(f" Error: Metric '{metric_name}' not found in {filename}")
        return None, None, None


def main(args: Args) -> None:
    # 1. Automatically find all matching files in child directories
    file_paths = sorted(list(args.root_dir.rglob(args.filename_pattern)))

    if not file_paths:
        print(f"No files found matching '{args.filename_pattern}' in {args.root_dir}")
        return

    print(f"Found {len(file_paths)} runs. Plotting...")

    plt.figure(figsize=(10, 6))
    colors = plt.cm.get_cmap("tab10", len(file_paths))

    for i, file_path in enumerate(file_paths):
        # Use the parent directory name as the label (e.g., 'beta_0.01')
        label = file_path.parent.name

        steps, mean, ci = process_metrics(file_path, args.metric, args.smooth)

        if steps is not None:
            plt.plot(steps, mean, label=label, color=colors(i), linewidth=2)
            plt.fill_between(steps, mean - ci, mean + ci, color=colors(i), alpha=0.2)

    plt.xlabel("Timesteps")
    plt.ylabel(args.metric.replace("_", " ").title())
    plt.title(args.title or f"Comparison: {args.metric}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    save_path = Path("graphs", args.output)
    plt.savefig(save_path )
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
