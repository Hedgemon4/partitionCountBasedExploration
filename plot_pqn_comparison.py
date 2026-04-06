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
    output: Path = Path("cartpole_beta_sweep")

    filename: Path = Path("extrinsic_return_comparison.png")

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
    file_paths = sorted(list(args.root_dir.rglob(args.filename_pattern)))

    if not file_paths:
        print(f"No files found matching '{args.filename_pattern}' in {args.root_dir}")
        return

    # Use the Object-Oriented interface (fig and ax) to keep plots separate
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.get_cmap("tab20", len(file_paths))

    for i, file_path in enumerate(file_paths):
        label = file_path.parent.name
        steps, mean, ci = process_metrics(file_path, args.metric, args.smooth)

        if steps is not None:
            # We plot on 'ax', which belongs to 'fig'
            ax.plot(steps, mean, label=label, color=colors(i), linewidth=2)
            ax.fill_between(steps, mean - ci, mean + ci, color=colors(i), alpha=0.2)

    # Format the main plot
    ax.set_xlabel("Timesteps")
    ax.set_ylabel(args.metric.replace("_", " ").title())
    ax.set_title(args.title or f"Comparison: {args.metric}")
    ax.grid(True, linestyle="--", alpha=0.7)

    # --- THE TRICK TO SEPARATE ---

    # 1. Grab the handles from the main axis
    handles, labels = ax.get_legend_handles_labels()

    # 2. Save the main plot WITHOUT calling ax.legend()
    save_folder = Path("graphs", args.output)
    save_folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_folder / args.filename, dpi=300, bbox_inches='tight')

    # 3. Create a brand new figure for the legend ONLY
    fig_leg = plt.figure()
    # Add the legend to the new figure using the handles from the old one
    fig_leg.legend(handles, labels, loc='center')

    # 4. Save the legend figure
    legend_path = save_folder / "legend.png"
    fig_leg.savefig(legend_path, dpi=300, bbox_inches='tight')

    # Cleanup
    plt.close(fig)
    plt.close(fig_leg)


if __name__ == "__main__":
    main(tyro.cli(Args))
