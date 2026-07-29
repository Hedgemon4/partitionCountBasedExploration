"""Plot intrinsic and extrinsic reward curves for the atari57 beta sweep.

For every game in the sweep this produces two figures:
  - intrinsic.png : intrinsic_return_per_game_ema vs env_step, one line per beta
  - extrinsic.png : extrinsic_return_per_game_ema vs env_step, one line per beta,
                    plus a dashed horizontal PQN baseline (from atari_scores.csv)

Seeds are averaged with a Student-t 95% confidence band. Runs whose metrics.npz
never got written (unfinished sweep configs) are collected and written out to a
failed-configs CSV so they can be re-run.

Standalone script -- deliberately does not import the other (cluttered) plotting
modules in this repo.
"""

import csv
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tyro
from scipy import stats

# The intended sweep grid (see job_scripts/atari/atari57_beta_sweep).
# Beta strings are the literal folder suffixes -- do NOT reformat them.
DEFAULT_BETAS: Tuple[str, ...] = ("1.0", "0.5", "0.25", "0.1", "0.05", "0.01", "0.005", "0.0")
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)

INTRINSIC_METRIC = "intrinsic_return_per_game_ema"
EXTRINSIC_METRIC = "extrinsic_return_per_game_ema"
STEP_KEY = "env_step"


@dataclass
class Args:
    data_dir: Path = Path("data/atari57_beta_sweep")
    """Root directory containing <game>/beta_<value>/seed_<n>/metrics.npz."""
    games_file: Path = Path("atari57_games.txt")
    """Newline-separated list of snake_case game names (blank lines skipped)."""
    baseline_csv: Path = Path("atari_scores.csv")
    """CSV with columns Game,Rainbow,PQN (CamelCase game names)."""
    output_dir: Path = Path("graphs/atari57_beta_sweep_rewards")
    """Where per-game figure folders are written."""
    failed_csv: Path = Path("graphs/atari57_beta_sweep_rewards/failed_configs.csv")
    """Where the list of missing (game, beta, seed) configs is written."""
    smooth_window: int = 1
    """Moving-average window over the time axis (1 = no smoothing)."""
    betas: Tuple[str, ...] = DEFAULT_BETAS
    """Beta folder suffixes to plot, in legend order."""
    seeds: Tuple[int, ...] = DEFAULT_SEEDS
    """Seeds expected per (game, beta)."""


def load_games(games_file: Path) -> List[str]:
    """Return snake_case game names from the games file, skipping blank lines."""
    with open(games_file) as f:
        return [line.strip() for line in f if line.strip()]


def _game_key(name: str) -> str:
    """Normalise a game name for matching across snake_case / CamelCase."""
    return name.replace("_", "").lower()


def load_pqn_baselines(csv_path: Path) -> Dict[str, float]:
    """Map normalised game key -> PQN baseline score from the scores CSV."""
    baselines: Dict[str, float] = {}
    if not csv_path.exists():
        print(f"[warn] baseline csv {csv_path} not found; skipping baselines")
        return baselines
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                baselines[_game_key(row["Game"])] = float(row["PQN"])
            except (KeyError, ValueError):
                continue
    return baselines


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """NaN-aware moving average along a 1-D array; returns same length via 'same'."""
    if w <= 1:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def load_seed_curves(
    data_dir: Path,
    game: str,
    beta: str,
    seeds: Tuple[int, ...],
    metric: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[int]]:
    """Load per-seed curves for one (game, beta).

    Returns (steps, values, missing_seeds) where values is a (n_present, T) array,
    or (None, None, missing) if no seed produced usable data. missing_seeds lists
    seeds whose metrics.npz is absent.
    """
    steps: Optional[np.ndarray] = None
    curves: List[np.ndarray] = []
    missing: List[int] = []
    for seed in seeds:
        metrics_path = data_dir / game / f"beta_{beta}" / f"seed_{seed}" / "metrics.npz"
        if not metrics_path.exists():
            missing.append(seed)
            continue
        try:
            data = np.load(metrics_path)
            s = np.asarray(data[STEP_KEY], dtype=np.float64)
            v = np.asarray(data[metric], dtype=np.float64)
        except (OSError, KeyError, ValueError) as e:
            print(f"[warn] could not read {metrics_path}: {e}")
            missing.append(seed)
            continue
        if steps is None:
            steps = s
        # Defensive: align lengths if a run ever differs.
        n = min(len(steps), len(v))
        if n != len(steps):
            steps = steps[:n]
            curves = [c[:n] for c in curves]
        curves.append(v[:n])

    if not curves:
        return None, None, missing
    values = np.vstack(curves)
    return steps, values, missing


def aggregate(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """NaN-aware mean and Student-t 95% CI across seeds (axis 0)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(values, axis=0)
        n_valid = np.sum(~np.isnan(values), axis=0)
        std = np.nanstd(values, axis=0, ddof=1)
        sem = np.where(n_valid > 1, std / np.sqrt(n_valid), np.nan)
        # t multiplier uses the max valid seed count (constant across the run).
        n_max = int(np.nanmax(n_valid)) if np.any(n_valid > 0) else 0
        t_mult = stats.t.ppf(0.975, max(n_max - 1, 1))
        ci = sem * t_mult
    return mean, ci


def plot_metric(
    game: str,
    metric: str,
    label: str,
    args: Args,
    baselines: Dict[str, float],
    failed: List[Tuple[str, str, int]],
) -> bool:
    """Build one figure (intrinsic or extrinsic) for a game. Returns True if drawn."""
    colors = plt.cm.get_cmap("tab20", len(args.betas))
    fig, ax = plt.subplots(figsize=(10, 6))
    any_line = False

    for i, beta in enumerate(args.betas):
        steps, values, missing = load_seed_curves(
            args.data_dir, game, beta, args.seeds, metric
        )
        for seed in missing:
            failed.append((game, beta, seed))
        if steps is None or values is None:
            continue

        mean, ci = aggregate(values)
        if args.smooth_window > 1:
            mean = moving_average(mean, args.smooth_window)
            ci = moving_average(ci, args.smooth_window)

        # Drop leading columns where the EMA is undefined for every seed.
        valid = ~np.isnan(mean)
        if not np.any(valid):
            continue

        color = colors(i)
        ax.plot(steps[valid], mean[valid], linewidth=2, color=color, label=f"beta={beta}")
        band = ci[valid]
        ax.fill_between(
            steps[valid],
            mean[valid] - band,
            mean[valid] + band,
            color=color,
            alpha=0.15,
        )
        any_line = True

    if not any_line:
        plt.close(fig)
        return False

    # Extrinsic-only PQN baseline.
    if metric == EXTRINSIC_METRIC:
        pqn = baselines.get(_game_key(game))
        if pqn is not None:
            ax.axhline(pqn, linestyle="--", color="black", linewidth=1.5, label="PQN baseline")

    ax.set_xlabel("Env Step")
    ax.set_ylabel(f"{label} Reward")
    ax.set_title(f"{game} - {label} reward")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(fontsize=8, loc="best")

    out_dir = args.output_dir / game
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{label.lower()}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def write_failed_csv(failed: List[Tuple[str, str, int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["game", "beta", "seed"])
        for game, beta, seed in sorted(failed):
            writer.writerow([game, beta, seed])


def main(args: Args) -> None:
    games = load_games(args.games_file)
    baselines = load_pqn_baselines(args.baseline_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    failed: List[Tuple[str, str, int]] = []
    metrics = [(INTRINSIC_METRIC, "Intrinsic"), (EXTRINSIC_METRIC, "Extrinsic")]

    games_plotted = 0
    for game in games:
        drawn = False
        for metric, label in metrics:
            # Only the first metric should record failures, else seeds double-count.
            fail_sink = failed if metric == INTRINSIC_METRIC else []
            if plot_metric(game, metric, label, args, baselines, fail_sink):
                drawn = True
        if drawn:
            games_plotted += 1
            print(f"[ok] {game}")
        else:
            print(f"[skip] {game} (no completed runs)")

    write_failed_csv(failed, args.failed_csv)
    print(
        f"\nDone. Plotted {games_plotted}/{len(games)} games. "
        f"{len(failed)} failed configs written to {args.failed_csv}"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
