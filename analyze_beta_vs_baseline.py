"""Analyze which betas outperformed the beta=0.0 baseline in the atari57 beta sweep.

For every game we compute a per-seed final extrinsic score (mean over the last
`last_frac` of training), then compare each beta against beta=0.0:
  - outperformed : beta's mean score > beta0's mean score
  - significant  : beta's 95% CI does not overlap beta0's 95% CI
                   (beta_mean - beta_ci > beta0_mean + beta0_ci)

Results are written to a CSV and a summary is printed. Reuses the NaN-aware data
loaders from plot_beta_sweep_rewards.py.
"""

import csv
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import tyro
from scipy import stats

from plot_beta_sweep_rewards import (
    DEFAULT_BETAS,
    DEFAULT_SEEDS,
    EXTRINSIC_METRIC,
    load_games,
    load_seed_curves,
)


@dataclass
class Args:
    data_dir: Path = Path("data/atari57_beta_sweep")
    """Root directory containing <game>/beta_<value>/seed_<n>/metrics.npz."""
    games_file: Path = Path("atari57_games.txt")
    """Newline-separated list of snake_case game names."""
    output_csv: Path = Path("graphs/atari57_beta_sweep_rewards/beta_vs_baseline.csv")
    """Where the comparison table is written."""
    baseline_beta: str = "0.0"
    """The beta treated as the no-intrinsic-reward baseline."""
    last_frac: float = 0.10
    """Fraction of the run (final window) used to compute each seed's score."""
    betas: Tuple[str, ...] = DEFAULT_BETAS
    """Beta folder suffixes to consider."""
    seeds: Tuple[int, ...] = DEFAULT_SEEDS
    """Seeds expected per (game, beta)."""


def per_seed_final_scores(values: np.ndarray, last_frac: float) -> np.ndarray:
    """Mean of the final `last_frac` columns per seed -> 1-D array (one per seed)."""
    n_cols = values.shape[1]
    window = max(1, math.ceil(last_frac * n_cols))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(values[:, -window:], axis=1)


def mean_ci(scores: np.ndarray) -> Tuple[float, float, int]:
    """Return (mean, student-t 95% CI half-width, n_valid) over per-seed scores.

    CI is NaN when fewer than 2 valid seeds (can't estimate spread).
    """
    valid = scores[~np.isnan(scores)]
    n = int(valid.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(np.mean(valid))
    if n < 2:
        return mean, float("nan"), n
    sem = float(np.std(valid, ddof=1)) / math.sqrt(n)
    ci = sem * float(stats.t.ppf(0.975, n - 1))
    return mean, ci, n


def score_for(
    args: Args, game: str, beta: str
) -> Tuple[Optional[float], Optional[float], int]:
    """Load a (game, beta) and return (mean, ci, n_seeds). n=0 if no usable data."""
    _, values, _ = load_seed_curves(
        args.data_dir, game, beta, args.seeds, EXTRINSIC_METRIC
    )
    if values is None:
        return None, None, 0
    return mean_ci(per_seed_final_scores(values, args.last_frac))


def main(args: Args) -> None:
    games = load_games(args.games_file)

    rows: List[dict] = []
    winners_by_game: dict = {}
    skipped_no_baseline: List[str] = []

    for game in games:
        base_mean, base_ci, base_n = score_for(args, game, args.baseline_beta)
        if base_n < 2 or base_mean is None or math.isnan(base_ci):
            skipped_no_baseline.append(game)
            continue

        game_winners: List[str] = []
        for beta in args.betas:
            if beta == args.baseline_beta:
                continue
            b_mean, b_ci, b_n = score_for(args, game, beta)
            if b_n == 0 or b_mean is None:
                continue  # beta has no data for this game

            diff = b_mean - base_mean
            outperformed = diff > 0
            significant = (
                b_n >= 2
                and not math.isnan(b_ci)
                and (b_mean - b_ci) > (base_mean + base_ci)
            )
            if outperformed and significant:
                game_winners.append(beta)

            rows.append(
                {
                    "game": game,
                    "beta": beta,
                    "n_seeds": b_n,
                    "beta_mean": round(b_mean, 4),
                    "beta_ci": round(b_ci, 4) if not math.isnan(b_ci) else "",
                    "baseline_n_seeds": base_n,
                    "baseline_mean": round(base_mean, 4),
                    "baseline_ci": round(base_ci, 4),
                    "diff": round(diff, 4),
                    "outperformed": outperformed,
                    "significant": significant,
                }
            )
        winners_by_game[game] = game_winners

    # Write CSV.
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "game", "beta", "n_seeds", "beta_mean", "beta_ci",
        "baseline_n_seeds", "baseline_mean", "baseline_ci",
        "diff", "outperformed", "significant",
    ]
    rows.sort(key=lambda r: (r["game"], args.betas.index(r["beta"])))
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary.
    n_outperformed = sum(1 for r in rows if r["outperformed"])
    n_significant = sum(1 for r in rows if r["significant"])
    games_with_winner = [g for g, w in winners_by_game.items() if w]

    # Per-game rollup: does ANY beta beat beta=0.0 (on the mean / significantly)?
    games_any_better = {
        g: [r["beta"] for r in rows if r["game"] == g and r["outperformed"]]
        for g in winners_by_game
    }
    games_better_on_mean = [g for g, bs in games_any_better.items() if bs]
    games_none_better = [g for g in winners_by_game if g not in games_better_on_mean]

    print("\n=== Per-game: is there at least one beta better than beta=0.0? ===")
    print(
        f"  better on mean       : {len(games_better_on_mean)}/{len(winners_by_game)} games"
    )
    print(
        f"  significantly better : {len(games_with_winner)}/{len(winners_by_game)} games"
    )
    print("  no beta beats beta0 (even on mean):")
    print("    " + (", ".join(sorted(games_none_better)) if games_none_better else "(none)"))

    print("\n=== Games with a beta that SIGNIFICANTLY outperformed beta=0.0 ===")
    for game in sorted(games_with_winner):
        betas = ", ".join(winners_by_game[game])
        print(f"  {game:20s} -> {betas}")

    print("\n=== Games compared but with NO significant winner ===")
    no_winner = [
        g for g in winners_by_game if g not in games_with_winner
    ]
    print("  " + (", ".join(sorted(no_winner)) if no_winner else "(none)"))

    if skipped_no_baseline:
        print("\n=== Skipped (no usable beta=0.0 baseline, <2 seeds) ===")
        print("  " + ", ".join(sorted(skipped_no_baseline)))

    print(
        f"\nComparisons: {len(rows)} (game, beta) pairs across "
        f"{len(winners_by_game)} games with a baseline.\n"
        f"  outperformed beta0 : {n_outperformed}\n"
        f"  significant winners: {n_significant}\n"
        f"  games with >=1 beta better on mean: {len(games_better_on_mean)}\n"
        f"  games with >=1 significant winner : {len(games_with_winner)}\n"
        f"  games skipped (no baseline)       : {len(skipped_no_baseline)}\n"
        f"CSV written to {args.output_csv}"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
