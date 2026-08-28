"""Per-game summary of the Atari-57 count-layer sweep.

Layout it expects (as produced by job_scripts/atari/atari57_count_layer_sweep):

    <root>/<game>/<position>/beta_<b>/next_0.0/seed_<s>/metrics.npz

57 games x 3 count positions x 8 betas, next_state_coef fixed at 0.0, so 24
combinations per game and 1368 in total.

This is the wide-sweep counterpart to plot_count_layer_sweep.py, which was built
for two games: at 57 games that script emits 1140 figures, and with only 24 cells
per game its "top 10 of 45" and per-position "top 5 of 15" selections stop being
selective. Here each game gets four figures and one row per selected combination
in a shared table, and the per-position figure family is dropped.

For each ranking metric it writes:

    <score>/summary.csv                         top 10 + best-per-beta, all games
    <score>/<game>/top10_extrinsic.png
    <score>/<game>/top10_intrinsic.png
    <score>/<game>/best_per_beta_extrinsic.png
    <score>/<game>/best_per_beta_intrinsic.png

Combinations are always ranked by the *extrinsic* metric; the intrinsic figure
shows the same selection, so a pair of figures describes one set of runs. The
best-per-beta series are ordered by beta rather than by rank, so a beta keeps its
colour between the two metrics.

Everything except the selection and reporting is imported from
plot_count_layer_sweep, the way analyze_beta_vs_baseline.py builds on
plot_beta_sweep_rewards.py -- so smoothing, seed aggregation, the Student-t band,
the palette and the axis styling stay defined in exactly one place.

While the sweep is still running some cells have fewer than five finished seeds.
Those are excluded from ranking by default (--min-seeds) and listed in
incomplete.csv, because a two-seed cell can otherwise take a top-10 slot on luck.
"""

from __future__ import annotations

import csv
import dataclasses
import math
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import tyro

from plot_count_layer_sweep import Combo, load_combos, plot_curves, score_combo


@dataclasses.dataclass(frozen=True)
class Args:
    root_dir: Path = Path("data/atari57_count_layer_sweep")
    """Sweep directory containing <game>/<position>/beta_*/next_*/seed_*/."""
    output_dir: Path = Path("graphs/atari57_count_layer_sweep_final")
    """Written to <output_dir>/<score>/ -- summary.csv plus one dir per game."""
    scores: tuple[Literal["final", "auc"], ...] = ("final", "auc")
    """Ranking metrics; each gets its own output tree."""
    games: tuple[str, ...] = ()
    """Restrict to these games. Empty means all of them."""
    top_k: int = 10
    """How many combinations in the top-N figures and rows."""
    min_seeds: int = 5
    """Combinations with fewer finished seeds are excluded and reported instead."""
    final_frac: float = 0.1
    """Fraction of the run averaged for the "final" score."""
    smooth: int = 750
    """Rolling-mean window in updates. 1 disables smoothing."""
    band_max_series: int = 10
    """Cap above which 95% CI bands are dropped."""
    theme: Literal["light", "dark"] = "light"
    dpi: int = 300
    figsize: tuple[float, float] = (10.0, 6.0)


def series_label(combo: Combo) -> str:
    """next_state_coef is 0.0 sweep-wide, so it is left out of the legend."""
    return f"{combo.position}  β={combo.beta}"


def write_rows(rows: list[dict], fieldnames: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


SUMMARY_FIELDS = (
    "game",
    "selection",
    "rank",
    "position",
    "beta",
    "n_seeds",
    "score",
    "final_extrinsic",
    "final_intrinsic",
)


def main(args: Args) -> None:
    combos, failures = load_combos(args.root_dir)
    if not combos:
        raise SystemExit(f"no runs found under {args.root_dir}")
    if args.games:
        wanted = set(args.games)
        combos = [c for c in combos if c.game in wanted]
        if not combos:
            raise SystemExit(f"no runs for games {sorted(wanted)}")

    complete = [c for c in combos if c.n_seeds >= args.min_seeds]
    incomplete = [c for c in combos if c.n_seeds < args.min_seeds]
    games = sorted({c.game for c in complete})
    print(
        f"{len(combos)} combinations over {len({c.game for c in combos})} games; "
        f"{len(complete)} complete (>= {args.min_seeds} seeds), {len(incomplete)} not"
    )

    if failures:
        write_rows(
            failures,
            ("game", "position", "beta", "next", "seed", "error"),
            args.output_dir / "unreadable_runs.csv",
        )
        print(f"  {len(failures)} unreadable runs -> unreadable_runs.csv")

    if incomplete:
        write_rows(
            [
                {
                    "game": c.game,
                    "position": c.position,
                    "beta": c.beta,
                    "n_seeds": c.n_seeds,
                }
                for c in sorted(incomplete, key=lambda c: (c.game, c.position, float(c.beta)))
            ],
            ("game", "position", "beta", "n_seeds"),
            args.output_dir / "incomplete.csv",
        )
        print(f"  {len(incomplete)} incomplete combinations -> incomplete.csv")

    if not games:
        raise SystemExit(
            f"every combination has fewer than {args.min_seeds} seeds; "
            "lower --min-seeds to plot the sweep as it stands"
        )

    def tail_mean(combo: Combo, metric: str) -> float:
        mean, _ = combo.aggregated(metric, args.smooth)
        tail = math.ceil(mean.shape[0] * args.final_frac)
        return float(np.nanmean(mean[-tail:]))

    for score in args.scores:
        score_label = {"final": "final performance", "auc": "area under the curve"}[
            score
        ]
        out = args.output_dir / score
        rows: list[dict] = []

        for game in games:
            ranked = sorted(
                (c for c in complete if c.game == game),
                key=lambda c: score_combo(c, score, args.final_frac, args.smooth),
                reverse=True,
            )
            top = ranked[: args.top_k]
            # Best combination at each beta: `ranked` is already sorted, so the
            # first match for a beta is its best. Ordered by beta, not by rank.
            best_per_beta = [
                next(c for c in ranked if c.beta == beta)
                for beta in sorted({c.beta for c in ranked}, key=float)
            ]

            for metric in ("extrinsic", "intrinsic"):
                plot_curves(
                    top,
                    metric=metric,
                    title=f"{game} — top {len(top)} combinations by {score_label}",
                    subtitle=(
                        f"{metric} return, mean of {args.min_seeds} seeds with 95% CI"
                    ),
                    path=out / game / f"top{args.top_k}_{metric}.png",
                    args=args,
                    label_fn=series_label,
                )
                plot_curves(
                    best_per_beta,
                    metric=metric,
                    title=f"{game} — best combination per β by {score_label}",
                    subtitle=(
                        f"{metric} return, mean of {args.min_seeds} seeds with "
                        "95% CI · series ordered by β"
                    ),
                    path=out / game / f"best_per_beta_{metric}.png",
                    args=args,
                    label_fn=series_label,
                )

            def row(combo: Combo, selection: str, rank: object) -> dict:
                return {
                    "game": combo.game,
                    "selection": selection,
                    "rank": rank,
                    "position": combo.position,
                    "beta": combo.beta,
                    "n_seeds": combo.n_seeds,
                    "score": round(
                        score_combo(combo, score, args.final_frac, args.smooth), 3
                    ),
                    "final_extrinsic": round(tail_mean(combo, "extrinsic"), 3),
                    "final_intrinsic": round(tail_mean(combo, "intrinsic"), 3),
                }

            rows += [row(c, "top10", i + 1) for i, c in enumerate(top)]
            rows += [row(c, "best_per_beta", "") for c in best_per_beta]

        write_rows(rows, SUMMARY_FIELDS, out / "summary.csv")
        print(
            f"  {score}: {len(rows)} rows, "
            f"{len(list(out.glob('*/*.png')))} figures -> {out}"
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
