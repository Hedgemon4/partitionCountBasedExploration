"""Histogram of FTA-bin usage for the best-performing config per beta value.

Scans data/mountaincar_longer_runs, filters runs with total_time_steps == 5e5,
finds the best run per beta (ranked by last-10% mean of extrinsic_return_ema),
and produces a bin-usage histogram for each.

The counts array in each run has shape (seeds, actions, hidden_units, bins).
Each bar in the histogram is the mean across seeds of the per-seed per-bin
total (summed across neurons), and is stacked by action so each colour shows
the contribution from one action. Error bars come from the std across seeds
of the total per-bin counts, and individual seeds are overlaid as dots.

Helper functions (`aggregate_counts`, `find_outlier_seeds`,
`plot_histogram_with_actions`) are reused by top_k_plotting_script.py so every
histogram across the project has the same format.

Outputs go to graphs/mountaincar_longer_runs/count_histograms/.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tyro
import yaml


ACTION_COLORS = ["#4C72B0", "#DD8452", "#55A467"]  # up to 3 actions (MountainCar)
ACTION_LABELS_DEFAULT = {
    3: ["Action 0 (push left)", "Action 1 (no-op)", "Action 2 (push right)"],
    2: ["Action 0 (left)", "Action 1 (right)"],
}


@dataclass
class Args:
    root_dir: Path = Path("data/mountaincar_longer_runs")
    output_dir: Path = Path("graphs/mountaincar_longer_runs/count_histograms")
    metric: str = "extrinsic_return_ema"
    total_time_steps: float = 5e5
    # If set, only plot this beta. Otherwise plot the best config per beta.
    beta: Optional[float] = None


# ---------------------------------------------------------------------------
# Generic helpers (also imported by top_k_plotting_script.py)
# ---------------------------------------------------------------------------


def score_run(folder: Path, metric: str) -> Optional[float]:
    m_file = folder / "metrics.npz"
    if not m_file.exists():
        return None
    data = np.load(m_file)
    if metric not in data:
        return None
    values = data[metric]
    last_10 = max(1, int(values.shape[1] * 0.1))
    return float(np.mean(values[:, -last_10:]))


def aggregate_counts(counts: np.ndarray):
    """Aggregate a counts array with shape (seeds, actions, neurons, bins).

    Returns a dict with:
        per_seed_action : (seeds, actions, bins)  sum over neurons
        per_seed        : (seeds, bins)           per_seed_action summed over actions
        action_mean     : (actions, bins)         per_seed_action mean over seeds
        per_bin_mean    : (bins,)                 per_seed mean over seeds
        per_bin_std     : (bins,)                 per_seed std  over seeds (ddof=1)
    """
    per_seed_action = counts.sum(axis=2)  # (seeds, actions, bins)
    per_seed = per_seed_action.sum(axis=1)  # (seeds, bins)
    action_mean = per_seed_action.mean(axis=0)  # (actions, bins)
    per_bin_mean = per_seed.mean(axis=0)
    per_bin_std = per_seed.std(axis=0, ddof=1) if per_seed.shape[0] > 1 else np.zeros_like(per_bin_mean)
    return {
        "per_seed_action": per_seed_action,
        "per_seed": per_seed,
        "action_mean": action_mean,
        "per_bin_mean": per_bin_mean,
        "per_bin_std": per_bin_std,
    }


def find_outlier_seeds(
    per_seed: np.ndarray,
    per_seed_action: np.ndarray,
    z_threshold: float = 2.0,
) -> List[dict]:
    """Detect seeds that look anomalous.

    Flags a seed if:
    (a) it has a bin with zero total counts (summed over actions and neurons), or
    (b) any (action, bin) cell (summed over neurons) is zero, or
    (c) any per-bin total is more than `z_threshold` std away from the seed mean.

    Returns a list of dicts, one per flagged seed.
    """
    n_seeds, n_bins = per_seed.shape
    mean = per_seed.mean(axis=0)
    std = per_seed.std(axis=0, ddof=1) if n_seeds > 1 else np.zeros_like(mean)

    outliers: List[dict] = []
    for s in range(n_seeds):
        reasons: List[str] = []
        zero_bins = np.where(per_seed[s] == 0)[0]
        if zero_bins.size > 0:
            reasons.append(f"zero-count bins {zero_bins.tolist()}")
        zero_action_bins: List[Tuple[int, int]] = []
        for a in range(per_seed_action.shape[1]):
            for b in np.where(per_seed_action[s, a] == 0)[0]:
                zero_action_bins.append((a, int(b)))
        if zero_action_bins:
            reasons.append(f"zero (action,bin) cells {zero_action_bins}")
        if std.max() > 0:
            z = np.where(std > 0, (per_seed[s] - mean) / np.where(std > 0, std, 1), 0)
            big = np.where(np.abs(z) > z_threshold)[0]
            if big.size > 0:
                details = ", ".join(f"bin {b} z={z[b]:+.2f}" for b in big)
                reasons.append(f"|z|>{z_threshold}: {details}")
        if reasons:
            outliers.append({"seed": s, "reasons": reasons})
    return outliers


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _action_labels(n_actions: int) -> List[str]:
    if n_actions in ACTION_LABELS_DEFAULT:
        return ACTION_LABELS_DEFAULT[n_actions]
    return [f"Action {a}" for a in range(n_actions)]


def plot_histogram_with_actions(
    counts: np.ndarray,
    title: str,
    out_path: Path,
    highlight: bool = False,
    ax: Optional[plt.Axes] = None,
    show_legend: bool = True,
    show_outlier_box: bool = True,
) -> Tuple[plt.Axes, List[dict], dict]:
    """Plot a histogram of bin usage, with bars stacked by action.

    * Each bin has one stacked bar showing the mean per-seed counts for each action.
    * A single error bar on top of each stack shows the seed std of the total.
    * Each seed's total is overlaid as a jittered dot so outliers are visible.
    * A text label above each bar shows the percentage of total counts in that bin.
    * A text box lists outlier seeds (if any).

    Returns (ax, outliers, aggregates).
    """
    agg = aggregate_counts(counts)
    per_bin_mean = agg["per_bin_mean"]
    per_bin_std = agg["per_bin_std"]
    per_seed = agg["per_seed"]
    per_seed_action = agg["per_seed_action"]
    action_mean = agg["action_mean"]

    n_bins = per_bin_mean.shape[0]
    n_actions = action_mean.shape[0]
    bins = np.arange(n_bins)
    labels = _action_labels(n_actions)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 6))
        created_fig = True
    else:
        fig = ax.figure

    # Stacked action bars
    bottom = np.zeros(n_bins)
    edgecolor = "black" if highlight else "#333333"
    for a in range(n_actions):
        ax.bar(
            bins,
            action_mean[a],
            bottom=bottom,
            color=ACTION_COLORS[a % len(ACTION_COLORS)],
            edgecolor=edgecolor,
            linewidth=0.7 if not highlight else 1.2,
            label=labels[a],
        )
        bottom = bottom + action_mean[a]

    # Error bar on totals
    ax.errorbar(
        bins,
        per_bin_mean,
        yerr=per_bin_std,
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=4,
        zorder=5,
    )

    # Per-seed dots (totals)
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.18, 0.18, size=per_seed.shape[0])
    for seed_idx in range(per_seed.shape[0]):
        ax.scatter(
            bins + jitter[seed_idx],
            per_seed[seed_idx],
            s=8,
            color="black",
            alpha=0.28,
            zorder=6,
        )

    # Percentage labels above each bar
    total = per_bin_mean.sum()
    ymax = (per_bin_mean + per_bin_std).max()
    for i, (m, s) in enumerate(zip(per_bin_mean, per_bin_std)):
        pct = 100.0 * m / total if total > 0 else 0.0
        ax.text(
            i,
            m + s + 0.02 * ymax,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xlabel("FTA Bin Index")
    ax.set_ylabel("Per-seed count (summed across neurons)")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(bins)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.margins(y=0.20)
    if show_legend:
        ax.legend(loc="upper center", fontsize=8, ncol=n_actions, frameon=True)

    # Outlier annotation
    outliers = find_outlier_seeds(per_seed, per_seed_action)
    if show_outlier_box:
        if outliers:
            lines = [f"Outlier seeds ({len(outliers)}):"]
            for o in outliers[:6]:
                lines.append(f"  seed {o['seed']}: " + "; ".join(o["reasons"]))
            if len(outliers) > 6:
                lines.append(f"  ... and {len(outliers) - 6} more")
            msg = "\n".join(lines)
            ax.text(
                0.01,
                0.98,
                msg,
                transform=ax.transAxes,
                fontsize=7,
                va="top",
                ha="left",
                family="monospace",
                bbox=dict(facecolor="#fff5cc", edgecolor="#cc9900", boxstyle="round,pad=0.4"),
            )
        else:
            ax.text(
                0.01,
                0.98,
                "No outlier seeds (no zero bins, no |z|>2)",
                transform=ax.transAxes,
                fontsize=7,
                va="top",
                ha="left",
                family="monospace",
                bbox=dict(facecolor="#e6f4ea", edgecolor="#44aa66", boxstyle="round,pad=0.4"),
            )

    if created_fig:
        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}")
    return ax, outliers, agg


def plot_neuron_summary(
    counts: np.ndarray,
    title_prefix: str,
    out_path: Path,
):
    """Summarize how bin activations vary from neuron to neuron and across seeds.

    counts shape: (seeds, actions, neurons, bins).

    Produces a 4-panel figure:
      1. Heatmap of the seed-mean bin distribution per neuron. Each row is a
         neuron, each column a bin, cell colour is the probability of that bin
         given the neuron fired (sum over actions, normalised per neuron).
         Neurons are sorted by their dominant (argmax) bin so similar neurons
         cluster together.
      2. Heatmap of the seed std of the same per-neuron distribution
         (same sorting as panel 1).
      3. Per-neuron distribution entropy (nats) with a shaded +/- 1 std
         band across seeds. Low entropy = neuron concentrates on few bins,
         high entropy = neuron spreads activity across many bins.
      4. Boxplot of per-seed dominant-bin indices across neurons: for each
         bin, how many neurons (out of N) pick it as their argmax on each
         seed, giving a sense of how stable the neuron-to-bin assignment is.
    """
    S, A, N, B = counts.shape

    # Per-seed per-neuron bin distribution (sum over actions, normalise)
    per_seed_neuron = counts.sum(axis=1)  # (S, N, B)
    totals = per_seed_neuron.sum(axis=2, keepdims=True)
    totals = np.where(totals == 0, 1, totals)
    probs = per_seed_neuron / totals  # (S, N, B)

    mean_probs = probs.mean(axis=0)  # (N, B)
    std_probs = probs.std(axis=0, ddof=1) if S > 1 else np.zeros_like(mean_probs)

    # Per-neuron per-seed entropy (nats); clip for numerical safety
    safe = np.clip(probs, 1e-12, 1.0)
    entropy = -(probs * np.log(safe)).sum(axis=2)  # (S, N)
    ent_mean = entropy.mean(axis=0)  # (N,)
    ent_std = entropy.std(axis=0, ddof=1) if S > 1 else np.zeros_like(ent_mean)

    # Sort neurons by their seed-mean argmax bin (stable cross-panel ordering)
    order = np.argsort(mean_probs.argmax(axis=1))
    mean_probs_sorted = mean_probs[order]
    std_probs_sorted = std_probs[order]
    ent_mean_sorted = ent_mean[order]
    ent_std_sorted = ent_std[order]

    # Per-seed "how many neurons pick each bin as their argmax"
    argmax_per_seed = probs.argmax(axis=2)  # (S, N)
    bin_counts_per_seed = np.zeros((S, B), dtype=int)
    for s in range(S):
        for b in range(B):
            bin_counts_per_seed[s, b] = int((argmax_per_seed[s] == b).sum())

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0], hspace=0.32, wspace=0.25)

    # --- Panel 1: mean probability heatmap ---------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(
        mean_probs_sorted,
        aspect="auto",
        cmap="viridis",
        origin="lower",
    )
    ax1.set_title("(a) Mean bin probability per neuron  (averaged over seeds)", fontsize=11)
    ax1.set_xlabel("FTA Bin Index")
    ax1.set_ylabel(f"Neuron index (sorted by argmax bin; {N} neurons)")
    ax1.set_xticks(np.arange(B))
    fig.colorbar(im1, ax=ax1, label="P(bin | neuron)")

    # --- Panel 2: std of probability heatmap -------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(
        std_probs_sorted,
        aspect="auto",
        cmap="magma",
        origin="lower",
    )
    ax2.set_title("(b) Seed std of bin probability per neuron", fontsize=11)
    ax2.set_xlabel("FTA Bin Index")
    ax2.set_ylabel(f"Neuron index (same sort as (a))")
    ax2.set_xticks(np.arange(B))
    fig.colorbar(im2, ax=ax2, label="std across seeds")

    # --- Panel 3: per-neuron entropy with +/- std band ---------------------
    ax3 = fig.add_subplot(gs[1, 0])
    x = np.arange(N)
    ax3.plot(x, ent_mean_sorted, color="#2b7bba", linewidth=1.8, label="mean")
    ax3.fill_between(
        x,
        ent_mean_sorted - ent_std_sorted,
        ent_mean_sorted + ent_std_sorted,
        color="#2b7bba",
        alpha=0.25,
        label="+/- 1 std across seeds",
    )
    max_ent = np.log(B)
    ax3.axhline(
        max_ent,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label=f"uniform entropy ln({B}) = {max_ent:.3f}",
    )
    ax3.set_xlabel("Neuron index (sorted by argmax bin)")
    ax3.set_ylabel("Entropy of bin distribution (nats)")
    ax3.set_title(
        "(c) Per-neuron distribution entropy  "
        "(low = concentrated, high = spread out)",
        fontsize=11,
    )
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="best", fontsize=8)

    # --- Panel 4: how many neurons pick each bin as their argmax -----------
    ax4 = fig.add_subplot(gs[1, 1])
    bp = ax4.boxplot(
        [bin_counts_per_seed[:, b] for b in range(B)],
        positions=np.arange(B),
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="#cce5ff", edgecolor="#1f5c99"),
        medianprops=dict(color="#c0392b", linewidth=2),
        flierprops=dict(marker="o", markersize=3, markerfacecolor="black", alpha=0.5),
    )
    # Overlay per-seed dots jittered
    rng = np.random.default_rng(0)
    for b in range(B):
        jitter = rng.uniform(-0.15, 0.15, size=S)
        ax4.scatter(
            np.full(S, b) + jitter,
            bin_counts_per_seed[:, b],
            s=10,
            color="black",
            alpha=0.35,
            zorder=3,
        )
    ax4.set_xticks(np.arange(B))
    ax4.set_xlabel("FTA Bin Index")
    ax4.set_ylabel(f"Neurons with this bin as argmax  (out of {N})")
    ax4.set_title(
        "(d) Per-seed count of neurons whose dominant bin is k  "
        "(shows neuron-to-bin assignment stability)",
        fontsize=11,
    )
    ax4.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle(
        f"Per-neuron bin-activation summary  |  {title_prefix}\n"
        f"shape: {S} seeds x {A} actions x {N} neurons x {B} bins",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_combined(best: dict, out_path: Path, best_beta: float):
    """Side-by-side subplots: one histogram per beta, with a shared y-axis."""
    betas = sorted(best.keys())
    n = len(betas)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), sharey=True)
    if n == 1:
        axes = [axes]

    last_cfg = None
    for idx, (ax, beta) in enumerate(zip(axes, betas)):
        score, folder, cfg = best[beta]
        last_cfg = cfg
        counts = np.load(folder / "counts.npy")
        star = "  *" if beta == best_beta else ""
        sub_title = f"beta={beta}{star}  |  {folder.name}  (score={score:.1f})"
        plot_histogram_with_actions(
            counts,
            title=sub_title,
            out_path=out_path,
            highlight=(beta == best_beta),
            ax=ax,
            show_legend=(idx == 0),
            show_outlier_box=True,
        )
        if idx != 0:
            ax.set_ylabel("")

    fig.suptitle(
        f"FTA bin usage: best config per beta  "
        f"(tts={int(last_cfg['total_time_steps']):,}, "
        f"error bars = seed std, dots = individual seeds, stacks = actions)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def find_best_per_beta(root: Path, tts: float, metric: str):
    """Return {beta: (score, folder, cfg)} for the best run per beta at tts."""
    by_beta = defaultdict(list)
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        cf = d / "config.yaml"
        if not cf.exists():
            continue
        with open(cf) as f:
            cfg = yaml.safe_load(f)
        if cfg.get("total_time_steps") != tts:
            continue
        if not (d / "counts.npy").exists():
            continue
        s = score_run(d, metric)
        if s is None:
            continue
        by_beta[cfg["beta"]].append((s, d, cfg))

    best = {}
    for beta, runs in by_beta.items():
        runs.sort(key=lambda x: -x[0])
        best[beta] = runs[0]
    return best


def main(args: Args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = find_best_per_beta(args.root_dir, args.total_time_steps, args.metric)
    if not best:
        print("No matching runs found.")
        return

    if args.beta is not None:
        best = {args.beta: best[args.beta]}

    nonzero = {b: v for b, v in best.items() if b > 0}
    best_beta = (
        max(nonzero, key=lambda b: nonzero[b][0])
        if nonzero
        else max(best, key=lambda b: best[b][0])
    )

    print(
        f"Best beta (by {args.metric}): {best_beta}  "
        f"(config={best[best_beta][1].name}, score={best[best_beta][0]:.2f})"
    )

    # Per-beta plots
    for beta in sorted(best.keys()):
        score, folder, cfg = best[beta]
        counts = np.load(folder / "counts.npy")
        title = (
            f"FTA bin usage  |  beta={beta}  |  {folder.name}\n"
            f"hidden={cfg['network']['hidden_size']}, "
            f"bound={cfg['network']['activation1']['bound']}, "
            f"eta={cfg['network']['activation1']['eta']}, "
            f"score={score:.2f}  (error bars: seed std; stacks: per-action contribution)"
        )
        out = args.output_dir / f"hist_beta_{beta}_{folder.name}.png"
        plot_histogram_with_actions(
            counts, title=title, out_path=out,
            highlight=(beta == best_beta),
        )
        # Per-neuron summary for the same config
        neuron_out = args.output_dir / f"neuron_summary_beta_{beta}_{folder.name}.png"
        plot_neuron_summary(
            counts,
            title_prefix=f"beta={beta}  |  {folder.name}  (score={score:.2f})",
            out_path=neuron_out,
        )

    if len(best) > 1:
        plot_combined(
            best,
            args.output_dir / "hist_best_per_beta_comparison.png",
            best_beta,
        )

    # Dedicated plot for the single best beta
    score, folder, cfg = best[best_beta]
    counts = np.load(folder / "counts.npy")
    title = (
        f"FTA bin usage for BEST beta  |  beta={best_beta}  |  {folder.name}\n"
        f"last-10% {args.metric} = {score:.2f}  (error bars: seed std; stacks: per-action)"
    )
    plot_histogram_with_actions(
        counts,
        title=title,
        out_path=args.output_dir / f"hist_BEST_beta_{best_beta}.png",
        highlight=True,
    )
    plot_neuron_summary(
        counts,
        title_prefix=f"BEST beta={best_beta}  |  {folder.name}  (score={score:.2f})",
        out_path=args.output_dir / f"neuron_summary_BEST_beta_{best_beta}.png",
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
