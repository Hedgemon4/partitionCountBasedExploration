"""Group single-seed run folders into multi-seed logical runs.

Some sweeps (e.g. ``data/freeway_sweep``) write **one run folder per
(hyperparameter, seed) combination**. Each ``metrics.npz`` in those folders
holds 1-D arrays for that single seed.

The plotting scripts (``top_k_plotting_script.py`` and ``best_beta.py``) were
written for the *other* layout, where a single folder already contains a
2-D ``(n_seeds, n_timesteps)`` array so they can draw a mean +/- CI band
across seeds.

This module bridges the two layouts. It scans a root directory, groups the
folders that share the same *varied* hyperparameters, and stacks their
per-seed metric arrays into the 2-D layout the plotting code expects.

For the freeway sweep only ``beta`` and ``network.next_state_coef`` were
varied, so a "group" is one ``(beta, next_state_coef)`` pair and it collects
all 10 seeds for that pair.

Typical use from a plotting script::

    from sweep_grouping import build_grouped_runs, format_group_label

    runs = build_grouped_runs(
        root_dir,
        group_keys=("beta", "network.next_state_coef"),
        metric_names=("extrinsic_return_ema", "intrinsic_return_ema"),
    )
    for g in runs:
        ext = g.metrics["extrinsic_return_ema"]   # (n_seeds, n_timesteps)
        steps = g.steps                            # (n_timesteps,)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

# Hyperparameters that were varied in the freeway sweep. Dot notation is used
# for nested config keys (so ``network.next_state_coef`` reaches into the
# ``network`` sub-mapping of config.yaml).
DEFAULT_GROUP_KEYS: Tuple[str, ...] = ("beta", "network.next_state_coef")

# Sentinel group key: group by the run folder's top-level subdirectory name
# (relative to ``root_dir``) instead of a config value. Used by sweeps that
# encode the varied dimension in the directory layout rather than a scalar
# config key — e.g. ``data/venture_arch_sweep/<arch>/seed_<N>``, where the
# architecture is only distinguishable by the ``<arch>`` directory.
DIR_GROUP_KEY: str = "__dir__"

# The intrinsic + extrinsic reward metrics. These are the only metrics we plot
# right now, so they are also the only ones the grouper loads by default.
DEFAULT_METRICS: Tuple[str, ...] = ("extrinsic_return_ema", "intrinsic_return_ema")


def flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """Flatten a nested dict using dot notation for nested keys.

    ``{"network": {"next_state_coef": 0.5}}`` becomes
    ``{"network.next_state_coef": 0.5}``.
    """
    items: List[Tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


@dataclass
class GroupedRun:
    """One logical run: all seeds that share the same varied hyperparameters.

    Attributes
    ----------
    group_keys / group_values
        The hyperparameter names and the specific values that define this
        group, e.g. ``("beta", "network.next_state_coef")`` and ``(0.1, 0.5)``.
    config
        A flattened copy of one member's ``config.yaml`` (representative of the
        whole group; only the seed differs between members).
    folders
        The run folders that were merged into this group, ordered by seed.
    seeds
        The seed value of each member, in the same order as ``folders``.
    steps
        The shared 1-D ``env_step`` array (truncated to the shortest member).
    metrics
        Maps a metric name to a 2-D ``(n_seeds, n_timesteps)`` array, with one
        row per seed. This is exactly what the plotting scripts expect.
    """

    group_keys: Tuple[str, ...]
    group_values: Tuple[Any, ...]
    config: Dict[str, Any]
    folders: List[Path]
    seeds: List[Any]
    steps: np.ndarray
    metrics: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_seeds(self) -> int:
        return len(self.seeds)

    def get(self, key: str, default: Any = None) -> Any:
        """Look up a (flattened, dot-notation) config key for this group."""
        return self.config.get(key, default)

    def label(self) -> str:
        """A short, human-readable label built from the varied hyperparameters."""
        return format_group_label(self)


def format_group_label(grouped: GroupedRun) -> str:
    """Build a concise legend label from a group's varied hyperparameters.

    Only the keys that actually define the group are shown (plus the seed
    count), so legends stay readable instead of listing every constant config
    value. Example: ``"beta=0.1 | next_state_coef=0.5 | n=10 seeds"``.
    """
    parts = []
    for key, value in zip(grouped.group_keys, grouped.group_values):
        short = key.split(".")[-1]
        parts.append(f"{short}={value}")
    parts.append(f"n={grouped.n_seeds} seeds")
    return " | ".join(parts)


def filter_by_config(
    groups: List[GroupedRun], key: str, allowed: Optional[Sequence[Any]]
) -> List[GroupedRun]:
    """Keep only groups whose flattened config value for ``key`` is allowed.

    Parameters
    ----------
    groups
        The grouped runs to filter.
    key
        A flattened (dot-notation) config key, e.g. ``network.next_state_coef``.
    allowed
        The values to keep. If ``None`` the list is returned unchanged, so a
        plotting script can pass an optional CLI argument straight through.

    Useful, for instance, to drop the ``next_state_coef == 0.0`` baseline and
    compare only the runs that use the next-state-prediction loss.
    """
    if allowed is None:
        return list(groups)
    allowed_set = set(allowed)
    return [g for g in groups if g.config.get(key) in allowed_set]


def _seed_rows(arr: np.ndarray) -> np.ndarray:
    """Coerce one member's metric array into 2-D ``(n_seed_rows, n_timesteps)``.

    A single-seed folder stores a 1-D ``(T,)`` array -> one row. A folder that
    already holds multiple seeds stores a 2-D ``(S, T)`` array -> kept as-is.
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Expected a 1-D or 2-D metric array, got shape {arr.shape}")


def build_grouped_runs(
    root_dir: Path,
    group_keys: Sequence[str] = DEFAULT_GROUP_KEYS,
    metric_names: Sequence[str] = DEFAULT_METRICS,
    step_key: str = "env_step",
    seed_key: str = "seed",
    config_filters: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    optional_metric_names: Sequence[str] = (),
) -> List[GroupedRun]:
    """Scan ``root_dir`` and merge single-seed folders into :class:`GroupedRun`s.

    Parameters
    ----------
    root_dir
        Directory holding the run subfolders (each with ``config.yaml`` and
        ``metrics.npz``). Searched recursively.
    group_keys
        Flattened (dot-notation) config keys whose combination defines a group.
        Runs that agree on all of these are merged; ``seed`` is what varies
        within a group.
    metric_names
        Required metrics to load and stack. Each becomes a
        ``(n_seeds, n_timesteps)`` entry in :attr:`GroupedRun.metrics`. A seed
        whose ``metrics.npz`` is missing any required metric is skipped.
    step_key
        The npz key holding the x-axis (environment step) values.
    seed_key
        The flattened config key holding the seed (used only to order members).
    config_filters
        Optional ``{flat_key: value}`` map; folders whose config disagrees with
        any entry are skipped before grouping.
    verbose
        Print a short grouping summary.
    optional_metric_names
        Metrics to load if present, but not required. Seeds missing these
        metrics are kept (not skipped). At the group level the metric is only
        included in :attr:`GroupedRun.metrics` when every kept seed in the
        group has it; otherwise the key is simply absent from the group. Use
        this for metrics like ``intrinsic_return_ema`` that only exist on a
        subset of runs (e.g. exploration vs. baselines).

    Returns
    -------
    list[GroupedRun]
        One per unique ``group_keys`` combination, sorted by ``group_values``.
    """
    root_dir = Path(root_dir)
    config_paths = sorted(root_dir.rglob("config.yaml"))
    if not config_paths:
        if verbose:
            print(f"[grouping] no config.yaml files found under {root_dir}")
        return []

    group_keys = tuple(group_keys)
    metric_names = tuple(metric_names)
    optional_metric_names = tuple(optional_metric_names)

    # members[group_values] -> list of (seed, folder, flat_config, metrics_path)
    members: Dict[Tuple[Any, ...], List[Tuple[Any, Path, Dict[str, Any], Path]]] = {}

    n_skipped_no_metrics = 0
    n_skipped_filtered = 0
    for config_path in config_paths:
        folder = config_path.parent
        metrics_path = folder / "metrics.npz"
        if not metrics_path.exists():
            n_skipped_no_metrics += 1
            continue

        try:
            with open(config_path, "r") as f:
                raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"[grouping] could not parse {config_path}: {e}")
            continue

        flat = flatten_dict(raw_config or {})

        # Optional pre-grouping filters on non-varied config keys.
        if config_filters:
            mismatch = any(flat.get(k) != v for k, v in config_filters.items())
            if mismatch:
                n_skipped_filtered += 1
                continue

        group_values = tuple(
            folder.relative_to(root_dir).parts[0] if k == DIR_GROUP_KEY else flat.get(k)
            for k in group_keys
        )
        seed = flat.get(seed_key)
        members.setdefault(group_values, []).append((seed, folder, flat, metrics_path))

    grouped_runs: List[GroupedRun] = []
    for group_values in sorted(
        members.keys(), key=lambda gv: [_sort_key(v) for v in gv]
    ):
        entries = members[group_values]
        # Order members by seed so seed rows are deterministic.
        entries.sort(key=lambda e: _sort_key(e[0]))

        seeds = [e[0] for e in entries]
        folders = [e[1] for e in entries]
        rep_config = entries[0][2]

        # --- Collect per-seed rows for every requested metric ----------------
        # Required metrics — a seed missing any of these is dropped from the
        # group. Optional metrics are loaded if present, otherwise left absent
        # for that seed and resolved later at the group level.
        per_metric_rows: Dict[str, List[np.ndarray]] = {m: [] for m in metric_names}
        per_optional_rows: Dict[str, List[Optional[np.ndarray]]] = {
            m: [] for m in optional_metric_names
        }
        step_rows: List[np.ndarray] = []
        kept_seeds: List[Any] = []
        kept_folders: List[Path] = []

        for seed, folder, _flat, metrics_path in entries:
            try:
                npz = np.load(metrics_path)
            except Exception as e:  # noqa: BLE001 - report and skip the seed
                print(f"[grouping] could not load {metrics_path}: {e}")
                continue

            if step_key not in npz:
                print(f"[grouping] {metrics_path} has no '{step_key}' key, skipping")
                continue
            if any(m not in npz for m in metric_names):
                missing = [m for m in metric_names if m not in npz]
                print(
                    f"[grouping] {metrics_path} missing metrics {missing}, "
                    "skipping this seed"
                )
                continue

            step_arr = np.asarray(npz[step_key])
            step_arr = step_arr[0] if step_arr.ndim == 2 else step_arr

            metric_arrs = {m: _seed_rows(npz[m]) for m in metric_names}

            step_rows.append(step_arr)
            for m in metric_names:
                per_metric_rows[m].append(metric_arrs[m])
            for m in optional_metric_names:
                per_optional_rows[m].append(_seed_rows(npz[m]) if m in npz else None)
            kept_seeds.append(seed)
            kept_folders.append(folder)

        if not kept_seeds:
            print(f"[grouping] group {group_values} has no usable seeds, skipping")
            continue

        # --- Align lengths: truncate every row to the shortest one -----------
        lengths = [len(s) for s in step_rows]
        for rows in per_metric_rows.values():
            for r in rows:
                lengths.append(r.shape[1])
        for rows in per_optional_rows.values():
            for r in rows:
                if r is not None:
                    lengths.append(r.shape[1])
        min_t = min(lengths)

        steps = step_rows[0][:min_t]
        metrics: Dict[str, np.ndarray] = {}
        for m in metric_names:
            stacked = np.vstack([r[:, :min_t] for r in per_metric_rows[m]])
            metrics[m] = stacked
        # Optional metrics: only include if every kept seed has them. Mixing
        # would silently drop rows from the seed-aggregated curves and confuse
        # downstream plotting (CIs, n_seeds, etc.).
        for m in optional_metric_names:
            rows = per_optional_rows[m]
            if rows and all(r is not None for r in rows):
                metrics[m] = np.vstack([r[:, :min_t] for r in rows])
            elif any(r is not None for r in rows):
                present = sum(r is not None for r in rows)
                print(
                    f"[grouping] group {group_values}: optional metric '{m}' "
                    f"present in {present}/{len(rows)} seeds — omitting from group"
                )

        grouped_runs.append(
            GroupedRun(
                group_keys=group_keys,
                group_values=group_values,
                config=rep_config,
                folders=kept_folders,
                seeds=kept_seeds,
                steps=steps,
                metrics=metrics,
            )
        )

    if verbose:
        print(
            f"[grouping] {len(config_paths)} run folders under {root_dir} "
            f"-> {len(grouped_runs)} groups by {group_keys}"
        )
        if n_skipped_no_metrics:
            print(f"[grouping]   ({n_skipped_no_metrics} folders had no metrics.npz)")
        if n_skipped_filtered:
            print(
                f"[grouping]   ({n_skipped_filtered} folders dropped by config filters)"
            )
        for g in grouped_runs:
            print(f"[grouping]   {format_group_label(g)}")

    return grouped_runs


def _sort_key(value: Any) -> Tuple[int, Any]:
    """Sort helper that tolerates ``None`` and mixed numeric/string values."""
    if value is None:
        return (0, 0)
    if isinstance(value, (int, float)):
        return (1, float(value))
    return (2, str(value))
