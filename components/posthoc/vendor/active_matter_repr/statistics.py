"""Hierarchical means and paired parameter-cluster bootstrap."""
from collections import defaultdict
import numpy as np


def grouped_mean(values, trajectory_ids, rows):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Cannot summarize empty/nonfinite or incomplete measurements")
    by_traj = defaultdict(list)
    for value, tid in zip(values, trajectory_ids):
        by_traj[int(tid)].append(value)
    if set(by_traj) != set(range(len(rows))):
        raise ValueError("Incomplete trajectory coverage")
    groups = defaultdict(list)
    for tid, observations in by_traj.items():
        groups[rows[tid]["group"]].append(np.mean(observations, axis=0))
    return np.mean([np.mean(v, axis=0) for v in groups.values()], axis=0)


def paired_cluster_bootstrap(a, b, samples=2000, seed=1729):
    """Inputs [seed, group], matched ordering; positive difference means b is worse."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 2 or not np.isfinite(a + b).all():
        raise ValueError("Require finite matched [seed,group] arrays")
    d = (b - a).mean(0)
    if len(d) < 2:
        return {"difference_b_minus_a": float(d.mean()), "ci95": None, "groups": len(d)}
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), (samples, len(d)))].mean(1)
    return {"difference_b_minus_a": float(d.mean()), "ci95": np.quantile(boot, [.025, .975]).tolist(),
            "groups": len(d), "interpretation": "Conditional on trained seeds; parameter-group cluster bootstrap"}
