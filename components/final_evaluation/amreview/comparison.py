"""Paired parameter-group uncertainty, conditional on the fitted seeds."""

import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from active_matter_repr.statistics import paired_cluster_bootstrap
from .reporting import write_csv

METRICS = (
    "R_rec",
    "F",
    "Q",
    "velocity_intensity_mae",
    "enstrophy_mae",
    "order_mae",
    "c_nonpositive_fraction",
    "D_negative_fraction",
    "order_above_one_fraction",
)


def compare(root, lock):
    root = Path(root)
    summaries = []
    group_rows = []
    lookup = {}
    for entry in lock["runs"]:
        d = entry["dimension"]
        out = root / f"d{d}"
        with open(out / "summary.csv") as f:
            summaries.extend(dict(dimension=d, **r) for r in csv.DictReader(f))
        for path in sorted(out.glob("*.jsonl")):
            buckets = defaultdict(list)
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    if r["kind"] == "spatial":
                        continue
                    k = (
                        r["kind"],
                        r["seed"],
                        r["method"],
                        r.get("predictor", "none"),
                        r["variant"],
                        r["group"],
                    )
                    buckets[k].append(r)
            for key, rr in buckets.items():
                kind, seed, method, pred, var, group = key
                for metric in METRICS:
                    traj = defaultdict(list)
                    for r in rr:
                        traj[r["trajectory_id"]].append(r.get(metric))
                    invalid = sum(
                        v is None or not np.isfinite(v)
                        for vs in traj.values()
                        for v in vs
                    )
                    value = (
                        None
                        if invalid
                        else float(np.mean([np.mean(v) for v in traj.values()]))
                    )
                    group_rows.append(
                        dict(
                            dimension=d,
                            kind=kind,
                            seed=seed,
                            method=method,
                            predictor=pred,
                            variant=var,
                            group=group,
                            metric=metric,
                            value=value,
                            invalid_records=invalid,
                        )
                    )
                    lookup[(d, kind, seed, method, pred, var, group, metric)] = value
    write_csv(root / "comparison_summary.csv", summaries)
    write_csv(root / "group_metrics.csv", group_rows)
    comparisons = []
    # Positive difference b-a means the second method has larger error.
    for entry in lock["runs"]:
        d = entry["dimension"]
        for kind, pred, metrics in [
            ("reconstruction", "none", METRICS),
            ("short_forecast", "linear", ("F", "Q")),
            ("rollout", "linear", ("F", "Q")),
        ]:
            for b, bv in [
                ("ae", "legacy"),
                ("ae", "projected"),
                ("pc", "legacy"),
                ("rae", "direct_corrected"),
                ("pod", "projected"),
            ]:
                for metric in metrics:
                    keys = sorted(
                        (k[2], k[6])
                        for k in lookup
                        if k[0] == d
                        and k[1] == kind
                        and k[3:6] == ("rpc", pred, "direct_corrected")
                        and k[7] == metric
                    )
                    if not keys:
                        continue
                    seeds = sorted({k[0] for k in keys})
                    groups = sorted({k[1] for k in keys})
                    a = np.array(
                        [
                            [
                                lookup.get(
                                    (
                                        d,
                                        kind,
                                        s,
                                        "rpc",
                                        pred,
                                        "direct_corrected",
                                        g,
                                        metric,
                                    ),
                                    np.nan,
                                )
                                for g in groups
                            ]
                            for s in seeds
                        ],
                        dtype=float,
                    )
                    bb = np.array(
                        [
                            [
                                lookup.get((d, kind, s, b, pred, bv, g, metric), np.nan)
                                for g in groups
                            ]
                            for s in seeds
                        ],
                        dtype=float,
                    )
                    if np.isfinite(a).all() and np.isfinite(bb).all():
                        res = paired_cluster_bootstrap(
                            a,
                            bb,
                            lock["protocol"]["bootstrap_samples"],
                            lock["protocol"]["bootstrap_seed"],
                        )
                        ci = res["ci95"]
                        delta = res["difference_b_minus_a"]
                    else:
                        delta = None
                        ci = None
                    comparisons.append(
                        dict(
                            dimension=d,
                            kind=kind,
                            predictor=pred,
                            metric=metric,
                            a="rpc/direct_corrected",
                            b=f"{b}/{bv}",
                            difference_b_minus_a=delta,
                            ci95_low=ci[0] if ci else None,
                            ci95_high=ci[1] if ci else None,
                            groups=len(groups),
                            seeds=len(seeds),
                        )
                    )
    write_csv(root / "paired_comparisons.csv", comparisons)
    # Keep seed scatter visible; deterministic POD repetitions are not independent.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    colors = dict(zip(("ae", "pc", "rae", "rpc", "pod"), ("C0", "C1", "C2", "C3", "C4")))
    if lock["synthetic"]:
        fig.suptitle("SYNTHETIC SOFTWARE TEST — not research results")
    for ax, (kind, metric, title) in zip(
        axs,
        [
            ("reconstruction", "Q", "Observable reconstruction"),
            ("short_forecast", "F", "All-origin 10-step field error"),
            ("rollout", "F", "First-origin long field error"),
        ],
    ):
        for method, var in [
            ("ae", "legacy"),
            ("pc", "legacy"),
            ("rae", "direct_corrected"),
            ("rpc", "direct_corrected"),
            ("pod", "projected"),
        ]:
            xs = []
            ys = []
            for d in sorted({e["dimension"] for e in lock["runs"]}):
                rows = [
                    r
                    for r in summaries
                    if r["dimension"] == d
                    and r["kind"] == kind
                    and r["method"] == method
                    and r["variant"] == var
                    and (kind == "reconstruction" or r["predictor"] == "linear")
                ]
                vv = [float(r[metric]) if r[metric] else np.nan for r in rows]
                xs.append(d)
                ys.append(np.mean(vv) if vv else np.nan)
                ax.scatter([d] * len(vv), vv, s=10, alpha=0.5, color=colors[method])
            ax.plot(xs, ys, marker="o", label=method.upper(), color=colors[method])
        ax.set(xlabel="Latent dimension", ylabel=metric, title=title)
        ax.set_xticks(xs)
    axs[0].legend(fontsize=8)
    for ext in ("svg", "png"):
        fig.savefig(root / f"compression_comparison.{ext}", dpi=200)
    plt.close(fig)
    (root / "INTERPRETATION.txt").write_text(
        "Official test split; no ranking-based selection is performed.\nshort_forecast F is all complete origins and leads 1..10; rollout F is first origin, all remaining leads.\nPaired intervals resample parameter groups after averaging matched seeds: conditional on these trained seeds, descriptive and not multiplicity-adjusted.\nFrames are not independent replicates. POD/persistence repetitions are not independent fits.\nNull means retain invalid records; see each dimension coverage.csv. Spatial diagnostics describe reconstructions only.\nHistorical test exposure in earlier project stages must be disclosed separately.\n"
    )
