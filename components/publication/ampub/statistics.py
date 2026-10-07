"""Publication summaries of frozen metrics; no model fitting or metric recomputation."""

from __future__ import annotations
import numpy as np
import pandas as pd

SEEDS = (1729, 2718, 31415)
KEY = ["dimension", "kind", "method", "predictor", "variant"]


def strict_mean(values):
    a = np.asarray(values, dtype=float)
    return float(a.mean()) if a.size and np.isfinite(a).all() else np.nan


def select(df, dimension, kind, method, variant, predictor="none"):
    x = df[
        (df.dimension == dimension)
        & (df.kind == kind)
        & (df.method == method)
        & (df.variant == variant)
    ]
    if "predictor" in x:
        x = x[x.predictor == predictor]
    return x.copy()


def seed_stats(rows, metric, expected=SEEDS):
    if rows.seed.duplicated().any():
        raise ValueError(
            "Duplicate seed rows: refine the dimension/kind/variant/predictor selection"
        )
    x = rows.set_index("seed").reindex(expected)
    a = pd.to_numeric(
        x.get(metric, pd.Series(index=x.index, dtype=float)), errors="coerce"
    ).to_numpy()
    badcol = metric + "_invalid_records"
    invalid = (
        int(pd.to_numeric(rows[badcol], errors="coerce").fillna(0).sum())
        if badcol in rows
        else 0
    )
    mean = strict_mean(a) if invalid == 0 else np.nan
    sd = float(np.std(a, ddof=1)) if np.isfinite(mean) and len(a) > 1 else np.nan
    return dict(
        mean=mean,
        sd=sd,
        n_finite=int(np.isfinite(a).sum()),
        n_expected=len(expected),
        invalid_records=invalid,
    )


def group_matrix(data, dimension, kind, method, variant, metric, predictor="none"):
    x = select(data.group_metrics, dimension, kind, method, variant, predictor)
    x = x[x.metric == metric]
    if x.empty:
        raise ValueError("No group-level frozen records for selected comparison")
    if x.duplicated(["seed", "group"]).any():
        raise ValueError("Duplicate group/seed records")
    p = x.pivot(index="seed", columns="group", values="value").reindex(SEEDS)
    # Any recorded failure invalidates the complete comparison; do not drop groups.
    if x.invalid_records.sum() > 0:
        p[:] = np.nan
    return p.sort_index(axis=1)


def publication_contrast(
    data, dimension, kind, metric, comparator, predictor="linear", percent=False
):
    """Matched group bootstrap, conditional on all fitted seeds. Publication-only derivation."""
    a = group_matrix(
        data, dimension, kind, "rpc", "direct_corrected", metric, predictor
    )
    method, variant, pred = comparator
    b = group_matrix(data, dimension, kind, method, variant, metric, pred)
    groups = sorted(set(a.columns) | set(b.columns))
    aa, bb = a.reindex(columns=groups).to_numpy(), b.reindex(columns=groups).to_numpy()
    result = dict(
        dimension=dimension,
        kind=kind,
        metric=metric,
        comparator=f"{method}/{variant}/{pred}",
        estimate=np.nan,
        ci95_low=np.nan,
        ci95_high=np.nan,
        groups=len(groups),
        seeds=len(SEEDS),
        derivation="publication-only matched group bootstrap; 2000 draws; random seed 1729",
        quantity=(
            "100*(comparator-R-PC-AE)/comparator"
            if percent
            else "comparator minus R-PC-AE"
        ),
    )
    if not np.isfinite(aa).all() or not np.isfinite(bb).all():
        return result
    av, bv = aa.mean(axis=0), bb.mean(axis=0)
    rng = np.random.default_rng(1729)
    idx = rng.integers(0, len(groups), size=(2000, len(groups)))
    if percent:
        if bv.mean() <= 0 or np.any(bv[idx].mean(axis=1) <= 0):
            return result
        estimate = 100 * (bv.mean() - av.mean()) / bv.mean()
        samples = (
            100 * (bv[idx].mean(axis=1) - av[idx].mean(axis=1)) / bv[idx].mean(axis=1)
        )
    else:
        estimate = (bv - av).mean()
        samples = (bv - av)[idx].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    result.update(estimate=float(estimate), ci95_low=float(low), ci95_high=float(high))
    return result


def uncertainty_tables(data):
    rows = []
    for key, x in data.summary.groupby(KEY, dropna=False, sort=True):
        for metric in [
            "R_rec",
            "F",
            "Q",
            "D_negative_fraction",
            "order_above_one_fraction",
            "divergence_rms",
            "strain_mismatch_rms",
        ]:
            if metric in x:
                rows.append(
                    dict(
                        zip(KEY, key),
                        metric=metric,
                        **seed_stats(x, metric),
                        uncertainty="sample SD across fitted neural seeds; deterministic labels are repeated evaluations",
                    )
                )
    seed = pd.DataFrame(rows)
    extra = []
    if not data.group_metrics.empty:
        for d in data.dimensions:
            extra.append(
                publication_contrast(
                    data,
                    d,
                    "reconstruction",
                    "Q",
                    ("ae", "legacy", "none"),
                    predictor="none",
                    percent=True,
                )
            )
            for kind in ["short_forecast", "rollout"]:
                for metric in ["F", "Q"]:
                    extra.append(
                        publication_contrast(
                            data, d, kind, metric, ("rpc", "legacy", "persistence")
                        )
                    )
    return seed, pd.DataFrame(extra)
