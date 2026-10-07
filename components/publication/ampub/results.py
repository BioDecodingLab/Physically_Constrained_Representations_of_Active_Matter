from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pandas as pd
from .statistics import strict_mean, seed_stats
from .schema import (
    FinalData,
    METHOD_LABELS,
    PRIMARY_VARIANTS,
    ORDERED_METHODS,
    primary_rows,
)


def mean_seed_table(
    data: FinalData, kind: str, metric: str, predictor: str | None = None
) -> pd.DataFrame:
    x = primary_rows(data.summary, kind=kind, predictor=predictor)
    if metric not in x.columns:
        return pd.DataFrame()
    grp = (
        x.groupby(["dimension", "method"], as_index=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grp["method_label"] = grp["method"].map(METHOD_LABELS)
    return grp


def exact_method_mean(
    data: FinalData,
    *,
    dimension: int,
    kind: str,
    method: str,
    metric: str,
    predictor: str = "none",
    variant: str | None = None,
) -> float:
    variant = variant or PRIMARY_VARIANTS[method]
    x = data.summary
    sel = x[
        (x["dimension"] == dimension)
        & (x["kind"] == kind)
        & (x["method"] == method)
        & (x["variant"] == variant)
    ]
    if "predictor" in sel.columns:
        sel = sel[sel["predictor"] == predictor]
    if sel.empty or metric not in sel:
        return float("nan")
    vals = pd.to_numeric(sel[metric], errors="coerce")
    return seed_stats(sel, metric)["mean"]


def key_results(data: FinalData) -> pd.DataFrame:
    rows = []
    for d in data.dimensions:
        ae_q = exact_method_mean(
            data, dimension=d, kind="reconstruction", method="ae", metric="Q"
        )
        rpc_q = exact_method_mean(
            data, dimension=d, kind="reconstruction", method="rpc", metric="Q"
        )
        ae_r = exact_method_mean(
            data, dimension=d, kind="reconstruction", method="ae", metric="R_rec"
        )
        rpc_r = exact_method_mean(
            data, dimension=d, kind="reconstruction", method="rpc", metric="R_rec"
        )
        reduction = (
            100 * (ae_q - rpc_q) / ae_q
            if np.isfinite(ae_q) and ae_q != 0 and np.isfinite(rpc_q)
            else np.nan
        )
        rchange = (
            100 * (rpc_r - ae_r) / ae_r
            if np.isfinite(ae_r) and ae_r != 0 and np.isfinite(rpc_r)
            else np.nan
        )
        vio_cols = [
            "c_nonpositive_fraction",
            "D_negative_fraction",
            "order_above_one_fraction",
        ]
        rpc_sel = primary_rows(data.summary, kind="reconstruction")
        rpc_sel = rpc_sel[(rpc_sel["dimension"] == d) & (rpc_sel["method"] == "rpc")]
        violation_max = np.nan
        vals = []
        for c in vio_cols:
            if c in rpc_sel:
                vals.extend(
                    pd.to_numeric(rpc_sel[c], errors="coerce").dropna().tolist()
                )
        if vals:
            violation_max = float(max(vals))
        rows.append(
            {
                "dimension": d,
                "AE_Q": ae_q,
                "R-PC-AE_Q": rpc_q,
                "Q_reduction_percent": reduction,
                "AE_R_rec": ae_r,
                "R-PC-AE_R_rec": rpc_r,
                "R_rec_relative_change_percent": rchange,
                "R-PC-AE_max_detected_admissibility_fraction": violation_max,
            }
        )
    return pd.DataFrame(rows)


def dynamics_key_table(data: FinalData, dimension: int = 256) -> pd.DataFrame:
    rows = []
    specs = [
        ("short_forecast", "linear", "rpc", "direct_corrected", "R-PC-AE + linear"),
        ("rollout", "linear", "rpc", "direct_corrected", "R-PC-AE + linear"),
        ("short_forecast", "linear", "pod", "projected", "POD + linear"),
        ("rollout", "linear", "pod", "projected", "POD + linear"),
        ("short_forecast", "persistence", "rpc", "legacy", "Physical persistence"),
        ("rollout", "persistence", "rpc", "legacy", "Physical persistence"),
    ]
    for kind, pred, method, variant, label in specs:
        x = data.summary
        sel = x[
            (x["dimension"] == dimension)
            & (x["kind"] == kind)
            & (x["method"] == method)
            & (x["variant"] == variant)
            & (x["predictor"] == pred)
        ]
        if sel.empty:
            continue
        rows.append(
            {
                "dimension": dimension,
                "kind": kind,
                "model": label,
                "F": seed_stats(sel, "F")["mean"],
                "Q": seed_stats(sel, "Q")["mean"],
                "n_seed_rows": int(len(sel)),
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["dimension", "kind", "model"])


