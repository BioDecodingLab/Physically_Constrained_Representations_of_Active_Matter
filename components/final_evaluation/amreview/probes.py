"""Incremental latent information; fit train only, select regularization on valid."""

import numpy as np
from active_matter_repr.representations import (
    fit_scaler,
    scaled,
    select_ridge,
    ridge_predict,
)
from active_matter_repr.pipeline import nuisance
from active_matter_repr.statistics import grouped_mean


def combined_probes(source, seed, method):
    tr, tr_rows = source.cache(seed, method, "train")
    va, va_rows = source.cache(seed, method, "valid")
    result = {
        "split": "valid",
        "seed": seed,
        "method": method,
        "models": {},
        "interpretation": "Validation-selected descriptive probes; not an unbiased final test or coordinate interpretation.",
    }
    ntr, nva = nuisance(tr, tr_rows), nuisance(va, va_rows)
    for name, x, xv in [
        ("parameters_time", ntr, nva),
        ("latent", tr["z"], va["z"]),
        ("combined", np.column_stack((ntr, tr["z"])), np.column_stack((nva, va["z"]))),
    ]:
        scaler = fit_scaler(x)
        fit = select_ridge(
            scaled(x, scaler),
            tr["observables"],
            scaled(xv, scaler),
            va["observables"],
            source.cfg,
            va["trajectory"],
            va_rows,
            source.obs_scale,
        )
        predicted = ridge_predict(fit, scaled(xv, scaler))
        active = np.isfinite(source.obs_scale)
        errors = (np.abs(predicted - va["observables"]) / source.obs_scale)[
            :, active
        ].mean(1)
        result["models"][name] = {
            "Q": float(grouped_mean(errors, va["trajectory"], va_rows)),
            "scaler": scaler,
            "fit": fit,
        }
    result["Q_gain_combined_vs_parameters_time"] = (
        result["models"]["parameters_time"]["Q"] - result["models"]["combined"]["Q"]
    )
    return result
