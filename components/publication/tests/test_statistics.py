import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
from ampub.statistics import seed_stats, strict_mean, publication_contrast
from ampub.export_visuals import pca_view, physical_views


def test_failure_never_becomes_survivor_mean():
    x = pd.DataFrame(
        {
            "seed": [1729, 2718, 31415],
            "Q": [1, np.nan, 2],
            "Q_invalid_records": [0, 1, 0],
        }
    )
    assert np.isnan(seed_stats(x, "Q")["mean"])
    assert np.isnan(strict_mean([1, np.nan]))


def test_missing_seed_stays_undefined():
    x = pd.DataFrame({"seed": [1729, 2718], "Q": [1, 2]})
    assert np.isnan(seed_stats(x, "Q")["mean"])


def test_duplicate_seed_selection_fails():
    with pytest.raises(ValueError):
        seed_stats(pd.DataFrame({"seed": [1729, 1729], "Q": [1, 2]}), "Q")


def group_fixture():
    rows = []
    for seed in [1729, 2718, 31415]:
        for g, base in [("a", 1), ("b", 2), ("c", 3)]:
            for m, var, val in [
                ("rpc", "direct_corrected", base * 0.5),
                ("ae", "legacy", base),
            ]:
                rows.append(
                    dict(
                        dimension=256,
                        kind="reconstruction",
                        seed=seed,
                        method=m,
                        predictor="none",
                        variant=var,
                        group=g,
                        metric="Q",
                        value=val,
                        invalid_records=0,
                    )
                )
    return SimpleNamespace(group_metrics=pd.DataFrame(rows))


def test_ratio_bootstrap_preserves_pairing():
    d = group_fixture()
    r = publication_contrast(
        d,
        256,
        "reconstruction",
        "Q",
        ("ae", "legacy", "none"),
        predictor="none",
        percent=True,
    )
    np.testing.assert_allclose(
        [r["estimate"], r["ci95_low"], r["ci95_high"]], 50, atol=1e-12
    )
    assert r == publication_contrast(
        d,
        256,
        "reconstruction",
        "Q",
        ("ae", "legacy", "none"),
        predictor="none",
        percent=True,
    )


def test_failed_group_invalidates_contrast():
    d = group_fixture()
    d.group_metrics.loc[0, "value"] = np.nan
    r = publication_contrast(
        d, 256, "reconstruction", "Q", ("ae", "legacy", "none"), predictor="none"
    )
    assert np.isnan(r["estimate"])


def test_unmatched_group_not_silently_dropped():
    d = group_fixture()
    d.group_metrics = d.group_metrics.drop(index=[1])
    r = publication_contrast(
        d, 256, "reconstruction", "Q", ("ae", "legacy", "none"), predictor="none"
    )
    assert np.isnan(r["estimate"])


def test_pca_fit_has_no_validation_leakage():
    rng = np.random.default_rng(4)
    train = rng.normal(size=(40, 6))
    valid = rng.normal(size=(12, 6))
    a = pca_view(train, valid)
    b = pca_view(train, valid * 100)
    for i in [1, 2, 3, 4]:
        np.testing.assert_allclose(a[i], b[i])
    assert a[0].shape == (12, 2)


def test_physical_vorticity_uses_correct_axes_and_spacing():
    n = 32
    L = 2 * np.pi
    x = np.zeros((11, n, n))
    grid = np.arange(n) * L / n
    x[0] = 1
    x[3] = x[6] = 0.5
    x[2] = np.sin(grid)[:, None]
    views = physical_views(x, [L / n, L / n])
    np.testing.assert_allclose(
        views[2], np.broadcast_to(np.cos(grid)[:, None], (n, n)), atol=1e-12
    )
    assert np.max(views[3]) == 0


def test_nonpositive_c_not_clamped_to_valid_order():
    x = np.zeros((11, 8, 8))
    x[0] = -1
    assert np.isnan(physical_views(x, [1, 1])[3]).all()
