import numpy as np
import matplotlib.pyplot as plt
from ampub.schema import load_final
from ampub.figures import curves
from ampub.statistics import select


def test_curve_never_pools_dimension_variant_or_horizon(final_fixture):
    d = load_final(final_fixture)
    # Deliberately introduce very different distractor scores.
    distract = d.lead[
        (d.lead.dimension != 256)
        | (d.lead.kind != "rollout")
        | (d.lead.variant != "direct_corrected")
    ]
    d.lead.loc[distract.index, "F"] = 999
    expected = (
        select(d.lead, 256, "rollout", "rpc", "direct_corrected", "linear")
        .groupby("time_since_origin")
        .F.mean()
    )
    fig, ax = plt.subplots()
    curves(ax, d, 256, "rollout", "F")
    line = next(l for l in ax.lines if l.get_label() == "R-PC-AE + linear")
    np.testing.assert_allclose(line.get_ydata(), expected.values)
    plt.close(fig)


def test_undefined_seed_creates_curve_gap(final_fixture):
    d = load_final(final_fixture)
    z = select(d.lead, 256, "rollout", "rpc", "direct_corrected", "linear")
    ix = z[(z.seed == 1729) & (z.lead == 5)].index
    d.lead.loc[ix, "F"] = np.nan
    fig, ax = plt.subplots()
    curves(ax, d, 256, "rollout", "F")
    line = next(l for l in ax.lines if l.get_label() == "R-PC-AE + linear")
    assert np.isnan(line.get_ydata()[4])
    plt.close(fig)
