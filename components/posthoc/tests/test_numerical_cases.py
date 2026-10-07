import numpy as np
import pytest
import torch
from active_matter_repr.physics import project_moment_fields
from active_matter_repr.models import Autoencoder
from active_matter_repr.transforms import FieldTransform
from amreview.numerics import project_realizable, physical_variants, measure, FLOOR
from amreview.spatial import structure
from amreview.reporting import aggregate


@pytest.fixture(autouse=True)
def runtime():
    torch.set_num_threads(1)
    torch.manual_seed(17)


def isotropic(dtype=torch.float32):
    x = torch.zeros(1, 11, 4, 4, dtype=dtype)
    x[:, 0] = 1
    x[:, 3] = x[:, 6] = 0.5
    return x


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_isotropic_gradient_regression(dtype):
    old = isotropic(dtype).requires_grad_()
    project_moment_fields(old, smooth_concentration=True).square().sum().backward()
    assert not torch.isfinite(old.grad).all()  # Reproduce the original defect.
    fixed = isotropic(dtype).requires_grad_()
    project_realizable(fixed).square().sum().backward()
    assert torch.isfinite(fixed.grad).all()
    torch.testing.assert_close(
        project_realizable(fixed), fixed.double(), atol=1e-12, rtol=1e-12
    )


@pytest.mark.parametrize("magnitude", [0.0, 1e-12, 1.0, 1e10, 1e20])
def test_projection_extremes(magnitude):
    x = isotropic()
    x[:, 0] = -1.0
    x[:, 3] = magnitude
    x[:, 6] = -magnitude
    x[:, 4] = magnitude
    x.requires_grad_()
    y = project_realizable(x)
    assert torch.isfinite(y).all() and bool((y[:, 0] >= FLOOR).all())
    a, b = y[:, 3] - y[:, 6], y[:, 4] + y[:, 5]
    norm = torch.linalg.vector_norm(torch.stack((a, b), dim=1), dim=1)
    assert bool(((y[:, 3] + y[:, 6] - norm) / 2 >= -1e-20).all())
    torch.testing.assert_close(y[:, 3] + y[:, 6], y[:, 0], atol=1e-20, rtol=1e-12)
    y.square().sum().backward()
    assert torch.isfinite(x.grad).all()


def test_roundtrip_loses_concentration():
    transform = FieldTransform([1.0] + [0.0] * 10, [1.0] * 4)
    raw = isotropic()
    raw[:, 0] = -1.0
    projected = project_moment_fields(raw, smooth_concentration=True)
    roundtrip = transform.inverse(transform.normalize(projected))
    assert bool((projected[:, 0] > 0).all())
    assert bool((roundtrip[:, 0] == 0).all())
    corrected = project_realizable(raw)
    assert bool((corrected[:, 0] > 0).all())
    metrics = measure(corrected[0], isotropic()[0], transform, (1.0, 1.0), [1.0] * 3)
    assert metrics["c_nonpositive_fraction"] == 0
    assert metrics["c_near_floor_fraction"] == 1
    assert metrics["Q"] is None  # Repair does not conceal unreliable concentration.


def test_historical_decoder_exact_and_ordinary_correction_small():
    transform = FieldTransform([1.0, 0, 0, 0.5, 0, 0, 0.5, 0, 0, 0, 0], [1.0] * 4)
    model = Autoencoder((16, 16), 2, (2, 3, 4, 5), True, transform).eval()
    z = torch.randn(2, 2)
    variants, diagnostics = physical_variants(model, z, transform, "rpc")
    old = transform.inverse(model.decode(z))
    torch.testing.assert_close(old, variants["legacy"], atol=0, rtol=0)
    torch.testing.assert_close(
        old.double(), variants["direct_corrected"], atol=2e-7, rtol=2e-6
    )
    assert bool((diagnostics["legacy_direct_max_abs_change"] < 1e-6).all())


def test_nonfinite_fields_are_not_admissible_zeros():
    x = isotropic()[0]
    x[1, 0, 0] = float("nan")
    d = measure(
        x, isotropic()[0], FieldTransform([0.0] * 11, [1.0] * 4), (1.0, 1.0), [1.0] * 3
    )
    assert d["status"] == "nonfinite_field"
    assert d["D_negative_fraction"] is None and d["F"] is None and d["Q"] is None


def test_invalid_records_not_averaged_away():
    rows = [
        {"trajectory_id": "a", "group": "g", "F": 1.0},
        {"trajectory_id": "b", "group": "g", "F": None},
    ]
    mean, invalid = aggregate(rows, "F")
    assert mean is None and invalid == 1


def test_spatial_parseval_and_constant_orientation():
    x = torch.zeros(11, 16, 16, dtype=torch.float64)
    x[0] = 1
    x[3] = 0.7
    x[6] = 0.3
    xx = torch.arange(16, dtype=torch.float64) * 2 * torch.pi / 16
    x[1] = torch.sin(xx)[:, None]
    result = structure(x, (1.0, 1.0))
    assert sum(result["energy_shell"]) == pytest.approx(0.25, abs=1e-12)
    assert result["energy_shell"][1] == pytest.approx(0.25, abs=1e-12)
    np.testing.assert_allclose(result["correlation"], 1.0, atol=1e-12)


def test_spatial_zero_order_and_shift_invariance():
    x = isotropic(torch.float64)[0]
    assert structure(x, (1.0, 1.0))["correlation"] is None
    x[3] += 0.1 * torch.randn(4, 4, dtype=x.dtype)
    x[6] = 1 - x[3]
    a = structure(x, (1.0, 1.0))
    b = structure(torch.roll(x, (1, 2), (-2, -1)), (1.0, 1.0))
    np.testing.assert_allclose(a["correlation"], b["correlation"], atol=1e-12)
