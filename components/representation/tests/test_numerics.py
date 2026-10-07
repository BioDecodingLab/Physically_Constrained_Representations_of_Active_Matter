import math
import numpy as np
import pytest
import torch
from active_matter_repr.physics import (derivative, strain, residuals, correct_output,
                                        observables, admissibility, project_moment_fields)
from active_matter_repr.transforms import FieldTransform, reconstruction, fit_transform
from active_matter_repr.models import Autoencoder, LatentMLP, POD
from active_matter_repr.training import optimize_epoch
from active_matter_repr.statistics import grouped_mean, paired_cluster_bootstrap


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)
    torch.manual_seed(17)


@pytest.mark.parametrize("n", [15, 16, 32])
@pytest.mark.parametrize("axis", [-2, -1])
def test_spectral_derivative_axes(n, axis):
    a = torch.arange(n, dtype=torch.float64) * 2 * math.pi / n
    x, y = torch.meshgrid(a, a, indexing="ij")
    f = torch.sin(2*x) + .7*torch.cos(3*y)
    target = 2*torch.cos(2*x) if axis == -2 else -2.1*torch.sin(3*y)
    torch.testing.assert_close(derivative(f, axis, 2*math.pi/n), target, atol=1e-12, rtol=1e-12)


def test_nyquist_and_constant():
    x = ((-1.) ** torch.arange(16)).expand(16, -1)
    assert derivative(x, -1, 1).abs().max() == 0
    assert derivative(torch.ones(16, 16), -2, 1).abs().max() == 0


def test_finite_difference_converges():
    errors = []
    for n in (32, 64):
        a = torch.arange(n, dtype=torch.float64) * 2*math.pi/n
        x = torch.sin(a)[None].expand(n, -1)
        errors.append((derivative(x, -1, 2*math.pi/n, "finite_difference") - torch.cos(a)).square().mean().sqrt())
    assert 3.9 < errors[0] / errors[1] < 4.1


def test_physics_identity_and_wrong_channel():
    n = 32
    a = torch.arange(n) * 2*math.pi/n
    xx, yy = torch.meshgrid(a, a, indexing="ij")
    x = torch.zeros(2, 11, n, n)
    x[:, 0] = 1
    x[:, 1] = torch.sin(xx)*torch.cos(yy)
    x[:, 2] = -torch.cos(xx)*torch.sin(yy)
    x[:, 3] = .6
    x[:, 6] = .4
    x[:, 7:11] = strain(x, (2*math.pi/n,)*2)
    scales = torch.ones(4)
    assert residuals(x, scales, (2*math.pi/n,)*2).max() < 1e-11
    o = observables(x, (2*math.pi/n,)*2)
    torch.testing.assert_close(o[0], torch.tensor([.25, .5, .2]), atol=1e-6, rtol=1e-6)
    bad = x.clone()
    bad[:, [7, 8]] = bad[:, [8, 7]]
    assert residuals(bad, scales, (2*math.pi/n,)*2).max() > .01
    bad[:, 4] = .1
    corrected = correct_output(bad, (2*math.pi/n,)*2)
    torch.testing.assert_close(corrected[:, 1:3], bad[:, 1:3])
    assert residuals(corrected, scales, (2*math.pi/n,)*2).max() < 1e-11


def test_physics_gradient_and_invalid_order():
    x = torch.randn(2, 11, 16, 16, requires_grad=True)
    loss = residuals(x, torch.ones(4), (1., 1.)).mean()
    loss.backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="concentration"):
        observables(x.detach(), (1., 1.))


def test_moment_projection_is_exact_and_preserves_valid_fields():
    x = torch.randn(3, 11, 8, 8, dtype=torch.float64)
    x[:, 0] = .2 + torch.rand(3, 8, 8, dtype=torch.float64)
    before_other = x[:, [1, 2, 7, 8, 9, 10]].clone()
    projected = project_moment_fields(x)
    ad = admissibility(projected)
    assert ad["c_nonpositive_fraction"].max() == 0
    assert ad["D_negative_eigen_fraction"].max() == 0
    assert ad["local_order_above_one_fraction"].max() == 0
    assert ad["D_trace_c_rmse"].max() < 1e-12
    assert ad["D_asymmetry_rmse"].max() < 1e-12
    torch.testing.assert_close(projected[:, [1, 2, 7, 8, 9, 10]], before_other)

    valid = projected.clone()
    twice = project_moment_fields(valid)
    torch.testing.assert_close(twice, valid, atol=1e-12, rtol=1e-12)


def test_realizable_decoder_has_gradients_and_physical_output():
    transform = FieldTransform(torch.zeros(11), torch.ones(4))
    model = Autoencoder((16, 16), 2, (2, 3, 4, 5), realizable=True, transform=transform)
    x = torch.randn(2, 11, 16, 16)
    pred = transform.inverse(model(transform.normalize(x)))
    ad = admissibility(pred)
    assert ad["D_negative_eigen_fraction"].max() == 0
    assert ad["D_trace_c_rmse"].max() < 2e-6
    pred.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_transform_and_block_weighting():
    transform = FieldTransform(torch.arange(11), [1, 2, 3, 4])
    x = torch.randn(3, 11, 16, 16)
    torch.testing.assert_close(transform.inverse(transform.normalize(x)), x, atol=1e-6, rtol=1e-5)
    y = x + transform.channel_scale[:, None, None]
    torch.testing.assert_close(reconstruction(y, x, transform), torch.ones(3))
    fitted = fit_transform([{"x": torch.ones(2, 11, 4, 4)}, {"x": 2*torch.ones(1, 11, 4, 4)}])
    torch.testing.assert_close(fitted.mean, torch.full((11,), 4/3))
    torch.testing.assert_close(fitted.scales, torch.full((4,), math.sqrt(2)))


def test_decoder_identity_and_frozen_gradient():
    model = Autoencoder((16, 16), 2, (2, 3, 4, 5))
    x = torch.randn(2, 11, 16, 16)
    torch.testing.assert_close(model(x), model.decode(model.encode(x)), rtol=0, atol=0)
    for p in model.parameters():
        p.requires_grad_(False)
    temporal = LatentMLP(2, 4, 8)
    y = temporal(torch.randn(2, 4, 2), torch.randn(2, 2))
    model.decode(y).square().mean().backward()
    assert all(p.grad is None for p in model.parameters())
    assert sum(float(p.grad.abs().sum()) for p in temporal.parameters()) > 0


def test_full_resolution_architecture_forward_backward():
    model = Autoencoder()
    x = torch.randn(1, 11, 256, 256)
    pred = model(x)
    assert pred.shape == x.shape and model.encode(x).shape == (1, 256)
    loss = (pred-x).square().mean() + .01*residuals(pred, torch.ones(4), (10/256,)*2).mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_nonfinite_training_loss_fails():
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=.1)
    with pytest.raises(FloatingPointError, match="finite"):
        optimize_epoch(model, [None], opt, lambda _: torch.tensor([float("nan")]), {"effective_batch":1, "grad_clip":1})


def test_pod_weighted_objective():
    shape = (2, 2)
    components = torch.eye(44)[:5]
    model = POD(components, torch.zeros(44), shape)
    x = torch.randn(3, 11, *shape)
    pred = model(x)
    transform = FieldTransform(torch.zeros(11), torch.ones(4))
    weighted_sq = ((pred-x)*model.weight).flatten(1).square().sum(1)
    torch.testing.assert_close(reconstruction(pred, x, transform), weighted_sq / (4*4))


def test_accumulation_matches_large_batch_with_partial_tail():
    x, y = torch.randn(7, 2), torch.randn(7, 1)
    a, b = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    b.load_state_dict(a.state_dict())
    cfg = {"effective_batch": 4, "grad_clip": 100.}
    for model, size in ((a, 2), (b, 4)):
        batches = [(x[i:i+size], y[i:i+size]) for i in range(0, 7, size)]
        opt = torch.optim.SGD(model.parameters(), lr=.01)
        optimize_epoch(model, batches, opt, lambda batch: (model(batch[0])-batch[1]).square().mean(1), cfg)
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb)


def test_group_aggregation_not_frame_weighted():
    rows = [{"group": "a"}, {"group": "a"}, {"group": "b"}]
    assert grouped_mean([0, 0, 0, 2, 10], [0, 0, 0, 1, 2], rows) == 5.5
    with pytest.raises(ValueError, match="coverage"):
        grouped_mean([1], [0], rows)
    out = paired_cluster_bootstrap(np.zeros((3, 4)), np.ones((3, 4)))
    assert out["ci95"] == [1., 1.]
