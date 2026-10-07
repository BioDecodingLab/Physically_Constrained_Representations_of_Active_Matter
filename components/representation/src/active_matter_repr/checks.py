"""Fast analytic operator gate executed before every new real-data audit."""
import math
import torch
from .physics import derivative, residuals, strain
from .transforms import FieldTransform
from .models import Autoencoder, LatentMLP


def verify_operators():
    a = torch.arange(16, dtype=torch.float64) * 2*math.pi/16
    x, y = torch.meshgrid(a, a, indexing="ij")
    spacing = 2*math.pi/16
    torch.testing.assert_close(derivative(torch.sin(x)+torch.cos(2*y), -2, spacing), torch.cos(x), atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(derivative(torch.sin(x)+torch.cos(2*y), -1, spacing), -2*torch.sin(2*y), atol=1e-11, rtol=1e-11)
    field = torch.zeros(1,11,16,16, dtype=torch.float64)
    field[:,0] = 1
    field[:,1], field[:,2] = torch.sin(x)*torch.cos(y), -torch.cos(x)*torch.sin(y)
    field[:,3], field[:,6] = .6, .4
    field[:,7:11] = strain(field, (spacing, spacing))
    if residuals(field, torch.ones(4), (spacing, spacing)).max() > 1e-20:
        raise AssertionError("Analytic strain/divergence operator gate failed")
    # Local RNG context avoids changing the caller's initialization stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        ae = Autoencoder((16,16), 2, (2,3,4,5))
        sample = torch.randn(2,11,16,16)
        torch.testing.assert_close(ae(sample), ae.decode(ae.encode(sample)), rtol=0, atol=0)
        tf = FieldTransform(torch.arange(11), [1,2,3,4])
        torch.testing.assert_close(tf.inverse(tf.normalize(sample)), sample, atol=1e-6, rtol=1e-5)
        for p in ae.parameters():
            p.requires_grad_(False)
        temporal = LatentMLP(2,4,8)
        ae.decode(temporal(torch.randn(2,4,2), torch.randn(2,2))).square().mean().backward()
        if not sum(float(p.grad.abs().sum()) for p in temporal.parameters()) > 0:
            raise AssertionError("Frozen decoder failed to transmit predictor gradient")
    return {"analytic_derivatives": True, "strain_and_divergence": True,
            "decoder_identity": True, "transform_round_trip": True, "frozen_decoder_gradient": True}
