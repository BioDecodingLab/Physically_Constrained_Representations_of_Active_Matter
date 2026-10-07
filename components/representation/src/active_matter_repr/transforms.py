"""Fixed training-only transforms and balanced losses."""
import torch
from .physics import BLOCKS


class FieldTransform(torch.nn.Module):
    def __init__(self, mean, scales):
        super().__init__()
        mean, scales = torch.as_tensor(mean, dtype=torch.float32), torch.as_tensor(scales, dtype=torch.float32)
        if mean.shape != (11,) or scales.shape != (4,) or not torch.isfinite(mean).all() or not torch.isfinite(scales).all() or (scales <= 1e-12).any():
            raise ValueError("Invalid training field transform")
        self.register_buffer("mean", mean)
        self.register_buffer("scales", scales)
        self.register_buffer("channel_scale", torch.repeat_interleave(scales, torch.tensor([1, 2, 4, 4])))

    def normalize(self, x):
        return (x - self.mean[:, None, None]) / self.channel_scale[:, None, None]

    def inverse(self, x):
        return x * self.channel_scale[:, None, None] + self.mean[:, None, None]

    def as_dict(self):
        return {"mean": self.mean.cpu().tolist(), "scales": self.scales.cpu().tolist()}


def fit_transform(loader):
    total = torch.zeros(11, dtype=torch.float64)
    sq = torch.zeros(11, dtype=torch.float64)
    count = 0
    for batch in loader:
        x = batch["x"].double()
        if not torch.isfinite(x).all():
            raise ValueError("Nonfinite training data")
        total += x.sum((0, 2, 3))
        sq += x.square().sum((0, 2, 3))
        count += x.shape[0] * x.shape[-2] * x.shape[-1]
    if count == 0:
        raise ValueError("Empty training data")
    return FieldTransform(total / count, torch.stack([(sq[b] / count).mean().sqrt() for b in BLOCKS]))


def reconstruction(pred, truth, transform):
    err = (pred - truth) / transform.channel_scale[:, None, None]
    return torch.stack([err[..., b, :, :].square().mean((-3, -2, -1)) for b in BLOCKS], -1).mean(-1)


def forecast_error(pred, truth, transform):
    err = (pred - truth) / transform.channel_scale[:, None, None]
    return (err[..., 1:3, :, :].square().mean((-3, -2, -1)).sqrt() +
            err[..., 3:7, :, :].square().mean((-3, -2, -1)).sqrt()) / 2
