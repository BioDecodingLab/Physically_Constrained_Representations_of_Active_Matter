"""P02-derived global AE and common latent temporal model."""
import torch
from torch import nn
from .physics import project_moment_fields


class Autoencoder(nn.Module):
    def __init__(self, shape=(256, 256), latent_dim=256, widths=(32, 64, 128, 256),
                 realizable=False, transform=None):
        super().__init__()
        if len(widths) != 4 or any(n % 16 for n in shape):
            raise ValueError("Four downsampling stages require spatial sizes divisible by 16")
        self.small = (widths[-1], shape[0] // 16, shape[1] // 16)
        n = self.small[0] * self.small[1] * self.small[2]
        layers, last = [], 11
        for width in widths:
            layers += [nn.Conv2d(last, width, 3, padding=1, padding_mode="circular"), nn.ReLU(), nn.MaxPool2d(2)]
            last = width
        self.encoder_conv = nn.Sequential(*layers)
        self.fc_enc, self.fc_dec = nn.Linear(n, latent_dim), nn.Linear(latent_dim, n)
        # Preserve the students' transposed-convolution decoder in both arms.
        layers = []
        for width in (*reversed(widths[:-1]), 11):
            layers.append(nn.ConvTranspose2d(last, width, 3, stride=2, padding=1, output_padding=1))
            if width != 11:
                layers.append(nn.ReLU())
            last = width
        self.decoder_conv = nn.Sequential(*layers)
        self.realizable = bool(realizable)
        if self.realizable:
            if transform is None:
                raise ValueError("A realizable decoder requires the fitted physical-field transform")
            # Nonpersistent: these values are provenance-controlled by transform.json,
            # and omitting them keeps warm-up checkpoints architecture-compatible.
            self.register_buffer("physical_mean", transform.mean.detach().clone(), persistent=False)
            self.register_buffer("physical_scale", transform.channel_scale.detach().clone(), persistent=False)

    def encode(self, x):
        return self.fc_enc(self.encoder_conv(x).flatten(1))

    def decode(self, z):
        normalized = self.decoder_conv(self.fc_dec(z).reshape(-1, *self.small))
        if not self.realizable:
            return normalized
        physical = normalized * self.physical_scale[:, None, None] + self.physical_mean[:, None, None]
        physical = project_moment_fields(physical, smooth_concentration=True)
        return (physical - self.physical_mean[:, None, None]) / self.physical_scale[:, None, None]

    def forward(self, x):
        return self.decode(self.encode(x))


class LatentMLP(nn.Module):
    def __init__(self, latent_dim, history=4, width=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(history * latent_dim + 2, width), nn.ReLU(),
                                 nn.Linear(width, width), nn.ReLU(), nn.Linear(width, latent_dim))

    def forward(self, history, parameters):
        return history[:, -1] + self.net(torch.cat((history.flatten(1), parameters), 1))


class POD(nn.Module):
    """Basis acts on normalized fields with additional block weights."""
    def __init__(self, components, center, shape):
        super().__init__()
        self.shape = tuple(shape)
        self.register_buffer("components", torch.as_tensor(components, dtype=torch.float32))
        self.register_buffer("center", torch.as_tensor(center, dtype=torch.float32))
        self.register_buffer("weight", torch.tensor([1., 2**-.5, 2**-.5] + [0.5] * 8)[:, None, None])

    def encode(self, x):
        return ((x * self.weight).flatten(1) - self.center) @ self.components.T

    def decode(self, z):
        return (z @ self.components + self.center).reshape(-1, 11, *self.shape) / self.weight

    def forward(self, x):
        return self.decode(self.encode(x))
