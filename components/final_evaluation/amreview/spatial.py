"""Periodic shell spectra and amplitude-weighted nematic correlations."""

import torch
from .numerics import FLOOR


def structure(x, spacing):
    """Single physical frame; return arrays suitable for hierarchical averaging.

    Spectrum shell sums integrate to <|v|²>/2 (including the mean mode).
    C(r)=<q(x) dot q(x+r)>/<|q|²>, q=(Dxx-Dyy,Dxy+Dyx)/c.
    Correlation is unconnected and amplitude weighted, not a unit-director
    correlation; it is well defined at isotropic pixels without dividing by S.
    """
    if (
        x.ndim != 3
        or x.shape[0] != 11
        or x.shape[-1] != x.shape[-2]
        or abs(spacing[0] - spacing[1]) > 1e-10
    ):
        raise ValueError("Spatial diagnostic requires a square isotropic periodic grid")
    if not torch.isfinite(x).all():
        return {"status": "nonfinite_field"}
    x = x.double()
    n, dx = x.shape[-1], spacing[0]
    f = torch.fft.fftfreq(n, d=1 / n, device=x.device, dtype=x.dtype)
    radius = torch.sqrt(f[:, None].square() + f[None, :].square())
    shell = radius.round().long()
    count = int(shell.max()) + 1
    vf = torch.fft.fft2(x[1:3], norm="forward")
    energy = 0.5 * vf.abs().square().sum(0)
    # CPU bincount avoids nondeterministic CUDA scatter reductions.
    spectrum = torch.bincount(
        shell.flatten().cpu(), weights=energy.flatten().cpu(), minlength=count
    )
    result = {
        "status": "ok",
        "k": (torch.arange(count) * 2 * torch.pi / (n * dx)).tolist(),
        "energy_shell": spectrum.tolist(),
        "correlation": None,
        "r": None,
    }
    if (x[0] <= FLOOR * (1 + 1e-6)).any():
        result["status"] = "invalid_concentration_for_correlation"
        return result
    q = torch.stack((x[3] - x[6], x[4] + x[5])) / x[0]
    power = torch.fft.fft2(q, norm="forward").abs().square().sum(0)
    corr = torch.fft.ifft2(power, norm="forward").real
    if not torch.isfinite(corr).all() or corr[0, 0] <= 1e-20:
        result["status"] = "undefined_zero_order_correlation"
        return result
    corr = corr / corr[0, 0]
    sums = torch.bincount(
        shell.flatten().cpu(), weights=corr.flatten().cpu(), minlength=count
    )
    counts = torch.bincount(shell.flatten().cpu(), minlength=count)
    stop = n // 2 + 1
    result["correlation"] = (sums[:stop] / counts[:stop]).tolist()
    result["r"] = (torch.arange(stop) * dx).tolist()
    return result
