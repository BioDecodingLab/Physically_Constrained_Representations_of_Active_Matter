"""Physical fields are [..., channel, x, y]; derivatives use periodic grids."""
import math
import torch
import torch.nn.functional as F

FIELDS = ("c", "vx", "vy", "Dxx", "Dxy", "Dyx", "Dyy", "Exx", "Exy", "Eyx", "Eyy")
C, VX, VY, DXX, DXY, DYX, DYY, EXX, EXY, EYX, EYY = range(11)
BLOCKS = (slice(0, 1), slice(1, 3), slice(3, 7), slice(7, 11))


def derivative(x, axis, spacing, method="spectral"):
    """Real first derivative; even-grid Nyquist multiplier is zero."""
    if axis not in (-2, -1) or spacing <= 0:
        raise ValueError("Derivative requires spatial axis -2/-1 and positive spacing")
    if method == "finite_difference":
        return (torch.roll(x, -1, axis) - torch.roll(x, 1, axis)) / (2 * spacing)
    if method != "spectral":
        raise ValueError(f"Unknown derivative {method}")
    # FFT outside reduced-precision autocast; gradients still propagate.
    x = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    # MKL in supported older PyTorch builds rejects expanded zero-stride inputs.
    x = x.contiguous()
    n = x.shape[axis]
    k = 2 * math.pi * torch.fft.fftfreq(n, d=spacing, device=x.device, dtype=x.dtype)
    if n % 2 == 0:
        k[n // 2] = 0
    shape = [1] * x.ndim
    shape[axis] = n
    return torch.fft.ifft(torch.fft.fft(x, dim=axis) * (1j * k.reshape(shape)), dim=axis).real


def strain(x, spacing, method="spectral"):
    u, v = x[..., VX, :, :], x[..., VY, :, :]
    ux = derivative(u, -2, spacing[0], method)
    uy = derivative(u, -1, spacing[1], method)
    vx = derivative(v, -2, spacing[0], method)
    vy = derivative(v, -1, spacing[1], method)
    shear = (uy + vx) / 2
    return torch.stack((ux, shear, shear, vy), dim=-3)


def residuals(x, scales, spacing, method="spectral"):
    """Four per-snapshot dimensionless squared residual families."""
    d = (x[..., DXY, :, :] - x[..., DYX, :, :]) / scales[2]
    e = (x[..., EXY, :, :] - x[..., EYX, :, :]) / scales[3]
    s = strain(x, spacing, method)
    div = (s[..., 0, :, :] + s[..., 3, :, :]) / scales[3]
    mismatch = (x[..., 7:11, :, :] - s) / scales[3]
    return torch.stack((d.square().mean((-2, -1)), e.square().mean((-2, -1)),
                        div.square().mean((-2, -1)), mismatch.square().mean((-3, -2, -1))), -1)


def project_moment_fields(x, concentration_floor=1e-8, smooth_concentration=False):
    """Project ``(c,D)`` onto the two-dimensional realizable moment cone.

    The returned fields satisfy c>0, D=D.T, D positive semidefinite and
    trace(D)=c pointwise.  Other channels are unchanged.  For a symmetric
    trace-c tensor this is the radial projection of its spin-2 anisotropy
    vector onto the disk of radius c.  It is differentiable almost everywhere
    and can therefore be used as the final layer of a trained decoder.

    ``smooth_concentration`` is intended for training.  It is a smooth maximum
    that is effectively the identity at the concentrations in this dataset.
    The diagnostic/post-hoc projection uses an exact clamp instead.
    """
    if x.shape[-3] != 11 or concentration_floor <= 0:
        raise ValueError("Expected eleven channels and a positive concentration floor")
    y = x.clone()
    c_raw = x[..., C, :, :]
    if smooth_concentration:
        # Stable smooth max(c, floor), effectively identity for c well above 0.
        c = concentration_floor + F.softplus(c_raw - concentration_floor, beta=50.)
    else:
        c = c_raw.clamp_min(concentration_floor)
    anisotropy_x = x[..., DXX, :, :] - x[..., DYY, :, :]
    anisotropy_y = x[..., DXY, :, :] + x[..., DYX, :, :]
    magnitude = torch.sqrt(anisotropy_x.square() + anisotropy_y.square())
    radial_scale = torch.clamp(c / magnitude.clamp_min(torch.finfo(x.dtype).eps), max=1.)
    anisotropy_x = anisotropy_x * radial_scale
    anisotropy_y = anisotropy_y * radial_scale
    y[..., C, :, :] = c
    y[..., DXX, :, :] = (c + anisotropy_x) / 2
    y[..., DYY, :, :] = (c - anisotropy_x) / 2
    y[..., DXY, :, :] = y[..., DYX, :, :] = anisotropy_y / 2
    return y


def correct_output(x, spacing, method="spectral"):
    y = x.clone()
    y[..., DXY, :, :] = y[..., DYX, :, :] = (x[..., DXY, :, :] + x[..., DYX, :, :]) / 2
    y[..., 7:11, :, :] = strain(x, spacing, method)
    return y


def observables(x, spacing, method="spectral", concentration_floor=1e-8):
    """K_v, enstrophy, mean local order. Invalid c is explicitly an error."""
    c = x[..., C, :, :]
    if not torch.isfinite(x).all() or (c <= concentration_floor).any():
        raise ValueError("Nonfinite field or nonpositive/near-zero concentration in order diagnostic")
    kv = 0.5 * x[..., 1:3, :, :].square().sum(-3).mean((-2, -1))
    omega = derivative(x[..., VY, :, :], -2, spacing[0], method) - derivative(x[..., VX, :, :], -1, spacing[1], method)
    enstrophy = 0.5 * omega.square().mean((-2, -1))
    off = (x[..., DXY, :, :] + x[..., DYX, :, :]) / 2
    order = torch.sqrt((x[..., DXX, :, :] - x[..., DYY, :, :]).square() + 4 * off.square()) / c
    return torch.stack((kv, enstrophy, order.mean((-2, -1))), -1)


def admissibility(x):
    c = x[..., C, :, :]
    off = (x[..., DXY, :, :] + x[..., DYX, :, :]) / 2
    trace = x[..., DXX, :, :] + x[..., DYY, :, :]
    delta = torch.sqrt((x[..., DXX, :, :] - x[..., DYY, :, :]).square() + 4 * off.square())
    order = delta / c.clamp_min(1e-8)
    return {"c_nonpositive_fraction": (c <= 1e-8).float().mean((-2, -1)),
            "D_negative_eigen_fraction": ((trace - delta) / 2 < -1e-6).float().mean((-2, -1)),
            "D_trace_c_rmse": (trace - c).square().mean((-2, -1)).sqrt(),
            "D_asymmetry_rmse": (x[..., DXY, :, :] - x[..., DYX, :, :]).square().mean((-2, -1)).sqrt(),
            "local_order_above_one_fraction": (order > 1. + 1e-6).float().mean((-2, -1)),
            "D_min_eigenvalue_over_c": (((trace - delta) / 2) / c.clamp_min(1e-8)).amin((-2, -1))}
