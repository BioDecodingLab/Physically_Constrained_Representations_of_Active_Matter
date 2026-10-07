"""Physical-space decoding and explicit numerical/physical failure diagnostics.

No checkpoint parameters or latent transitions are changed by these routines.
Corrected projection uses float64 intermediates and avoids the final normalized
round trip. This is a versioned inference intervention, not a retrained model.
"""

import math
import torch
import torch.nn.functional as F
from active_matter_repr.physics import project_moment_fields, residuals, derivative

FLOOR = 1e-8


def project_realizable(x, smooth=True):
    """Differentiable projection; finite gradient at isotropy, physical output.

    Returns float64. The norm has the zero subgradient at the origin. Projection
    acts on the anisotropy at fixed predicted concentration; it is not claimed
    to be the Euclidean projection of the full joint (c,D) vector.
    """
    if x.shape[-3] != 11:
        raise ValueError("Expected eleven physical field channels")
    x = x.double()
    c = (
        FLOOR + F.softplus(x[..., 0, :, :] - FLOOR, beta=50.0)
        if smooth
        else x[..., 0, :, :].clamp_min(FLOOR)
    )
    a = x[..., 3, :, :] - x[..., 6, :, :]
    b = x[..., 4, :, :] + x[..., 5, :, :]
    norm = torch.linalg.vector_norm(torch.stack((a, b), dim=-3), dim=-3)
    # c>=FLOOR: for norm<FLOOR the exact factor is necessarily one.
    # A tiny floating-point denominator would overflow the backward quotient.
    factor = (c / norm.clamp_min(FLOOR)).clamp_max(1.0)
    a, b = a * factor, b * factor
    parts = [x[..., j, :, :] for j in range(11)]
    parts[0], parts[3], parts[6] = c, (c + a) / 2, (c - a) / 2
    parts[4] = parts[5] = b / 2
    return torch.stack(parts, dim=-3)


def physical_variants(rep, latent, transform, method):
    """Use identical raw network outputs to isolate decoder arithmetic changes."""
    if method == "pod":
        normalized = rep.decode(latent)
    else:
        normalized = rep.decoder_conv(rep.fc_dec(latent).reshape(-1, *rep.small))
    # Reproduce v0.2 float32 raw physical fields exactly before the intervention.
    raw = transform.inverse(normalized)
    if method in ("rae", "rpc"):
        historical_projected = project_moment_fields(raw, smooth_concentration=True)
        legacy = transform.inverse(transform.normalize(historical_projected))
        corrected = project_realizable(raw)
        # Higher precision before projection is a diagnostic, not another model.
        raw64 = (
            normalized.double() * transform.channel_scale.double()[:, None, None]
            + transform.mean.double()[:, None, None]
        )
        reference = project_realizable(raw64)
        diagnostic = {
            "raw_c_min": raw[:, 0].amin((-2, -1)),
            "before_roundtrip_c_min": historical_projected[:, 0].amin((-2, -1)),
            "after_roundtrip_c_min": legacy[:, 0].amin((-2, -1)),
            "roundtrip_c_max_abs_change": (legacy[:, 0] - historical_projected[:, 0])
            .abs()
            .amax((-2, -1)),
            "legacy_direct_max_abs_change": (legacy.double() - corrected)
            .abs()
            .amax((-3, -2, -1)),
            "direct_high_precision_reference_max_abs_change": (corrected - reference)
            .abs()
            .amax((-3, -2, -1)),
        }
        return {"legacy": legacy, "direct_corrected": corrected}, diagnostic
    return {"legacy": raw, "projected": project_realizable(raw, smooth=False)}, {}


def finite_scalar(x):
    v = float(x)
    return v if math.isfinite(v) else None


@torch.no_grad()
def measure(pred, truth, transform, spacing, obs_scale, derivative_method="spectral"):
    """One frame. Undefined metrics stay null; no survivor-only primary means.

    Legacy concentration threshold is retained for Q. Positive-domain order is
    also reported, but flooring concentration does not make Q trustworthy.
    """
    x, y = pred.double(), truth.double()
    result = {
        "finite_field": bool(torch.isfinite(x).all()),
        "R_rec": None,
        "F": None,
        "Q": None,
        "order_mae": None,
        "velocity_intensity_mae": None,
        "enstrophy_mae": None,
        "c_nonpositive_fraction": None,
        "c_near_floor_fraction": None,
        "c_min": None,
        "D_negative_fraction": None,
        "order_above_one_fraction": None,
        "D_trace_c_rmse": None,
        "D_asymmetry_rmse": None,
        "D_min_eigenvalue": None,
        "divergence_rms": None,
        "strain_mismatch_rms": None,
    }
    if not result["finite_field"]:
        result["status"] = "nonfinite_field"
        return result
    c, trace = x[0], x[3] + x[6]
    delta = torch.linalg.vector_norm(torch.stack((x[3] - x[6], x[4] + x[5])), dim=0)
    eig = (trace - delta) / 2
    result.update(
        c_min=finite_scalar(c.min()),
        c_nonpositive_fraction=float((c <= 0).double().mean()),
        c_near_floor_fraction=float((c <= FLOOR * (1 + 1e-6)).double().mean()),
        D_negative_fraction=float((eig < -1e-6).double().mean()),
        D_min_eigenvalue=finite_scalar(eig.min()),
        D_trace_c_rmse=finite_scalar((trace - c).square().mean().sqrt()),
        D_asymmetry_rmse=finite_scalar((x[4] - x[5]).square().mean().sqrt()),
    )
    if bool((c > 0).all()):
        result["order_above_one_fraction"] = float(
            (delta / c > 1 + 1e-6).double().mean()
        )
    err = (x - y) / transform.channel_scale.double()[:, None, None]
    blocks = [
        err[0:1].square().mean(),
        err[1:3].square().mean(),
        err[3:7].square().mean(),
        err[7:11].square().mean(),
    ]
    result["R_rec"] = finite_scalar(torch.stack(blocks).mean())
    result["F"] = finite_scalar((blocks[1].sqrt() + blocks[2].sqrt()) / 2)
    p = residuals(x[None], transform.scales.double(), spacing, derivative_method)[0]
    result["divergence_rms"], result["strain_mismatch_rms"] = (
        finite_scalar(p[2].sqrt()),
        finite_scalar(p[3].sqrt()),
    )

    def kv_enstrophy(f):
        omega = derivative(f[2], -2, spacing[0], derivative_method) - derivative(
            f[1], -1, spacing[1], derivative_method
        )
        return torch.stack(
            (0.5 * f[1:3].square().sum(0).mean(), 0.5 * omega.square().mean())
        )

    absolute = (kv_enstrophy(x) - kv_enstrophy(y)).abs()
    result["velocity_intensity_mae"], result["enstrophy_mae"] = [
        finite_scalar(v) for v in absolute
    ]
    if bool((c > FLOOR * (1 + 1e-6)).all()) and bool((y[0] > FLOOR).all()):
        yd = torch.linalg.vector_norm(torch.stack((y[3] - y[6], y[4] + y[5])), dim=0)
        order_error = ((delta / c).mean() - (yd / y[0]).mean()).abs()
        result["order_mae"] = finite_scalar(order_error)
        errors = torch.cat((absolute, order_error[None]))
        scale = torch.as_tensor(obs_scale, dtype=x.dtype, device=x.device)
        result["Q"] = finite_scalar(
            (errors[torch.isfinite(scale)] / scale[torch.isfinite(scale)]).mean()
        )
    result["status"] = (
        "ok"
        if result["Q"] is not None and result["F"] is not None
        else "invalid_observable_or_overflow"
    )
    return result
