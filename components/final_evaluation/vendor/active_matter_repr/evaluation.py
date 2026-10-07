"""Common reconstruction evaluator; retain raw rows and invalid-observable status."""
import numpy as np
import torch
from .data import loader
from .physics import residuals, observables, correct_output, admissibility, project_moment_fields, BLOCKS
from .transforms import reconstruction
from .statistics import grouped_mean


@torch.no_grad()
def evaluate_reconstruction(model, dataset, transform, cfg, obs_scale, correction=None):
    if correction not in (None, "kinematic", "realizability"):
        raise ValueError("Unknown reconstruction correction")
    model.eval()
    records = []
    spacing = dataset.rows[0]["spacing"]
    for batch in loader(dataset, cfg):
        x = batch["x"].to(cfg["device"])
        pred = transform.inverse(model(transform.normalize(x)))
        if correction == "kinematic":
            pred = correct_output(pred, spacing, cfg["derivative"])
        elif correction == "realizability":
            pred = project_moment_fields(pred)
        if not torch.isfinite(pred).all():
            raise FloatingPointError("Nonfinite reconstruction; evaluation incomplete")
        r = reconstruction(pred, x, transform)
        p = residuals(pred, transform.scales, spacing, cfg["derivative"])
        truth_o = observables(x, spacing, cfg["derivative"])
        ad = admissibility(pred)
        for i in range(len(x)):
            try:
                op = observables(pred[i:i+1], spacing, cfg["derivative"])[0]
                errors = ((op - truth_o[i]).abs().cpu().numpy() / obs_scale)
                active = np.isfinite(obs_scale)
                q = float(errors[active].mean())
                oe = [float(v) if a else None for v, a in zip(errors, active)]
            except ValueError:
                q, oe = None, [None] * 3
            records.append({"trajectory": int(batch["trajectory"][i]), "frame": int(batch["frame"][i]),
                            "R_rec": float(r[i]), "Q_rec": q, "observable_errors": oe,
                            "physics_mse": p[i].cpu().tolist(),
                            "block_rmse": [float((pred[i, b] - x[i, b]).square().mean().sqrt()) for b in BLOCKS],
                            **{k: float(v[i]) for k, v in ad.items()}})
    tids = [r["trajectory"] for r in records]
    score = {"R_rec": float(grouped_mean([r["R_rec"] for r in records], tids, dataset.rows)),
             "Q_rec": None, "invalid_observable_frames": sum(r["Q_rec"] is None for r in records)}
    if not score["invalid_observable_frames"]:
        score["Q_rec"] = float(grouped_mean([r["Q_rec"] for r in records], tids, dataset.rows))
    for key in admissibility(torch.zeros(1, 11, 1, 1)):
        score[key] = float(grouped_mean([r[key] for r in records], tids, dataset.rows))
    return {"summary": score, "records": records}
