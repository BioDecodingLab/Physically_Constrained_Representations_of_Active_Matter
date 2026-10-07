"""Four-frame latent histories; decoded supervision and pure latent recursion."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.linear_model import Ridge
from .data import loader
from .models import LatentMLP
from .representations import scaled
from .transforms import reconstruction, forecast_error
from .physics import correct_output, observables, residuals, admissibility, project_moment_fields
from .statistics import grouped_mean
from .training import optimize_epoch
from .artifacts import save_checkpoint, load_checkpoint, write_json, rng_state, restore_rng, digest


class Windows(Dataset):
    def __init__(self, snapshots, cache, scaler, parameter_scaler, history, horizon):
        self.snapshots, self.rows = snapshots, snapshots.rows
        self.history, self.horizon = history, horizon
        self.z = np.asarray(scaled(cache["z"], scaler), dtype=np.float32)
        self.params = np.asarray(scaled(np.array([r["parameters"] for r in self.rows]), parameter_scaler), dtype=np.float32)
        self.lookup = {(int(i), int(t)): j for j, (i, t) in enumerate(zip(cache["trajectory"], cache["frame"]))}
        if len(self.lookup) != len(snapshots) or set(self.lookup) != set(snapshots.index):
            raise ValueError("Missing/duplicate/cross-split cache frames")
        self.origins = [(i, t) for i, r in enumerate(self.rows) for t in range(history - 1, len(r["times"]) - horizon)]
        if not self.origins:
            raise ValueError("No eligible forecast windows")

    def __len__(self):
        return len(self.origins)

    def close(self):
        self.snapshots.close()

    def __getitem__(self, index):
        i, t = self.origins[index]
        row = self.rows[i]
        history = torch.from_numpy(self.z[[self.lookup[i, j] for j in range(t - self.history + 1, t + 1)]])
        targets = torch.stack([self.snapshots.frame(row, j) for j in range(t + 1, t + self.horizon + 1)])
        return {"history": history, "parameters": torch.from_numpy(self.params[i]), "truth": targets,
                "last_field": self.snapshots.frame(row, t), "trajectory": i, "origin": t,
                "next_z": torch.from_numpy(self.z[self.lookup[i, t + 1]])}


class LinearAR(torch.nn.Module):
    def __init__(self, coef, intercept):
        super().__init__()
        self.register_buffer("coef", torch.tensor(coef, dtype=torch.float32))
        self.register_buffer("intercept", torch.tensor(intercept, dtype=torch.float32))

    def forward(self, history, parameters):
        x = torch.cat((history.flatten(1), parameters), 1)
        return history[:, -1] + x @ self.coef.T + self.intercept


def decode_latent(rep, z, scaler, transform):
    mean = torch.as_tensor(scaler["mean"], device=z.device, dtype=z.dtype)
    scale = torch.as_tensor(scaler["scale"], device=z.device, dtype=z.dtype)
    return transform.inverse(rep.decode(z * scale + mean))


@torch.no_grad()
def evaluate_forecast(predictor, rep, windows, scaler, transform, cfg, correction=None):
    if correction not in (None, "kinematic", "realizability"):
        raise ValueError("Unknown forecast correction")
    if predictor is not None:
        predictor.eval()
    rep.eval()
    records = []
    for batch in loader(windows, cfg):
        history = batch["history"].to(cfg["device"])
        params = batch["parameters"].to(cfg["device"])
        truth = batch["truth"].to(cfg["device"])
        for lead in range(windows.horizon):
            if predictor is None:
                pred = batch["last_field"].to(cfg["device"])
            else:
                z = predictor(history, params)
                pred = decode_latent(rep, z, scaler, transform)
                history = torch.cat((history[:, 1:], z[:, None]), 1)
            if correction == "kinematic":
                pred = correct_output(pred, windows.rows[0]["spacing"], cfg["derivative"])
            elif correction == "realizability":
                pred = project_moment_fields(pred)
            errors = forecast_error(pred, truth[:, lead], transform)
            ad = admissibility(pred)
            for i, error in enumerate(errors):
                records.append({"trajectory": int(batch["trajectory"][i]), "origin": int(batch["origin"][i]),
                                "lead": lead + 1, "error": float(error) if torch.isfinite(error) else None,
                                **{k: float(v[i]) for k, v in ad.items()}})
    failures = sum(r["error"] is None for r in records)
    value = None if failures else float(grouped_mean([r["error"] for r in records], [r["trajectory"] for r in records], windows.rows))
    summary = {"F_10" if windows.horizon == 10 else f"F_{windows.horizon}": value,
               "failed_lead_records": failures, "origins": len(windows)}
    for key in admissibility(torch.zeros(1, 11, 1, 1)):
        summary[key] = float(grouped_mean([r[key] for r in records],
                                           [r["trajectory"] for r in records], windows.rows))
    return {"summary": summary, "records": records}


def train_forecaster(rep, transform, training, validation, scaler, cfg, directory, seed):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for p in rep.parameters():
        p.requires_grad_(False)
    rep.eval()
    model = LatentMLP(cfg["latent_dim"], cfg["history"], cfg["forecast_width"]).to(cfg["device"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    identity = digest({"config": cfg["id"], "run": cfg.get("run_id"), "seed": seed, "scaler": scaler})
    start, best, history = 0, float("inf"), []
    last = directory / "last.pt"
    if last.exists():
        state = load_checkpoint(last)
        if state["identity"] != identity:
            raise ValueError("Forecaster resume mismatch")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        start, best, history = state["epoch"], state["best"], state["history"]
        best_path = directory / "best.pt"
        if state["is_best"] and (not best_path.exists() or load_checkpoint(best_path)["epoch"] != state["epoch"]):
            save_checkpoint(directory / "best.pt", state)
    for epoch in range(start, cfg["forecast_epochs"]):
        def loss(batch):
            z = model(batch["history"].to(cfg["device"]), batch["parameters"].to(cfg["device"]))
            # Decoder weights frozen; graph MUST remain enabled here.
            pred = decode_latent(rep, z, scaler, transform)
            return reconstruction(pred, batch["truth"][:, 0].to(cfg["device"]), transform)
        train_loss = optimize_epoch(model, loader(training, cfg, True, seed + epoch), opt, loss, cfg)
        score = evaluate_forecast(model, rep, validation, scaler, transform, cfg)["summary"]
        value = score[f"F_{cfg['horizon']}"]
        if value is None:
            raise FloatingPointError("Nonfinite validation forecast")
        is_best = value < best
        if is_best:
            best = value
        history.append({"epoch": epoch + 1, "train_loss": train_loss, **score})
        state = {"identity": identity, "model": model.state_dict(), "optimizer": opt.state_dict(),
                 "epoch": epoch + 1, "best": best, "is_best": is_best, "history": history, "rng": rng_state()}
        save_checkpoint(last, state)
        if is_best:
            save_checkpoint(directory / "best.pt", state)
        write_json(directory / "history.json", history)
        print(f"{directory.name}: epoch {epoch+1}, F_{cfg['horizon']}={value:.6g}", flush=True)
    model.load_state_dict(load_checkpoint(directory / "best.pt")["model"])
    return model


def fit_linear_forecaster(rep, transform, training, validation, scaler, cfg):
    features, targets = [], []
    # Only latent arrays required for fitting; avoid reading truth fields here.
    for i, t in training.origins:
        h = training.z[[training.lookup[i, j] for j in range(t - training.history + 1, t + 1)]]
        features.append(np.concatenate((h.flatten(), training.params[i])))
        targets.append(training.z[training.lookup[i, t+1]] - h[-1])
    candidates = []
    for alpha in cfg["ridge_alphas"]:
        fitted = Ridge(alpha=alpha).fit(np.stack(features), np.stack(targets))
        model = LinearAR(fitted.coef_, fitted.intercept_).to(cfg["device"])
        value = evaluate_forecast(model, rep, validation, scaler, transform, cfg)["summary"][f"F_{cfg['horizon']}"]
        if value is not None:
            candidates.append((value, alpha, fitted))
    if not candidates:
        raise FloatingPointError("All linear AR validation forecasts failed")
    value, alpha, fit = min(candidates, key=lambda x: (x[0], x[1]))
    return {"alpha": alpha, "coef": fit.coef_.tolist(), "intercept": fit.intercept_.tolist(),
            "validation_score": value, "candidates": [{"alpha": a, "score": v} for v, a, _ in candidates]}


@torch.no_grad()
def long_rollouts(predictor, rep, ds, cache, scaler, parameter_scaler, transform, cfg):
    """First-history origin only; stream truth one lead at a time, bounded RAM."""
    rep.eval()
    predictor.eval()
    z = scaled(cache["z"], scaler).astype(np.float32)
    lookup = {(int(i), int(t)): j for j, (i,t) in enumerate(zip(cache["trajectory"], cache["frame"]))}
    records = []
    for i, row in enumerate(ds.rows):
        history = torch.as_tensor(z[[lookup[i,t] for t in range(cfg["history"])]], device=cfg["device"])[None]
        params = torch.as_tensor(scaled(np.array([row["parameters"]]), parameter_scaler), dtype=torch.float32, device=cfg["device"])
        for t in range(cfg["history"], len(row["times"])):
            future_z = predictor(history, params)
            pred = decode_latent(rep, future_z, scaler, transform)
            history = torch.cat((history[:,1:], future_z[:,None]), 1)
            truth = ds.frame(row, t)[None].to(cfg["device"])
            oracle = transform.inverse(rep(transform.normalize(truth)))
            error = forecast_error(pred, truth, transform)[0]
            floor = forecast_error(oracle, truth, transform)[0]
            true_obs = observables(truth, row["spacing"], cfg["derivative"])[0].cpu().tolist()
            try:
                po = observables(pred, row["spacing"], cfg["derivative"])[0].cpu().tolist()
                if not np.isfinite(po).all():
                    po = None
            except ValueError:
                po = None
            physical = residuals(pred, transform.scales, row["spacing"], cfg["derivative"])[0].cpu().tolist()
            ad = {k: float(v[0]) for k, v in admissibility(pred).items()}
            records.append({"trajectory": i, "origin": cfg["history"]-1, "lead": t-cfg["history"]+1,
                            "time": row["times"][t], "forecast_error": float(error) if torch.isfinite(error) else None,
                            "encoded_true_future_error": float(floor) if torch.isfinite(floor) else None,
                            "truth_observables": true_obs, "predicted_observables": po,
                            "physics_mse": [v if np.isfinite(v) else None for v in physical], **ad})
    return {"origin_rule": "First four stored frames; recursive MLP; no teacher forcing",
            "encoded_true_future_note": "Uses future truth: reconstruction reference, not a forecasting baseline or mathematical error lower bound",
            "records": records}
