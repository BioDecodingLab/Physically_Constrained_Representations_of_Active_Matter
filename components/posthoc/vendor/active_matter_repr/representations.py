"""Out-of-core POD, reusable latent caches, and validation-selected ridge probes."""
from pathlib import Path
import os
import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import Ridge
from scipy.stats import rankdata
from .artifacts import save_checkpoint, load_checkpoint, write_json, read_json, digest
from .data import loader
from .models import POD
from .physics import observables
from .statistics import grouped_mean


def fit_pod(dataset, transform, cfg, path):
    path = Path(path)
    if path.exists():
        state = load_checkpoint(path)
        if state["config_id"] != cfg["id"] or state.get("run_id") != cfg.get("run_id"):
            raise ValueError("POD configuration mismatch")
        return POD(state["components"], state["center"], cfg["expected_shape"]).to(cfg["device"])
    d, n = cfg["latent_dim"], len(dataset)
    if n <= d or d >= 11 * np.prod(cfg["expected_shape"]):
        raise ValueError("POD needs more training snapshots/features than latent dimensions")
    batch = max(cfg["pod_batch"], d)
    # Balanced chunks ensure the last partial_fit also contains at least d rows.
    chunks = np.array_split(np.arange(n), max(1, n // batch))
    estimator = IncrementalPCA(n_components=d, batch_size=batch)
    weight = torch.tensor([1., 2**-.5, 2**-.5] + [.5] * 8, device=cfg["device"])[:, None, None]
    for indices in chunks:
        # Memory is O(pod_batch * features), not O(features**2). See README RAM estimate.
        rows = []
        for i in indices:
            x = dataset[int(i)]["x"].to(cfg["device"])
            rows.append((transform.normalize(x) * weight).flatten().cpu().numpy())
        estimator.partial_fit(np.stack(rows))
    state = {"components": torch.from_numpy(estimator.components_.astype(np.float32)),
             "center": torch.from_numpy(estimator.mean_.astype(np.float32)), "config_id": cfg["id"], "run_id": cfg.get("run_id"),
             "explained_variance": torch.from_numpy(estimator.explained_variance_.astype(np.float32))}
    save_checkpoint(path, state)
    return POD(state["components"], state["center"], cfg["expected_shape"]).to(cfg["device"])


@torch.no_grad()
def cache_latents(model, dataset, transform, cfg, directory, identity):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    meta = {"identity": identity, "config_id": cfg["id"], "run_id": cfg.get("run_id"), "ids": [r["id"] for r in dataset.rows],
            "index": [list(i) for i in dataset.index]}
    key = digest(meta)
    path = directory / "latents.npz"
    if path.exists() and (directory / "metadata.json").exists():
        if read_json(directory / "metadata.json")["id"] != key:
            raise ValueError("Incompatible latent cache; use a new run directory")
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    model.eval()
    z, obs, tids, frames = [], [], [], []
    for batch in loader(dataset, cfg):
        x = batch["x"].to(cfg["device"])
        z.append(model.encode(transform.normalize(x)).cpu().numpy())
        obs.append(observables(x, dataset.rows[0]["spacing"], cfg["derivative"]).cpu().numpy())
        tids.extend(batch["trajectory"].tolist())
        frames.extend(batch["frame"].tolist())
    result = {"z": np.concatenate(z), "observables": np.concatenate(obs), "trajectory": np.asarray(tids), "frame": np.asarray(frames)}
    if not all(np.isfinite(v).all() for v in result.values()):
        raise ValueError("Nonfinite latent cache")
    tmp = directory / "latents.tmp.npz"
    np.savez(tmp, **result)
    os.replace(tmp, path)
    write_json(directory / "metadata.json", {"id": key, **meta})
    return result


def fit_scaler(x):
    mean, scale = np.mean(x, axis=0), np.std(x, axis=0)
    inactive = scale <= 1e-8
    return {"mean": mean.tolist(), "scale": np.where(inactive, 1., scale).tolist(), "inactive": inactive.tolist()}


def scaled(x, scaler):
    return (x - np.asarray(scaler["mean"])) / np.asarray(scaler["scale"])


def select_ridge(x, y, xv, yv, cfg, tids, rows, obs_scale=None):
    candidates = []
    for alpha in cfg["ridge_alphas"]:
        model = Ridge(alpha=alpha).fit(x, y)
        pred = model.predict(xv)
        error = (pred - yv) ** 2 if obs_scale is None else np.abs(pred - yv) / obs_scale
        if obs_scale is not None:
            error = error[:, np.isfinite(obs_scale)]
        score = float(np.mean(grouped_mean(error, tids, rows)))
        candidates.append((score, alpha, model))
    score, alpha, model = min(candidates, key=lambda a: (a[0], a[1]))
    return {"alpha": alpha, "validation_score": score, "coef": model.coef_.tolist(), "intercept": model.intercept_.tolist(),
            "candidates": [{"alpha": a, "score": s} for s, a, _ in candidates]}


def ridge_predict(model, x):
    return x @ np.asarray(model["coef"]).T + np.asarray(model["intercept"])


def latent_correlations(z, obs):
    """Descriptive associations only: no coordinate selection or p-value claims."""
    def correlate(a, b):
        a, b = a - a.mean(0), b - b.mean(0)
        denominator = np.sqrt((a*a).sum(0)[:, None] * (b*b).sum(0)[None])
        out = np.divide(a.T @ b, denominator, out=np.full((a.shape[1], b.shape[1]), np.nan), where=denominator > 1e-12)
        return [[None if not np.isfinite(v) else float(v) for v in row] for row in out]
    return {"pearson": correlate(z, obs), "spearman": correlate(rankdata(z, axis=0), rankdata(obs, axis=0)),
            "interpretation": "Descriptive frame-pooled correlations; dependent frames, non-identifiable latent axes; no causal or coordinate-interpretability claim"}
