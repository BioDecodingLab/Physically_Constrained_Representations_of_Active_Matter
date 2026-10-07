"""Explicit stages of the controlled representation experiment."""
from pathlib import Path
import platform
import sys
import numpy as np
import torch
from . import __version__
from .artifacts import (write_json, read_json, digest, file_hash, seed_all,
                        load_checkpoint, save_checkpoint)
from .data import build_manifest, Snapshots, loader
from .download import ensure_data
from .transforms import FieldTransform, fit_transform
from .physics import residuals, observables, admissibility
from .models import Autoencoder, LatentMLP
from .checks import verify_operators
from .training import train_ae
from .evaluation import evaluate_reconstruction
from .representations import (fit_pod, cache_latents, fit_scaler, scaled,
                              select_ridge, ridge_predict, latent_correlations)
from .statistics import grouped_mean
from .forecasting import (Windows, train_forecaster, fit_linear_forecaster,
                          LinearAR, evaluate_forecast, long_rollouts)


def source_id():
    return digest({p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def configure_runtime(cfg):
    torch.set_num_threads(cfg["threads"])
    if cfg["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def audit_data(cfg):
    """G1: full manifest/hash scan; training transforms; train/valid physical checks."""
    configure_runtime(cfg)
    root = Path(cfg["run_dir"])
    root.mkdir(parents=True, exist_ok=True)
    analytic_checks = verify_operators()
    ensure_data(cfg)
    manifest = build_manifest(cfg)
    identity = digest({"config": cfg["id"], "manifest": manifest["id"], "source": source_id()})
    if (root / "audit.json").exists():
        old = read_json(root / "audit.json")
        if old["identity"] != identity:
            raise ValueError("Run belongs to different data/config/code; choose a new run_dir")
        if old["passed"]:
            print("Compatible data audit already complete", flush=True)
            return
    write_json(root / "resolved_config.json", cfg)
    write_json(root / "manifest.json", manifest)
    train = Snapshots(cfg, manifest, "train")
    transform = fit_transform(loader(train, cfg))
    all_obs, report = [], {}
    passed = True
    for split in ("train", "valid"):
        ds = train if split == "train" else Snapshots(cfg, manifest, split)
        maxima, traces, negative = np.zeros(4), [], []
        for batch in loader(ds, cfg):
            x = batch["x"]
            rms = residuals(x, transform.scales, ds.rows[0]["spacing"], cfg["derivative"]).sqrt().numpy()
            maxima = np.maximum(maxima, rms.max(0))
            obs = observables(x, ds.rows[0]["spacing"], cfg["derivative"])
            if split == "train":
                all_obs.append(obs.numpy())
            ad = admissibility(x)
            traces.extend(ad["D_trace_c_rmse"].tolist())
            negative.extend(ad["D_negative_eigen_fraction"].tolist())
        good = bool(np.all(maxima <= cfg["residual_rms_max"]) and max(traces) <= cfg["trace_rmse_max"] and max(negative) == 0)
        passed &= good
        report[split] = {"max_frame_residual_rms": maxima.tolist(), "max_trace_rmse": max(traces),
                         "max_negative_D_fraction": max(negative), "passed": good}
        ds.close()
    values = np.concatenate(all_obs).astype(np.float64)
    std = values.std(0)
    active = std > np.maximum(1e-10, np.abs(values.mean(0)) * 1e-7)
    if not active.any():
        raise ValueError("No variable physical observables")
    stats = {"mean": values.mean(0).tolist(), "std": [float(s) if a else None for s, a in zip(std, active)],
             "active": active.tolist(), "names": ["velocity_intensity", "enstrophy", "mean_local_order"]}
    stamps = {name: {"size": (Path(cfg["data_root"]) / name).stat().st_size,
                     "mtime_ns": (Path(cfg["data_root"]) / name).stat().st_mtime_ns} for name in manifest["files"]}
    write_json(root / "transform.json", transform.as_dict())
    write_json(root / "observable_stats.json", stats)
    write_json(root / "environment.json", {"python": sys.version, "platform": platform.platform(),
               "torch": str(torch.__version__), "cuda": torch.version.cuda, "package": __version__,
               "source": source_id(), "device": cfg["device"], "precision": "float32; statistics float64"})
    write_json(root / "audit.json", {"identity": identity, "config_id": cfg["id"], "source": source_id(),
               "passed": passed, "analytic_checks": analytic_checks, "physical_checks": report, "file_stamps": stamps,
               "synthetic": cfg["synthetic"], "note": "Reference checks are train/validation only. Thresholds are configurable engineering gates, not physical laws."})
    if not passed:
        raise ValueError("Physical measurement gate failed; inspect audit.json before changing any tolerance or training")
    print("Data and physical measurement gates passed", flush=True)


def context(cfg):
    configure_runtime(cfg)
    root = Path(cfg["run_dir"])
    audit = read_json(root / "audit.json")
    if not audit["passed"] or audit["config_id"] != cfg["id"] or audit["source"] != source_id():
        raise ValueError("Missing/stale measurement gate; use audit-data and a compatible run")
    cfg["run_id"] = audit["identity"]
    for name, stamp in audit["file_stamps"].items():
        stat = (Path(cfg["data_root"]) / name).stat()
        if stat.st_size != stamp["size"] or stat.st_mtime_ns != stamp["mtime_ns"]:
            raise ValueError("Source data changed; rerun a full audit in a new run directory")
    manifest = read_json(root / "manifest.json")
    transform = FieldTransform(**read_json(root / "transform.json")).to(cfg["device"])
    stats = read_json(root / "observable_stats.json")
    obs_scale = np.array([float("inf") if x is None else x for x in stats["std"]])
    return root, manifest, transform, obs_scale


def neural_methods(cfg):
    return ("ae", "pc", "rae", "rpc") if cfg["include_realizable"] else ("ae", "pc")


def all_methods(cfg):
    return (*neural_methods(cfg), "pod")


def make_ae(cfg, transform=None, realizable=False):
    return Autoencoder(cfg["expected_shape"], cfg["latent_dim"], cfg["widths"],
                       realizable=realizable, transform=transform).to(cfg["device"])


def train_representations(cfg, pilot_only=False):
    root, manifest, transform, obs_scale = context(cfg)
    train, valid = Snapshots(cfg, manifest, "train"), Snapshots(cfg, manifest, "valid")
    evaluator = lambda model: evaluate_reconstruction(model, valid, transform, cfg, obs_scale)
    selection_path = root / "selection.json"
    selection = read_json(selection_path) if selection_path.exists() else {
        "seeds": {}, "selected_lambda": None, "selected_realizable_lambda": None}
    for j, seed in enumerate(cfg["seeds"][:1] if pilot_only else cfg["seeds"]):
        if str(seed) in selection["seeds"] and all(k in selection["seeds"][str(seed)] for k in neural_methods(cfg)):
            continue
        directory = root / "representations" / str(seed)
        seed_all(seed)
        model = make_ae(cfg)
        warm = train_ae(model, transform, train, valid, cfg, directory / "warmup", seed, 0., cfg["warmup_epochs"], evaluator, warmup=True)
        ae_path = train_ae(model, transform, train, valid, cfg, directory / "ae", seed, 0., cfg["branch_epochs"], evaluator, warm)
        ae_score = evaluator(model)
        write_json(directory / "ae" / "validation.json", ae_score)
        weights = cfg["lambdas"] if j == 0 else [selection["selected_lambda"]]
        if any(w is None for w in weights):
            raise ValueError("Pilot penalty not selected")
        candidates = []
        for weight in weights:
            path = train_ae(model, transform, train, valid, cfg, directory / f"pc_{weight:g}", seed, weight,
                            cfg["branch_epochs"], evaluator, warm)
            score = evaluator(model)
            write_json(path.parent / "validation.json", score)
            candidates.append({"weight": weight, "path": str(path.relative_to(root)), **score["summary"]})
        if j == 0:
            eligible = [r for r in candidates if r["Q_rec"] is not None and r["R_rec"] <= cfg["reconstruction_guard"] * ae_score["summary"]["R_rec"]]
            write_json(root / "penalty_candidates.json", candidates)
            if not eligible:
                raise RuntimeError("No PC candidate passes the prespecified fidelity/observable guard. Pilot recorded; stop for scientific review.")
            chosen = min(eligible, key=lambda r: (r["Q_rec"], r["R_rec"], r["weight"]))
            selection["selected_lambda"] = chosen["weight"]
        else:
            chosen = candidates[0]
        entry = {"ae": str(ae_path.relative_to(root)), "pc": chosen["path"],
                 "warmup_sha256": file_hash(warm)}
        if cfg["include_realizable"]:
            realizable = make_ae(cfg, transform, realizable=True)
            rae_path = train_ae(realizable, transform, train, valid, cfg, directory / "rae", seed, 0.,
                                cfg["branch_epochs"], evaluator, warm, physics_terms=(1, 2, 3))
            rae_score = evaluator(realizable)
            write_json(directory / "rae" / "validation.json", rae_score)
            weights = cfg["realizable_lambdas"] if j == 0 else [selection["selected_realizable_lambda"]]
            if any(w is None for w in weights):
                raise ValueError("Pilot realizable-physics penalty not selected")
            rcandidates = []
            for weight in weights:
                path = train_ae(realizable, transform, train, valid, cfg, directory / f"rpc_{weight:g}",
                                seed, weight, cfg["branch_epochs"], evaluator, warm,
                                physics_terms=(1, 2, 3))
                score = evaluator(realizable)
                write_json(path.parent / "validation.json", score)
                rcandidates.append({"weight": weight, "path": str(path.relative_to(root)), **score["summary"]})
            if j == 0:
                eligible = [r for r in rcandidates if r["Q_rec"] is not None and
                            r["R_rec"] <= cfg["reconstruction_guard"] * rae_score["summary"]["R_rec"]]
                write_json(root / "realizable_penalty_candidates.json", rcandidates)
                if not eligible:
                    raise RuntimeError("No R-PC candidate passes the prespecified fidelity/observable guard. Pilot recorded; stop for scientific review.")
                rchosen = min(eligible, key=lambda r: (r["Q_rec"], r["R_rec"], r["weight"]))
                selection["selected_realizable_lambda"] = rchosen["weight"]
            else:
                rchosen = rcandidates[0]
            entry.update({"rae": str(rae_path.relative_to(root)), "rpc": rchosen["path"]})
        selection["seeds"][str(seed)] = entry
        write_json(selection_path, selection)
    fit_pod(train, transform, cfg, root / "representations" / "pod.pt")
    train.close()
    valid.close()


def representations(cfg, root, manifest, transform):
    selection = read_json(root / "selection.json")
    for seed in cfg["seeds"]:
        if str(seed) not in selection["seeds"]:
            continue
        for method in all_methods(cfg):
            if method == "pod":
                path = root / "representations" / "pod.pt"
                model = fit_pod(Snapshots(cfg, manifest, "train"), transform, cfg, path)
            else:
                path = root / selection["seeds"][str(seed)][method]
                model = make_ae(cfg, transform, realizable=method in ("rae", "rpc"))
                model.load_state_dict(load_checkpoint(path)["model"])
            yield seed, method, model, file_hash(path)


def nuisance(cache, rows):
    return np.array([[*rows[int(i)]["parameters"], rows[int(i)]["times"][int(t)]] for i, t in zip(cache["trajectory"], cache["frame"])])


def analyze_representations(cfg):
    root, manifest, transform, obs_scale = context(cfg)
    train, valid = Snapshots(cfg, manifest, "train"), Snapshots(cfg, manifest, "valid")
    for seed, method, model, identity in representations(cfg, root, manifest, transform):
        directory = root / "analysis" / str(seed) / method
        tr = cache_latents(model, train, transform, cfg, directory / "train", identity)
        va = cache_latents(model, valid, transform, cfg, directory / "valid", identity)
        scaler = fit_scaler(tr["z"])
        probe = select_ridge(scaled(tr["z"], scaler), tr["observables"], scaled(va["z"], scaler), va["observables"],
                             cfg, va["trajectory"], valid.rows, obs_scale)
        ns = fit_scaler(nuisance(tr, train.rows))
        control = select_ridge(scaled(nuisance(tr, train.rows), ns), tr["observables"],
                               scaled(nuisance(va, valid.rows), ns), va["observables"], cfg, va["trajectory"], valid.rows, obs_scale)
        ps = fit_scaler(np.array([r["parameters"] for r in train.rows]))
        write_json(directory / "probes.json", {"latent_scaler": scaler, "parameter_scaler": ps,
                   "probe": probe, "nuisance_scaler": ns, "nuisance_probe": control, "checkpoint": identity})
        write_json(directory / "latent_correlations.json", {"train": latent_correlations(tr["z"], tr["observables"]),
                   "valid": latent_correlations(va["z"], va["observables"]), "observable_names": ["velocity_intensity", "enstrophy", "mean_local_order"]})
        result = evaluate_reconstruction(model, valid, transform, cfg, obs_scale)
        write_json(directory / "reconstruction_valid.json", result)
        if method == "ae":
            write_json(directory / "kinematic_reconstruction_valid.json",
                       evaluate_reconstruction(model, valid, transform, cfg, obs_scale, "kinematic"))
    train.close()
    valid.close()


def train_forecasters(cfg):
    root, manifest, transform, _ = context(cfg)
    train, valid = Snapshots(cfg, manifest, "train"), Snapshots(cfg, manifest, "valid")
    for seed, method, model, identity in representations(cfg, root, manifest, transform):
        directory = root / "analysis" / str(seed) / method
        metadata = read_json(directory / "probes.json")
        if metadata["checkpoint"] != identity:
            raise ValueError("Probe/encoder identity mismatch")
        tr = cache_latents(model, train, transform, cfg, directory / "train", identity)
        va = cache_latents(model, valid, transform, cfg, directory / "valid", identity)
        scaler, ps = metadata["latent_scaler"], metadata["parameter_scaler"]
        training = Windows(train, tr, scaler, ps, cfg["history"], 1)
        validation = Windows(valid, va, scaler, ps, cfg["history"], cfg["horizon"])
        seed_all(seed)
        target = root / "forecasters" / str(seed) / method
        train_forecaster(model, transform, training, validation, scaler, cfg, target, seed)
        if not (target / "linear.json").exists():
            linear = fit_linear_forecaster(model, transform, training, validation, scaler, cfg)
            write_json(target / "linear.json", linear)
    train.close()
    valid.close()


def _lock_files(root):
    files = [root / "selection.json", root / "transform.json", root / "observable_stats.json", root / "manifest.json"]
    files += list((root / "representations").rglob("best.pt"))
    files += [root / "representations" / "pod.pt"]
    files += list((root / "analysis").rglob("probes.json"))
    files += list((root / "forecasters").rglob("best.pt")) + list((root / "forecasters").rglob("linear.json"))
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(files)}


def lock_experiment(cfg):
    root, manifest, _, _ = context(cfg)
    if not (root / "evaluation_valid.json").exists():
        raise ValueError("Evaluate validation before locking")
    if set(read_json(root / "selection.json")["seeds"]) != {str(s) for s in cfg["seeds"]}:
        raise ValueError("Complete configured confirmation seeds before locking")
    validation = read_json(root / "evaluation_valid.json")
    required = {(s, m) for s in cfg["seeds"] for m in all_methods(cfg)}
    if {(r["seed"], r["method"]) for r in validation["results"]} != required:
        raise ValueError("Evaluate all configured seeds/representations before locking")
    lock = {"config_id": cfg["id"], "source": source_id(), "files": _lock_files(root),
            "endpoints": ["R_rec", "Q_rec", "Q_probe", f"F_{cfg['horizon']}",
                          "D_negative_eigen_fraction", "D_trace_c_rmse",
                          "local_order_above_one_fraction"],
            "history": cfg["history"], "horizon": cfg["horizon"], "all_eligible_origins": True,
            "historical_test_inspection": True, "synthetic": cfg["synthetic"]}
    if (root / "experiment_lock.json").exists() and read_json(root / "experiment_lock.json") != lock:
        raise ValueError("Existing experiment lock differs; do not overwrite a locked experiment")
    write_json(root / "experiment_lock.json", lock)


def evaluate(cfg, split="valid"):
    root, manifest, transform, obs_scale = context(cfg)
    if split == "test":
        lock = read_json(root / "experiment_lock.json")
        if lock["config_id"] != cfg["id"] or lock["source"] != source_id() or lock["files"] != _lock_files(root):
            raise ValueError("Test lock is stale")
    elif split != "valid":
        raise ValueError("Evaluation split must be valid or test")
    ds = Snapshots(cfg, manifest, split)
    results = []
    for seed, method, model, identity in representations(cfg, root, manifest, transform):
        analysis = root / "analysis" / str(seed) / method
        metadata = read_json(analysis / "probes.json")
        if metadata["checkpoint"] != identity:
            raise ValueError("Evaluation checkpoint mismatch")
        cache = cache_latents(model, ds, transform, cfg, analysis / split, identity)
        windows = Windows(ds, cache, metadata["latent_scaler"], metadata["parameter_scaler"], cfg["history"], cfg["horizon"])
        rec = evaluate_reconstruction(model, ds, transform, cfg, obs_scale)
        probes = {}
        for name, features, fit in (("latent", scaled(cache["z"], metadata["latent_scaler"]), metadata["probe"]),
                                     ("parameters_time", scaled(nuisance(cache, ds.rows), metadata["nuisance_scaler"]), metadata["nuisance_probe"])):
            pred = ridge_predict(fit, features)
            errors = np.abs(pred - cache["observables"]) / obs_scale
            active = np.isfinite(obs_scale)
            q = errors[:, active].mean(1)
            residual = ((pred - cache["observables"])**2).sum(0)
            total = ((cache["observables"] - cache["observables"].mean(0))**2).sum(0)
            probes[name] = {"Q_probe": float(grouped_mean(q, cache["trajectory"], ds.rows)),
                            "R2_pooled_frames": [float(1-r/t) if t > 1e-12 else None for r, t in zip(residual, total)],
                            "records": [{"trajectory": int(t), "frame": int(f), "Q_probe": float(v)} for t, f, v in zip(cache["trajectory"], cache["frame"], q)]}
        target = root / "forecasters" / str(seed) / method
        mlp = LatentMLP(cfg["latent_dim"], cfg["history"], cfg["forecast_width"]).to(cfg["device"])
        mlp.load_state_dict(load_checkpoint(target / "best.pt")["model"])
        linear = read_json(target / "linear.json")
        forecasts = {"mlp": evaluate_forecast(mlp, model, windows, metadata["latent_scaler"], transform, cfg),
                     "linear": evaluate_forecast(LinearAR(linear["coef"], linear["intercept"]).to(cfg["device"]), model, windows, metadata["latent_scaler"], transform, cfg)}
        if method == "ae":
            forecasts["persistence"] = evaluate_forecast(None, model, windows, metadata["latent_scaler"], transform, cfg)
        item = {"seed": seed, "method": method, "reconstruction": rec, "probes": probes, "forecasts": forecasts}
        if method in ("ae", "pc", "pod"):
            item["projected_reconstruction"] = evaluate_reconstruction(
                model, ds, transform, cfg, obs_scale, "realizability")
            item["projected_forecasts"] = {
                "mlp": evaluate_forecast(mlp, model, windows, metadata["latent_scaler"], transform, cfg, "realizability"),
                "linear": evaluate_forecast(LinearAR(linear["coef"], linear["intercept"]).to(cfg["device"]),
                                            model, windows, metadata["latent_scaler"], transform, cfg, "realizability")}
        if method == "ae":
            item["kinematic_reconstruction"] = evaluate_reconstruction(model, ds, transform, cfg, obs_scale, "kinematic")
            item["kinematic_forecast"] = evaluate_forecast(mlp, model, windows, metadata["latent_scaler"], transform, cfg, "kinematic")
        results.append(item)
    write_json(root / f"evaluation_{split}.json", {"split": split, "config_id": cfg["id"], "synthetic": cfg["synthetic"],
               "trajectories": ds.rows, "results": results})
    ds.close()


def diagnose_rollouts(cfg, split="valid"):
    root, manifest, transform, _ = context(cfg)
    if split == "test":
        lock = read_json(root / "experiment_lock.json")
        if lock["config_id"] != cfg["id"] or lock["source"] != source_id() or lock["files"] != _lock_files(root):
            raise ValueError("Test lock is stale")
    elif split != "valid":
        raise ValueError("Invalid rollout split")
    ds = Snapshots(cfg, manifest, split)
    result = []
    for seed, method, model, identity in representations(cfg, root, manifest, transform):
        analysis = root / "analysis" / str(seed) / method
        metadata = read_json(analysis / "probes.json")
        if metadata["checkpoint"] != identity:
            raise ValueError("Rollout representation mismatch")
        cache = cache_latents(model, ds, transform, cfg, analysis / split, identity)
        mlp = LatentMLP(cfg["latent_dim"], cfg["history"], cfg["forecast_width"]).to(cfg["device"])
        mlp.load_state_dict(load_checkpoint(root / "forecasters" / str(seed) / method / "best.pt")["model"])
        result.append({"seed": seed, "method": method, **long_rollouts(mlp, model, ds, cache,
                      metadata["latent_scaler"], metadata["parameter_scaler"], transform, cfg)})
    write_json(root / f"rollouts_{split}.json", {"split": split, "synthetic": cfg["synthetic"], "results": result})
    ds.close()
