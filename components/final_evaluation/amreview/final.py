"""Freeze validation-selected artifacts, then evaluate the official test split."""

import argparse
import fcntl
import json
import os
from pathlib import Path
from types import SimpleNamespace
from importlib.metadata import version
import time
import numpy as np
import torch
from active_matter_repr.artifacts import read_json, write_json, file_hash, digest
from active_matter_repr.data import Snapshots, loader
from active_matter_repr.representations import scaled, ridge_predict
from active_matter_repr.physics import observables
from .provenance import SourceRun, METHODS, inside
from .run import (
    reconstruction,
    rollouts,
    emit,
    task_file,
    task_complete,
    own_fingerprint,
)
from .numerics import physical_variants, measure, finite_scalar
from .reporting import make_reports, aggregate

STAGES = ("reconstruction", "short_forecast", "rollouts", "probes")
V030_SOURCE = "8fd9d61549bb41840f8bc1bd06520beec9124b4d575f0b0187c89af56b7957c0"


def software():
    return {n: version(n) for n in ("torch", "numpy", "scipy", "scikit-learn", "h5py")}


def separate(out, *roots):
    out = Path(out).resolve()
    for root in roots:
        root = Path(root).resolve()
        if out == root or out.is_relative_to(root) or root.is_relative_to(out):
            raise ValueError(
                "Final output must be separate from source runs, data and posthoc outputs"
            )


def validation_for(posthoc, source, synthetic=False):
    matches = []
    for p in Path(posthoc).rglob("analysis_manifest.json"):
        m = read_json(p)
        if (
            m.get("source_run_id") != source.cfg["run_id"]
            or not (p.parent / "COMPLETE.json").exists()
        ):
            continue
        q = m["protocol"]
        if (
            m["split"] != "valid"
            or q["quick"]
            or q["max_trajectories"]
            or q["max_leads"]
            or q["frame_stride"] != 1
        ):
            continue
        if set(q["seeds"]) != set(source.cfg["seeds"]) or set(q["methods"]) != set(
            METHODS
        ):
            continue
        if set(q["stages"]) != {"reconstruction", "rollouts", "probes"}:
            continue
        if set(q["predictors"]) != {"linear", "mlp", "persistence"}:
            continue
        if not synthetic and m["analysis_source"] != V030_SOURCE:
            raise ValueError("Expected the reviewed v0.3 validation analysis")
        matches.append((p.parent, m))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one completed full validation posthoc for {source.root}; found {len(matches)}"
        )
    folder, m = matches[0]
    body = dict(m)
    mid = body.pop("id")
    if digest(body) != mid:
        raise ValueError("Validation manifest digest mismatch")
    if read_json(folder / "COMPLETE.json")["id"] != m["id"]:
        raise ValueError("Validation completion identity mismatch")
    for seed in source.cfg["seeds"]:
        for method in METHODS:
            for stage in ("reconstruction", "rollouts", "probes"):
                if not task_complete(folder, f"{stage}_{seed}_{method}", m["id"]):
                    raise ValueError("Validation task missing")
    for name, expected in source.files.items():
        if m["source_files"].get(name) != expected:
            raise ValueError(f"Validation/source mismatch: {name}")
    return folder, m


def freeze(project, posthoc, out, run256=None, synthetic_runs=None):
    """No test field access. Synthetic override is internal to software tests."""
    project, posthoc, out = map(lambda p: Path(p).resolve(), (project, posthoc, out))
    roots = synthetic_runs or [
        project / "runs/compression_d128",
        Path(run256).resolve() if run256 else project / "runs/realizable_d256",
        project / "runs/compression_d512",
    ]
    data = project / "data/active_matter/data"
    separate(out, data, posthoc, *roots)
    if out.exists():
        raise ValueError(
            "Freeze destination already exists; reuse its lock with evaluate, or choose a new destination"
        )
    entries = []
    for root in roots:
        source = SourceRun(root, data, "cpu", True)
        if source.cfg["synthetic"] != bool(synthetic_runs):
            raise ValueError("Real and synthetic experiments cannot be mixed")
        if not synthetic_runs and (
            source.cfg["seeds"] != [1729, 2718, 31415]
            or source.cfg["history"] != 4
            or source.cfg["horizon"] != 10
        ):
            raise ValueError("Unexpected seeds/history/horizon")
        source.prepare(source.cfg["seeds"], METHODS, True)
        folder, m = validation_for(posthoc, source, bool(synthetic_runs))
        probes = {}
        for seed in source.cfg["seeds"]:
            for method in METHODS:
                # Also validates checkpoint and forecaster identities before test access.
                rep, meta, predictors = source.load(seed, method)
                del rep, meta, predictors
                probes[f"{seed}/{method}"] = read_json(
                    folder / f"probes_{seed}_{method}.json"
                )
        source.verify_unchanged()
        entries.append(
            dict(
                dimension=source.cfg["latent_dim"],
                run_dir=str(source.root),
                run_id=source.cfg["run_id"],
                source_files=source.files,
                validation_id=m["id"],
                validation_dir=str(folder),
                probes=probes,
            )
        )
    if not synthetic_runs and sorted(e["dimension"] for e in entries) != [
        128,
        256,
        512,
    ]:
        raise ValueError("Expected dimensions 128,256,512")
    manifests = [read_json(Path(e["run_dir"]) / "manifest.json")["id"] for e in entries]
    if len(set(manifests)) != 1:
        raise ValueError("Different data manifests across dimensions")
    body = dict(
        version="0.4.0",
        split="test",
        analysis_source=own_fingerprint(),
        software=software(),
        data_root=str(data),
        runs=entries,
        synthetic=bool(synthetic_runs),
        protocol=dict(
            stages=list(STAGES),
            methods=list(METHODS),
            predictors=["linear", "mlp", "persistence"],
            primary_predictor="linear",
            batch_size=16,
            spatial_stride=10,
            frame_stride=1,
            max_leads=0,
            max_trajectories=0,
            bootstrap_samples=2000,
            bootstrap_seed=1729,
        ),
        interpretation="Frozen after validation. Historical test exposure in earlier project stages is not certified absent. No tuning on these test results. Bootstrap conditional on the trained seeds.",
    )
    out.mkdir(parents=True)
    write_json(out / "protocol_lock.json", {"id": digest(body), **body})
    print(f'Frozen: {out / "protocol_lock.json"}', flush=True)
    return out / "protocol_lock.json"


@torch.no_grad()
def short_forecasts(source, rep, metadata, predictors, seed, method, cfg, args, stream):
    """All complete H-step origins; batch only observed histories, never future truth."""
    ds = Snapshots(cfg, source.manifest, "test")
    h, H = cfg["history"], cfg["horizon"]
    scaler = metadata["latent_scaler"]
    mean = torch.tensor(scaler["mean"], device=cfg["device"], dtype=torch.float32)
    scale = torch.tensor(scaler["scale"], device=cfg["device"], dtype=torch.float32)
    try:
        for row in ds.rows:
            origins = list(range(h - 1, len(row["times"]) - H))
            if not origins:
                raise ValueError("Trajectory has no complete forecast origins")
            for start in range(0, len(origins), args.batch_size):
                oo = origins[start : start + args.batch_size]
                observed = torch.stack(
                    [
                        torch.stack([ds.frame(row, t) for t in range(o - h + 1, o + 1)])
                        for o in oo
                    ]
                ).to(cfg["device"])
                enc = (
                    rep.encode(source.transform.normalize(observed.flatten(0, 1)))
                    .cpu()
                    .numpy()
                )
                initial = (
                    torch.from_numpy(scaled(enc, scaler).astype(np.float32))
                    .to(cfg["device"])
                    .reshape(len(oo), h, -1)
                )
                params = torch.tensor(
                    scaled(
                        np.array([row["parameters"]] * len(oo)),
                        metadata["parameter_scaler"],
                    ),
                    device=cfg["device"],
                    dtype=torch.float32,
                )
                for pred in args.predictors:
                    history = initial.clone()
                    for lead in range(1, H + 1):
                        if pred == "persistence":
                            variants = {"legacy": observed[:, -1]}
                        else:
                            z = predictors[pred](history, params)
                            history = torch.cat((history[:, 1:], z[:, None]), 1)
                            variants, _ = physical_variants(
                                rep, z * scale + mean, source.transform, method
                            )
                        for j, o in enumerate(oo):
                            truth = ds.frame(row, o + lead).to(cfg["device"])
                            for var, x in variants.items():
                                emit(
                                    stream,
                                    dict(
                                        kind="short_forecast",
                                        seed=seed,
                                        method=method,
                                        predictor=pred,
                                        variant=var,
                                        trajectory_id=row["id"],
                                        group=row["group"],
                                        origin=o,
                                        lead=lead,
                                        time_since_origin=row["times"][o + lead]
                                        - row["times"][o],
                                        **measure(
                                            x[j],
                                            truth,
                                            source.transform,
                                            row["spacing"],
                                            source.obs_scale,
                                            cfg["derivative"],
                                        ),
                                    ),
                                )
    finally:
        ds.close()


@torch.no_grad()
def frozen_probes(source, rep, frozen, seed, method, cfg):
    ds = Snapshots(cfg, source.manifest, "test")
    result = dict(split="test", seed=seed, method=method, models={})
    records = {k: [] for k in frozen["models"]}
    try:
        for batch in loader(ds, cfg):
            truth = batch["x"].to(cfg["device"])
            z = rep.encode(source.transform.normalize(truth)).cpu().numpy()
            nuisance = np.array(
                [
                    [*ds.rows[int(i)]["parameters"], ds.rows[int(i)]["times"][int(t)]]
                    for i, t in zip(batch["trajectory"], batch["frame"])
                ]
            )
            features = {
                "parameters_time": nuisance,
                "latent": z,
                "combined": np.column_stack((nuisance, z)),
            }
            for name, model in frozen["models"].items():
                pred = ridge_predict(
                    model["fit"], scaled(features[name], model["scaler"])
                )
                for j, (i, t) in enumerate(zip(batch["trajectory"], batch["frame"])):
                    row = ds.rows[int(i)]
                    obs = (
                        observables(
                            truth[j].double(), row["spacing"], cfg["derivative"]
                        )
                        .cpu()
                        .numpy()
                    )
                    active = np.isfinite(source.obs_scale)
                    q = finite_scalar(
                        np.mean((np.abs(pred[j] - obs) / source.obs_scale)[active])
                    )
                    records[name].append(
                        dict(
                            trajectory_id=row["id"],
                            group=row["group"],
                            frame=int(t),
                            Q=q,
                        )
                    )
        for name, rr in records.items():
            q, invalid = aggregate(rr, "Q")
            result["models"][name] = dict(Q=q, invalid_records=invalid, records=rr)
        a = result["models"]["parameters_time"]["Q"]
        b = result["models"]["combined"]["Q"]
        result["Q_gain_combined_vs_parameters_time"] = (
            a - b if a is not None and b is not None else None
        )
        return result
    finally:
        ds.close()


def evaluate(lock_path, device="cuda:0", workers=8, threads=8):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    lock_path = Path(lock_path).resolve()
    lock = read_json(lock_path)
    body = dict(lock)
    ident = body.pop("id")
    if (
        digest(body) != ident
        or lock["analysis_source"] != own_fingerprint()
        or lock["software"] != software()
    ):
        raise ValueError("Frozen protocol/code/environment changed")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if workers < 0 or threads < 1:
        raise ValueError("Invalid workers/threads")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    root = lock_path.parent
    with open(root / ".final.lock", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for entry in lock["runs"]:
            print(f"Verifying dimension {entry['dimension']}...", flush=True)
            source = SourceRun(entry["run_dir"], lock["data_root"], device, True)
            separate(root, source.root, source.data_root, entry["validation_dir"])
            source.prepare(source.cfg["seeds"], METHODS, True)
            if (
                source.cfg["run_id"] != entry["run_id"]
                or source.files != entry["source_files"]
            ):
                raise ValueError("Source artifacts changed after freeze")
            # Verify test bytes only AFTER protocol and source checks pass.
            for name, expected in source.manifest["files"].items():
                if (
                    Path(name).parts[0] == "test"
                    and file_hash(inside(source.data_root, name)) != expected
                ):
                    raise ValueError(f"Test data hash mismatch: {name}")
            out = root / f"d{entry['dimension']}"
            out.mkdir(exist_ok=True)
            info = dict(
                id=ident,
                split="test",
                scope="full test",
                synthetic=lock["synthetic"],
                source_run_id=entry["run_id"],
                dimension=entry["dimension"],
            )
            if (out / "analysis_manifest.json").exists() and read_json(
                out / "analysis_manifest.json"
            ) != info:
                raise ValueError("Output identity mismatch")
            write_json(out / "analysis_manifest.json", info)
            cfg = {
                **source.cfg,
                "batch_size": 16,
                "workers": workers,
                "threads": threads,
            }
            args = SimpleNamespace(
                **{
                    k: v
                    for k, v in lock["protocol"].items()
                    if k
                    not in (
                        "stages",
                        "methods",
                        "primary_predictor",
                        "bootstrap_samples",
                        "bootstrap_seed",
                    )
                },
                split="test",
            )
            started = time.time()
            for seed in source.cfg["seeds"]:
                for method in METHODS:
                    pending = [
                        s
                        for s in STAGES
                        if not task_complete(out, f"{s}_{seed}_{method}", ident)
                    ]
                    if not pending:
                        continue
                    print(
                        f"d{entry['dimension']} {seed}/{method}: {pending}", flush=True
                    )
                    rep, meta, predictors = source.load(seed, method)
                    for stage in pending:
                        with task_file(
                            out,
                            f"{stage}_{seed}_{method}",
                            ident,
                            "json" if stage == "probes" else "jsonl",
                        ) as stream:
                            if stage == "reconstruction":
                                reconstruction(
                                    source, rep, seed, method, cfg, args, stream
                                )
                            elif stage == "rollouts":
                                rollouts(
                                    source,
                                    rep,
                                    meta,
                                    predictors,
                                    seed,
                                    method,
                                    cfg,
                                    args,
                                    stream,
                                )
                            elif stage == "short_forecast":
                                short_forecasts(
                                    source,
                                    rep,
                                    meta,
                                    predictors,
                                    seed,
                                    method,
                                    cfg,
                                    args,
                                    stream,
                                )
                            else:
                                json.dump(
                                    frozen_probes(
                                        source,
                                        rep,
                                        entry["probes"][f"{seed}/{method}"],
                                        seed,
                                        method,
                                        cfg,
                                    ),
                                    stream,
                                    allow_nan=False,
                                )
                    del rep, meta, predictors
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()
            source.verify_unchanged()
            make_reports(out)
            write_json(
                out / "COMPLETE.json",
                dict(
                    id=ident,
                    split="test",
                    elapsed_seconds_this_invocation=time.time() - started,
                    device=device,
                    workers=workers,
                    threads=threads,
                    gpu=(
                        torch.cuda.get_device_name(torch.device(device))
                        if device.startswith("cuda")
                        else None
                    ),
                ),
            )
        from .comparison import compare

        compare(root, lock)
        write_json(
            root / "COMPLETE.json",
            dict(
                id=ident,
                split="test",
                dimensions=[e["dimension"] for e in lock["runs"]],
            ),
        )
        print(f"Complete: {root}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--project", required=True, type=Path)
    f.add_argument("--posthoc", required=True, type=Path)
    f.add_argument("--out", required=True, type=Path)
    f.add_argument("--run256", type=Path)
    e = sub.add_parser("evaluate")
    e.add_argument("--lock", required=True, type=Path)
    e.add_argument("--device", default="cuda:0")
    e.add_argument("--workers", type=int, default=8)
    e.add_argument("--threads", type=int, default=8)
    a = p.parse_args(argv)
    if a.command == "freeze":
        freeze(a.project, a.posthoc, a.out, a.run256)
    else:
        evaluate(a.lock, a.device, a.workers, a.threads)


if __name__ == "__main__":
    main()
