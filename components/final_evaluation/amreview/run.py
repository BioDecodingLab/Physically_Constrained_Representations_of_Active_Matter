"""Validation-only posthoc CLI. Explicit source import and separate output lock."""

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import platform
import time
from importlib.metadata import version
import numpy as np
import torch
from tqdm.auto import tqdm
from active_matter_repr.artifacts import read_json, write_json, digest, file_hash
from active_matter_repr.data import Snapshots, loader
from active_matter_repr.representations import scaled
from . import __version__
from .numerics import physical_variants, measure, finite_scalar
from .provenance import SourceRun, METHODS
from .spatial import structure
from .probes import combined_probes
from .reporting import make_reports


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Completed original runs/realizable_d256",
    )
    p.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="Original directory containing train/, valid/, test/",
    )
    p.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Separate output directory, outside the source run",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument(
        "--stages",
        nargs="+",
        choices=("reconstruction", "rollouts", "probes"),
        default=["reconstruction", "rollouts", "probes"],
    )
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument(
        "--predictors",
        nargs="+",
        choices=("linear", "mlp", "persistence"),
        default=["linear", "mlp", "persistence"],
    )
    p.add_argument(
        "--max-trajectories",
        type=int,
        default=0,
        help="0 = all validation trajectories; otherwise diagnostic subset",
    )
    p.add_argument("--max-leads", type=int, default=0, help="0 = all remaining steps")
    p.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Reconstruction snapshots only; 1 = every frame",
    )
    p.add_argument(
        "--spatial-stride",
        type=int,
        default=10,
        help="Spatial diagnostic every Nth reconstructed frame; fixed before test",
    )
    p.add_argument(
        "--verify-data-hashes",
        action="store_true",
        help="Rehash train/valid files even when original size/mtime match",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="First seed, AE/R-PC/POD, two trajectories, 12 leads; diagnostic only",
    )
    return p


def emit(stream, record):
    stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")


def task_complete(out, name, identity):
    marker = out / (name + ".done.json")
    if not marker.exists():
        return False
    meta = read_json(marker)
    path = out / meta["file"]
    if (
        meta["identity"] != identity
        or not path.exists()
        or file_hash(path) != meta["sha256"]
    ):
        raise ValueError(f"Completed task differs from its fingerprint: {name}")
    return True


@contextmanager
def task_file(out, name, identity, extension="jsonl"):
    path = out / (name + "." + extension)
    temporary = path.with_suffix(path.suffix + ".partial")
    with open(temporary, "w") as stream:
        yield stream
    os.replace(temporary, path)
    write_json(
        out / (name + ".done.json"),
        {"identity": identity, "file": path.name, "sha256": file_hash(path)},
    )


@torch.no_grad()
def reconstruction(source, rep, seed, method, cfg, args, stream):
    ds = Snapshots(cfg, source.manifest, getattr(args, "split", "valid"))
    if args.max_trajectories:
        ds.rows = ds.rows[: args.max_trajectories]
    ds.index = [
        (i, t)
        for i, r in enumerate(ds.rows)
        for t in range(0, len(r["times"]), args.frame_stride)
    ]
    try:
        for batch in tqdm(
            loader(ds, cfg), desc=f"reconstruction {seed}/{method}", leave=False
        ):
            truth = batch["x"].to(cfg["device"])
            z = rep.encode(source.transform.normalize(truth))
            variants, diagnostics = physical_variants(rep, z, source.transform, method)
            for j in range(len(truth)):
                i, t = int(batch["trajectory"][j]), int(batch["frame"][j])
                row = ds.rows[i]
                tag = {
                    "seed": seed,
                    "method": method,
                    "trajectory_id": row["id"],
                    "group": row["group"],
                    "frame": t,
                }
                for variant, x in variants.items():
                    measurements = measure(
                        x[j],
                        truth[j],
                        source.transform,
                        row["spacing"],
                        source.obs_scale,
                        cfg["derivative"],
                    )
                    extra = {
                        key: finite_scalar(value[j])
                        for key, value in diagnostics.items()
                    }
                    emit(
                        stream,
                        {
                            "kind": "reconstruction",
                            **tag,
                            "variant": variant,
                            **measurements,
                            **extra,
                        },
                    )
                    if t % args.spatial_stride == 0:
                        emit(
                            stream,
                            {
                                "kind": "spatial",
                                **tag,
                                "variant": variant,
                                **structure(x[j], row["spacing"]),
                            },
                        )
                if t % args.spatial_stride == 0:
                    emit(
                        stream,
                        {
                            "kind": "spatial",
                            **tag,
                            "variant": "truth",
                            **structure(truth[j], row["spacing"]),
                        },
                    )
    finally:
        ds.close()


@torch.no_grad()
def rollouts(source, rep, metadata, predictors, seed, method, cfg, args, stream):
    ds = Snapshots(cfg, source.manifest, getattr(args, "split", "valid"))
    if args.max_trajectories:
        ds.rows = ds.rows[: args.max_trajectories]
    h = cfg["history"]
    scaler = metadata["latent_scaler"]
    mean = torch.as_tensor(scaler["mean"], device=cfg["device"], dtype=torch.float32)
    scale = torch.as_tensor(scaler["scale"], device=cfg["device"], dtype=torch.float32)
    try:
        for i, row in enumerate(
            tqdm(ds.rows, desc=f"rollouts {seed}/{method}", leave=False)
        ):
            first = torch.stack([ds.frame(row, t) for t in range(h)]).to(cfg["device"])
            # NumPy scaling and float32 conversion match legacy Windows precisely.
            initial = scaled(
                rep.encode(source.transform.normalize(first)).cpu().numpy(), scaler
            ).astype(np.float32)
            initial = torch.from_numpy(initial).to(cfg["device"])[None]
            params = torch.as_tensor(
                scaled(np.array([row["parameters"]]), metadata["parameter_scaler"]),
                dtype=torch.float32,
                device=cfg["device"],
            )
            histories = {
                k: initial.clone() for k in args.predictors if k != "persistence"
            }
            first_failure = {}
            available = len(row["times"]) - h
            count = min(available, args.max_leads) if args.max_leads else available
            for lead in range(1, count + 1):
                t = h + lead - 1
                truth = ds.frame(row, t).to(cfg["device"])
                tag = {
                    "kind": "rollout",
                    "seed": seed,
                    "method": method,
                    "trajectory_id": row["id"],
                    "group": row["group"],
                    "origin": h - 1,
                    "lead": lead,
                    "time_since_origin": row["times"][t] - row["times"][h - 1],
                }
                for predictor in args.predictors:
                    if predictor == "persistence":
                        variants, diagnostics = {"legacy": first[-1:]}, {}
                        latent_max = None
                    elif predictor in first_failure:
                        variants = {
                            k: torch.full_like(truth[None], float("nan"))
                            for k in (
                                ("legacy", "direct_corrected")
                                if method in ("rae", "rpc")
                                else ("legacy", "projected")
                            )
                        }
                        diagnostics = {}
                        latent_max = None
                    else:
                        z = predictors[predictor](histories[predictor], params)
                        latent_max = finite_scalar(z.abs().max())
                        if not torch.isfinite(z).all():
                            first_failure[predictor] = lead
                        histories[predictor] = torch.cat(
                            (histories[predictor][:, 1:], z[:, None]), 1
                        )
                        variants, diagnostics = physical_variants(
                            rep, z * scale + mean, source.transform, method
                        )
                    for variant, x in variants.items():
                        extra = {k: finite_scalar(v[0]) for k, v in diagnostics.items()}
                        emit(
                            stream,
                            {
                                **tag,
                                "predictor": predictor,
                                "variant": variant,
                                "max_abs_standardized_latent": latent_max,
                                "first_nonfinite_latent_lead": first_failure.get(
                                    predictor
                                ),
                                **measure(
                                    x[0],
                                    truth,
                                    source.transform,
                                    row["spacing"],
                                    source.obs_scale,
                                    cfg["derivative"],
                                ),
                                **extra,
                            },
                        )
    finally:
        ds.close()


def own_fingerprint():
    return digest(
        {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    )


def main(argv=None):
    args = parser().parse_args(argv)
    if (
        min(args.batch_size, args.threads, args.frame_stride, args.spatial_stride) < 1
        or min(args.workers, args.max_leads, args.max_trajectories) < 0
    ):
        raise ValueError("Invalid execution/subsampling settings")
    torch.set_num_threads(args.threads)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    source = SourceRun(
        args.run_dir, args.data_root, args.device, args.verify_data_hashes
    )
    args.seeds = args.seeds or source.cfg["seeds"]
    if args.quick:
        args.seeds = [source.cfg["seeds"][0]]
        args.methods = ["ae", "rpc", "pod"]
        args.max_trajectories = 2
        args.max_leads = 12
        args.frame_stride = 20
        args.spatial_stride = 20
    for name in ("seeds", "methods", "predictors", "stages"):
        if len(set(getattr(args, name))) != len(getattr(args, name)):
            raise ValueError(f"Duplicate {name}")
    if not set(args.seeds).issubset(set(source.cfg["seeds"])):
        raise ValueError("Unknown source seed")
    out = args.out.resolve()
    if (
        out == source.root
        or out.is_relative_to(source.root)
        or source.root.is_relative_to(out)
    ):
        raise ValueError(
            "Output must be separate from the completed source run (not its ancestor or descendant)"
        )
    if (
        out == source.data_root
        or out.is_relative_to(source.data_root)
        or source.data_root.is_relative_to(out)
    ):
        raise ValueError("Output must be separate from source data")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / ".analysis.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "Another process is writing this analysis directory"
            ) from None
        print("Checking source checkpoints and cache fingerprints...", flush=True)
        source.prepare(args.seeds, args.methods, "probes" in args.stages)
        protocol = {
            k: v
            for k, v in vars(args).items()
            if k
            not in (
                "run_dir",
                "data_root",
                "out",
                "device",
                "workers",
                "threads",
                "verify_data_hashes",
            )
        }
        info = {
            "version": __version__,
            "source_run_id": source.cfg["run_id"],
            "source_config_id": source.cfg["id"],
            "source_files": source.files,
            "analysis_source": own_fingerprint(),
            "split": "valid",
            "protocol": protocol,
            "software_versions": {
                name: version(name)
                for name in ("torch", "numpy", "scipy", "scikit-learn", "h5py")
            },
            "scope": (
                "diagnostic subset"
                if args.max_trajectories
                or args.max_leads
                or args.frame_stride != 1
                or len(args.seeds) != len(source.cfg["seeds"])
                else "full validation"
            ),
            "synthetic": source.cfg["synthetic"],
        }
        identity = digest(info)
        if (out / "analysis_manifest.json").exists():
            if read_json(out / "analysis_manifest.json") != {"id": identity, **info}:
                raise ValueError(
                    "Output belongs to different inputs/code/settings. Choose a new output directory."
                )
        else:
            if any(p.name != ".analysis.lock" for p in out.iterdir()):
                raise ValueError("Refusing a nonempty unknown output directory")
            write_json(out / "analysis_manifest.json", {"id": identity, **info})
        cfg = {
            **source.cfg,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "threads": args.threads,
        }
        started = time.time()
        for seed in args.seeds:
            for method in args.methods:
                names = {stage: f"{stage}_{seed}_{method}" for stage in args.stages}
                pending = [
                    stage
                    for stage, name in names.items()
                    if not task_complete(out, name, identity)
                ]
                if not pending:
                    print(f"Skip completed {seed}/{method}", flush=True)
                    continue
                print(f"Analyzing {seed}/{method}: {', '.join(pending)}", flush=True)
                rep, metadata, predictors = source.load(seed, method)
                for stage in pending:
                    with task_file(
                        out,
                        names[stage],
                        identity,
                        "json" if stage == "probes" else "jsonl",
                    ) as f:
                        if stage == "reconstruction":
                            reconstruction(source, rep, seed, method, cfg, args, f)
                        elif stage == "rollouts":
                            rollouts(
                                source,
                                rep,
                                metadata,
                                predictors,
                                seed,
                                method,
                                cfg,
                                args,
                                f,
                            )
                        else:
                            json.dump(
                                combined_probes(source, seed, method),
                                f,
                                allow_nan=False,
                                indent=2,
                            )
                del rep, metadata, predictors
                if cfg["device"].startswith("cuda"):
                    torch.cuda.empty_cache()
        print(
            "Verifying unchanged source artifacts and generating reports...", flush=True
        )
        source.verify_unchanged()
        make_reports(out)
        write_json(
            out / "COMPLETE.json",
            {
                "id": identity,
                "split": "valid",
                "elapsed_seconds_this_invocation": time.time() - started,
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "numpy": np.__version__,
                "device": args.device,
                "batch_size": args.batch_size,
                "workers": args.workers,
                "threads": args.threads,
                "synthetic": source.cfg["synthetic"],
            },
        )
        print(f"Complete: {out}", flush=True)


if __name__ == "__main__":
    main()
