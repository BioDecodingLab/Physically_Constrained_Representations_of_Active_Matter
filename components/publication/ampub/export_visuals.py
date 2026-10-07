"""Read-only visual export from the frozen protocol and fitted representations."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from .provenance import inventory, file_hash


def pca_view(train, valid):
    from sklearn.decomposition import PCA

    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale <= 1e-8] = 1
    pca = PCA(n_components=2, svd_solver="full")
    pca.fit((train - mean) / scale)
    return (
        pca.transform((valid - mean) / scale),
        pca.explained_variance_ratio_,
        mean,
        scale,
        pca.components_,
    )


def physical_views(x, spacing):
    c = x[0]
    vx = x[1]
    vy = x[2]

    def deriv(a, axis, dx):
        n = a.shape[axis]
        k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
        k[n // 2] = 0 if n % 2 == 0 else k[n // 2]
        shape = [1, 1]
        shape[axis] = n
        return np.fft.ifft(
            np.fft.fft(a, axis=axis) * (1j * k.reshape(shape)), axis=axis
        ).real

    speed = np.sqrt(vx * vx + vy * vy)
    omega = deriv(vy, 0, spacing[0]) - deriv(vx, 1, spacing[1])
    delta = np.hypot(x[3] - x[6], x[4] + x[5])
    order = np.divide(delta, c, out=np.full_like(delta, np.nan), where=c > 1e-8)
    strain_norm = np.sqrt(np.sum(x[7:11] ** 2, axis=0))
    return np.stack([c, speed, omega, order, strain_norm])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lock", required=True)
    p.add_argument("--evaluator", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dimension", type=int, default=256)
    p.add_argument("--device", default="cpu")
    a = p.parse_args(argv)
    evaluator = Path(a.evaluator).expanduser().resolve()
    out = Path(a.out).expanduser().resolve()
    lockpath = Path(a.lock).expanduser().resolve()
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(evaluator), str(evaluator / "vendor")]
    from active_matter_repr.artifacts import digest, read_json
    from active_matter_repr.data import Snapshots
    from amreview.provenance import SourceRun, METHODS, inside
    from amreview.final import own_fingerprint
    from amreview.numerics import physical_variants
    import torch

    lock = read_json(lockpath)
    body = dict(lock)
    ident = body.pop("id")
    if (
        digest(body) != ident
        or lock["analysis_source"] != own_fingerprint()
        or lock["synthetic"]
    ):
        raise ValueError("Frozen lock identity or evaluator source mismatch")
    entry = next(e for e in lock["runs"] if e["dimension"] == a.dimension)
    protected = [evaluator, lockpath.parent, Path(lock["data_root"]).resolve()] + [
        Path(e["run_dir"]).resolve() for e in lock["runs"]
    ]
    if any(out == r or r in out.parents or out in r.parents for r in protected):
        raise ValueError("Export destination must be separate from source artifacts")
    before_eval = inventory(evaluator)
    before_final = inventory(lockpath.parent)
    source = SourceRun(entry["run_dir"], lock["data_root"], a.device, True)
    source.prepare(source.cfg["seeds"], METHODS, True)
    if source.files != entry["source_files"] or source.cfg["run_id"] != entry["run_id"]:
        raise ValueError("Source changed after freeze")
    for name, h in source.manifest["files"].items():
        if (
            Path(name).parts[0] == "test"
            and file_hash(inside(source.data_root, name)) != h
        ):
            raise ValueError("Test file hash mismatch")
    cfg = source.cfg
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    arrays = {}
    views = []
    ids = []
    selected = []
    ds = Snapshots(cfg, source.manifest, "test")
    # Stable design-based selection: first trajectory at minimum, median, maximum parameter tuple.
    groups = sorted({tuple(r["parameters"]) for r in ds.rows})
    chosen = [groups[0], groups[len(groups) // 2], groups[-1]]
    rows = [
        next(
            r
            for r in sorted(ds.rows, key=lambda r: r["id"])
            if tuple(r["parameters"]) == g
        )
        for g in chosen
    ]
    try:
        for row in rows:
            t = len(row["times"]) // 2
            x = ds.frame(row, t).numpy()
            views.append(physical_views(x, row["spacing"]))
            ids.append(row["id"])
            selected.append(
                {
                    "trajectory_id": row["id"],
                    "parameters": row["parameters"],
                    "frame": t,
                    "time": row["times"][t],
                }
            )
        arrays["dataset_views"] = np.array(views, dtype=np.float32)
        # Gallery uses the median parameter group, midpoint frame, fixed seed 1729.
        row = rows[1]
        t = len(row["times"]) // 2
        truth = ds.frame(row, t)[None].to(a.device)
        arrays["gallery_truth"] = physical_views(
            truth[0].cpu().numpy(), row["spacing"]
        ).astype(np.float32)
        for method in ["ae", "rpc", "pod"]:
            with torch.no_grad():
                rep, _, _ = source.load(1729, method)
                z = rep.encode(source.transform.normalize(truth))
                vv, _ = physical_variants(rep, z, source.transform, method)
                variant = {
                    "ae": "legacy",
                    "rpc": "direct_corrected",
                    "pod": "projected",
                }[method]
                arrays["gallery_" + method] = physical_views(
                    vv[variant][0].cpu().numpy(), row["spacing"]
                ).astype(np.float32)
        movie = []
        for t in range(len(row["times"])):
            x = ds.frame(row, t).numpy()
            movie.append(physical_views(x, row["spacing"])[1:4, ::2, ::2])
        arrays["reference_movie"] = np.array(movie, dtype=np.float32)
        arrays["reference_times"] = np.array(row["times"])
    finally:
        ds.close()
    cache_records = {}
    for seed in source.cfg["seeds"]:
        for method in ["ae", "rpc"]:
            train, _ = source.cache(seed, method, "train")
            valid, vr = source.cache(seed, method, "valid")
            coords, variance, mean, scale, components = pca_view(train["z"], valid["z"])
            prefix = f"pca_{seed}_{method}_"
            arrays.update(
                {
                    prefix + "xy": coords,
                    prefix + "variance": variance,
                    prefix + "mean": mean,
                    prefix + "scale": scale,
                    prefix + "components": components,
                    prefix + "observables": valid["observables"],
                    prefix + "trajectory": valid["trajectory"],
                    prefix + "frame": valid["frame"],
                }
            )
            cache_records[f"{seed}/{method}"] = {
                "train_rows": len(train["z"]),
                "valid_rows": len(valid["z"]),
                "valid_ids": [r["id"] for r in vr],
            }
    source.verify_unchanged()
    if before_eval != inventory(evaluator) or before_final != inventory(
        lockpath.parent
    ):
        raise RuntimeError("Protected source changed")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "visuals.npz", **arrays)
    protocols = {}
    for e in lock["runs"]:
        rr = Path(e["run_dir"])
        for name in ["resolved_config.json", "selection.json", "observable_stats.json"]:
            if file_hash(rr / name) != e["source_files"][name]:
                raise ValueError("Changed protocol artifact")
        protocols[str(e["dimension"])] = {
            name: read_json(rr / name)
            for name in [
                "resolved_config.json",
                "selection.json",
                "observable_stats.json",
            ]
        }
    meta = {
        "protocols": protocols,
        "final_evaluation_id": ident,
        "dimension": a.dimension,
        "seeds": source.cfg["seeds"],
        "methods": ["ae", "rpc"],
        "classification": "publication-only descriptive visualisation; no tuning or inferential latent interpretation",
        "selection": "minimum/median/maximum sorted parameter tuple; first trajectory ID; midpoint frame",
        "dataset_selected": selected,
        "gallery": selected[1],
        "gallery_seed": 1729,
        "cache_records": cache_records,
        "PCA": "standardized train latents; fit train only; transform valid; separate basis for each method and seed",
        "length": cfg["expected_length"],
        "fields": cfg["fields"],
        "source_files": source.files,
        "data_files": source.manifest["files"],
        "visuals_sha256": file_hash(out / "visuals.npz"),
        "protected_sources_unchanged": True,
    }
    (out / "VISUAL_EXPORT.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("Read-only visual export complete:", out)


if __name__ == "__main__":
    main()
