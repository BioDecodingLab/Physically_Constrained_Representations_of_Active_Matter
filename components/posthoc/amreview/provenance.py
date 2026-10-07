"""Explicit read-only import of audited v0.2 artifacts; no gate bypasses."""

from pathlib import Path
from collections import Counter
import numpy as np
from active_matter_repr.artifacts import read_json, file_hash, digest, load_checkpoint
from active_matter_repr.pipeline import source_id
from active_matter_repr.models import Autoencoder, POD, LatentMLP
from active_matter_repr.forecasting import LinearAR
from active_matter_repr.transforms import FieldTransform

V020_SOURCE = "9e85ef699942c8679b712c0af82ab7d8bc3bc4c50897758eb3dbd8809cf5475b"
METHODS = ("ae", "pc", "rae", "rpc", "pod")


def inside(root, relative):
    root = Path(root).resolve()
    p = (root / relative).resolve()
    if Path(relative).is_absolute() or not p.is_relative_to(root):
        raise ValueError(f"Unsafe artifact path: {relative}")
    return p


class SourceRun:
    def __init__(self, run_dir, data_root, device, verify_hashes=False):
        self.root, self.data_root = Path(run_dir).resolve(), Path(data_root).resolve()
        self.cfg = read_json(self.root / "resolved_config.json")
        self.audit = read_json(self.root / "audit.json")
        self.manifest = read_json(self.root / "manifest.json")
        self.selection = read_json(self.root / "selection.json")
        self.files = {}
        if source_id() != V020_SOURCE or self.audit["source"] != V020_SOURCE:
            raise ValueError(
                "This importer supports the fingerprinted v0.2.0 source only"
            )
        body = dict(self.manifest)
        identity = body.pop("id")
        if digest(body) != identity:
            raise ValueError("Manifest fingerprint mismatch")
        scientific = {
            k: v
            for k, v in self.cfg.items()
            if k
            not in {
                "id",
                "run_id",
                "device",
                "workers",
                "threads",
                "worker_transport",
                "run_dir",
                "data_root",
                "auto_download",
            }
        }
        if (
            digest(scientific) != self.cfg["id"]
            or self.audit["config_id"] != self.cfg["id"]
            or not self.audit["passed"]
        ):
            raise ValueError("Invalid source audit/configuration")
        run_id = digest(
            {"config": self.cfg["id"], "manifest": identity, "source": V020_SOURCE}
        )
        if run_id != self.audit["identity"]:
            raise ValueError("Invalid source run identity")
        counts = Counter(r["split"] for r in self.manifest["trajectories"])
        if [counts[s] for s in ("train", "valid", "test")] != self.cfg[
            "expected_counts"
        ]:
            raise ValueError("Manifest split counts differ from source configuration")
        if len({r["id"] for r in self.manifest["trajectories"]}) != sum(
            counts.values()
        ):
            raise ValueError("Duplicate trajectory IDs")
        for r in self.manifest["trajectories"]:
            if Path(r["path"]).parts[0] != r["split"]:
                raise ValueError("Trajectory directory does not match official split")
            inside(self.data_root, r["path"])
        # Only training/validation file contents may be accessed by this release.
        for name, expected in self.manifest["files"].items():
            if Path(name).parts[0] not in ("train", "valid"):
                continue
            p = inside(self.data_root, name)
            stat, stamp = p.stat(), self.audit["file_stamps"][name]
            if (
                verify_hashes
                or stat.st_size != stamp["size"]
                or stat.st_mtime_ns != stamp["mtime_ns"]
            ):
                if file_hash(p) != expected:
                    raise ValueError(f"Data checksum differs from original run: {name}")
        for name in (
            "resolved_config.json",
            "audit.json",
            "manifest.json",
            "selection.json",
            "transform.json",
            "observable_stats.json",
        ):
            self.record(name)
        self.cfg = {
            **self.cfg,
            "data_root": str(self.data_root),
            "device": device,
            "run_id": run_id,
        }
        self.transform = FieldTransform(**read_json(self.root / "transform.json")).to(
            device
        )
        obs = read_json(self.root / "observable_stats.json")
        self.obs_scale = np.array([np.inf if x is None else x for x in obs["std"]])
        if set(self.selection["seeds"]) != {str(s) for s in self.cfg["seeds"]}:
            raise ValueError(
                "All configured confirmation seeds must have selected models"
            )

    def record(self, name):
        p = inside(self.root, name)
        self.files[name] = file_hash(p)
        return p

    def prepare(self, seeds, methods, need_probes):
        for s in seeds:
            for m in methods:
                self.record(self.rep_path(s, m))
                self.record(f"analysis/{s}/{m}/probes.json")
                self.record(f"forecasters/{s}/{m}/best.pt")
                self.record(f"forecasters/{s}/{m}/linear.json")
                if need_probes:
                    for split in ("train", "valid"):
                        self.record(f"analysis/{s}/{m}/{split}/metadata.json")
                        self.record(f"analysis/{s}/{m}/{split}/latents.npz")

    def rep_path(self, seed, method):
        return (
            "representations/pod.pt"
            if method == "pod"
            else self.selection["seeds"][str(seed)][method]
        )

    def load(self, seed, method):
        path = self.rep_path(seed, method)
        state = load_checkpoint(inside(self.root, path))
        if method == "pod":
            if (
                state["config_id"] != self.cfg["id"]
                or state.get("run_id") != self.cfg["run_id"]
            ):
                raise ValueError("POD identity mismatch")
            rep = POD(state["components"], state["center"], self.cfg["expected_shape"])
        else:
            if (
                state["config"]["id"] != self.cfg["id"]
                or state["config"].get("run_id") != self.cfg["run_id"]
            ):
                raise ValueError("Representation identity mismatch")
            if state["transform"] != self.transform.as_dict():
                raise ValueError("Representation transform mismatch")
            rep = Autoencoder(
                self.cfg["expected_shape"],
                self.cfg["latent_dim"],
                self.cfg["widths"],
                realizable=method in ("rae", "rpc"),
                transform=self.transform,
            )
            rep.load_state_dict(state["model"])
        rep = rep.to(self.cfg["device"]).eval()
        metadata = read_json(self.root / f"analysis/{seed}/{method}/probes.json")
        if metadata["checkpoint"] != self.files[path]:
            raise ValueError("Probe/representation fingerprint mismatch")
        mlp = LatentMLP(
            self.cfg["latent_dim"], self.cfg["history"], self.cfg["forecast_width"]
        ).to(self.cfg["device"])
        state = load_checkpoint(self.root / f"forecasters/{seed}/{method}/best.pt")
        expected = digest(
            {
                "config": self.cfg["id"],
                "run": self.cfg["run_id"],
                "seed": seed,
                "scaler": metadata["latent_scaler"],
            }
        )
        if state["identity"] != expected:
            raise ValueError("Forecaster identity mismatch")
        mlp.load_state_dict(state["model"])
        linear = read_json(self.root / f"forecasters/{seed}/{method}/linear.json")
        predictors = {
            "linear": LinearAR(linear["coef"], linear["intercept"])
            .to(self.cfg["device"])
            .eval(),
            "mlp": mlp.eval(),
        }
        return rep, metadata, predictors

    def cache(self, seed, method, split):
        if split not in ("train", "valid"):
            raise ValueError(
                "Test access is not implemented in this validation release"
            )
        folder = self.root / f"analysis/{seed}/{method}/{split}"
        meta = read_json(folder / "metadata.json")
        rows = [r for r in self.manifest["trajectories"] if r["split"] == split]
        index = [[i, t] for i, r in enumerate(rows) for t in range(len(r["times"]))]
        expected = {
            "identity": self.files[self.rep_path(seed, method)],
            "config_id": self.cfg["id"],
            "run_id": self.cfg["run_id"],
            "ids": [r["id"] for r in rows],
            "index": index,
        }
        if meta != {"id": digest(expected), **expected}:
            raise ValueError("Latent cache belongs to a different split/run/checkpoint")
        with np.load(folder / "latents.npz", allow_pickle=False) as f:
            cache = {k: f[k] for k in f.files}
        if list(zip(cache["trajectory"].tolist(), cache["frame"].tolist())) != [
            tuple(v) for v in index
        ]:
            raise ValueError("Latent cache ordering mismatch")
        if not all(np.isfinite(v).all() for v in cache.values()):
            raise ValueError("Nonfinite cached values")
        return cache, rows

    def verify_unchanged(self):
        for name, expected in self.files.items():
            if file_hash(inside(self.root, name)) != expected:
                raise ValueError(f"Source artifact changed during analysis: {name}")
