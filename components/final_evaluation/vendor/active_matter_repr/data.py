"""Strict direct reader of The Well HDF5 layout, with bounded worker-local handles."""
from collections import OrderedDict
from pathlib import Path
import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from .artifacts import file_hash, digest


def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def _array(ds, sample, frame=None):
    ix = []
    if bool(ds.attrs.get("sample_varying", False)):
        ix.append(sample)
    if bool(ds.attrs.get("time_varying", False)):
        if frame is None:
            raise ValueError(f"Unexpected time-varying parameter/coordinate: {ds.name}")
        ix.append(frame)
    return np.asarray(ds[tuple(ix)] if ix else ds[()])


def read_frame(handle, sample, frame, mapping, shape):
    parts = []
    for block, order, count in (("c", 0, 1), ("v", 1, 2), ("D", 2, 4), ("E", 2, 4)):
        ds = handle[f"t{order}_fields/{mapping[block]}"]
        if "dim_varying" not in ds.attrs or not all(ds.attrs["dim_varying"]):
            raise ValueError(f"Only full spatial fields supported: {ds.name}")
        arr = _array(ds, sample, frame)
        if tuple(arr.shape[:2]) != tuple(shape) or arr.size != shape[0] * shape[1] * count:
            raise ValueError(f"Unsupported shape for {ds.name}: {arr.shape}")
        parts.append(arr.reshape(*shape, count))
    x = np.concatenate(parts, axis=-1).transpose(2, 0, 1).copy()
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite source field")
    return torch.from_numpy(x).float()


def build_manifest(cfg):
    root = Path(cfg["data_root"])
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root missing: {root}; expected train/, valid/, test/")
    rows, hashes, initials, shared_initials = [], {}, {}, {}
    for split, expected in zip(("train", "valid", "test"), cfg["expected_counts"]):
        files = sorted(set((root / split).glob("*.h5")) | set((root / split).glob("*.hdf5")))
        count = 0
        for path in files:
            rel = str(path.relative_to(root))
            hashes[rel] = file_hash(path)
            with h5py.File(path, "r") as h:
                if text(h.attrs.get("dataset_name", "")) != "active_matter":
                    raise ValueError(f"Unexpected dataset_name in {rel}")
                if bool(h.attrs.get("synthetic_fixture", False)) != cfg["synthetic"]:
                    raise ValueError("Synthetic/real configuration mismatch")
                dims = [text(x) for x in h["dimensions"].attrs["spatial_dims"]]
                if dims != ["x", "y"]:
                    raise ValueError(f"Expected explicit x,y axes, found {dims}")
                periodic = set()
                for bc in h["boundary_conditions"].values():
                    if text(bc.attrs.get("bc_type", "")).lower() == "periodic":
                        periodic.update(text(x) for x in bc.attrs["associated_dims"])
                if periodic != {"x", "y"}:
                    raise ValueError("Both spatial axes must be periodic")
                n = int(h.attrs["n_trajectories"])
                for sample in range(n):
                    coords = [_array(h[f"dimensions/{dim}"], sample).astype(float) for dim in dims]
                    coordinate_spacing = [float(np.diff(a).mean()) for a in coords]
                    if any(s <= 0 or not np.allclose(np.diff(a), s, rtol=1e-4, atol=1e-7) for a, s in zip(coords, coordinate_spacing)):
                        raise ValueError("Nonuniform/nonascending grid")
                    shape = [len(a) for a in coords]
                    time_ds = h["dimensions/time"]
                    times = np.asarray(time_ds[sample] if time_ds.attrs.get("sample_varying", False) else time_ds[()]).astype(float)
                    if shape != cfg["expected_shape"] or len(times) != cfg["expected_frames"]:
                        raise ValueError(f"Unexpected grid/time size: {shape}, {len(times)}")
                    if not np.allclose(np.diff(times), cfg["expected_dt"]):
                        raise ValueError("Stored time interval differs from configured expectation")
                    conventions = []
                    for a, dx, ngrid, length in zip(coords, coordinate_spacing, shape, cfg["expected_length"]):
                        if np.isclose(dx * ngrid, length, rtol=1e-5):
                            conventions.append("endpoint_excluded")
                        elif np.isclose(a[-1] - a[0], length, rtol=1e-5):
                            conventions.append("endpoint_inclusive_labels; periodic operator uses L/N")
                        else:
                            raise ValueError("Coordinate range incompatible with configured physical period")
                    if "L" in h.attrs and not np.allclose(cfg["expected_length"], float(h.attrs["L"])):
                        raise ValueError("Configured period disagrees with stored L")
                    # The public data use inclusive coordinate labels. Derivatives use
                    # the physical periodic domain L/N; validate against stored E at G1.
                    spacing = [length / ngrid for length, ngrid in zip(cfg["expected_length"], shape)]
                    params = []
                    for name in ("alpha", "zeta"):
                        value = _array(h[f"scalars/{name}"], sample) if f"scalars/{name}" in h else np.asarray(h.attrs[name])
                        if value.size != 1 or not np.isfinite(value).all():
                            raise ValueError(f"Invalid parameter {name}")
                        params.append(float(value.item()))
                    first = read_frame(h, sample, 0, cfg["fields"], shape)
                    # Shared initial conditions at DIFFERENT control parameters can
                    # be valid experimental design; record them, do not deduplicate.
                    import hashlib
                    initial_hash = hashlib.sha256(first.numpy().tobytes()).hexdigest()
                    initial_key = (initial_hash, tuple(params))
                    if initial_key in initials:
                        raise ValueError(f"Duplicate initial state at identical parameters: {rel}:{sample} and {initials[initial_key]}")
                    initials[initial_key] = f"{rel}:{sample}"
                    shared_initials.setdefault(initial_hash, []).append({"id": f"{rel}:{sample}", "parameters": params, "split": split})
                    rows.append({"id": f"{rel}:{sample}", "path": rel, "sample": sample,
                                 "split": split, "parameters": params, "group": str(tuple(params)),
                                 "shape": shape, "spacing": spacing, "coordinate_spacing": coordinate_spacing,
                                 "coordinate_convention": conventions, "times": times.tolist(), "initial_sha256": initial_hash})
                    count += 1
        if count != expected:
            raise ValueError(f"{split}: found {count} trajectories, expected {expected}; check root and explicit counts")
    body = {"trajectories": rows, "files": hashes, "fields": cfg["fields"],
            "shared_initial_conditions": [v for v in shared_initials.values() if len(v) > 1]}
    body["id"] = digest(body)
    return body


class Snapshots(Dataset):
    def __init__(self, cfg, manifest, split):
        self.root, self.mapping = cfg["data_root"], cfg["fields"]
        self.rows = [r for r in manifest["trajectories"] if r["split"] == split]
        self.index = [(i, t) for i, r in enumerate(self.rows) for t in range(len(r["times"]))]
        self._handles, self._pid = OrderedDict(), os.getpid()

    def __len__(self):
        return len(self.index)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        return state

    def frame(self, row, t):
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        path = row["path"]
        if path not in self._handles:
            self._handles[path] = h5py.File(Path(self.root) / path, "r")
        self._handles.move_to_end(path)
        while len(self._handles) > 4:
            _, handle = self._handles.popitem(last=False)
            handle.close()
        return read_frame(self._handles[path], row["sample"], t, self.mapping, row["shape"])

    def __getitem__(self, index):
        i, t = self.index[index]
        return {"x": self.frame(self.rows[i], t), "trajectory": i, "frame": t}

    def close(self):
        for h in self._handles.values():
            h.close()
        self._handles.clear()


def _numpy_collate(samples):
    """Optional pipe transfer avoids tensor-sharing sockets in restricted hosts."""
    return {key: np.stack([np.asarray(sample[key]) for sample in samples]) for key in samples[0]}


class _TensorBatches:
    def __init__(self, batches, pin):
        self.batches, self.pin = batches, pin

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for batch in self.batches:
            result = {key: torch.from_numpy(value) for key, value in batch.items()}
            yield {key: value.pin_memory() for key, value in result.items()} if self.pin else result


def loader(dataset, cfg, shuffle=False, seed=0, batch_size=None):
    # Closing parent handles before workers fork avoids inherited active HDF5 state.
    if hasattr(dataset, "close"):
        dataset.close()
    numpy_transfer = cfg.get("worker_transport", "torch") == "numpy" and cfg["workers"] > 0
    from torch.utils.data import default_collate
    batches = DataLoader(dataset, batch_size=batch_size or cfg["batch_size"], shuffle=shuffle,
                      num_workers=cfg["workers"], pin_memory=cfg["device"].startswith("cuda"),
                      generator=torch.Generator().manual_seed(seed), persistent_workers=False,
                      collate_fn=_numpy_collate if numpy_transfer else default_collate,
                      multiprocessing_context="spawn" if cfg["workers"] else None,
                      timeout=120 if cfg["workers"] else 0)
    return _TensorBatches(batches, cfg["device"].startswith("cuda")) if numpy_transfer else batches
