"""Analytic periodic HDF5 fixtures, explicitly marked as non-research data."""
from pathlib import Path
import math
import numpy as np
import h5py
import torch
import yaml
from .physics import strain


def create_fixture(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    data_root = directory / "data"
    if data_root.exists():
        raise FileExistsError("Fixture data already exist; use a fresh directory")
    n, frames, length = 16, 15, 2 * math.pi
    coordinates = np.arange(n, dtype=np.float32) * length / n
    xx, yy = torch.meshgrid(torch.from_numpy(coordinates), torch.from_numpy(coordinates), indexing="ij")
    for split, count, offset in (("train", 4, 0), ("valid", 2, 10), ("test", 2, 20)):
        target = data_root / split
        target.mkdir(parents=True)
        fields = torch.zeros(count, frames, 11, n, n)
        for i in range(count):
            for t in range(frames):
                phase = .13 * (i + offset) + .09 * t
                amplitude = .3 + .08 * (i % 2) + .003 * t
                x = fields[i, t]
                x[0] = 1
                x[1] = amplitude * torch.sin(xx + phase) * torch.cos(yy - phase)
                x[2] = -amplitude * torch.cos(xx + phase) * torch.sin(yy - phase)
                order = .1 + .03 * torch.sin(xx + yy + phase)
                x[3], x[6] = .5 + order, .5 - order
                x[4] = x[5] = .02 * torch.sin(xx - yy + phase)
                x[7:11] = strain(x, (length/n, length/n))
        with h5py.File(target / "fixture.h5", "w") as h:
            h.attrs.update(dataset_name="active_matter", n_trajectories=count, synthetic_fixture=True)
            dims = h.create_group("dimensions")
            dims.attrs["spatial_dims"] = ["x", "y"]
            dims.create_dataset("x", data=coordinates)
            dims.create_dataset("y", data=coordinates)
            dims.create_dataset("time", data=np.arange(frames, dtype=np.float32) * .25)
            bcs = h.create_group("boundary_conditions")
            for dim in ("x", "y"):
                bc = bcs.create_group(dim)
                bc.attrs["associated_dims"] = [dim]
                bc.attrs["bc_type"] = "PERIODIC"
            scalars = h.create_group("scalars")
            for name, values in (("alpha", [-1 - i % 2 for i in range(count)]), ("zeta", [1 + 2 * (i % 2) for i in range(count)])):
                ds = scalars.create_dataset(name, data=np.array(values, dtype=np.float32))
                ds.attrs["sample_varying"] = True
            for name, order, block in (("concentration", 0, slice(0, 1)), ("velocity", 1, slice(1, 3)), ("D", 2, slice(3, 7)), ("E", 2, slice(7, 11))):
                group = h.require_group(f"t{order}_fields")
                arr = fields[:, :, block].numpy().transpose(0, 1, 3, 4, 2)
                if order == 0:
                    arr = arr[..., 0]
                elif order == 2:
                    arr = arr.reshape(count, frames, n, n, 2, 2)
                ds = group.create_dataset(name, data=arr)
                ds.attrs.update(sample_varying=True, time_varying=True, dim_varying=[True, True])
    cfg = {"data_root": "data", "run_dir": "run", "synthetic": True,
           "expected_counts": [4, 2, 2], "expected_shape": [n, n], "expected_frames": frames,
           "expected_length": [length, length], "latent_dim": 2, "widths": [2, 3, 4, 5],
           "warmup_epochs": 1, "branch_epochs": 3, "ramp_epochs": 1,
           "lambdas": [.01, .1, 1.], "batch_size": 8, "effective_batch": 16,
           "include_realizable": True, "realizable_lambdas": [.01, .1, 1.],
           "threads": 1, "forecast_epochs": 2, "forecast_width": 8, "pod_batch": 16,
           "ridge_alphas": [.1, 1.], "bootstrap_samples": 100}
    config_path = directory / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return config_path
