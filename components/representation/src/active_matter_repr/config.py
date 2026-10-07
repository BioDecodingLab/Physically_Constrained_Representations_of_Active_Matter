"""Strict configuration; paths are relative to the config file."""
from pathlib import Path
import copy
import math
import yaml
from .artifacts import digest

DEFAULTS = {
    "data_root": "data/active_matter", "run_dir": "runs/pilot", "synthetic": False,
    "auto_download": True,
    "fields": {"c": "concentration", "v": "velocity", "D": "D", "E": "E"},
    "expected_counts": [175, 24, 26], "expected_shape": [256, 256],
    "expected_frames": 81, "expected_length": [10., 10.], "expected_dt": 0.25,
    "residual_rms_max": [0.001, 0.001, 0.05, 0.05], "trace_rmse_max": 0.01,
    "derivative": "spectral", "seeds": [1729], "latent_dim": 256,
    "widths": [32, 64, 128, 256], "warmup_epochs": 20, "branch_epochs": 60,
    "ramp_epochs": 5, "lambdas": [0.01, 0.1, 1.], "batch_size": 8,
    "include_realizable": False, "realizable_lambdas": [0.001, 0.01, 0.1],
    "reconstruction_guard": 1.05,
    "effective_batch": 32, "workers": 0, "threads": 4, "device": "cpu", "worker_transport": "torch",
    "learning_rate": 1e-4, "weight_decay": 1e-8, "grad_clip": 1.,
    "history": 4, "horizon": 10, "forecast_epochs": 60, "forecast_width": 256,
    "ridge_alphas": [0.001, 0.01, 0.1, 1., 10., 100.],
    "pod_batch": 512, "bootstrap_samples": 2000,
}


def load_config(path):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise ValueError(f"Unknown/invalid configuration keys: {set(raw or {}) - set(DEFAULTS)}")
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(raw)
    if type(cfg["auto_download"]) is not bool:
        raise ValueError("auto_download must be true or false")
    if type(cfg["include_realizable"]) is not bool:
        raise ValueError("include_realizable must be true or false")
    if set(cfg["fields"]) != {"c", "v", "D", "E"}:
        raise ValueError("fields must map c, v, D, E explicitly")
    if any(not isinstance(v, str) or not v for v in cfg["fields"].values()):
        raise ValueError("Field names must be nonempty strings")
    for key in ("latent_dim", "warmup_epochs", "branch_epochs", "ramp_epochs", "batch_size", "effective_batch", "threads", "history", "horizon", "forecast_epochs", "pod_batch", "bootstrap_samples"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["branch_epochs"] <= cfg["ramp_epochs"]:
        raise ValueError("Need eligible epochs after ramp")
    if cfg["effective_batch"] % cfg["batch_size"]:
        raise ValueError("effective_batch must be divisible by batch_size")
    if type(cfg["workers"]) is not int or cfg["workers"] < 0 or cfg["history"] + cfg["horizon"] > cfg["expected_frames"]:
        raise ValueError("Invalid workers/history/horizon")
    if cfg["worker_transport"] not in ("torch", "numpy"):
        raise ValueError("worker_transport must be torch or numpy")
    if not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("Unique seeds and positive physics weights required")
    if any(type(s) is not int or not 0 <= s < 2**32 for s in cfg["seeds"]):
        raise ValueError("Seeds must be integers in [0, 2**32)")
    for key in ("learning_rate", "grad_clip", "expected_dt", "trace_rmse_max", "reconstruction_guard"):
        cfg[key] = float(cfg[key])
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if cfg["reconstruction_guard"] < 1:
        raise ValueError("reconstruction_guard must be at least one")
    cfg["weight_decay"] = float(cfg["weight_decay"])
    if not math.isfinite(cfg["weight_decay"]) or cfg["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if len(cfg["expected_counts"]) != 3 or any(type(n) is not int or n < 1 for n in cfg["expected_counts"]):
        raise ValueError("Three positive integer split counts required")
    if len(cfg["expected_shape"]) != 2 or any(type(n) is not int or n < 16 or n % 16 for n in cfg["expected_shape"]):
        raise ValueError("Two spatial sizes divisible by 16 required")
    if len(cfg["expected_length"]) != 2 or any(not math.isfinite(n) or n <= 0 for n in cfg["expected_length"]):
        raise ValueError("Positive finite physical periods required")
    for key in ("lambdas", "realizable_lambdas", "ridge_alphas", "residual_rms_max"):
        if not cfg[key] or any(not math.isfinite(v) or v <= 0 for v in cfg[key]):
            raise ValueError(f"{key} must contain finite positive values")
    if len(cfg["residual_rms_max"]) != 4:
        raise ValueError("Four physical residual thresholds required")
    if len(cfg["widths"]) != 4 or any(x <= 0 for x in cfg["widths"]) or cfg["derivative"] not in ("spectral", "finite_difference"):
        raise ValueError("Invalid architecture/operator")
    for key in ("data_root", "run_dir"):
        cfg[key] = str((path.parent / cfg[key]).resolve())
    # Execution-only resources may change on resume; scientific configuration may not.
    cfg["id"] = digest({k: v for k, v in cfg.items() if k not in {"device", "workers", "threads", "worker_transport", "run_dir", "data_root", "auto_download"}})
    return cfg
