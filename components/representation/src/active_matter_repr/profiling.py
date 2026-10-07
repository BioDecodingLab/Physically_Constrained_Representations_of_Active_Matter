"""A bounded forward/backward benchmark; never a scientific training result."""
import time
from pathlib import Path
import torch
from .models import Autoencoder
from .physics import residuals
from .artifacts import write_json, seed_all


def profile(cfg, batch_size=1):
    if batch_size < 1:
        raise ValueError("Positive profile batch size required")
    seed_all(cfg["seeds"][0])
    torch.set_num_threads(cfg["threads"])
    device = cfg["device"]
    model = Autoencoder(cfg["expected_shape"], cfg["latent_dim"], cfg["widths"]).to(device)
    x = torch.randn(batch_size, 11, *cfg["expected_shape"], device=device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    y = model(x)
    spacing = [l/n for l,n in zip(cfg["expected_length"], cfg["expected_shape"])]
    loss = (y-x).square().mean() + .1*residuals(y, torch.ones(4, device=device), spacing, cfg["derivative"]).mean()
    loss.backward()
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    output = {"kind": "synthetic forward/backward cost check", "batch_size": batch_size,
              "parameters": sum(p.numel() for p in model.parameters()), "seconds": time.perf_counter()-start,
              "finite_gradients": all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else None,
              "note": "One first iteration; excludes optimizer state, data loading and validation; not an epoch-time estimate"}
    write_json(Path(cfg["run_dir"]) / "profile.json", output)
    print(output)
    if not output["finite_gradients"]:
        raise FloatingPointError("Profile produced invalid gradients")
