#!/usr/bin/env python3
"""Report installed numerical libraries and enforce the execution backend."""
import argparse
import importlib.metadata as metadata
import json
import platform
import sys
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    if not torch.__version__.split("+")[0] == "2.6.0":
        raise RuntimeError("This release requires PyTorch 2.6.0")
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; install the CUDA environment or select cpu")
    if a.device.startswith("cuda"):
        torch.empty(1, device=a.device)
    versions = {name: metadata.version(name) for name in
                ("torch", "numpy", "scipy", "scikit-learn", "h5py", "PyYAML", "matplotlib",
                 "pandas", "imageio", "imageio-ffmpeg", "tabulate", "tqdm", "pytest",
                 "active-matter-repr", "active-matter-publication")}
    print(json.dumps({"python": sys.version, "platform": platform.platform(), "device": a.device,
                      "cuda_runtime": torch.version.cuda, "dependencies": versions}, indent=2))


if __name__ == "__main__":
    main()
