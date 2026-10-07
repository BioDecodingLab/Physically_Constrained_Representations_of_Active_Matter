#!/usr/bin/env python3
"""Stage orchestration for the fixed active-matter study protocol."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DIMENSIONS = (256, 128, 512)
RUN_NAMES = {128: "compression_d128", 256: "realizable_d256", 512: "compression_d512"}
STAGES = ("download", "audit", "representations", "analysis", "forecasters",
          "validation", "posthoc", "freeze", "test", "visuals", "publication")
EXECUTION_KEYS = {"id", "device", "workers", "threads", "worker_transport",
                  "run_dir", "data_root", "auto_download"}


def digest(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_repository():
    manifest = json.loads((ROOT / "provenance/source_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if not (ROOT / name).is_file() or file_hash(ROOT / name) != expected:
            raise ValueError(f"Reference source changed or missing: {name}")
    for d, expected in manifest["config_ids"].items():
        import yaml
        cfg = yaml.safe_load((ROOT / f"configs/paper/d{d}.yaml").read_text())
        if digest({k: v for k, v in cfg.items() if k not in EXECUTION_KEYS}) != expected:
            raise ValueError(f"Scientific configuration differs from the paper: d{d}")


def environment(component, threads=8):
    env = os.environ.copy()
    base = ROOT / "components" / component
    if component == "representation":
        paths = [base / "src"]
    elif component in ("posthoc", "final_evaluation"):
        paths = [base, base / "vendor"]
    else:
        paths = [base]
    # Isolate packages that share names between reference releases.
    env["PYTHONPATH"] = os.pathsep.join(map(str, paths))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["MPLBACKEND"] = "Agg"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = str(threads)
    return env


def runtime_config(dimension, args):
    import yaml
    cfg = yaml.safe_load((ROOT / f"configs/paper/d{dimension}.yaml").read_text())
    cfg.update(data_root=str(args.work_dir / "project/data/active_matter/data"),
               run_dir=str(args.work_dir / "project/runs" / RUN_NAMES[dimension]),
               device=args.device, workers=args.workers, threads=args.threads,
               auto_download=False)
    path = args.work_dir / f"configs/d{dimension}.yaml"
    if not args.dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(cfg, sort_keys=True))
    return path


def execute(component, command, args, label):
    command = [str(x) for x in command]
    print(f"[{label}] {shlex.join(command)}", flush=True)
    if args.dry_run:
        return
    args.work_dir.mkdir(parents=True, exist_ok=True)
    log = args.work_dir / "execution.jsonl"
    record = {"stage": label, "command": command,
              "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    with log.open("a") as stream:
        stream.write(json.dumps({**record, "status": "started"}) + "\n")
    result = subprocess.run(command, env=environment(component, args.threads), cwd=ROOT)
    with log.open("a") as stream:
        stream.write(json.dumps({**record, "status": "completed" if result.returncode == 0 else "failed",
                                 "returncode": result.returncode,
                                 "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat()}) + "\n")
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


def stage(name, args):
    python = sys.executable
    work = args.work_dir
    data = work / "project/data/active_matter/data"
    final = work / "final_test"
    lock = final / "protocol_lock.json"
    if name == "download":
        execute("representation", [python, ROOT / "scripts/download_data.py", "--out", data], args, name)
    elif name in ("audit", "representations", "analysis", "forecasters", "validation"):
        commands = {
            "audit": ["audit-data"], "representations": ["train-representations"],
            "analysis": ["analyze-representations"], "forecasters": ["train-forecasters"],
            "validation": ["evaluate", "diagnose-rollouts", "report", "lock"],
        }[name]
        for d in args.dimensions:
            config = runtime_config(d, args)
            for command in commands:
                cmd = [python, "-m", "active_matter_repr.cli", command, "--config", config]
                if command in ("evaluate", "diagnose-rollouts", "report"):
                    cmd += ["--split", "valid"]
                execute("representation", cmd, args, f"{name}/d{d}/{command}")
    elif name == "posthoc":
        for d in args.dimensions:
            execute("posthoc", [python, "-m", "amreview.run",
                    "--run-dir", work / "project/runs" / RUN_NAMES[d],
                    "--data-root", data, "--out", work / f"posthoc/d{d}",
                    "--device", args.device, "--workers", args.workers,
                    "--threads", args.threads, "--batch-size", "16",
                    "--frame-stride", "1", "--spatial-stride", "10",
                    "--verify-data-hashes"], args, f"posthoc/d{d}")
    elif name == "freeze":
        if lock.exists() and not args.dry_run:
            body = json.loads(lock.read_text())
            ident = body.pop("id")
            expected_runs = {str(work / "project/runs" / RUN_NAMES[d]) for d in DIMENSIONS}
            if (digest(body) != ident or body["synthetic"] or
                {r["run_dir"] for r in body["runs"]} != expected_runs or
                body["data_root"] != str(data)):
                raise ValueError("Existing protocol lock does not match this workspace")
            print(f"Reusing protocol lock: {lock}. Evaluation verifies locked source files.", flush=True)
        else:
            execute("final_evaluation", [python, "-m", "amreview.final", "freeze",
                    "--project", work / "project", "--posthoc", work / "posthoc",
                    "--out", final], args, name)
    elif name == "test":
        execute("final_evaluation", [python, "-m", "amreview.final", "evaluate",
                "--lock", lock, "--device", args.device, "--workers", args.workers,
                "--threads", args.threads], args, name)
    elif name == "visuals":
        execute("publication", [python, "-m", "ampub.export_visuals", "--lock", lock,
                "--evaluator", ROOT / "components/final_evaluation", "--out", work / "visual_export",
                "--dimension", "256", "--device", args.device], args, name)
    elif name == "publication":
        cmd = [python, "-m", "ampub", "--input", final, "--visuals", work / "visual_export",
               "--out", work / "publication"]
        if args.no_videos:
            cmd.append("--no-videos")
        execute("publication", cmd, args, name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(*STAGES, "all", "check", "tests"))
    parser.add_argument("--work-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--dimensions", type=int, nargs="+", choices=DIMENSIONS, default=list(DIMENSIONS))
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running stages")
    parser.add_argument("--no-videos", action="store_true", help="Omit video rendering")
    args = parser.parse_args(argv)
    args.work_dir = args.work_dir.expanduser().resolve()
    if args.workers < 0 or args.threads < 1 or len(args.dimensions) != len(set(args.dimensions)):
        parser.error("Require nonnegative workers, positive threads, and unique dimensions")
    if args.stage in ("all", "freeze", "test", "visuals", "publication") and set(args.dimensions) != set(DIMENSIONS):
        parser.error("The complete paper protocol requires dimensions 128, 256, and 512")
    if args.work_dir == ROOT or ROOT in args.work_dir.parents and args.work_dir.parts[len(ROOT.parts)] != "artifacts":
        parser.error("Keep generated work outside source directories; use artifacts/ or an external directory")
    if args.work_dir in ROOT.parents:
        parser.error("Work directory cannot contain the source repository")
    verify_repository()
    if args.stage == "check":
        execute("representation", [sys.executable, ROOT / "scripts/check_environment.py",
                "--device", args.device], args, "check")
    elif args.stage == "tests":
        for component in ("representation", "posthoc", "final_evaluation", "publication"):
            execute(component, [sys.executable, "-m", "pytest", "-q",
                    ROOT / "components" / component / "tests"], args, f"tests/{component}")
        execute("publication", [sys.executable, "-m", "pytest", "-q", ROOT / "tests"], args, "tests/orchestration")
    else:
        if args.stage == "all":
            execute("representation", [sys.executable, ROOT / "scripts/check_environment.py",
                    "--device", args.device], args, "check")
        for name in STAGES if args.stage == "all" else (args.stage,):
            stage(name, args)


if __name__ == "__main__":
    main()
