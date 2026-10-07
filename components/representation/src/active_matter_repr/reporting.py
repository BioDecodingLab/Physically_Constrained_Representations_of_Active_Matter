"""Diagnostic figures, exact tables, and matched statistical comparisons."""
from collections import defaultdict
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .artifacts import read_json, write_json
from .statistics import paired_cluster_bootstrap


def _groups(records, key, rows):
    trajectories = defaultdict(list)
    for record in records:
        if record[key] is None:
            return None
        trajectories[record["trajectory"]].append(record[key])
    if set(trajectories) != set(range(len(rows))):
        raise ValueError("Incomplete reporting coverage")
    groups = defaultdict(list)
    for tid, values in trajectories.items():
        groups[rows[tid]["group"]].append(np.mean(values))
    return {k: float(np.mean(v)) for k, v in groups.items()}


def report(cfg, split="valid"):
    root = Path(cfg["run_dir"])
    data = read_json(root / f"evaluation_{split}.json")
    if data["config_id"] != cfg["id"]:
        raise ValueError("Report configuration mismatch")
    directory = root / "reports" / split
    directory.mkdir(parents=True, exist_ok=True)
    summary, detail, grouped, sensitivity = [], [], {}, []
    fkey = f"F_{cfg['horizon']}"
    for item in data["results"]:
        seed, method = item["seed"], item["method"]
        summary.append({"seed": seed, "method": method, **item["reconstruction"]["summary"],
                        "Q_probe": item["probes"]["latent"]["Q_probe"],
                        "Q_nuisance": item["probes"]["parameters_time"]["Q_probe"],
                        fkey: item["forecasts"]["mlp"]["summary"][fkey],
                        "forecast_D_negative_eigen_fraction": item["forecasts"]["mlp"]["summary"]["D_negative_eigen_fraction"],
                        "forecast_D_trace_c_rmse": item["forecasts"]["mlp"]["summary"]["D_trace_c_rmse"],
                        "failed_forecast_leads": item["forecasts"]["mlp"]["summary"]["failed_lead_records"]})
        sources = [("R_rec", item["reconstruction"]["records"], "R_rec"),
                   ("Q_rec", item["reconstruction"]["records"], "Q_rec"),
                   ("Q_probe", item["probes"]["latent"]["records"], "Q_probe"),
                   (fkey, item["forecasts"]["mlp"]["records"], "error")]
        for name, records, key in sources:
            grouped[method, seed, name] = _groups(records, key, data["trajectories"])
            values_by_trajectory = defaultdict(list)
            for record in records:
                values_by_trajectory[record["trajectory"]].append(record[key])
            valid = grouped[method, seed, name] is not None
            sensitivity.append({"seed": seed, "method": method, "metric": name,
                "equal_group_mean": float(np.mean(list(grouped[method, seed, name].values()))) if valid else None,
                "equal_trajectory_mean": float(np.mean([np.mean(v) for v in values_by_trajectory.values()])) if valid else None,
                "status": "complete" if valid else "incomplete"})
            for r in records:
                detail.append({"seed": seed, "method": method, "metric": name, "trajectory_id": data["trajectories"][r["trajectory"]]["id"],
                               "group": data["trajectories"][r["trajectory"]]["group"], "frame_or_origin": r.get("frame", r.get("origin")),
                               "lead": r.get("lead"), "value": r[key]})
    for filename, rows in (("summary.csv", summary), ("measurements.csv", detail)):
        with open(directory / filename, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    comparisons = {}
    completed_seeds = sorted(set(item["seed"] for item in data["results"]))
    pairs = [("pc_vs_ae", "ae", "pc")]
    if cfg["include_realizable"]:
        pairs += [("rae_vs_ae", "ae", "rae"), ("rpc_vs_rae", "rae", "rpc"),
                  ("rpc_vs_pc", "pc", "rpc")]
    for label, baseline, candidate in pairs:
        comparisons[label] = {}
        for metric in ("R_rec", "Q_rec", "Q_probe", fkey):
            a = [grouped[baseline, s, metric] for s in completed_seeds]
            b = [grouped[candidate, s, metric] for s in completed_seeds]
            if any(x is None for x in a + b):
                comparisons[label][metric] = {"status": "incomplete; no survivor-only comparison"}
            else:
                keys = sorted(a[0])
                if any(sorted(x) != keys for x in a + b):
                    raise ValueError("Mismatched parameter groups")
                comparisons[label][metric] = paired_cluster_bootstrap(
                    [[x[k] for k in keys] for x in a], [[x[k] for k in keys] for x in b],
                    cfg["bootstrap_samples"])
    write_json(directory / "paired_statistics.json", comparisons)
    write_json(directory / "aggregation_sensitivity.json", sensitivity)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), layout="constrained")
    methods = ("pod", "ae", "pc", "rae", "rpc") if cfg["include_realizable"] else ("pod", "ae", "pc")
    labels = {"pod": "POD", "ae": "AE", "pc": "PC-AE", "rae": "R-AE", "rpc": "R-PC-AE"}
    for ax, metric in zip(axes, ("Q_rec", "Q_probe", fkey)):
        for i, method in enumerate(methods):
            vals = [row[metric] for row in summary if row["method"] == method and row[metric] is not None]
            ax.scatter([i] * len(vals), vals, s=30)
        ax.set_xticks(range(len(methods)), [labels[m] for m in methods], rotation=20 if len(methods) > 3 else 0)
        ax.set_ylabel(metric)
        ax.spines[["top", "right"]].set_visible(False)
    title = "SYNTHETIC SOFTWARE TEST — NOT SCIENTIFIC RESULTS" if cfg["synthetic"] else f"{split}: individual trained seeds"
    fig.suptitle(title, fontsize=10)
    fig.savefig(directory / "diagnostic_comparison.svg")
    fig.savefig(directory / "diagnostic_comparison.png", dpi=180)
    plt.close(fig)
