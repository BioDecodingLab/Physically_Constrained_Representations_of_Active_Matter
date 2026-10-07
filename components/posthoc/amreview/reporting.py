"""Coverage-aware hierarchical summaries. No failed samples silently discarded."""

import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from active_matter_repr.artifacts import write_json

METRICS = (
    "R_rec",
    "F",
    "Q",
    "velocity_intensity_mae",
    "enstrophy_mae",
    "order_mae",
    "c_nonpositive_fraction",
    "c_near_floor_fraction",
    "D_negative_fraction",
    "order_above_one_fraction",
    "D_trace_c_rmse",
    "divergence_rms",
    "strain_mismatch_rms",
)


def aggregate(records, key):
    trajectories = defaultdict(list)
    groups = {}
    for r in records:
        trajectories[r["trajectory_id"]].append(r.get(key))
        groups[r["trajectory_id"]] = r["group"]
    invalid = sum(
        v is None or not np.isfinite(v)
        for values in trajectories.values()
        for v in values
    )
    if invalid:
        return None, invalid
    grouped = defaultdict(list)
    for trajectory, values in trajectories.items():
        grouped[groups[trajectory]].append(float(np.mean(values)))
    return float(np.mean([np.mean(v) for v in grouped.values()])), 0


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_reports(out):
    out = Path(out)
    summaries, lead_summaries, coverage, spatial_rows, probe_rows, diagnostic_rows = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for path in sorted(out.glob("*.jsonl")):
        records = [json.loads(line) for line in path.open() if line.strip()]
        buckets = defaultdict(list)
        for r in records:
            if r["kind"] == "spatial":
                continue
            key = (
                r["kind"],
                r["seed"],
                r["method"],
                r.get("predictor", "none"),
                r["variant"],
            )
            buckets[key].append(r)
        for (kind, seed, method, predictor, variant), rows in buckets.items():
            tag = dict(
                kind=kind,
                seed=seed,
                method=method,
                predictor=predictor,
                variant=variant,
            )
            summary = {
                **tag,
                "records": len(rows),
                "trajectories": len({r["trajectory_id"] for r in rows}),
            }
            for metric in METRICS:
                summary[metric], summary[metric + "_invalid_records"] = aggregate(
                    rows, metric
                )
            summaries.append(summary)
            if variant == "legacy" and any("raw_c_min" in r for r in rows):
                diagnostic = {**tag, "records": len(rows)}
                for metric in (
                    "raw_c_min",
                    "before_roundtrip_c_min",
                    "after_roundtrip_c_min",
                    "roundtrip_c_max_abs_change",
                    "legacy_direct_max_abs_change",
                    "direct_high_precision_reference_max_abs_change",
                ):
                    vals = [r.get(metric) for r in rows]
                    finite = [v for v in vals if v is not None and np.isfinite(v)]
                    diagnostic[metric + "_minimum"] = min(finite, default=None)
                    diagnostic[metric + "_maximum"] = max(finite, default=None)
                    diagnostic[metric + "_invalid_records"] = len(vals) - len(finite)
                diagnostic_rows.append(diagnostic)
            by_traj = defaultdict(list)
            for r in rows:
                by_traj[r["trajectory_id"]].append(r)
            for tid, obs in by_traj.items():
                bad = [r for r in obs if r["status"] != "ok"]
                violations = [
                    r
                    for r in obs
                    if (r.get("order_above_one_fraction") or 0) > 0
                    or (r.get("D_negative_fraction") or 0) > 0
                ]
                coverage.append(
                    {
                        **tag,
                        "trajectory_id": tid,
                        "records": len(obs),
                        "invalid_records": len(bad),
                        "first_invalid_lead": min(
                            (r.get("lead", r.get("frame", 0)) for r in bad),
                            default=None,
                        ),
                        "first_violation_lead": min(
                            (r.get("lead", r.get("frame", 0)) for r in violations),
                            default=None,
                        ),
                    }
                )
            if kind == "rollout":
                by_lead = defaultdict(list)
                for r in rows:
                    by_lead[r["lead"]].append(r)
                for lead, obs in sorted(by_lead.items()):
                    s = {
                        **tag,
                        "lead": lead,
                        "time_since_origin": obs[0]["time_since_origin"],
                        "records": len(obs),
                    }
                    for metric in (
                        "F",
                        "Q",
                        "c_near_floor_fraction",
                        "order_above_one_fraction",
                        "D_negative_fraction",
                    ):
                        s[metric], s[metric + "_invalid_records"] = aggregate(
                            obs, metric
                        )
                    lead_summaries.append(s)
        # Spatial curves: frame -> trajectory -> parameter group, at each bin.
        spatial = defaultdict(list)
        for r in records:
            if r["kind"] == "spatial":
                spatial[(r["seed"], r["method"], r["variant"])].append(r)
        for (seed, method, variant), obs in spatial.items():
            for metric, axis in [("energy_shell", "k"), ("correlation", "r")]:
                template = next((r for r in obs if r.get(metric) is not None), None)
                if template is None:
                    spatial_rows.append(
                        dict(
                            seed=seed,
                            method=method,
                            variant=variant,
                            metric=metric,
                            coordinate=None,
                            value=None,
                            invalid_records=len(obs),
                        )
                    )
                    continue
                for j, coord in enumerate(template[axis]):
                    vals = [
                        {
                            **r,
                            "value": r[metric][j]
                            if r.get(metric) is not None
                            else None,
                        }
                        for r in obs
                    ]
                    value, invalid = aggregate(vals, "value")
                    spatial_rows.append(
                        dict(
                            seed=seed,
                            method=method,
                            variant=variant,
                            metric=metric,
                            coordinate=coord,
                            value=value,
                            invalid_records=invalid,
                        )
                    )
    for path in sorted(out.glob("probes_*.json")):
        if path.name.endswith(".done.json"):
            continue
        d = json.loads(path.read_text())
        probe_rows.append(
            {
                "seed": d["seed"],
                "method": d["method"],
                **{k: v["Q"] for k, v in d["models"].items()},
                "combined_gain": d["Q_gain_combined_vs_parameters_time"],
            }
        )
    for name, rows in [
        ("summary", summaries),
        ("lead_summary", lead_summaries),
        ("coverage", coverage),
        ("spatial_summary", spatial_rows),
        ("probe_summary", probe_rows),
        ("decoder_diagnostics", diagnostic_rows),
    ]:
        write_csv(out / f"{name}.csv", rows)
    write_json(
        out / "report_notes.json",
        {
            "aggregation": "Equal frames per trajectory, trajectories per parameter group, groups per seed. Seeds are reported individually.",
            "failure_rule": "A primary mean is null if any contributing record is undefined. See coverage.csv.",
            "rollout_rule": "One first-history origin; aggregate F here is NOT the original all-origin F_10.",
            "POD": "The POD basis and linear predictor are shared/deterministic across seed labels. Do not count these as independent fits.",
        },
    )
    plots(out, lead_summaries, spatial_rows)


def plots(out, leads, spatial):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = json.loads((out / "analysis_manifest.json").read_text())
    label = "SYNTHETIC SOFTWARE TEST — " if manifest["synthetic"] else ""
    if manifest["scope"] == "diagnostic subset":
        label += "DIAGNOSTIC SUBSET — "
    for predictor in ("linear", "mlp", "persistence"):
        obs = [r for r in leads if r["predictor"] == predictor]
        if not obs:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
        for metric, ax in zip(("F", "Q", "order_above_one_fraction"), axes):
            groups = defaultdict(list)
            for r in obs:
                groups[(r["seed"], r["method"], r["variant"])].append(r)
            for (seed, method, variant), rows in groups.items():
                rows = sorted(rows, key=lambda r: r["lead"])
                ax.plot(
                    [r["lead"] for r in rows],
                    [np.nan if r[metric] is None else r[metric] for r in rows],
                    lw=1,
                    alpha=0.8,
                    label=f"{method}/{variant}/{seed}",
                )
            ax.set(xlabel="Lead (stored steps)", ylabel=metric)
            if metric in ("F", "Q"):
                ax.set_yscale("symlog", linthresh=0.01)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=6)
        fig.suptitle(
            label
            + f"Validation first-origin rollouts: {predictor}; gaps = undefined metrics",
            fontsize=10,
        )
        for ext in ("png", "svg"):
            fig.savefig(out / f"rollouts_{predictor}.{ext}", dpi=180)
        plt.close(fig)
    if spatial:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        for metric, ax in zip(("energy_shell", "correlation"), axes):
            groups = defaultdict(list)
            for r in spatial:
                if r["metric"] == metric and r["coordinate"] is not None:
                    groups[(r["seed"], r["method"], r["variant"])].append(r)
            for key, rows in groups.items():
                ax.plot(
                    [r["coordinate"] for r in rows],
                    [np.nan if r["value"] is None else r["value"] for r in rows],
                    label="/".join(map(str, key)),
                    alpha=0.7,
                )
            ax.set(
                xlabel="Wavenumber k" if metric == "energy_shell" else "Separation r",
                ylabel=metric,
            )
            if metric == "energy_shell":
                ax.set_yscale("log")
        fig.suptitle(
            label + "Validation reconstruction: spatial diagnostics", fontsize=10
        )
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=6)
        for ext in ("png", "svg"):
            fig.savefig(out / f"spatial_reconstruction.{ext}", dpi=180)
        plt.close(fig)
