from pathlib import Path
import pandas as pd
import pytest


@pytest.fixture
def final_fixture(tmp_path: Path):
    root = tmp_path / "final"
    root.mkdir()
    dims = [128, 256, 512]
    seeds = [1729, 2718, 31415]
    methods = [
        ("ae", "legacy"),
        ("pc", "legacy"),
        ("rae", "direct_corrected"),
        ("rpc", "direct_corrected"),
        ("pod", "projected"),
    ]
    all_summary = []
    all_paired = []
    group = []
    for d in dims:
        dd = root / f"d{d}"
        dd.mkdir()
        rows = []
        leads = []
        probes = []
        decoder = []
        cov = []
        spatial = []
        for seed_i, seed in enumerate(seeds):
            for mi, (m, var) in enumerate(methods):
                qbase = {"ae": 0.40, "pc": 0.32, "rae": 0.27, "rpc": 0.20, "pod": 0.24}[
                    m
                ] * (256 / d) ** 0.4
                rbase = 0.10 * (256 / d) ** 0.3 * (1.0 + 0.01 * (m == "rpc"))
                rows.append(
                    dict(
                        kind="reconstruction",
                        seed=seed,
                        method=m,
                        predictor="none",
                        variant=var,
                        records=30,
                        trajectories=2,
                        R_rec=rbase * (1 + 0.01 * seed_i),
                        R_rec_invalid_records=0,
                        F=0.5,
                        F_invalid_records=0,
                        Q=qbase * (1 + 0.02 * seed_i),
                        Q_invalid_records=0,
                        velocity_intensity_mae=0.02,
                        enstrophy_mae=0.03,
                        order_mae=0.01,
                        c_nonpositive_fraction=0,
                        D_negative_fraction=0,
                        order_above_one_fraction=0,
                        D_trace_c_rmse=0,
                        divergence_rms=0.01,
                        strain_mismatch_rms=0.01,
                    )
                )
                for kind, steps in [("short_forecast", 10), ("rollout", 20)]:
                    for pred in ["linear", "mlp", "persistence"]:
                        f = (
                            0.6
                            + 0.08 * (pred == "persistence")
                            + 0.20 * (pred == "mlp")
                            + 0.02 * mi
                        )
                        q = (
                            0.5
                            - 0.25 * (pred == "persistence")
                            + 0.5 * (pred == "mlp")
                            + 0.03 * mi
                        )
                        rows.append(
                            dict(
                                kind=kind,
                                seed=seed,
                                method=m,
                                predictor=pred,
                                variant=var,
                                records=20,
                                trajectories=2,
                                R_rec="",
                                R_rec_invalid_records=0,
                                F=f,
                                F_invalid_records=0,
                                Q=q,
                                Q_invalid_records=0,
                                velocity_intensity_mae=0.02,
                                enstrophy_mae=0.03,
                                order_mae=0.01,
                                c_nonpositive_fraction=0,
                                D_negative_fraction=0,
                                order_above_one_fraction=0,
                                D_trace_c_rmse=0,
                                divergence_rms=0.01,
                                strain_mismatch_rms=0.01,
                            )
                        )
                        for lead in range(1, steps + 1):
                            leads.append(
                                dict(
                                    kind=kind,
                                    seed=seed,
                                    method=m,
                                    predictor=pred,
                                    variant=var,
                                    lead=lead,
                                    time_since_origin=0.25 * lead,
                                    records=2,
                                    F=f * (1 + 0.01 * lead),
                                    F_invalid_records=0,
                                    Q=q * (1 + 0.015 * lead),
                                    Q_invalid_records=0,
                                    c_near_floor_fraction=0,
                                    c_near_floor_fraction_invalid_records=0,
                                    order_above_one_fraction=0,
                                    order_above_one_fraction_invalid_records=0,
                                    D_negative_fraction=0,
                                    D_negative_fraction_invalid_records=0,
                                )
                            )
                probes.append(
                    dict(
                        seed=seed,
                        method=m,
                        combined=0.18 + 0.01 * mi,
                        latent=0.4 + 0.01 * mi,
                        parameters_time=0.30,
                        combined_gain=0.12 - 0.01 * mi,
                    )
                )
                decoder.append(
                    dict(
                        kind="reconstruction",
                        seed=seed,
                        method=m,
                        predictor="none",
                        variant=var,
                        records=30,
                        raw_c_min_minimum=0.8,
                        legacy_direct_max_abs_change_maximum=1e-8,
                        direct_high_precision_reference_max_abs_change_maximum=2e-9,
                        roundtrip_c_max_abs_change_maximum=0,
                    )
                )
                cov.append(
                    dict(
                        kind="reconstruction",
                        seed=seed,
                        method=m,
                        predictor="none",
                        variant=var,
                        trajectory_id=f"t{seed}",
                        records=30,
                        invalid_records=0,
                        first_invalid_lead="",
                        first_violation_lead="",
                    )
                )
                for coord in range(6):
                    spatial.append(
                        dict(
                            seed=seed,
                            method=m,
                            variant=var,
                            metric="energy_shell",
                            coordinate=coord,
                            value=0.1 / (1 + coord) + 0.001 * mi,
                            invalid_records=0,
                        )
                    )
        pd.DataFrame(rows).to_csv(dd / "summary.csv", index=False)
        pd.DataFrame(leads).to_csv(dd / "lead_summary.csv", index=False)
        pd.DataFrame(probes).to_csv(dd / "probe_summary.csv", index=False)
        pd.DataFrame(decoder).to_csv(dd / "decoder_diagnostics.csv", index=False)
        pd.DataFrame(cov).to_csv(dd / "coverage.csv", index=False)
        pd.DataFrame(spatial).to_csv(dd / "spatial_summary.csv", index=False)
        for r in rows:
            all_summary.append(dict(dimension=d, **r))
        for b in ["ae/legacy", "pc/legacy", "rae/direct_corrected", "pod/projected"]:
            all_paired.append(
                dict(
                    dimension=d,
                    kind="reconstruction",
                    predictor="none",
                    metric="Q",
                    a="rpc/direct_corrected",
                    b=b,
                    difference_b_minus_a=0.10,
                    ci95_low=0.05,
                    ci95_high=0.15,
                    groups=3,
                    seeds=3,
                )
            )
    pd.DataFrame(all_summary).to_csv(root / "comparison_summary.csv", index=False)
    pd.DataFrame(all_paired).to_csv(root / "paired_comparisons.csv", index=False)
    pd.DataFrame(group).to_csv(root / "group_metrics.csv", index=False)
    return root
