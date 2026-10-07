import json
from pathlib import Path
import h5py
import pytest
import torch
from active_matter_repr.synthetic import create_fixture
from active_matter_repr.config import load_config
from active_matter_repr.artifacts import file_hash
from active_matter_repr import pipeline
from amreview.run import main


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("complete")
    cfg = load_config(create_fixture(root / "fixture"))
    torch.set_num_threads(1)
    pipeline.audit_data(cfg)
    pipeline.train_representations(cfg)
    pipeline.analyze_representations(cfg)
    pipeline.train_forecasters(cfg)
    return cfg, root


def snapshot(root):
    return {
        str(p.relative_to(root)): file_hash(p)
        for p in Path(root).rglob("*")
        if p.is_file()
    }


def test_readonly_complete_resume_and_no_test_access(completed, monkeypatch):
    cfg, root = completed
    before = snapshot(cfg["run_dir"])
    original = h5py.File

    def guard(path, *args, **kwargs):
        if "test" in Path(path).parts:
            raise AssertionError("Test data accessed")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(h5py, "File", guard)
    out = root / "review"
    args = [
        "--run-dir",
        cfg["run_dir"],
        "--data-root",
        cfg["data_root"],
        "--out",
        str(out),
        "--device",
        "cpu",
        "--workers",
        "0",
        "--threads",
        "1",
        "--batch-size",
        "8",
        "--spatial-stride",
        "2",
    ]
    main(args)
    assert snapshot(cfg["run_dir"]) == before
    assert (out / "COMPLETE.json").exists()
    assert (out / "rollouts_linear.svg").exists() and (
        out / "spatial_reconstruction.png"
    ).exists()
    rows = [
        json.loads(line)
        for line in (out / "rollouts_1729_rpc.jsonl").read_text().splitlines()
    ]
    assert (
        len(rows) == 2 * 11 * 5
    )  # two trajectories, 11 leads, 2 variants each + persistence
    assert all(r["origin"] == 3 for r in rows)
    reconstruction = [
        json.loads(line)
        for line in (out / "reconstruction_1729_rpc.jsonl").read_text().splitlines()
    ]
    assert (
        len([r for r in reconstruction if r["kind"] == "reconstruction"]) == 2 * 15 * 2
    )
    assert all(r["frame"] < 15 for r in reconstruction)
    probes = json.loads((out / "probes_1729_rpc.json").read_text())
    assert set(probes["models"]) == {"parameters_time", "latent", "combined"}
    saved = file_hash(out / "rollouts_1729_rpc.jsonl")
    main(args)
    assert file_hash(out / "rollouts_1729_rpc.jsonl") == saved
    assert snapshot(cfg["run_dir"]) == before
    with pytest.raises(ValueError, match="different inputs"):
        main(args + ["--max-leads", "3"])


def test_reject_output_inside_source(completed):
    cfg, _ = completed
    with pytest.raises(ValueError, match="Output must be separate"):
        main(
            [
                "--run-dir",
                cfg["run_dir"],
                "--data-root",
                cfg["data_root"],
                "--out",
                str(Path(cfg["run_dir"]) / "posthoc"),
                "--device",
                "cpu",
                "--workers",
                "0",
            ]
        )


def test_tampered_source_config_rejected(completed, tmp_path):
    import shutil

    cfg, _ = completed
    destination = tmp_path / "copy"
    shutil.copytree(cfg["run_dir"], destination)
    config = json.loads((destination / "resolved_config.json").read_text())
    config["latent_dim"] = 999
    (destination / "resolved_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="configuration"):
        main(
            [
                "--run-dir",
                str(destination),
                "--data-root",
                cfg["data_root"],
                "--out",
                str(tmp_path / "out"),
                "--device",
                "cpu",
            ]
        )


def test_nonfinite_rollouts_retain_every_lead(completed):
    import io
    from amreview.provenance import SourceRun
    from amreview.run import rollouts, parser

    cfg, root = completed
    source = SourceRun(cfg["run_dir"], cfg["data_root"], "cpu")
    source.prepare([1729], ["rpc"], False)
    rep, meta, predictors = source.load(1729, "rpc")

    class Fails(torch.nn.Module):
        def forward(self, history, parameters):
            return torch.full_like(history[:, -1], float("inf"))

    predictors["linear"] = Fails()
    args = parser().parse_args(
        [
            "--run-dir",
            cfg["run_dir"],
            "--data-root",
            cfg["data_root"],
            "--out",
            str(root / "unused"),
            "--predictors",
            "linear",
            "--max-leads",
            "3",
        ]
    )
    stream = io.StringIO()
    rollouts(source, rep, meta, predictors, 1729, "rpc", source.cfg, args, stream)
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert len(records) == 2 * 3 * 2
    assert all(r["first_nonfinite_latent_lead"] == 1 for r in records)
    assert all(r["F"] is None and r["D_negative_fraction"] is None for r in records)


def test_future_truth_does_not_change_latent_forecast(completed, monkeypatch):
    import io
    from amreview.provenance import SourceRun
    from amreview.run import rollouts, parser
    from active_matter_repr.data import Snapshots

    cfg, root = completed
    source = SourceRun(cfg["run_dir"], cfg["data_root"], "cpu")
    source.prepare([1729], ["rpc"], False)
    rep, meta, predictors = source.load(1729, "rpc")
    args = parser().parse_args(
        [
            "--run-dir",
            cfg["run_dir"],
            "--data-root",
            cfg["data_root"],
            "--out",
            str(root / "unused"),
            "--predictors",
            "linear",
            "--max-leads",
            "3",
        ]
    )
    before = io.StringIO()
    rollouts(source, rep, meta, predictors, 1729, "rpc", source.cfg, args, before)
    original = Snapshots.frame

    def changed(self, row, t):
        x = original(self, row, t)
        if t >= cfg["history"]:
            x = x.clone()
            x[1:3] += 2
        return x

    monkeypatch.setattr(Snapshots, "frame", changed)
    after = io.StringIO()
    rollouts(source, rep, meta, predictors, 1729, "rpc", source.cfg, args, after)
    a = [json.loads(line) for line in before.getvalue().splitlines()]
    b = [json.loads(line) for line in after.getvalue().splitlines()]
    assert [r["max_abs_standardized_latent"] for r in a] == [
        r["max_abs_standardized_latent"] for r in b
    ]
    assert [r["raw_c_min"] for r in a] == [r["raw_c_min"] for r in b]
    assert any(x["F"] != y["F"] for x, y in zip(a, b))
