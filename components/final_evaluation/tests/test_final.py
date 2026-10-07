import json
from pathlib import Path
import h5py
import pytest
from sklearn.linear_model import Ridge
from test_end_to_end import completed, snapshot
from amreview.run import main as validation
from amreview.final import freeze, evaluate, frozen_probes, short_forecasts
from active_matter_repr.artifacts import file_hash


def test_frozen_test_end_to_end(completed, monkeypatch):
    cfg, root = completed
    project = root / "project"
    data = project / "data/active_matter"
    data.mkdir(parents=True)
    (data / "data").symlink_to(Path(cfg["data_root"]), target_is_directory=True)
    posthoc = root / "validation_for_final"
    validation(
        [
            "--run-dir",
            cfg["run_dir"],
            "--data-root",
            cfg["data_root"],
            "--out",
            str(posthoc),
            "--device",
            "cpu",
            "--workers",
            "0",
            "--threads",
            "1",
            "--batch-size",
            "16",
        ]
    )
    before = snapshot(cfg["run_dir"])
    original = h5py.File

    def no_test(path, *a, **kw):
        if "test" in Path(path).parts:
            raise AssertionError("Test opened during freeze")
        return original(path, *a, **kw)

    monkeypatch.setattr(h5py, "File", no_test)
    lock = freeze(
        project, posthoc, root / "final", synthetic_runs=[Path(cfg["run_dir"])]
    )
    monkeypatch.setattr(h5py, "File", original)

    def forbidden(*a, **kw):
        raise AssertionError("Fitting during final evaluation")

    monkeypatch.setattr(Ridge, "fit", forbidden)
    evaluate(lock, "cpu", 0, 1)
    out = lock.parent / f"d{cfg['latent_dim']}"
    assert snapshot(cfg["run_dir"]) == before
    assert (lock.parent / "COMPLETE.json").exists()
    assert (lock.parent / "paired_comparisons.csv").exists()
    rr = [
        json.loads(l)
        for l in (out / "short_forecast_1729_rpc.jsonl").read_text().splitlines()
    ]
    assert (
        len(rr) == 2 * 2 * 10 * 5
    )  # 2 trajectories, 2 origins, 10 leads, 5 variant/predictor pairs
    assert {r["origin"] for r in rr} == {3, 4}
    assert all(r["trajectory_id"].startswith("test/") for r in rr)
    assert all(r["F"] is not None for r in rr)
    probe = json.loads((out / "probes_1729_rpc.json").read_text())
    assert probe["split"] == "test"
    assert all(len(m["records"]) == 30 for m in probe["models"].values())
    h = file_hash(out / "short_forecast_1729_rpc.jsonl")
    evaluate(lock, "cpu", 0, 1)
    assert h == file_hash(out / "short_forecast_1729_rpc.jsonl")
    with pytest.raises(ValueError, match="already exists"):
        freeze(project, posthoc, root / "final", synthetic_runs=[Path(cfg["run_dir"])])
    content = lock.read_text()
    tampered = json.loads(content)
    tampered["protocol"]["batch_size"] = 1
    lock.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="protocol/code/environment"):
        evaluate(lock, "cpu", 0, 1)
    lock.write_text(content)
    source = Path(cfg["run_dir"]) / "forecasters/1729/rpc/linear.json"
    saved = source.read_bytes()
    source.write_bytes(saved + b" ")
    with pytest.raises(ValueError, match="Source artifacts changed"):
        evaluate(lock, "cpu", 0, 1)
    source.write_bytes(saved)
    testfile = Path(cfg["data_root"]) / "test/fixture.h5"
    saved = testfile.read_bytes()
    testfile.write_bytes(saved + b" ")
    with pytest.raises(ValueError, match="Test data hash mismatch"):
        evaluate(lock, "cpu", 0, 1)
    testfile.write_bytes(saved)


def test_short_forecast_no_future_feedback(completed, monkeypatch):
    import io
    from types import SimpleNamespace
    from active_matter_repr.data import Snapshots
    from amreview.provenance import SourceRun

    cfg, root = completed
    source = SourceRun(cfg["run_dir"], cfg["data_root"], "cpu")
    source.prepare([1729], ["rpc"], False)
    rep, meta, predictors = source.load(1729, "rpc")
    args = SimpleNamespace(batch_size=16, predictors=["linear"])
    a = io.StringIO()
    short_forecasts(source, rep, meta, predictors, 1729, "rpc", source.cfg, args, a)
    original = Snapshots.frame

    def changed(self, row, t):
        x = original(self, row, t)
        # t>=5 is beyond both observed histories in this fixture.
        if t >= 5:
            x = x.clone()
            x[1:3] += 2
        return x

    monkeypatch.setattr(Snapshots, "frame", changed)
    b = io.StringIO()
    short_forecasts(source, rep, meta, predictors, 1729, "rpc", source.cfg, args, b)
    aa = [json.loads(l) for l in a.getvalue().splitlines()]
    bb = [json.loads(l) for l in b.getvalue().splitlines()]
    assert [r["divergence_rms"] for r in aa] == [r["divergence_rms"] for r in bb]
    assert any(x["F"] != y["F"] for x, y in zip(aa, bb))


def test_short_score_matches_legacy_definition(completed):
    import io
    import numpy as np
    import torch
    from types import SimpleNamespace
    from active_matter_repr.data import Snapshots
    from active_matter_repr.forecasting import Windows, evaluate_forecast
    from amreview.provenance import SourceRun
    from amreview.reporting import aggregate

    cfg, _ = completed
    source = SourceRun(cfg["run_dir"], cfg["data_root"], "cpu")
    source.prepare([1729], ["rpc"], False)
    rep, meta, predictors = source.load(1729, "rpc")
    ds = Snapshots(source.cfg, source.manifest, "test")
    cache = {"z": [], "trajectory": [], "frame": []}
    with torch.no_grad():
        for i, t in ds.index:
            cache["z"].append(
                rep.encode(
                    source.transform.normalize(ds.frame(ds.rows[i], t)[None])
                ).numpy()[0]
            )
            cache["trajectory"].append(i)
            cache["frame"].append(t)
    cache = {k: np.array(v) for k, v in cache.items()}
    windows = Windows(ds, cache, meta["latent_scaler"], meta["parameter_scaler"], 4, 10)
    legacy = evaluate_forecast(
        predictors["linear"],
        rep,
        windows,
        meta["latent_scaler"],
        source.transform,
        {**source.cfg, "workers": 0},
    )["summary"]["F_10"]
    stream = io.StringIO()
    short_forecasts(
        source,
        rep,
        meta,
        predictors,
        1729,
        "rpc",
        source.cfg,
        SimpleNamespace(batch_size=16, predictors=["linear"]),
        stream,
    )
    rr = [json.loads(l) for l in stream.getvalue().splitlines()]
    rr = [r for r in rr if r["variant"] == "legacy"]
    assert aggregate(rr, "F")[0] == pytest.approx(legacy, abs=1e-6)

    class Fails(torch.nn.Module):
        def forward(self, h, p):
            return torch.full_like(h[:, -1], float("nan"))

    stream = io.StringIO()
    short_forecasts(
        source,
        rep,
        meta,
        {"linear": Fails()},
        1729,
        "rpc",
        source.cfg,
        SimpleNamespace(batch_size=16, predictors=["linear"]),
        stream,
    )
    rr = [json.loads(l) for l in stream.getvalue().splitlines()]
    assert len(rr) == 80
    assert all(r["F"] is None for r in rr)
    ds.close()
