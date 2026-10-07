from pathlib import Path
import h5py
import numpy as np
import pytest
import torch
import yaml
from active_matter_repr.synthetic import create_fixture
from active_matter_repr.config import load_config
from active_matter_repr.data import build_manifest, Snapshots, loader
from active_matter_repr.forecasting import Windows
from active_matter_repr.representations import fit_scaler, cache_latents
from active_matter_repr.models import Autoencoder
from active_matter_repr.transforms import FieldTransform
from active_matter_repr.artifacts import write_json, read_json, load_checkpoint, file_hash
from active_matter_repr import pipeline


@pytest.fixture
def config(tmp_path):
    torch.set_num_threads(1)
    return load_config(create_fixture(tmp_path / "fixture"))


def test_manifest_windows_and_worker_reads(config):
    manifest = build_manifest(config)
    ds = Snapshots(config, manifest, "train")
    assert len(ds) == 60
    assert ds[0]["x"].shape == (11, 16, 16)
    assert len(set(r["id"] for r in manifest["trajectories"])) == 8
    cache = {"z": np.arange(len(ds)*2).reshape(-1, 2), "trajectory": np.array([i for i,t in ds.index]), "frame": np.array([t for i,t in ds.index])}
    scaler = fit_scaler(cache["z"])
    params = fit_scaler(np.array([r["parameters"] for r in ds.rows]))
    windows = Windows(ds, cache, scaler, params, 4, 10)
    assert windows.origins == [(i, t) for i in range(4) for t in (3, 4)]
    sample = windows[0]
    torch.testing.assert_close(sample["truth"][0], ds.frame(ds.rows[0], 4))
    torch.testing.assert_close(sample["truth"][-1], ds.frame(ds.rows[0], 13))
    config["workers"] = 2
    config["worker_transport"] = "numpy"
    assert sum(len(b["x"]) for b in loader(ds, config)) == 60
    assert sum(len(b["truth"]) for b in loader(windows, config)) == 8
    bad = dict(cache)
    bad["frame"] = cache["frame"].copy()
    bad["frame"][0] = 1
    with pytest.raises(ValueError, match="cache"):
        Windows(ds, bad, scaler, params, 4, 10)
    ds.close()


def test_reject_axis_swap_and_wrong_mode(config):
    config["synthetic"] = False
    with pytest.raises(ValueError, match="Synthetic"):
        build_manifest(config)
    config["synthetic"] = True
    with h5py.File(Path(config["data_root"])/"train/fixture.h5", "r+") as h:
        h["dimensions"].attrs["spatial_dims"] = ["y", "x"]
    with pytest.raises(ValueError, match="axes"):
        build_manifest(config)


def test_shared_initial_conditions_and_duplicate_parameters(config):
    p = Path(config["data_root"])/"train/fixture.h5"
    with h5py.File(p, "r+") as h:
        for group in ("t0_fields", "t1_fields", "t2_fields"):
            for ds in h[group].values():
                ds[1, 0] = ds[0, 0]
    manifest = build_manifest(config)
    assert len(manifest["shared_initial_conditions"]) == 1
    with h5py.File(p, "r+") as h:
        h["scalars/alpha"][1] = h["scalars/alpha"][0]
        h["scalars/zeta"][1] = h["scalars/zeta"][0]
    with pytest.raises(ValueError, match="identical parameters"):
        build_manifest(config)


def test_config_unknown_key(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("learing_rate: 0.1\n")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(p)


def test_inclusive_coordinate_labels_use_physical_period(config):
    for split in ("train", "valid", "test"):
        with h5py.File(Path(config["data_root"])/split/"fixture.h5", "r+") as h:
            for name in ("x", "y"):
                h["dimensions/"+name][:] = np.linspace(0, config["expected_length"][0], 16)
    manifest = build_manifest(config)
    row = manifest["trajectories"][0]
    assert np.isclose(row["spacing"][0], config["expected_length"][0] / 16)
    assert np.isclose(row["coordinate_spacing"][0], config["expected_length"][0] / 15)
    assert "inclusive" in row["coordinate_convention"][0]


def test_interrupted_epoch_resume_matches_uninterrupted(config, tmp_path):
    from active_matter_repr.artifacts import seed_all
    from active_matter_repr.training import train_ae
    from active_matter_repr.evaluation import evaluate_reconstruction
    pipeline.audit_data(config)
    root, manifest, transform, obs = pipeline.context(config)
    train, valid = Snapshots(config, manifest, "train"), Snapshots(config, manifest, "valid")
    def evaluator(model):
        return evaluate_reconstruction(model, valid, transform, config, obs)
    seed_all(1729)
    full = pipeline.make_ae(config)
    full_path = train_ae(full, transform, train, valid, config, tmp_path/"full", 1729, 0., 3, evaluator, warmup=True)
    counter = [0]
    def interrupted(model):
        counter[0] += 1
        if counter[0] == 2:
            raise RuntimeError("simulated interruption")
        return evaluator(model)
    seed_all(1729)
    partial = pipeline.make_ae(config)
    with pytest.raises(RuntimeError, match="simulated"):
        train_ae(partial, transform, train, valid, config, tmp_path/"resumed", 1729, 0., 3, interrupted, warmup=True)
    resumed = pipeline.make_ae(config)
    resumed_path = train_ae(resumed, transform, train, valid, config, tmp_path/"resumed", 1729, 0., 3, evaluator, warmup=True)
    a, b = load_checkpoint(full_path), load_checkpoint(resumed_path)
    assert a["history"] == b["history"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    train.close()
    valid.close()


def test_cache_identity(config, tmp_path):
    manifest = build_manifest(config)
    ds = Snapshots(config, manifest, "train")
    model = Autoencoder((16,16), 2, (2,3,4,5))
    transform = FieldTransform(torch.zeros(11), torch.ones(4))
    cache_latents(model, ds, transform, config, tmp_path / "cache", "first")
    with pytest.raises(ValueError, match="Incompatible latent"):
        cache_latents(model, ds, transform, config, tmp_path / "cache", "second")
    ds.close()


def test_data_gate_and_stale_source_data(config):
    pipeline.audit_data(config)
    pipeline.context(config)
    with h5py.File(Path(config["data_root"])/"train/fixture.h5", "r+") as h:
        h.attrs["changed"] = 1
    with pytest.raises(ValueError, match="changed"):
        pipeline.context(config)


def test_full_synthetic_pipeline_and_resume(config):
    """Actual train/probe/forecast/test execution, not mocked model outputs."""
    from active_matter_repr.reporting import report
    pipeline.audit_data(config)
    pipeline.train_representations(config)
    pipeline.analyze_representations(config)
    pipeline.train_forecasters(config)
    pipeline.evaluate(config)
    pipeline.diagnose_rollouts(config)
    report(config)
    root = Path(config["run_dir"])
    result = read_json(root / "evaluation_valid.json")
    assert len(result["results"]) == 5
    assert {r["method"] for r in result["results"]} == {"ae", "pc", "rae", "rpc", "pod"}
    for item in result["results"]:
        if item["method"] in {"rae", "rpc"}:
            assert item["reconstruction"]["summary"]["D_negative_eigen_fraction"] == 0
            assert item["reconstruction"]["summary"]["local_order_above_one_fraction"] == 0
        if item["method"] in {"ae", "pc", "pod"}:
            assert item["projected_reconstruction"]["summary"]["D_negative_eigen_fraction"] == 0
    rollouts = read_json(root / "rollouts_valid.json")
    assert len(rollouts["results"][0]["records"]) == 2 * (15 - 4)
    assert all(r["forecasts"]["mlp"]["summary"]["F_10"] is not None for r in result["results"])
    # Finished-stage resume must preserve selected tensors and history length.
    checkpoint = root / "representations/1729/ae/best.pt"
    before = load_checkpoint(checkpoint)
    pipeline.train_representations(config)
    after = load_checkpoint(checkpoint)
    assert before["epoch"] == after["epoch"]
    for name in before["model"]:
        torch.testing.assert_close(before["model"][name], after["model"][name], rtol=0, atol=0)
    with pytest.raises(FileNotFoundError):
        pipeline.evaluate(config, "test")
    pipeline.lock_experiment(config)
    pipeline.evaluate(config, "test")
    report(config, "test")
    assert (root / "reports/test/summary.csv").exists()
    selection = read_json(root / "selection.json")
    selection["selected_lambda"] = 99
    write_json(root / "selection.json", selection)
    with pytest.raises(ValueError, match="stale"):
        pipeline.evaluate(config, "test")


def test_pilot_expansion_preserves_first_seed(tmp_path):
    path = create_fixture(tmp_path / "fixture")
    raw = yaml.safe_load(path.read_text())
    raw["seeds"] = [1729, 2718, 31415]
    path.write_text(yaml.safe_dump(raw))
    config = load_config(path)
    pipeline.audit_data(config)
    pipeline.train_representations(config, pilot_only=True)
    root = Path(config["run_dir"])
    first = file_hash(root / "representations/1729/ae/best.pt")
    pilot_selection = read_json(root / "selection.json")
    assert list(pilot_selection["seeds"]) == ["1729"]
    pipeline.train_representations(config)
    assert file_hash(root / "representations/1729/ae/best.pt") == first
    selection = read_json(root / "selection.json")
    assert set(selection["seeds"]) == {"1729", "2718", "31415"}
    assert selection["selected_lambda"] == pilot_selection["selected_lambda"]
    assert len(list((root / "representations").glob("*/pc_*/best.pt"))) == 5
    assert len(list((root / "representations").glob("*/rpc_*/best.pt"))) == 5
    assert len(list((root / "representations").glob("*/rae/best.pt"))) == 3
    assert selection["selected_realizable_lambda"] == pilot_selection["selected_realizable_lambda"]
