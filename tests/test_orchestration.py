"""Public workflow guarantees that do not require the full scientific dataset."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("study", ROOT / "scripts/study.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_reference_integrity_and_pinned_dataset():
    study.verify_repository()
    catalog = json.loads((ROOT / "provenance/dataset_catalog.json").read_text())
    hashes = json.loads((ROOT / "provenance/paper_data_sha256.json").read_text())
    assert len(catalog["files"]) == 82
    assert {e["path"]: e["sha256"] for e in catalog["files"]} == hashes
    assert sum(e["size"] for e in catalog["files"]) == 55809409024


def test_runtime_configuration_preserves_science(tmp_path):
    manifest = json.loads((ROOT / "provenance/source_manifest.json").read_text())
    args = SimpleNamespace(work_dir=tmp_path, device="cpu", workers=0, threads=1, dry_run=False)
    for dimension in study.DIMENSIONS:
        cfg = yaml.safe_load(study.runtime_config(dimension, args).read_text())
        body = {k: v for k, v in cfg.items() if k not in study.EXECUTION_KEYS}
        assert study.digest(body) == manifest["config_ids"][str(dimension)]
        assert Path(cfg["run_dir"]).name == study.RUN_NAMES[dimension]
        assert not cfg["auto_download"]


def test_full_plan_freezes_before_test_without_execution(tmp_path, capsys, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dry run executed a computation")
    monkeypatch.setattr(study.subprocess, "run", forbidden)
    study.main(["all", "--work-dir", str(tmp_path / "work"), "--dry-run", "--device", "cpu"])
    plan = capsys.readouterr().out
    assert plan.index("[check]") < plan.index("[download]")
    assert plan.index("[posthoc/d512]") < plan.index("[freeze]") < plan.index("[test]")
    assert "--split test" not in plan
    assert "amreview.final evaluate" in plan
    assert "--dimension 256" in plan
    assert not (tmp_path / "work").exists()


def test_incomplete_budget_is_not_a_paper_protocol(tmp_path):
    with pytest.raises(SystemExit):
        study.main(["freeze", "--dimensions", "256", "--work-dir", str(tmp_path / "work")])


def test_shared_package_names_are_isolated():
    p = study.environment("posthoc")["PYTHONPATH"]
    f = study.environment("final_evaluation")["PYTHONPATH"]
    assert "posthoc" in p and "final_evaluation" not in p
    assert "final_evaluation" in f and "posthoc" not in f
