import hashlib
import io
from pathlib import Path
import pytest
from active_matter_repr import download as dl
from active_matter_repr.synthetic import create_fixture
from active_matter_repr.config import load_config

REVISION = "a" * 40
PAYLOAD = b"official-file-content"


def entry(split="train"):
    return {"path": f"{split}/sample.hdf5", "size": len(PAYLOAD), "sha256": hashlib.sha256(PAYLOAD).hexdigest()}


class Response(io.BytesIO):
    def __init__(self, value, status=200, headers=None):
        super().__init__(value)
        self.status, self.headers = status, headers or {}


def test_download_checksum_and_existing_reuse(tmp_path, monkeypatch):
    calls = []
    def fetch(req, timeout):
        calls.append(req.full_url)
        return Response(PAYLOAD)
    monkeypatch.setattr(dl, "urlopen", fetch)
    dl.download_file(entry(), REVISION, tmp_path)
    dl.download_file(entry(), REVISION, tmp_path)
    assert len(calls) == 1 and REVISION in calls[0]
    assert (tmp_path / entry()["path"]).read_bytes() == PAYLOAD


@pytest.mark.parametrize("supports_range", [True, False])
def test_partial_resume_and_range_ignored(tmp_path, monkeypatch, supports_range):
    target = tmp_path / entry()["path"]
    target.parent.mkdir()
    partial = target.with_name(target.name + "." + REVISION + ".part")
    partial.write_bytes(PAYLOAD[:5])
    def fetch(req, timeout):
        assert req.get_header("Range") == "bytes=5-"
        return Response(PAYLOAD[5:], 206, {"Content-Range": f"bytes 5-{len(PAYLOAD)-1}/{len(PAYLOAD)}"}) if supports_range else Response(PAYLOAD)
    monkeypatch.setattr(dl, "urlopen", fetch)
    dl.download_file(entry(), REVISION, tmp_path)
    assert target.read_bytes() == PAYLOAD and not partial.exists()


def test_interruption_retains_partial_then_resumes(tmp_path, monkeypatch):
    class Interrupted(Response):
        def read(self, n):
            if self.tell():
                raise TimeoutError("network interrupted")
            return super().read(5)
    monkeypatch.setattr(dl, "urlopen", lambda *a, **k: Interrupted(PAYLOAD))
    with pytest.raises(TimeoutError):
        dl.download_file(entry(), REVISION, tmp_path)
    target = tmp_path / entry()["path"]
    assert not target.exists()
    monkeypatch.setattr(dl, "urlopen", lambda *a, **k: Response(PAYLOAD[5:], 206, {"Content-Range": f"bytes 5-{len(PAYLOAD)-1}/{len(PAYLOAD)}"}))
    dl.download_file(entry(), REVISION, tmp_path)
    assert target.read_bytes() == PAYLOAD


def test_wrong_checksum_never_becomes_final(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "urlopen", lambda *a, **k: Response(b"x" * len(PAYLOAD)))
    with pytest.raises(ValueError, match="SHA256"):
        dl.download_file(entry(), REVISION, tmp_path)
    assert not (tmp_path / entry()["path"]).exists()


def test_existing_wrong_size_is_preserved(tmp_path):
    target = tmp_path / entry()["path"]
    target.parent.mkdir()
    target.write_bytes(b"user-data")
    with pytest.raises(ValueError, match="Existing file"):
        dl.download_file(entry(), REVISION, tmp_path)
    assert target.read_bytes() == b"user-data"


def test_complete_local_and_synthetic_are_offline(tmp_path, monkeypatch):
    cfg = load_config(create_fixture(tmp_path / "fixture"))
    def forbidden():
        raise AssertionError("Network should not be used")
    monkeypatch.setattr(dl, "official_catalog", forbidden)
    dl.ensure_data(cfg)
    cfg["synthetic"] = False
    dl.ensure_data(cfg)
    cfg["data_root"] = str(tmp_path / "missing")
    cfg["auto_download"] = False
    dl.ensure_data(cfg)


def test_missing_files_catalog_pinning(tmp_path, monkeypatch):
    catalog = {"repository": dl.REPOSITORY, "revision": REVISION, "files": [entry(s) for s in ("train", "valid", "test")]}
    monkeypatch.setattr(dl, "official_catalog", lambda: catalog)
    monkeypatch.setattr(dl, "local_complete", lambda cfg: False)
    monkeypatch.setattr(dl, "urlopen", lambda *a, **k: Response(PAYLOAD))
    cfg = {"data_root": str(tmp_path), "synthetic": False, "auto_download": True}
    dl.ensure_data(cfg)
    assert (tmp_path / "download_manifest.json").exists()
    monkeypatch.setattr(dl, "official_catalog", lambda: pytest.fail("Must reuse pinned catalog"))
    (tmp_path / "valid/sample.hdf5").unlink()
    dl.ensure_data(cfg)
    assert (tmp_path / "valid/sample.hdf5").read_bytes() == PAYLOAD


def test_catalog_rejects_unsafe_paths():
    catalog = {"repository": dl.REPOSITORY, "revision": REVISION, "files": [{**entry(), "path": "../outside.hdf5"}]}
    with pytest.raises(ValueError, match="path"):
        dl.validate_catalog(catalog)
