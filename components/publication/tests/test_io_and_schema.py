from pathlib import Path
import tarfile
from ampub.io import materialized_input
from ampub.schema import load_final


def test_directory_load(final_fixture):
    d = load_final(final_fixture)
    assert d.dimensions == [128, 256, 512]
    assert set(d.summary.method.unique()) >= {"ae", "rpc", "pod"}


def test_tar_load(final_fixture, tmp_path):
    arc = tmp_path / "final.tar.gz"
    with tarfile.open(arc, "w:gz") as tf:
        for p in final_fixture.rglob("*"):
            tf.add(p, arcname=p.relative_to(final_fixture))
    with materialized_input(arc) as root:
        d = load_final(root)
        assert 256 in d.dimensions


def test_archive_traversal_rejected(tmp_path):
    import io
    from ampub.io import _safe_extract
    import pytest

    arc = tmp_path / "unsafe.tar"
    with tarfile.open(arc, "w") as tf:
        info = tarfile.TarInfo("../escape")
        info.size = 1
        tf.addfile(info, io.BytesIO(b"x"))
    with tarfile.open(arc) as tf:
        with pytest.raises(ValueError):
            _safe_extract(tf, tmp_path / "dest")


def test_archive_link_rejected(tmp_path):
    from ampub.io import _safe_extract
    import pytest

    arc = tmp_path / "unsafe.tar"
    with tarfile.open(arc, "w") as tf:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/tmp"
        tf.addfile(info)
    with tarfile.open(arc) as tf:
        with pytest.raises(ValueError):
            _safe_extract(tf, tmp_path / "dest")
