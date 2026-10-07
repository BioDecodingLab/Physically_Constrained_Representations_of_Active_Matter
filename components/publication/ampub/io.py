from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import shutil
import tarfile
import tempfile


def _safe_extract(tf: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in tf.getmembers():
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"Unsupported archive member type: {member.name}")
        target = (destination / member.name).resolve()
        try:
            target.relative_to(destination)
        except ValueError as exc:
            raise ValueError(f"Unsafe archive member: {member.name}") from exc
    try:
        tf.extractall(destination, filter="data")
    except (
        TypeError
    ):  # Python 3.10/3.11 compatibility after explicit path validation above.
        tf.extractall(destination)


def locate_final_root(path: Path) -> Path:
    """Locate a frozen-final result root or a single per-dimension result directory."""
    path = path.resolve()
    candidates = [path]
    candidates.extend(p for p in path.rglob("*") if p.is_dir())
    # Prefer full roots with dimension subdirectories and then per-dimension roots.
    for p in candidates:
        if any((p / f"d{d}" / "summary.csv").exists() for d in (128, 256, 512)):
            return p
    for p in candidates:
        if (p / "summary.csv").exists() and (p / "lead_summary.csv").exists():
            return p
    raise FileNotFoundError(f"Could not locate final-evaluation outputs under {path}")


@contextmanager
def materialized_input(source: str | Path):
    """Yield an extracted result directory from a tar.gz/tgz or an existing directory."""
    source = Path(source).expanduser().resolve()
    if source.is_dir():
        yield locate_final_root(source)
        return
    if not source.is_file():
        raise FileNotFoundError(source)
    suffixes = "".join(source.suffixes[-2:])
    if suffixes not in {".tar.gz", ".tgz"} and source.suffix != ".tar":
        raise ValueError(
            "Input must be a final-evaluation directory or .tar.gz/.tgz/.tar archive"
        )
    with tempfile.TemporaryDirectory(prefix="active_matter_publication_") as td:
        dest = Path(td)
        with tarfile.open(source, "r:*") as tf:
            _safe_extract(tf, dest)
        yield locate_final_root(dest)


def copy_source_metadata(root: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in (
        "COMPLETE.json",
        "analysis_manifest.json",
        "protocol_lock.json",
        "INTERPRETATION.txt",
    ):
        src = root / name
        if src.exists():
            shutil.copy2(src, out / name)
