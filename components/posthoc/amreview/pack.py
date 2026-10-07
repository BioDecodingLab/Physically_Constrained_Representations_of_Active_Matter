"""Create a review archive only for a complete, unchanged analysis."""

import argparse
import fcntl
from pathlib import Path
import tarfile
from active_matter_repr.artifacts import read_json, file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("archive", type=Path)
    a = p.parse_args()
    out = a.out.resolve()
    archive = a.archive.resolve()
    if archive.is_relative_to(out):
        raise ValueError("Place the archive outside the analysis directory")
    with open(out / ".analysis.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Analysis is still running") from None
        complete = read_json(out / "COMPLETE.json")
        manifest = read_json(out / "analysis_manifest.json")
        if complete["id"] != manifest["id"]:
            raise ValueError("Completion/manifest mismatch")
        for marker in out.glob("*.done.json"):
            d = read_json(marker)
            if (
                d["identity"] != complete["id"]
                or file_hash(out / d["file"]) != d["sha256"]
            ):
                raise ValueError("Incomplete or changed result file")
        with tarfile.open(archive, "x:gz") as t:
            for f in sorted(out.iterdir()):
                if f.is_file() and f.suffix in (
                    ".json",
                    ".jsonl",
                    ".csv",
                    ".svg",
                    ".png",
                ):
                    t.add(f, arcname=f.name, recursive=False)
    print(f"Review archive: {archive}")


if __name__ == "__main__":
    main()
