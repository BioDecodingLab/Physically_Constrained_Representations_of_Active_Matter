#!/usr/bin/env python3
"""Acquire the content-pinned HDF5 files used by the study."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "components/representation/src"))
from active_matter_repr.download import validate_catalog, download_file
from active_matter_repr.artifacts import file_hash, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    root = args.out.expanduser().resolve()
    catalog = json.loads((ROOT / "provenance/dataset_catalog.json").read_text())
    expected = json.loads((ROOT / "provenance/paper_data_sha256.json").read_text())
    validate_catalog(catalog)
    if {entry["path"]: entry["sha256"] for entry in catalog["files"]} != expected:
        raise ValueError("Catalog does not match study data hashes")
    root.mkdir(parents=True, exist_ok=True)
    receipt = root / "download_manifest.json"
    if receipt.exists() and json.loads(receipt.read_text()) != catalog:
        raise ValueError("Existing download receipt differs from the pinned catalog")
    extras = {str(p.relative_to(root)) for p in root.glob("*/*.h*") if p.suffix in (".h5", ".hdf5")} - set(expected)
    if extras:
        raise ValueError(f"Unexpected HDF5 files in dataset directory: {sorted(extras)}")
    write_json(receipt, catalog)
    remaining = sum(max(0, entry["size"] -
                    ((root / (entry["path"] + "." + catalog["revision"] + ".part")).stat().st_size
                     if (root / (entry["path"] + "." + catalog["revision"] + ".part")).exists() else 0))
                    for entry in catalog["files"] if not (root / entry["path"]).exists())
    if shutil.disk_usage(root).free < remaining + 1024**3:
        raise OSError(f"Insufficient free space: {remaining / 1024**3:.2f} GiB of data remain")
    for entry in catalog["files"]:
        download_file(entry, catalog["revision"], root)
        if file_hash(root / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Existing data checksum mismatch: {entry['path']}")
    write_json(root / "DATA_VERIFIED.json", {"revision": catalog["revision"], "files": len(expected),
               "total_bytes": sum(entry["size"] for entry in catalog["files"]), "sha256": expected})
    print(f"Verified {len(expected)} files at revision {catalog['revision']}", flush=True)


if __name__ == "__main__":
    main()
