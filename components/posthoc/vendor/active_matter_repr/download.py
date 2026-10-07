"""Missing-file acquisition from the official Well mirror, with pinned resumes."""
import json
from pathlib import Path, PurePosixPath
import re
import shutil
from urllib.request import Request, urlopen
import h5py
from tqdm.auto import tqdm
from .artifacts import read_json, write_json, file_hash

REPOSITORY = "polymathic-ai/active_matter"
API = f"https://huggingface.co/api/datasets/{REPOSITORY}/revision/main?blobs=true"


def local_complete(cfg):
    """No network for an existing complete dataset; scientific checks follow."""
    root = Path(cfg["data_root"])
    for split, expected in zip(("train", "valid", "test"), cfg["expected_counts"]):
        count = 0
        files = sorted(set((root/split).glob("*.h5")) | set((root/split).glob("*.hdf5")))
        for path in files:
            # Corrupt existing files require inspection, never silent replacement.
            with h5py.File(path, "r") as h:
                count += int(h.attrs["n_trajectories"])
        if count < expected:
            return False
    return True


def official_catalog():
    with urlopen(API, timeout=60) as response:
        metadata = json.load(response)
    entries = []
    for entry in metadata["siblings"]:
        name = PurePosixPath(entry["rfilename"])
        if len(name.parts) != 3 or name.parts[0] != "data" or name.parts[1] not in ("train", "valid", "test") or name.suffix not in (".h5", ".hdf5"):
            continue
        entries.append({"path": str(name.relative_to("data")), "size": entry["size"], "sha256": entry["lfs"]["sha256"]})
    catalog = {"repository": REPOSITORY, "revision": metadata["sha"], "files": entries}
    validate_catalog(catalog)
    return catalog


def validate_catalog(catalog):
    if catalog["repository"] != REPOSITORY or not re.fullmatch(r"[0-9a-f]{40}", catalog["revision"]):
        raise ValueError("Invalid official download revision/repository")
    seen, splits = set(), set()
    for entry in catalog["files"]:
        p = PurePosixPath(entry["path"])
        if len(p.parts) != 2 or p.parts[0] not in ("train", "valid", "test") or p.suffix not in (".h5", ".hdf5") or "\\" in str(p) or str(p) in seen:
            raise ValueError("Invalid/duplicate dataset download path")
        if type(entry["size"]) is not int or entry["size"] <= 0 or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError("Invalid download size/checksum")
        seen.add(str(p))
        splits.add(p.parts[0])
    if splits != {"train", "valid", "test"}:
        raise ValueError("Official catalog is missing a split")


def download_file(entry, revision, root):
    target = Path(root) / entry["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != entry["size"]:
            raise ValueError(f"Existing file has unexpected size; inspect/move it before retrying: {target}")
        return
    # Partial data are associated with this pinned revision; complete files are
    # exposed under their final name only after size and SHA256 verification.
    part = target.with_name(target.name + "." + revision + ".part")
    offset = part.stat().st_size if part.exists() else 0
    if offset > entry["size"]:
        raise ValueError(f"Oversized partial file; inspect/remove it: {part}")
    if offset < entry["size"]:
        url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/data/{entry['path']}"
        request = Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        with urlopen(request, timeout=120) as response:
            status = response.status
            if status == 206:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if not match or int(match[1]) != offset or int(match[3]) != entry["size"]:
                    raise ValueError("Server returned an incompatible partial response")
            elif status == 200:
                offset = 0  # Server ignored Range: restart, never append a full file.
            else:
                raise ValueError(f"Unexpected download HTTP status {status}")
            with open(part, "ab" if offset else "wb") as stream, tqdm(total=entry["size"], initial=offset, unit="B", unit_scale=True, desc=target.name) as bar:
                while chunk := response.read(4 * 1024 * 1024):
                    if stream.tell() + len(chunk) > entry["size"]:
                        raise ValueError("Response exceeds the official file size")
                    stream.write(chunk)
                    bar.update(len(chunk))
    if part.stat().st_size != entry["size"]:
        raise IOError(f"Incomplete download retained for resume: {part}")
    if file_hash(part) != entry["sha256"]:
        part.unlink()  # Only our failed partial file; keep existing source files intact.
        raise ValueError(f"Official SHA256 verification failed: {target.name}; rerun to download again")
    part.replace(target)


def ensure_data(cfg):
    if cfg["synthetic"] or not cfg.get("auto_download", True):
        return
    if local_complete(cfg):
        print("Local dataset present; no download needed", flush=True)
        return
    root = Path(cfg["data_root"])
    if "/absolute/path/" in str(root):
        raise ValueError("Set data_root to your desired download directory first")
    root.mkdir(parents=True, exist_ok=True)
    receipt = root / "download_manifest.json"
    catalog = read_json(receipt) if receipt.exists() else official_catalog()
    validate_catalog(catalog)
    write_json(receipt, catalog)  # Pin before transfers, including interrupted ones.
    remaining = 0
    for entry in catalog["files"]:
        target = root / entry["path"]
        if not target.exists():
            part = target.with_name(target.name + "." + catalog["revision"] + ".part")
            remaining += max(0, entry["size"] - (part.stat().st_size if part.exists() else 0))
    if shutil.disk_usage(root).free < remaining + 1024**3:
        raise OSError(f"Insufficient disk space: need approximately {remaining/1024**3:.1f} GiB plus 1 GiB reserve for missing data")
    print(f"Official {REPOSITORY}: downloading missing files into {root} ({remaining/1024**3:.1f} GiB remaining)", flush=True)
    for entry in catalog["files"]:
        download_file(entry, catalog["revision"], root)
