from pathlib import Path
import hashlib, json


def file_hash(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory(root):
    root = Path(root)
    return {
        str(p.relative_to(root)): file_hash(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def validate_final(root, data):
    root = Path(root)
    complete = json.loads((root / "COMPLETE.json").read_text())
    if complete.get("split") != "test":
        raise ValueError("Input must be frozen test results")
    ident = complete["id"]
    for d in data.dimensions:
        c = json.loads((root / f"d{d}" / "COMPLETE.json").read_text())
        m = json.loads((root / f"d{d}" / "analysis_manifest.json").read_text())
        if (
            c["id"] != ident
            or m["id"] != ident
            or m["split"] != "test"
            or m["synthetic"]
        ):
            raise ValueError("Mismatched final-evaluation identity or synthetic input")
    keys = ["dimension", "kind", "seed", "method", "predictor", "variant"]
    if data.summary.duplicated(keys).any():
        raise ValueError("Duplicate logical summary rows")
    if data.dimensions != [128, 256, 512]:
        raise ValueError("Primary release requires all three dimensions")
    # Aggregate summary must reproduce equal parameter-group aggregation, including nulls.
    from .statistics import strict_mean, select
    import numpy as np

    for _, row in data.summary.iterrows():
        for metric in ["R_rec", "F", "Q"]:
            z = select(
                data.group_metrics,
                row.dimension,
                row.kind,
                row.method,
                row.variant,
                row.predictor,
            )
            z = z[(z.seed == row.seed) & (z.metric == metric)]
            if z.empty:
                raise ValueError("Missing group records")
            val = strict_mean(z.value)
            if not np.isclose(val, row[metric], rtol=1e-9, atol=1e-12, equal_nan=True):
                raise ValueError("Summary and group aggregation disagree")
    return ident
