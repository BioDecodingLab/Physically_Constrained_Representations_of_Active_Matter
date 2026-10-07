from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import pandas as pd

METHOD_LABELS = {
    "ae": "AE",
    "pc": "PC-AE",
    "rae": "R-AE",
    "rpc": "R-PC-AE",
    "pod": "POD",
}
PRIMARY_VARIANTS = {
    "ae": "legacy",
    "pc": "legacy",
    "rae": "direct_corrected",
    "rpc": "direct_corrected",
    "pod": "projected",
}
ORDERED_METHODS = ["ae", "pc", "rae", "rpc", "pod"]


@dataclass
class FinalData:
    root: Path
    dimensions: list[int]
    summary: pd.DataFrame
    lead: pd.DataFrame
    probes: pd.DataFrame
    decoder: pd.DataFrame
    coverage: pd.DataFrame
    spatial: pd.DataFrame
    paired: pd.DataFrame
    group_metrics: pd.DataFrame


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _dimension_from_path(path: Path) -> int | None:
    m = re.fullmatch(r"d(\d+)", path.name)
    return int(m.group(1)) if m else None


def _per_dim(root: Path, filename: str) -> pd.DataFrame:
    frames = []
    dims = []
    for d in (128, 256, 512):
        p = root / f"d{d}" / filename
        if p.exists():
            df = _read(p)
            if not df.empty:
                df.insert(0, "dimension", d)
                frames.append(df)
                dims.append(d)
    if frames:
        return pd.concat(frames, ignore_index=True)
    # Allow a single per-dimension root in synthetic/testing mode.
    p = root / filename
    if p.exists():
        df = _read(p)
        d = _dimension_from_path(root)
        if d is not None and "dimension" not in df.columns:
            df.insert(0, "dimension", d)
        return df
    return pd.DataFrame()


def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    categorical = {
        "kind",
        "method",
        "predictor",
        "variant",
        "trajectory_id",
        "group",
        "metric",
        "a",
        "b",
    }
    for c in out.columns:
        if c not in categorical:
            try:
                out[c] = pd.to_numeric(out[c])
            except (ValueError, TypeError):
                pass
    return out


def load_final(root: Path) -> FinalData:
    # Prefer root comparison summary because it carries dimension explicitly.
    summary = _read(root / "comparison_summary.csv")
    if summary.empty:
        summary = _per_dim(root, "summary.csv")
    lead = _per_dim(root, "lead_summary.csv")
    probes = _per_dim(root, "probe_summary.csv")
    decoder = _per_dim(root, "decoder_diagnostics.csv")
    coverage = _per_dim(root, "coverage.csv")
    spatial = _per_dim(root, "spatial_summary.csv")
    paired = _read(root / "paired_comparisons.csv")
    group_metrics = _read(root / "group_metrics.csv")

    frames = [summary, lead, probes, decoder, coverage, spatial, paired, group_metrics]
    summary, lead, probes, decoder, coverage, spatial, paired, group_metrics = [
        _numeric(f) for f in frames
    ]
    dims = sorted(
        {
            int(x)
            for x in summary.get("dimension", pd.Series(dtype=int)).dropna().unique()
        }
    )
    if not dims:
        for df in (lead, probes, decoder):
            if "dimension" in df:
                dims = sorted({int(x) for x in df["dimension"].dropna().unique()})
                if dims:
                    break
    if summary.empty:
        raise ValueError("No summary.csv/comparison_summary.csv was found")
    required = {"kind", "seed", "method", "predictor", "variant"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Summary is missing required columns: {sorted(missing)}")
    return FinalData(
        root,
        dims,
        summary,
        lead,
        probes,
        decoder,
        coverage,
        spatial,
        paired,
        group_metrics,
    )


def primary_rows(
    df: pd.DataFrame, *, kind: str | None = None, predictor: str | None = None
) -> pd.DataFrame:
    if df.empty:
        return df
    x = df.copy()
    if kind is not None:
        x = x[x["kind"] == kind]
    if predictor is not None and "predictor" in x.columns:
        x = x[x["predictor"] == predictor]
    mask = pd.Series(False, index=x.index)
    for method, variant in PRIMARY_VARIANTS.items():
        mask |= (x["method"] == method) & (x["variant"] == variant)
    return x[mask].copy()
