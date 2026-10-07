from __future__ import annotations
import argparse, json, platform, sys, importlib.metadata
from pathlib import Path
from .io import materialized_input, copy_source_metadata
from .schema import load_final
from .figures import generate_all_figures
from .tables import generate_tables
from .videos import generate_videos
from .provenance import inventory, validate_final


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Publication-only renderer for frozen ActiveMatter v0.4.0 results"
    )
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--visuals", help="Optional read-only export from ampub.export_visuals"
    )
    p.add_argument("--no-videos", action="store_true")
    p.add_argument(
        "--software-test",
        action="store_true",
        help="Synthetic fixtures only; outputs are not scientific evidence",
    )
    a = p.parse_args(argv)
    out = Path(a.out).expanduser().resolve()
    with materialized_input(a.input) as root:
        if out == root or root in out.parents or out in root.parents:
            raise ValueError("Publication output must be separate from frozen input")
        data = load_final(root)
        before = inventory(root)
        ident = "SOFTWARE_TEST" if a.software_test else validate_final(root, data)
        out.mkdir(parents=True, exist_ok=True)
        generate_all_figures(data, out / "figures")
        generate_tables(data, out / "tables")
        visual_status = {
            "dataset_examples": False,
            "latent_PCA": False,
            "reconstruction_gallery": False,
        }
        if a.visuals:
            from .visuals import generate_visuals

            visual_status = generate_visuals(Path(a.visuals), out / "figures", ident, data)
        videos = [] if a.no_videos else generate_videos(data, out / "videos")
        if a.visuals and not a.no_videos:
            from .videos import generate_reference_video

            videos.append(generate_reference_video(Path(a.visuals), out / "videos"))
        copy_source_metadata(root, out / "source_metadata")
        if before != inventory(root):
            raise RuntimeError("Frozen input changed during rendering")
        versions = {
            k: importlib.metadata.version(k)
            for k in [
                "numpy",
                "pandas",
                "matplotlib",
                "imageio",
                "imageio-ffmpeg",
                "scikit-learn",
            ]
        }
        manifest = {
            "renderer": "active-matter-publication 1.4.1",
            "final_evaluation_id": ident,
            "scientific_logic_modified": False,
            "dimensions": data.dimensions,
            "source_sha256": before,
            "python": sys.version,
            "platform": platform.platform(),
            "dependencies": versions,
            "generated": {
                "figures": len(list((out / "figures").rglob("*.pdf"))),
                "tables": len(list((out / "tables").rglob("*.csv"))),
                "videos": len(videos),
            },
            "visual_inputs": visual_status,
            "synthetic_software_test": a.software_test,
            "uncertainty": "seed SD is fitted-model variation; paired group intervals conditional on fitted seeds; no calibrated prediction intervals",
        }
        (out / "PUBLICATION_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        checks = inventory(out)
        checks.pop("SHA256SUMS.txt", None)
        (out / "SHA256SUMS.txt").write_text(
            "\n".join(f"{h}  {name}" for name, h in checks.items()) + "\n"
        )
        print(
            json.dumps(
                {k: v for k, v in manifest.items() if k != "source_sha256"}, indent=2
            )
        )
