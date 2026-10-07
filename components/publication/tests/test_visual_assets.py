"""Synthetic fixtures test visual adapters only; never used as scientific evidence."""

import json
import numpy as np
from ampub.provenance import file_hash
from ampub.visuals import generate_visuals
from ampub.videos import generate_reference_video


def test_optional_visual_adapter_and_reference_movie(tmp_path, final_fixture):
    root = tmp_path / "synthetic_export"
    root.mkdir()
    rng = np.random.default_rng(1729)
    views = np.ones((3, 5, 16, 16), dtype=np.float32)
    views[:, 1] = rng.uniform(0, 1, (3, 16, 16))
    views[:, 2] = rng.normal(size=(3, 16, 16))
    views[:, 3] = rng.uniform(0, 1, (3, 16, 16))
    arrays = {
        "dataset_views": views,
        "gallery_truth": views[1],
        "gallery_ae": views[1],
        "gallery_rpc": views[1],
        "gallery_pod": views[1],
        "reference_movie": np.repeat(views[1, 1:4][None], 3, axis=0),
        "reference_times": np.array([0, 0.25, 0.5]),
    }
    for seed in [1729, 2718, 31415]:
        for m in ["ae", "rpc"]:
            prefix = f"pca_{seed}_{m}_"
            arrays.update(
                {
                    prefix + "xy": rng.normal(size=(24, 2)),
                    prefix + "observables": rng.uniform(0, 1, (24, 3)),
                    prefix + "variance": np.array([0.2, 0.1]),
                    prefix + "trajectory": np.repeat([0, 1], 12),
                    prefix + "frame": np.tile(np.arange(12), 2),
                }
            )
    np.savez_compressed(root / "visuals.npz", **arrays)
    meta = {
        "final_evaluation_id": "SOFTWARE_TEST",
        "visuals_sha256": file_hash(root / "visuals.npz"),
        "length": [10, 10],
        "dimension": 256,
        "seeds": [1729, 2718, 31415],
        "dataset_selected": [{"parameters": [-i - 1, 1]} for i in range(3)],
    }
    (root / "VISUAL_EXPORT.json").write_text(json.dumps(meta))
    out = tmp_path / "synthetic_only"
    out.mkdir()
    from ampub.schema import load_final
    status = generate_visuals(root, out / "figures", "SOFTWARE_TEST", load_final(final_fixture))
    assert all(status.values())
    assert len(list((out / "figures").rglob("*.pdf"))) == 6
    assert (out / "figures/main/Figure_5_latent_information.svg").exists()
    videos = out / "videos"
    videos.mkdir()
    movie = generate_reference_video(root, videos)
    assert movie.stat().st_size > 1000
