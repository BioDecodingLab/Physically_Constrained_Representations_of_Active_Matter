from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from .style import configure, save, panel
from .provenance import file_hash


def generate_visuals(root, out, ident, data=None):
    meta = json.loads((root / "VISUAL_EXPORT.json").read_text())
    if (
        meta["final_evaluation_id"] != ident
        or file_hash(root / "visuals.npz") != meta["visuals_sha256"]
    ):
        raise ValueError("Visual input identity or hash mismatch")
    a = np.load(root / "visuals.npz", allow_pickle=False)
    configure()
    L = meta["length"][0]
    Ly = meta["length"][1]
    fields = [
        "Concentration c",
        "Speed |v|",
        "Vorticity ω",
        "Local order S",
        "Strain norm |E|",
    ]
    cmaps = ["viridis", "magma", "RdBu_r", "cividis", "viridis"]
    fig, axs = plt.subplots(3, 4, figsize=(7.3, 6.3), layout="constrained")
    for col, k in enumerate([0, 1, 3, 4]):
        values = a["dataset_views"][:, k]
        lim = (np.nanmin(values), np.nanmax(values))
        if k == 3:
            lim = (0, 1)
        for row in range(3):
            ax = axs[row, col]
            im = ax.imshow(
                values[row].T,
                origin="lower",
                extent=[0, L, 0, Ly],
                vmin=lim[0],
                vmax=lim[1],
                cmap=cmaps[k],
            )
            ax.set_xticks([0, L])
            ax.set_yticks([0, Ly])
            panel(ax, chr(97 + 4 * row + col))
            if row == 0:
                ax.set_xlabel(fields[k])
                ax.xaxis.set_label_position("top")
            if col == 0:
                params = meta["dataset_selected"][row]["parameters"]
                ax.set_ylabel(f"α={params[0]:g}, ζ={params[1]:g}\ny")
            if row == 2:
                ax.set_xlabel("x")
        fig.colorbar(
            im, ax=list(axs[:, col]), shrink=0.75, pad=0.025, orientation="horizontal"
        )
    save(fig, out / "supplementary/Figure_S1_dataset_examples")
    fig, axs = plt.subplots(3, 4, figsize=(7.3, 6.4), layout="constrained")
    mats = [a["gallery_truth"]] + [a["gallery_" + m] for m in ["ae", "rpc", "pod"]]
    for row, k in enumerate([1, 2, 3]):
        vmax = np.nanmax(np.abs(mats[0][k]))
        lim = (-vmax, vmax) if k == 2 else (0, 1 if k == 3 else vmax)
        for col, mat in enumerate(mats):
            ax = axs[row, col]
            im = ax.imshow(
                mat[k].T,
                origin="lower",
                extent=[0, L, 0, Ly],
                vmin=lim[0],
                vmax=lim[1],
                cmap=cmaps[k],
            )
            ax.set_xticks([0, L])
            ax.set_yticks([0, Ly])
            sat = np.mean((mat[k] < lim[0]) | (mat[k] > lim[1]))
            ax.text(.02, .02, f"outside scale: {100*sat:.1f}%", transform=ax.transAxes, fontsize=6, bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
            panel(ax, chr(97 + 4 * row + col))
            if row == 0:
                ax.set_xlabel(["Reference", "AE", "R-PC-AE", "POD projected"][col])
            if col == 0:
                ax.set_ylabel(fields[k] + "\ny")
        fig.colorbar(im, ax=list(axs[row]), shrink=0.8, pad=0.02)
    save(fig, out / "supplementary/Figure_S2_reconstruction_gallery")
    fig, axs = plt.subplots(3, 2, figsize=(7.3, 8.5), layout="constrained")
    for row, seed in enumerate(meta["seeds"]):
        for col, method in enumerate(["ae", "rpc"]):
            ax = axs[row, col]
            pre = f"pca_{seed}_{method}_"
            xy = a[pre + "xy"]
            obs = a[pre + "observables"]
            var = a[pre + "variance"]
            ids = a[pre + "trajectory"]
            frames = a[pre + "frame"]
            for tid in np.unique(ids):
                ix = np.where(ids == tid)[0]
                ix = ix[np.argsort(frames[ix])]
                ax.plot(
                    xy[ix, 0], xy[ix, 1], color="#AAAAAA", lw=0.4, alpha=0.5, zorder=1
                )
            im = ax.scatter(
                xy[:, 0],
                xy[:, 1],
                c=obs[:, 2],
                cmap="cividis",
                s=4,
                rasterized=True,
                vmin=0,
                vmax=1,
                zorder=2,
            )
            ax.set(xlabel=f"PC1 ({100*var[0]:.1f}%)", ylabel=f"PC2 ({100*var[1]:.1f}%)")
            ax.text(
                0.02,
                0.98,
                f"{'AE' if method=='ae' else 'R-PC-AE'} · d={meta.get('dimension',256)} · seed {seed}",
                va="top",
                transform=ax.transAxes,
            )
            ax.set_aspect("equal", adjustable="box")
            ax.text(.02,.02,f"{len(np.unique(ids))} trajectories · {len(ids):,} frames",transform=ax.transAxes,fontsize=6)
            panel(ax, chr(97 + row * 2 + col))
    fig.colorbar(im, ax=list(axs.flat), shrink=0.5, label="Mean local nematic order")
    save(fig, out / "supplementary/Figure_S10_latent_PCA")
    from .presentation import reconstruction_errors, main_latent_figure
    reconstruction_errors(a, meta, out)
    if data is not None:
        main_latent_figure(data, a, meta, out)
    from .figure1 import main as figure1
    figure1(out / "main", root, "Figure_1_dataset")
    # Portable copy of verified manifest; renderer never reads model files.
    target = out.parent / "source_metadata"
    target.mkdir(exist_ok=True)
    (target / "VISUAL_EXPORT.json").write_text(json.dumps(meta, indent=2) + "\n")
    return {
        "pca_trajectories": int(len(np.unique(a["pca_1729_ae_trajectory"]))),
        "pca_points": int(len(a["pca_1729_ae_xy"])),
        "pca_dimension": meta.get("dimension",256),
        "pca_total_variance": {m: float(a[f"pca_1729_{m}_variance"].sum()) for m in ["ae", "rpc"]},
        "dataset_examples": True,
        "latent_PCA": True,
        "reconstruction_gallery": True,
    }
