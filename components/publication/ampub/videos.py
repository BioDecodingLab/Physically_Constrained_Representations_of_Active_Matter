"""Animations of aggregate frozen error curves, not simulated field movies."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from .figures import curves
from .style import configure


def generate_videos(data, out):
    configure()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for i, (kind, metric) in enumerate([("rollout", "F"), ("short_forecast", "Q")], 2):
        path = out / f"Supplementary_Video_{i}_aggregate_{metric}.mp4"
        leads = sorted(
            data.lead[
                (data.lead.dimension == 256) & (data.lead.kind == kind)
            ].lead.unique()
        )
        with imageio.get_writer(
            path, fps=6, codec="libx264", quality=8, macro_block_size=2
        ) as writer:
            for lead in leads:
                fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
                curves(ax, data, 256, kind, metric)
                ax.set_xlim(0, float(lead) * 0.25)
                ax.legend(frameon=False)
                ax.set_title(f"Aggregate test errors at d=256 · lead {int(lead)}")
                fig.text(
                    0.5,
                    0.01,
                    "Shading: ±1 seed SD; deterministic controls have no seed band. Not prediction intervals.",
                    ha="center",
                    fontsize=8,
                )
                fig.subplots_adjust(bottom=0.18)
                fig.canvas.draw()
                frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
                writer.append_data(frame)
                plt.close(fig)
        files.append(path)
    return files


def generate_reference_video(root, out):
    import json

    root = Path(root)
    meta = json.loads((root / "VISUAL_EXPORT.json").read_text())
    a = np.load(root / "visuals.npz", allow_pickle=False)
    movie = a["reference_movie"]
    times = a["reference_times"]
    L = meta["length"][0]
    path = Path(out) / "Supplementary_Video_1_reference_fields.mp4"
    maxima = [float(np.nanmax(movie[:, i])) for i in range(3)]
    omega = float(np.nanmax(np.abs(movie[:, 1])))
    limits = [(0, maxima[0]), (-omega, omega), (0, 1)]
    with imageio.get_writer(
        path, fps=8, codec="libx264", quality=8, macro_block_size=2
    ) as writer:
        for j, fields in enumerate(movie):
            fig = reference_frame(fields, times[j], meta, limits)
            fig.canvas.draw()
            writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
            plt.close(fig)
    return path


def reference_frame(fields, time, meta, limits):
    """Fixed image and color-bar axes leave room for coordinate labels."""
    fig=plt.figure(figsize=(9,4.8),dpi=120)
    Lx,Ly=meta['length']
    for i,x in enumerate([.07,.38,.69]):
        ax=fig.add_axes([x,.34,.24,.45])
        im=ax.imshow(fields[i].T,origin='lower',extent=[0,Lx,0,Ly],cmap=['magma','RdBu_r','cividis'][i],vmin=limits[i][0],vmax=limits[i][1])
        ax.set_xlabel('x',labelpad=5);ax.set_ylabel('y',labelpad=3)
        ax.set_title(['Speed','Vorticity','Local order'][i],pad=8)
        cbax=fig.add_axes([x,.19,.24,.025])
        cb=fig.colorbar(im,cax=cbax,orientation='horizontal')
        cb.set_label(['|v|','ω','S'][i],labelpad=3)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle(f'Reference active-fluid trajectory · t={time:g}',y=.94)
    fig.text(.5,.035,'Observed simulation fields · fixed color limits · display sampled at 128 × 128',ha='center',fontsize=8)
    return fig
