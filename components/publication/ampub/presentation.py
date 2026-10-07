"""Publication-only views of frozen scores and authenticated visual exports."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .statistics import publication_contrast
from .style import COLORS, MARKERS, panel, save
from .schema import ORDERED_METHODS, METHOD_LABELS

ENDPOINTS = [("reconstruction", "Q", "none", "Snapshot Q"),
             ("short_forecast", "F", "linear", "10-step F"),
             ("short_forecast", "Q", "linear", "10-step Q"),
             ("rollout", "F", "linear", "77-step F"),
             ("rollout", "Q", "linear", "77-step Q")]

def endpoint_profile(data):
    if data.group_metrics.empty:
        return pd.DataFrame(columns=["dimension", "endpoint", "estimate", "ci95_low", "ci95_high"])
    rows = []
    for d in data.dimensions:
        for kind, metric, pred, label in ENDPOINTS:
            try:
                r = publication_contrast(data, d, kind, metric,
                      ("ae", "legacy", pred), predictor=pred, percent=True)
            except ValueError as exc:
                if "No group-level" not in str(exc):
                    raise
                continue
            rows.append(dict(r, endpoint=label))
    return pd.DataFrame(rows)

def plot_endpoint_profile(data, out):
    rows = endpoint_profile(data)
    fig, axs = plt.subplots(1, len(data.dimensions), figsize=(7.3, 3.8), layout="constrained", squeeze=False)
    for ax, d, letter in zip(axs.flat, data.dimensions, "abc"):
        ax.axvspan(-100, 0, color="#F6F6F6", zorder=0)
        ax.axvline(0, color="#888888", lw=.8)
        for i, (_, _, _, label) in enumerate(ENDPOINTS):
            z = rows[(rows.dimension == d) & (rows.endpoint == label)] if not rows.empty else rows
            if z.empty or not np.isfinite(z.iloc[0].estimate):
                ax.text(.03, i, "undefined", transform=ax.get_yaxis_transform(), fontsize=7)
                continue
            r = z.iloc[0]
            ax.hlines(i, r.ci95_low, r.ci95_high, color=COLORS["rpc"], lw=1.5)
            ax.scatter(r.estimate, i, color=COLORS["rpc"], marker="D", s=25)
        ax.set_yticks(range(len(ENDPOINTS)), [e[3] for e in ENDPOINTS])
        ax.invert_yaxis()
        ax.set_title(f"d = {d}")
        ax.set_xlabel("Error reduction vs AE (%)")
        panel(ax, letter)
    # Common axes prevent visual amplification of differences across budgets.
    finite = rows[["ci95_low", "ci95_high"]].to_numpy().ravel() if not rows.empty else np.array([])
    finite = finite[np.isfinite(finite)]
    low, high = (min(-5, finite.min()), max(5, finite.max())) if len(finite) else (-5, 5)
    pad = .07 * (high-low)
    for ax in axs.flat:
        ax.set_xlim(low-pad, high+pad)
    save(fig, out / "Figure_S7_endpoint_profile")
    return rows

def pca_panel(ax, arrays, seed, method, dimension=256):
    prefix = f"pca_{seed}_{method}_"
    xy, obs = arrays[prefix+"xy"], arrays[prefix+"observables"]
    ids, frames = arrays[prefix+"trajectory"], arrays[prefix+"frame"]
    for tid in np.unique(ids):
        ix = np.flatnonzero(ids == tid)
        ix = ix[np.argsort(frames[ix])]
        ax.plot(xy[ix,0], xy[ix,1], color="#999999", lw=.4, alpha=.35)
    im = ax.scatter(xy[:,0], xy[:,1], c=frames / max(1, frames.max()), cmap="cividis", s=5,
                    rasterized=True, vmin=0, vmax=1)
    variance = arrays[prefix+"variance"]
    ax.set(xlabel=f"PC1 ({100*variance[0]:.1f}%)", ylabel=f"PC2 ({100*variance[1]:.1f}%)")
    ax.set_title(METHOD_LABELS[method] + f" · d={dimension} · seed {seed}", fontsize=9)
    ax.text(.02, .02, f"{len(np.unique(ids))} validation trajectories · {len(ids):,} frames", transform=ax.transAxes, fontsize=6)
    ax.set_aspect("equal", adjustable="box")
    return im

def main_latent_figure(data, arrays, meta, out):
    from .statistics import seed_stats
    fig = plt.figure(figsize=(7.3, 6.8), layout="constrained")
    gs = fig.add_gridspec(2, 6, height_ratios=[1.4,1])
    # Fixed seed, never selected for a visually attractive projection.
    seed = 1729
    top = [fig.add_subplot(gs[0,:3]), fig.add_subplot(gs[0,3:])]
    for ax, method, letter in zip(top, ["ae","rpc"], "ab"):
        im = pca_panel(ax, arrays, seed, method, meta.get("dimension",256))
        panel(ax, letter)
    fig.colorbar(im, ax=top, shrink=.75, label="Normalized stored time")
    for j, (d, letter) in enumerate(zip(data.dimensions, "cde")):
        ax = fig.add_subplot(gs[1, 2*j:2*j+2])
        for i, m in enumerate(ORDERED_METHODS):
            x = data.probes[(data.probes.dimension == d)&(data.probes.method == m)]
            st = seed_stats(x,"combined_gain")
            ax.errorbar(i, st["mean"], yerr=st["sd"] if m != "pod" else None,
                        fmt=MARKERS[m], color=COLORS[m], capsize=2)
            if m != "pod":
                ax.scatter(i+np.linspace(-.07,.07,len(x)), x.combined_gain,
                            s=10, color=COLORS[m], alpha=.55)
        ax.axhline(0,color="#888",lw=.7)
        ax.set_xticks(range(5),[METHOD_LABELS[m] for m in ORDERED_METHODS],rotation=55,ha="right")
        ax.set_title(f"d = {d}")
        ax.set_ylabel("Control Q − combined Q")
        panel(ax,letter)
    save(fig,out / "main/Figure_5_latent_information")

def reconstruction_errors(arrays, meta, out):
    reference = arrays["gallery_truth"]
    extent = [0,meta["length"][0],0,meta["length"][1]]
    methods = ["ae","rpc","pod"]
    fig, axs = plt.subplots(3,3,figsize=(7.3,6.7),layout="constrained")
    for row,k in enumerate([1,2,3]):
        errors = [arrays["gallery_"+m][k]-reference[k] for m in methods]
        finite = np.concatenate([e[np.isfinite(e)].ravel() for e in errors])
        limit = float(np.max(np.abs(finite))) if finite.size else 1.
        limit = max(limit,np.finfo(float).eps)
        for col,(method,error) in enumerate(zip(methods,errors)):
            ax = axs[row,col]
            im = ax.imshow(error.T,origin="lower",extent=extent,cmap="RdBu_r",vmin=-limit,vmax=limit)
            ax.set_xticks([0,extent[1]]); ax.set_yticks([0,extent[3]])
            ax.set_title(METHOD_LABELS[method] if row == 0 else "")
            if col == 0:
                ax.set_ylabel(["Speed error","Vorticity error","Order error"][row])
            if row == 2:
                ax.set_xlabel("x")
            panel(ax,chr(97+3*row+col))
        fig.colorbar(im,ax=list(axs[row]),shrink=.8,label="Decoded − reference")
    save(fig,out / "supplementary/Figure_S3_reconstruction_errors")
