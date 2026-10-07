"""Figures with explicit model variants, matched contrasts, and visible seed spread."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .style import configure, COLORS, MARKERS, panel, save
from .schema import METHOD_LABELS, PRIMARY_VARIANTS, ORDERED_METHODS, primary_rows
from .statistics import (
    select,
    seed_stats,
    strict_mean,
    uncertainty_tables,
    group_matrix,
)


def method_legend(ax, ncol=3):
    ax.legend(frameon=False, ncol=ncol, loc="best")


def seed_line(ax, data, kind, metric, methods=ORDERED_METHODS, predictor="none"):
    for method in methods:
        means = []
        sds = []
        for i, d in enumerate(data.dimensions):
            x = select(
                data.summary, d, kind, method, PRIMARY_VARIANTS[method], predictor
            )
            st = seed_stats(x, metric)
            means.append(st["mean"])
            sds.append(st["sd"])
            if method != "pod":
                ax.scatter(
                    i + (np.arange(len(x)) - (len(x) - 1) / 2) * 0.05,
                    x[metric],
                    s=13,
                    alpha=0.55,
                    color=COLORS[method],
                    marker=MARKERS[method],
                    zorder=3,
                )
        ax.errorbar(
            range(len(means)),
            means,
            yerr=sds if method != "pod" else None,
            color=COLORS[method],
            marker=MARKERS[method],
            capsize=2,
            label=METHOD_LABELS[method],
            zorder=4,
        )
    ax.set_xticks(range(len(data.dimensions)), data.dimensions)
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel(metric + " (lower is better)")


def figure2_reconstruction(data, out):
    fig, axs = plt.subplots(2, 2, figsize=(7.3, 6.6), layout="constrained")
    seed_line(axs[0, 0], data, "reconstruction", "Q")
    seed_line(axs[0, 1], data, "reconstruction", "R_rec")
    method_legend(axs[0, 0], 2)
    _, extra = uncertainty_tables(data)
    ax = axs[1, 0]
    if not extra.empty:
        z = extra[extra.kind == "reconstruction"]
        xx = np.arange(len(z))
        ax.vlines(xx, z.ci95_low, z.ci95_high, color=COLORS["rpc"])
        ax.scatter(xx, z.estimate, color=COLORS["rpc"], marker="D")
        ax.set_xticks(xx, z.dimension)
    ax.axhline(0, color="#888", lw=0.7)
    ax.set(xlabel="Latent dimension", ylabel="Q reduction relative to AE (%)")
    ax = axs[1, 1]
    if not data.group_metrics.empty:
        for i, d in enumerate(data.dimensions):
            a = group_matrix(data, d, "reconstruction", "ae", "legacy", "Q").mean(
                axis=0, skipna=False
            )
            b = group_matrix(
                data, d, "reconstruction", "rpc", "direct_corrected", "Q"
            ).mean(axis=0, skipna=False)
            diff = a - b
            offset = np.linspace(-0.18, 0.18, len(diff))
            ax.scatter(i + offset, diff, s=12, c=COLORS["rpc"], alpha=0.75)
            ax.plot([i - 0.2, i + 0.2], [diff.mean()] * 2, color="black", lw=1.5)
        ax.set_xticks(range(len(data.dimensions)), data.dimensions)
    ax.axhline(0, color="#888", lw=0.7)
    ax.set(xlabel="Latent dimension", ylabel="Group Q difference (AE − R-PC-AE)")
    for ax, l in zip(axs.flat, "abcd"):
        panel(ax, l)
    save(fig, out / "Figure_2_reconstruction_compression")


def figure3_constraints(data, out):
    fig, axs = plt.subplots(2, 2, figsize=(7.3, 7), layout="constrained")
    ax = axs[0, 0]
    p = data.paired
    if not p.empty:
        y = 0
        ticks = []
        labels = []
        for d in data.dimensions:
            for m, var in [
                ("ae", "legacy"),
                ("ae", "projected"),
                ("pc", "legacy"),
                ("rae", "direct_corrected"),
                ("pod", "projected"),
            ]:
                z = p[
                    (p.dimension == d)
                    & (p.kind == "reconstruction")
                    & (p.predictor == "none")
                    & (p.metric == "Q")
                    & (p.a == "rpc/direct_corrected")
                    & (p.b == m + "/" + var)
                ]
                if len(z) != 1:
                    continue
                r = z.iloc[0]
                ax.hlines(y, r.ci95_low, r.ci95_high, color=COLORS[m])
                ax.scatter(
                    r.difference_b_minus_a, y, color=COLORS[m], marker=MARKERS[m], s=22
                )
                ticks.append(y)
                labels.append(
                    f"{d}  "
                    + METHOD_LABELS[m]
                    + (" + projection" if m == "ae" and var == "projected" else "")
                )
                y += 1
            y += 0.5
        ax.set_yticks(ticks, labels, fontsize=7)
        ax.invert_yaxis()
        ax.axvline(0, color="#888", lw=0.7)
    ax.set_xlabel("Comparator Q − R-PC-AE Q")
    for ax, metric, lab in [
        (axs[0, 1], "D_negative_fraction", "Fraction with negative D eigenvalue"),
        (axs[1, 0], "divergence_rms", "Normalized divergence RMS"),
        (axs[1, 1], "strain_mismatch_rms", "Normalized strain mismatch RMS"),
    ]:
        z = primary_rows(data.summary, kind="reconstruction")
        z = z[z.dimension == 256]
        for i, m in enumerate(ORDERED_METHODS):
            x = z[z.method == m]
            st = seed_stats(x, metric)
            ax.errorbar(
                i,
                st["mean"],
                yerr=st["sd"] if m != "pod" else None,
                color=COLORS[m],
                fmt=MARKERS[m],
                capsize=3,
            )
            if m != "pod":
                ax.scatter(
                    i + np.linspace(-0.08, 0.08, len(x)),
                    x[metric],
                    s=13,
                    alpha=0.5,
                    color=COLORS[m],
                )
        ax.set_xticks(
            range(5),
            [METHOD_LABELS[m] for m in ORDERED_METHODS],
            rotation=35,
            ha="right",
        )
        ax.set_ylabel(lab)
    for ax, l in zip(axs.flat, "abcd"):
        panel(ax, l)
    save(fig, out / "Figure_3_constraint_ablation")


def curves(ax, data, dimension, kind, metric, mlp=False):
    specs = [
        ("rpc", "direct_corrected", "linear", "R-PC-AE + linear", COLORS["rpc"], "-"),
        ("pod", "projected", "linear", "POD + linear", COLORS["pod"], "--"),
        ("rpc", "legacy", "persistence", "Persistence", COLORS["persistence"], ":"),
    ]
    if mlp:
        specs.append(
            ("rpc", "direct_corrected", "mlp", "R-PC-AE + MLP", COLORS["pc"], "-.")
        )
    for m, var, pred, lab, col, ls in specs:
        z = select(data.lead, dimension, kind, m, var, pred)
        if z.empty:
            continue
        stats = []
        for (lead, t), x in z.groupby(["lead", "time_since_origin"]):
            stats.append(dict(lead=lead, t=t, **seed_stats(x, metric)))
        g = pd.DataFrame(stats).sort_values("lead")
        ax.plot(g.t, g["mean"], color=col, ls=ls, label=lab)
        if pred != "persistence" and not (m == "pod" and pred == "linear"):
            ax.fill_between(
                g.t, g["mean"] - g.sd, g["mean"] + g.sd, color=col, alpha=0.14
            )
    ax.set(xlabel="Time since origin", ylabel=metric + " (lower is better)")


def figure4_dynamics(data, out, dimension=256):
    fig, axs = plt.subplots(2, 3, figsize=(7.3, 5.8), layout="constrained")
    for i, metric in enumerate(["F", "Q"]):
        curves(axs[i, 0], data, dimension, "short_forecast", metric)
        curves(axs[i, 1], data, dimension, "rollout", metric)
        ax = axs[i, 2]
        _, extra = uncertainty_tables(data)
        y = 0
        labs = []
        for kind in ["short_forecast", "rollout"]:
            if extra.empty:
                continue
            z = extra[
                (extra.dimension == dimension)
                & (extra.kind == kind)
                & (extra.metric == metric)
            ]
            if len(z) == 1:
                r = z.iloc[0]
                ax.hlines(y, r.ci95_low, r.ci95_high, color=COLORS["rpc"])
                ax.scatter(r.estimate, y, color=COLORS["rpc"], marker="D")
                labs.append("10-step" if kind == "short_forecast" else "77-step")
                y += 1
        ax.axvline(0, color="#888", lw=0.7)
        ax.set_yticks(range(len(labs)), labs)
        ax.set_xlabel(f"Persistence {metric} −\nR-PC-AE linear {metric}")
        ax.set_ylim(-0.6, 1.6)
    axs[0, 0].legend(frameon=False, fontsize=7)
    for i in range(2):
        axs[i, 0].set_xlabel("Time (10-step forecast)")
        axs[i, 1].set_xlabel("Time (77-step rollout)")
    for ax, l in zip(axs.flat, "abcdef"):
        panel(ax, l)
    save(fig, out / "Figure_4_dynamics")


def supplementary_figures(data, out):
    out.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(7.3, 3.4), layout="constrained")
    for ax, metric, l in zip(axs, ["combined", "latent", "parameters_time"], "abc"):
        for m in ORDERED_METHODS:
            means = []
            sds = []
            for d in data.dimensions:
                x = data.probes[
                    (data.probes.dimension == d) & (data.probes.method == m)
                ]
                st = seed_stats(x, metric)
                means.append(st["mean"])
                sds.append(st["sd"])
            ax.errorbar(
                range(3),
                means,
                yerr=sds if m != "pod" else None,
                fmt=MARKERS[m] + "-",
                color=COLORS[m],
                capsize=2,
                label=METHOD_LABELS[m],
            )
        ax.set_xticks(range(3), data.dimensions)
        ax.set(xlabel="Latent dimension", ylabel=metric.replace("_", " ") + " probe Q")
        panel(ax, l)
    method_legend(axs[0], 2)
    save(fig, out / "Figure_S9_latent_probes")
    for name, mlp in [("S6_predictor_stability", True)]:
        fig, axs = plt.subplots(3, 4, figsize=(7.3, 8.5), layout="constrained")
        for row, d in enumerate(data.dimensions):
            for col, (kind, metric) in enumerate(
                [
                    ("short_forecast", "F"),
                    ("short_forecast", "Q"),
                    ("rollout", "F"),
                    ("rollout", "Q"),
                ]
            ):
                ax = axs[row, col]
                curves(ax, data, d, kind, metric, mlp=mlp)
                ax.text(
                    0.03,
                    0.96,
                    f"d={d} " + ("10-step" if kind == "short_forecast" else "77-step"),
                    va="top",
                    transform=ax.transAxes,
                    fontsize=7,
                )
                panel(ax, chr(97 + 4 * row + col))
                if mlp:
                    ax.set_yscale("symlog", linthresh=1)
        axs[0, 0].legend(frameon=False, fontsize=6)
        save(fig, out / ("Figure_" + name))
    decoder_diagnostics(data.decoder, data.dimensions, out)
    fig, axs = plt.subplots(3, 2, figsize=(7.3, 7.6), layout="constrained")
    for row, d in enumerate(data.dimensions):
        for col, metric in enumerate(["energy_shell", "correlation"]):
            ax = axs[row, col]
            for m in ["truth"] + ORDERED_METHODS:
                z = data.spatial[
                    (data.spatial.dimension == d)
                    & (data.spatial.method == ("rpc" if m == "truth" else m))
                    & (data.spatial.metric == metric)
                ]
                z = z[z.variant == ("truth" if m == "truth" else PRIMARY_VARIANTS[m])]
                if z.empty:
                    continue
                p = z.pivot(index="seed", columns="coordinate", values="value").reindex(
                    [1729, 2718, 31415]
                )
                g = p.mean(axis=0, skipna=False)
                sd = p.std(axis=0, skipna=False)
                ax.plot(
                    g.index,
                    g.values,
                    color=COLORS[m],
                    label="Reference" if m == "truth" else METHOD_LABELS[m],
                    ls="--" if m == "pod" else "-",
                )
                if m not in ["truth", "pod"]:
                    ax.fill_between(
                        g.index, g - sd, g + sd, color=COLORS[m], alpha=0.12
                    )
            ax.set(
                xlabel="Wavenumber k" if col == 0 else "Separation r",
                ylabel="Velocity shell energy" if col == 0 else "Nematic correlation",
            )
            ax.text(0.97, 0.94, f"d={d}", ha="right", va="top", transform=ax.transAxes)
            panel(ax, chr(97 + row * 2 + col))
            if col == 0:
                ax.set_yscale("log")
                ax.set_xlim(left=0, right=40)
    method_legend(axs[0, 0], 2)
    save(fig, out / "Figure_S4_spatial_reconstruction")
    fig, axs = plt.subplots(1, 3, figsize=(7.3, 3.8), layout="constrained")
    for ax, d, l in zip(axs, data.dimensions, "abc"):
        z = primary_rows(data.summary)
        z = z[
            (z.dimension == d)
            & (z.predictor.isin(["linear", "mlp"]))
            & (z.kind == "rollout")
        ]
        vals = []
        labels = []
        for m in ORDERED_METHODS:
            for pred in ["linear", "mlp"]:
                x = z[(z.method == m) & (z.predictor == pred)]
                vals.append(x.Q_invalid_records.sum())
                labels.append(METHOD_LABELS[m] + " " + pred)
        ax.barh(range(len(vals)), vals, color=["#595959", "#E69F00"] * 5)
        ax.set_yticks(range(len(vals)), labels, fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel("Undefined Q records")
        ax.text(0.98, 0.02, f"d={d}", ha="right", transform=ax.transAxes)
        panel(ax, l)
    save(fig, out / "Figure_S8_coverage_failures")


def generate_all_figures(data, out):
    configure()
    out = Path(out)
    main = out / "main"
    supp = out / "supplementary"
    figure2_reconstruction(data, main)
    figure3_constraints(data, main)
    figure4_dynamics(data, main)
    supplementary_figures(data, supp)
    from .presentation import plot_endpoint_profile
    plot_endpoint_profile(data, supp)


DECODER_METRICS = [
    'legacy_direct_max_abs_change_maximum',
    'direct_high_precision_reference_max_abs_change_maximum',
    'roundtrip_c_max_abs_change_maximum',
]
DECODER_TASKS = [('reconstruction','none','Snapshot reconstruction'),
                 ('rollout','linear','77-step linear rollout'),
                 ('rollout','mlp','77-step MLP rollout')]


def decoder_task_records(frame, kind, predictor):
    """Retain one recorded extremum per fitted run for the requested task."""
    z=frame[(frame.kind==kind)&(frame.predictor==predictor)&frame.method.isin(['rae','rpc'])].copy()
    if z.duplicated(['dimension','seed','method']).any():
        raise ValueError('Ambiguous decoder arithmetic records for the requested task')
    return z


def decoder_diagnostics(frame, dimensions, out):
    from matplotlib.lines import Line2D
    fig, axs=plt.subplots(3,3,figsize=(8,8.2))
    seed_markers={1729:'o',2718:'s',31415:'^'}
    for row,(kind,predictor,title) in enumerate(DECODER_TASKS):
        z=decoder_task_records(frame,kind,predictor)
        for col,(metric,label) in enumerate(zip(DECODER_METRICS,['Legacy − direct','Direct − reference','Concentration round trip'])):
            ax=axs[row,col]
            for i,d in enumerate(dimensions):
                for method,offset in [('rae',-.16),('rpc',.16)]:
                    for j,(seed,marker) in enumerate(seed_markers.items()):
                        x=z[(z.dimension==d)&(z.method==method)&(z.seed==seed)]
                        if len(x):ax.scatter(i+offset+(j-1)*.045,x[metric],s=27,color=COLORS[method],marker=marker,edgecolors='white',linewidths=.3,zorder=3)
            if z.empty:ax.text(.5,.5,'No records',ha='center',transform=ax.transAxes)
            ax.set_xticks(range(len(dimensions)),dimensions)
            ax.set_xlabel('Latent dimension')
            ax.set_ylabel('Maximum absolute difference')
            ax.set_title(title+'\n'+label,fontsize=8.5)
            values=z[metric].to_numpy(dtype=float) if len(z) else np.array([0.])
            maximum=float(np.nanmax(values))
            if maximum>1e-6:
                ax.set_yscale('symlog',linthresh=1e-9)
                from matplotlib.ticker import FixedLocator, LogFormatterMathtext
                ax.yaxis.set_major_locator(FixedLocator([0.,1e-9,1e-6,1e-3,1.,10.]))
                ax.yaxis.set_major_formatter(LogFormatterMathtext())
            else:
                ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3),useOffset=False)
            ax.set_ylim(0,maximum*1.25 if maximum else 1e-9)
            ax.grid(axis='y',alpha=.15)
            panel(ax,chr(97+3*row+col))
    handles=[Line2D([],[],ls='',marker='o',color=COLORS[m],label=METHOD_LABELS[m]) for m in ['rae','rpc']]
    handles += [Line2D([],[],ls='',marker=mk,color='#444444',label=f'Seed {seed}') for seed,mk in seed_markers.items()]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.99),ncol=5,frameon=False,fontsize=8)
    fig.subplots_adjust(top=.90,bottom=.07,left=.10,right=.98,hspace=.65,wspace=.55)
    save(fig,Path(out)/'Figure_S5_decoder_diagnostics')
