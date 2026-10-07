from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "ae": "#0072B2",
    "pc": "#E69F00",
    "rae": "#009E73",
    "rpc": "#CC79A7",
    "pod": "#595959",
    "persistence": "#222222",
    "truth": "#000000",
}
MARKERS = {"ae": "o", "pc": "s", "rae": "^", "rpc": "D", "pod": "v", "persistence": "X"}


def configure():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.4,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "svg.hashsalt": "ampub-v1.4.0",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def panel(ax, letter):
    ax.text(
        -0.15,
        1.04,
        letter,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=11,
        va="bottom",
    )


def save(fig, base):
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "svg", "png"]:
        import io, os

        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format=ext,
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
            metadata=(
                {"Creator": "ampub 1.4.0", "CreationDate": None, "ModDate": None}
                if ext == "pdf"
                else ({"Date": None} if ext == "svg" else None)
            ),
        )
        target = base.with_suffix("." + ext)
        temporary = target.with_suffix("." + ext + ".tmp")
        temporary.write_bytes(buffer.getvalue())
        os.replace(temporary, target)
    plt.close(fig)
