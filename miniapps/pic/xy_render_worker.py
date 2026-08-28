"""Worker for parallel x-y particle scatter rendering.

Kept as a standalone module so ProcessPoolExecutor can pickle it from
Jupyter (notebook __main__ functions are not picklable under spawn).
"""

from __future__ import annotations


def render_one_frame(args):
    """Load one cached frame and write a PNG."""
    (
        frame_idx,
        npz_path,
        png_path,
        xlim,
        ylim,
        rank_lim,
        title,
        dpi,
        scatter_size,
    ) = args

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import numpy as np

    data = np.load(npz_path)
    x, y, rank = data["x"], data["y"], data["rank"]

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(
        x,
        y,
        s=scatter_size,
        c=rank,
        cmap="viridis",
        norm=Normalize(vmin=rank_lim[0], vmax=rank_lim[1]),
        edgecolors="none",
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("rank")
    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(fig)
    return frame_idx
