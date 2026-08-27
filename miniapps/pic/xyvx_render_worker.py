"""Worker for parallel x-y-v_x frame rendering.

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
        vxlim,
        rho_lim,
        time,
        dpi,
        scatter_size,
        scatter_alpha,
    ) = args

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data = np.load(npz_path)
    x, y, vx, rho = data["x"], data["y"], data["vx"], data["rho"]

    # Stretch v_x relative to its data span (not a taller empty figure).
    vx_stretch = 4.0
    dx = max(xlim[1] - xlim[0], 1e-12)
    dy = max(ylim[1] - ylim[0], 1e-12)
    dv = max(vxlim[1] - vxlim[0], 1e-12)

    fig = plt.figure(figsize=(20, 14))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        x,
        y,
        vx,
        s=scatter_size,
        alpha=scatter_alpha,
        c=rho,
        cmap="inferno",
        vmin=rho_lim[0],
        vmax=rho_lim[1],
        edgecolors="none",
        depthshade=False,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*vxlim)
    ax.set_box_aspect((dx, dy, dv * vx_stretch))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$v_x$")
    # Low elevation: camera near the floor, looking across x-y.
    ax.view_init(elev=4, azim=-60)
    ax.set_title(rf"$x$–$y$–$v_x$  |  $t = {time:.4g}$")
    fig.colorbar(
        sc,
        ax=ax,
        pad=0.04,
        shrink=0.65,
        label=r"phase-space density $f(x,y,v_x)$",
    )
    # Crop figure whitespace so the 3D box fills the frame.
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return frame_idx
