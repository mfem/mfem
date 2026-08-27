"""Worker for parallel x-v_x frame rendering (2x3 panel layout).

Kept as a standalone module so ProcessPoolExecutor can pickle it from
Jupyter (notebook __main__ functions are not picklable under spawn).
"""

from __future__ import annotations

# y-slice fractions of L for panels 1-5 (panel 0 is the full domain).
Y_SLICE_FRACS = (
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 1.0),
)


def _slice_mask(y: "np.ndarray", y0: float, L: float, lo_frac: float, hi_frac: float):
    import numpy as np

    lo = y0 + lo_frac * L
    hi = y0 + hi_frac * L
    if hi_frac >= 1.0:
        return (y >= lo) & (y <= hi)
    return (y >= lo) & (y < hi)


def _phase_density(x, vx, x_edges, vx_edges, mass: float, y_extent: float):
    """Estimate f(x,y,v_x) using the y extent of the selected particles."""
    import numpy as np

    hist, _, _ = np.histogram2d(x, vx, bins=(x_edges, vx_edges))
    dx = float(x_edges[1] - x_edges[0])
    dv = float(vx_edges[1] - vx_edges[0])
    bin_vol = max(dx * dv * max(y_extent, 1e-30), 1e-30)
    dens = hist * (mass / bin_vol)
    x_idx = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, len(x_edges) - 2)
    vx_idx = np.clip(np.searchsorted(vx_edges, vx, side="right") - 1, 0, len(vx_edges) - 2)
    return dens[x_idx, vx_idx]


def render_one_frame(args):
    """Load one cached frame and write a PNG with a 2x3 panel layout."""
    (
        frame_idx,
        npz_path,
        png_path,
        xlim,
        vxlim,
        f_lim,
        y0,
        L,
        mass,
        x_edges,
        vx_edges,
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
    x, y, vx = data["x"], data["y"], data["vx"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes_flat = axes.ravel()

    panel_specs = [("all", None, None)] + [
        (f"slice{i}", lo, hi) for i, (lo, hi) in enumerate(Y_SLICE_FRACS)
    ]

    sc_ref = None
    for ax, (tag, lo_frac, hi_frac) in zip(axes_flat, panel_specs):
        if lo_frac is None:
            mask = np.ones_like(y, dtype=bool)
            subtitle = r"all $y$"
        else:
            mask = _slice_mask(y, y0, L, lo_frac, hi_frac)
            if hi_frac >= 1.0:
                subtitle = rf"${lo_frac:.1f}L \leq y \leq L$"
            else:
                subtitle = rf"${lo_frac:.1f}L \leq y < {hi_frac:.1f}L$"

        x_p = x[mask]
        vx_p = vx[mask]
        if x_p.size == 0:
            f_p = np.array([], dtype=np.float32)
        else:
            if lo_frac is None:
                y_extent = L
            else:
                y_extent = (hi_frac - lo_frac) * L
            f_p = _phase_density(x_p, vx_p, x_edges, vx_edges, mass, y_extent)

        sc = ax.scatter(
            x_p,
            vx_p,
            s=scatter_size,
            alpha=scatter_alpha,
            c=f_p,
            cmap="inferno",
            vmin=f_lim[0],
            vmax=f_lim[1],
            edgecolors="none",
        )
        sc_ref = sc
        ax.set_xlim(*xlim)
        ax.set_ylim(*vxlim)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$v_x$")
        ax.set_title(subtitle, fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle(rf"$x$–$v_x$  |  $t = {time:.4g}$", fontsize=14)
    if sc_ref is not None:
        fig.colorbar(
            sc_ref,
            ax=axes_flat.tolist(),
            shrink=0.85,
            pad=0.02,
            label=r"phase-space density $f(x,y,v_x)$",
        )

    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(fig)
    return frame_idx
