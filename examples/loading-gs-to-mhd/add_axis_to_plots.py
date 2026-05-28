# %%
"""Add r/z axis labels to ParaView field PNGs (fixed physical size, DPI scales with input)."""
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PNG_DIRS = [
    Path("/Users/rsz/Documents/GT_Research/tds/notes_GS_to_MHD/Paper_Draft_Revise/figs/GS_GS_align_refine"),
    Path("/Users/rsz/Documents/GT_Research/tds/notes_GS_to_MHD/Paper_Draft_Revise/figs/GS_GS_unalign"),
    Path("/Users/rsz/Documents/GT_Research/tds/notes_GS_to_MHD/Paper_Draft_Revise/figs/GS_GS_align"),
    Path("/Users/rsz/Documents/GT_Research/tds/notes_GS_to_MHD/Paper_Draft_Revise/figs/GS_GS_iter"),
]

# Fixed physical size (inches) — same on the page for every figure.
FIG_WIDTH_IN = 5.0
FIG_HEIGHT_IN = 7.0

# Base DPI for standard field screenshots (1000 x 1400 from paraview_vis.ipynb).
# Mesh screenshots (2000 x 2800) get 2x DPI automatically via output_dpi().
BASE_DPI = 300
_REF_WIDTH = 1000
_REF_HEIGHT = 1400

# Camera settings from paraview_vis.ipynb (ResetCamera, Zoom(3.7), shift).
_MESH_CENTER_R = 7.996205806732178
_MESH_CENTER_Z = 0.0
_CAMERA_DISTANCE = 69.10947113661028
_INITIAL_VIEW_ANGLE_DEG = 30.0
_ZOOM_FACTOR = 3.7
_SHIFT_R = -1.75
_SHIFT_Z = 0.25


def output_dpi(width, height):
    """Scale DPI with input resolution so high-res mesh PNGs stay sharp."""
    scale = max(width / _REF_WIDTH, height / _REF_HEIGHT)
    return int(BASE_DPI * scale)


def zoomed_view_bounds(aspect):
    """World-space (r_min, r_max, z_min, z_max); depends on aspect ratio, not pixel count."""
    focal_r = _MESH_CENTER_R + _SHIFT_R
    focal_z = _MESH_CENTER_Z + _SHIFT_Z
    view_angle = math.radians(_INITIAL_VIEW_ANGLE_DEG / _ZOOM_FACTOR)
    half_height = _CAMERA_DISTANCE * math.tan(view_angle / 2.0)
    half_width = half_height * aspect
    return [
        focal_r - half_width,
        focal_r + half_width,
        focal_z - half_height,
        focal_z + half_height,
    ]


def add_axes_to_png(png_path):
    png_path = Path(png_path)
    with Image.open(png_path) as img:
        width, height = img.size

    dpi = output_dpi(width, height)
    aspect = width / height
    bounds = zoomed_view_bounds(aspect)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=dpi)
    ax.imshow(plt.imread(png_path), extent=bounds, origin="upper", aspect="auto")
    ax.set_xlabel("r", fontsize=14)
    ax.set_ylabel("z", fontsize=14)
    ax.tick_params(labelsize=12)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.10, top=0.98)

    out_path = png_path.with_name(png_path.stem + "_ax.pdf")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(
        f"Wrote {out_path}  size={FIG_WIDTH_IN}x{FIG_HEIGHT_IN} in  dpi={dpi}  "
        f"input={width}x{height}  bounds={[round(b, 3) for b in bounds]}"
    )


for PNG_DIR in PNG_DIRS:
    if not PNG_DIR.is_dir():
        print(f"Skip missing dir {PNG_DIR}")
        continue
    for png_path in sorted(PNG_DIR.glob("*.png")):
        name = png_path.name.lower()
        if "colorbar" in name:
            print(f"Skip {png_path.name}")
            continue
        add_axes_to_png(png_path)
