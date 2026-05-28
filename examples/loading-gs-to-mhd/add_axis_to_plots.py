# %%
"""Add r/z axis labels to ParaView field PNGs in png_output/ (overwrites in place)."""
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PNG_DIR = Path("png_output")

# Camera settings from paraview_vis.ipynb (ResetCamera, Zoom(3.7), shift).
_MESH_CENTER_R = 7.996205806732178
_MESH_CENTER_Z = 0.0
_CAMERA_DISTANCE = 69.10947113661028
_INITIAL_VIEW_ANGLE_DEG = 30.0
_ZOOM_FACTOR = 3.7
_SHIFT_R = -1.75
_SHIFT_Z = 0.25


def zoomed_view_bounds(width, height):
    focal_r = _MESH_CENTER_R + _SHIFT_R
    focal_z = _MESH_CENTER_Z + _SHIFT_Z
    view_angle = math.radians(_INITIAL_VIEW_ANGLE_DEG / _ZOOM_FACTOR)
    half_height = _CAMERA_DISTANCE * math.tan(view_angle / 2.0)
    half_width = half_height * (width / height)
    return [focal_r - half_width, focal_r + half_width,
            focal_z - half_height, focal_z + half_height]


def add_axes_to_png(png_path):
    png_path = Path(png_path)
    width, height = Image.open(png_path).size
    bounds = zoomed_view_bounds(width, height)

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.imshow(plt.imread(png_path), extent=bounds, origin="upper", aspect="auto")
    ax.set_xlabel("r", fontsize=12)
    ax.set_ylabel("z", fontsize=12)
    ax.tick_params(labelsize=10)
    fig.patch.set_facecolor("white")

    fig.savefig(str(png_path).replace(".png", "_ax.pdf"), dpi=100, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Wrote {png_path}  bounds={[round(b, 3) for b in bounds]}")


for png_path in sorted(PNG_DIR.glob("*.png")):
    name = png_path.name.lower()
    if "colorbar" in name or name.startswith("mesh_view"):
        print(f"Skip {png_path.name}")
        continue
    add_axes_to_png(png_path)
