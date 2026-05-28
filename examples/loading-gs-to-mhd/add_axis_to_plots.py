# %%
"""
Add r/z axis labels to existing ParaView field PNGs in png_output/.

Uses matplotlib only (no ParaView). Bounds match paraview_vis.ipynb camera:
  ResetCamera on mesh, Zoom(3.7), shift r by -1.75 and z by +0.25.
"""
import argparse
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


# Mesh-centered camera after ResetCamera (same for all fields on new_2d_mesh_iter).
_MESH_CENTER_R = 7.996205806732178
_MESH_CENTER_Z = 0.0
_CAMERA_DISTANCE = 69.10947113661028
_INITIAL_VIEW_ANGLE_DEG = 30.0
_ZOOM_FACTOR = 3.7
_SHIFT_R = -1.75
_SHIFT_Z = 0.25

# Field screenshots from paraview_vis.ipynb
_FIELD_IMAGE_SIZE = (1000, 1400)


def zoomed_view_bounds(width, height, zoom_factor=_ZOOM_FACTOR):
    """World-space (r_min, r_max, z_min, z_max) for the standard zoomed field view."""
    focal_r = _MESH_CENTER_R + _SHIFT_R
    focal_z = _MESH_CENTER_Z + _SHIFT_Z
    view_angle = math.radians(_INITIAL_VIEW_ANGLE_DEG / zoom_factor)
    half_height = _CAMERA_DISTANCE * math.tan(view_angle / 2.0)
    half_width = half_height * (width / height)
    return [
        focal_r - half_width,
        focal_r + half_width,
        focal_z - half_height,
        focal_z + half_height,
    ]


def add_axes_to_png(
    image_path,
    output_path=None,
    bounds=None,
    xlabel="r",
    ylabel="z",
):
    """
    Add spatial axis labels to an existing PNG.

    Args:
        image_path: Input PNG path
        output_path: Output path (default: overwrite image_path)
        bounds: [r_min, r_max, z_min, z_max]; inferred from image size if None
        xlabel, ylabel: Axis titles
    """
    image_path = Path(image_path)
    output_path = Path(output_path) if output_path else image_path

    with Image.open(image_path) as img:
        width, height = img.size

    if bounds is None:
        bounds = zoomed_view_bounds(width, height)

    fig_w = width / 100.0
    fig_h = height / 100.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    # ParaView PNGs have row 0 at the top; origin='upper' matches that layout.
    ax.imshow(plt.imread(image_path), extent=bounds, origin="upper", aspect="auto")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(labelsize=10)
    fig.patch.set_facecolor("white")
    out = Path(output_path)
    tmp = out.with_name(out.stem + "_tmp" + out.suffix) if out.resolve() == image_path.resolve() else out
    fig.savefig(tmp, dpi=100, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    if tmp != out:
        tmp.replace(out)
    print(f"Wrote {out}  bounds={[round(b, 3) for b in bounds]}")


def should_add_axes(filename):
    """Skip colorbars and mesh-only figures."""
    name = filename.lower()
    if "colorbar" in name:
        return False
    if name.startswith("mesh_view"):
        return False
    return True


def process_directory(
    input_dir="png_output",
    output_dir=None,
    zoom_factor=_ZOOM_FACTOR,
    overwrite=True,
):
    """
    Add axes to all eligible PNGs in a directory.

    Args:
        input_dir: Folder containing ParaView PNG exports
        output_dir: If set, write labeled PNGs here; else overwrite inputs
        zoom_factor: Must match the Zoom() used when creating the PNGs
        overwrite: When output_dir is None, replace files in input_dir
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else None

    png_files = sorted(input_dir.glob("*.png"))
    if not png_files:
        print(f"No PNG files in {input_dir}")
        return

    for png_path in png_files:
        if not should_add_axes(png_path.name):
            print(f"Skip {png_path.name}")
            continue

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / png_path.name
        elif overwrite:
            out_path = png_path
        else:
            out_path = png_path.with_name(png_path.stem + "_axes.png")

        with Image.open(png_path) as img:
            w, h = img.size

        bounds = zoomed_view_bounds(w, h, zoom_factor=zoom_factor)
        add_axes_to_png(png_path, out_path, bounds=bounds)


def main():
    parser = argparse.ArgumentParser(
        description="Add r/z axes to ParaView field PNGs in png_output/"
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        default="png_output",
        help="Directory of input PNG files",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory (default: overwrite inputs)",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=_ZOOM_FACTOR,
        help="Camera zoom used when the PNGs were created",
    )
    parser.add_argument(
        "images",
        nargs="*",
        help="Optional specific PNG paths (instead of whole directory)",
    )
    args = parser.parse_args()

    if args.images:
        for path in args.images:
            with Image.open(path) as img:
                w, h = img.size
            bounds = zoomed_view_bounds(w, h, zoom_factor=args.zoom)
            out = args.output_dir
            if out and len(args.images) == 1:
                out_path = Path(out) / Path(path).name
                Path(out).mkdir(parents=True, exist_ok=True)
            elif out:
                out_path = Path(out) / Path(path).name
                Path(out).mkdir(parents=True, exist_ok=True)
            else:
                out_path = None
            add_axes_to_png(path, output_path=out_path, bounds=bounds)
    else:
        process_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            zoom_factor=args.zoom,
        )


if __name__ == "__main__":
    main()
