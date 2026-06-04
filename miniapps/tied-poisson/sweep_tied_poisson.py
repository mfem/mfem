#!/usr/bin/env python3
"""
Sweep tied-poisson solver options and plot PCG iterations vs alpha.

Features:
- CLI for mesh/refinement/tied-attr and sweep settings
- Saves plot to PNG
- Optionally saves CSV
- Compares:
  - AMG
  - AMG one-level
  - AMGF
  - AMGF + iterative filter
- AMGF variants always use one AMG level
- color = solver
- linestyle = diffusion ratio
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class SolverConfig:
    name: str
    color: str
    extra_args: Tuple[str, ...]


SOLVERS: List[SolverConfig] = [
    SolverConfig(
        name="AMG",
        color="tab:blue",
        extra_args=(
            "--no-amgf",
            "--no-one-level-amg",
            "--no-iterative-filter",
        ),
    ),
    #SolverConfig(
    #    name="AMG one-level",
    #    color="tab:orange",
    #    extra_args=(
    #        "--no-amgf",
    #        "--one-level-amg",
    #        "--no-iterative-filter",
    #    ),
    #),
    SolverConfig(
        name="AMGF",
        color="tab:green",
        extra_args=(
            "--amgf",
            #"--one-level-amg",
            "--no-iterative-filter",
        ),
    ),
    SolverConfig(
        name="AMGF w/ iterative filter",
        color="tab:red",
        extra_args=(
            "--amgf",
            #"--one-level-amg",
            "--iterative-filter",
        ),
    ),
][:-1]

LINESTYLES = ["-", "--", "-.", ":"]
MARKERS = [".", "^", "+", "D", "v"]

ITER_RE = re.compile(r"Iteration\s*:\s*(\d+)\s+\(B r, r\)\s*=")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep tied-poisson and plot PCG iterations vs alpha."
    )

    parser.add_argument("--exe", default="./tied-poisson", help="Path to tied-poisson executable")
    parser.add_argument("--mesh", default="../../data/beam-tri.mesh", help="Mesh file")
    parser.add_argument("--refine", type=int, default=4, help="Uniform refinement level")
    parser.add_argument("--tied-attr", type=int, default=3, help="Boundary attribute to tie")
    parser.add_argument("--pcg-max-iters", type=int, default=10000, help="Max PCG iterations")
    parser.add_argument("--visualization", action="store_true", help="Enable visualization")

    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000],
        help="Alpha values to sweep",
    )
    parser.add_argument(
        "--diffusion-ratios",
        type=float,
        nargs="+",
        default=[1, 10, 100, 1000],
        help="Diffusion ratios to sweep",
    )

    parser.add_argument("--title", default="PCG iterations vs alpha", help="Plot title")
    parser.add_argument("--output", default="pcg_iterations_vs_alpha.png", help="PNG output file")
    parser.add_argument("--csv", default=None, help="Optional CSV output file")
    parser.add_argument("--dpi", type=int, default=150, help="PNG DPI")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")

    parser.add_argument(
        "--print-failed-stdout",
        action="store_true",
        help="Print stdout/stderr when parsing fails",
    )

    return parser.parse_args()


def parse_final_iteration(output: str) -> Optional[int]:
    matches = ITER_RE.findall(output)
    if not matches:
        return None
    return int(matches[-1])


def build_command(
    executable: str,
    mesh: str,
    refine: int,
    tied_attr: int,
    alpha: float,
    diffusion_ratio: float,
    pcg_max_iters: int,
    visualization: bool,
    solver: SolverConfig,
) -> List[str]:
    cmd = [
        "mpirun",
        "-np",
        "1",
        executable,
        "--mesh", mesh,
        "--refine", str(refine),
        "--tied-attr", str(tied_attr),
        "--alpha", f"{alpha:g}",
        "--diffusion-ratio", f"{diffusion_ratio:g}",
        "--pcg-max-iters", str(pcg_max_iters),
    ]

    cmd.append("--visualization" if visualization else "--no-visualization")
    cmd.extend(solver.extra_args)
    return cmd


def run_case(
    executable: str,
    mesh: str,
    refine: int,
    tied_attr: int,
    alpha: float,
    diffusion_ratio: float,
    pcg_max_iters: int,
    visualization: bool,
    solver: SolverConfig,
    print_failed_stdout: bool,
) -> int:
    cmd = build_command(
        executable=executable,
        mesh=mesh,
        refine=refine,
        tied_attr=tied_attr,
        alpha=alpha,
        diffusion_ratio=diffusion_ratio,
        pcg_max_iters=pcg_max_iters,
        visualization=visualization,
        solver=solver,
    )

    print("Running:", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    combined_output = (result.stdout or "") + "\n" + (result.stderr or "")
    iters = parse_final_iteration(combined_output)

    if iters is None:
        print(f"\nERROR: Could not parse iteration count for command:\n  {' '.join(cmd)}", file=sys.stderr)
        print(f"Return code: {result.returncode}", file=sys.stderr)
        if print_failed_stdout:
            print("\n--- STDOUT ---", file=sys.stderr)
            print(result.stdout, file=sys.stderr)
            print("\n--- STDERR ---", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
        raise RuntimeError("Failed to parse PCG iteration count.")

    return iters


def save_csv(
    csv_path: str,
    alphas: np.ndarray,
    diffusion_ratios: List[float],
    solvers: List[SolverConfig],
    results: Dict[Tuple[str, float], List[int]],
) -> None:
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["solver", "diffusion_ratio", "alpha", "pcg_iterations"])

        for dr in diffusion_ratios:
            for solver in solvers:
                y = results[(solver.name, dr)]
                for alpha, iters in zip(alphas, y):
                    writer.writerow([solver.name, dr, alpha, iters])


from matplotlib.lines import Line2D


def plot_results(
    alphas: np.ndarray,
    diffusion_ratios: List[float],
    solvers: List[SolverConfig],
    results: Dict[Tuple[str, float], List[int]],
    title: str,
    output_png: str,
    dpi: int,
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, dr in enumerate(diffusion_ratios):
        linestyle = LINESTYLES[i % len(LINESTYLES)]
        marker = MARKERS[i % len(MARKERS)]
        for solver in solvers:
            y = results[(solver.name, dr)]
            ax.plot(
                alphas,
                y,
                marker=marker,
                color=solver.color,
                linestyle=linestyle,
                linewidth=2,
                markersize=5,
            )

    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("alpha")
    ax.set_ylabel("PCG iterations")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)

    solver_handles = [
        Line2D(
            [0], [0],
            color=solver.color,
            linestyle="-",
            linewidth=2,
            marker="o",
            markersize=5,
            label=solver.name,
        )
        for solver in solvers
    ]

    diffusion_handles = [
        Line2D(
            [0], [0],
            color="black",
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            linewidth=2,
            marker=MARKERS[i % len(MARKERS)],
            markersize=5,
            label=f"d={dr:g}",
        )
        for i, dr in enumerate(diffusion_ratios)
    ]

    legend1 = ax.legend(
        handles=solver_handles,
        title="Solver",
        loc="upper left",
    )
    ax.add_artist(legend1)

    ax.legend(
        handles=diffusion_handles,
        title="Diffusion ratio",
        loc="upper right",
    )

    fig.tight_layout()
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    print(f"Saved plot to: {output_png}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()

    exe_path = Path(args.exe)
    if not exe_path.exists():
        raise FileNotFoundError(f"Executable not found: {args.exe}")

    alphas = np.array(args.alphas, dtype=float)
    diffusion_ratios = list(args.diffusion_ratios)

    results: Dict[Tuple[str, float], List[int]] = {}

    for dr in diffusion_ratios:
        for solver in SOLVERS:
            curve: List[int] = []
            for alpha in alphas:
                iters = run_case(
                    executable=args.exe,
                    mesh=args.mesh,
                    refine=args.refine,
                    tied_attr=args.tied_attr,
                    alpha=float(alpha),
                    diffusion_ratio=float(dr),
                    pcg_max_iters=args.pcg_max_iters,
                    visualization=args.visualization,
                    solver=solver,
                    print_failed_stdout=args.print_failed_stdout,
                )
                curve.append(iters)
            results[(solver.name, dr)] = curve

    plot_results(
        alphas=alphas,
        diffusion_ratios=diffusion_ratios,
        solvers=SOLVERS,
        results=results,
        title=args.title,
        output_png=args.output,
        dpi=args.dpi,
        show=args.show,
    )

    if args.csv:
        save_csv(
            csv_path=args.csv,
            alphas=alphas,
            diffusion_ratios=diffusion_ratios,
            solvers=SOLVERS,
            results=results,
        )
        print(f"Saved CSV to: {args.csv}")


if __name__ == "__main__":
    main()
