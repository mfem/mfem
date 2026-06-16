#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ITER_RE = re.compile(r"Iteration\s*:\s*(\d+)\s+\(B r, r\)\s*=")
NO_CONV_RE = re.compile(r"PCG:\s*No convergence!")


@dataclass(frozen=True)
class SolverConfig:
    name: str
    linestyle: str
    marker: str
    extra_args: Tuple[str, ...]


@dataclass
class RunResult:
    alpha: float
    subdomain_iters: int
    solver: str
    no_convergence: bool
    pcg_iters: Optional[int]
    returncode: int
    stdout: str
    stderr: str
    cmd: List[str]


SOLVERS: List[SolverConfig] = [
    SolverConfig(
        name="Direct AMGF",
        linestyle="-",
        marker="o",
        extra_args=(
            "--amgf",
            "--no-schwarz-filter",
            "--no-amg-filter",
        ),
    ),
    SolverConfig(
        name="AMGF + Schwarz filter",
        linestyle="--",
        marker="s",
        extra_args=(
            "--amgf",
            "--schwarz-filter",
            "--no-amg-filter",
        ),
    ),
    SolverConfig(
        name="AMGF + AMG filter",
        linestyle="-.",
        marker="^",
        extra_args=(
            "--amgf",
            "--no-schwarz-filter",
            "--amg-filter",
        ),
    ),
]


def parse_final_iteration(output: str) -> Optional[int]:
    matches = ITER_RE.findall(output)
    if not matches:
        return None
    return int(matches[-1])


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    group = parser.add_mutually_exclusive_group()
    dest = name.replace("-", "_")
    group.add_argument(f"--{name}", dest=dest, action="store_true", default=default, help=help_text)
    group.add_argument(f"--no-{name}", dest=dest, action="store_false")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep tied-poisson subdomain PCG iterations and compare AMGF solver variants."
    )

    parser.add_argument("--executable", default="./tied-poisson", help="Path to executable.")
    parser.add_argument("--cwd", default=None, help="Working directory for running the executable.")
    parser.add_argument("--mesh", default="../../data/beam-tet.mesh", help="Mesh file to use.")
    parser.add_argument("--refine", type=int, default=4, help="Number of uniform refinements.")
    parser.add_argument("--tied-attr", type=int, default=1, help="Boundary attribute to tie.")
    parser.add_argument("--separation", type=float, default=0.0, help="Separation distance for visualization.")
    parser.add_argument("--pcg-max-iters", type=int, default=10000, help="Max outer PCG iterations.")
    parser.add_argument("--diffusion-ratio", type=float, default=1.0, help="Fixed diffusion ratio.")
    parser.add_argument(
        "--alphas",
        type=parse_float_list,
        default=[1.0, 1000.0, 1000000.0],
        help="Comma-separated alpha values, example: 1,1000,1000000",
    )
    parser.add_argument(
        "--subdomain-iters",
        type=parse_int_list,
        default=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        help="Comma-separated subdomain PCG max iterations to sweep.",
    )
    parser.add_argument("--output", default="pcg_vs_subdomain_iters.png", help="Output plot filename.")
    parser.add_argument("--csv", default=None, help="Optional CSV output filename.")
    parser.add_argument("--title", default=None, help="Optional plot title.")
    parser.add_argument(
        "--extra-args",
        default="",
        help='Extra raw args to append to executable invocation, example: "--foo 3 --bar".',
    )
    parser.add_argument(
        "--show-failures",
        action="store_true",
        help="Print stdout/stderr for failed or unparsed runs.",
    )

    add_bool_arg(parser, "visualization", False, "Enable visualization.")
    add_bool_arg(parser, "eigenvalues", False, "Enable eigenvalue computation.")
    add_bool_arg(parser, "precondition-subspace-cg", True, "Enable PCG for subspace iterative solver.")
    add_bool_arg(parser, "select-subdomains-from-spectrum", False, "Enable spectral subdomain selection.")
    add_bool_arg(parser, "uniform-alpha", True, "Use uniform alpha weighting.")
    add_bool_arg(parser, "vis-spectrum", False, "Enable spectrum visualization.")
    add_bool_arg(parser, "one-level-amg", False, "Enable one-level AMG.")
    add_bool_arg(parser, "symmetric-tie", False, "Enable symmetric tie.")
    add_bool_arg(parser, "even-weighting", False, "Enable even weighting.")

    return parser


def build_command(
    args: argparse.Namespace,
    alpha: float,
    subdomain_iters: int,
    solver: SolverConfig,
) -> List[str]:
    cmd = [args.executable]

    cmd += ["--mesh", args.mesh]
    cmd += ["--refine", str(args.refine)]
    cmd += ["--alpha", str(alpha)]
    cmd += ["--tied-attr", str(args.tied_attr)]
    cmd += ["--separation", str(args.separation)]
    cmd += ["--pcg-max-iters", str(args.pcg_max_iters)]
    cmd += ["--diffusion-ratio", str(args.diffusion_ratio)]
    cmd += ["--subdomain-pcg-max-iters", str(subdomain_iters)]

    cmd.append("--visualization" if args.visualization else "--no-visualization")
    cmd.append("--eigenvalues" if args.eigenvalues else "--no-eigenvalues")
    cmd.append("--precondition-subspace-cg" if args.precondition_subspace_cg else "--no-precondition-subspace-cg")
    cmd.append(
        "--select-subdomains-from-spectrum"
        if args.select_subdomains_from_spectrum else "--no-select-subdomains-from-spectrum"
    )
    cmd.append("--uniform-alpha" if args.uniform_alpha else "--nonuniform-alpha")
    cmd.append("--vis-spectrum" if args.vis_spectrum else "--no-vis-spectrum")
    cmd.append("--one-level-amg" if args.one_level_amg else "--no-one-level-amg")
    cmd.append("--symmetric-tie" if args.symmetric_tie else "--no-symmetric-tie")
    cmd.append("--even-weighting" if args.even_weighting else "--no-even-weighting")

    cmd += list(solver.extra_args)

    if args.extra_args:
        cmd += shlex.split(args.extra_args)

    return cmd


def run_case(
    args: argparse.Namespace,
    alpha: float,
    subdomain_iters: int,
    solver: SolverConfig,
) -> RunResult:
    cmd = build_command(args, alpha, subdomain_iters, solver)

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=args.cwd,
    )

    combined_output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    pcg_iters = parse_final_iteration(combined_output)
    no_convergence = NO_CONV_RE.search(combined_output) is not None

    return RunResult(
        alpha=alpha,
        subdomain_iters=subdomain_iters,
        solver=solver.name,
        no_convergence=no_convergence,
        pcg_iters=pcg_iters,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        cmd=cmd,
    )


def report_run_issue(result: RunResult, show_failures: bool) -> None:
    print("  Run issue detected.", file=sys.stderr)
    print("  Command:", " ".join(shlex.quote(x) for x in result.cmd), file=sys.stderr)
    print("  Return code:", result.returncode, file=sys.stderr)
    print("  Parsed iterations:", result.pcg_iters, file=sys.stderr)
    print("  No convergence:", result.no_convergence, file=sys.stderr)

    if show_failures:
        print("  STDOUT:", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        print("  STDERR:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)


def save_csv(csv_path: str, results: List[RunResult]) -> None:
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "alpha",
            "subdomain_pcg_max_iters",
            "solver",
            "pcg_iterations",
            "no_convergence",
            "returncode",
        ])
        for r in results:
            writer.writerow([
                r.alpha,
                r.subdomain_iters,
                r.solver,
                r.pcg_iters,
                r.no_convergence,
                r.returncode,
            ])


def plot_results(args: argparse.Namespace, results: List[RunResult]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not color_cycle:
        color_cycle = ["C0", "C1", "C2", "C3", "C4", "C5"]

    alpha_colors: Dict[float, str] = {
        alpha: color_cycle[i % len(color_cycle)]
        for i, alpha in enumerate(args.alphas)
    }

    solver_map = {solver.name: solver for solver in SOLVERS}

    for alpha in args.alphas:
        color = alpha_colors[alpha]

        for solver in SOLVERS:
            subset = [
                r for r in results
                if r.alpha == alpha and r.solver == solver.name and r.pcg_iters is not None
            ]
            subset.sort(key=lambda r: r.subdomain_iters)

            if not subset:
                continue

            xs = [r.subdomain_iters for r in subset]
            ys = [r.pcg_iters for r in subset]

            ax.plot(
                xs,
                ys,
                color=color,
                linestyle=solver.linestyle,
                linewidth=2,
            )

            conv = [r for r in subset if not r.no_convergence]
            if conv:
                ax.plot(
                    [r.subdomain_iters for r in conv],
                    [r.pcg_iters for r in conv],
                    color=color,
                    marker=solver.marker,
                    linestyle="None",
                    markersize=6,
                )

            no_conv = [r for r in subset if r.no_convergence]
            if no_conv:
                ax.plot(
                    [r.subdomain_iters for r in no_conv],
                    [r.pcg_iters for r in no_conv],
                    color=color,
                    marker="x",
                    linestyle="None",
                    markersize=8,
                    markeredgewidth=2,
                )

    ax.set_xlabel("Subdomain PCG max iterations")
    ax.set_ylabel("Outer PCG iterations")
    ax.set_yscale("log")
    ax.set_title(args.title or "PCG iterations vs subdomain PCG max iterations")
    ax.grid(True, alpha=0.3)

    alpha_handles = [
        Line2D(
            [0], [0],
            color=alpha_colors[alpha],
            linestyle="-",
            linewidth=2,
            label=f"alpha={alpha:g}",
        )
        for alpha in args.alphas
    ]

    solver_handles = [
        Line2D(
            [0], [0],
            color="black",
            linestyle=solver.linestyle,
            marker=solver.marker,
            linewidth=2,
            markersize=6,
            label=solver.name,
        )
        for solver in SOLVERS
    ]

    failure_handle = Line2D(
        [0], [0],
        color="black",
        linestyle="None",
        marker="x",
        markersize=8,
        markeredgewidth=2,
        label="No convergence",
    )

    legend1 = ax.legend(handles=alpha_handles, title="Alpha", loc="upper right")
    ax.add_artist(legend1)
    legend2 = ax.legend(handles=solver_handles + [failure_handle], title="Solver", loc="upper left")
    ax.add_artist(legend2)

    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"Saved plot to {args.output}")


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()

    exe_path = Path(args.executable)
    if args.cwd is None and not exe_path.is_absolute() and not exe_path.exists():
        print(f"Warning: executable not found at {args.executable}", file=sys.stderr)

    results: List[RunResult] = []
    total_runs = len(args.alphas) * len(args.subdomain_iters) * len(SOLVERS)
    run_idx = 0

    for alpha in args.alphas:
        for solver in SOLVERS:
            for subdomain_iters in args.subdomain_iters:
                run_idx += 1
                print(
                    f"[{run_idx}/{total_runs}] alpha={alpha:g}, s={subdomain_iters}, solver={solver.name}",
                    flush=True,
                )
                result = run_case(args, alpha, subdomain_iters, solver)
                results.append(result)

                if result.returncode != 0 or result.pcg_iters is None:
                    report_run_issue(result, args.show_failures)

    plot_results(args, results)

    if args.csv:
        save_csv(args.csv, results)
        print(f"Saved CSV to {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())