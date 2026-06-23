#!/usr/bin/env python3

import argparse
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt


ITER_RE = re.compile(r"Iteration\s*:\s*(\d+)\s+\(B r, r\)\s*=")
NO_CONV_RE = re.compile(r"PCG:\s*No convergence!")


def parse_final_iteration(output: str) -> Optional[int]:
    matches = ITER_RE.findall(output)
    if not matches:
        return None
    return int(matches[-1])


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


@dataclass
class RunResult:
    alpha: float
    subdomain_iters: int
    schwarz_filter: bool
    iterative_filter: bool
    no_convergence: bool
    pcg_iters: Optional[int]
    returncode: int
    stdout: str
    stderr: str
    cmd: List[str]


def build_command(
    args: argparse.Namespace,
    alpha: float,
    subdomain_iters: int,
    schwarz_filter: bool,
    iterative_filter: Optional[bool] = None,
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
    cmd.append("--amgf" if args.amgf else "--no-amgf")

    use_iterative_filter = args.iterative_filter if iterative_filter is None else iterative_filter
    #cmd.append("--schwarz-filter" if use_iterative_filter else "--no-schwarz-filter")

    cmd.append("--one-level-amg" if args.one_level_amg else "--no-one-level-amg")
    cmd.append("--symmetric-tie" if args.symmetric_tie else "--no-symmetric-tie")
    cmd.append("--even-weighting" if args.even_weighting else "--no-even-weighting")
    cmd.append("--schwarz-filter" if schwarz_filter or iterative_filter else "--no-schwarz-filter")
    cmd.append("--precondition-subspace-cg" if schwarz_filter else "--no-precondition-subspace-cg")

    if args.extra_args:
        cmd += shlex.split(args.extra_args)

    return cmd


def run_case(
    args: argparse.Namespace,
    alpha: float,
    subdomain_iters: int,
    schwarz_filter: bool,
    iterative_filter: Optional[bool] = None,
) -> RunResult:
    cmd = build_command(args, alpha, subdomain_iters, schwarz_filter, iterative_filter=iterative_filter)

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=args.cwd,
    )

    combined_output = proc.stdout + "\n" + proc.stderr
    pcg_iters = parse_final_iteration(combined_output)
    no_convergence = NO_CONV_RE.search(combined_output) is not None
    used_iterative_filter = args.iterative_filter if iterative_filter is None else iterative_filter

    return RunResult(
        alpha=alpha,
        subdomain_iters=subdomain_iters,
        schwarz_filter=schwarz_filter,
        iterative_filter=used_iterative_filter,
        no_convergence=no_convergence,
        pcg_iters=pcg_iters,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        cmd=cmd,
    )


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=name.replace("-", "_"), action="store_true", default=default, help=help_text)
    group.add_argument(f"--no-{name}", dest=name.replace("-", "_"), action="store_false")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep tied-poisson subdomain PCG max iterations and plot outer PCG iterations."
    )

    parser.add_argument("--executable", default="./tied-poisson", help="Path to executable.")
    parser.add_argument("--cwd", default=None, help="Working directory for running the executable.")
    parser.add_argument("--mesh", default="../../data/beam-tet.mesh", help="Mesh file to use.")
    parser.add_argument("--refine", type=int, default=4, help="Number of uniform refinements.")
    parser.add_argument("--tied-attr", type=int, default=1, help="Boundary attribute to tie.")
    parser.add_argument("--separation", type=float, default=0.0, help="Separation distance for visualization.")
    parser.add_argument("--pcg-max-iters", type=int, default=10000, help="Max outer PCG iterations.")
    parser.add_argument("--diffusion-ratio", type=float, default=1.0, help="Diffusion ratio.")
    parser.add_argument(
        "--alphas",
        type=parse_float_list,
        default=[1.0, 1000.0, 1000000.0],
        help="Comma-separated alpha values, example: 1,1000,1000000",
    )
    parser.add_argument(
        "--subdomain-iters",
        type=parse_int_list,
        default=[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        help="Comma-separated subdomain max iters to sweep, example: 0,1,2,5,10,20",
    )
    parser.add_argument(
        "--iterative-filter-off-s",
        type=int,
        default=None,
        help="Subdomain max iters value to use for the one-shot no-iterative-filter test per alpha. Defaults to max(subdomain-iters).",
    )
    parser.add_argument(
        "--output",
        default="pcg_vs_subdomain_iters.png",
        help="Output plot filename.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title.",
    )
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
    add_bool_arg(parser, "amgf", True, "Enable AMG with Filtering.")
    add_bool_arg(parser, "iterative-filter", True, "Enable iterative solver on subspace in AMGF.")
    add_bool_arg(parser, "one-level-amg", False, "Enable one-level AMG.")
    add_bool_arg(parser, "symmetric-tie", False, "Enable symmetric tie.")
    add_bool_arg(parser, "even-weighting", False, "Enable even weighting.")

    return parser


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


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()

    exe_path = Path(args.executable)
    if args.cwd is None and not exe_path.is_absolute() and not exe_path.exists():
        print(f"Warning: executable not found at {args.executable}", file=sys.stderr)

    results: List[RunResult] = []

    iterative_filter_off_s = (
        args.iterative_filter_off_s
        if args.iterative_filter_off_s is not None
        else max(args.subdomain_iters)
    )

    total_runs = len(args.alphas) * len(args.subdomain_iters) * 2 + len(args.alphas)
    run_idx = 0

    for alpha in args.alphas:
        for schwarz_filter in [False, True]:
            for subdomain_iters in args.subdomain_iters:
                run_idx += 1
                label = "schwarz" if schwarz_filter else "no-schwarz"
                print(
                    f"[{run_idx}/{total_runs}] alpha={alpha:g}, s={subdomain_iters}, {label}, iterative-filter={'on' if args.iterative_filter else 'off'}",
                    flush=True,
                )
                result = run_case(
                    args,
                    alpha,
                    subdomain_iters,
                    schwarz_filter,
                    iterative_filter=args.iterative_filter,
                )
                results.append(result)

                if result.returncode != 0 or result.pcg_iters is None:
                    report_run_issue(result, args.show_failures)

        run_idx += 1
        print(
            f"[{run_idx}/{total_runs}] alpha={alpha:g}, s={iterative_filter_off_s}, iterative-filter=off, one-shot reference",
            flush=True,
        )
        result = run_case(
            args,
            alpha,
            iterative_filter_off_s,
            schwarz_filter=False,
            iterative_filter=False,
        )
        results.append(result)

        if result.returncode != 0 or result.pcg_iters is None:
            report_run_issue(result, args.show_failures)

    fig, ax = plt.subplots(figsize=(10, 6))

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not color_cycle:
        color_cycle = ["C0", "C1", "C2", "C3", "C4", "C5"]

    alpha_colors = {
        alpha: color_cycle[i % len(color_cycle)]
        for i, alpha in enumerate(args.alphas)
    }

    style_map = {
        False: {"marker": "o", "linestyle": "-"},
        True: {"marker": "s", "linestyle": "--"},
    }

    for alpha in args.alphas:
        color = alpha_colors[alpha]

        for schwarz_filter in [False, True]:
            subset = [
                r for r in results
                if r.alpha == alpha
                and r.schwarz_filter == schwarz_filter
                and r.iterative_filter == args.iterative_filter
                and r.pcg_iters is not None
            ]
            subset.sort(key=lambda r: r.subdomain_iters)

            label = f"alpha={alpha:g}, {'schwarz' if schwarz_filter else 'no schwarz'}"

            if subset:
                xs = [r.subdomain_iters for r in subset]
                ys = [r.pcg_iters for r in subset]

                # Draw connecting line through all points
                ax.plot(
                    xs,
                    ys,
                    color=color,
                    linestyle=style_map[schwarz_filter]["linestyle"],
                    linewidth=2,
                    label=label,
                )

                # Overlay converged markers
                conv = [r for r in subset if not r.no_convergence]
                if conv:
                    ax.plot(
                        [r.subdomain_iters for r in conv],
                        [r.pcg_iters for r in conv],
                        color=color,
                        marker=style_map[schwarz_filter]["marker"],
                        linestyle="None",
                        markersize=6,
                    )

                # Overlay non-converged markers
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

        baseline = [
            r for r in results
            if r.alpha == alpha
            and not r.iterative_filter
        ]
        baseline.sort(key=lambda r: r.subdomain_iters)

        by = [r.pcg_iters for r in baseline if r.pcg_iters is not None]

        if by:
            ax.axhline(
                y=by[0],
                color=color,
                linestyle=":",
                linewidth=2,
                alpha=0.8,
                label=f"alpha={alpha:g}, direct filter",
            )

    ax.set_xlabel("Subdomain PCG max iterations (-s)")
    ax.set_ylabel("Outer PCG iterations")
    ax.set_yscale("log")
    ax.set_title(args.title or "PCG iterations vs subdomain PCG max iterations")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"Saved plot to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())