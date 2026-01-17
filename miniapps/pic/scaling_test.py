#!/usr/bin/env python3
"""
scaling_test.py

Automate strong/weak scaling tests for:
  make electrostatic-2d2v && srun -n <ranks> ./electrostatic-2d2v ...

Notes:
- -npt is per rank.
- Measures end-to-end wall time externally (time.time()).
- Stores per-run stdout/stderr logs, plus summary CSV/JSON.

Examples:
  # Weak scaling: keep -npt per rank fixed (default), sweep ranks
  python scaling_test.py --mode weak --ranks 1,2,4,8,16,32,64,128,256

  # Strong scaling: keep total particles fixed at base_total_npt (default = 10240*256)
  python scaling_test.py --mode strong --ranks 16,32,64,128,256

  # Weak scaling AND scale the grid with sqrt(ranks) (optional)
  python scaling_test.py --mode weak --scale-grid

  # Skip build step
  python scaling_test.py --no-build
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple


# ---- Default case parameters (from your command) ----
DEFAULT_CASE_ARGS = [
    "-rdf", "1",
    "-npt", "10240",             # per rank (will be replaced per run)
    "-k", "0.2855993321",
    "-a", "0.05",
    "-nt", "400",
    "-nx", "64",
    "-ny", "64",
    "-O", "1",
    "-q", "0.00009453125",
    "-m", "0.00009453125",
    "-ocf", "1000",
    "-dt", "0.05",
    "-no-vis",
]

DEFAULT_EXE = "./electrostatic-2d2v"
DEFAULT_MAKE_TARGET = "electrostatic-2d2v"


@dataclass
class RunResult:
    timestamp: str
    mode: str
    ranks: int
    npt_per_rank: int
    total_particles: int
    nx: int
    ny: int
    nt: int
    wall_seconds: float
    particle_steps: int
    particle_steps_per_sec: float
    returncode: int
    log_dir: str
    cmd: str


def parse_ranks(s: str) -> List[int]:
    # Allow "1,2,4,8" or "1:256:2" (geometric) style
    s = s.strip()
    if ":" in s:
        # format: start:end:factor  (geometric progression)
        parts = s.split(":")
        if len(parts) != 3:
            raise ValueError("Rank range must be start:end:factor (e.g., 1:256:2)")
        start, end, factor = map(int, parts)
        out = []
        r = start
        while r <= end:
            out.append(r)
            r *= factor
        return out
    return [int(x) for x in s.split(",") if x.strip()]


def build_target(make_target: str) -> None:
    print(f"[build] make {make_target}")
    p = subprocess.run(["make", make_target], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        print(p.stdout)
        raise RuntimeError(f"Build failed: make {make_target} (rc={p.returncode})")


def kv_from_args(args: List[str]) -> Dict[str, str]:
    """Parse simple '-flag value' pairs (and flags like '-no-vis')."""
    d: Dict[str, str] = {}
    i = 0
    while i < len(args):
        tok = args[i]
        if tok.startswith("-"):
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                d[tok] = args[i + 1]
                i += 2
            else:
                d[tok] = "true"
                i += 1
        else:
            i += 1
    return d


def replace_arg(args: List[str], flag: str, value: str) -> List[str]:
    out = args[:]
    i = 0
    while i < len(out):
        if out[i] == flag:
            # if next is value token, replace; else insert
            if i + 1 < len(out) and not out[i + 1].startswith("-"):
                out[i + 1] = value
                return out
            out.insert(i + 1, value)
            return out
        i += 1
    # not present, append
    out.extend([flag, value])
    return out


def run_case(
    exe: str,
    ranks: int,
    case_args: List[str],
    log_root: Path,
    extra_env: Dict[str, str] | None = None,
) -> Tuple[int, float, str, str]:
    """Run srun command, return (rc, wall_seconds, stdout_path, stderr_path)."""
    log_dir = log_root / f"ranks_{ranks}"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "stdout.txt"
    stderr_path = log_dir / "stderr.txt"

    cmd = ["srun", "-n", str(ranks), exe] + case_args

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    t0 = time.time()
    with stdout_path.open("w") as out, stderr_path.open("w") as err:
        p = subprocess.run(cmd, stdout=out, stderr=err, env=env, text=True)
    t1 = time.time()

    return p.returncode, (t1 - t0), str(stdout_path), str(stderr_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["weak", "strong"], default="weak",
                    help="weak: keep -npt per rank fixed; strong: keep total particles fixed")
    ap.add_argument("--ranks", default="1:256:2",
                    help="Comma list (e.g. 1,2,4,8) or geometric start:end:factor (e.g. 1:256:2)")
    ap.add_argument("--exe", default=DEFAULT_EXE)
    ap.add_argument("--make-target", default=DEFAULT_MAKE_TARGET)
    ap.add_argument("--no-build", action="store_true")

    ap.add_argument("--base-npt-per-rank", type=int, default=10240,
                    help="Base -npt per rank (used in weak scaling; also used to define default total for strong)")
    ap.add_argument("--base-total-npt", type=int, default=10240 * 256,
                    help="Total particles to hold fixed for strong scaling (default matches your 256-rank run)")
    ap.add_argument("--min-npt-per-rank", type=int, default=256,
                    help="For strong scaling, lower bound to avoid too tiny per-rank particle counts")

    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--ny", type=int, default=64)
    ap.add_argument("--nt", type=int, default=400)

    ap.add_argument("--scale-grid", action="store_true",
                    help="(weak scaling) scale nx,ny by sqrt(ranks/base_ranks) and keep them powers of two-ish")
    ap.add_argument("--base-ranks", type=int, default=1,
                    help="Reference ranks for grid scaling when --scale-grid is set")

    ap.add_argument("--outdir", default="scaling_results",
                    help="Output directory for logs and summary files")
    ap.add_argument("--repeat", type=int, default=1,
                    help="Repeat each configuration N times (keeps separate logs, records best/median etc. separately per run)")
    ap.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()

    ranks_list = parse_ranks(args.ranks)
    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    if not args.no_build:
        build_target(args.make_target)

    # Prepare base args and force nt/nx/ny from CLI (so metrics compute correctly)
    case_args = DEFAULT_CASE_ARGS[:]
    case_args = replace_arg(case_args, "-nt", str(args.nt))
    case_args = replace_arg(case_args, "-nx", str(args.nx))
    case_args = replace_arg(case_args, "-ny", str(args.ny))

    summary: List[RunResult] = []
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"{args.mode}_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] mode={args.mode} ranks={ranks_list} out={run_dir}")

    for r in ranks_list:
        for rep in range(1, args.repeat + 1):
            rep_dir = run_dir / f"rep_{rep}"
            rep_dir.mkdir(parents=True, exist_ok=True)

            # Determine npt per rank
            if args.mode == "weak":
                npt_per_rank = args.base_npt_per_rank
                total_particles = npt_per_rank * r

                nx, ny = args.nx, args.ny
                if args.scale_grid:
                    scale = math.sqrt(r / max(1, args.base_ranks))
                    nx = max(1, int(round(args.nx * scale)))
                    ny = max(1, int(round(args.ny * scale)))
                    # keep them even (common for domain decompositions)
                    nx += nx % 2
                    ny += ny % 2
                this_args = replace_arg(case_args, "-nx", str(nx))
                this_args = replace_arg(this_args, "-ny", str(ny))
                this_args = replace_arg(this_args, "-npt", str(npt_per_rank))

            else:  # strong
                npt_per_rank = max(args.min_npt_per_rank, args.base_total_npt // r)
                total_particles = npt_per_rank * r
                # Keep grid fixed (typical strong scaling)
                nx, ny = args.nx, args.ny
                this_args = replace_arg(case_args, "-npt", str(npt_per_rank))

            # Compute particle-steps (very rough “work” proxy)
            particle_steps = int(total_particles) * int(args.nt)

            cmd_str = " ".join(map(shlex.quote, ["srun", "-n", str(r), args.exe] + this_args))

            if args.dry_run:
                print(f"[dry-run] {cmd_str}")
                continue

            print(f"[run] ranks={r} rep={rep} npt_per_rank={npt_per_rank} "
                  f"total_particles={total_particles} grid={nx}x{ny}")

            rc, wall, stdout_path, stderr_path = run_case(
                exe=args.exe,
                ranks=r,
                case_args=this_args,
                log_root=rep_dir / "logs",
            )

            psps = (particle_steps / wall) if wall > 0 else 0.0

            summary.append(RunResult(
                timestamp=datetime.now().isoformat(timespec="seconds"),
                mode=args.mode,
                ranks=r,
                npt_per_rank=npt_per_rank,
                total_particles=total_particles,
                nx=nx,
                ny=ny,
                nt=args.nt,
                wall_seconds=wall,
                particle_steps=particle_steps,
                particle_steps_per_sec=psps,
                returncode=rc,
                log_dir=str((rep_dir / "logs" / f"ranks_{r}").resolve()),
                cmd=cmd_str,
            ))

            if rc != 0:
                print(f"[warn] run failed (rc={rc}). See logs:\n  {stdout_path}\n  {stderr_path}")

    # Write summary files
    json_path = run_dir / "summary.json"
    csv_path = run_dir / "summary.csv"

    with json_path.open("w") as f:
        json.dump([asdict(x) for x in summary], f, indent=2)

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(summary[0]).keys()) if summary else [])
        if summary:
            w.writeheader()
            for row in summary:
                w.writerow(asdict(row))

    print(f"[done] wrote {json_path} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())