#!/usr/bin/env python3
"""Check empty contact, self-contact onset, and unloading of a strip or ring.

Run after building contact. The optional launcher accepts a command prefix:
  --launcher './run-contact-bound.sh --ranks 4 --cores 4 --'
"""

import argparse
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=("strip", "ring"), default="strip")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--launcher", default="")
    parser.add_argument("--solver", default="mumps")
    parser.add_argument("--gpu-aware-mpi", action="store_true")
    parser.add_argument("--loading-steps", type=int, default=4)
    parser.add_argument("--unloading-steps", type=int, default=2)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.loading_steps < 1 or args.unloading_steps < 1:
        parser.error("Loading and unloading step counts must be positive")
    app_dir = Path(__file__).resolve().parent
    output = args.output_dir or Path(tempfile.mkdtemp(prefix="self-contact-test-"))
    output.mkdir(parents=True, exist_ok=True)
    prefix = shlex.split(args.launcher) or [str(app_dir / "contact")]
    ring = args.problem == "ring"
    label = "Ring" if ring else "Folded strip"
    pressure = "-ring-pressure" if ring else "-u-pressure"
    common = ["-prob", "5" if ring else "4", "-sr", "0", "-pr", "0",
              "-d", args.device, "-no-vis", "-no-paraview", "-pl", "2"]
    if args.gpu_aware_mpi:
        common.append("-hypre-gpu-aware-mpi")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               SPINDLE_FLUXOPT="disable")
    env["MPICH_GPU_SUPPORT_ENABLED"] = "1" if args.gpu_aware_mpi else "0"

    def run(name, options, steps):
        command = prefix + common + options
        log = output / (name + ".log")
        with log.open("w") as stream:
            stream.write(shlex.join(command) + "\n")
            stream.flush()
            # Allow the larger strip mesh time for all geometry updates.
            result = subprocess.run(command, cwd=app_dir, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT,
                                    timeout=600 if ring else 1200,
                                    check=False)
        text = log.read_text()
        if result.returncode:
            raise RuntimeError(f"{name} exited {result.returncode}; see {log}")
        if f"Converged load steps             = {steps} / {steps}" not in text:
            raise AssertionError(f"Incomplete loading sequence; see {log}")
        updates = [float(v) for v in re.findall(
            r"Final displacement update\s*=\s*(\S+)", text)]
        if len(updates) != steps or any(not math.isfinite(v) or v > 1e-6
                                        for v in updates):
            raise AssertionError(f"Geometry updates did not converge; see {log}")
        print(f"PASS {name}: {log}", flush=True)
        return text

    # Success with unavailable cuDSS verifies that the empty contact subspace
    # never constructs or validates the filtered direct solver on CPU/HIP.
    empty_options = ["-nsteps", "1", pressure, "0.005" if ring else "0.0001"]
    filtered = run("empty-amgf", empty_options +
                   ["-amgf", "-amgf-fsolver", "cudss"], 1)
    plain = run("empty-amg", empty_options + ["-no-amgf"], 1)
    if "global contact subspace is empty; using AMG" not in filtered:
        raise AssertionError("The empty-contact AMG fallback was not exercised")
    if "gap rows = 0, minimum mortar gap = n/a" not in filtered:
        raise AssertionError("The small-pressure test unexpectedly has contact rows")
    if not re.search(r"Total Newton solves\s*=\s*[1-9]", filtered):
        raise AssertionError("The empty-contact test did not exercise a linear solve")
    energy_pattern = r"Final Energy objective\s*=\s*(\S+)"
    e_filtered = float(re.findall(energy_pattern, filtered)[-1])
    e_plain = float(re.findall(energy_pattern, plain)[-1])
    if not math.isclose(e_filtered, e_plain, rel_tol=1e-7, abs_tol=1e-10):
        raise AssertionError("Empty-contact AMGF and plain AMG energies disagree")

    steps = args.loading_steps + args.unloading_steps
    cycle = run("loading-unloading", ["-nsteps", str(args.loading_steps),
                "-msteps", str(args.unloading_steps), "-amgf",
                "-amgf-fsolver", args.solver], steps)
    gaps = re.findall(re.escape(label) +
                      r": gap rows = (\d+), minimum mortar gap = (.*)", cycle)
    finite_gaps = [float(gap) for rows, gap in gaps if int(rows) > 0]
    if len(gaps) != steps or not finite_gaps or min(finite_gaps) > 1e-5:
        raise AssertionError("The loading cycle did not reach self-contact")
    if any(not math.isfinite(g) or g < -2e-6 for g in finite_gaps):
        raise AssertionError("The cycle has an invalid or penetrated mortar gap")
    if int(gaps[-1][0]):
        raise AssertionError("The surfaces did not separate on full unloading")
    final_energy = float(re.findall(energy_pattern, cycle)[-1])
    if not math.isfinite(final_energy) or abs(final_energy) > 1e-8:
        raise AssertionError("Unloading did not recover the unloaded elastic state")
    print(f"PASS self-contact onset and release; logs in {output}", flush=True)


if __name__ == "__main__":
    main()
