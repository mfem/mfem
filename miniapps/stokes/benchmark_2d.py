#!/usr/bin/env python3
"""Sequential, resumable screening and refinement of the smooth 2D Stokes baseline."""

import argparse
import collections
import contextlib
import csv
import datetime
import fcntl
import hashlib
import html
import json
import math
import os
from pathlib import Path
import platform
import random
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import time


SCHEMA_VERSION = 2
SOURCE = Path(__file__).resolve().parent
WORKSPACE = SOURCE.parents[2]
METHODS = ("pspg", "gls", "cip", "taylor-hood")
SOLVERS = ("gmres", "fgmres", "mumps")
PRECONDITIONERS = (
    "block-diagonal-amg", "block-diagonal-p1-amg", "block-upper-p1-amg",
    "block-lower-p1-amg", "block-diagonal-p1-amg-assembled",
    "block-upper-p1-amg-assembled", "block-lower-p1-amg-assembled",
)
TIMINGS = (
    "assembly_seconds", "preconditioner_setup_seconds", "direct_setup_seconds",
    "linear_solve_seconds", "finalization_seconds", "solve_seconds",
    "boundary_projection_seconds", "run_wall_seconds",
)
ERRORS = ("velocity_l2", "velocity_h1", "pressure_l2", "divergence_l2", "boundary_l2_error")
BOUNDS = {
    "velocity_l2": 0.001, "velocity_h1": 0.05, "pressure_l2": 0.02,
    "boundary_l2_error": 0.01, "boundary_dof_error": 1e-8,
    "boundary_flux": 1e-8, "gauge_multiplier": 1e-8, "projection_correction_l2": 0.01,
}
REFINEMENT_ORDERS = {
    "velocity_l2": 3, "velocity_h1": 2, "pressure_l2": 2,
    "boundary_l2_error": 3, "projection_correction_l2": 3,
}
INTEGER_FIELDS = (
    "dimension", "level", "periodic", "internal_interface_bcs", "elements",
    "velocity_dofs", "pressure_dofs", "outer_iterations", "converged", "geometry_order",
    "integration_order", "pressure_nullspace_detected", "pressure_gauge_active",
    "assembly_performed", "order", "pressure_order",
)
FLOAT_FIELDS = tuple(dict.fromkeys(TIMINGS[:-1] + ERRORS + tuple(BOUNDS) + (
    "relative_residual", "absolute_residual", "hole_velocity_l2", "projection_flux_before",
)))
STRING_FIELDS = ("method", "solver", "preconditioner", "mms_case", "pressure_gauge")


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def manifest_fingerprint(record):
    stable = json.loads(json.dumps(record))
    for field in ("fingerprint", "created_utc"):
        stable.pop(field, None)
    stable["provenance"].pop("linked_libraries", None)
    return digest(stable)


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def command_output(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"


def provenance(executable, mesh, launcher, environment):
    sources = sorted(set(SOURCE.glob("stokes_*.cpp")) | set(SOURCE.glob("stokes_*.hpp")) |
                     {Path(__file__).resolve(), SOURCE / "test_benchmark_2d.py", SOURCE / "CMakeLists.txt"})
    build_files = [WORKSPACE / "build/CMakeCache.txt", WORKSPACE / "build/config/_config.hpp",
                   WORKSPACE / "build/miniapps/stokes/CMakeFiles/stokes_mms.dir/flags.make"]
    selected_environment = {
        key: value for key, value in environment.items()
        if key in {"PATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "PETSC_DIR", "PETSC_ARCH"} or
        key.startswith(("OMP_", "OPENBLAS_", "MKL_", "BLIS_", "VECLIB_", "HYDRA_",
                        "I_MPI_", "OMPI_", "MPICH_", "MV2_", "HYPRE_"))
    }
    allocation = {key: environment[key] for key in (
        "SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_JOB_NUM_NODES", "SLURM_CPUS_ON_NODE",
        "SLURM_NTASKS", "SLURM_JOB_PARTITION", "SLURM_JOB_CPUS_PER_NODE", "PBS_JOBID",
        "PBS_NODEFILE", "LSB_JOBID", "FLUX_URI",
    ) if key in environment}
    launcher_path = shutil.which(launcher[0], path=environment.get("PATH")) if launcher else None
    libraries = command_output(["ldd", str(executable)])
    library_paths = [Path(token) for line in libraries.splitlines() for token in line.split()
                     if token.startswith("/") and Path(token).is_file()]
    cpu = "\n".join(line for line in command_output(["lscpu"]).splitlines()
                    if "scaling MHz" not in line and not line.startswith("CPU MHz:"))
    return {
        "executable": {"path": str(executable), "sha256": file_hash(executable)},
        "mesh": {"path": str(mesh), "sha256": file_hash(mesh)},
        "launcher": {"tokens": launcher, "path": launcher_path,
                     "sha256": file_hash(launcher_path) if launcher_path else None},
        "source_revision": command_output(["git", "-C", str(SOURCE.parents[1]), "rev-parse", "HEAD"]),
        "source_status": command_output(["git", "-C", str(SOURCE.parents[1]), "status", "--short"]),
        "source_hashes": {str(path.relative_to(SOURCE)): file_hash(path) for path in sources},
        "build_files": {str(path): file_hash(path) for path in build_files if path.is_file()},
        "build_flags": next((path.read_text() for path in build_files if path.name == "flags.make" and path.is_file()), "unavailable"),
        "linked_libraries": libraries,
        "library_hashes": {str(path): file_hash(path) for path in library_paths},
        "hostname": platform.node(), "platform": platform.platform(),
        "cpu": cpu,
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "allocation": allocation, "environment": selected_environment,
        "python": sys.version,
    }


def configuration_matrix(settings, methods, solvers, preconditioners):
    configurations = []
    for refinement in settings.get("refinement_levels", [1]):
        for method in methods:
            for solver in solvers:
                for preconditioner in (("not-applicable",) if solver == "mumps" else preconditioners):
                    case = {"method": method, "solver": solver, "preconditioner": preconditioner,
                            "serial_refinements": refinement}
                    case["id"] = f"{method}-{solver}-{preconditioner}-r{refinement}-{digest([settings, case])[:12]}"
                    configurations.append(case)
    return configurations


def case_refinement(case, settings):
    return case.get("serial_refinements", settings.get("serial_refinements", 1))


def refinement_bounds(refinement):
    return {field: bound * 2 ** ((1 - refinement) * REFINEMENT_ORDERS.get(field, 0))
            for field, bound in BOUNDS.items()}


def case_settings(case, settings):
    refinement = case_refinement(case, settings)
    effective = dict(settings)
    if "accuracy_bounds_by_refinement" in settings:
        effective["accuracy_bounds"] = settings["accuracy_bounds_by_refinement"][str(refinement)]
    if "input_elements" in settings:
        effective["expected_elements"] = settings["input_elements"] * 4 ** refinement
    return effective


def schedule(configurations, warmups, repetitions, seed):
    generator = random.Random(seed)
    jobs = []
    for warmup, count, prefix in ((True, warmups, "warmup"), (False, repetitions, "measured")):
        for repetition in range(1, count + 1):
            shuffled = list(configurations)
            generator.shuffle(shuffled)
            jobs.extend({"configuration_id": case["id"], "warmup": warmup,
                         "repetition": repetition, "sample_id": f"{prefix}-{repetition:03d}"}
                        for case in shuffled)
    return jobs


def input_element_count(path):
    found = {}
    with path.open() as stream:
        lines = (line.split("#", 1)[0].strip() for line in stream)
        entries = (line for line in lines if line)
        for line in entries:
            if line in {"dimension", "elements"}:
                found[line] = int(next(entries))
                if len(found) == 2:
                    break
    if found.get("dimension") != 2 or found.get("elements", 0) <= 0:
        raise ValueError("Expected a native MFEM 2D mesh with a positive element count")
    return found["elements"]


def make_manifest(args):
    executable, mesh = Path(args.executable).resolve(), Path(args.mesh).resolve()
    if not executable.is_file() or not mesh.is_file():
        raise ValueError("The executable and mesh must exist, including for a dry run")
    launcher = shlex.split(args.launcher)
    if launcher and shutil.which(launcher[0]) is None:
        raise ValueError(f"Launcher not found: {launcher[0]}")
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       BLIS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    settings = {
        "ranks": args.ranks, "relative_tolerance": args.relative_tolerance,
        "absolute_tolerance": args.absolute_tolerance, "restart": args.restart,
        "max_iterations": args.max_iterations, "order": 2, "refinement_levels": sorted(args.refinements),
        "parallel_refinements": 0, "mms_case": "smooth", "variable_viscosity": True,
        "mu": 1, "pressure_gauge": "auto", "p1_smoothing_steps": 2,
        "amg_interp_refine": True, "quadrature_order": 6,
        "pspg_constant": 1, "gls_tau_scale": 1, "cip_gamma": 0.01,
        "accuracy_bounds_by_refinement": {str(level): refinement_bounds(level) for level in sorted(args.refinements)},
        "minimum_hole_velocity_l2": 0.01, "input_elements": input_element_count(mesh),
    }
    configurations = configuration_matrix(settings, args.methods, args.solvers, args.preconditioners)
    record = {
        "schema_version": SCHEMA_VERSION, "settings": settings,
        "protocol": {"warmups": args.warmups, "repetitions": args.repetitions, "seed": args.seed,
                     "timeout_seconds": args.timeout, "execution_class": args.execution_class,
                     "allocation_note": args.allocation_note, "binding_note": args.binding_note},
        "provenance": provenance(executable, mesh, launcher, environment),
        "configurations": configurations,
        "schedule": schedule(configurations, args.warmups, args.repetitions, args.seed),
    }
    record["fingerprint"] = manifest_fingerprint(record)
    record["created_utc"] = utc_now()
    return record, environment


def build_command(manifest, case, csv_path):
    settings = manifest["settings"]
    provenance_record = manifest["provenance"]
    command = [token.replace("{ranks}", str(settings["ranks"]))
               for token in provenance_record["launcher"]["tokens"]]
    command.append(provenance_record["executable"]["path"])
    options = {
        "mesh": provenance_record["mesh"]["path"], "mms-case": "smooth",
        "method": case["method"], "solver": case["solver"],
        "preconditioner": case["preconditioner"] if case["solver"] != "mumps" else "block-diagonal-amg",
        "order": 2, "mu": 1, "pressure-gauge": "auto", "relative-tolerance": settings["relative_tolerance"],
        "absolute-tolerance": settings["absolute_tolerance"], "restart": settings["restart"],
        "max-iterations": settings["max_iterations"], "p1-smoothing-steps": 2,
        "quadrature-order": settings["quadrature_order"], "tau-constant": 1, "gls-tau-scale": 1,
        "cip-gamma": 0.01, "refine-serial": case_refinement(case, settings), "refine-parallel": 0,
        "print-level": 0, "csv": str(csv_path),
    }
    for key, value in options.items():
        command.extend([f"--{key}", str(value)])
    return command + ["--variable-viscosity", "--amg-interp-refine", "--no-paraview"]


def read_driver_csv(path):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError("Duplicate CSV columns")
        required = set(INTEGER_FIELDS + FLOAT_FIELDS + STRING_FIELDS)
        if not required.issubset(fields):
            raise ValueError(f"Missing CSV columns: {sorted(required - set(fields))}")
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one level, got {len(rows)}")
    row = rows[0]
    if None in row or any(value is None for value in row.values()):
        raise ValueError("CSV row length does not match header")
    for field in INTEGER_FIELDS:
        row[field] = int(row[field])
    for field in FLOAT_FIELDS:
        row[field] = float(row[field])
        if not math.isfinite(row[field]):
            raise ValueError(f"Nonfinite {field}")
    return row


def validate_row(row, case, settings):
    settings = case_settings(case, settings)
    expected = {
        "dimension": 2, "level": 0, "periodic": 0, "internal_interface_bcs": 0,
        "method": case["method"], "solver": case["solver"], "mms_case": "smooth",
        "pressure_gauge": "auto", "pressure_nullspace_detected": 1, "pressure_gauge_active": 1,
        "assembly_performed": 1, "geometry_order": 1, "integration_order": 6, "order": 2,
        "pressure_order": 1 if case["method"] == "taylor-hood" else 2,
        "preconditioner": case["preconditioner"] if case["solver"] != "mumps" else "block-diagonal-amg",
    }
    for field, value in expected.items():
        if row[field] != value:
            raise ValueError(f"Wrong {field}: expected {value}, got {row[field]}")
    for field in ("elements", "velocity_dofs", "pressure_dofs"):
        if row[field] <= 0:
            raise ValueError(f"Nonpositive {field}")
    if "expected_elements" in settings and row["elements"] != settings["expected_elements"]:
        raise ValueError("Wrong refined element count")
    if row["converged"] not in (0, 1) or not 0 <= row["outer_iterations"] <= settings["max_iterations"]:
        raise ValueError("Invalid convergence or iteration count")
    signed = {"boundary_flux", "gauge_multiplier", "projection_flux_before"}
    if any(row[field] < 0 for field in FLOAT_FIELDS if field not in signed):
        raise ValueError("Negative norm or time")
    for field in TIMINGS[:5]:
        if row[field] > row["solve_seconds"] * (1 + 1e-12):
            raise ValueError(f"{field} exceeds inclusive solve time")
    active, inactive = (("direct_setup_seconds", "preconditioner_setup_seconds") if case["solver"] == "mumps"
                        else ("preconditioner_setup_seconds", "direct_setup_seconds"))
    if row[active] <= 0 or row[inactive] != 0 or row["boundary_projection_seconds"] <= 0:
        raise ValueError("Incorrect backend/projection timing")
    if case["solver"] == "mumps" and row["outer_iterations"] != 0:
        raise ValueError("MUMPS has a nonzero iteration count")
    failures = []
    if row["converged"] and not (row["absolute_residual"] <= settings["absolute_tolerance"] * (1 + 1e-12) or
                                 row["relative_residual"] <= settings["relative_tolerance"] * (1 + 1e-12)):
        failures.append("true residual exceeds both tolerances")
    for field, bound in settings["accuracy_bounds"].items():
        if abs(row[field]) > bound:
            failures.append(f"{field} exceeds {bound}")
    if row["hole_velocity_l2"] < settings["minimum_hole_velocity_l2"]:
        failures.append("hole trace is not nonzero")
    return failures


def diagnose_exit(returncode, text):
    messages = (
        ("MUMPS is unavailable", "unavailable"),
        ("incompatible net boundary flux", "incompatible_boundary"),
        ("Jacobian", "invalid_geometry"),
        ("out of memory", "resource_limit"),
        ("OUT_OF_MEMORY", "resource_limit"),
        ("std::bad_alloc", "resource_limit"),
    )
    for message, status in messages:
        if message.lower() in text.lower():
            return status
    return "nonconverged" if returncode == 2 else "execution_failed"


def stop_process_group(process):
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def execute_attempt(directory, manifest, case, job, environment):
    directory.mkdir(parents=True, exist_ok=False)
    command = build_command(manifest, case, directory / "driver.csv")
    (directory / "command.txt").write_text(shlex.join(command) + "\n")
    result = {**job, "status": "running", "started_utc": utc_now(), "command": command,
              "hostname": platform.node(), "load_average": list(os.getloadavg()),
              "returncode": None, "diagnostics": [], "row": None}
    write_json(directory / "status.json", result)
    started = time.monotonic()
    process = None
    interrupted = False
    try:
        with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
            process = subprocess.Popen(command, cwd=WORKSPACE, env=environment, stdout=stdout, stderr=stderr,
                                       start_new_session=True)
            result["returncode"] = process.wait(timeout=manifest["protocol"]["timeout_seconds"])
        result["status"] = "completed"
    except subprocess.TimeoutExpired:
        stop_process_group(process)
        result["status"] = "timeout"
        result["returncode"] = process.returncode
    except KeyboardInterrupt:
        if process is not None:
            stop_process_group(process)
            result["returncode"] = process.returncode
        result["status"] = "interrupted"
        interrupted = True
    except OSError as error:
        result["status"] = "execution_failed"
        result["diagnostics"].append(str(error))
    result["run_wall_seconds"] = time.monotonic() - started
    if result["status"] == "completed":
        row = None
        try:
            row = read_driver_csv(directory / "driver.csv")
            result["row"] = row
            result["diagnostics"] = validate_row(row, case, manifest["settings"])
        except (OSError, ValueError) as error:
            result["diagnostics"].append(str(error))
            result["status"] = "invalid_output"
        if result["returncode"] != 0:
            text = "\n".join((directory / name).read_text(errors="replace")[-65536:]
                             for name in ("stdout.log", "stderr.log"))
            result["status"] = diagnose_exit(result["returncode"], text)
            if result["status"] == "nonconverged" and row and row["outer_iterations"] >= manifest["settings"]["max_iterations"]:
                result["status"] = "iteration_limit"
        elif result["status"] != "invalid_output":
            result["status"] = ("nonconverged" if not row["converged"] else
                                "accuracy_failed" if result["diagnostics"] else "success")
    result["finished_utc"] = utc_now()
    result["artifacts"] = {path.name: file_hash(path) for path in directory.iterdir()
                           if path.name in {"command.txt", "stdout.log", "stderr.log", "driver.csv"}}
    write_json(directory / "status.json", result)
    if interrupted:
        raise KeyboardInterrupt
    return result


def attempts_for(root, job):
    return sorted((root / "runs" / job["configuration_id"] / job["sample_id"]).glob("attempt-*"))


def load_attempt(path):
    status_path = path / "status.json"
    if not status_path.exists():
        return {"status": "incomplete", "row": None, "diagnostics": ["No final status"], "returncode": None}
    result = read_json(status_path)
    for name, expected in result.get("artifacts", {}).items():
        artifact = path / name
        if not artifact.is_file() or file_hash(artifact) != expected:
            raise ValueError(f"Modified/missing raw artifact: {artifact}")
    return result


@contextlib.contextmanager
def study_lock(root):
    with (root / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another runner/report process is using this study") from error
        yield


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def statistic_fields(values, field):
    if not values:
        return {f"{field}_{name}": "" for name in ("median", "min", "max", "mad")}
    center = statistics.median(values)
    return {f"{field}_{name}": value for name, value in (
        ("median", center), ("min", min(values)), ("max", max(values)),
        ("mad", statistics.median(abs(value - center) for value in values)),
    )}


def write_refinement_summary(root, summary):
    groups = collections.defaultdict(list)
    for row in summary:
        groups[(row["method"], row["solver"], row["preconditioner"])].append(row)
    comparisons = []
    for key, rows in groups.items():
        rows.sort(key=lambda row: row["serial_refinements"])
        for coarse, fine in zip(rows, rows[1:]):
            eligible = coarse["ranking_eligible"] and fine["ranking_eligible"]
            gap = fine["serial_refinements"] - coarse["serial_refinements"]
            output = dict(zip(("method", "solver", "preconditioner"), key))
            output.update(from_refinement=coarse["serial_refinements"], to_refinement=fine["serial_refinements"],
                          from_configuration=coarse["id"], to_configuration=fine["id"],
                          from_status_counts=coarse["status_counts"], to_status_counts=fine["status_counts"],
                          from_system_dofs=coarse["system_dofs"], to_system_dofs=fine["system_dofs"],
                          comparison_eligible=eligible, execution_class=fine["execution_class"])
            output["dof_ratio"] = fine["system_dofs"] / coarse["system_dofs"] if eligible else ""
            for field in TIMINGS + ("outer_iterations",):
                previous, current = coarse[f"{field}_median"], fine[f"{field}_median"]
                output[f"{field}_ratio"] = current / previous if eligible and previous > 0 else ""
            for field in ERRORS:
                previous, current = coarse[f"{field}_median"], fine[f"{field}_median"]
                output[f"{field}_rate"] = math.log2(previous / current) / gap if eligible and previous > 0 and current > 0 else ""
            comparisons.append(output)
    if comparisons:
        write_csv(root / "refinement.csv", comparisons)
    return comparisons


def summarize(root):
    manifest = read_json(root / "manifest.json")
    if manifest["schema_version"] not in (1, SCHEMA_VERSION):
        raise ValueError("Unsupported manifest schema")
    if manifest["fingerprint"] != manifest_fingerprint(manifest):
        raise ValueError("Manifest content does not match its fingerprint")
    configurations = {case["id"]: case for case in manifest["configurations"]}
    samples, history = [], []
    for job in manifest["schedule"]:
        case = configurations[job["configuration_id"]]
        attempts = attempts_for(root, job)
        result = {"status": "pending", "row": None, "diagnostics": [], "returncode": None}
        for attempt in attempts:
            result = load_attempt(attempt)
            history.append({**job, "serial_refinements": case_refinement(case, manifest["settings"]),
                            "attempt": attempt.name, "status": result["status"],
                            "returncode": result["returncode"], "path": str(attempt.relative_to(root))})
        row = result.get("row") or {}
        samples.append({
            **job, "method": case["method"], "solver": case["solver"],
            "serial_refinements": case_refinement(case, manifest["settings"]),
            "preconditioner": case["preconditioner"], "status": result["status"],
            "attempts": len(attempts), "returncode": result["returncode"],
            "diagnostics": "; ".join(result["diagnostics"]),
            "run_wall_seconds": result.get("run_wall_seconds", ""),
            "ranks": manifest["settings"]["ranks"], "source_fingerprint": manifest["fingerprint"],
            "execution_class": manifest["protocol"]["execution_class"],
            **{f"driver_{key}": value for key, value in row.items()},
        })
    summary = []
    for case in configurations.values():
        selected = [row for row in samples if row["configuration_id"] == case["id"] and not row["warmup"]]
        successes = [row for row in selected if row["status"] == "success"]
        counts = collections.Counter(row["status"] for row in selected)
        warmup_failures = sum(row["status"] != "success" for row in samples
                              if row["configuration_id"] == case["id"] and row["warmup"])
        complete = len(successes) == len(selected) and not warmup_failures
        historical_failures = sum(row["status"] != "success" for row in history if row["configuration_id"] == case["id"])
        output = {**case, "serial_refinements": case_refinement(case, manifest["settings"]),
                  "expected_samples": len(selected), "successful_samples": len(successes),
                  "success_fraction": len(successes) / len(selected),
                  "status_counts": json.dumps(counts, sort_keys=True), "warmup_failures": warmup_failures,
                  "historical_failed_attempts": historical_failures,
                  "ranking_eligible": complete and not historical_failures,
                  "execution_class": manifest["protocol"]["execution_class"]}
        for field in TIMINGS + ("outer_iterations",) + ERRORS:
            values = [float(row[field if field == "run_wall_seconds" else f"driver_{field}"]) for row in successes]
            output.update(statistic_fields(values, field))
        for field in ("elements", "velocity_dofs", "pressure_dofs"):
            output[field] = successes[0][f"driver_{field}"] if successes else ""
        output["system_dofs"] = (int(output["velocity_dofs"]) + int(output["pressure_dofs"]) + 1) if successes else ""
        summary.append(output)
    for row in summary:
        reference = next((other for other in summary if other["method"] == row["method"] and
                          other["serial_refinements"] == row["serial_refinements"] and
                          other["solver"] == "mumps" and other["ranking_eligible"]), None)
        row["direct_reference"] = "available" if reference else "unavailable"
        row["mms_norm_difference_max"] = max(abs(row[f"{field}_median"] - reference[f"{field}_median"])
                                             for field in ERRORS) if reference and row["successful_samples"] else ""
    write_csv(root / "samples.csv", samples)
    write_csv(root / "attempts.csv", history)
    write_csv(root / "summary.csv", summary)
    write_refinement_summary(root, summary)
    render_report(root, manifest, summary)
    return samples, summary


def render_report(root, manifest, summary):
    levels = sorted({row["serial_refinements"] for row in summary})
    lines = ["# 2D Stokes screening and refinement", "", f"Execution class: **{manifest['protocol']['execution_class']}**.",
             f"Fixed degree-two smooth, variable-viscosity, nonperiodic triangular void baseline; serial refinements: {levels}.",
             "Each level is an independent cold solve, refined before partitioning; driver level stays zero.",
             "Times are seconds; medians use successful measured runs only, never warm-ups.",
             "Partial-success configurations are not ranking eligible; all attempts and failures remain in CSV/logs.",
             "MUMPS ignores preconditioners. Its iteration count is not applicable.",
             "Phase MPI maxima are not additive. Direct-reference differences compare MMS norms, not solution vectors.",
             "Exploratory runs do not establish controlled performance rankings.", "",
             "Direct references are matched by method AND refinement. Refinement ratios use fully successful adjacent selected levels only.",
             "Multi-level studies also write refinement.csv and refinement SVGs: DOF growth, timing ratios, iteration ratios, and error rates.", "",
             "| Refinement | Method | Solver | Preconditioner | System DOFs | Successes | Failed attempts | Eligible | Assembly | Setup | Linear | Total | Iterations |",
             "|---:|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        def value(field):
            number = row[f"{field}_median"]
            return f"{number:.6g}" if number != "" else "--"
        setup = "direct_setup_seconds" if row["solver"] == "mumps" else "preconditioner_setup_seconds"
        iterations = "n/a" if row["solver"] == "mumps" else value("outer_iterations")
        lines.append(f"| {row['serial_refinements']} | {row['method']} | {row['solver']} | {row['preconditioner']} | {row['system_dofs']} | "
                     f"{row['successful_samples']}/{row['expected_samples']} | {row['historical_failed_attempts']} | {row['ranking_eligible']} | "
                     f"{value('assembly_seconds')} | {value(setup)} | {value('linear_solve_seconds')} | "
                     f"{value('solve_seconds')} | {iterations} |")
    (root / "report.md").write_text("\n".join(lines) + "\n")
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    for method, refinement in dict.fromkeys((row["method"], row["serial_refinements"]) for row in summary):
        rows = sorted((row for row in summary if row["method"] == method and
                       row["serial_refinements"] == refinement and row["ranking_eligible"]),
                      key=lambda row: row["solve_seconds_median"])
        largest = max((row["solve_seconds_max"] for row in rows), default=1.0) or 1.0
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{100 + len(rows) * 32}" viewBox="0 0 1100 {100 + len(rows) * 32}">',
               '<rect width="100%" height="100%" fill="white"/>',
               f'<text x="16" y="26" font-family="sans-serif" font-size="18">{html.escape(method)}, refinement {refinement}: inclusive solve time (seconds)</text>',
               f'<text x="16" y="49" font-family="sans-serif" font-size="12">{html.escape(manifest["protocol"]["execution_class"])}; bars: median; whiskers: min/max; only fully successful configurations</text>']
        for index, row in enumerate(rows):
            vertical = 70 + index * 32
            label = html.escape(row["solver"] + " / " + row["preconditioner"])
            width = 470 * row["solve_seconds_median"] / largest
            lower = 550 + 470 * row["solve_seconds_min"] / largest
            upper = 550 + 470 * row["solve_seconds_max"] / largest
            svg.extend([
                f'<text x="16" y="{vertical + 14}" font-family="monospace" font-size="12">{label}</text>',
                f'<rect x="550" y="{vertical}" width="{width:.3f}" height="20" fill="#377eb8"/>',
                f'<path d="M {lower:.3f} {vertical+10} H {upper:.3f}" stroke="#111"/>',
                f'<text x="1030" y="{vertical + 14}" font-family="sans-serif" font-size="12">{row["solve_seconds_median"]:.4f}</text>',
            ])
        suffix = "" if levels == [1] else f"-r{refinement}"
        (figures / f"{method}{suffix}-total.svg").write_text("\n".join(svg + ["</svg>"]))
    if len(levels) > 1:
        render_refinement_figures(figures, manifest, summary)


def render_refinement_figures(figures, manifest, summary):
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
              "#7f7f7f", "#bcbd22", "#17becf", "#003f5c", "#7a5195", "#ef5675", "#ffa600", "#222222")
    for method in dict.fromkeys(row["method"] for row in summary):
        method_rows = [row for row in summary if row["method"] == method]
        for field, suffix, label in (("solve_seconds", "total", "Inclusive solve time (seconds)"),
                                     ("outer_iterations", "iterations", "Outer iterations (MUMPS not applicable)")):
            rows = [row for row in method_rows if field != "outer_iterations" or row["solver"] != "mumps"]
            eligible = [row for row in rows if row["ranking_eligible"]]
            groups = collections.defaultdict(list)
            for row in rows:
                groups[(row["solver"], row["preconditioner"])].append(row)
            svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1250" height="620" viewBox="0 0 1250 620">',
                   '<rect width="100%" height="100%" fill="white"/>',
                   f'<text x="20" y="28" font-family="sans-serif" font-size="18">{html.escape(method)}: {label}</text>',
                   f'<text x="20" y="50" font-family="sans-serif" font-size="12">{html.escape(manifest["protocol"]["execution_class"])}; medians and min/max; failed or incomplete cells omitted, with broken lines</text>',
                   '<path d="M 85 90 V 500 H 650" fill="none" stroke="#222"/>',
                   '<text x="160" y="585" font-family="sans-serif" font-size="13">System DOFs including gauge (log2 scale); r = serial refinements</text>']
            if eligible:
                minimum = min(math.log2(row["system_dofs"]) for row in eligible)
                maximum = max(math.log2(row["system_dofs"]) for row in eligible)
                span = maximum - minimum or 1
                ceiling = max(row[f"{field}_max"] for row in eligible) * 1.05 or 1
                for tick in range(6):
                    height = 500 - tick * 78
                    svg.extend([f'<path d="M 85 {height} H 650" stroke="#ddd"/>',
                                f'<text x="76" y="{height+4}" text-anchor="end" font-family="sans-serif" font-size="12">{ceiling * tick / 5:.3g}</text>'])
                ticks = sorted({(row["serial_refinements"], row["system_dofs"]) for row in eligible})
                for refinement, dofs in ticks:
                    horizontal = 100 + 530 * (math.log2(dofs) - minimum) / span
                    svg.extend([f'<text x="{horizontal:.2f}" y="527" text-anchor="middle" font-family="sans-serif" font-size="12">{dofs}</text>',
                                f'<text x="{horizontal:.2f}" y="545" text-anchor="middle" font-family="sans-serif" font-size="12">r={refinement}</text>'])
                for index, (key, series) in enumerate(groups.items()):
                    color = colors[index % len(colors)]
                    dash = ' stroke-dasharray="6 3"' if key[0] == "fgmres" else ""
                    legend_height = 90 + index * 29
                    svg.extend([f'<path d="M 695 {legend_height} H 725" stroke="{color}" stroke-width="2"{dash}/>',
                                f'<text x="735" y="{legend_height+4}" font-family="monospace" font-size="12">{html.escape(" / ".join(key))}</text>'])
                    previous = None
                    for row in sorted(series, key=lambda row: row["serial_refinements"]):
                        if not row["ranking_eligible"]:
                            previous = None
                            continue
                        horizontal = 100 + 530 * (math.log2(row["system_dofs"]) - minimum) / span
                        vertical = 500 - 390 * row[f"{field}_median"] / ceiling
                        lower = 500 - 390 * row[f"{field}_min"] / ceiling
                        upper = 500 - 390 * row[f"{field}_max"] / ceiling
                        if previous:
                            svg.append(f'<path d="M {previous[0]:.2f} {previous[1]:.2f} L {horizontal:.2f} {vertical:.2f}" stroke="{color}" fill="none" stroke-width="2"{dash}/>')
                        svg.extend([f'<path d="M {horizontal:.2f} {lower:.2f} V {upper:.2f}" stroke="{color}"/>',
                                    f'<circle cx="{horizontal:.2f}" cy="{vertical:.2f}" r="3" fill="{color}"/>'])
                        previous = (horizontal, vertical)
            else:
                svg.append('<text x="110" y="280" font-family="sans-serif" font-size="18">No fully successful configurations</text>')
            (figures / f"{method}-refinement-{suffix}.svg").write_text("\n".join(svg + ["</svg>"]))


def run_study(root, manifest, environment, resume=False, retry_failed=False):
    root.mkdir(parents=True, exist_ok=True)
    with study_lock(root):
        path = root / "manifest.json"
        if path.exists():
            if not resume:
                raise ValueError("Study exists; use --resume with identical settings or choose a new output directory")
            saved = read_json(path)
            if saved.get("fingerprint") != manifest_fingerprint(saved):
                raise ValueError("Manifest content does not match its fingerprint")
            if saved.get("fingerprint") != manifest["fingerprint"]:
                raise ValueError("Resume refused: source, executable, mesh, settings, host, allocation, or environment changed")
            manifest = saved
        else:
            if resume:
                raise ValueError("Cannot resume: manifest.json is missing")
            if any(child.name != ".lock" for child in root.iterdir()):
                raise ValueError("New study output must be empty")
            write_json(path, manifest)
        configurations = {case["id"]: case for case in manifest["configurations"]}
        try:
            for ordinal, job in enumerate(manifest["schedule"], start=1):
                attempts = attempts_for(root, job)
                latest = load_attempt(attempts[-1]) if attempts else None
                if latest and latest["status"] not in {"running", "incomplete", "interrupted"}:
                    if latest["status"] == "success" or not retry_failed:
                        continue
                if file_hash(manifest["provenance"]["executable"]["path"]) != manifest["provenance"]["executable"]["sha256"]:
                    raise ValueError("Executable changed during study")
                mesh = manifest["provenance"]["mesh"]
                if "sha256" in mesh and file_hash(mesh["path"]) != mesh["sha256"]:
                    raise ValueError("Mesh changed during study")
                directory = root / "runs" / job["configuration_id"] / job["sample_id"] / f"attempt-{len(attempts) + 1:03d}"
                result = execute_attempt(directory, manifest, configurations[job["configuration_id"]], job, environment)
                print(f"[{ordinal}/{len(manifest['schedule'])}] {job['configuration_id']} {job['sample_id']}: {result['status']}", flush=True)
        finally:
            samples, _ = summarize(root)
        return 0 if all(row["status"] == "success" for row in samples) else 2


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def positive_float(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("plan", "run"):
        command = subparsers.add_parser(action)
        command.add_argument("--executable", default=str(WORKSPACE / "build/miniapps/stokes/stokes_mms"))
        command.add_argument("--mesh", default=str(WORKSPACE / "meshes/holes-2d/triangles/level-1/void/nonperiodic.mesh"),
                             help="Baseline-compatible 2D nonperiodic straight triangular void mesh")
        command.add_argument("--launcher", default="mpirun -n {ranks}", help="Shell-quoted tokens; {ranks} is substituted; no shell is run")
        command.add_argument("--ranks", type=positive_int, default=4)
        command.add_argument("--refinements", nargs="+", type=int, choices=(0, 1, 2), default=[1],
                             help="Independent serial-before-partition cold solves; default: 1; refinement study: 0 1 2")
        command.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
        command.add_argument("--solvers", nargs="+", choices=SOLVERS, default=list(SOLVERS))
        command.add_argument("--preconditioners", nargs="+", choices=PRECONDITIONERS, default=list(PRECONDITIONERS))
        command.add_argument("--repetitions", type=positive_int, default=3)
        command.add_argument("--warmups", type=positive_int, default=1)
        command.add_argument("--seed", type=int, default=20261101)
        command.add_argument("--timeout", type=positive_float, default=180.0, help="Per MPI launch, seconds; entire local process group is terminated on timeout")
        command.add_argument("--relative-tolerance", type=positive_float, default=1e-12)
        command.add_argument("--absolute-tolerance", type=positive_float, default=1e-14)
        command.add_argument("--restart", type=positive_int, default=100)
        command.add_argument("--max-iterations", type=positive_int, default=2000)
        command.add_argument("--execution-class", choices=("exploratory", "controlled"), default="exploratory")
        command.add_argument("--allocation-note", default="No exclusive allocation independently verified")
        command.add_argument("--binding-note", default="No explicit binding independently verified")
        if action == "run":
            command.add_argument("--output", required=True)
            command.add_argument("--resume", action="store_true")
            command.add_argument("--retry-failed", action="store_true")
        else:
            command.add_argument("--json", action="store_true", help="Print the complete manifest without creating a study or launching a solver")
    report = subparsers.add_parser("report")
    report.add_argument("--output", required=True)
    args = parser.parse_args(arguments)
    if args.action != "report":
        for field in ("methods", "solvers", "preconditioners", "refinements"):
            if len(getattr(args, field)) != len(set(getattr(args, field))):
                parser.error(f"--{field} must not contain duplicates")
        if args.action == "run" and args.retry_failed and not args.resume:
            parser.error("--retry-failed requires --resume")
        if args.execution_class == "controlled" and any(not note.strip() or note.startswith("No ")
                                                       for note in (args.allocation_note, args.binding_note)):
            parser.error("Controlled runs require explicit --allocation-note and --binding-note; this is an operator assertion")
    return args


def main(arguments=None):
    args = parse_arguments(arguments)
    try:
        if args.action == "report":
            root = Path(args.output).resolve()
            with study_lock(root):
                samples, _ = summarize(root)
            return 0 if all(row["status"] == "success" for row in samples) else 2
        manifest, environment = make_manifest(args)
        if args.action == "plan":
            if args.json:
                print(json.dumps(manifest, indent=2, sort_keys=True))
            else:
                print(f"{len(manifest['configurations'])} configurations; {len(manifest['schedule'])} launches; "
                      f"{len(manifest['configurations']) * args.repetitions} measured samples")
                for case in manifest["configurations"]:
                    print(case["id"])
                print("Dry run only. No solver launched; no output directory created.")
            return 0
        return run_study(Path(args.output).resolve(), manifest, environment, args.resume, args.retry_failed)
    except (OSError, ValueError, KeyError, csv.Error) as error:
        print(f"benchmark: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("benchmark: interrupted; raw attempts retained; use --resume", file=sys.stderr)
        return 130


def handle_termination(signum, frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, handle_termination)
    sys.exit(main())
