#!/usr/bin/env python3
"""
General experiment runner for the updated test-contact miniapp.

Features:
- Runs arbitrary parameter sweeps
- Parses PCG iterations and linear solve times
- Saves raw output and structured metadata
- Plots both per-solve behavior and aggregate summaries
- Supports grouping runs into curves by arbitrary parameters

Example uses:
- Compare solver variants
- Sweep refinement levels
- Sweep MPI ranks
- Sweep Schwarz weight / variant / cg iterations
- Compare problems or nonlinear vs linear modes
"""

import argparse
import hashlib
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# USER CONFIGURATION
# =============================================================================

BASE_CONFIG = {
    "np": 8,
    "prob": 0,
    "model": "linear",          # "linear" or "nonlinear"
    "sr": 0,
    "pr": 0,
    "nsteps": 1,
    "msteps": 0,
    "tr": 2.0,
    "vis": False,
    "paraview": False,
    "amgf": False,
    "amgf_fsolver": "auto",
    "schwarz": False,
    "schwarz_expand": False,
    "schwarz_cg_iters": 0,
    "schwarz_variant": 2,
    "schwarz_weight": 1.0,
    "schwarz_min_diag": 0.0,
    "schwarz_uniform_weight": 1.0,
    "subspace_pl": 0
}

# Arbitrary parameter sweep.
# Each key maps to a list of values to test.
SWEEP_PARAMETERS = {
    # Example solver sweep:
    "amgf": [True],
    "schwarz": [True],
    "schwarz_variant": [2],
    "schwarz_uniform_weight": [0.1],

    # Uncomment for other experiments:
    # "np": [1, 2, 4],
    "sr": [3, 4],
    # "pr": [0, 1],
    # "prob": [0, 1, 2],
}

# Which parameters define distinct curves in plots
CURVE_KEYS = [
    "amgf",
    "schwarz",
    "schwarz_uniform_weight",
]

# Which parameter should be used for x-axis in summary plots.
# Options:
#   - a parameter name from configs, such as "sr", "np", "pr", "schwarz_weight"
#   - "run_index", meaning just one point per experiment in run order
X_AXIS_MODE = "run_index"

# Output directory prefix
OUTPUT_PREFIX = "contact_experiments"

# Executable path
EXECUTABLE = "./test-contact"

# Timeout per run
TIMEOUT_SECONDS = 3600


# =============================================================================
# UTILS
# =============================================================================

def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def short_hash(obj, n=10):
    return hashlib.md5(canonical_json(obj).encode()).hexdigest()[:n]


def sanitize_filename(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def build_output_dir(base_config, sweep_parameters, user_output_dir=None):
    if user_output_dir:
        return Path(user_output_dir)
    signature = {
        "base": base_config,
        "sweep": sweep_parameters,
    }
    return Path(f"{OUTPUT_PREFIX}_{short_hash(signature)}")


def expand_sweep(base_config, sweep_parameters):
    if not sweep_parameters:
        return [base_config.copy()]

    keys = list(sweep_parameters.keys())
    values_product = itertools.product(*(sweep_parameters[k] for k in keys))

    configs = []
    for values in values_product:
        cfg = base_config.copy()
        for k, v in zip(keys, values):
            cfg[k] = v
        configs.append(cfg)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for cfg in configs:
        key = canonical_json(cfg)
        if key not in seen:
            seen.add(key)
            unique.append(cfg)
    return unique


def config_to_label(config, keys):
    parts = []
    for k in keys:
        if k in config:
            parts.append(f"{k}={config[k]}")
    return ", ".join(parts) if parts else "default"


def config_to_command(config):
    cmd = [
        "mpirun", "-np", str(config["np"]),
        EXECUTABLE,
        "-prob", str(config["prob"]),
        "--nonlinear" if config["model"] == "nonlinear" else "--linear",
        "-sr", str(config["sr"]),
        "-pr", str(config["pr"]),
        "-nsteps", str(config["nsteps"]),
        "-msteps", str(config["msteps"]),
        "-tr", str(config["tr"]),
        "--visualization" if config["vis"] else "--no-visualization",
        "--paraview" if config["paraview"] else "--no-paraview",
        "--amgf" if config["amgf"] else "--no-amgf",
        "-amgf-fsolver", str(config["amgf_fsolver"]),
        "--schwarz" if config["schwarz"] else "--no-schwarz",
        "--schwarz-expand" if config["schwarz_expand"] else "--no-schwarz-expand",
        "-schwarz-cg-iters", str(config["schwarz_cg_iters"]),
        "-schwarz-variant", str(config["schwarz_variant"]),
        "-schwarz-weight", str(config["schwarz_weight"]),
        "-schwarz-min-diag", str(config["schwarz_min_diag"]),
        "-schwarz-uniform-weight", str(config["schwarz_uniform_weight"]),
        "-subspace-pl", str(config["subspace_pl"])
    ]
    return cmd


def run_command(cmd, timeout=TIMEOUT_SECONDS):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "combined_output": result.stdout + result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": -999,
            "stdout": "",
            "stderr": f"Timeout after {timeout} seconds",
            "combined_output": "",
        }
    except Exception as e:
        return {
            "returncode": -998,
            "stdout": "",
            "stderr": str(e),
            "combined_output": "",
        }


# =============================================================================
# PARSING
# =============================================================================

def parse_int_list_line(text, label):
    pattern = rf"{re.escape(label)}\s*=\s*([0-9\s]+)"
    matches = re.findall(pattern, text)
    values = []
    for match in matches:
        values.extend(int(x) for x in match.split())
    return values


def parse_float_list_line(text, label):
    pattern = rf"{re.escape(label)}\s*=\s*([0-9eE+.\-\s]+)"
    matches = re.findall(pattern, text)
    values = []
    for match in matches:
        values.extend(float(x) for x in match.split())
    return values


def parse_scalar_int(text, label):
    pattern = rf"{re.escape(label)}\s*=\s*(\d+)"
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


def parse_scalar_float(text, label):
    pattern = rf"{re.escape(label)}\s*=\s*([0-9eE+.\-]+)"
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def parse_run_output(output_text):
    data = {
        "optimizer_iterations": parse_scalar_int(output_text, "Optimizer number of iterations"),
        "initial_energy": parse_scalar_float(output_text, "Initial Energy objective"),
        "final_energy": parse_scalar_float(output_text, "Final Energy objective"),
        "pcg_iterations": parse_int_list_line(output_text, "PCG number of iterations"),
        "linear_solve_times": parse_float_list_line(output_text, "Linear solve times [s]"),
    }

    data["num_linear_solves_from_iters"] = len(data["pcg_iterations"])
    data["num_linear_solves_from_times"] = len(data["linear_solve_times"])

    if data["pcg_iterations"]:
        data["pcg_total"] = int(sum(data["pcg_iterations"]))
        data["pcg_mean"] = float(np.mean(data["pcg_iterations"]))
        data["pcg_median"] = float(np.median(data["pcg_iterations"]))
        data["pcg_min"] = int(min(data["pcg_iterations"]))
        data["pcg_max"] = int(max(data["pcg_iterations"]))
    else:
        data["pcg_total"] = None
        data["pcg_mean"] = None
        data["pcg_median"] = None
        data["pcg_min"] = None
        data["pcg_max"] = None

    if data["linear_solve_times"]:
        data["time_total"] = float(sum(data["linear_solve_times"]))
        data["time_mean"] = float(np.mean(data["linear_solve_times"]))
        data["time_median"] = float(np.median(data["linear_solve_times"]))
        data["time_min"] = float(min(data["linear_solve_times"]))
        data["time_max"] = float(max(data["linear_solve_times"]))
    else:
        data["time_total"] = None
        data["time_mean"] = None
        data["time_median"] = None
        data["time_min"] = None
        data["time_max"] = None

    if data["pcg_iterations"] and data["linear_solve_times"]:
        n = min(len(data["pcg_iterations"]), len(data["linear_solve_times"]))
        if n > 0:
            ratios = [data["linear_solve_times"][i] / data["pcg_iterations"][i]
                      for i in range(n) if data["pcg_iterations"][i] > 0]
            data["time_per_iter_mean"] = float(np.mean(ratios)) if ratios else None
        else:
            data["time_per_iter_mean"] = None
    else:
        data["time_per_iter_mean"] = None

    return data


# =============================================================================
# DATA STORAGE
# =============================================================================

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_run_artifacts(output_dir, run_id, config, result, parsed):
    run_dir = output_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    save_json(run_dir / "config.json", config)
    save_json(run_dir / "parsed.json", parsed)

    (run_dir / "stdout.txt").write_text(result["stdout"])
    (run_dir / "stderr.txt").write_text(result["stderr"])
    (run_dir / "combined_output.txt").write_text(result["combined_output"])

    metadata = {
        "run_id": run_id,
        "returncode": result["returncode"],
    }
    save_json(run_dir / "run_metadata.json", metadata)


def load_existing_runs(output_dir):
    runs_root = output_dir / "runs"
    records = []

    if not runs_root.exists():
        return records

    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue

        config_file = run_dir / "config.json"
        parsed_file = run_dir / "parsed.json"
        meta_file = run_dir / "run_metadata.json"

        if config_file.exists() and parsed_file.exists() and meta_file.exists():
            records.append({
                "run_id": run_dir.name,
                "config": load_json(config_file),
                "parsed": load_json(parsed_file),
                "metadata": load_json(meta_file),
            })

    return records


# =============================================================================
# PLOTTING
# =============================================================================

def group_records_by_curve(records, curve_keys):
    grouped = {}
    for record in records:
        label = config_to_label(record["config"], curve_keys)
        grouped.setdefault(label, []).append(record)
    return grouped


def get_x_value(record, x_axis_mode, fallback_index):
    if x_axis_mode == "run_index":
        return fallback_index
    return record["config"].get(x_axis_mode, fallback_index)


def sort_curve_records(records, x_axis_mode):
    decorated = []
    for i, rec in enumerate(records):
        x = get_x_value(rec, x_axis_mode, i)
        decorated.append((x, rec))
    decorated.sort(key=lambda t: t[0])
    return [rec for _, rec in decorated]


def plot_per_solve_curves(records, output_dir):
    valid = [r for r in records if r["parsed"]["pcg_iterations"] or r["parsed"]["linear_solve_times"]]
    if not valid:
        print("No per-solve data to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for rec in valid:
        label = config_to_label(rec["config"], CURVE_KEYS)
        pcg = rec["parsed"]["pcg_iterations"]
        times = rec["parsed"]["linear_solve_times"]

        if pcg:
            x = np.arange(1, len(pcg) + 1)
            axes[0].plot(x, pcg, marker="o", linewidth=1.8, alpha=0.9, label=label)

        if times:
            x = np.arange(1, len(times) + 1)
            axes[1].plot(x, times, marker="o", linewidth=1.8, alpha=0.9, label=label)

    axes[0].set_title("PCG iterations per linear solve")
    axes[0].set_xlabel("Linear solve index")
    axes[0].set_ylabel("PCG iterations")
    axes[0].grid(True, alpha=0.3, linestyle="--")

    axes[1].set_title("Linear solve time per linear solve")
    axes[1].set_xlabel("Linear solve index")
    axes[1].set_ylabel("Time [s]")
    axes[1].grid(True, alpha=0.3, linestyle="--")

    for ax in axes:
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = output_dir / "per_solve_curves.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.show()


def plot_summary_curves(records, output_dir, x_axis_mode):
    valid = [r for r in records if r["metadata"]["returncode"] == 0]
    if not valid:
        print("No successful runs to summarize.")
        return

    grouped = group_records_by_curve(valid, CURVE_KEYS)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    metrics = [
        ("pcg_total", "Total PCG iterations"),
        ("pcg_mean", "Mean PCG iterations"),
        ("time_total", "Total linear solve time [s]"),
        ("time_mean", "Mean linear solve time [s]"),
    ]

    for ax, (metric_key, metric_title) in zip(axes, metrics):
        for label, recs in grouped.items():
            recs_sorted = sort_curve_records(recs, x_axis_mode)

            xvals = []
            yvals = []

            for i, rec in enumerate(recs_sorted):
                x = get_x_value(rec, x_axis_mode, i)
                y = rec["parsed"].get(metric_key)
                if y is not None:
                    xvals.append(x)
                    yvals.append(y)

            if xvals and yvals:
                ax.plot(xvals, yvals, marker="o", linewidth=2.0, label=label)

        ax.set_title(metric_title)
        ax.set_xlabel(x_axis_mode)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = output_dir / "summary_curves.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.show()


def plot_scatter_time_vs_iterations(records, output_dir):
    valid = [r for r in records if r["parsed"]["pcg_total"] is not None and r["parsed"]["time_total"] is not None]
    if not valid:
        print("No aggregate timing vs iteration data to plot.")
        return

    plt.figure(figsize=(8, 6))

    grouped = group_records_by_curve(valid, CURVE_KEYS)
    for label, recs in grouped.items():
        x = [r["parsed"]["pcg_total"] for r in recs]
        y = [r["parsed"]["time_total"] for r in recs]
        plt.scatter(x, y, s=70, alpha=0.85, label=label)

    plt.xlabel("Total PCG iterations")
    plt.ylabel("Total linear solve time [s]")
    plt.title("Timing vs iteration")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(fontsize=8)

    out = output_dir / "time_vs_iterations.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.show()


# =============================================================================
# REPORTING
# =============================================================================

def print_summary_table(records):
    headers = [
        "run_id", "returncode", "curve_label", "pcg_total", "pcg_mean",
        "time_total", "time_mean", "n_iters", "n_times"
    ]
    rows = []

    for rec in records:
        parsed = rec["parsed"]
        rows.append([
            rec["run_id"],
            rec["metadata"]["returncode"],
            config_to_label(rec["config"], CURVE_KEYS),
            parsed["pcg_total"],
            parsed["pcg_mean"],
            parsed["time_total"],
            parsed["time_mean"],
            parsed["num_linear_solves_from_iters"],
            parsed["num_linear_solves_from_times"],
        ])

    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]

    def fmt_row(r):
        return " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(r))

    print()
    print(fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(fmt_row(r))
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run and plot parameter sweeps for test-contact."
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Do not execute experiments, load existing run artifacts only.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Explicit output directory.",
    )
    args = parser.parse_args()

    output_dir = build_output_dir(BASE_CONFIG, SWEEP_PARAMETERS, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(output_dir / "experiment_definition.json", {
        "base_config": BASE_CONFIG,
        "sweep_parameters": SWEEP_PARAMETERS,
        "curve_keys": CURVE_KEYS,
        "x_axis_mode": X_AXIS_MODE,
        "executable": EXECUTABLE,
    })

    records = []

    if args.skip_run:
        print(f"Loading existing runs from {output_dir}")
        records = load_existing_runs(output_dir)
    else:
        if not Path(EXECUTABLE).exists():
            print(f"Error: executable not found: {EXECUTABLE}")
            sys.exit(1)

        configs = expand_sweep(BASE_CONFIG, SWEEP_PARAMETERS)
        print(f"Running {len(configs)} experiment(s)")
        print(f"Output directory: {output_dir}")

        for i, config in enumerate(configs):
            label = config_to_label(config, CURVE_KEYS)
            run_id = f"run_{i:03d}_{short_hash(config)}"
            cmd = config_to_command(config)

            print()
            print("=" * 80)
            print(f"Run {i + 1}/{len(configs)}")
            print(f"Run ID: {run_id}")
            print(f"Label: {label}")
            print("Command:")
            print(" ".join(cmd))

            result = run_command(cmd)
            parsed = parse_run_output(result["combined_output"])

            save_run_artifacts(output_dir, run_id, config, result, parsed)

            records.append({
                "run_id": run_id,
                "config": config,
                "parsed": parsed,
                "metadata": {"returncode": result["returncode"]},
            })

            print(f"Return code: {result['returncode']}")
            print(f"PCG solves parsed: {parsed['num_linear_solves_from_iters']}")
            print(f"Timing entries parsed: {parsed['num_linear_solves_from_times']}")
            print(f"Total PCG iterations: {parsed['pcg_total']}")
            print(f"Total linear solve time [s]: {parsed['time_total']}")

    if not records:
        print("No records found.")
        sys.exit(1)

    print_summary_table(records)

    plot_per_solve_curves(records, output_dir)
    plot_summary_curves(records, output_dir, X_AXIS_MODE)
    plot_scatter_time_vs_iterations(records, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()