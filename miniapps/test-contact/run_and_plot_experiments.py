#!/usr/bin/env python3
"""
Run test-contact experiments with different Schwarz configurations
and plot PCG iteration results.

This script runs the miniapp with various threshold configurations,
collects the PCG iteration data, and generates comparison plots.
"""

import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import argparse
import hashlib
import json


# ============================================================================
# Configuration: Adjust these parameters
# ============================================================================

# Problem parameters
PROBLEM_CONFIG = {
    'prob': 0,        # Problem number (0=two-block, 1=ironing, 2=beam-sphere)
    'sr': 2,          # Serial refinements
    'pr': 0,          # Parallel refinements
    'tr': 2,          # Tribol proximity ratio
    'nsteps': 4,      # Number of timesteps
    'np': 1,          # Number of MPI ranks
}

# Schwarz eigendecomposition thresholds to test
# Format: (eigenvalue_threshold, support_threshold, label, color, marker, linestyle, linewidth)
# Color = eigenvalue threshold, Marker & Linestyle = support threshold
THRESHOLD_CONFIGS = [
    # Baseline: row-based subdomains (no eigendecomposition)
    (None, None, 'Row-based', 'gray', 'x', '-.', 2),

    # Vary eigenvalue threshold (fixed support = 1e-4)
    (1e3, 1e-4, 'Eigen=1e3, Supp=1e-4', 'red', 'o', '-', 1.5),
    (1e4, 1e-4, 'Eigen=1e4, Supp=1e-4', 'blue', 'o', '-', 1.5),
    (1e5, 1e-4, 'Eigen=1e5, Supp=1e-4', 'green', 'o', '-', 1.5),
    (1e6, 1e-4, 'Eigen=1e6, Supp=1e-4', 'purple', 'o', '-', 1.5),
    (1e7, 1e-4, 'Eigen=1e7, Supp=1e-4', 'orange', 'o', '-', 1.5),

    # Vary support threshold (fixed eigenvalue = 1e4)
    (1e4, 1e-3, 'Eigen=1e4, Supp=1e-3', 'blue', 's', '--', 1.5),
    (1e4, 1e-2, 'Eigen=1e4, Supp=1e-2', 'blue', '^', ':', 1.5),

    (1e6, 1e-3, 'Eigen=1e6, Supp=1e-3', 'purple', 's', '--', 1.5),
    (1e6, 1e-2, 'Eigen=1e6, Supp=1e-2', 'purple', '^', ':', 1.5),

    # Lower eigenvalue with coarser support
    (1e3, 1e-3, 'Eigen=1e3, Supp=1e-3', 'red', 's', '--', 1.5),
][:1]

# Reference configurations
RUN_AMGF_REFERENCE = True  # AMGF with direct filter (no Schwarz)
RUN_AMG_REFERENCE = True   # Plain AMG (no AMGF, no Schwarz)

# ============================================================================


def hash_config(config):
    """
    Create a short hash of the problem configuration.

    Args:
        config: Dictionary of problem parameters

    Returns:
        str: 8-character hash of the configuration
    """
    config_str = json.dumps(config, sort_keys=True)
    hash_obj = hashlib.md5(config_str.encode())
    return hash_obj.hexdigest()[:8]


def get_output_dir(config):
    """
    Get output directory path based on problem configuration.

    Args:
        config: Dictionary of problem parameters

    Returns:
        Path: Output directory path with config hash
    """
    config_hash = hash_config(config)
    return Path(f"experiment_results_{config_hash}")


def save_config(output_dir, config):
    """
    Save problem configuration to the output directory.

    Args:
        output_dir: Output directory path
        config: Dictionary of problem parameters
    """
    config_file = output_dir / "problem_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Problem configuration saved to: {config_file}")


def load_config(output_dir):
    """
    Load problem configuration from the output directory.

    Args:
        output_dir: Output directory path

    Returns:
        dict: Problem configuration, or None if not found
    """
    config_file = output_dir / "problem_config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            return json.load(f)
    return None


def run_test_contact(config_name, extra_flags, verbose=True):
    """
    Run the test-contact miniapp with specified flags.

    Args:
        config_name: Name for this configuration
        extra_flags: Additional command-line flags
        verbose: Print progress messages

    Returns:
        str: Output from the miniapp
    """
    cmd = [
        'mpirun', '-np', str(PROBLEM_CONFIG['np']),
        './test-contact',
        '-prob', str(PROBLEM_CONFIG['prob']),
        '-sr', str(PROBLEM_CONFIG['sr']),
        '-pr', str(PROBLEM_CONFIG['pr']),
        '-tr', str(PROBLEM_CONFIG['tr']),
        '-nsteps', str(PROBLEM_CONFIG['nsteps']),
        '-no-vis'
    ] + extra_flags

    if verbose:
        print(f"Running: {config_name}")
        print(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        output = result.stdout + result.stderr

        if result.returncode != 0:
            print(f"  Warning: Non-zero exit code ({result.returncode})")

        return output

    except subprocess.TimeoutExpired:
        print(f"  Error: Timeout after 300 seconds")
        return ""
    except Exception as e:
        print(f"  Error: {e}")
        return ""


def parse_pcg_iterations(output_text):
    """
    Parse PCG iteration counts from test-contact output.

    Returns:
        list: PCG iteration counts across all timesteps (flattened)
    """
    pcg_iters = []

    for line in output_text.split('\n'):
        match = re.search(r'PCG number of iterations\s*=\s*([\d\s]+)', line)
        if match:
            numbers = [int(x) for x in match.group(1).split()]
            pcg_iters.extend(numbers)

    return pcg_iters


def parse_subdomain_stats(output_text):
    """
    Parse Schwarz subdomain statistics from output.

    Returns:
        dict: Statistics including num_subdomains, min_size, max_size, avg_size
    """
    stats = {}

    for line in output_text.split('\n'):
        if 'Number of subdomains:' in line:
            match = re.search(r'Number of subdomains:\s*(\d+)', line)
            if match:
                stats['num_subdomains'] = int(match.group(1))

        if 'Subdomain sizes' in line:
            match = re.search(r'min:\s*(\d+),\s*max:\s*(\d+),\s*avg:\s*([\d.]+)', line)
            if match:
                stats['min_size'] = int(match.group(1))
                stats['max_size'] = int(match.group(2))
                stats['avg_size'] = float(match.group(3))

    return stats


def parse_subdomain_stats_per_step(output_text):
    """
    Parse per-Newton-step subdomain statistics from output.

    Returns:
        dict: Dictionary with lists of max_size and coverage per Newton step
              {'max_sizes': [int, ...], 'coverages': [float, ...]}
    """
    max_sizes = []
    coverages = []

    for line in output_text.split('\n'):
        # Look for lines like: "Subdomain sizes: min: X, max: Y, avg: Z"
        if 'Subdomain sizes' in line:
            match = re.search(r'max:\s*(\d+)', line)
            if match:
                max_sizes.append(int(match.group(1)))

        # Look for lines like: "Coverage: X" or "Subdomain coverage: X"
        coverage_match = re.search(r'[Cc]overage:\s*([\d.]+)', line)
        if coverage_match:
            coverages.append(float(coverage_match.group(1)))

    return {'max_sizes': max_sizes, 'coverages': coverages}


def read_existing_results(output_dir):
    """
    Read PCG iteration data and subdomain statistics from existing output files.

    Args:
        output_dir: Path to the output directory

    Returns:
        tuple: (results, subdomain_results)
               results: List of (config_name, pcg_iters, color, marker, linestyle, linewidth) tuples
               subdomain_results: List of (config_name, subdomain_stats, color, marker, linestyle, linewidth) tuples
    """
    results = []
    subdomain_results = []

    print("=" * 60)
    print("Reading existing output files")
    print("=" * 60)
    print()

    if not output_dir.exists():
        print(f"Error: Output directory '{output_dir}' does not exist.")
        return results, subdomain_results

    # Reference configurations
    print("=== Reference Configurations ===")

    if RUN_AMG_REFERENCE:
        output_file = output_dir / "output_amg.txt"
        if output_file.exists():
            output = output_file.read_text()
            pcg_iters = parse_pcg_iterations(output)
            subdomain_stats = parse_subdomain_stats_per_step(output)
            if pcg_iters:
                results.append(("AMG", pcg_iters, 'black', 'o', '-', 2.5))
                subdomain_results.append(("AMG", subdomain_stats, 'black', 'o', '-', 2.5))
                print(f"  AMG: Found {len(pcg_iters)} PCG iterations")
            else:
                print(f"  Warning: No PCG iterations found in {output_file}")
        else:
            print(f"  Warning: File not found: {output_file}")

    if RUN_AMGF_REFERENCE:
        output_file = output_dir / "output_amgf_direct.txt"
        if output_file.exists():
            output = output_file.read_text()
            pcg_iters = parse_pcg_iterations(output)
            subdomain_stats = parse_subdomain_stats_per_step(output)
            if pcg_iters:
                results.append(("AMGF (direct filter)", pcg_iters, 'black', 'd', '-', 2.5))
                subdomain_results.append(("AMGF (direct filter)", subdomain_stats, 'black', 'd', '-', 2.5))
                print(f"  AMGF (direct filter): Found {len(pcg_iters)} PCG iterations")
            else:
                print(f"  Warning: No PCG iterations found in {output_file}")
        else:
            print(f"  Warning: File not found: {output_file}")
    print()

    # Schwarz configurations
    print("=== Schwarz Configurations ===")

    for eigen_thresh, support_thresh, label, color, marker, linestyle, linewidth in THRESHOLD_CONFIGS:
        safe_label = label.replace(' ', '_').replace('=', '').replace(',', '')
        output_file = output_dir / f"output_{safe_label}.txt"

        if output_file.exists():
            output = output_file.read_text()
            pcg_iters = parse_pcg_iterations(output)
            subdomain_stats = parse_subdomain_stats_per_step(output)
            if pcg_iters:
                results.append((label, pcg_iters, color, marker, linestyle, linewidth))
                subdomain_results.append((label, subdomain_stats, color, marker, linestyle, linewidth))
                print(f"  {label}: Found {len(pcg_iters)} PCG iterations")
            else:
                print(f"  Warning: No PCG iterations found in {output_file}")
        else:
            print(f"  Warning: File not found: {output_file}")

    print()
    print("=" * 60)
    print("Finished reading output files!")
    print("=" * 60)
    print()

    return results, subdomain_results


def run_experiments(output_dir):
    """
    Run all experiments and collect results.

    Args:
        output_dir: Path to the output directory

    Returns:
        tuple: (results, subdomain_results)
               results: List of (config_name, pcg_iters, color, marker, linestyle, linewidth) tuples
               subdomain_results: List of (config_name, subdomain_stats, color, marker, linestyle, linewidth) tuples
    """
    output_dir.mkdir(exist_ok=True)
    results = []
    subdomain_results = []

    print("=" * 60)
    print("Running test-contact experiments")
    print("=" * 60)
    print()

    # Reference configurations
    print("=== Reference Configurations ===")

    if RUN_AMG_REFERENCE:
        output = run_test_contact("AMG", [])

        if output:
            pcg_iters = parse_pcg_iterations(output)
            subdomain_stats = parse_subdomain_stats_per_step(output)
            if pcg_iters:
                results.append(("AMG", pcg_iters, 'black', 'o', '-', 2.5))
                subdomain_results.append(("AMG", subdomain_stats, 'black', 'o', '-', 2.5))
                print(f"  Found {len(pcg_iters)} PCG iterations")

                # Save output
                output_file = output_dir / "output_amg.txt"
                output_file.write_text(output)

    if RUN_AMGF_REFERENCE:
        output = run_test_contact("AMGF (direct filter)", ['-amgf'])

        if output:
            pcg_iters = parse_pcg_iterations(output)
            subdomain_stats = parse_subdomain_stats_per_step(output)
            if pcg_iters:
                results.append(("AMGF (direct filter)", pcg_iters, 'black', 'd', '-', 2.5))
                subdomain_results.append(("AMGF (direct filter)", subdomain_stats, 'black', 'd', '-', 2.5))
                print(f"  Found {len(pcg_iters)} PCG iterations")

                # Save output
                output_file = output_dir / "output_amgf_direct.txt"
                output_file.write_text(output)
    print()

    # Schwarz configurations
    print("=== Schwarz Configurations ===")

    for eigen_thresh, support_thresh, label, color, marker, linestyle, linewidth in THRESHOLD_CONFIGS:
        if eigen_thresh is None:
            # Row-based subdomains (no eigendecomposition)
            flags = ['-amgf', '-schwarz']
        else:
            # Eigendecomposition-based
            flags = [
                '-amgf', '-schwarz', '-schwarz-eigen',
                '-schwarz-eigen-thresh', str(eigen_thresh),
                '-schwarz-support-thresh', str(support_thresh)
            ]

        output = run_test_contact(label, flags)

        if output:
            pcg_iters = parse_pcg_iterations(output)
            subdomain_stats_summary = parse_subdomain_stats(output)
            subdomain_stats_per_step = parse_subdomain_stats_per_step(output)

            if pcg_iters:
                results.append((label, pcg_iters, color, marker, linestyle, linewidth))
                subdomain_results.append((label, subdomain_stats_per_step, color, marker, linestyle, linewidth))
                print(f"  Found {len(pcg_iters)} PCG iterations")

                if subdomain_stats_summary:
                    print(f"  Subdomains: {subdomain_stats_summary.get('num_subdomains', 'N/A')}, "
                          f"Avg size: {subdomain_stats_summary.get('avg_size', -1):.1f}")

                # Save output
                safe_label = label.replace(' ', '_').replace('=', '').replace(',', '')
                output_file = output_dir / f"output_{safe_label}.txt"
                output_file.write_text(output)
        print()

    print("=" * 60)
    print("Experiments completed!")
    print("=" * 60)
    print()

    return results, subdomain_results


def plot_results(results, output_dir, output_file='pcg_iterations_comparison.png'):
    """
    Plot PCG iteration results for all configurations.

    Args:
        results: List of (config_name, pcg_iters, color, marker, linestyle, linewidth) tuples
        output_dir: Path to the output directory
        output_file: Output plot filename
    """
    if not results:
        print("No results to plot!")
        return

    plt.figure(figsize=(14, 8))

    for config_name, pcg_iters, color, marker, linestyle, linewidth in results:
        solve_indices = range(1, len(pcg_iters) + 1)

        plt.plot(solve_indices, pcg_iters,
                label=config_name,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                marker=marker,
                markersize=8,
                alpha=1.0 if "AMGF" in config_name else 0.75)

        # Print statistics
        print(f"{config_name}:")
        print(f"  Total solves: {len(pcg_iters)}")
        print(f"  Mean PCG iters: {np.mean(pcg_iters):.1f}")
        print(f"  Median: {np.median(pcg_iters):.1f}")
        print(f"  Min: {min(pcg_iters)}, Max: {max(pcg_iters)}")
        print(f"  Total PCG iters: {sum(pcg_iters)}")
        print()

    plt.xlabel('Solve Index (across all timesteps)', fontsize=13)
    plt.ylabel('PCG Iterations', fontsize=13)
    plt.yscale('log')
    plt.title('PCG Iterations: Eigendecomposition-based vs Row-based Schwarz Subdomains',
             fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--', which='both')
    plt.tight_layout()

    # Save plot
    plot_path = output_dir / output_file
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")

    plt.show()


def plot_max_subdomain_sizes(subdomain_results, output_dir, output_file='max_subdomain_sizes.png'):
    """
    Plot maximum subdomain sizes per Newton step for all configurations.

    Args:
        subdomain_results: List of (config_name, subdomain_stats, color, marker, linestyle, linewidth) tuples
        output_dir: Path to the output directory
        output_file: Output plot filename
    """
    if not subdomain_results:
        print("No subdomain results to plot!")
        return

    plt.figure(figsize=(14, 8))

    for config_name, subdomain_stats, color, marker, linestyle, linewidth in subdomain_results:
        max_sizes = subdomain_stats.get('max_sizes', [])
        if not max_sizes:
            continue

        step_indices = range(1, len(max_sizes) + 1)

        plt.plot(step_indices, max_sizes,
                label=config_name,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                marker=marker,
                markersize=8,
                markevery=max(1, len(max_sizes) // 20),  # Show ~20 markers
                alpha=1.0 if "AMGF" in config_name else 0.75)

    plt.xlabel('Newton Step Index', fontsize=13)
    plt.ylabel('Maximum Subdomain Size', fontsize=13)
    plt.yscale('log')
    plt.title('Maximum Subdomain Size per Newton Step',
             fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--', which='both')
    plt.tight_layout()

    # Save plot
    plot_path = output_dir / output_file
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Max subdomain size plot saved to: {plot_path}")

    plt.show()


def plot_coverage(subdomain_results, output_dir, output_file='coverage.png'):
    """
    Plot subdomain coverage per Newton step for all configurations.

    Args:
        subdomain_results: List of (config_name, subdomain_stats, color, marker, linestyle, linewidth) tuples
        output_dir: Path to the output directory
        output_file: Output plot filename
    """
    if not subdomain_results:
        print("No subdomain results to plot!")
        return

    plt.figure(figsize=(14, 8))

    for config_name, subdomain_stats, color, marker, linestyle, linewidth in subdomain_results:
        coverages = subdomain_stats.get('coverages', [])
        if not coverages:
            continue

        step_indices = range(1, len(coverages) + 1)

        plt.plot(step_indices, coverages,
                label=config_name,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                marker=marker,
                markersize=8,
                markevery=max(1, len(coverages) // 20),  # Show ~20 markers
                alpha=1.0 if "AMGF" in config_name else 0.75)

    plt.xlabel('Newton Step Index', fontsize=13)
    plt.ylabel('Coverage', fontsize=13)
    plt.yscale('log')
    plt.title('Subdomain Coverage per Newton Step',
             fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Save plot
    plot_path = output_dir / output_file
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Coverage plot saved to: {plot_path}")

    plt.show()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run test-contact experiments and plot PCG iteration results.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--skip-run',
        action='store_true',
        help='Skip running experiments and only read from existing output files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Specify output directory (overrides auto-generated hash-based directory)'
    )
    args = parser.parse_args()

    print(f"Problem configuration: {PROBLEM_CONFIG}")
    print(f"Testing {len(THRESHOLD_CONFIGS)} Schwarz configurations")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        print(f"Using specified output directory: {output_dir}")
    else:
        output_dir = get_output_dir(PROBLEM_CONFIG)
        config_hash = hash_config(PROBLEM_CONFIG)
        print(f"Configuration hash: {config_hash}")
        print(f"Output directory: {output_dir}")
    print()

    if args.skip_run:
        # Read from existing output files
        print("Mode: Reading from existing output files (--skip-run)\n")

        if not output_dir.exists():
            print(f"Error: Output directory '{output_dir}' does not exist.")
            sys.exit(1)

        # Load and display saved configuration
        saved_config = load_config(output_dir)
        if saved_config:
            print(f"Loaded configuration from {output_dir / 'problem_config.json'}:")
            print(json.dumps(saved_config, indent=2))
            print()
        else:
            print(f"Warning: No configuration file found in {output_dir}")
            print()

        results, subdomain_results = read_existing_results(output_dir)
    else:
        # Run experiments
        print("Mode: Running experiments\n")

        # Check if test-contact exists
        if not Path('./test-contact').exists():
            print("Error: test-contact executable not found in current directory.")
            print("Please run 'make test-contact' first or cd to the miniapps/test-contact directory.")
            sys.exit(1)

        results, subdomain_results = run_experiments(output_dir)

        # Save configuration
        save_config(output_dir, PROBLEM_CONFIG)
        print()

    if not results:
        print("Error: No results collected. Check if test-contact runs successfully or if output files exist.")
        sys.exit(1)

    # Plot results
    print("Generating plots...")
    plot_results(results, output_dir)
    plot_max_subdomain_sizes(subdomain_results, output_dir)
    plot_coverage(subdomain_results, output_dir)

    print("\nAll done!")


if __name__ == "__main__":
    main()
