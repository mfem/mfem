#!/usr/bin/env python3
"""
Run test-contact experiments comparing different solver variants and plot results.

This script compares:
- AMG (baseline)
- AMGF with direct filter
- Row-based Schwarz (variant 0)
- Variant 2 Schwarz with different relaxation weights
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
    'sr': 0,          # Serial refinements
    'pr': 0,          # Parallel refinements
    'tr': 2,          # Tribol proximity ratio
    'nsteps': 1,      # Number of timesteps
    'msteps': 0,      # Extra steps (for ironing problem)
    'np': 1,          # Number of MPI ranks
}

# Solver configurations to test
# Format: (name, flags, color, marker)
SOLVER_CONFIGS = [
    # Baseline solvers
    ('AMG', [], 'black', 'o'),
    ('AMGF', ['-amgf'], 'gray', 's'),

    # Row-based Schwarz (variant 0 = multiplicative)
    ('Multiplicative Schwarz', ['-amgf', '-schwarz'], 'blue', '^'),

    # Variant 2 with different relaxation weights
    ('Additive Schwarz unweighted', ['-amgf', '-schwarz', '-schwarz-variant', '2', '-schwarz-weight', '1.0'], 'red', 'D'),
    #('Additive Schwarz (w=1e-7)', ['-amgf', '-schwarz', '-schwarz-variant', '2', '-schwarz-weight', '0.0000001'], 'orange', '^'),
    #('Additive Schwarz (w=1e-1)', ['-amgf', '-schwarz', '-schwarz-variant', '2', '-schwarz-weight', '0.1'], 'green', '^'),
    #('Additive Schwarz (w=0.2)', ['-amgf', '-schwarz', '-schwarz-variant', '2', '-schwarz-weight', '0.2'], 'blue', '^'),
    #('Additive Schwarz (w=0.4)', ['-amgf', '-schwarz', '-schwarz-variant', '2', '-schwarz-weight', '0.4'], 'brown', '^'),
    #('Additive Schwarz (w=0.6)', ['-amgf', '-schwarz', '-schwarz-variant', '2', '-schwarz-weight', '0.6'], 'silver', '^')

    # Different minimum diagonal values
    #('Min D = 0 (no filter)', ['-amgf', '-schwarz', '-schwarz-examine-diag', '-schwarz-min-diag', '0'], 'black', 'o'),
    #('Min D = 1', ['-amgf', '-schwarz', '-schwarz-examine-diag', '-schwarz-min-diag', '1'], 'blue', 's'),
    #('Min D = 1e3', ['-amgf', '-schwarz', '-schwarz-examine-diag', '-schwarz-min-diag', '1000'], 'green', '^'),
    #('Min D = 1e6', ['-amgf', '-schwarz', '-schwarz-examine-diag', '-schwarz-min-diag', '1e6'], 'red', 'v'),
    #('Min D = 1e9', ['-amgf', '-schwarz', '-schwarz-examine-diag', '-schwarz-min-diag', '1e9'], 'orange', 'd'),

]

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
    return Path(f"variant_experiment_{config_hash}")


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
        '-msteps', str(PROBLEM_CONFIG['msteps']),
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
        print(f"  Error: Timeout after 3600 seconds")
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


def read_existing_results(output_dir):
    """
    Read PCG iteration data from existing output files.

    Args:
        output_dir: Path to the output directory

    Returns:
        list: List of (config_name, pcg_iters, color, marker) tuples
    """
    results = []

    print("=" * 60)
    print("Reading existing output files")
    print("=" * 60)
    print()

    if not output_dir.exists():
        print(f"Error: Output directory '{output_dir}' does not exist.")
        return results

    for config_name, _, color, marker in SOLVER_CONFIGS:
        safe_name = config_name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
        output_file = output_dir / f"output_{safe_name}.txt"

        if output_file.exists():
            output = output_file.read_text()
            pcg_iters = parse_pcg_iterations(output)
            if pcg_iters:
                results.append((config_name, pcg_iters, color, marker))
                print(f"  {config_name}: Found {len(pcg_iters)} PCG iterations")
            else:
                print(f"  Warning: No PCG iterations found in {output_file}")
        else:
            print(f"  Warning: File not found: {output_file}")

    print()
    print("=" * 60)
    print("Finished reading output files!")
    print("=" * 60)
    print()

    return results


def run_experiments(output_dir):
    """
    Run all experiments and collect results.

    Args:
        output_dir: Path to the output directory

    Returns:
        list: List of (config_name, pcg_iters, color, marker) tuples
    """
    output_dir.mkdir(exist_ok=True)
    results = []

    print("=" * 60)
    print("Running test-contact experiments")
    print("=" * 60)
    print()

    for config_name, flags, color, marker in SOLVER_CONFIGS:
        output = run_test_contact(config_name, flags)

        if output:
            pcg_iters = parse_pcg_iterations(output)
            if pcg_iters:
                results.append((config_name, pcg_iters, color, marker))
                print(f"  Found {len(pcg_iters)} PCG iterations")

                # Save output
                safe_name = config_name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
                output_file = output_dir / f"output_{safe_name}.txt"
                output_file.write_text(output)
        print()

    print("=" * 60)
    print("Experiments completed!")
    print("=" * 60)
    print()

    return results


def plot_results(results, output_dir, output_file='solver_variants_comparison.png'):
    """
    Plot PCG iteration results for all solver configurations.

    Args:
        results: List of (config_name, pcg_iters, color, marker) tuples
        output_dir: Path to the output directory
        output_file: Output plot filename
    """
    if not results:
        print("No results to plot!")
        return

    plt.figure(figsize=(14, 8))

    # Print header
    print("=" * 80)
    print("PCG Iteration Statistics")
    print("=" * 80)
    print()

    for config_name, pcg_iters, color, marker in results:
        solve_indices = range(1, len(pcg_iters) + 1)

        # Determine line style and width based on solver type
        if 'AMG' in config_name and 'AMGF' not in config_name:
            linestyle = '-'
            linewidth = 2.5
            alpha = 1.0
        elif 'AMGF' in config_name:
            linestyle = '-'
            linewidth = 2.5
            alpha = 1.0
        elif 'Row' in config_name:
            linestyle = '-'
            linewidth = 2.0
            alpha = 0.85
        else:  # Variant 2 with different weights
            linestyle = '--'
            linewidth = 1.5
            alpha = 0.75

        plt.plot(solve_indices, pcg_iters,
                label=config_name,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                marker=marker,
                markersize=8,
                alpha=alpha)

        # Print statistics
        print(f"{config_name}:")
        print(f"  Total solves: {len(pcg_iters)}")
        print(f"  Mean PCG iters: {np.mean(pcg_iters):.1f}")
        print(f"  Median: {np.median(pcg_iters):.1f}")
        print(f"  Min: {min(pcg_iters)}, Max: {max(pcg_iters)}")
        print(f"  Total PCG iters: {sum(pcg_iters)}")
        print()

    print("=" * 80)
    print()

    plt.xlabel('Solve Index (across all timesteps)', fontsize=13)
    plt.ylabel('PCG Iterations', fontsize=13)
    plt.yscale('log')
    plt.title('PCG Iterations: Solver Variant Comparison',
             fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--', which='both')
    plt.tight_layout()

    # Save plot
    plot_path = output_dir / output_file
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")

    plt.show()


def plot_summary_bar_chart(results, output_dir, output_file='solver_variants_summary.png'):
    """
    Create a bar chart showing total PCG iterations for each solver configuration.

    Args:
        results: List of (config_name, pcg_iters, color, marker) tuples
        output_dir: Path to the output directory
        output_file: Output plot filename
    """
    if not results:
        print("No results to plot!")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Extract data
    names = [name for name, _, _, _ in results]
    totals = [sum(iters) for _, iters, _, _ in results]
    means = [np.mean(iters) for _, iters, _, _ in results]
    colors_list = [color for _, _, color, _ in results]

    # Plot 1: Total PCG iterations
    bars1 = ax1.barh(names, totals, color=colors_list, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Total PCG Iterations', fontsize=12)
    ax1.set_title('Total PCG Iterations', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels
    for bar, total in zip(bars1, totals):
        ax1.text(total + max(totals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{total}', va='center', fontsize=9)

    # Plot 2: Mean PCG iterations per solve
    bars2 = ax2.barh(names, means, color=colors_list, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Mean PCG Iterations per Solve', fontsize=12)
    ax2.set_title('Mean PCG Iterations per Solve', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels
    for bar, mean in zip(bars2, means):
        ax2.text(mean + max(means) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{mean:.1f}', va='center', fontsize=9)

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / output_file
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Summary plot saved to: {plot_path}")

    plt.show()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run test-contact experiments comparing solver variants.',
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
    print(f"Testing {len(SOLVER_CONFIGS)} solver configurations")

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

        results = read_existing_results(output_dir)
    else:
        # Run experiments
        print("Mode: Running experiments\n")

        # Check if test-contact exists
        if not Path('./test-contact').exists():
            print("Error: test-contact executable not found in current directory.")
            print("Please run 'make test-contact' first or cd to the miniapps/test-contact directory.")
            sys.exit(1)

        results = run_experiments(output_dir)

        # Save configuration
        save_config(output_dir, PROBLEM_CONFIG)
        print()

    if not results:
        print("Error: No results collected. Check if test-contact runs successfully or if output files exist.")
        sys.exit(1)

    # Plot results
    print("Generating plots...")
    plot_results(results, output_dir)
    plot_summary_bar_chart(results, output_dir)

    print("\nAll done!")


if __name__ == "__main__":
    main()
