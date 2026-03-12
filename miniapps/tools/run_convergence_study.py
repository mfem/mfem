#!/usr/bin/env python3
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.ticker as mticker
import json
import os
import argparse

def run_reconstruction(refinement_level, ho_value, method):
    """
    Run the reconstruction executable with given refinement level, ho value,
    and reconstruction method.
    Returns the L2 error value.
    """
    try:
        # Run the executable
        result = subprocess.run(
            ['./reconstruction', '-r', str(refinement_level), '-f', '2',
             '-ho', str(ho_value), '-m', method],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Parse the output to extract the error
        output = result.stdout
        
        # Look for the line: || u_h - u ||_{L^2} =
        # The error value is on the next line
        lines = output.split('\n')
        for i, line in enumerate(lines):
            if '|| u_h - u ||_{L^2} =' in line:
                # Get the next non-empty line
                if i + 1 < len(lines):
                    error_line = lines[i + 1].strip()
                    if error_line:
                        # Extract the numerical value (handles scientific notation)
                        error_value = float(error_line)
                        return error_value
        
        print(f"Warning: Could not extract error for refinement level {refinement_level}, ho {ho_value}")
        return None
        
    except subprocess.TimeoutExpired:
        print(f"Error: Execution timed out for refinement level {refinement_level}, ho {ho_value}")
        return None
    except Exception as e:
        print(f"Error running refinement level {refinement_level}, ho {ho_value}: {e}")
        return None

def save_errors_to_file(errors_dict, refinement_levels, filename='convergence_errors.json'):
    """
    Save the error dictionary to a JSON file.
    """
    data = {
        'refinement_levels': refinement_levels,
        'errors': {str(k): v for k, v in errors_dict.items()}
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nErrors saved to '{filename}'")

def load_errors_from_file(filename='convergence_errors.json'):
    """
    Load the error dictionary from a JSON file.
    Returns (errors_dict, refinement_levels) or (None, None) if file doesn't exist.
    """
    if not os.path.exists(filename):
        return None, None
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        errors_dict = {int(k): v for k, v in data['errors'].items()}
        refinement_levels = data['refinement_levels']
        
        print(f"\nLoaded errors from '{filename}'")
        return errors_dict, refinement_levels
    except Exception as e:
        print(f"\nError loading file '{filename}': {e}")
        return None, None

def plot_convergence(refinement_levels, errors_dict, method):
    """
    Plot the convergence curves for different ho values.
    errors_dict: dictionary with ho values as keys and error lists as values
    """
    fs = 24
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', serif='times')
    plt.rcParams.update({'font.family': 'serif', 'font.size': fs, 'font.weight': 'normal'})

    fig, ax = plt.subplots(figsize=(8, 6))
    font_label = {'family': 'serif', 'size': 1.2 * fs}
    font_tick = font_manager.FontProperties(family='serif', size=0.8 * fs)

    # Define colors and markers for different ho values
    plot_colors = ['k', 'b', 'r', 'g', 'm', 'c']
    plot_markers = ['o', 's', '^', 'D', 'v', 'p']
    styles = {
        ho: {'color': plot_colors[idx % len(plot_colors)],
             'marker': plot_markers[idx % len(plot_markers)],
             'label': fr'$p={ho}$'}
        for idx, ho in enumerate(errors_dict.keys())
    }

    h_values = np.array([1.0 / (2**(r + 1)) for r in refinement_levels])

    # Plot the data for each ho value
    for ho, errors in errors_dict.items():
        style = styles[ho]
        ax.plot(
            h_values,
            errors,
            linestyle='-',
            color=style['color'],
            linewidth=1.2,
            marker=style['marker'],
            markersize=6,
            markerfacecolor=style['color'],
            markeredgecolor=style['color'],
            label=style['label']
        )

    # Add reference lines for convergence rates O(h^2) and O(h^4)
    # Choose a reference point (use the middle of the h range)
    h_ref_idx = len(h_values) // 2
    h_ref = h_values[h_ref_idx]
    
    # For each ho value, use its error at the reference point
    for idx, (ho, errors) in enumerate(errors_dict.items()):
        if not np.isnan(errors[h_ref_idx]):
            e_ref = errors[h_ref_idx]

            rate = ho + 1
            linestyle = '--'
            alpha = 0.5
            color = styles[ho]['color']
            
            # Calculate reference line: error = C * h^rate
            # At h_ref: e_ref = C * h_ref^rate, so C = e_ref / h_ref^rate
            C = e_ref / (h_ref ** rate)
            reference_line = C * (h_values ** rate)
            
            ax.plot(
                h_values,
                reference_line,
                linestyle=linestyle,
                color=color,
                linewidth=1.0,
                alpha=alpha,
                label=f'$O(h^{rate})$'
            )

    # Legend
    plt.legend(
        loc='lower left',
        frameon=False,
        numpoints=1,
        fontsize=fs,
        borderaxespad=0.2,
        ncol=1,
        handlelength=1.4,
        handletextpad=0.6,
        columnspacing=1.0
    )

    # Labels
    plt.xlabel(r'$h$', labelpad=10, fontdict=font_label)
    plt.ylabel(r'$\|u_h - u\|_{L^2}$', labelpad=10, fontdict=font_label)

    # Tick settings
    ax.tick_params(axis="both", pad=10, top=False, right=False)

    # Determine appropriate axis limits
    all_errors = [float(e) for errors in errors_dict.values() for e in errors
                  if np.isfinite(e) and float(e) > 0.0]
    if not all_errors:
        print("\nNo finite positive errors to plot.")
        return
    max_error = max(all_errors)
    min_error = min(all_errors)
    
    # Set x-axis limits
    plt.xlim(1, 5.e-3)
    
    # Use log scale if errors span multiple orders of magnitude
    if max_error / min_error > 100:
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.ylabel(r'$\|u_h - u\|_{L^2}$', labelpad=10, fontdict=font_label)
        plt.xlabel(r'$h$', labelpad=10, fontdict=font_label)
        # Enforce consistent sub-interval ticks on log axes.
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10)))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    else:
        # Linear scale with padding
        y_padding = (max_error - min_error) * 0.1
        plt.ylim([max(0, min_error - y_padding), max_error + y_padding])
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_tick)

    plt.tight_layout()
    figure_name = f'Figure_L2error_{method}.png'
    plt.savefig(figure_name, format='png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as '{figure_name}'")

def main():
    """
    Main function to run convergence study.
    """
    parser = argparse.ArgumentParser(description="Run or print reconstruction convergence study results.")
    parser.add_argument(
        '--print-only',
        action='store_true',
        default=False,
        help='Only read and print results from convergence_errors.json without running reconstruction.'
    )
    parser.add_argument(
        '-m', '--method',
        default='element_least_squares',
        help='Method passed to ./reconstruction via -m (default: element_least_squares).'
    )
    cli_args = parser.parse_args()

    # Refinement levels to test
    refinement_levels = [0, 1, 2, 3, 4]
    
    # ho values to test
    ho_values = [1, 7]
    
    # Error data filename
    error_file = 'convergence_errors.json'

    all_errors = None
    if cli_args.print_only:
        all_errors, loaded_refinement_levels = load_errors_from_file(error_file)

        if all_errors is None:
            print(f"\nNo cached data found in '{error_file}'.")
            return

        if loaded_refinement_levels != refinement_levels:
            print("\nWarning: refinement levels in cached data differ from current defaults.")

        missing_ho = [ho for ho in ho_values if ho not in all_errors]
        if missing_ho:
            print(f"\nCached data is missing ho values: {missing_ho}")
            return

        print("\nPrint-only mode enabled: using cached error data.")
        print("=" * 60)
    
    # Calculate errors if needed
    if all_errors is None:
        all_errors = {}
        
        print("Running convergence study...")
        print("=" * 60)
        
        for ho in ho_values:
            print(f"\n{'='*60}")
            print(f"Testing with -ho {ho}")
            print(f"{'='*60}")
            
            errors = []
            for r in refinement_levels:
                print(f"\nRunning refinement level {r} with ho {ho}...")
                error = run_reconstruction(r, ho, cli_args.method)
                
                if error is not None:
                    errors.append(error)
                    print(f"  Error: {error:.6e}")
                else:
                    print(f"  Failed to get error for refinement level {r}, ho {ho}")
                    errors.append(np.nan)
            
            all_errors[ho] = errors
        
        # Save the errors to file
        save_errors_to_file(all_errors, refinement_levels, error_file)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("Convergence Study Results:")
    print("-" * 80)
    header = f"{'Refinement':>12} |"
    for ho in ho_values:
        header += f" {'ho=' + str(ho):>18} |"
    print(header)
    print("-" * 80)
    
    for i, r in enumerate(refinement_levels):
        row = f"{r:>12} |"
        for ho in ho_values:
            e = all_errors[ho][i]
            if not np.isnan(e):
                row += f" {e:>18.6e} |"
            else:
                row += f" {'N/A':>18} |"
        print(row)
    print("=" * 80)
    
    # Calculate convergence rates
    print("\nConvergence Rates:")
    print("-" * 80)
    for ho in ho_values:
        print(f"\nho = {ho}:")
        errors = all_errors[ho]
        for i in range(1, len(errors)):
            if not np.isnan(errors[i]) and not np.isnan(errors[i-1]) and errors[i] > 0 and errors[i-1] > 0:
                rate = np.log(errors[i-1] / errors[i]) / np.log(2)
                print(f"  Rate between levels {refinement_levels[i-1]} and {refinement_levels[i]}: {rate:.3f}")
    
    # Prepare data for plotting (remove NaN values)
    plot_errors = {}
    valid_refinement = refinement_levels
    
    for ho in ho_values:
        errors = all_errors[ho]
        # Check if we have valid data
        if not all(np.isnan(e) for e in errors):
            plot_errors[ho] = errors
    
    if len(plot_errors) > 0:
        plot_convergence(valid_refinement, plot_errors, cli_args.method)
    else:
        print("\nInsufficient valid data to create plot.")

if __name__ == "__main__":
    main()
