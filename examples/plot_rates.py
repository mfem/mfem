import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np

def run_simulation_and_get_errors():
    executable_path = './dual_L2' 
    
    refinement_levels = range(1, 5) 

    l2_errors = []
    div_errors = []
    
    for refs in refinement_levels:
        try:
            command = [
                executable_path, 
                '--refs', f'{refs}',
                '-mi', '1000',
                '-tol', '1e-14', 
                '-o', '2',
                '-o2', '1',
                '-no-vis'
            ]
            print(f"Running command: {' '.join(command)}")

            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                check=True
            )

            output = result.stdout
            
            l2_match = re.search(r"L2 error: (\d+\.\d+e[+-]\d+|\d+\.\d+)", output)
            div_match = re.search(r"div error: (\d+\.\d+e[+-]\d+|\d+\.\d+)", output)

            if l2_match and div_match:
                l2_error = float(l2_match.group(1))
                div_error = float(div_match.group(1))
                
                l2_errors.append(l2_error)
                div_errors.append(div_error)
                
                print(f"  Refinement: {refs} -> L2 Error: {l2_error:.4e}, Div Error: {div_error:.4e}")
            else:
                print(f"  ERROR: Could not parse errors for refinement level {refs}.")
                print("  --- Full Output ---")
                print(output)
                print("  -------------------")

        except subprocess.CalledProcessError as e:
            print(f"FATAL ERROR: The executable returned a non-zero exit code for refinement level {refs}.")
            print(f"  Return Code: {e.returncode}")
            print(f"  Stdout: {e.stdout}")
            print(f"  Stderr: {e.stderr}")
            return
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return

    print("\n--- Simulation Runs Complete ---")

    if not l2_errors or not div_errors:
        print("No data was collected. Cannot generate plot.")
        return

    print("Generating convergence plot...")
    
    h_values = [1 / (2**r) for r in refinement_levels]
    # h_values = refinement_levels

    plt.style.use('seaborn-v0_8-dark')
    fig, ax = plt.subplots(figsize=(10, 8))

    l2_line, = ax.loglog(h_values, l2_errors, 'o-', label='L2 Error', markersize=8, linewidth=2)
    div_line, = ax.loglog(h_values, div_errors, 'o-', label='Divergence Error', markersize=8, linewidth=2)

    # if l2_errors:
        # h_squared = [l2_errors[0] * (h / h_values[0])**2 for h in h_values]
        # ax.loglog(h_values, h_squared, 'k--', label=r'$O(h^2)$')
    # if div_errors:
        # h_linear = [div_errors[0] * (h / h_values[0])**2 for h in h_values]
        # ax.loglog(h_values, h_linear, 'k:', label=r'$O(h)$')

    if len(h_values) > 3:
        def draw_slope_triangle(h_vals, error_vals, line):
            line_color = line.get_color()

            observed_slope = (np.log(error_vals[3]) - np.log(error_vals[2])) / (np.log(h_vals[3]) - np.log(h_vals[2]))

            h_pos = [h_vals[2], h_vals[3]] 
            y_pos = (error_vals[2] + error_vals[3]) * 1.15
            
            tri_x = [h_pos[0], h_pos[1], h_pos[1], h_pos[0]]
            tri_y = [y_pos, y_pos, y_pos * (h_pos[1]/h_pos[0])**observed_slope, y_pos]

            ax.plot(tri_x, tri_y, color=line_color, linestyle='--')
            
            ax.fill(tri_x, tri_y, color=line_color, alpha=0.2)
            
            ax.text((h_pos[0] + h_pos[1]) / 2, tri_y[0] * 0.9, '1', color=line_color, ha='center', va='top', fontsize=12, fontweight='bold')
            ax.text(h_pos[1] * 1.1, (tri_y[1] + tri_y[2]) / 2, f'{observed_slope:.1f}', color=line_color, ha='left', va='center', fontsize=12, fontweight='bold')

        draw_slope_triangle(h_values, l2_errors, l2_line)
        
        draw_slope_triangle(h_values, div_errors, div_line)


    ax.set_xlabel('Mesh size (h)', fontsize=14)
    ax.set_ylabel('Error', fontsize=14)
    ax.set_title('Convergence Rates', fontsize=16, fontweight='bold')
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, which="both", ls="--", c='0.7')
    
    ax.invert_xaxis()

    legend = ax.legend(fontsize=12, frameon=True, facecolor='white', framealpha=0.8)
    legend.get_frame().set_edgecolor('black')

    plt.tight_layout()
    plt.show()


def draw_slope_triangle(ax, h_vals, error_vals, line):
    try:
        observed_slope = (np.log(error_vals[3]) - np.log(error_vals[2])) / (np.log(h_vals[3]) - np.log(h_vals[2]))
    except (IndexError, ValueError) as e:
        print(f"  Warning: Could not calculate slope. Not enough data points? Error: {e}")
        return

    line_color = line.get_color()
    h_pos = [h_vals[2], h_vals[3]] 
    y_pos = (error_vals[2] + error_vals[3]) * 1.15
    
    tri_x = [h_pos[0], h_pos[1], h_pos[1], h_pos[0]]
    tri_y = [y_pos, y_pos, y_pos * (h_pos[1]/h_pos[0])**observed_slope, y_pos]

    ax.plot(tri_x, tri_y, color=line_color, linestyle='--')
    ax.fill(tri_x, tri_y, color=line_color, alpha=0.2)
    
    ax.text((h_pos[0] + h_pos[1]) / 2, tri_y[0] * 0.9, '1', color=line_color, ha='center', va='top', fontsize=12, fontweight='bold')
    ax.text(h_pos[1] * 1.05, (tri_y[1] + tri_y[2]) / 2, f'{observed_slope:.1f}', color=line_color, ha='left', va='center', fontsize=12, fontweight='bold')


def run_and_plot_case(case_config):
    executable_path = './dual_L2' 
    refinement_levels = range(1, 5)

    l2_errors = []
    div_errors = []
    
    print(f"\n--- Starting Case: {case_config['title']} ---")

    for refs in refinement_levels:
        try:
            command = [
                executable_path, 
                '--refs',  f'{refs}',
                '-ex', str(case_config['problem_num']),
                '-o', str(case_config['rt_order']),
                '-o2', '1', 
                '-tol', '1e-14', 
                '-mi', '1000',
                '-no-vis'
            ]
            print(f"  Running command: {' '.join(command)}")

            result = subprocess.run(command, capture_output=True, text=True, check=True)
            output = result.stdout
            
            l2_match = re.search(r"L2 error: (\d+\.\d+e[+-]\d+|\d+\.\d+)", output)
            div_match = re.search(r"div error: (\d+\.\d+e[+-]\d+|\d+\.\d+)", output)

            if l2_match and div_match:
                l2_errors.append(float(l2_match.group(1)))
                div_errors.append(float(div_match.group(1)))
            else:
                print(f"  ERROR: Could not parse errors for refinement level {refs}.")
        
        except (FileNotFoundError, subprocess.CalledProcessError, Exception) as e:
            print(f"  FATAL ERROR during simulation run: {e}")
            return 

    if not l2_errors or not div_errors:
        print("  No data collected for this case. Skipping plot.")
        return

    h_values = [1 / (2**r) for r in refinement_levels]
    plt.style.use('seaborn-v0_8-dark')
    fig, ax = plt.subplots(figsize=(8, 6))

    l2_line, = ax.loglog(h_values, l2_errors, 'o-', label='L2 Error', markersize=8, linewidth=2)
    div_line, = ax.loglog(h_values, div_errors, 'o-', label='Divergence Error', markersize=8, linewidth=2)

    if len(h_values) > 3:
        draw_slope_triangle(ax, h_values, l2_errors, l2_line)
        draw_slope_triangle(ax, h_values, div_errors, div_line)

    ax.set_xlabel('Mesh size (h)', fontsize=14)
    ax.set_ylabel('Error', fontsize=14)
    ax.set_title(case_config['title'], fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, which="both", ls="--", c='0.7')
    ax.invert_xaxis()
    legend = ax.legend(fontsize=12, frameon=True, facecolor='white', framealpha=0.8)
    legend.get_frame().set_edgecolor('black')
    plt.tight_layout()
    # plt.show()

    output_filename = case_config['filename']
    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"  Successfully saved plot to '{output_filename}'")
    except Exception as e:
        print(f"  Error saving plot: {e}")
        
    plt.close(fig) 

if __name__ == '__main__':
    simulation_cases = [
        {
            'problem_num': 1, 
            'rt_order': 1, 
            'title': 'Convergence: Trig Example #1 (Div-Free), RT Order 1',
            'filename': 'trig1_div_free_rt1.png'
        },
        {
            'problem_num': 1, 
            'rt_order': 2, 
            'title': 'Convergence: Trig Example #1 (Div-Free), RT Order 2',
            'filename': 'trig1_div_free_rt2.png'
        },
        {
            'problem_num': 2, 
            'rt_order': 1, 
            'title': 'Convergence: Trig Example #2 (Non Div-Free), RT Order 1',
            'filename': 'trig2_non_div_free_rt1.png'
        },
        {
            'problem_num': 2, 
            'rt_order': 2, 
            'title': 'Convergence: Trig Example #2 (Non Div-Free), RT Order 2',
            'filename': 'trig2_non_div_free_rt2.png'
        }
    ]

    for sim in simulation_cases: 
        run_and_plot_case(sim)

    # run_simulation_and_get_errors()