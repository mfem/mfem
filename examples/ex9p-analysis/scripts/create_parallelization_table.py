"""
This python script file performs a parallelization analsyis by running
a reading output files and composing a table from those values.

Example:
> python create_parallelization_table.py "/Users/sheridan7/Workspace/mfem/examples/ex9p-analysis/temp_output/"
"""

import numpy as np
import re
import os
import sys
from tabulate import tabulate


# check command line inputs
assert len(sys.argv) == 2, "This file needs 1 input argument: directory, but " + str(len(sys.argv)) + " were given."
directory = str(sys.argv[1])
# iterations = int(sys.argv[2])
# increment = int(sys.argv[3])
# plot_organization = False

# now we define the main function to be called at the end
def main():
    # comment out "run_simuations" if you only want to compute the errors
    vals = gather_vals()
    compute_rates(vals)


def gather_vals():
    vals = {'Processor_Runtime': [],
            'Speedup': [],
            'n_processes': [],
            'n_refinements': [],
            'n_Dofs': [],
            'h': [],
            'L1_Error': [],
            'L1_Rates': [],
            'L2_Error': [],
            'L2_Rates': [],
            'Linf_Error': [],
            'Linf_Rates': [],
            'dt': [],
            'Endtime': []}
    for filename in sorted(os.listdir(directory)):
        f = os.path.join(directory, filename)
        with open(f) as fp:
            for cnt, ln in enumerate(fp):
                l = ln.strip().split()
                vals[l[0]].append(float(l[1]))

    return vals

def compute_rates(vals):

    # Use tabulate to create a formatted table
    p_table = []
    s_table = []
    serial_time = 0
    for i in range(len(vals['h'])):
        if i == 0:
            serial_time = vals['Processor_Runtime'][i]
            vals['Speedup'].append(0.)
        else:
            _time = vals['Processor_Runtime'][i]
            # assert(_time != 0, "Division by Zero.")
            
            vals['Speedup'].append(serial_time / _time)
            print('_time = ', _time)
            print('speedup: ', serial_time / _time)

        p_table.append([vals['n_processes'][i], vals['Endtime'][i], vals['n_Dofs'][i],
                      vals['L1_Error'][i], vals['L2_Error'][i], vals['Linf_Error'][i]])

        s_table.append([vals['n_processes'][i], 
                        vals['Processor_Runtime'][i],
                        vals['Speedup'][i]])

    p_table_tab = tabulate(p_table, 
                       floatfmt=(".0f", ".3f", ".0f", ".12f", ".12f", ".12f"), 
                       headers=["# processors",
                                "Endtime", "# Dofs", "L1 Error",
                                "L2 Error", "L-Inf Error"],
                       tablefmt="latex")
    
    s_table_tab = tabulate(s_table, 
                           floatfmt=(".0f", ".3f", ".3f"),
                           headers=["# processors", "Wall-clock time", "Speedup"],
                           tablefmt="latex")

    # Output table to console
    print("             ")
    print(p_table_tab)
    print("             ")
    print(s_table_tab)

    # Output table to txt file
    f = open("../parallelization_table.txt", "w+")
    f.write(p_table_tab)
    f.close()

    g = open("../speedup_table.txt", "w+")
    g.write(s_table_tab)
    g.close()

# then we put main at the bottom to run everything
main()
