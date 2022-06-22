"""
This python script file performs a convergence rates analsyis by running
a reading output files and composing a table from those values.

Example:
> python create_convergence_table.py "/Users/sheridan7/Workspace/mfem/examples/ex9p-analysis/temp_output/"
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
    table = []
    for i in range(len(vals['h'])):
        if i == 0:
            table.append([vals['n_Dofs'][i], vals['L1_Error'][i], "---", vals['L2_Error'][i], "---",
                          vals['Linf_Error'][i], "---"])
        else:
            L1_rate = np.log(vals['L1_Error'][i]/vals['L1_Error'][i-1]) / np.log(vals['n_Dofs'][i-1]/vals['n_Dofs'][i])
            L2_rate = np.log(vals['L2_Error'][i]/vals['L2_Error'][i-1]) / np.log(vals['n_Dofs'][i-1]/vals['n_Dofs'][i])
            Linf_rate = np.log(vals['Linf_Error'][i]/vals['Linf_Error'][i-1]) / np.log(vals['n_Dofs'][i-1]/vals['n_Dofs'][i])
            table.append([vals['n_Dofs'][i], vals['L1_Error'][i], L1_rate,
                          vals['L2_Error'][i], L2_rate,
                          vals['Linf_Error'][i], Linf_rate])

    s_table = tabulate(table, 
                       headers=["# dof", "L1 Error", "Rate", "L2 Error",
                                "Rate", "L-Inf Error", "Rate"],
                       tablefmt="latex")

    # Output table to console
    print("             ")
    print(s_table)

    # Output table to txt file
    f = open("../convergence_rates.txt", "w+")
    f.write(s_table)
    f.close()

# then we put main at the bottom to run everything
main()
