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
import matplotlib.pyplot as plt
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
    plot_1(vals)


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

##
# Function to plot the L2 error with respect to space discretization.
##
def plot_1(vals):
    plt.plot(vals['n_refinements'], vals['L1_Error'], label='$L_1$ Error')
    plt.plot(vals['n_refinements'], vals['L2_Error'], label='$L_2$ Error')
    plt.plot(vals['n_refinements'], vals['Linf_Error'], label='$L_{\infty}$ Error')
    plt.xlabel('# Refinements', fontsize=16)
    plt.ylabel('Error', fontsize=16)
    plt.title('Approximation Error', fontsize=20)
    plt.legend()
    plt.yscale('log')
    plt.show()

    plt.plot(vals['n_refinements'], vals['L1_Rates'], label='$L_1$')
    plt.plot(vals['n_refinements'], vals['L2_Rates'], label='$L_2$')
    plt.plot(vals['n_refinements'], vals['Linf_Rates'], label='$L_{\infty}$')
    plt.xlabel('# Refinements', fontsize=16)
    plt.ylabel('Convergence Rate', fontsize=16)
    plt.title('Convergence Rates', fontsize=20)
    plt.legend()
    plt.show()

def compute_rates(vals):
    for i in range(len(vals['h'])):
        if i == 0:
            L1_rate = 0.
            L2_rate = 0.
            Linf_rate = 0.
        else:
            L1_rate = np.log(vals['L1_Error'][i]/vals['L1_Error'][i-1]) / np.log(vals['h'][i]/vals['h'][i-1])
            L2_rate = np.log(vals['L2_Error'][i]/vals['L2_Error'][i-1]) / np.log(vals['h'][i]/vals['h'][i-1])
            Linf_rate = np.log(vals['Linf_Error'][i]/vals['Linf_Error'][i-1]) / np.log(vals['h'][i]/vals['h'][i-1])
            
        vals['L1_Rates'].append(L1_rate)
        vals['L2_Rates'].append(L2_rate)
        vals['Linf_Rates'].append(Linf_rate)

# then we put main at the bottom to run everything
main()
