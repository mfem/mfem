#!/usr/bin/env python3

# Author: William F Godoy godoywf@ornl.gov
# Documents the list of mfem examples with adios2 outputs
# ex12p: adios2stream interface for saving temporaries
# all others in this list: adios2datacollection interface

# Usage: Run from mfem-build/examples
# ./run-adios2-examples.py example
# Example:
# To run examples ex9p: ./run_adios2stream.py ex9p
# To run all examples: ./run_adios2stream.py all

import argparse
import subprocess
import os
import glob
import shutil
from genericpath import exists

# cases covered with Paraview, if commented out it's either not covered or a bug
# not covered: mixed element type cases, 1D segment elements

cases = {
         # THIS IS A LARGE CASE FOR MY PC, run paraview in parallel using a server-client mode
         'ex5p': ['square-disc.mesh',
                  'star.mesh',
                  'beam-tet.mesh',
                  'beam-hex.mesh',
                  'escher.mesh',
                  'fichera.mesh',
                  ],

         'ex9p': [# Not supported 'star-mixed.mesh -p 1 -rp 1 -dt 0.004 -tf 9',
                  'periodic-square.mesh -p 0 -dt 0.01',
                  'periodic-segment.mesh -p 0 -dt 0.005',
                  'periodic-hexagon.mesh -p 0 -dt 0.01',
                  'periodic-square.mesh -p 1 -dt 0.005 -tf 9',
                  'periodic-hexagon.mesh -p 1 -dt 0.005 -tf 9',
                  'amr-quad.mesh -p 1 -rp 1 -dt 0.002 -tf 9',
                  'star-q3.mesh -p 1 -rp 1 -dt 0.004 -tf 9',
                  'disc-nurbs.mesh -p 1 -rp 1 -dt 0.005 -tf 9',
                  'disc-nurbs.mesh -p 2 -rp 1 -dt 0.005 -tf 9',
                  'periodic-square.mesh -p 3 -rp 2 -dt 0.0025 -tf 9 -vs 20',
                  'periodic-cube.mesh -p 0 -o 2 -rp 1 -dt 0.01 -tf 8',
                  ],

         # Uses adios2stream interface to save temporaries
         'ex12p': ['beam-tri.mesh',
                   'beam-quad.mesh',
                   'beam-tet.mesh -s 462 -n 10 -o 2 -elast',
                   'beam-hex.mesh -s 3878',
                   'beam-wedge.mesh -s 81',
                   'beam-tri.mesh -s 3876 -o 2 -sys',
                   'beam-quad.mesh -s 4544 -n 6 -o 3 -elast',
                   'beam-quad-nurbs.mesh',
                   'beam-hex-nurbs.mesh',
                   ],

         'ex16p': [# Not supported '-m ../data/fichera-mixed.mesh',
                   # Seg fault '-m ../data/amr-hex.mesh -o 2 -rs 0 -rp 0',
                   '-m ../data/inline-tri.mesh',
                   '-m ../data/disc-nurbs.mesh -tf 2',
                   '-s 1 -a 0.0 -k 1.0',
                   '-s 2 -a 1.0 -k 0.0',
                   '-s 3 -a 0.5 -k 0.5 -o 4',
                   '-s 14 -dt 1.0e-4 -tf 4.0e-2 -vs 40',
                   '-m ../data/fichera-q2.mesh',
                   '-m ../data/escher-p2.mesh',
                   '-m ../data/beam-tet.mesh -tf 10 -dt 0.1',
                   '-m ../data/amr-quad.mesh -o 4 -rs 0 -rp 0',
                   ],
         }


def ArgParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("example", type=str, help='example executable')
    args = parser.parse_args()
    return args


def Run(example):
    if example.endswith('p'):
        run_command = 'mpirun '
        run_prefix = '-n 4 ./' + example
    else:
        run_command = './' + example
        run_prefix = ''

    for case in cases[example]:
        if example == 'ex7p' or example == 'ex16p' or example == 'ex18p' or example == 'ex20p':
            run_args = run_prefix + ' ' + case
        else:
            run_args = run_prefix + ' -m ../data/' + case
        
        run_args += ' -adios2'
        subprocess.check_call(run_command + run_args, shell=True)

    outdir = 'mfem-adios2_dc-'+example
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for bp in glob.glob(r'*.bp'):
        shutil.move(bp, outdir+'/'+bp)


if __name__ == "__main__":

    args = ArgParser()
    example = str(args.example)
    
    if example == 'all':
        for case in cases.keys():
            Run(case)
    else:
        if example in cases.keys():
            Run(example)
        else:
            raise ValueError('Example binary' + example + 'not found, check input')
        
