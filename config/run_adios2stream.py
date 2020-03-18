#!/usr/bin/env python3

# Author: William F Godoy godoywf@ornl.gov
# Documents the list of mfem examples for adios2stream output
# that have Paraview support

# Usage: Run from mfem-build/examples
# ./run_adios2stream.py example
# Example:
# To run examples ex2p: ./run_adios2stream.py ex2p
# To run all examples: ./run_adios2stream.py all

import argparse
import subprocess
import os
import glob
import shutil
from genericpath import exists

# cases covered with Paraview, if commented out it's either not covered or a bug
# not covered: mixed element type cases, 1D segment elements

cases = {'ex1p': [# Not yet supported 'star-mixed.mesh',
                  # Not yet supported 'fichera-mixed.mesh',
                  # Not yet supported 'star-mixed-p2.mesh -o 2',
                  # Not yet supported 'fichera-mixed-p2.mesh -o 2',
                  # Seg fault in Paraview 'mobius-strip.mesh -o -1 -sc',
                  # Seg fault in Paraview 'mobius-strip.mesh',
                  # Seg fault in Paraview 'star-surf.mesh',
                  # Seg fault in Paraview 'square-disc-surf.mesh',
                  'star.mesh',
                  'escher.mesh',
                  'fichera.mesh',
                  'toroid-wedge.mesh',
                  'square-disc.mesh',
                  'square-disc-p2.vtk -o 2',
                  'square-disc-p3.mesh -o 3',
                  'square-disc-nurbs.mesh -o -1',
                  'disc-nurbs.mesh -o -1',
                  'pipe-nurbs.mesh -o -1',
                  'ball-nurbs.mesh -o 2',
                  'inline-segment.mesh',
                  'amr-quad.mesh',
                  'amr-hex.mesh',
                  ],

         'ex2p': ['beam-tri.mesh',
                  'beam-quad.mesh',
                  'beam-tet.mesh',
                  'beam-hex.mesh',
                  'beam-wedge.mesh',
                  'beam-tri.mesh -o 2 -sys',
                  'beam-quad.mesh -o 3 -elast',
                  'beam-quad.mesh -o 3 -sc',
                  'beam-quad-nurbs.mesh',
                  'beam-hex-nurbs.mesh',
                  ],

         'ex3p': [# Not supported missing points 'star-surf.mesh -o 2',
                  # Not supported 'mobius-strip.mesh -o 2 -f 0.1',
                  # Not supported 'klein-bottle.mesh -o 2 -f 0.1'
                  'star.mesh',
                  'square-disc.mesh -o 2',
                  'beam-tet.mesh',
                  'beam-hex.mesh',
                  'escher.mesh',
                  'escher.mesh -o 2',
                  'fichera.mesh',
                  'fichera-q2.vtk',
                  'fichera-q3.mesh',
                  'square-disc-nurbs.mesh',
                  'beam-hex-nurbs.mesh',
                  'amr-quad.mesh -o 2',
                  'amr-hex.mesh',
                  ],

         'ex4p': [# Not supported 'star-surf.mesh -o 3 -hb',
                  'square-disc.mesh',
                  'star.mesh',
                  'beam-tet.mesh',
                  'beam-hex.mesh',
                  'escher.mesh -o 2 -sc',
                  'fichera.mesh -o 2 -hb',
                  'fichera-q2.vtk',
                  'fichera-q3.mesh -o 2 -sc',
                  'square-disc-nurbs.mesh -o 3',
                  'beam-hex-nurbs.mesh -o 3',
                  'periodic-square.mesh -no-bc',
                  'periodic-cube.mesh -no-bc',
                  'amr-quad.mesh',
                  'amr-hex.mesh -o 2 -sc',
                  'amr-hex.mesh -o 2 -hb',
                  ],

         # THIS IS A LARGE CASE FOR MY PC, run paraview in parallel using a server-client mode
         'ex5p': ['square-disc.mesh',
                  'star.mesh',
                  'beam-tet.mesh',
                  'beam-hex.mesh',
                  'escher.mesh',
                  'fichera.mesh',
                  ],

         # NOT WORKING, Problems with Refined grid and moving mesh
#                 'ex6p': ['square-disc.mesh -o 1',
#                          'square-disc.mesh -o 2',
#                          'square-disc-nurbs.mesh -o 2',
#                          'star.mesh -o 3',
#                          'escher.mesh -o 2',
#                          'fichera.mesh -o 2',
#                          'disc-nurbs.mesh -o 2',
#                          'ball-nurbs.mesh',
#                          'pipe-nurbs.mesh',
#                          'star-surf.mesh -o 2',
#                          'square-disc-surf.mesh -o 2',
#                          'amr-quad.mesh',
#                          ],

         # WORKING WITH FULL DATA, Order::byNODES -> Order::byVDIM
         # connectivity element ID blows up with refinement
         'ex7p': ['-e 0 -o 2 -r 4',
                  '-e 1 -o 2 -r 4 -snap',
                  '-e 0 -amr 1',
                  '-e 1 -amr 2 -o 2',
                  ],

         'ex8p': [# Not supported 'star-mixed.mesh',
                  # Not supported 'fichera-mixed.mesh',
                  # Not supported 'star-surf.mesh -o 2'
                  'square-disc.mesh',
                  'star.mesh',
                  'escher.mesh',
                  'fichera.mesh',
                  'square-disc-p2.vtk',
                  'square-disc-p3.mesh',
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

         'ex10p': ['beam-quad.mesh -s 3 -rs 2 -dt 3',
                   'beam-tri.mesh -s 3 -rs 2 -dt 3',
                   'beam-hex.mesh -s 2 -rs 1 -dt 3',
                   'beam-tet.mesh -s 2 -rs 1 -dt 3',
                   'beam-wedge.mesh -s 2 -rs 1 -dt 3',
                   'beam-quad.mesh -s 14 -rs 2 -dt 0.03 -vs 20',
                   'beam-hex.mesh -s 14 -rs 1 -dt 0.05 -vs 20',
                   'beam-quad-amr.mesh -s 3 -rs 2 -dt 3',
                   ],

         # Uses adios2stream interface to save temporaries
         'ex11p': [# Not supported 'star-mixed.mesh',
                   # Not working 'star-surf.mesh',
                   # Not supported 'fichera-mixed.mesh',
                   'square-disc.mesh',
                   'star.mesh',
                   'escher.mesh',
                   'fichera.mesh',
                   'toroid-wedge.mesh -o 2',
                   'square-disc-p2.vtk -o 2',
                   'square-disc-p3.mesh -o 3',
                   'square-disc-nurbs.mesh -o -1',
                   'disc-nurbs.mesh -o -1 -n 20',
                   'pipe-nurbs.mesh -o -1',
                   'ball-nurbs.mesh -o 2',
                   'square-disc-surf.mesh',
                   'inline-segment.mesh',
                   'inline-quad.mesh',
                   'inline-tri.mesh',
                   'inline-hex.mesh',
                   'inline-tet.mesh',
                   'inline-wedge.mesh -s 83',
                   'amr-quad.mesh',
                   'amr-hex.mesh',
                   # Refinement not working 'mobius-strip.mesh -n 8',
                   # Refinement not working 'klein-bottle.mesh -n 10',
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

         # Uses adios2stream interface to save temporaries
         'ex13p': [# Refinement not working 'mobius-strip.mesh -n 8 -o 2',
                   # Refinement not working 'klein-bottle.mesh -n 10 -o 2',
                   'star.mesh',
                   'square-disc.mesh -o 2',
                   'beam-tet.mesh',
                   'beam-hex.mesh',
                   'escher.mesh',
                   'fichera.mesh',
                   'fichera-q2.vtk',
                   'fichera-q3.mesh',
                   'square-disc-nurbs.mesh',
                   'beam-hex-nurbs.mesh',
                   'amr-quad.mesh -o 2',
                   'amr-hex.mesh',
                   ],

         'ex14p': [# not supported 'star-mixed.mesh -o 2',
                   # not supported 'fichera-mixed.mesh -s 1 -k 1',
                   'star.mesh -o 2',
                   'inline-quad.mesh -o 0',
                   'escher.mesh -s 1',
                   'fichera.mesh -s 1 -k 1',
                   'square-disc-p2.vtk -o 2',
                   'square-disc-p3.mesh -o 3',
                   'square-disc-nurbs.mesh -o 1',
                   'disc-nurbs.mesh -rs 4 -o 2 -s 1 -k 0',
                   'pipe-nurbs.mesh -o 1',
                   'inline-segment.mesh -rs 5',
                   'amr-quad.mesh -rs 3',
                   'amr-hex.mesh',
                   ],
         # Bugs as in ex6p
         #   'ex15p': []

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

         'ex17p': ['beam-tri.mesh',
                   'beam-quad.mesh',
                   'beam-tet.mesh',
                   'beam-hex.mesh',
                   'beam-wedge.mesh',
                   'beam-quad.mesh -rs 2 -rp 2 -o 3 -elast',
                   'beam-quad.mesh -rs 2 -rp 3 -o 2 -a 1 -k 1',
                   'beam-hex.mesh -rs 2 -rp 1 -o 2',
                   ],

         'ex18p': [# Seg fault '-p 1 -rs 1 -rp 1 -o 5 -s 6',
                   '',
                   '-p 1 -rs 2 -rp 1 -o 1 -s 3',
                   '-p 1 -rs 1 -rp 1 -o 3 -s 4',
                   '-p 2 -rs 1 -rp 1 -o 1 -s 3',
                   '-p 2 -rs 1 -rp 1 -o 3 -s 3',
                   ],

         'ex19p': ['beam-quad.mesh',
                   'beam-tri.mesh',
                   'beam-hex.mesh',
                   'beam-tet.mesh',
                   'beam-wedge.mesh',
                   ],
         # Connectivity issues, can't refine data, works with full data
         'ex20p': ['']
         }


def ArgParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("example", type=str, help='example executable')
    args = parser.parse_args()
    return args


def Run(example):
    if example.endswith('p'):
        run_command = 'mpirun '
        run_prefix = '-n 4 ' + example
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
        
