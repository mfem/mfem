# Code Overview

This miniapp solves extended MHD equations use finite element methods and physics-based 
preconditioning. It is a part of the SciDAC project of [Tokamak Disruption Simulation](https://www.scidac.gov/projects/2018/fusion-energy-sciences/tokamak-disruption-simulation.html)

It currently supports a 2D reduced resistive MHD model. For details of the model as well as physics-based
preconditioning, see [Chacón, Luis, Dana A. Knoll, and J. M. Finn. "An implicit, nonlinear reduced resistive MHD solver." JCP 2002].
Other MHD models will be developed soon.

### Some highlights of the miniapp
* Both implicit and explicit MHD solvers are available
  * Implicit solvers support backward Euler, implicit midpoint and SDIRK
  * Explicit solvers support a predictor-corrector Brailovskaya’s scheme
* Scalable physics-based preconditioning
* Structured and unstructured (high-order) meshes
* Dynamic AMR: conforming and nonconforming; refine and derefine; load-balancing
* Demonstrated excellent scalability
* [GLVis](https://glvis.org) and [Visit](https://wci.llnl.gov/simulation/computer-codes/visit/) plotting routines

*For more questions, contact Qi Tang (qtang@lanl.gov or @tangqi)*

# Installation Instructions

To obtain a clone of the MHD miniapp, use:

    git clone https://github.com/mfem/mfem.git
    cd mfem
    git checkout --track origin/tds-mhd-dev

see [MFEM building page](https://mfem.org/building) for further details on how to compile MFEM.
Note that it is necessary to (re)build mfem in the tds-mhd-dev branch.
The master version may miss some key functions.

The following instructions are ordered based on the complexity 
* A serial solver (exMHD)
* A parallel explicit solver (exMHDp)
* A parallel implicit solver (imMHDp)
TODO
*Instructions on amr solvers (imAMRMHDp and exAMRMHDp)*

## Serial building

A serial MHD solver can be compiled and tested using:
```
cd mfem
git checkout tds-mhd-dev
make serial -j
cd miniapps/mhd
make exMHD
./exMHD -r 2 -tf 10 -dt .004 -i 2
glvis -m refined.mesh -g phi.sol -k "mmc" -ww 800 -wh 800
```
GLVis should show a surface plot of omega at t=10.

We recommend users start from the serial version,
since the serial code is simple and self-contained.
However, the serial code only supports explicit solvers (although linked with AMR).


## Parallel building



### Building explicit solvers only
The parallel explicit solvers only need hypre and metis. 
See [MFEM building page](https://mfem.org/building) for further details on how to compile and link these two packages. An example identical to the serial case can be built and tested by
```
make parallel -j
cd miniapps/mhd
make exMHDp
mpirun -np 4 exMHDp -rs 2 -tf 10 -dt .004 -i 2
glvis -np 4 -m mesh -g sol_phi -k "mmc" -ww 800 -wh 800
```
One should obtain a surface plot identical to the serial case.

The parallel AMR solver can be built and tested by
```
make exAMRMHDp
mpirun -n 4 exAMRMHDp -m Meshes/xperiodicR1.mesh -o 4 -tf 1 -dt .0001 -i 3 -amrl 3 -ltol 2e-3 -derefine
```

### Building both explicit and implicit solvers
Install PETSc with hypre
```
cd petsc-3.11.1
./configure --with-prefix=$PWD --with-mpi-dir=/packages/mpi/openmpi-2.1.2-gcc-7.2.0 --with-fc=0 --download-hypre --with-valgrind=0 --with-debugging=0
make PETSC_DIR=/nh/u/qtang/code/test/petsc-3.11.1 PETSC_ARCH=arch-linux2-c-debug all
```
Link PETSc and hypre to the directory above mfem
```
export PETSC_DIR=/nh/u/qtang/code/test/petsc-3.11.1
export PETSC_ARCH=arch-linux2-c-debug
ln -s $PETSC_DIR/$PETSC_ARCH/externalpackages/git.hypre hypre
ln -s $PETSC_DIR petsc
```
Build MFEM
```
make config MFEM_USE_MPI=YES MPICXX=mpicxx MFEM_USE_PETSC=YES
make -j
cd miniapps/mhd
make exMHDp
make imMHDp
mpirun -n 4 imMHDp -rs 4 -o 2 -i 2 -tf 10 -dt 5 -usepetsc --petscopts petscrc/rc_full -s 3 -shell
glvis -np 4 -m mesh -g sol_phi -k "mmc" -ww 800 -wh 800
```
This example is similar with the explicit run except for refined by a factor of 2, but we note that we take a much larger time step dt=5. It demonstrates the advantage of physics-based preconditioning.
Implicit solvers will show a even more significant speedup as we refine the mesh.

#### Building on NERSC
The changes are
```
./configure CC=cc CXX=CC --with-fc=0 --download-hypre --with-debugging=0 --with-valgrind=0 --with-cxx-dialect=C++11
make config MFEM_USE_MPI=YES MFEM_DEBUG=YES MFEM_USE_PETSC=YES MPICXX=CC MFEM_MPIEXEC=srun MFEM_MPIEXEC_NP=-n
```
The rest steps are identical.

# FAQs

#### How could I visualize the solutions?
Two options are provided:
* GLVis
  * Before the run, if one opens a glvis in another terminal, some solutions will be plotted automatically.
  * Alternatively, after the run is finished,  solutions can be plotted by, for instance, 
```
glvis -np 4 -m mesh -g sol_phi -k "mmc" -ww 800 -wh 800
```
* Visit: 
  adding a flag `-visit` will output all the solutions in the visit format.


#### What examples are provided?
* Case `-i 1`: Wave Propagation (5.1.1) in [An implicit, nonlinear reduced resistive MHD solver. JCP 2002]. Identical setup.
* Case `-i 2` Tearing Mode (5.1.2) in [An implicit, nonlinear reduced resistive MHD solver. JCP 2002]. Identical setup.
* Case `-i 3`: Island Coalescence (5.2) in [Implicit adaptive mesh refinement for 2D reduced resistive magnetohydrodynamics, JCP 2008]. Identical setup.

#### Where has the code been built and tested?
* Mac OS
* Linux 
* Many HPC including LANL and NERSC machines

#### Why do the implicit solvers complain about hypre? 
It is likely there are two different versions of hypre linked in the code.
One needs to make sure there is only one version of hypre seen by mfem.
We recommend to link mfem with the hypre installed by PETSc.
Please also avoid linking your local hypre with PETSc.



#### Why mfem produces an error on cori at NERSC?
There is a known compiler bug (https://github.com/mfem/mfem/issues/956) on cori. It can be fixed by
```
--- a/linalg/densemat.cpp
+++ b/linalg/densemat.cpp
@@ -3967,7 +3967,10 @@ void LUFactors::Factor(int m)
             // swap rows i and piv in both L and U parts
             for (int j = 0; j < m; j++)
             {
-               Swap<double>(data[i+j*m], data[piv+j*m]);
+               double tmp=data[i+j*m];
+               data[i+j*m]=data[piv+j*m];
+               data[piv+j*m]=tmp;
             }
```

#### Could I use glvis on cori?
Yes, recently we were able to use glvis on cori. Visit or paraview is also another option.



