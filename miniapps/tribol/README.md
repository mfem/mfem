# Tribol Interface Physics Library Mini-applications in MFEM

High fidelity simulations modeling complex interactions of moving bodies require
specialized contact algorithms to enforce zero-interpenetration constraints
between surfaces. Tribol provides a unified interface for various contact
algorithms, including contact search, detection and enforcement, thereby
enabling the research and development of advanced contact algorithms. More
information about Tribol can be found on [its Github
repo](https://github.com/LLNL/Tribol).

Tribol uses MFEM's data structures to simplify interoperability with MFEM-based
finite element codes. As a result, Tribol has a dependency on MFEM. Other than
this miniapp, MFEM has no dependencies on Tribol. To satisfy these somewhat
circular dependencies, the MFEM library must be installed first, followed by
Tribol (pointing to the MFEM library already built), and finally, the MFEM
miniapps, pointing to the newly built Tribol library. This readme details this
process.

This directory contains an example using Tribol's mortar method to solve a
contact patch test. A contact patch test places two aligned, linear elastic
cubes in contact, then verifies the exact elasticity solution for this problem
is recovered. The exact solution requires transmission of a uniform pressure
field across a (not necessarily conforming) interface (i.e. the contact
surface). Mortar methods (including the one implemented in Tribol) are generally
able to pass the contact patch test. The test assumes small deformations and no
accelerations, so the relationship between forces/contact pressures and
deformations/contact gaps is linear and, therefore, the problem can be solved
exactly with a single linear solve. The mortar implementation is based on [Puso
and Laursen (2004)](https://doi.org/10.1016/j.cma.2003.10.010). A description of
the Tribol implementation is available in [Serac
documentation](https://serac.readthedocs.io/en/latest/sphinx/theory_reference/solid.html#contact-mechanics).
Lagrange multipliers are used to solve for the pressure required to prevent
violation of the contact constraints.

## Installation

Tribol has dependencies on [Axom](https://github.com/LLNL/axom) and MFEM. Tribol
provides build instructions using uberenv and spack to automate the process of
building its TPLs. However, the use of spack complicates interacting with the
MFEM library built as a Tribol TPL. For running the miniapp, you may find it
simpler to manually build Axom and the MFEM library, then point a Tribol host
config file (i.e. a CMake Toolchain file) to them. This is the method described
in this section. The steps are as follows:

1. Download MFEM, hypre, and metis and install the library (and not the
   miniapps) with options `MFEM_USE_MPI`, `MFEM_USE_METIS`, and
   `MFEM_USE_TRIBOL`. See the file `INSTALL` and the following subsections for
   more details.
2. Download [Axom](https://github.com/LLNL/axom) and install without TPLs using
   the instructions below.
   1. Clone the repo. Starting from the MFEM root directory (we assume this
      directory is named `mfem`), type `cd .. && git clone --recursive
      https://github.com/LLNL/axom.git axom-repo && cd axom-repo`.
   2. Inspect (and edit, as needed) the `axom-gcc-notpl.cmake` host config file
      found in the current directory (i.e. the directory this readme file is
      in). Then run `python3 ./config-build.py -hc
      ../mfem/miniapps/tribol/axom-gcc-notpl.cmake -bt Release
      -DCMAKE_INSTALL_PREFIX=../../axom`.
   3. Build and install axom by typing `cd build-axom-gcc-notpl-release && make
      -j install`.
3. Download [Tribol](https://github.com/LLNL/Tribol) and install using the
   following instructions.
   1. Starting from the mfem root directory, type `cd .. && git clone
      --recursive https://github.com/LLNL/Tribol.git tribol-repo && cd
      tribol-repo` to clone Tribol.
   2. Inspect (and edit, as needed) the `tribol-gcc-basictpl.cmake` host config
      file found in the current directory (i.e. the directory this readme file
      is in). Then run `python3 ./config-build.py -hc
      ../mfem/miniapps/tribol/tribol-gcc-basictpl.cmake -bt Release
      -DCMAKE_INSTALL_PREFIX=../../tribol`.
   3. Finally, build and install Tribol: `cd build-tribol-gcc-basictpl-release
      && make -j install`.
4. Build the MFEM miniapps (usually by running `make -j miniapps` from the MFEM
   build directory if it exists, or the MFEM root directory).

### Using GNU Make

Refer to the MFEM instructions in `mfem/INSTALL` on how to modify the
configuration in `mfem/config/`. Particularly, `MFEM_USE_TRIBOL` and
`MFEM_USE_MPI` need to be set to `YES` and the variables `AXOM_DIR` and
`TRIBOL_DIR` need to be set to the install directories for Axom and Tribol,
respectively. Note the default install directories do not need to be changed if
the instructions above are followed.

After making these modifications, the miniapp should build using the
instructions above.

### Using CMake

Refer to the MFEM instructions in `mfem/INSTALL` on how to modify the
configuration in `mfem/config/`. Particularly, `MFEM_USE_TRIBOL` and
`MFEM_USE_MPI` need to be set to `ON` and the variables `AXOM_DIR` and
`TRIBOL_DIR` need to be set to the install directories for Axom and Tribol,
respectively. Note the default install directories do not need to be changed if
the instructions above are followed.

After making these modifications, the miniapp should build using the
instructions above.

## Usage

The miniapp can be run by invoking e.g. `mpirun -n 2
<MFEM_BUILD>/miniapps/tribol/contact-patch-test`. Note the mesh file
`mfem/miniapps/tribol/two-hex.mesh` must be in your path for the miniapp to
execute correctly. This mesh file contains two cubes occupying $[0,1]^3$ and
$[0,1] \times [0,1] \times [0.99,1.99]$. By default, the miniapp will uniformly
refine the mesh twice, then split it across MPI ranks. Additional command line
options can be viewed in the source file,
`mfem/miniapps/tribol/contact-patch-test.cpp`. An elasticity bilinear form will be
created over the volume mesh and mortar contact constraints will be formed along
the $z=1$ and $z=0.99$ surfaces of the blocks.

Given the elasticity stiffness matrix and the gap constraint and constraint
derivatives from Tribol, the miniapp will form and solve a linear system of
equations for updated displacements and contact pressures. Finally, it will
verify force equilibrium and that the gap constraints are satisfied and save
output in VisIt format.
