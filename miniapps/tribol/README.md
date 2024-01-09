# Tribol Interface Physics Library Mini-applications in MFEM

High fidelity simulations modeling complex interactions of moving bodies require
specialized contact algorithms to enforce constraints between surfaces that come
into contact in order to prevent penetration and to compute the associated
contact response forces. Tribol aims to provide a unified interface for various
contact algorithms, specifically, contact detection and enforcement, and serve
as a common infrastructure enabling the research and development of advanced
contact algorithms. More information about Tribol can be found on [its Github
repo](https://github.com/LLNL/Tribol).

Tribol uses MFEM's data structures to simplify interoperability with MFEM-based
finite element codes.  As a result, Tribol has a dependency on MFEM.  Other than
this miniapp, MFEM has no dependencies on Tribol.  While a single MFEM install
can be used to build and link both Tribol and this miniapp, it is generally
simpler to allow Tribol's build system to build its own MFEM, then build the
miniapp in a separately cloned repo. The installation instructions below follow
this procedure.

This directory contains an example using Tribol's mortar method to solve a
contact patch test.  The mortar implementation is based on [Puso and Laursen
(2004)](https://doi.org/10.1016/j.cma.2003.10.010).  A description of the Tribol
implementation is available in [Serac
documentation](https://serac.readthedocs.io/en/latest/sphinx/theory_reference/solid.html#contact-mechanics).
Lagrange multipliers are used to solve for the pressure required to prevent
violation of the contact constraints.

## Installation

Tribol has dependencies on [Axom](https://github.com/LLNL/axom) and MFEM.  The
simplest way to satisfy these dependencies is to follow the build instructions
provided in the Tribol repo.  This will use uberenv and spack to build
dependencies (and their dependencies) mostly automatically.  To simplify Axom's
dependencies, the following spack spec is recommended: `^axom~examples~tools`.
Furthermore, the version of MFEM built by spack should match the version of MFEM
you are using.  For instance, if you are tracking the latest `master` branch,
the following spack spec is recommended: `^mfem@develop`.  After building
Tribol, issue the command `make install` to simplify building and linking in
MFEM.

For the Tribol MFEM miniapps to successfully build and link, the MFEM build must
be aware of the locations of not only Tribol, but also its dependencies Axom and
Conduit. If the uberenv and spack directions are used to build Tribol, these
dependencies will be located in
`<TRIBOL_ROOT>/install-<COMPUTER>-<OS>-<COMPILER>`,
`<TRIBOL_ROOT>/../tribol_libs/spack/opt/spack/<OS>/<COMPILER>/axom-<AXOM_VER>-<AXOM_HASH>`,
and
`<TRIBOL_ROOT>/../tribol_libs/spack/opt/spack/<OS>/<COMPILER>/conduit-<CONDUIT_VER>-<CONDUIT_HASH>`,
where `<TRIBOL_ROOT>`, `<COMPUTER>`, `<OS>`, `<COMPILER>`, `<AXOM_VER>`,
`<AXOM_HASH>`, `<CONDUIT_VER>`, and `<CONDUIT_HASH>` will be specific to each
computer. Below, instructions are provided for both building with GNU Make and
with CMake.

### Using GNU Make

Refer to the MFEM instructions in `<MFEM_ROOT>/INSTALL` on how to modify the
configuration in `<MFEM_ROOT>/config/`. Particularly, `MFEM_USE_TRIBOL` and
`MFEM_USE_MPI` need to be set to `YES` and the variables `CONDUIT_DIR`,
`AXOM_DIR`, and `TRIBOL_DIR` need to be set to the install directories for
Conduit, Axom, and Tribol, respectively.

After making these modifications, the miniapp should build using the
instructions in `<MFEM_ROOT>/INSTALL`.

### Using CMake

Refer to the MFEM instructions in `<MFEM_ROOT>/INSTALL` on how to modify the
configuration in `<MFEM_ROOT>/config/`. Particularly, `MFEM_USE_TRIBOL` and
`MFEM_USE_MPI` need to be set to `ON` and the variables `CONDUIT_DIR`,
`AXOM_DIR`, and `TRIBOL_DIR` need to be set to the install directories for
Conduit, Axom, and Tribol, respectively.

After making these modifications, the miniapp should build using the
instructions in `<MFEM_ROOT>/INSTALL`.

## Usage

The miniapp can be run by invoking e.g. `mpirun -n 2
<MFEM_BUILD>/miniapps/tribol/ContactPatchTest`. Note the mesh file
`<MFEM_ROOT>/miniapps/tribol/two-hex.mesh` must be in your path for the miniapp
to execute correctly. This mesh file contains two cubes occupying $[0,1]^3$ and
$[0,1] \times [0,1] \times [0.99,1.99]$. By default, the miniapp will uniformly
refine the mesh twice, then split it across MPI ranks. Additional command line
options can be viewed in the source file,
`<MFEM_ROOT>/miniapps/tribol/ContactPatchTest.cpp`. An elasticity bilinear form
will be created over the volume mesh and mortar contact constraints will be
formed along the $z=1$ and $z=0.99$ surfaces of the blocks.

Given the elasticity stiffness matrix and the gap constraint and constraint
derivatives from Tribol, the miniapp will form and solve a linear system of
equations for updated displacements and pressures. Finally, it will verify force
equilibrium and that the gap constraints are satisfied and save output in VisIt
format.