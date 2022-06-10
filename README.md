                    Finite Element Discretization Library
                                   __
                       _ __ ___   / _|  ___  _ __ ___
                      | '_ ` _ \ | |_  / _ \| '_ ` _ \
                      | | | | | ||  _||  __/| | | | | |
                      |_| |_| |_||_|   \___||_| |_| |_|

                               https://mfem.org

[MFEM](https://mfem.org) is a modular parallel C++ library for finite element
methods. Its goal is to enable high-performance scalable finite element
discretization research and application development on a wide variety of
platforms, ranging from laptops to supercomputers.

We welcome contributions and feedback from the community. Please see the file
[CONTRIBUTING.md](CONTRIBUTING.md) for additional details about our development
process.

* For building instructions, see the file [INSTALL](INSTALL), or type "make help".

* Copyright and licensing information can be found in files [LICENSE](LICENSE) and [NOTICE](NOTICE).

* The best starting point for new users interested in MFEM's features is to
  review the examples and miniapps at https://mfem.org/examples.

* Instructions for learning with Docker are in [config/docker](config/docker).

Conceptually, MFEM can be viewed as a finite element toolbox that provides the
building blocks for developing finite element algorithms in a manner similar to
that of MATLAB for linear algebra methods. In particular, MFEM provides support
for arbitrary high-order H1-conforming, discontinuous (L2), H(div)-conforming,
H(curl)-conforming and NURBS finite element spaces in 2D and 3D, as well as many
bilinear, linear and nonlinear forms defined on them. It enables the quick
prototyping of various finite element discretizations, including Galerkin
methods, mixed finite elements, Discontinuous Galerkin (DG), isogeometric
analysis, hybridization and Discontinuous Petrov-Galerkin (DPG) approaches.

MFEM includes classes for dealing with a wide range of mesh types: triangular,
quadrilateral, tetrahedral and hexahedral, as well as surface and topologically
periodical meshes. It has general support for mesh refinement, including local
conforming and non-conforming (AMR) adaptive refinement. Arbitrary element
transformations, allowing for high-order mesh elements with curved boundaries,
are also supported.

When used as a "finite element to linear algebra translator", MFEM can take a
problem described in terms of finite element-type objects, and produce the
corresponding linear algebra vectors and fully or partially assembled operators,
e.g. in the form of global sparse matrices or matrix-free operators. The library
includes simple smoothers and Krylov solvers, such as PCG, MINRES and GMRES, as
well as support for sequential sparse direct solvers from the SuiteSparse
library. Nonlinear solvers (the Newton method), eigensolvers (LOBPCG), and
several explicit and implicit Runge-Kutta time integrators are also available.

MFEM supports MPI-based parallelism throughout the library, and can readily be
used as a scalable unstructured finite element problem generator. Starting with
version 4.0, MFEM offers support for GPU acceleration, and programming models,
such as CUDA, HIP, OCCA, RAJA and OpenMP. MFEM-based applications require
minimal changes to switch from a serial to a highly-performant MPI-parallel
version of the code, where they can take advantage of the integrated linear
solvers from the hypre library. Comprehensive support for other external
packages, e.g. PETSc, SUNDIALS and libCEED is also included, giving access to
additional linear and nonlinear solvers, preconditioners, time integrators, etc.

For examples of using MFEM, see the [examples/](examples) and [miniapps/](miniapps)
directories, as well as the OpenGL visualization tool GLVis which is available
at https://glvis.org.

## License

MFEM is distributed under the terms of the BSD-3 license. All new contributions
must be made under this license. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for
details.

SPDX-License-Identifier: BSD-3-Clause <br>
LLNL Release Number: LLNL-CODE-806117 <br>
DOI: 10.11578/dc.20171025.1248
