# MFEM - Modular Finite Element Methods Library

## Project Overview

MFEM is a modular parallel C++ library for finite element methods. Its goal is to enable high-performance scalable finite element discretization research and application development on a wide variety of platforms, ranging from laptops to supercomputers.

**Website:** https://mfem.org
**Documentation:** https://docs.mfem.org
**License:** BSD-3-Clause

## What MFEM Does

MFEM provides the building blocks for developing finite element algorithms in a manner similar to MATLAB for linear algebra. Key capabilities:

- **Finite Element Spaces:** Arbitrary high-order H1-conforming, discontinuous (L2), H(div)-conforming, H(curl)-conforming, and NURBS finite element spaces in 2D and 3D
- **Mesh Support:** Triangular, quadrilateral, tetrahedral, hexahedral, surface, and topologically periodic meshes with support for conforming and non-conforming (AMR) adaptive refinement
- **Discretizations:** Galerkin methods, mixed finite elements, Discontinuous Galerkin (DG), isogeometric analysis, hybridization, and Discontinuous Petrov-Galerkin (DPG)
- **Parallel Computing:** MPI-based parallelism throughout the library
- **GPU Acceleration:** Support for CUDA, HIP, OCCA, RAJA, and OpenMP backends
- **Linear Algebra:** Integration with hypre, PETSc, SUNDIALS, SuiteSparse, SuperLU, and other libraries
- **Time Integration:** Explicit and implicit Runge-Kutta methods, ODE and DAE solvers

## Repository Structure

```
.
├── fem/              # Finite element classes and operators
│   ├── ceed/        # libCEED integration
│   ├── dfem/        # Discontinuous finite elements
│   ├── fe/          # Finite element definitions
│   ├── integ/       # Bilinear and linear form integrators
│   └── tmop/        # Target-matrix optimization paradigm
├── mesh/            # Mesh classes and operations
│   └── submesh/     # Submesh support
├── linalg/          # Linear algebra classes
│   ├── batched/     # Batched linear algebra operations
│   └── simd/        # SIMD vectorization support
├── general/         # Utility classes (arrays, communication, timing, etc.)
├── config/          # Build configuration files
│   ├── cmake/       # CMake configuration
│   ├── docker/      # Docker configurations
│   └── githooks/    # Git hooks for development
├── data/            # Example mesh files
├── doc/             # Doxygen documentation configuration
├── examples/        # Simple example codes demonstrating features
├── miniapps/        # More complex application demonstrations
│   ├── electromagnetics/  # EM solvers
│   ├── meshing/           # Mesh generation and optimization
│   ├── navier/            # Navier-Stokes solver
│   ├── nurbs/             # NURBS examples
│   ├── performance/       # Performance benchmarks
│   └── solvers/           # Advanced solver examples
└── tests/           # Unit tests and benchmarks
    ├── unit/        # Unit test suite
    ├── benchmarks/  # Performance benchmarks
    └── scripts/     # Test scripts
```

## Building MFEM

### Quick Start (GNU Make)

**Serial build:**
```bash
make serial -j 4
```

**Parallel build (requires hypre and METIS):**
```bash
make parallel -j 4
```

**CUDA build:**
```bash
make cuda -j 4
# For specific compute capability: make cuda -j 4 CUDA_ARCH=sm_70
```

**HIP build:**
```bash
make hip -j 4
# For specific AMD GPU: make hip -j 4 HIP_ARCH=gfx900
```

**Build and test:**
```bash
make all -j 4      # Build library, examples, and miniapps
make check         # Quick test with Example 1
make test          # Run all tests
```

### Quick Start (CMake)

**Serial build:**
```bash
mkdir build && cd build
cmake ..
make -j 4
```

**Parallel build:**
```bash
mkdir build && cd build
cmake .. -DMFEM_USE_MPI=YES
make -j 4
```

**CUDA build (requires CMake 3.17+):**
```bash
mkdir build && cd build
cmake .. -DMFEM_USE_CUDA=YES -DCUDA_ARCH=sm_70
make -j 4
```

**Test the build:**
```bash
make check         # Quick test
make exec -j 4     # Build all examples/miniapps
make test          # Run full test suite
```

## Key Dependencies

### Required (for parallel build)
- **hypre** (>= 2.10.0b): High-performance preconditioners
- **METIS** (4.0.3 or 5.1.0): Graph partitioning

### Optional Libraries
- **SUNDIALS** (>= 5.0.0): ODE/DAE solvers and nonlinear solvers
- **PETSc** (>= 3.21.0): Linear/nonlinear solvers, preconditioners
- **SuiteSparse** (>= 4.5.4): Sparse direct solvers (UMFPACK, KLU)
- **SuperLU_DIST** (>= 5.1.0): Parallel sparse direct solver
- **libCEED**: High-order operator evaluation
- **OCCA**, **RAJA**: Performance portability layers
- **Ginkgo** (>= 1.9.0): GPU linear solvers
- **PUMI**: Parallel unstructured mesh infrastructure
- **Conduit**, **ADIOS2**, **HDF5**: I/O and visualization

## Development Workflow

### Code Style
- Run `make style` before committing (requires Artistic Style 3.1)
- See `config/mfem.astylerc` for style configuration

### Testing
```bash
make unittest              # Run unit tests
./tests/scripts/runtest    # Run additional test scripts
```

### Branch Strategy
- **master**: Stable, release-quality code
- **next**: Integration branch for testing approved PRs
- Feature branches: `feature-name-dev`

### Contributing
1. Fork or create a branch in the MFEM organization (preferred)
2. Follow the [Developer Guidelines](CONTRIBUTING.md)
3. Submit PR to `mfem:master`
4. Add `ready-for-review` label when ready
5. PRs are reviewed by 2+ reviewers (3-week review window)
6. After approval, PRs are tested in `next` for 1 week before merging to `master`

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## Common Development Tasks

### Adding a New Example
1. Create the example file in `examples/`
2. Update `examples/makefile` and `examples/CMakeLists.txt`
3. Add documentation in `.dox` files
4. Create companion PR in [mfem/web](https://github.com/mfem/web) repo

### Adding a New Miniapp
1. Create directory and source files in `miniapps/`
2. Update makefiles and CMake files
3. Add to `.gitignore`
4. Document in `doc/CodeDocumentation.dox`
5. Create companion PR in [mfem/web](https://github.com/mfem/web) repo

### Documentation
- All public/protected/private members need Doxygen documentation
- Use LaTeX (`$...$`, `$$...$$`) or Unicode for math
- Include ownership/lifetime information for pointers
- Document limitations and assumptions
- Build docs: `cd doc && make`

## Important Configuration Options

### GNU Make Options
```makefile
MFEM_USE_MPI        = YES/NO    # Enable MPI parallelism
MFEM_USE_CUDA       = YES/NO    # Enable CUDA support
MFEM_USE_HIP        = YES/NO    # Enable HIP (AMD GPU) support
MFEM_USE_OPENMP     = YES/NO    # Enable OpenMP
MFEM_USE_SUNDIALS   = YES/NO    # Enable SUNDIALS integrators
MFEM_USE_PETSC      = YES/NO    # Enable PETSc solvers
MFEM_USE_LAPACK     = YES/NO    # Use LAPACK
MFEM_DEBUG          = YES/NO    # Debug vs optimized build
MFEM_PRECISION      = double/single  # Floating-point precision
```

### CMake Options
```bash
-DMFEM_USE_MPI=YES              # Enable MPI
-DMFEM_USE_CUDA=YES             # Enable CUDA
-DMFEM_USE_HIP=YES              # Enable HIP
-DMFEM_FETCH_TPLS=YES           # Auto-fetch dependencies
-DMFEM_ENABLE_TESTING=YES       # Enable ctest framework
-DMFEM_ENABLE_EXAMPLES=YES      # Build examples by default
-DMFEM_ENABLE_MINIAPPS=YES      # Build miniapps by default
```

## Key Classes

### Mesh Classes
- `Mesh`: Serial mesh representation
- `ParMesh`: Parallel mesh (MPI)
- `NCMesh`: Non-conforming mesh (AMR support)
- `Element`: Base element class
- `ElementTransformation`: Element mappings

### Finite Element Classes
- `FiniteElement`: Base FE class
- `FiniteElementCollection`: Collection of FEs
- `FiniteElementSpace`: FE space on a mesh
- `GridFunction`: FE function
- `BilinearForm`, `LinearForm`: Discrete forms
- `BilinearFormIntegrator`, `LinearFormIntegrator`: Weak form integrators

### Linear Algebra Classes
- `Vector`, `DenseMatrix`, `SparseMatrix`: Basic linear algebra
- `Operator`: Abstract operator interface
- `Solver`: Abstract solver interface
- `HypreParMatrix`, `HypreParVector`: Parallel hypre wrappers
- Various preconditioners and Krylov solvers

### Parallel Classes (MPI)
- `ParMesh`, `ParNCMesh`
- `ParFiniteElementSpace`
- `ParGridFunction`
- `ParBilinearForm`, `ParLinearForm`
- `HypreParMatrix`, `HypreParVector`

## Performance and GPU Support

MFEM uses a device abstraction layer for performance portability:
- `Device` class: Manage execution and memory policies
- `MemoryManager`: Device/host memory management
- `mfem::forall`: Portable parallel loops
- Backends: CUDA, HIP, RAJA, OCCA, OpenMP

## Running Examples

Examples are in `examples/` with sample runs in comments at the top of each file:

```bash
cd examples
make ex1
./ex1 -m ../data/star.mesh
./ex1 -m ../data/beam-tri.mesh -o 2
```

For parallel examples:
```bash
make ex1p
mpirun -np 4 ./ex1p -m ../data/star.mesh
```

## Visualization

MFEM works with **GLVis** (OpenGL visualization tool):
- Website: https://glvis.org
- Real-time visualization of meshes and solutions
- Start GLVis server: `glvis`
- MFEM examples automatically connect to GLVis when available

## Resources

- **Website:** https://mfem.org
- **Documentation:** https://docs.mfem.org
- **Examples:** https://mfem.org/examples
- **Code overview:** https://mfem.org/code-overview
- **Building guide:** https://mfem.org/building
- **GitHub Issues:** https://github.com/mfem/mfem/issues
- **Mailing List:** mfem-dev@llnl.gov

## Citing MFEM

If you use MFEM in your research, please cite the paper and DOI listed in [CITATION.cff](CITATION.cff).
