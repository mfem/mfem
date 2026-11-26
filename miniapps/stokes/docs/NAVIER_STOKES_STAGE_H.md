# Stage H: geometric velocity multigrid

`steady_navier_stokes_gmg.hpp` provides an actual nested-mesh velocity inverse
for the existing forward and supplied-state-adjoint block FGMRES solvers. It
does not change the nonlinear residual, exact Jacobian, pressure Schur strategy,
default P1-AMG inverse, or legacy miniapp defaults. This is velocity-block GMG,
not a coupled saddle-point or pressure multigrid.

The subsequent [scaling follow-up](GMG_SCALING.md) adds an opt-in
`GeometricCoarseSolver::MumpsRoot` strategy. The Stage H measurements below
describe the original distributed-MUMPS/fixed-Jacobi baseline.

The [linear Brinkman follow-up](STOKES_BRINKMAN_LINEAR.md) also adds symmetric
fixed-cycle BoomerAMG and a user coarse-inverse factory, plus explicit
auxiliary-space constraints for prescribed velocity DOF lists. Its tests cover
the new choices; the historical Stage H measurements below are unchanged.

## Supported hierarchy and ownership

`UniformVelocityHierarchy(coarse_mesh, order, refinements)` copies the caller's
parallel coarse mesh, then constructs successively uniformly refined meshes,
same-degree H1 velocity spaces and assembled Hypre true-DOF transfers. Degrees
2–4 and 1–8 refinements are accepted. The concrete route supports conforming
triangles/quadrilaterals and tetrahedra/hexahedra, affine or quadratic continuous
H1 geometry, CPU/full assembly, and velocity `Ordering::byVDIM`.

The hierarchy owns its copied meshes, collection, spaces and raw transfers.
The original coarse mesh can subsequently be destroyed or changed. Its finest
space must be the **exact object** used by `SteadyNavierStokesProblem`; this is
not a facility for inferring refinement history from an unrelated fine space.
Construct the pressure space on that finest mesh. Keep the hierarchy alive
longer than the pressure space, fields, problem, forward and adjoint solvers.
The existing coefficient-borrowing and revision contracts remain unchanged.

All levels use the original communicator and inherited refinement partitioning.
Every rank must own at least one coarse element. Nonconforming/hanging-node
meshes, embedded meshes, unsupported cell families, geometry degree above two,
and invalid level/degree choices are rejected. There is no repartitioning,
rank agglomeration, adaptive h/p hierarchy, or rediscretized level policy.
Periodic and mixed-cell hierarchies are outside the validation matrix.

Mesh/space sequence changes and in-place geometry/domain/boundary attribute
changes invalidate the hierarchy. Checked hierarchy getters and inverse
applications are collective; cache getters before uneven rank-local loops.
Geometry/attribute fingerprints are checked on application and their cost is
included in apply timing. Coefficient changes require the existing setters or
`NotifyCoefficientsChanged()`; retained inverses then reject stale revisions.
Transfers may be retained while numerical matrices and coarse solvers are
rebuilt for each new linear setup. No address-only numerical cache is used.

## Integration

The following pattern assumes an already constructed parallel `coarse_mesh`
and caller-owned viscosity coefficient:

```cpp
UniformVelocityHierarchy hierarchy(coarse_mesh, 2, 2);
auto &velocity_space = hierarchy.GetFinestSpace();
H1_FECollection pressure_collection(1, coarse_mesh.Dimension());
ParFiniteElementSpace pressure_space(velocity_space.GetParMesh(), &pressure_collection);
SteadyNavierStokesProblem problem(velocity_space, pressure_space, viscosity);

GeometricMultigridOptions multigrid;
multigrid.pre_sweeps = 3;
multigrid.post_sweeps = 3;
multigrid.coarse_solver = GeometricCoarseSolver::Mumps;
multigrid.num_levels = 2;
IterativeOptions iterative;
iterative.hierarchy = &hierarchy;
iterative.velocity_inverse_factory = MakeGeometricVelocityInverse(multigrid);
SteadyNavierStokesOptions options;
options.linear.backend = iterative;
```

Register valid boundary data, force, Brinkman coefficients and pressure gauge
before solving. Equal-order methods use a same-degree pressure collection;
nonlinear PSPG/GLS still require `ExperimentalResidualOptions` opt-in. The
experimental reaction Schur remains optional and still requires MUMPS.
Setting only `iterative.hierarchy` does not select GMG: the inverse factory
must also be set. Unsupported provider types are rejected by this factory.

For adjoints, copy the iterative linear settings into `AdjointOptions::linear`
and pass the actual forward residual options to `SteadyNavierStokesAdjointSolver`.
The context's transpose flag selects the explicitly oriented GMG action.
Repeated objectives reuse the validated setup. No forward solve is introduced
into the supplied-state adjoint path, and no checkpoint format changes occur.
Stage G imported meshes do not carry a GMG hierarchy; its existing workflow
continues to use P1-AMG. A caller must retain or reconstruct a supported coarse
recipe before choosing this inverse.

## Specified levels with parallel MUMPS

`GeometricCoarseSolver::Mumps` uses **all ranks of the problem communicator**
for coarse factorization and solves. It is the default and does not gather to
one rank. `MumpsRoot` is a separate explicit opt-in; do not select it when a
distributed coarse solve is required.

Set `GeometricMultigridOptions::num_levels` to the number of active levels,
**including both finest and coarse**. Zero (the default) uses the entire supplied
hierarchy. A positive count retains that many finest levels without changing
the finest mesh, FE spaces, nonlinear residual, pressure inverse or partitioning.
For a five-level hierarchy, `num_levels=3` uses original levels 2, 3, 4 and runs
parallel MUMPS on original level 2. It does not assemble discarded coarse
Galerkin matrices. The hierarchy still owns all its geometric meshes/spaces.

`num_levels=1` means one mesh level, **not one MPI process**: the velocity proxy
on the finest level is solved by parallel MUMPS inside the existing block
preconditioner. It is not a direct solve of the whole coupled system. Negative,
rank-inconsistent or greater-than-available counts are rejected. The finest
problem size remains controlled by the hierarchy's refinement argument.

The inverse's level/operator/transfer indices are relative to the retained
coarse-to-fine sequence. Provider indices remain relative to the complete
hierarchy. Statistics report retained `level_dofs` and `coarse_solver_ranks`.
Forward and supplied-state adjoint factories use the same level-count option;
the explicit transpose, constraints and Galerkin conventions are unchanged.

The benchmark exposes this as `--gmg-levels`. For example, four MPI ranks,
four available levels, three active levels and a four-rank MUMPS coarse solve:

```bash
"$PETSC_DIR/bin/mpiexec" -launcher fork -rmk user -n 4 \
  build/miniapps/stokes/steady_navier_stokes_gmg_benchmark \
  --backend gmg --coarse distributed --refinements 3 --gmg-levels 3
```

The JSON reports `requested_gmg_levels`, actual `gmg_levels`, and actual coarse
solver communicator size. Pass this option directly to the benchmark executable;
P1-AMG records zero geometric levels and zero GMG coarse ranks because this
option does not alter its own inverse. CTest checks the specified-level cases
without an interpreted benchmark runner.
See `build/parallel-mumps-levels-20261109/` for this follow-up's evidence.

## Operators, corrections and transpose

Levels are numbered coarse to fine. Raw primal interpolation is `P`; dual
residual restriction is its algebraic transpose, not coefficient restriction.
At each level the current boundary attributes define homogeneous Cartesian
component constraints. The cycle uses `P_c = E_f P E_c`, where each `E` zeros
essential coefficients. Every constrained level operator has identity rows
and columns on essential DOFs. Inverse applications preserve that identity
block exactly, including nonzero algebraic essential RHS entries.

The finest operator is a copy of the supplied symmetric positive velocity
proxy, **not** the nonsymmetric Newton block. Coarser matrices are Galerkin
products of the finer constrained operator and raw interpolation, followed by
the level's essential elimination. On homogeneous corrections their action is
`P_c^T A_f P_c`. Thus integrated fine-grid viscosity and Brinkman contributions
survive coarsening; no coarse-point resampling can erase a thin penalty region.
The tests compare both the full action and the difference in energy when a
single coarse-cell Brinkman region is removed. Targets affect the residual,
not the SPD proxy, and are retained by the unchanged forward assembly.

The exposed initial choices are `GeometricCycle::V` and
`GeometricSmoother::DampedJacobi`, with independent positive pre/post counts.
For each level the row-sum bound `b = max_i sum_j |A_ij| / A_ii` is reduced
globally. The Jacobi weight is `damping/(b*A_ii)`, with `0 < damping < 1`.
Nonfinite or nonpositive diagonal entries are errors, not clipped pivots.
This is a conservative stable polynomial smoother for the SPD proxy, not a
promise of optimal high-order smoothing or Reynolds-independent convergence.

The default coarse solve is SPD MUMPS on the full problem communicator.
`GeometricCoarseSolver::FixedJacobi`
instead performs exactly `coarse_sweeps` zero-start weighted Jacobi iterations
(default 100), allowing the inverse itself to build without MUMPS. It is a
fixed linear symmetric polynomial, not a tolerance-dependent inner Krylov solve.
It is a correctness/fallback option, not a scalable coarse-solver recommendation.

With unequal pre/post counts the forward cycle is generally nonsymmetric.
`MultTranspose` swaps those counts recursively. The symmetric level operators,
Jacobi updates and coarse inverse give the actual transposed cycle; tests use
unequal counts to avoid accidentally testing only a self-transpose special case.
The adjoint factory honors `context.transpose`. This does not assert that the
entire upper block preconditioner is self-transpose: Stage E/F still assemble
the separately oriented nonsymmetric block system.

## Validation and reproduction

From the workspace, source `set_up_local.sh`, set one OpenMP/OpenBLAS thread,
and use the configured MPI launcher:

```bash
cmake -S mfem -B build
cmake --build build --target steady_navier_stokes_gmg_tests --parallel 12
export HYDRA_LAUNCHER=fork HYDRA_RMK=user OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$PETSC_DIR/bin/mpiexec" -launcher fork -rmk user -n 4 \
  build/miniapps/stokes/steady_navier_stokes_gmg_tests --suite operators --simplex
"$PETSC_DIR/bin/mpiexec" -launcher fork -rmk user -n 4 \
  build/miniapps/stokes/steady_navier_stokes_gmg_tests --suite solves --method 3 --projected
"$PETSC_DIR/bin/mpiexec" -launcher fork -rmk user -n 4 \
  build/miniapps/stokes/steady_navier_stokes_gmg_tests --suite scaling --tensor
ctest --test-dir build -L '^navier-stokes-gmg$' --output-on-failure -j 16
```

The validation target requires MUMPS for independent references. It compares
GMG-FGMRES, existing P1-AMG-FGMRES and MUMPS on identical forward/adjoint
systems, including all four methods, curved simplex/tensor geometry, component
constraints, open traction, augmented/projected gauges, moving Brinkman targets,
both Schur choices and both coarse strategies. Operator tests include orders
2–4; solved comparisons add degree-three TH/GLS to degree-two coverage.
Diagnostics cover invalid options, provider/finest-space mismatch, stale
revisions, mesh/geometry mutations, invalid levels and malformed input vectors.

The scaling suite uses three geometric refinements of the same coarse recipe
on 1/2/4 ranks. It prints iteration counts, actual residuals, setup/apply/solve
time, coarse setup/apply time, operator/transfer CSR storage and maximum process
high-water RSS. GMG timing counters accumulate over forward and repeated
adjoint setups/applications, and coarse times are subsets, not additive costs.
`csr_bytes` sums the maximum local cycle matrix/transfer storage across ranks;
it excludes mesh/FE metadata, raw hierarchy transfers, vectors and MUMPS factors.
RSS includes those allocations plus the reference/other solver work and is a
process lifetime high-water mark, not isolated GMG memory. P1-AMG rows have zero
GMG-only counters; their forward/adjoint setup and solve timings remain reported.

No mesh-independent or multi-node scalability claim follows from these small
correctness/refinement runs. All ranks participate in the coarse solve, whose
cost is reported explicitly. Further smoothers, coarse rank reduction and
larger refinement studies remain performance work, not silently enabled features.
Measured results and artifacts are recorded in `build/stage-h-20261109/`.

### Measured checkpoint: November 9, 2026

The focused suite passes **200/200 in 73.27 seconds** at `ctest -j 16`.
There are 30 operator, 123 single-mesh solve, six three-depth refinement, and
41 invalid-input cases. The solved checks contain 141 independent MUMPS
forward/adjoint references, each compared against P1-AMG and GMG.
The full Stokes regression passes **1179/1179 in 601.23 seconds**, including
all 979 pre-existing cases, with no failures/skips or runner errors.

Across the 18 GMG refinement configurations, total Newton-correction linear
iterations range from **24–35**, versus **23–32** for P1-AMG; adjoint iterations
range from **29–40**, versus **28–38**. Separate Stokes-initialization counts
are reported, not included in those forward totals. On four ranks, triangular
forward totals are 32/35/35 and quadrilateral totals 24/26/26 at depths 1/2/3.
Coarse applications consume about **35–81%** of GMG apply time in those four-rank
samples. This demonstrates bounded behavior for the small tested family, not
an advantage over the existing inverse or a general convergence/scaling result.
Larger isolated and multi-node studies, and coarse-rank reduction, are the
recommended next performance scope. The MUMPS-free build configuration has not
been separately exercised; the fixed-Jacobi path is tested in the current build.
