# Mixed null-space and coarse-space deflation

`linalg/deflation.hpp` provides `DeflatedCGSolver`, `DeflatedGMRESSolver`, and
`DeflatedFGMRESSolver`. Exact null-space bases control RHS compatibility and
the solution gauge. Coarse bases are optional acceleration candidates. Either
category can be set, replaced, or cleared without changing the other.

```cpp
mfem::DeflatedCGSolver solver;
solver.SetOperator(A);
solver.SetNullSpace(N);       // N spans ker(A) and ker(A^T).
solver.SetCoarseSpace(Z);     // Z contains acceleration candidates.
solver.SetRelTol(1e-8);
solver.SetMaxIter(100);
solver.Mult(b, x);
if (!solver.GetConverged()) { /* inspect solver.GetSolveInfo() */ }
```

Use CG only when `A` is symmetric and positive definite on the complement of
its null space. Its fine preconditioner must be fixed and positive definite
there. For a nonsymmetric singular operator whose left and right null spaces
differ, use GMRES or FGMRES with paired null bases and an explicit
preconditioner that maps the left complement invertibly to the right one:

```cpp
mfem::DeflatedFGMRESSolver solver(comm);
solver.SetOperator(A);
solver.SetPreconditioner(M);
solver.SetNullSpaces(N_right, N_left);
solver.SetCoarseSpace(Z); // A shared coarse input is allowed.
solver.SetKDim(30);
solver.Mult(b, x);
```

Use `SetCoarseSpaces(Z_right, Z_left)` if distinct trial and test candidates
are needed. All four configurations—neither space, either space alone, and
both together—are supported where the solver's mathematical assumptions
hold. `BALANCED` correction is the default. `PROJECTED` is selected with
`SetCoarseCorrectionType(CoarseCorrectionType::PROJECTED)`.

## Coarse-space representations

Two representations are available; both give the same correction
`Q = Pi_R Z_R (Z_L^T A Z_R)^{-1} Z_L^T Pi_L` in exact arithmetic.

**Dense.** `SetCoarseSpace(Z)` takes any operator mapping replicated
coordinates to owned true DOFs; only `Mult()` is needed. Setup materializes
the columns, projects them off the null spaces, orthonormalizes them, and
factors the replicated matrix `V^T A U`. It checks column rank and the
conditioning of that matrix, supports the SVD solve, and caches `A U`. It
suits tens to hundreds of global modes. A `VectorDeflationBasis` holds an
explicit set of vectors for this path and is copied directly, without one
`Mult()` per column:

```cpp
mfem::VectorDeflationBasis Z(n_local);
Z.Add(v0);  Z.Add(v1);        // copies; Add(std::move(v)) takes storage
solver.SetCoarseSpace(Z);
```

**Operator.** `SetCoarseSpace(Z, S, exact)` takes a general operator whose
coarse coordinates may be distributed and whose `MultTranspose()` is the
global transpose, for example a sparse `HypreParMatrix` prolongation, and a
borrowed solver `S` for `Z^T A Z`. Nothing is materialized. Setup forms
`E = Z_L^T A Z_R` with `RAP` or `ParMult` when `A` and `Z` are
`HypreParMatrix` objects, and as an `RAPOperator` otherwise, then calls
`S.SetOperator(E)`. With `HypreParMatrix` inputs it also caches `A Z_R`, so
the projectors need no extra operator application. Because the null spaces
are exact, `Pi_L A Pi_R = A`, and the candidates need no projection.
`SetCoarseSpaces(Z_R, Z_L, S, exact)` registers distinct trial and test
spaces.

```cpp
mfem::HypreParMatrix &Z = ...;   // e.g. aggregate prolongation
mfem::HypreBoomerAMG coarse;     // or a direct solver with exact = true
solver.SetCoarseSpace(Z, coarse);
```

The operator path suits large sparse subspaces, as in contact problems. Its
setup performs only dimension checks: rank deficiency, or candidates in the
null space, show up as a singular or ill-conditioned `E`, which the coarse
solver must handle. Declare `exact = true` only when `S` applies `E^{-1}`,
e.g. a direct solver. `PROJECTED` correction requires an exact coarse solve,
because its projected operator and solution recovery rely on `P_L` and `P_R`
being projectors; `Mult()` rejects it otherwise. `BALANCED` correction
accepts an inexact `S`; CG then requires `S` to be fixed and symmetric
positive definite, while FGMRES also allows a varying `S`. The SVD and
transpose-caching setup options apply only to the dense path.
`GetCoarseOperator()` returns the coarse operator on either path, and
`GetCoarseSpaceDimension()` sums the local widths of `Z_R` on the operator
path.

For a controlled example, take `A = diag(0.001, 2, 5)`, `b = (1, 0, 0)`,
and `Z = (1, 0, 0)`. With `PROJECTED` correction, the coarse solve returns
`x = (1000, 0, 0)` at a zero iteration budget. With no coarse space, the
same budget returns the zero initial guess. The unit test checks this case.

`Mult()` rejects a RHS with NaN or infinite entries on every rank. The
diagnostics `IsConsistent()` and `ValidateNullSpaces()` report such values as
failures rather than treating them as zero.
The default RHS policy rejects an incompatible component above its tolerance.
Its default relative tolerance is `max(1e-10, 32*epsilon)` to account for
projection roundoff in single precision.
To solve the modified problem `A x = Pi_L b`, set
`DeflationRHSOptions::action` to `IncompatibleRHSAction::PROJECT` and pass the
options to `SetRHSOptions`. `GetSolveInfo()` distinguishes working-system and
original-system residuals and convergence. Both modes return the Euclidean
gauge `N_right^T x = 0` when exact right null modes are registered.

`A`, input basis operators, and the fine preconditioner are borrowed and must
outlive registration. Basis operators map replicated coarse coordinates to
local owned true-DOF rows; `MultTranspose()` is unnecessary for basis setup.
Call `Update()` after changing a registered operator or basis in place. `b`
and `x` must not share storage in `Mult()`; partially overlapping `Vector`
views are rejected too. Individual projector methods accept
input/output aliasing. The combined residual-projector/coarse-correction
method rejects outputs whose storage overlaps, including distinct `Vector`
views of the same data. Solvers and projector objects are non-reentrant.
`DeflationSpaces::Setup(true)` asserts that `A` is symmetric and uses shared
bases; the solution projector relies on that symmetry, so pass `false` for a
nonsymmetric operator.
Changing RHS policy or correction mode clears solve statistics without
rebuilding the fine preconditioner. `Update()` also rebinds that preconditioner
to the original operator after in-place operator changes.

Automatic basis-rank and coarse reciprocal-condition thresholds use
`max(1e-10, 32*k*epsilon)` and `max(1e-12, 32*k*epsilon)` respectively, where
`k` is the relevant number of columns and `epsilon` is machine epsilon for
`real_t`. In `DeflationSetupOptions`, `-1` requests this default; explicit
nonnegative values, including zero, take precedence. These thresholds cannot
prove that supplied null bases are exact or complete. Optional
`ValidateNullSpaces()` applies `A` and `A^T` to the registered null bases.

The default coarse solve uses Cholesky for CG and pivoted LU for GMRES or
FGMRES. With MFEM built using LAPACK, users can opt into an SVD coarse solve:

```cpp
mfem::DeflationSetupOptions setup;
setup.coarse_solve_method = mfem::CoarseSolveMethod::SVD;
solver.SetSetupOptions(setup);
```

SVD mode checks the smallest-to-largest singular-value ratio against
`coarse_rcond_min` and rejects a singular or insufficiently conditioned coarse
matrix. The default direct mode estimates reciprocal condition in the 1-norm;
SVD uses the 2-norm, so borderline cases may be accepted differently. CG
still checks positive definiteness with Cholesky before using SVD. Selecting
SVD in a build without LAPACK fails at setup with a clear error. SVD does not
form a pseudoinverse or change the coarse-space dimension. All coarse factors
and SVD data remain small, replicated host objects.

The MPI constructor uses the supplied borrowed communicator for Krylov dot
products and deflation reductions. All ranks must use matching column counts,
options, and call order. Borrowed operators must themselves obey collective
call behavior. Whether setup or preconditioner binding must be rebuilt is
decided collectively, so a rank with stale state never enters a different
reduction sequence; all ranks then rebuild. Either every rank must attach a
solver controller or none may;
mixed controller registration is rejected before the solve. Controller
callbacks run on every rank; a stop requested on any rank stops the collective
solve.

The implementation uses MFEM device-aware `Vector` operations for basis
orthogonalization, projection, restriction, expansion, coarse correction, and
Krylov fine-vector work. With device-backed vectors and device-capable input
operators and preconditioners, these fine-vector operations are intended to
stay on the configured device. Setup materializes basis columns using the
registered basis operators, so those operators must also support the device
for a device-only setup path. Replicated coarse coordinates, the coarse matrix
and its factorization, GMRES Hessenberg bookkeeping, and scalar reductions
remain on the host. MPI receives host scalars or small host coarse-coordinate
buffers; GPU-aware MPI is not required. Device scalar products can synchronize
to return their host scalar result. Solver controller callbacks may access
the vectors on the host if the controller chooses to do so.

The implemented path uses MFEM's CPU and backend-neutral device memory and
kernel APIs; it contains no CUDA- or HIP-specific kernels and adds no external
dependency. Device-oriented unit cases are provided for CG, GMRES, FGMRES,
setup/update, paired spaces, both correction modes, and MPI reductions. CUDA,
HIP, and MPI+GPU execution are intended targets; builds, runtime correctness,
and transfer behavior on those configurations have not yet been validated.
Setup caches the fine-grid images `A U` for residual projection. GMRES and
FGMRES use their internal residual estimates to decide when to verify a
physical residual; controller callbacks still receive physical residuals and
recovered solutions on every iteration. In left-preconditioned GMRES, the
internal estimate may delay detection of physical convergence until the end
of a restart cycle. A transient dip below tolerance can be missed if the
physical residual rises again before that check. Attach a controller or
request per-iteration output when physical residual verification is needed
after every iteration. For symmetric
operators, the solution projector also uses cached `A U`. For nonsymmetric
operators, `DeflationSetupOptions::cache_transpose_images = true` caches
`A^T V` during setup, so the solution projector needs no operator
application. It calls `A.MultTranspose()` once per coarse column and stores
that many extra fine vectors. It is off by default because `MultTranspose()`
is otherwise optional, and all MPI ranks must use the same setting. The
balanced preconditioner shares one coarse restriction between residual
projection and coarse correction. Projectors skip input scratch copies when
the input and output storage is disjoint. To decide this without moving data,
the output, which is about to be overwritten, is marked valid in the input's
memory space without copying. Only an input valid in no memory space keeps
the conservative copy. Each CG iteration obtains `||r||` and `r^T z` from one
global reduction. The next preconditioner application therefore happens
before the stopping test, and it is wasted only on the iteration that
converges.
