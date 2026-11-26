# Stage C: steady Navier–Stokes with Newton/Picard and MUMPS

Stage C adds a genuine nonlinear forward solve to the Stage A problem API and
Stage B Stokes/Brinkman assembly. `SteadyNavierStokesSolver` enables **advective
convection by default**. The new `steady_navier_stokes` executable demonstrates
this solve; `steady_navier_stokes_setup` remains setup-only, and
`StokesBrinkmanSolver` remains explicitly convection-off.

This document records the Stage C checkpoint. The subsequent
[Stage D extension](NAVIER_STOKES_STAGE_D.md) provides explicitly experimental
PSPG/GLS support; without its opt-in, the restrictions described here remain.
[Stage E](NAVIER_STOKES_STAGE_E.md) separately supersedes the historical
MUMPS-only and unsupported-projection restrictions in this checkpoint.

## Scope and equations

The density-normalized equations use density one, kinematic viscosity `nu`,
density-normalized pressure, and prescribed, state-independent coefficients:

```text
-div(2*nu*epsilon(u)) + (u dot grad)u + beta*(u - u_target) + grad(p) = f,
div(u) = 0.
```

The Brinkman coefficient has units of inverse time in dimensional equations.
Conversions such as `beta = nu/permeability` are the caller's responsibility.
There is no hidden density multiplier, transient term, conservative-convection
boundary flux, turbulence model, or change of stabilization.

- With convection on, Taylor–Hood and equal-order CIP are supported, including
  variable viscosity, nonnegative Brinkman reaction, moving targets, component
  velocity constraints, arbitrary total traction, pressure-derived outlets,
  and admissible nonzero mean-pressure gauges from Stage B.
- Nonlinear PSPG/GLS are rejected until Stage D supplies their full residual,
  Jacobian, and scaling extensions. Pressure-only Stokes stabilization is not
  silently substituted for nonlinear residual stabilization.
- `options.convection = false` recovers Stage B exactly. All four Stokes
  discretizations work with beta zero; positive beta retains the Stage B
  Taylor–Hood/CIP restriction.
- MUMPS is the only Stage C backend. Iterative choices and nullspace projection
  remain rejected, not silently converted to direct solves. The legacy Stokes
  preconditioners are unchanged; new nonlinear iterative support is Stage E.
- The inherited conforming, CPU, full-assembly, H1, uniform-order, tri/quad/tet/hex
  and velocity `Ordering::byVDIM` restrictions remain in force.

## Residual and exact derivative

MFEM's existing `VectorConvectionNLFIntegrator` supplies

```text
n(u; v) = integral(((u dot grad)u) dot v),
n'(u)[du; v] = integral((((u dot grad)du) + ((du dot grad)u)) dot v).
```

`ParNonlinearForm` performs true-DOF residual and gradient assembly. The new code
does not duplicate the viscous, pressure, stabilization, Brinkman, boundary-load,
lifting, or gauge implementations. `stokes_assembly.hpp` exposes const shared
blocks and the essential velocity true DOFs through the Stage B adapter.

Let `K0*z - b0` be the existing retained-essential-row Stokes/Brinkman residual.
Let `u_hat` equal the velocity part of `z` on free DOFs and the prescribed lifting
`g` on essential DOFs. With `E` denoting injection into free velocity equations,
the full residual and exact Jacobian are

```text
R(z) = K0*z - b0 + E*n(u_hat),
J(z) = K0 + E*n'(u_hat)*E^T.
```

The essential equations are still `u_essential - g`. This defines the residual
even off the boundary constraint manifold and explains why the added convection
matrix has both essential rows and columns zeroed, with **no extra identity
diagonal**. The inherited identity rows occur exactly once. The velocity used to
evaluate convection always includes the nonzero boundary lifting. Pressure and
gauge blocks, ownership, signs, and source loads are inherited unchanged.

No convection term is assembled when explicitly disabled. This is a stronger
test than evaluating at a zero velocity, whose Newton derivative can mask an
incorrect implementation of the Stokes limit.

Automatic convection quadrature is

```text
max(existing Stokes quadrature, 3*k + 2*dim*(geometry_order - 1)).
```

The `3*k` bound covers transverse tensor-product degrees as well as simplex
cubic products. The geometry allowance is a practical overintegration policy,
not exact integration of arbitrary curved rational functions. The Stokes blocks
keep their existing quadrature. `problem.SetIntegrationOrder(q)` overrides both
with the requested positive order; the caller is responsible for adequate
explicit quadrature. `GetConvectionIntegrationOrder()` reports the resolved
convection order.

Signed element Jacobians are checked at the convection quadrature points in
addition to the baseline geometry checks. A regression mesh passes the lower
Stokes quadrature/geometry-node samples but is rejected at an added convection
point. These sampled checks are not a proof of global geometric validity.

## Nonlinear iteration and stopping

Newton solves `J(z)*step = -R(z)` with both convection derivative terms. Picard
uses MFEM's `ConvectiveVectorConvectionNLFIntegrator` for the Oseen term
`((u_hat dot grad)step, v)`, but always evaluates and stops on the **full nonlinear
residual**. Picard is a separately selected approximate linearization, not an
exact Newton Jacobian.

Both methods backtrack from `initial_damping` to `minimum_damping`, multiplying
by `line_search_reduction`, and accept when

```text
||R(z + alpha*step)|| <= (1 - armijo_constant*alpha)*||R(z)||,
```

or when the candidate already meets the nonlinear stopping tolerance. Trial
states retain the exact prescribed velocity. Nonfinite trials are rejected.
Picard directions are not guaranteed to be descent directions; a failed line
search is reported instead of switching methods or accepting a worse iterate.

The norm is the unweighted distributed Euclidean norm of the actual constrained
equation vector, including essential and gauge equations. This is a coordinate
residual, not a physical L2 norm or a mesh-independent block scaling. Convergence
requires

```text
||R_final|| <= max(absolute_tolerance, relative_tolerance*||R_initial||).
```

`R_initial` is measured after the selected initialization and essential-velocity
imposition. A small step, a converged linear solve, or an exhausted iteration
budget does not imply nonlinear convergence. MUMPS corrections also undergo an
independent true-linear-residual check against the linear options.

`InitialGuess::Stokes` is the default and uses the actual Stage B solve.
`InitialGuess::Provided` uses the caller's finite fields, replacing only essential
velocity DOFs. It does not silently normalize pressure. An omitted initial gauge
multiplier defaults to zero for a provided augmented state; a supplied multiplier
is rejected for a Stokes initialization or an unaugmented problem.

## API, ownership, and diagnostics

```cpp
#include "stokes.hpp"

SteadyNavierStokesOptions options;
options.linear.backend = MumpsOptions{};
options.nonlinear_method = NonlinearMethod::Newton;
options.initial_guess = InitialGuess::Stokes;
SteadyNavierStokesSolver solver(problem, options);
SolveReport report = solver.Solve(velocity, pressure);
MFEM_VERIFY(report.converged, "Nonlinear flow did not converge");
```

The constructor without options explicitly selects Newton/MUMPS. The general
options struct retains its Stage A iterative default, so an explicit options
object must select `MumpsOptions{}` for Stage C.

Output fields are caller-owned and must use the exact problem spaces. The solver
borrows the problem, spaces, and registered coefficients. Setters and
`NotifyCoefficientsChanged()` cause reassembly at the next solve/residual/Jacobian
request. Do not mutate coefficients, fields, or the problem during a collective
operation. Reconstruct after mesh or space changes.

`SolveReport` provides actual initial/final residuals, accepted residual and
damping histories, momentum/continuity/gauge residuals, quadrature-based physical
divergence L2 norm, flux, pressure mean, gauge error, and the replicated multiplier.
`nonlinear_iterations` counts accepted updates; `linear_solves` counts attempted
nonlinear correction solves, excluding the optional Stokes initialization.
MUMPS reports zero Krylov iterations, including in `linear_iterations_per_step`.

Termination distinguishes convergence, iteration limit, true-linear-residual
failure, line-search failure, and a nonfinite initial residual. Failed solves
return the last **accepted** iterate, not a rejected trial, with matching residual
diagnostics. Structural/configuration errors and fatal MUMPS errors still use
MFEM's error mechanism. No claim is made to recover from a fatal factorization.

MUMPS uses nonsymmetric factorization. Symbolic analysis is reused within a solve
only when every rank's assembled CSR graph and column map agree with the preceding
iteration; fixed problem spaces preserve the partition. Numeric factorization
is refreshed each iteration. Reports count symbolic factorizations. A new solve
starts a fresh direct-solver context.

`assembly_seconds`, `setup_seconds`, `solve_seconds`, and `total_seconds` are
MPI-max wall timings. Assembly covers problem preparation and nonlinear residual/
Jacobian assembly; setup and solve include direct work for initialization and
corrections. Total time also includes validation, residual checks, diagnostics,
and field copies. Separate phase maxima need not sum to the total.

## Linearization is not an adjoint solve

`AssembleResidual(FlowState, residual)` and `AssembleJacobian(FlowState)` work on
arbitrary finite compatible states. `AssembleJacobianBlock(state, row, column)`
returns an owned matrix for each velocity/pressure/gauge block, or null for an
identically zero block. These methods always expose the **exact Newton derivative**,
even when the chosen nonlinear iteration is Picard.

Include `stokes.hpp` when using `Linearize(state)`. It returns
an owning `FlowLinearization` with copied state/residual and the exact constrained
Jacobian and transpose. It remains usable after caller field changes or another
solver assembly, but rejects changed problem revisions or spaces. It does not
certify that the supplied state is converged and does not solve an adjoint or
assemble parameter sensitivities. Stage F still owns those acceptance gates.

## Build and tests

From the workspace root:

```bash
source ./set_up_local.sh
export HYDRA_LAUNCHER=fork HYDRA_RMK=user
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cmake --build build --target steady_navier_stokes steady_navier_stokes_nonlinear_tests -j 12
"$PETSC_DIR/bin/mpiexec" -launcher fork -rmk user -n 4 \
  build/miniapps/stokes/steady_navier_stokes
ctest --test-dir build -L '^navier-stokes-nonlinear$' --output-on-failure -j 8
```

Generated-mesh tests cover exact derivatives/Taylor remainders and transposes on
curved tri/quad/tet/hex meshes; reconstruction from all Jacobian/gauge blocks;
Newton/Picard manufactured solutions with nonzero velocity lifting, viscosity
gradients, traction, Brinkman targets, and nonzero means; exact convection-off
recovery for all four Stokes policies; smooth 2D convergence; initialization,
globalization and truthful failure reports; revision invalidation and unsupported
configuration rejection, including added-quadrature geometry checks. They run
on one, two, and four ranks.

These are small-problem verification tests, not evidence of high-Reynolds-number,
arbitrary penalty-contrast, or iterative mesh-independent robustness. Final
validation results and artifacts are recorded in the implementation-plan
checkpoint and `build/stage-c-20261108/`.

## Validation checkpoint: November 8, 2026

- All ten affected executables build using `set_up_local.sh` and its MPI/compiler
  installation. New numerical targets remain conditional on MPI/double/MUMPS.
- All **78/78 focused tests pass** on one, two, and four MPI ranks in 7.84 seconds.
- The full **625/625 Stokes-label tests pass** at `ctest -j 4` in 1609.13 seconds:
  424 legacy tests, 60 Stage A contracts, 63 Stage B tests, and 78 Stage C tests.
  All 547 previous test commands and properties remain unchanged.
- The four-rank Q2/Q1 example converges in two Newton corrections, with a true
  residual below `1e-14`. The stronger globalization case converges in six
  accepted updates with three backtracks and verifies the Armijo inequalities.
- Curved-mesh finite-difference/Taylor and transpose checks pass in 2D/3D for
  Taylor–Hood and CIP. Smooth Q2/P2 velocity refinement tests exceed order 2.6;
  pressure rates exceed 1.7 for Taylor–Hood and 2.2 for equal-order CIP.
- C++17 `-Wall -Wextra -Wpedantic` syntax checks pass without Stokes miniapp source
  warnings; existing MFEM headers emit warnings under these additional flags.
  Tracked-diff and all changed/new-file whitespace checks pass.

`build/stage-c-20261108/` retains the Stage B source snapshot, focused and full
logs/JUnit, earlier test-development attempts, test-definition comparison,
source/executable hashes, and `stage-c-only.patch`. No nonlinear PSPG/GLS,
iterative preconditioner, or new adjoint validation is implied by this checkpoint.
Stage D is next.
