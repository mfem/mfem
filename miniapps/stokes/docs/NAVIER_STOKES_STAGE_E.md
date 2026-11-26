# Stage E: iterative linear corrections

The original SPD-proxy design recorded here has been superseded for built-in
Navier–Stokes velocity inverses by [convection-aware velocity preconditioning](NAVIER_STOKES_PRECONDITIONER.md).
The symmetric proxy remains available to custom factories and Stokes solves.

Implemented with bounded validation and the legacy regression caveat recorded
below. Stage F adjoint solves and Stage H geometric multigrid are not implemented
by this change.

## Design recorded before implementation

- Outer iteration: MFEM flexible right-preconditioned GMRES, checked against the
  original constrained system residual after every solve.
- Velocity inverse: the existing fixed P1/Q1-AMG cycle applied to a separately
  assembled symmetric-strain viscosity + reaction mass proxy. GLS additionally
  includes the existing Stokes GLS normal term with the original cell tau. Never
  pass the Newton/Oseen velocity matrix to the SPD cycle. Only Galerkin coarse
  operators are offered by this preset; rebuild operators/cycles per correction.
- Diffusion Schur preset: pointwise inverse-viscosity mass plus the actual J_pp.
  The signed Schur is J_pp - J_pu J_uu^-1 J_up; the Stokes convention has positive
  mass/stabilization sign. Reuse the fixed pressure P1 inverse and upper gauge
  factor with negative scalar Schur -m^T B m.
- Experimental reaction Schur: D_A is the scalar true-DOF diagonal of the actual
  constrained J_uu (including reaction and nonlinear derivatives), in the native
  vector-space ordering. Require every diagonal entry finite and strictly
  positive; reject zero/negative pivots rather than clipping. Form
  J_pp - J_pu D_A^-1 J_up, retaining the actual nonsymmetric couplings. Augment
  with the integral-pressure gauge when present. Use nonsymmetric MUMPS for this
  auxiliary pressure solve, explicitly requiring a MUMPS-enabled build. This is
  not an AMG-only or penalty-independent robustness claim. Rebuild on each state
  or coefficient update; transposed solves use transposed actual blocks.
- Projection: preserve the public augmented coordinate layout for residuals,
  Jacobians and existing FlowState users, but solve the physical two-field
  quotient without a multiplier. Construct the constant pressure coefficient
  vector by FE projection; independently verify right and left null identities
  against the current physical matrix. Reject unsupported left-nullspace shapes
  and incompatible right-hand sides. Project residuals and preconditioner outputs,
  then restore the FE integral mean. The returned multiplier is zero.
- Factories borrow a per-correction context (spaces/communicator through problem,
  actual blocks, constrained SPD proxy, auxiliary pressure operator, layout,
  essential DOFs, state, gauge policy, transpose flag and optional hierarchy).
  Context data live through the linear solve only; factories return owned Solver
  objects. They must not retain references past destruction of that solver.
  No implicit SetOperator-based update/reuse contract is provided.

The measured parameter range and acceptance results are recorded below.

## Selecting the backend

```cpp
SteadyNavierStokesOptions options;
IterativeOptions iterative;
iterative.max_iterations = 1000;
iterative.restart_dimension = 100;
iterative.p1_smoothing_steps = 2;
iterative.schur = SchurApproximation::WeightedMass;
options.linear.backend = iterative;
options.linear.relative_tolerance = 1e-10;
options.linear.absolute_tolerance = 1e-12;
SteadyNavierStokesSolver solver(problem, options);
auto report = solver.Solve(velocity, pressure);
```

Explicit options default to FGMRES; the no-options solver constructor retains
the previous Newton/MUMPS behavior. MUMPS remains selectable explicitly with
`options.linear.backend = MumpsOptions{}`. The separate public
`StokesBrinkmanSolver` remains convection-off and MUMPS-only.

Use `SchurApproximation::ExperimentalReactionDiagonal` to select the algebraic
reaction auxiliary. It uses MUMPS **only on the pressure/gauge auxiliary**, not on
the full Newton system. The mass preset uses no direct factorization, including
for Stokes initialization. Neither choice silently switches outer backends.
Nonlinear PSPG/GLS still require `options.experimental_residual` explicitly.
Unsupported Schur enums, nonpositive iteration/smoothing counts and rank-varying
options are rejected collectively.

For a pressure-free physical problem with a prescribed mean:

```cpp
problem.SetPressureGauge(PressureGauge::MeanValue(3.0));
options.gauge_enforcement = GaugeEnforcement::NullspaceProjection;
SteadyNavierStokesSolver projected(problem, options);
auto report = projected.Solve(velocity, pressure);
```

Projection requires the iterative backend and a physical pressure nullspace.
The inherited flux, rigid-mode, connectivity and pressure-reference validation
still applies. A provided projected initial multiplier must be zero or omitted.
Pressure is normalized with the FE integral, never the arithmetic coefficient
mean. The public `TrueDofLayout`, `FlowState`, assembled residual/Jacobian and
`FlowLinearization` keep their augmented coordinate convention; only the internal
iterative solve eliminates the multiplier. The returned projected multiplier is
exactly zero. This avoids changing Stage A--D state and linearization contracts.

## Operators, signs, and constraints

For Taylor--Hood without residual stabilization, `D=B`, `G=-B^T`, and `C=0` in
this implementation, so `S=B A^-1 B^T` is positive. CIP adds its positive pressure
jump matrix. PSPG/GLS add a positive pressure-gradient block **and modify other
blocks**. With nonlinear residual stabilization, `G` and `D` include tau/test
derivatives and need not be signed transposes. The weighted-mass preset ignores
those changes to the eliminated velocity contribution, not to the actual
Newton matrix. It is an approximation to the signed Schur, not a replacement
continuity equation or a proof of stability.

Let `B_H` denote the fixed degree-one inverse for `H=M_(1/nu)+J_pp` and
`s=m^T B_H m > 0`. The mass preset recovers the legacy upper action:

```text
z_g = -r_g / s,
z_p = B_H (r_p - m z_g),
z_u = B_A (r_u - J_up z_p).
```

The rank-zero multiplier ownership is retained; every rank participates in its
broadcast and coupling. This is the legacy triangular approximation, not the
exact inverse of the complete constrained pressure block. The reaction preset
instead factors `[S_hat, m; m^T, 0]` nonsymmetrically, then performs the same
velocity back substitution. Essential velocity identity diagonals are retained
once; their eliminated columns in `D` ensure they cannot contaminate the Schur.

`AssembleVelocityProxy()` exposes the constrained, separately assembled proxy
for inspection/custom inverses. It includes reaction even for heterogeneous
penalties. For GLS the proxy uses the old, positive Stokes tau and viscous normal
term, not the nonsymmetric exact derivative of the new residual. The fine
Newton/Picard operator is unchanged. P1/Q1 transfers eliminate fine essential
rows and coarse essential columns component by component. Both proxy and
pressure cycles use fixed l1-Jacobi pre/post sweeps and one zero-tolerance AMG
coarse iteration. Their coarse matrices are exactly Galerkin projections with
homogeneous component constraints; no rediscretized nonlinear coarse option is
implicitly selected.

For projection, the physical block matrix is checked against separate right
and left candidate vectors `[0; c]`, where `c` is obtained by projecting the
constant function into the actual pressure space. Each null defect must be at
most `1e-11*max(1, ||J_phys||_F)*||[0;c]||_2`. A different left-nullspace shape is
explicitly unsupported rather than assumed equal. Orthogonal projectors act on
the Krylov operator, RHS and preconditioner output. RHS incompatibility larger
than the requested linear residual tolerance returns `LinearFailure`; it is not
silently repaired. After solving the quotient, add `alpha*c` to meet the gauge
row RHS using `alpha=(r_g-m^T delta_p)/(m^T c)`. The original full residual is
checked again, including the gauge equation.

## Factory and hierarchy contracts

`IterativeOptions::velocity_inverse_factory` and `pressure_inverse_factory`
replace the two inverse strategies independently. The velocity factory returns a
local true-velocity-size `Solver`; the pressure factory returns a solver of the
local pressure-plus-owned-multiplier size. All ranks must register the same
factory choices and participate in any collectives they require. Null/mismatched
results are rejected. The framework fixes the upper factorization and FGMRES;
custom factories replace inverses, not the nonlinear residual or exact Jacobian.

The borrowed `LinearSolverContext` supplies the actual oriented blocks, problem
and spaces/communicator, coefficient and boundary access, true-DOF layout,
essential DOFs, current packed state, SPD velocity proxy, assembled pressure
auxiliary, integral moments, constant pressure coefficients, verified null
vectors when projected, and a transpose flag. Setup and returned solver objects
are destroyed before any of those borrowed objects change. Every initialization
or correction constructs new operators and factories; there is no cache keyed
only by matrix address and no hidden generic `SetOperator` update.

The optional caller-owned `VelocityHierarchyProvider` exposes coarse-to-fine
levels, borrowed parallel spaces (hence meshes), true-DOF prolongations from each
level to the next, and `BuildLevelOperator(level, context)`, which returns an
owned current operator. The finest space must be the context's velocity space;
the last prolongation is null. Providers/custom factories own validation of MPI
layouts, homogeneous component BCs and transfer compatibility. `Galerkin` means
projection of the current finer operator; `Rediscretization` means restriction
of current state/material data and recomputation of level stabilization,
including detection/treatment of lost thin penalty regions. The transpose flag
requires the corresponding oriented operator. Spaces/transfers outlive every
returned inverse; level operators must be refreshed after state/material changes.
The built-in AMG preset does not inspect or construct this hierarchy. The
single-level factory test validates the interface, **not a geometric multigrid
implementation**.

## Transposes and diagnostics

The internal `detail::BlockFGMRESSolver` can set up a separate solve for the
algebraic transpose: `A^T`, `D^T`, `G^T`, `C^T`, including transposed gauge blocks.
The reaction auxiliary then equals the transpose of the forward auxiliary.
It does not call `MultTranspose()` on the legacy P1 cycle or pretend the upper
factorization is its own transpose. This is a separately assembled preconditioner
for `J^T`, not an exact transpose of the forward preconditioner. Public adjoint
state/objective validation and sensitivity solves still belong to Stage F.

FGMRES uses an internal target one tenth of the requested combined
absolute/relative residual target to leave room for roundoff. Acceptance uses the original requested tolerance,
`max(atol, rtol*||rhs||_2)`, measured against the original constrained matrix.
The nonlinear solver independently repeats this check before line search.
If the recursive estimate stops early but the true residual misses the target,
the driver restarts on the recomputed residual, retaining the original total
iteration budget. It stops on convergence, exhaustion or lack of residual
reduction. `linear_residual_restarts` counts these reliable residual updates,
including initialization. Tight-tolerance tensor-cell tests exercise this path.
An iteration limit never accepts an inaccurate correction. The last accepted
nonlinear state is preserved on linear failure.

`SolveReport` adds total/per-step correction iteration counts, separate
`initial_linear_iterations`, and per-correction `linear_residuals`.
`preconditioner_seconds` measures upper-inverse applications and is a subset of
`solve_seconds`; setup includes auxiliary construction/factorization. Reported
phase times are maximum-over-ranks. `symbolic_factorizations` still counts only
full-system MUMPS factorizations, not reaction-pressure auxiliary factorizations.

## Validation commands

From the workspace, source `./set_up_local.sh`, build the
`steady_navier_stokes_nonlinear_tests` target, then run:

```bash
export HYDRA_LAUNCHER=fork HYDRA_RMK=user OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
ctest --test-dir build -L '^navier-stokes-iterative$' --output-on-failure -j 4
```

All meshes are generated in the tests; no external fixtures are required.
The suite covers 1/2/4 MPI ranks, curved tri/quad/tet/hex operators, both Schur
choices, Newton/Picard, open traction and closed nonzero means, projected
forward/transposed corrections, exact legacy upper-cycle recovery, pointwise
inverse-viscosity mass, diagonal-Schur action, factory/lifecycle behavior, and
independent rejection of invalid left/right nullspaces and diagonal pivots.

The degree-two 2D sweep uses all four policies, 3/6/12 cells per direction,
viscosities 0.5/2, velocity scales 0.2/1, and penalties 0 or 100/1000 on alternating
element attributes. It compares both presets to identical MUMPS nonlinear
problems and solves random forward/transposed RHSs. The latter use tighter
`rtol=1e-13, atol=1e-14` because residual accuracy alone does not bound coefficient
error in refined equal-order systems. An initial `rtol=1e-10` pilot met all true
residual targets but failed eight coefficient-error checks (32 rank-local
failures); that evidence is preserved, not hidden by relaxing error tolerances.
These are polynomial-exact consistency/algebraic sweeps with a bounded contrast
of ten, not a general mesh/Reynolds/penalty robustness claim.

## Acceptance results

- Focused suite: **74/74 pass**, 241.63 seconds, on 1/2/4 MPI ranks.
- Full Stokes regression: **765/767 pass**, 1882.46 seconds. Every Stage E case
  passes in this run, as do the prior nonlinear/stabilization/adjoint suites.
- Two unchanged legacy failure-capture cases return their expected nonzero exit
  but capture empty diagnostics: `stokes_mms_2d_invalid_pressure_gauge_none` and
  `stokes_cli_invalid_order`. The CLI case passes ten isolated repeats; the gauge
  case fails the first retry's capture and then passes ten dedicated isolated
  repeats. The exact capture cause is not established. This is **not** reported
  as a clean 767/767 full regression, and the legacy helper is not changed.
- All 693 previous test names remain. There are 688 unchanged command/property
  definitions; only five old new-API rejection messages change deliberately
  (iterative support and projection-with-MUMPS validation). Exactly 74 are added.
- All relevant miniapps build without compiler warnings. The updated theory PDF
  builds without overfull-box or undefined-reference warnings.

Evidence, initial failures, final logs, source snapshots, the incremental patch
and hashes are in `build/stage-e-20261109/` in the development workspace. This
document and the generated-mesh tests are part of the checkout; that build
artifact directory is not required to use or test the solver.

## Standalone four-rank sweep measurements

Both complete geometry sweeps pass: 192 distinct MUMPS reference problems,
384 nonlinear preset comparisons and 768 forward/transposed correction solves.
Across these runs:

- Newton takes 2--3 corrections; maximum nonlinear residual is `4.47308e-13`.
- Maximum relative state difference from MUMPS is `1.63835e-10`.
- Maximum random-correction relative residual is `5.78660e-14`; maximum relative
  correction coefficient error is `2.74638e-9`.
- Weighted-mass correction iterations: 37--192 forward, 37--191 transposed.
- Reaction-diagonal correction iterations: 14--141 forward, 15--141 transposed.

The reaction approximation is useful but **not mesh independent**. On simplex
meshes with no drag its median correction count grows from 33 to 60 to 128 as
cells per direction increase from 3 to 6 to 12; the mass preset gives 50, 59.5,
64. With penalties 100/1000, the reaction medians are 21, 31.5, 62, versus
75.5, 94.5, 117 for mass. These medians pool four discretizations, two viscosity
values, two velocity scales, and both orientations. Tensor meshes show the same
qualitative behavior. Across viscosity 0.5 versus 2, mass median counts are 79.5
versus 66.5, and reaction medians are 36 versus 48.5; these pooled observations
are not independent viscosity-robustness estimates.

Median per-correction setup/apply/solve times are 1.62/7.26/14.19 ms for mass and
2.53/9.79/11.89 ms for reaction. Setup includes pressure-auxiliary construction,
factorization and P1 cycle setup, but excludes already-assembled Jacobian blocks
and the velocity proxy. Apply time is a subset of solve time, not additive.
These are single-run validation measurements on this allocation, not a scaling
benchmark or a universal runtime ranking. Full rows and min/median/max summaries
are in `sweep.csv`, `sweep-summary.log`, and `sweep-{simplex,tensor}.log` in the
Stage E artifact directory.
