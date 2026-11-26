# Linear Stokes-Brinkman solver

Include `stokes.hpp` and link the CMake target `stokes_core`. The consolidated
interfaces live in that single header; `stokes.cpp` contains all solver,
adjoint, discretization, and preconditioner implementations. All types use
`mfem::flow`; `mfem::navier_stokes` remains a namespace alias. Replace removed
forwarding/module includes with `stokes.hpp` and rebuild downstream code.
See the [package organization](../README.md) for the current layout.

`StokesBrinkmanSolver(problem, StokesBrinkmanOptions)` assembles and solves a
linear system directly; it never calls the nonlinear driver's `Solve`. It
reuses that driver's convection-off residual/Jacobian assembly. The existing
constructors taking no options or `LinearOptions` retain the Stage B,
MUMPS-only behavior and its stabilization restrictions.

## Coefficients and prescribed DOFs

The equations are `-div(2*mu*epsilon(u)) + beta*(u-target) + grad(p) = f` and
`div(u)=0`. `SetBrinkman(beta)` supplies a whole-domain coefficient and zero
target; `SetBrinkman(beta, target)` supplies a vector coefficient target.
Attribute-specific registrations override that default, including an explicit
zero penalty. Changing the default does not erase attribute overrides.

`SetPrescribedVelocityTrueDofs(dofs, values)` copies a list of rank-local,
owned velocity true-DOF indices and borrows the `VectorCoefficient` values.
Indices include the component in the space's `Ordering::byVDIM` layout; they
are not element-local, shared local-vector, or global indices. Every rank calls
the setter, including ranks with empty lists. The coefficient is interpolated
in the nodal H1 space, then mapped to true DOFs. Positive/Bernstein velocity
bases are rejected for this API. Values must agree on shared selected nodes.

The list replaces the previous list, is sorted/deduplicated, and is merged with
attribute-based velocity boundary conditions. Conflicting/nonfinite values
are rejected. Boundary and interior entries, including component-only entries,
are supported. `ClearPrescribedVelocityTrueDofs()` removes only the list.
Physical lifting includes momentum, continuity, and residual stabilization;
retained prescribed coordinates have identity equations.

Coefficient objects and spaces must outlive the problem and solvers. Call
`NotifyCoefficientsChanged()` after in-place coefficient changes. Setters
invalidate numerical setup automatically. Recreate spaces/solvers after mesh
or space mutation. Collective calls must occur in the same order on all ranks.

## Linear options

```cpp
using namespace mfem::flow;

StokesBrinkmanOptions options;
IterativeOptions iterative;
iterative.schur = SchurApproximation::ExperimentalLSC;
options.linear.backend = iterative;
StokesBrinkmanSolver solver(problem, options);
auto report = solver.Solve(velocity, pressure);
```

Use `MumpsOptions{}` instead for a distributed full-system direct solve.
Unchanged systems reuse their factorization/preconditioner between calls.
Both backends check the original constrained residual.

Taylor-Hood and CIP support coefficient-defined reaction directly. Positive-drag
PSPG/GLS require `options.experimental_residual = ExperimentalResidualOptions{}`;
these use the full linear Brinkman momentum residual and GLS reaction test
operator, not just an added reaction mass matrix. Their tau is independent of
the solution with convection disabled. The default pressure/velocity inverse
choices of the existing steady-flow solver remain unchanged.

## Auxiliary constraints and geometric MG

List-based constraints require an explicit auxiliary-space selector when a
P1/Q1 or multilevel geometric inverse is used:

```cpp
problem.SetAuxiliaryVelocityConstraints(
   [](ParFiniteElementSpace &level_space, Array<int> &owned_dofs)
   {
      SelectConstraintsOnThisSpace(level_space, owned_dofs);
   });
```

This callback receives each actual coarse space, including the same-mesh P1/Q1
auxiliary space. It returns owned true-DOF indices; existing boundary-attribute
constraints are added separately. Values are unnecessary for homogeneous
corrections. A single-level geometric direct inverse needs no coarse selector.

The first implementation requires nested correction spaces: every coarse basis
influencing a prescribed fine DOF must itself be constrained. It checks this
condition rather than guessing nearest nodes or accepting a potentially
rank-deficient transfer. Coarse selections may need more constrained DOFs than
the fine list. Arbitrary non-nested projected coarse spaces are not supported.
The matrices use the actual constrained transfers in the Galerkin product.

```cpp
GeometricMultigridOptions multigrid;
multigrid.num_levels = 3;
multigrid.coarse_solver = GeometricCoarseSolver::Mumps;
iterative.hierarchy = &hierarchy;
iterative.velocity_inverse_factory = MakeGeometricVelocityInverse(multigrid);
options.linear.backend = iterative;
```

The hierarchy's finest space must be the problem's velocity space. Zero active
levels means all supplied levels; a positive count retains that many finest
levels including coarse and finest. All levels keep the problem communicator.
One level means a parallel inverse of the fine velocity proxy, not one rank.
GMG does not coarsen the pressure block or full saddle system.

Coarse choices:

- `Mumps`: SPD factorization on every problem-communicator rank; default.
- `FixedJacobi`: `coarse_sweeps` fixed zero-start Jacobi sweeps.
- `BoomerAMG`: `coarse_amg_cycles` fixed zero-start correction cycles, with
  symmetric Galerkin V-cycles, equal Jacobi pre/post sweeps, Jacobi on the
  algebraic coarsest level, and no CF ordering. This configuration has a
  symmetric inverse action; transpose applications use the same action.
- `User`: `coarse_factory(matrix, space)` returns a `GeometricCoarseInverse`.
  The factory must use the supplied distributed layout/communicator; the
  inverse must implement both `Mult` and `MultTranspose` as fixed linear maps.
- `MumpsRoot`: separate explicit centralized option with existing size limits.

Unequal pre/post counts are supported; the transpose reverses them and applies
the transpose coarse action. Setup/application and coarse rank statistics are
available on `GeometricVelocityInverse`.

## Schur choices

With `K=[F,G;D,C]`, the signed Schur complement is `S=C-D*F^{-1}*G`.

- `WeightedMass`: inverse-viscosity pressure mass plus actual stabilization.
  This is a viscous baseline, not a penalty-robust guarantee.
- `ExperimentalReactionDiagonal`: `C-D*diag(F)^{-1}*G`, using actual couplings
  and positive finite pivots. Its pressure auxiliary uses distributed MUMPS.
- `ExperimentalLSC`: unstabilized Taylor-Hood only; stabilized discretizations
  are rejected. With `W=diag(M_u)^{-1}`, form `L=D*W*D^T` and
  `H=D*W*F*W*D^T`, then apply `L^{-1}*H*L^{-1}`. Brinkman reaction is in `F`.
  The shared backend also supports Navier–Stokes with the current Newton/Picard
  convection in `F`; see [Navier–Stokes preconditioning](NAVIER_STOKES_PRECONDITIONER.md).

LSC uses the consistent velocity mass diagonal on free velocity coordinates,
not the diagonal of the momentum operator. Select `lsc_solver` as
`LSCPressureSolver::Mumps` or `LSCPressureSolver::BoomerAMG`; the latter uses
`lsc_amg_cycles` fixed cycles. Both Laplacian applications share a setup. For a
constant-pressure mode, the Laplacian solve is pinned internally and projected
on both sides; the physical pressure mean is enforced separately. No pressure
pin is introduced into the physical system.

For constant coefficient vector `c`, pressure moments `m`, and `t=c^T*m`, the
augmented inverse action uses `zeta=c^T*r_p/t`, applies LSC to `r_p-m*zeta`,
and adds `c*(r_g-m^T*p)/t` to pressure. This replaces the weighted-mass gauge
formula, which would be inappropriate for a quotient inverse annihilating `c`.
The LSC transpose reverses the full composition, including approximate AMG
actions. These pressure solves are independent of the velocity coarse solver.
All new Schur choices remain experimental; stabilized LSC is deferred.

## Supplied-state adjoint

The public `stokes.hpp` header includes the adjoint interface. Use:

```cpp
AdjointOptions adjoint_options;
adjoint_options.linear = options.linear;
StokesBrinkmanAdjointSolver adjoint(problem, options, adjoint_options);
FlowState state{velocity, pressure, report.gauge_multiplier};
KineticEnergyObjective objective;
AdjointState multipliers;
auto adjoint_report = adjoint.Solve(state, objective, multipliers);
```

Omit the gauge value (`std::nullopt`) for ungauged problems. The exact transpose
includes all constraints, gauges, reaction, and selected residual stabilization.
The supplied forward state is validated; no forward solve is rerun. An upper
block preconditioner for the transposed system is not the transpose of the
entire forward upper preconditioner.

Existing objective covectors and expert parameter derivative interfaces apply:
`dQ/dalpha = Q_alpha - lambda^T R_alpha`. Derivatives must include prescribed
values/lifting, target/force loads, and any material-dependent stabilization.
Arbitrary coefficients are not automatically differentiated. Preconditioners
are not part of the differentiated physical residual.

## Example and validation commands

Build `stokes_brinkman_linear_example` for a standalone coefficient-driven
Taylor-Hood forward/adjoint solve. It supplies boundary DOF lists from a vector
coefficient and an explicit level selector, without a nonlinear solve:

```bash
"$PETSC_DIR/bin/mpiexec" -launcher fork -rmk user -n 4 \
  build/miniapps/stokes/stokes_brinkman_linear_example \
  --penalty 100 --refinements 2 --gmg-levels 2 --coarse mumps --schur lsc
```

Other example choices are `--coarse amg|jacobi|root` and
`--schur diagonal|mass`. The `User` coarse strategy is an API factory rather
than a command-line plugin.

Build `stokes_brinkman_tests` and `stokes_brinkman_linear_tests`. With the local
MPI environment set up, run:

```bash
ctest --test-dir build -L '^stokes-brinkman-linear$' --output-on-failure
ctest --test-dir build -R '^stokes_brinkman_(prescribed_dofs|invalid_dof_)' --output-on-failure
```

The linear test driver checks forward and adjoint direct-reference agreement,
nonzero prescribed values, coefficient updates, setup reuse, inverse transposes,
coarse rank/depth reporting, and re-solved sensitivity Taylor rates. Tests and
small manufactured examples do not prove uniform penalty or mesh robustness.

## Recorded validation

On November 9, 2026, the focused suite passes **196/196 in 26.17 seconds** and
the full Stokes regression passes **1578/1578 in 1240.15 seconds**, with no
failures or skips. The full run retains all 1358 previous cases and includes
the 196 linear cases plus 24 prescribed-DOF cases on 1/2/4 ranks. See
`VALIDATION.md` and `build/stokes-brinkman-20261109.LJBOCH/` for scope, raw logs,
JUnit results, and source/evidence fingerprints. Both theory PDFs are rebuilt.
