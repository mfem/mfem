# Stage F: supplied-state nonlinear discrete adjoint

`SteadyNavierStokesAdjointSolver` differentiates the implemented constrained
residual at a supplied state. It does not call the nonlinear forward solver,
correct the supplied solution, or overwrite its fields. None/Taylor–Hood, CIP,
and the explicitly experimental nonlinear PSPG/GLS policies use their exact
Newton Jacobian, even if `forward_options.nonlinear_method` is Picard.
The convection-off residual is also supported. The legacy Stokes API is unchanged.

## Usage and ownership

```cpp
FlowState state{velocity, pressure, forward_report.gauge_multiplier};
AdjointOptions options;
options.linear.backend = MumpsOptions{};
SteadyNavierStokesAdjointSolver adjoint(problem, forward_options, options);
KineticEnergyObjective objective;
AdjointState multipliers;
AdjointSolveReport report = adjoint.Solve(state, objective, multipliers);
```

The three-argument constructor records the forward residual policy, including
convection and experimental stabilization scales. Its independent `AdjointOptions`
replace only the linear backend/tolerances. The two-argument constructor uses
default forward residual options: convection enabled, no experimental residual.
Both constructor defaults select FGMRES through `AdjointOptions::linear`.

For an expert assembled objective derivative, call `Prepare(state)`, inspect
`GetLayout()`, assemble the derivative in those coordinates, and call
`Solve(state, gradient, multipliers)`. Solve prepares and validates again;
unchanged state/matrices reuse setup. The field-only overload is for ungauged
problems: a gauged problem requires an explicit signed primal multiplier from
`SolveReport`, including zero for a projected primal solution. Missing gauge
data are not guessed. A projected adjoint requires the iterative backend and a
zero supplied multiplier; MUMPS comparisons use the augmented policy.

The problem, spaces, mesh, coefficient objects, hierarchy/factory dependencies,
and supplied field objects must outlive the solver's use of them. The solver
copies true DOFs and owns its Jacobian and algebraic solution. Output
`AdjointState` owns a copy, independent of subsequent solves; its layout becomes
stale after a problem revision. Checked layout/state getters are collective:
cache their references before uneven local loops or quadrature callbacks.

## Coordinates, signs, objectives, and controls

All public vectors use uniquely owned velocity-equation true DOFs, followed by
pressure-equation true DOFs, and the augmented gauge entry on rank zero only:

```
J = partial R / partial z
J^T lambda = partial Q / partial z
dQ/dalpha = partial Q / partial alpha - lambda^T partial R / partial alpha
```

`lambda` is an algebraic multiplier on residual rows, not a primal FE solution.
Retained essential identity entries and gauge entries are neither discarded nor
overwritten. No primal boundary values or primal pressure-mean shift are imposed
on the adjoint. There is no automatic visualization/reconstruction map.

`ObjectiveFunctional::Evaluate` returns a globally evaluated scalar replicated
on every rank. `AssembleGradient` returns ordinary assembled coordinate
derivatives, not an L2/Riesz gradient. Its vector includes essential and gauge
slots. Usual objectives set the gauge derivative to zero. The built-in
`KineticEnergyObjective` computes the integral of one half the velocity norm
squared. `MeanPressureDifferenceObjective(positive, negative)` computes the
difference of domain-attribute pressure averages, not boundary averages.
Both use geometry-aware quadrature consistently for evaluation and gradient.
Nonfinite values, inconsistent MPI scalars, and invalid vector layouts fail
collectively. Custom callback errors must themselves be MPI-coherent.

`ParameterDerivative` supplies the partial residual direction and the optional
globally replicated explicit objective derivative at the stored state.
`ComputeDirectionalSensitivity` also accepts these assembled quantities directly.
The MPI-global contraction counts the gauge once and does not sum the explicit
scalar again. `ComputeParameterGradient` supports a distributed parameter-Jacobian
transpose, including zero local parameter DOFs and output/input aliasing. That
operator is responsible for its communication and unique parameter ownership.

These are **expert derivative interfaces**, not automatic coefficient or shape
differentiation. The callback must differentiate the same quadrature, lifted
residual, essential equations, boundary loads, stabilization and gauge data.
For a drag control include both the reaction and drag-aware tau/GLS-test terms;
for a moving target include `-beta * target_derivative`; for viscosity include
the viscosity-gradient, tau and GLS-test derivatives (and CIP weight if used).
Domain forces enter both Galerkin and stabilized loads. Prescribed traction and
pressure-as-traction have different load derivatives. A Dirichlet or mean-value
control must include its lifting/identity/gauge-row derivatives. Holding any of
these fixed is a different partial derivative, not the advertised total one.

The adjoint state Jacobian includes exact convection, tau and GLS test-operator
derivatives already implemented in Stage D. Material coefficients are prescribed
fields, not solution-dependent constitutive callbacks. No frozen/approximate
adjoint is offered. The expert callback's mathematical correctness cannot be
inferred from its vector shape; validation examples assemble independent analytic
force, drag, target, viscosity and boundary-load directions.

## Supplied-state validity and reuse

Every preparation validates the current problem and exact space identities,
packs a finite state, and recomputes the full nonlinear residual. Validity uses

```
||R(z)|| <= max(forward_absolute_tolerance,
               forward_relative_tolerance * ||R(0)||)
```

`R(0)` uses zero algebraic fields and zero multiplier, with the current primal
lifting and loads still present in the residual. This reference does not depend
on the history or stopping tolerance of the imported forward solve. The report
records this reference norm, threshold, full and equation-block residuals.
The default relative and absolute validity tolerances are `1e-8` and `1e-12`.
`allow_nonconverged_state` permits an exact local transpose solve for diagnostics;
`forward_valid` is false, and sensitivity methods reject that result.

Reuse requires an unchanged problem revision and bitwise-identical global packed
state. Changing a field in place, a setter, or `NotifyCoefficientsChanged()`
invalidates numerical setup. Mesh/space changes require reconstruction. Mutating
borrowed coefficient data without notification is outside the problem contract.
Factory-produced inverses are destroyed before their borrowed state/matrices.
Sensitivity access checks both revision and current supplied fields, and requires
a converged, forward-valid latest solve. It uses the solver-owned multipliers,
not an output vector that callers might subsequently replace.
Calling `Prepare` clears sensitivity readiness until the next successful solve,
even when it reuses the linearization.

## Backends and pressure projection

MUMPS factors the actual supplied-state nonsymmetric Jacobian and invokes the
verified `MultTranspose` wrapper. Repeated objectives reuse these factors.
FGMRES constructs a separate solver for the actual transposed blocks, with the
[convection-aware velocity inverse](NAVIER_STOKES_PRECONDITIONER.md) for Navier–Stokes
and either weighted-mass or experimental reaction-diagonal pressure auxiliaries.
Stokes and custom SPD-proxy factories retain their previous operator choice.
This upper preconditioner for `J^T` is
not the literal transpose of the forward upper approximate inverse. Factory
contexts expose `transpose=true`, oriented blocks, and the preserved proxy.

Projection keeps the augmented public layout. Internally it verifies left and
right constant-pressure null modes separately, applies the transposed projected
maps, and restores the adjoint pressure normalization dictated by the **objective
gauge derivative**, not by the primal mean. The supported null modes happen to
coincide; unsupported left-nullspace shapes are rejected, not assumed away.
The objective must be orthogonal to the primal right pressure null mode.
An incompatible objective returns `rhs_compatible=false`, `converged=false`,
and `LinearFailure`, with a measured compatibility defect. It is not silently
projected into a different objective. Sensitivities then remain unavailable.

Both backends accept convergence only after an actual full augmented
`||J^T lambda - gradient||` check. Reports include compatibility, signed gauge
multiplier, exactness, setup/linearization reuse, iterations, residual restarts,
and global maximum assembly/setup/solve/apply times. Preconditioner application
time is a subset of solve time.

## Validation and limits

The `navier-stokes-adjoint` CTest label covers curved 2D/3D simplex/tensor meshes,
one/two/four ranks, all four methods, closed mean gauges, open arbitrary traction
and pressure outlets, both Schur strategies, projected objective compatibility,
objective/state derivatives, actual transpose identities and retained essential
and gauge entries. Separate analytic partial-residual loads are compared with
centered differences; parameter-perturbed nonlinear problems are independently
resolved for centered sensitivities and second-order Taylor remainders. Lifecycle
and expected-failure cases cover reuse, invalid states, objective coordinates,
and collective diagnostics. Evidence is recorded under
`build/stage-f-20261109/`, without overwriting Stage E artifacts.

The final focused run passes **101/101 tests in 141.61 seconds** on one, two,
and four MPI ranks. This includes 240 end-to-end scalar control cases, each
with three centered-difference step sizes and independent nonlinear re-solves.
The largest successive Taylor remainder ratio is 0.254001. The full Stokes
regression passes **868/868 in 521.49 seconds**, including every pre-existing
case. The two historical diagnostic-capture cases pass here; their earlier
intermittency remains documented. All affected miniapps build without compiler
warnings; the theory PDF builds without overfull boxes or unresolved references.
The artifact README records a post-CTest shell-wrapper error caused by editing
the runner during execution; both saved test reports independently show no
failures. The final runner is syntax-checked and must not be edited while running.

Nonlinear PSPG/GLS and the reaction-diagonal Schur remain experimental. These
small correctness tests are not high-Reynolds-number, mesh-independent iteration,
large-contrast, shape-optimization, geometric-multigrid, or scaling evidence.
EssentialTrace, solution-dependent material/turbulence models, shape derivatives,
transient adjoints and Stage H geometric multigrid remain unsupported.
