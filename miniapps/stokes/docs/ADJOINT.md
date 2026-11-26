# Discrete flow adjoints

`FlowAdjointSolver` is the sole adjoint interface for Stokes, Stokes–Brinkman,
and steady Navier–Stokes. Include `stokes.hpp`, use `mfem::flow`, and link
`stokes_core`. There are no old namespace aliases or model-specific solvers.
Pass the same `FlowProblem` and forward options used for the primal model.
The short constructor assumes default nonconvective Stokes.

For the constrained discrete residual and objective:

```text
R(z, alpha) = 0
J = partial_R/partial_z
J^T lambda = dQ/dz
dQ/dalpha = partial_Q/partial_alpha - (partial_R/partial_alpha)^T lambda
```

The Jacobian is the exact derivative of the shared forward residual, including
stabilization, convection, drag, boundary lifting/elimination, essential identity
rows, and the pressure gauge. It is not a separately discretized continuous
adjoint. A Picard forward solve still uses the exact Newton Jacobian for adjoints.

## State and coordinate conventions

Every solve takes explicit `FlowState` fields; a preceding forward solve is not
required. Fields must use the identical finite-element-space objects held by the
problem. Supply the finite replicated primal gauge multiplier when a gauge is
active, and `std::nullopt` otherwise. The field-only convenience overload has no
multiplier and therefore applies only to gauge-free problems.

Vectors use uniquely owned velocity true DOFs, pressure true DOFs, and the
optional rank-zero-owned gauge entry. `Prepare(state)` validates the state and
prepares its exact Jacobian. `GetLayout().GetOffsets()` provides block offsets;
retain the layout or copy its offsets while constructing `BlockVector` views.
Automatic pressure gauges are resolved by assembly before validating the layout.

Objective inputs are assembled coordinate covectors, not L2/Riesz gradients.
An L2 gradient normally requires a mass-matrix action. Shared-DOF contributions
must already be assembled. Do not automatically zero essential objective entries
or shift adjoint pressure: essential and gauge-equation multipliers can be nonzero.

## Solving and retrieving results

```cpp
using namespace mfem;
using namespace mfem::flow;

AdjointOptions adjoint_options;
adjoint_options.linear = forward_options.linear;
FlowAdjointSolver adjoint(problem, forward_options, adjoint_options);
FlowState state{velocity, pressure, forward_report.gauge_multiplier};
AdjointState multipliers;
KineticEnergyObjective objective;
const AdjointSolveReport report = adjoint.Solve(state, objective, multipliers);
```

`KineticEnergyObjective` integrates `0.5*|u|^2`; its gradient is the velocity
mass action. `MeanPressureDifferenceObjective` subtracts volume-weighted means
on two volume attributes, not boundary patches. Custom `ObjectiveFunctional`
implementations return a global rank-replicated value and an assembled gradient.
For an already assembled objective covector, use:

```cpp
adjoint.Prepare(state);
const Array<int> offsets(adjoint.GetLayout().GetOffsets());
Vector objective_gradient(offsets.Last());
objective_gradient = 0.0;
BlockVector blocks(objective_gradient, offsets);
blocks.GetBlock(0) = assembled_velocity_objective_derivative;
const auto result = adjoint.Solve(state, objective_gradient, multipliers);
```

The result container owns algebraic multipliers:

```cpp
const Vector &packed = multipliers.GetMultipliers();
const Vector &velocity_equations = multipliers.GetVelocity();
const Vector &pressure_equations = multipliers.GetPressure();
const std::optional<real_t> gauge = multipliers.GetGaugeMultiplier();
```

These are not `ParGridFunction` fields. There is no prolongation, field
interpretation, pressure normalization, or Riesz conversion. Block getters are
views of packed storage. Later solves overwrite that result container; copy a
vector to retain a snapshot. Failed finite iterates remain available, but check
`report.converged` before using them as a solution.

## Validation, ownership, and reuse

The supplied forward residual is tested against
`max(forward_absolute_tolerance, forward_relative_tolerance*||R(0)||)` from
`AdjointOptions`. The reference includes constrained boundary/gauge data.
States are not corrected, transferred, or pressure-shifted. The adjoint never
runs a forward solve or overwrites primal fields. Setting
`allow_nonconverged_state=true` permits finite non-steady states for inspection,
but sensitivities still require a valid steady state.

The problem, spaces, mesh, coefficients, hierarchy, and supplied primal fields
must outlive their users, including sensitivity queries. Packed values are
copied, but source fields are retained for mutation checks. Calls are collective
and configuration must agree across ranks. Changing the problem revision or
source state requires preparation/solve again. In-place coefficient changes
require `problem.NotifyCoefficientsChanged()`; mesh/space mutation requires
reconstruction. Unchanged state/revision reuses the exact Jacobian and backend
setup for later objectives. Calling `Prepare` alone clears sensitivity readiness.

MUMPS factors the exact Jacobian and uses `MultTranspose()`. Iterative options
support GMRES/FGMRES and all seven block preconditioner variants, with actual
transposed upper/lower couplings. These are preconditioners for the transposed
system, not necessarily transposes of the forward approximate inverses. GMRES
requires fixed linear actions. Pressure-nullspace projection requires iterative
solves, an active mean gauge, compatible objective data, and a zero primal gauge
multiplier. No unavailable backend is silently substituted.

Reports contain the true adjoint residual, supplied-forward residual and
tolerance, block residuals, compatibility defect, iteration counts, problem
revision, optional objective value/multiplier, and cache-reuse flags. Wall times
are MPI maxima: `assembly_seconds`, `setup_seconds`, `solve_seconds`, and
`preconditioner_seconds` (applications, included in solve time). Setup is zero
when reused. Finite true residuals are checked even for direct solves; numerical
nonconvergence is returned, while invalid configuration/state uses MFEM errors.

## Caller-supplied parameter derivatives

For one parameter direction, supply `partial_R/partial_alpha` at fixed algebraic
state in the same residual layout:

```cpp
real_t derivative = adjoint.ComputeDirectionalSensitivity(
   explicit_objective_derivative, residual_direction);
```

The explicit objective derivative is a globally evaluated scalar replicated
consistently across ranks; it is not summed a second time. The adjoint contraction
uses a global dot product with unique ownership, including the gauge only once.

For a distributed parameter field:

```cpp
adjoint.ComputeParameterGradient(residual_parameter_jacobian,
                                 explicit_objective_gradient, total_gradient);
```

The supplied `Operator` has residual local height and parameter local width.
Its `MultTranspose()` must implement the distributed derivative action and all
required communication. Parameter gradients use uniquely owned parameter DOFs;
the adjoint API does not sum the resulting vector again. Output may alias the
explicit gradient. Scalar directional and distributed results use the same
minus-sign convention.

All derivatives must correspond to the last validated state/current problem.
Their mathematical correctness cannot be inferred from their vector sizes.
Include physical stabilization, lifting, and gauge derivatives when applicable;
do not differentiate auxiliary preconditioning matrices as part of the residual.
Automatic differentiation and built-in control/material/shape derivatives are
not provided.

## Example and tests

`stokes_adjoint_example` computes the derivative of velocity energy with respect
to a force amplitude for Taylor–Hood, using both stored and supplied states and
checking the analytic scaling. It explicitly assembles the objective covector
and residual direction through the public API. The same usage supports fields
loaded into the existing spaces.

```bash
set +u
source ./set_up_local.sh
export HYDRA_LAUNCHER=fork HYDRA_RMK=user
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cmake --build build --target stokes_pspg_tests stokes_adjoint_example --parallel 8
"$PETSC_DIR/bin/mpiexec" -launcher fork -rmk user -n 4 \
  build/miniapps/stokes/stokes_adjoint_example --solver fgmres
ctest --test-dir build -L '^stokes-adjoint$' --output-on-failure -j 8
```

The suites cover explicit-transpose references, all method/backend/preconditioner
combinations, higher-order/curved/periodic geometry, both gauge states, supplied
states, cache invalidation, zero RHS, failed iterations, derivative transpose
identities, and Taylor checks for force/viscosity/boundary amplitudes. See
`VALIDATION.md` for measured results and the preexisting CLI diagnostic issue.
