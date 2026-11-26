# Steady Navier-Stokes: Stage A contracts

Stage A implements problem registration, validation, options, reports, true-DOF
state layout, and reusable algebraic linearization containers. It does **not**
assemble or solve Navier-Stokes/Brinkman equations. There is deliberately no
placeholder `SteadyNavierStokesSolver::Solve` or nonlinear adjoint solve.
`steady_navier_stokes_setup` is a setup-only example, not a flow simulation.
The existing `mfem::stokes` solver and adjoint remain unchanged.

This document records the Stage A contract checkpoint. Subsequent forward
implementations are documented in [Stage B](NAVIER_STOKES_STAGE_B.md) and
[Stage C](NAVIER_STOKES_STAGE_C.md); Stage A's setup example remains setup-only.

## Location and build

The new interfaces live in `mfem::navier_stokes`:

- `steady_navier_stokes_solver.hpp/.cpp`: problem, registration, options, reports,
  primal state, layout, and the legacy field-copy adapter.
- `steady_navier_stokes_adjoint.hpp/.cpp`: adjoint options/reports, algebraic
  multiplier storage, objective interface, and linearization container.
- `steady_navier_stokes.cpp`: generated-mesh setup example.
- `steady_navier_stokes_tests.cpp`: generated-mesh MPI contract tests.

These are miniapp-side interfaces, not installed MFEM library headers or an
addition to `mfem.hpp`. The local CMake file defines explicit shared source/header
lists and reuses the existing Stokes solver/integrator/preconditioner sources.
The MPI/double guards, C++17 requirement, Hypre dependency, and optional MUMPS
configuration are inherited from the existing build. MUMPS option validation
rejects a build without MUMPS; numerical backend execution remains later work.

From the workspace containing `mfem`, `build`, and `set_up_local.sh`:

```bash
set +u
source ./set_up_local.sh
export HYDRA_LAUNCHER=fork HYDRA_RMK=user
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cmake -S mfem -B build
cmake --build build --target steady_navier_stokes_setup steady_navier_stokes_tests --parallel 16
"$PETSC_DIR/bin/mpiexec" -launcher fork -rmk user -n 4 build/miniapps/stokes/steady_navier_stokes_setup
ctest --test-dir build -L '^navier-stokes-contracts$' --output-on-failure -j 16
```

Every new test generates its mesh in memory. No new test depends on the workspace
`meshes/` collection. Existing external-mesh tests remain registered separately.
CTest `PROCESSORS` reservations account for each one- or four-rank MPI launch.

## Problem and registration

```cpp
using namespace mfem::navier_stokes;
SteadyNavierStokesProblem problem(velocity_space, pressure_space, 1.0);
BoundaryConditions boundary(PressureCondition::OutletTraction);
boundary.SetVelocity(1, inlet_velocity);
boundary.SetVelocity(2, Direction::X, 0.0);
boundary.SetTraction(2, Direction::Y, shear);
boundary.SetPressure(3, outlet_pressure);
problem.SetBoundaryConditions(boundary);
problem.SetBodyForce(1, body_force);
problem.SetBodyForce(2, Direction::X, force_amplitude);
problem.SetBrinkman(2, penalty, target_velocity);
problem.SetStabilization(PSPGOptions{});
problem.SetPressureGauge(PressureGauge::None());
problem.Validate();
```

The example assumes compatible equal-order spaces, boundary/domain attributes,
and caller-owned coefficients. It registers data only; it does not solve.

Boundary IDs are boundary attributes; force/Brinkman IDs are domain attributes.
Directions are Cartesian components. Whole-vector registration replaces all
components on one attribute; a later component setter replaces only that entry.
Other attributes are preserved. Repeated force calls replace rather than add.
Brinkman replacement replaces both penalty and target: the two-argument overload
resets the target to zero. Unregistered forces and penalties are zero.

Constants are copied. Coefficient objects, spaces, and meshes are borrowed and
must outlive every use. Coefficients must provide deterministic, prescribed
spatial data and valid MFEM evaluations on their volume or boundary context;
opaque state-dependent coefficients are not a nonlinear derivative model.
The owned piecewise force, penalty, and target coefficient adapters expose the
same registered data to future Galerkin and stabilization assembly.

`BoundaryConditions` is a local staging object. Its setters retain records;
validation occurs when it is attached. `SetBoundaryConditions` copies the records,
not coefficient objects. Changing the original builder has no effect until it is
reattached. Problem construction, setters, validation, and state/algebraic
operations are collective on the velocity-space communicator; call them in the
same order on every rank. Destroy borrowed-data users before their dependencies.

Before preparing layouts/operators, `Validate` checks replicated options and
registration structure/constants, global attribute existence, supported space
metadata, and sampled coefficient values. Coefficient identities/addresses are
not compared across ranks; the caller is responsible for one coherent global
coefficient definition. MPI checks reduce failures before issuing MFEM errors.

## Stage A decisions and limits

- Velocity is uniform H1 degree `k >= 2`, byVDIM, on conforming full-dimensional
  tri/quad/tet/hex meshes. Pressure is H1 degree `k-1` for `NoStabilization` and
  `k` for CIP/PSPG/GLS. Equal order does not automatically enable stabilization.
- PSPG/GLS retain the existing Hessian-basis restrictions. A scalar viscosity
  constructor owns a constant and zero gradient. The coefficient constructor
  requires `SetViscosityGradient` for PSPG/GLS, including an explicit zero-gradient
  coefficient when appropriate. Viscosity and penalty values are checked at the
  selected volume quadrature; boundary data are checked at boundary quadrature.
- Free velocity components default to zero natural traction, matching Stokes.
  Explicit velocity and traction on the same attribute/component are rejected;
  complementary components are allowed. Pressure plus total traction on one
  attribute is rejected. Pressure on an entirely prescribed-velocity attribute is
  redundant and rejected. `OutletTraction` means future `sigma*n = -p_ext*n`
  assembly; it never means an essential trace of the FE pressure.
- `EssentialTrace` is explicitly rejected: no compatible velocity closure has
  been derived. `NullspaceProjection` is explicitly rejected until its left/right
  nullspace implementation is validated. Only augmentation is an accepted
  gauge-enforcement contract. The requested gauge is explicitly `None()` or
  `MeanValue(value)`; there is no automatic gauge selection in this new API.
- Metadata/coefficient validation is **not** assembled-problem validation.
  Shared-DOF boundary conflicts across different attributes, rigid modes,
  boundary-flux compatibility, physical gauge admissibility, pressure null modes,
  signed-geometry checks, full stabilized lifting, and PDE consistency remain
  Stage B/shared-assembly gates. A successful setup does not prove solvability.

The assembly contract for later stages is density-normalized, with
`sigma = 2*nu*epsilon(u) - p*I`, advective convection, prescribed scalar
`beta >= 0`, and prescribed target velocity. The momentum residual contains
`beta*(u-u_target) - f` and the natural load has residual sign `-integral(t*v)`.
The unscaled continuity row is `+(q,div(u))`, giving unstabilized Stokes blocks
`[A,-D^T; D,0]`. A requested mean uses the FE integral vector `m`, with
`m^T*p = mean*volume`, not an arithmetic coefficient average. These are mathematical
contracts, not operators assembled by Stage A.

The exact Stokes recovery specification remains unchanged:

```text
tau_0,K = c_tau*h_K^2 / (nu_bar_K*(k+1)^4), c_tau = 1 by default
CIP weight = gamma*h_F^3*(0.5/nu_bar_plus + 0.5/nu_bar_minus), gamma = 0.01
h_K = mesh.GetElementSize(element,1); h_F = min(h_plus,h_minus)
automatic quadrature = 2*k + 2 + 2*d*(geometry_degree-1)
```

The existing geometry/order helper computes the automatic quadrature. Use
`SetIntegrationOrder(-1)` for automatic or a positive explicit order.
The formula and defaults above are a recovery contract, not newly implemented
nonlinear stabilization. No convection/drag tau formula, nonsymmetric velocity
inverse, or reaction-aware Schur preset is declared robust by Stage A. Resolve
those numerical decision gates in Stages D/E, and retain all seven legacy Stokes
presets. Same-mesh P1/Q1 degree coarsening is not geometric multigrid.

## Revisions, layouts, and primal state

Every problem setter and `NotifyCoefficientsChanged()` increments a revision.
After modifying borrowed coefficients in place, call `NotifyCoefficientsChanged`
before preparing new algebraic data. Such changes cannot be discovered from a
coefficient pointer alone. Mesh/space sequence changes invalidate the problem;
reconstruct it after any mesh/space modification, including manual metadata edits
that MFEM sequence counters cannot detect.

`TrueDofLayout(problem)` validates the problem and snapshots its revision and
local offsets/global sizes. Its order is velocity true DOFs, pressure true DOFs,
and (only when requested) a gauge DOF owned by rank zero. Non-root ranks retain
an explicit empty gauge block. Essential identity-row coordinates are retained;
this is not a free-DOF reduction. Layout metadata describes the snapshot;
`CheckCurrent` rejects using it after a problem change.

`FlowState` borrows primal `ParGridFunction`s in the identical space objects,
plus an optional replicated primal gauge scalar required exactly when active.
`Pack` copies true DOFs and places the gauge only in its owner slot. It does not
project fields, fix essential values, normalize pressure, or establish convergence.
Its output vector must not alias the primal fields.

`CopyStokesSolution` copies existing solver-owned fields into caller-owned fields
on identical spaces. It does not change legacy ownership, rerun Stokes, or transfer
between unrelated FE spaces. Carry any primal gauge scalar separately from the
existing solve report when constructing a supplied `FlowState`.

## Linearization, objectives, and adjoint state

`FlowLinearization(problem, state, jacobian, residual)` is an expert container:
it takes ownership of an already constructed `Operator`, copies the primal state
and residual, and binds them to the current problem revision. It does **not**
compute or certify a Navier-Stokes Jacobian. Its producer must supply the exact
constrained residual/Jacobian in the published coordinates, including any gauge,
and provide a valid distributed `MultTranspose`. Any data borrowed inside that
operator must remain alive and consistent with the snapshot. Forward/transpose
applications check sizes, finiteness, and revision validity. Changing the original
primal fields does not alter the saved state vector. Prepare a new linearization
for a different state or problem revision.

`ObjectiveFunctional` defines a globally reduced scalar evaluation and assembly
of a coordinate derivative in the supplied layout. Shared-DOF contributions must
be assembled exactly once. An L2/Riesz gradient requires an explicit mass-map
conversion; the interface does not apply one automatically.

`AdjointState::SetMultipliers` copies a packed algebraic vector. Its velocity and
pressure getters are read-only equation-block views, not FE fields. Essential
entries are retained, adjoint pressure is not shifted, and the optional gauge
getter broadcasts the single owned value. Storage is noncopyable to avoid dangling
block views; a subsequent assignment invalidates previously borrowed views.
No field-reconstruction/Riesz mapping is supplied. Adjoint options are independent
of forward options. Reports default to `NotRun`, not convergence.

The intended discrete sensitivity convention remains
`J^T*lambda = dQ/dz` and `dQ/dalpha = Q_alpha - lambda^T*R_alpha`.
Numerical supplied-state convergence checks and actual forward/adjoint solves
belong to later stages. Stage A tests use a synthetic nonsymmetric operator and a
coordinate-quadratic objective solely to verify the algebraic contracts.

## Validation checkpoint: November 8, 2026

- Both new targets and the four existing Stokes targets build successfully.
  Additional C++17 syntax checks with `-Wall -Wextra -Wpedantic` report no warnings
  in the new sources; existing MFEM headers emit warnings under these extra flags.
- All 60 new contract tests pass on one/four MPI ranks, both independently and
  within each full-suite attempt. All meshes are generated in memory.
- All 424 legacy test commands/properties, existing C++ sources/test helpers, and
  four Stokes executable hashes remain identical to the Stage 0 baseline.
- The first full run passes 483/484 tests. The unchanged
  `stokes_cli_invalid_order` exits as expected but loses its diagnostic text;
  ten standalone repeats then pass, as does that test in the second full run.
- The second full run passes 475/484. Several legacy processes enter prolonged
  apparent I/O wait, followed by SIGBUS, incomplete MMS output, or missing
  diagnostic failures. The precise runtime cause is not established. All nine
  failed cases subsequently pass with one four-rank test running at a time.

The follow-up conservative run passes **484/484 tests** at `ctest -j 4` in
1594.30 seconds, closing the Stage A regression gate. This run uses the unchanged
Stage A executables, before rebuilding legacy targets with Stage B assembly
extensions. No legacy numerical code or error-capture helper was changed to mask
the earlier failures. Stage B is documented separately in
`NAVIER_STOKES_STAGE_B.md`.

Logs, JUnit reports, the incremental Stage A patch, source hashes, and reproduction
commands are retained in the workspace's `build/stage-a-20261108/` directory:
`contracts.log`, `regression.log`, `diagnostic-retry.log`,
`regression-confirm.log`, `legacy-retry.log`, `regression-conservative.log`, and
`run.sh` preserve all attempts.
