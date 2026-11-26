# Parallel Stokes and steady-flow solver package

This package solves Stokes, coefficient-driven Stokes-Brinkman, and steady
Navier-Stokes problems with exact supplied-state adjoints. It supports the
existing Taylor-Hood, CIP, PSPG, and GLS discretizations within their documented
scopes. Numerical code remains in one header/source pair. All models now use
one `FlowProblem` / `FlowSolverOptions` / `FlowSolver` API, one forward
implementation, and one `FlowAdjointSolver`. There is no separate public Stokes
solver or model-specific configuration. See the migration table below.

## Organization

| Location | Responsibility |
| --- | --- |
| `stokes.hpp` | All problem, solver, adjoint, integrator, preconditioner, and GMG interfaces |
| `stokes.cpp` | All corresponding numerical implementations in one translation unit |
| `examples/` | C++ applications and benchmarks; shared manufactured fields in `support/` |
| `tests/` | C++ numerical checks and CMake driver, layout, and diagnostic tests |
| `docs/` | Usage guides, development history, and measured validation |

The numerical library is exactly one header and one source file. There are no
separate `core/`, `discretization/`, or `preconditioners/` implementation trees,
and no legacy forwarding headers. Stokes, Brinkman, nonlinear flow, adjoints,
viscous/reaction/convection/stabilization integrators, block FGMRES, P1/Q1 AMG,
geometric multigrid, and all Schur choices are included in this pair.

## Include and link

```cpp
#include "stokes.hpp"

mfem::flow::FlowSolverOptions options;
```

In the MFEM CMake build, link against `stokes_core`. Its public usage requirements
provide MFEM, C++17, and the package include directory:

```cmake
target_link_libraries(my_flow_application PRIVATE stokes_core)
```

The static library compiles `stokes.cpp` once and is shared
by all examples/tests. `stokes_example_support` separately holds manufactured
solution helpers; it is not part of the solver API. These are build-tree targets,
not a new installed/exported MFEM library package.

Build all package examples/tests with `cmake --build build --target stokes_package`.
Existing individual target names and binaries under `build/miniapps/stokes/`
are unchanged.

Replace all former solver/module header includes with `stokes.hpp`; the old
19 forwarding headers and ten module headers are removed. Replace old lists of
individual numerical `.cpp` files with the `stokes_core` target, or compile only
`stokes.cpp` when integrating outside this CMake build. Rebuild downstream code.
Example-only manufactured-field headers remain in `examples/support/`; they
are not part of the solver library. Historical stage documents describe older
file layouts. This package contains no Python scripts or interpreter dependency.
Run the C++ benchmark executables directly; the current commands are in
[the 2D benchmark guide](docs/BENCHMARK_2D.md) and
[the GMG scaling guide](docs/GMG_SCALING.md). CTest uses CMake wrappers and a
small C++ checker for channel/workflow diagnostics, checkpoint integrity,
numerical comparisons, and visualization output. The `stokes_package_layout`
test also rejects Python sources and caches in this directory.

## GNU make build

The `makefile` provides the same application names as CMake and compiles the
numerical implementation once into `libstokes_core.a`. Manufactured-solution
helpers are built separately as `libstokes_example_support.a`. It reads MFEM's
exported `config.mk`, using its compiler, flags, libraries, and MPI launcher.
MFEM must already be configured and built with MPI, double precision, and
C++17 support. MUMPS-only applications are enabled only with `MFEM_USE_MUMPS=YES`.

For an in-source MFEM make build:

```bash
make -C miniapps/stokes -j 8
make -C miniapps/stokes test-par MFEM_MPI_NP=4
```

The top-level MFEM `make miniapps` also includes this package. For an existing
out-of-source MFEM build, including one produced by CMake, run from the workspace:

```bash
make -C mfem/miniapps/stokes MFEM_BUILD_DIR="$PWD/build" -j 8
make -C mfem/miniapps/stokes MFEM_BUILD_DIR="$PWD/build" test-par MFEM_MPI_NP=4
```

These commands put the package objects, archives, and executables in
`mfem/miniapps/stokes/`. To keep them outside the source tree, use a separate
directory (not the existing CMake-generated `build/miniapps/stokes/`):

```bash
mkdir -p build/stokes-make
make -C build/stokes-make -f "$PWD/mfem/miniapps/stokes/makefile" \
  MFEM_DIR="$PWD/mfem" MFEM_BUILD_DIR="$PWD/build" -j 8
```

For an installed MFEM, point `MFEM_INSTALL_DIR` at its prefix and use an
`MFEM_BUILD_DIR` without a `config/config.mk`; an available build configuration
takes precedence over the installed `share/mfem/config.mk`.
`make lib` builds only the solver archive; individual executable targets are
also available. `make test-par` runs bounded linear, nonlinear/Brinkman, and
manufactured-solution smoke checks. `make test` (or `make check`) builds all
enabled applications first and then runs those checks. The full regression
suite remains available through CMake/CTest. Set the appropriate `MFEM_MPIEXEC`
and `MFEM_MPI_NP` for your MPI installation, and use one OpenMP/BLAS thread per
rank when comparing results. `make clean` removes only generated package build
products and smoke-test logs, not source files or user simulation output.

## Unified forward interface

Use `FlowProblem`, `FlowSolverOptions`, and `FlowSolver` for new forward solves.
They share one boundary-condition API and one `Solve(velocity, pressure)` call:

```cpp
using namespace mfem::flow;

FlowProblem problem(velocity_space, pressure_space, viscosity);
problem.SetBoundaryConditions(boundary);
problem.SetBodyForce(1, force);
problem.SetPressureGauge(PressureGauge::MeanValue(0.0));

FlowSolverOptions options;
FlowSolver solver(problem, options);
const auto report = solver.Solve(velocity, pressure);
```

Here the mesh, finite-element spaces, boundary data, coefficients, and output
grid functions are caller-supplied. The default discretization is unstabilized
Taylor-Hood: velocity degree `k >= 2`, pressure degree `k-1`. Equal-order spaces
require an explicit `CIPOptions`, `PSPGOptions`, or `GLSOptions` selection via
`problem.SetStabilization(...)`. A pressure mean is appropriate when the
boundary conditions leave a constant-pressure nullspace, not for every outlet.

| Physics | Configuration before constructing the solver |
| --- | --- |
| Pure Stokes | Defaults: no convection and no Brinkman penalty |
| Stokes-Brinkman | `problem.SetBrinkman(attribute, penalty)` |
| Steady Navier-Stokes | `options.convection = true` |
| Steady Navier-Stokes with drag | Enable convection and set a Brinkman penalty |

`FlowSolverOptions` is the single configuration struct for the shared forward
interface and defaults to **convection off**. `options.linear` controls the backend and linear
tolerances for every model; the default is iterative FGMRES. Set
`options.linear.backend = MumpsOptions{}` for a MUMPS-enabled build.
For an iterative solve, configure an `IterativeOptions` value and assign it to
`options.linear.backend`. Its `method` selects GMRES or FGMRES; `preconditioner`
selects any of the seven diagonal/upper/lower AMG or Galerkin/assembled-P1
variants. FGMRES with the upper P1-AMG variant remains the shared default.
Built-in Navier–Stokes velocity inverses include the current convection block;
see [convection-aware preconditioning](docs/NAVIER_STOKES_PRECONDITIONER.md).
An opt-in [pressure convection–diffusion approximation](docs/PCD_PRECONDITIONER.md)
is available alongside weighted mass and LSC for unstabilized Taylor–Hood flow.
GMRES requires fixed linear inverse actions, including user-supplied factories.
`GaugeEnforcement::NullspaceProjection` requires the iterative backend and an
active pressure-mean gauge; use the default enforcement for gauge-free outlets.
Newton/Picard, initial-guess, line-search, and outer convergence settings apply
when convection is enabled. All option fields are validated and must be
consistent across MPI ranks, including unused nonlinear settings.
The solver copies options at construction; construct a new solver to change them.

One `FlowSolver::Impl` owns the assembly and solve lifecycle. Its linear path
solves the constrained system once; its convection path performs Newton/Picard
updates using the same residual, Jacobian, boundary lifting, and gauge machinery.
The element integration kernels remain shared; models differ by active terms,
not by a chain of specialized solver wrappers.
Existing stabilization restrictions remain: residual-based PSPG/GLS with drag
or convection require the documented experimental residual opt-in.
For coefficient-valued viscosity, residual stabilization also requires its
physical gradient via `problem.SetViscosityGradient(...)`; supply a zero vector
coefficient for constant viscosity.

Both paths return the same `FlowSolveReport` directly, with convergence,
termination reason, iteration counts, residuals, pressure mean, boundary flux,
divergence norm, gauge diagnostics, and timings. There is no nested `details`
variant or second shared report definition. `linear_iterations` includes the
nonlinear path's initial Stokes iterations; `initial_linear_iterations` is a
breakdown of that total, not an additional count. Relative residuals retain
their meanings: linear solves normalize by the constrained RHS norm (or one
for a zero RHS); nonlinear solves normalize by the initial nonlinear residual
norm (or one if zero). Absolute residuals are unnormalized true residuals.
Check `report.converged` before using the output fields.
`linear_residuals` consistently records absolute linear-system residuals.
`setup_seconds` records backend setup, while `preconditioner_seconds` records
preconditioner application time included in `solve_seconds`. Linear solves
reuse backend setup until reassembly; `assembly_seconds` retains the most recent
assembly duration. `pressure_nullspace_detected` reports the assembly decision.

Use `problem.SetPressureGauge(PressureGauge::Auto())` to preserve automatic
constant-pressure nullspace detection. It imposes a mean only when needed;
`Auto(value)` requests a nonzero mean in that case. `MeanValue(value)` requires
a nullspace, while `None()` requires its absence. For automatic gauges, assemble
before constructing a standalone `TrueDofLayout` or querying `HasPressureGauge()`.

The problem and its borrowed spaces and coefficients must outlive the solver;
the mesh and FE collections must outlive their spaces. Problem setters
invalidate setup automatically. Call `problem.NotifyCoefficientsChanged()`
after changing a coefficient in place. Mesh/space mutation requires recreating
the problem and solver. Call collective operations in the same order on all
ranks. `Assemble()`, `GetLayout()`, and `GetEssentialVelocityDofs()` are available
on the common solver. Residual, Jacobian, linearization, and velocity-proxy
assembly are also available through `FlowSolver` and reuse its existing
assembly rather than constructing another solver. Linear-only
`GetSystemMatrix()`, `GetRHS()`, and `GetSystemBlock()` require `Assemble()` or
`Solve()` first and reject convection-enabled problems. A supplied initial gauge multiplier is supported only
with convection and `InitialGuess::Provided`, following the existing gauge rules.

`FlowProblem` is the actual problem class, not an alias. `FlowAdjointSolver`
uses the same problem and forward options for Stokes, Brinkman, and nonlinear
flow; there is no separate Brinkman adjoint forwarding class:

```cpp
AdjointOptions adjoint_options;
adjoint_options.linear = options.linear;
FlowAdjointSolver adjoint(problem, options, adjoint_options);
FlowState state{velocity, pressure, report.gauge_multiplier};
KineticEnergyObjective objective;
AdjointState multipliers;
const auto adjoint_report = adjoint.Solve(state, objective, multipliers);
```

Always pass the same forward options to the adjoint when enabling convection.
Without explicit forward options, the adjoint assumes the default Stokes model.

The complete [flow example](examples/flow.cpp) uses this same setup for each model:

```sh
build/miniapps/stokes/flow_example
build/miniapps/stokes/flow_example --drag 10
build/miniapps/stokes/flow_example --convection
build/miniapps/stokes/flow_example --convection --drag 10
```

## API migration

The redundant compatibility definitions are removed, not retained as aliases.
Update applications and rebuild downstream code:

| Removed name or access | Canonical replacement |
| --- | --- |
| `StokesPreconditionerType`, `StokesPreconditionerName` | `PreconditionerType`, `PreconditionerName` |
| `StokesStabilizationMethod`, `StokesStabilizationMethodName` | `Stabilization` carrying `NoStabilization`, `CIPOptions`, `PSPGOptions`, or `GLSOptions`; `StabilizationName` |
| `StokesPressureGauge`, `StokesPressureGaugeName` | `PressureGauge` factories and `PressureGaugeName`; inspect `PressureGauge::Mode` with `GetMode()` |
| `StokesSolver` | `FlowProblem` plus `FlowSolver` |
| `StokesSolverOptions` | Physical settings on `FlowProblem`; solve settings in `FlowSolverOptions::linear` |
| `StokesSolveInfo` | `FlowSolveReport` (`linear_iterations`, optional `gauge_multiplier`) |
| `StokesAdjointSolver`, `StokesAdjointOptions` | `FlowAdjointSolver`, `AdjointOptions` |
| `StokesForwardState`, `StokesAdjointSolveInfo` | `FlowState`, `AdjointSolveReport` |
| `GetAdjointVector/Velocity/Pressure()` | `AdjointState::GetMultipliers/GetVelocity/GetPressure()` |
| `AddVelocityBC`, `SetVolForce` | `BoundaryConditions::SetVelocity`, `FlowProblem::SetBodyForce` |
| `SteadyNavierStokesProblem` | `FlowProblem` |
| `SteadyNavierStokesOptions`, `StokesBrinkmanOptions` | `FlowSolverOptions` |
| `SteadyNavierStokesSolver`, `StokesBrinkmanSolver` | `FlowSolver` |
| `SolveReport`, `report.details` | Direct fields of `FlowSolveReport` |
| `SteadyNavierStokesAdjointSolver`, `StokesBrinkmanAdjointSolver` | `FlowAdjointSolver` |
| `mfem::stokes`, `mfem::navier_stokes` | `mfem::flow` |

Preconditioning, stabilization, and pressure reference remain independent choices:

```cpp
IterativeOptions iterative;
iterative.preconditioner = PreconditionerType::BlockUpperTriangularP1AMG;
FlowSolverOptions options;
options.linear.backend = iterative;
problem.SetStabilization(CIPOptions{0.02});
problem.SetPressureGauge(PressureGauge::MeanValue(3.0));
```

There is no second stabilization selector: the variant carries both the method
and its active parameters, and assembly reads that configuration directly.
`NoStabilization{}` selects the existing Taylor-Hood formulation. A pressure
configuration carries its mode and requested value together; its nested modes
are `Auto`, `MeanValue`, and `None`. Construct it through `Auto(value)`,
`MeanValue(value)`, or `None()`, not by passing a bare mode. The old `MeanZero`
mode name is gone because prescribed means can be nonzero. CLI method,
preconditioner, and `--pressure-gauge auto|mean-zero|none` spellings are unchanged;
`PressureGaugeName` returns `mean-value` for a nonzero prescribed mean and
`mean-zero` for zero. Examples and test fixtures also use these canonical types.

When migrating a former Navier-Stokes options declaration, explicitly set
`options.convection = true`; the canonical default is pure Stokes. Former
no-options specialized drivers selected MUMPS; select
`options.linear.backend = MumpsOptions{}` explicitly to retain that choice.
The canonical no-options driver selects FGMRES. Adjoint callers can pass the
same `FlowSolverOptions` directly; the old `ResidualOptions()` conversion is gone.

The old low-level API is removed, not retained as aliases. To retain its former
PSPG/automatic-gauge defaults, explicitly set `PSPGOptions{}` and
`PressureGauge::Auto()` on the problem. Scalar viscosity is copied; borrowed
viscosity uses `SetViscosityGradient(...)` for PSPG/GLS. Output grid functions
are caller-owned and passed to `Solve(velocity, pressure)`. A forward state is
always explicit when solving an adjoint, including a state produced in memory.

The former `LinearFlowSolver`, `NonlinearFlowSolver`, `LinearBrinkmanSystem`, and
separate low-level adjoint implementation are gone. The private `FlowAssembly`
contains assembly kernels only: no independent solve interface, options type,
solution storage, or backend. Regression-only fixtures in `tests/linear_case.hpp`
translate historical test notation into the public API; they contain no alternate
numerical implementation and are not linked into the library or examples.
Integrators and preconditioners remain because they perform distinct numerical
operations. Historical stage guides should be read using this migration table.

## Common namespace

All Stokes, Brinkman, steady Navier-Stokes, adjoint, and geometric multigrid
definitions now live in **`mfem::flow`**. Shared implementation helpers live in
`mfem::flow::detail`; anonymous namespaces keep other helpers file-local.
Examples and numerical tests use the common namespace.

There are no legacy namespace aliases. Use `mfem::flow` in qualified names,
`using` statements, and namespace declarations. Removed class and namespace
names are source-breaking changes; rebuild rather than linking objects compiled
against the former definitions. The public include and link target are unchanged.

## Guides

- [Mesh-driven channel flow with a Brinkman inclusion](docs/CHANNEL_FLOW.md)
- [Linear Stokes-Brinkman and geometric MG](docs/STOKES_BRINKMAN_LINEAR.md)
- [Full Stokes usage and existing APIs](docs/README.md)
- [Adjoint and sensitivity interfaces](docs/ADJOINT.md)
- [Steady-flow application and checkpoints](docs/NAVIER_STOKES_STAGE_G.md)
- [Geometric MG and coarse solvers](docs/NAVIER_STOKES_STAGE_H.md)
- [Measured validation](docs/VALIDATION.md)
- [Discretization and preconditioner theory](../../../theory/README.md)

LSC remains experimental and limited to linear Taylor-Hood. Complete reaction
PSPG/GLS remain experimental. Explicit fine DOF lists require auxiliary selectors
and nested correction spaces for P1/Q1 and multilevel geometric inverses.
