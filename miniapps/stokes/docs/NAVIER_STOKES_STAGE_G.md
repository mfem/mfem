# Stage G: runnable flow and supplied-state adjoint workflows

`steady_navier_stokes_workflow` is a separate CPU/full-assembly MPI example.
It leaves the existing `steady_navier_stokes`, `stokes_brinkman`, legacy Stokes
miniapps, and their defaults unchanged. It demonstrates Taylor–Hood and equal-order
CIP/PSPG/GLS, open and fully prescribed-velocity boundaries, stationary/moving
Brinkman regions, and adjoints of a solution imported by a **separate process**.
Its files are an example-specific restart format, not a general persistence API.

## Build and run

From the workspace containing `mfem`, `build`, and `set_up_local.sh`:

```bash
set +u
source ./set_up_local.sh
export HYDRA_LAUNCHER=fork HYDRA_RMK=user
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cmake -S mfem -B build
cmake --build build --target steady_navier_stokes_workflow --parallel 12
mpi=("$PETSC_DIR/bin/mpiexec" -launcher fork -rmk user -n 4)
app=build/miniapps/stokes/steady_navier_stokes_workflow
mkdir -p build/workflow-demo
"${mpi[@]}" "$app" --help
```

The local fork/user launcher is appropriate for the supplied environment. Use
the allocation's launcher for multi-node execution, without mixing MPI builds.
The target requires MPI and double precision. Built-in FGMRES works without
MUMPS; the direct backend and experimental reaction Schur require MUMPS.

Each export directory must be **new**, with an existing parent. The application
refuses to overwrite it, even when it contains only a partial previous export.

### Stable pair and open pressure outlet

```bash
"${mpi[@]}" "$app" --method taylor-hood --problem channel \
  --checkpoint build/workflow-demo/th-channel
```

This uses Q2/Q1, Newton, and block-preconditioned FGMRES. The inlet is a
parabolic velocity profile; walls are no-slip; the outlet has pressure-derived
traction. `--simplex` selects Pk/P(k-1); `--order 3` or `--order 4` raises degree;
`--curved` applies a quadratic coordinate map. Equal-order policies use Pk/Pk
or Qk/Qk. The example supports 2D, degrees 2–4, and at least three cells per
direction; it is not a generic mesh-file frontend.

### CIP, arbitrary traction, stationary Brinkman region

```bash
"${mpi[@]}" "$app" --method cip --problem channel --outlet traction \
  --normal-traction 0.03 --tangential-traction 0.015 --drag 15 \
  --simplex --curved --checkpoint build/workflow-demo/cip-traction
```

The traction vector is `(-pressure + normal_traction)*n + tangential_traction*t`,
where `t=(-n_y,n_x)`. This is a prescribed vector load, not a pressure trace.
It uses the same boundary load definition in the forward and adjoint residuals.
With no target speed, the penalized region's prescribed target velocity is zero.

### Experimental PSPG and a closed mean gauge

```bash
"${mpi[@]}" "$app" --method pspg --experimental --problem closed \
  --drag 20 --pressure 0.3 --curved \
  --checkpoint build/workflow-demo/pspg-closed
```

All velocity components are zero on the boundary, with a pressure mean of 0.3.
The gauge is augmented by default. Nonlinear PSPG/GLS require the explicit
`--experimental` acknowledgement; there is no silent fallback to a Stokes-only
stabilization formula or claim of production robustness.

### Experimental GLS, a moving target, and pressure projection

```bash
"${mpi[@]}" "$app" --method gls --experimental --problem closed \
  --drag 20 --target-speed 0.05 --gauge projected --curved \
  --checkpoint build/workflow-demo/gls-moving
```

The target is `(target_speed, 0.25*target_speed)` on the Brinkman attribute.
Projection requires a closed problem and FGMRES. Its public state still includes
the gauge slot, with zero primal multiplier. In this example a nonzero target
speed requires positive drag to avoid an ineffective input option.

## Adjoint-only reload: no forward solve

```bash
"${mpi[@]}" "$app" --mode adjoint \
  --checkpoint build/workflow-demo/gls-moving --experimental --gauge projected
"${mpi[@]}" "$app" --mode adjoint \
  --checkpoint build/workflow-demo/gls-moving --experimental --backend mumps
"${mpi[@]}" "$app" --mode adjoint \
  --checkpoint build/workflow-demo/cip-traction --schur reaction --objective pressure
```

Use the **same MPI rank count and shard-to-rank mapping** as the export. Physical
options come from the manifest; specifying `--method`, `--force`, mesh parameters,
or other physical overrides in adjoint-only mode is an error, even if they equal
the saved value. Backend, Schur strategy, gauge enforcement, objective selection,
and the FGMRES iteration limit remain independently selectable. Experimental
PSPG/GLS require acknowledgement again when importing the saved recipe.

The reload branch constructs the problem and invokes `Prepare`/`Solve` on
`SteadyNavierStokesAdjointSolver`, never the forward nonlinear solver. It prints
`forward_solves=0`. It recomputes the full nonlinear residual, verifies both
saved objective values, solves the requested adjoints, and checks that the
supplied algebraic state is unchanged (`state_unchanged=1`). Invalid or
nonconverged imported states fail; there is no repair solve.

The default `--objective both` solves integral kinetic energy, then a difference
of **domain-attribute pressure averages**, reusing one setup. The latter is the
average over attribute 1 minus that over attribute 2, not an inlet/outlet pressure
drop. `--objective kinetic` and `--objective pressure` select one objective.
The printed multiplier norm is an MPI-global algebraic norm. Multipliers are
not exported as velocity or pressure grid functions: they are dual variables
on residual rows, with retained essential and gauge entries.

The signs and coordinates are

```
J^T lambda = partial Q / partial z
dQ/dalpha = partial Q / partial alpha - lambda^T partial R / partial alpha
```

Gradients are assembled true-DOF coordinate derivatives, not unconverted Riesz
gradients. The pressure mean in the forward problem is not imposed on adjoint
pressure-equation multipliers. A projected objective incompatible with the primal
right nullspace is rejected instead of silently altered. See
`NAVIER_STOKES_STAGE_F.md` for validity tolerances, transpose maps, exact tau/GLS
derivatives, output semantics, and scalar/distributed sensitivity callbacks.

## Geometry, attributes, and coefficient recipe

The unwarped rectangle is `[0,2] x [0,1]`. Boundary attributes are 1 bottom,
2 right, 3 top, 4 left. Domain attribute 2 is the set of cells whose **unwarped
centers** lie inside `(0.65,1.35) x (0.25,0.75)`; all other cells are attribute 1.
This cellwise patch is not a cut-cell obstacle or a body-fitted interface.
The curved map is `(x,y) -> (x+0.03*x*y,y)` and preserves the attribute labels.

Viscosity is `viscosity*(1+0.2*x)` and its exact gradient is supplied. The body
force is `force*(sin(pi*y),-sin(pi*x))`, except that the y component on domain
attribute 2 is deliberately overwritten with zero to demonstrate a targeted
component update. The reaction is `drag*(u-u_target)` only on attribute 2.
In a channel, `--pressure` is the pressure-derived contribution to outlet
traction; in the closed case it is the pressure mean. It never means an
essential pressure trace.

## Setter and ownership contracts

- `SetVelocity(attribute, component, value)` updates only that boundary-component
  key; a vector overload sets all components on that attribute. Repeated calls
  replace their own entries and preserve unrelated attributes/components.
- `SetPressure(attribute, value)` replaces only that boundary's pressure load.
  `SetTraction` has the same component/vector update rules as velocity. These
  registrations do not erase conflicting boundary types. To replace a pressure
  outlet by arbitrary traction, construct compatible new boundary data; do not
  register both on the same attribute.
- `problem.SetBoundaryConditions(boundary)` copies the registration maps. Changing
  the local `BoundaryConditions` object afterward has no effect until it is
  registered again. Borrowed coefficient objects must still remain alive.
- `SetBodyForce` uses **domain** attributes, not boundary attributes. Components
  accumulate across distinct keys; replacing one component does not reset others.
  Missing domain components are zero.
- `SetBrinkman` replaces the reaction/target registration for its domain attribute.
  The overload without a target resets that attribute's target to zero, including
  when replacing an earlier moving target. Other attributes are unchanged.
- Spaces and mesh are caller-owned; coefficient references and callback/hierarchy
  objects are borrowed. Setters change the problem revision. Notify the problem
  with `NotifyCoefficientsChanged()` after in-place coefficient changes. Mesh or
  space changes require reconstruction; revision/state changes invalidate cached
  linearizations and inverses. Checked getters are collective and must be cached
  before uneven rank-local loops.

## Backend choices, diagnostics, and extension hooks

The new nonlinear solver's explicit options and this example default to **FGMRES**.
For compatibility, its no-options C++ constructor still selects Newton/MUMPS.
The legacy Stokes miniapps retain their own GMRES/FGMRES choices and documented
defaults. This workflow does not introduce a nonlinear GMRES backend.

`--backend mumps` is the direct nonsymmetric reference. `--schur mass` selects
the Stage E weighted-mass pressure auxiliary; `--schur reaction` selects the
experimental actual-block reaction-diagonal auxiliary and its MUMPS inverse.
The built-in Navier–Stokes velocity path now uses the full linearized velocity
block and a fixed same-mesh P1/Q1-AMG cycle with systems-AMG options, **not
geometric multigrid** or an SPD treatment of the nonsymmetric Newton matrix.
Stokes and custom SPD-proxy factories retain their previous operator choice;
see [convection-aware velocity preconditioning](NAVIER_STOKES_PRECONDITIONER.md).
The adjoint builds a separately oriented block preconditioner for the actual
transpose. Reaction-Schur options cannot be applied to MUMPS mode.

Forward output includes convergence/termination, nonlinear and linear iterations,
the actual full and block residuals, independent L2 divergence, boundary flux,
pressure mean/gauge error, signed multiplier, and assembly/setup/solve times.
Termination codes follow `TerminationReason`: 0 not run, 1 converged, 2 iteration
limit, 3 linear failure, 4 line-search failure, 5 nonfinite residual. Adjoint
output includes forward validity, actual transpose residual, RHS compatibility,
global multiplier norm, signed gauge multiplier, iterations and setup reuse.
A small stabilized continuity residual does not imply zero pointwise divergence.
A failed forward solve returns nonzero and does not publish a checkpoint.

Public C++ extension points already implemented are `ObjectiveFunctional`,
`ParameterDerivative`, the distributed parameter-Jacobian transpose interface,
and `IterativeOptions` velocity/pressure inverse factories. A factory receives
the oriented actual blocks, SPD proxy, state/layout, constraints, auxiliaries,
and `transpose` flag through `LinearSolverContext`; its returned solver must not
outlive those borrowed data. The hierarchy provider is a contract for later
work, not an implemented geometric hierarchy or an AMG scalability guarantee.

Material coefficients are prescribed fields. Fully coupled turbulence-model
derivatives, RANS stresses, LES/transient trajectories, time integrators, shape
derivatives, and EssentialTrace closure are not implemented by these hooks.
Frozen derivatives must not be described as total discrete sensitivities.
Stage H implements and validates an actual geometric multigrid inverse next;
it must address transfers, level constraints/operators, transpose behavior, and
refinement scaling rather than simply relabel the current P1-AMG cycle.

## Checkpoint contract and testing

A directory contains `manifest` and native MFEM `mesh.RANK`, `velocity.RANK`,
`pressure.RANK` files, using six-digit rank suffixes and 17-digit output precision.
The manifest stores workflow recipe version 1, rank count, all physical recipe
parameters, the signed replicated primal gauge multiplier, both global objective
values, and per-shard byte lengths/FNV-1a checksums. The manifest is published by
rename only after every shard is closed successfully. Missing, truncated or
inconsistent files are rejected collectively before loading their MFEM objects.
Checksums detect accidental corruption; this is not an authenticated file format.

Loaded grid functions own their reconstructed spaces/collections; the parallel
mesh outlives both. The native parallel format preserves sharing and curved
coordinates. This example does not repartition, migrate build/file-format
versions, import arbitrary material code, or guarantee cross-version replay.
The coefficient recipe must retain its meaning within a workflow schema version.
Use the Stage F `FlowState` API for caller-managed imports outside this recipe.

`ctest --test-dir build -L '^navier-stokes-workflow$' --output-on-failure` checks
separate-process round trips, both backends and Schur choices, projected gauges,
two-objective reuse and single-objective selection, immutable input files,
all four policies, both cell families, quadratic curved geometry, and degrees
two/three on 1/2/4 ranks. Failure tests cover invalid options, missing/corrupt
shards, rank/schema/space/gauge mismatches, modified forcing/nonconverged states,
objective mismatches, unsafe overwrites, and failed-solve publication.
These are small correctness examples, not convergence/scaling benchmarks.
Stage G evidence is recorded separately in `build/stage-g-20261109/`.

### Measured validation: November 9, 2026

The focused suite passes **111/111 tests in 62.39 seconds** at `ctest -j 8`:
108 separate-process round trips and three bundled invalid-input suites on
1/2/4 ranks. Degree-two coverage includes all four methods and all four physical
scenarios on curved simplex/tensor meshes; degree-three coverage adds
Taylor–Hood and GLS channel cases. The documented degree-four option is not
part of this integration matrix. The full Stokes regression passes **979/979
in 563.70 seconds** at `ctest -j 16` within a 112-process allocation, including
all 868 pre-existing cases, with no failures/skips or runner errors.
The workflow target builds without warnings; the updated theory PDF has no
overfull boxes or undefined-reference warnings. Stage E/F evidence remains
preserved; no core solver, legacy example or legacy test expectations changed.
