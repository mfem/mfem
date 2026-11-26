# Stage B: symmetric-strain Stokes/Brinkman baseline

This is the **convection-off, MUMPS-only** numerical baseline for the steady
Navier–Stokes extension. `StokesBrinkmanSolver` uses the Stage A problem and
boundary registrations; its name deliberately does not imply that nonlinear
Navier–Stokes is implemented. The setup-only `steady_navier_stokes_setup` driver
remains unchanged. No nonlinear or new adjoint solve is claimed here.

The subsequent [Stage C solver](NAVIER_STOKES_STAGE_C.md) adds nonlinear
Taylor–Hood/CIP solves using this baseline; this Stage B adapter remains
convection-off.

## Equations and supported combinations

The momentum and continuity equations are

```text
-div(2*mu*epsilon(u)) + beta*(u - u_target) + grad(p) = f,
div(u) = 0,
epsilon(u) = (grad(u) + grad(u)^T)/2.
```

Viscosity is positive and the Brinkman penalty is nonnegative at the assembly
quadrature points. Domain registrations use element attributes, not boundary
attributes. Repeated registrations replace the specified data rather than
accumulating duplicate integrators. `SetBrinkman(attribute, beta)` resets that
attribute's target to zero. Unregistered domains have zero force/penalty/target.

- Taylor–Hood H1(k)/H1(k-1), k >= 2: Stokes and positive-drag Brinkman.
- Equal-order H1(k)/H1(k) CIP: Stokes and positive-drag Brinkman.
- Equal-order PSPG and GLS: the existing complete symmetric-stress **Stokes**
  residual terms, viscosity gradients, and consistent loads, with beta = 0.
- Positive beta with PSPG/GLS is rejected. Their reaction residual, test operator,
  Jacobian, and scaling extensions belong to Stage D; simply adding a mass
  matrix would not be the specified stabilized Brinkman discretization.
- The new solver rejects iterative options. All seven legacy Stokes
  preconditioners remain available through the unchanged `stokes::StokesSolver`
  API. Reaction-aware preconditioning belongs to Stage E.

This retains the legacy CPU/full-assembly, conforming tri/quad/tet/hex, uniform
order, H1, and velocity `Ordering::byVDIM` restrictions. Residual stabilization
still requires a physical viscosity gradient and supported Hessian-capable
velocity bases. A gradient is not needed for Taylor–Hood or CIP.

## Shared assembly, not a second Stokes implementation

`stokes_assembly.hpp` is a miniapp-internal production bridge into the existing
`StokesSolver` assembly. It is distinct from the test-only hooks in
`stokes_solver_internal.hpp` and is not an installed MFEM API. The bridge borrows
coefficient pointers and configures optional reaction, total traction,
pressure-derived traction, and mean-target data before assembly.

The existing implementation still owns and builds the viscous, divergence,
pressure-gradient, CIP/PSPG/GLS, lifting, pressure-surrogate, and gauge blocks.
It also retains geometry validation, shared-DOF boundary conflict checks,
net-flux compatibility, actual pressure-nullspace detection, distributed
true-DOF assembly, MUMPS, and true-residual convergence checks.

For positive beta, the only new Galerkin volume contributions are

```text
velocity matrix: (beta*u, v)
velocity load:   (beta*u_target, v).
```

The reaction uses the same resolved volume quadrature as the Stokes blocks.
The effective source is `f + beta*u_target`. For identically zero sampled beta,
the reaction integrator is not added and the target is not added to the source:
the assembled Stokes matrix, RHS, lifting, gauge, and transpose actions recover
the existing implementation exactly. All legacy tau, CIP, and pressure-surrogate
scales are unchanged. See `NAVIER_STOKES_STAGE_A.md` for those formulas.

The rigid-mode check uses the old essential-DOF restrictions, augmented for
positive reaction by quadrature samples of `sqrt(beta*abs(weight))*rigid_mode`.
Their common normalization by the absolute-weight quadrature sum of beta makes
the rank check insensitive to a uniform rescaling of the penalty. Absolute
weights keep this rank probe well-defined for high-order simplex rules with
negative weights; the assembled reaction matrix uses the original signed rule.
Thus positive volumetric
reaction can remove rigid modes even without velocity Dirichlet conditions;
beta = 0 retains the old rigid-mode rejection. This is a sampled numerical rank
check, not a general proof of stability for disconnected/degenerate meshes.

## Boundary conditions and pressure means

`SetVelocity` retains componentwise strong constraints and full block lifting.
`SetTraction` supplies the **total stress traction**

```text
t = (2*mu*epsilon(u) - p*I)*n,
boundary momentum load = integral(t dot v).
```

Traction may be a vector coefficient or independent Cartesian components.
Unspecified free components have zero natural traction. Loads are assembled
before elimination, so essential velocity test components do not receive
natural loads, including at shared boundary corners.

`SetPressure(attribute, p_out)` remains an outlet-traction condition, not an
essential pressure trace. Its load is `-integral(p_out*n dot v)`, using MFEM's
oriented boundary normal and the chosen boundary quadrature. It prescribes
normal total traction `-p_out` and zero tangential total traction on free test
components, not separate conditions `p = p_out` and zero viscous normal stress.
Already prescribed velocity components are eliminated. Pressure
and explicit total traction on the same attribute remain incompatible. Fully
velocity-constrained pressure outlets and `EssentialTrace` remain rejected.

For a detected constant-pressure nullspace, `PressureGauge::MeanValue(p_bar)`
uses the existing augmented system

```text
[ F  G  0 ] [ u      ]   [ momentum load  ]
[ B  C  m ] [ p      ] = [ continuity load]
[ 0 m^T 0 ] [ lambda ]   [ p_bar*volume   ],
m_i = integral(phi_i), volume = m^T*1_h.
```

Only MPI rank zero owns the multiplier row. A mean is rejected if natural/open
boundaries already determine absolute pressure. Conversely, `None()` is
rejected when a pressure nullspace remains. There is no silent normalization,
pressure pinning, or projection. The legacy report's `MeanZero` policy enum
identifies this inherited augmentation path; `pressure_mean` reports the actual,
possibly nonzero mean. The problem retains the requested target.

## API and lifetime

Include `stokes.hpp` and use `mfem::flow` (`mfem::navier_stokes` remains an alias):

```cpp
SteadyNavierStokesProblem problem(velocity_space, pressure_space, viscosity);
problem.SetBoundaryConditions(boundary);
problem.SetBodyForce(1, force);
problem.SetBrinkman(2, penalty, target);
problem.SetPressureGauge(PressureGauge::MeanValue(2.0));

StokesBrinkmanSolver solver(problem);
auto report = solver.Solve(velocity, pressure);
MFEM_VERIFY(report.converged, "Stokes/Brinkman did not converge");
```

The default constructor selects MUMPS. The overload taking `LinearOptions`
accepts its tolerances/print level, but requires `backend = MumpsOptions{}`.
The output grid functions must be caller-owned and use the exact problem spaces.
`Solve()` returns `stokes::StokesSolveInfo`, including true residual, pressure
mean, gauge multiplier, flux, and the existing MPI-max assembly/setup/solve
timings. These are shared Stokes-kernel timings, not a complete profile of the
wrapper's registration/validation work. No nonlinear iteration count is
fabricated.

The solver borrows the problem, spaces, and registered coefficients. Keep them
alive and do not mutate them during a collective call. Problem setters trigger
reassembly on the next solve. Call `NotifyCoefficientsChanged()` after modifying
a borrowed coefficient in place. Mesh/space changes require reconstruction.
An unchanged repeated solve reuses assembly but currently refactors MUMPS.

After `Assemble()` or `Solve()`, `GetLayout()`, `GetSystemMatrix()`, and `GetRHS()`
give const, solver-owned views of the retained-essential-row system. Accessors
reject stale problem revisions. Reassembly destroys these objects: do not retain
their references across a reassembly or solver destruction. The raw matrix's
`MultTranspose()` is the exact algebraic transpose of this linear system; this
does **not** expose a validated Stage F adjoint solver or sensitivity API.

## Build and run

From the workspace root, source the local MPI/compiler environment first:

```bash
source ./set_up_local.sh
export HYDRA_LAUNCHER=fork HYDRA_RMK=user
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cmake --build build --target stokes_brinkman stokes_brinkman_tests -j 12
"$PETSC_DIR/bin/mpiexec" -launcher fork -rmk user -n 4 \
  build/miniapps/stokes/stokes_brinkman
ctest --test-dir build -L '^stokes-brinkman$' --output-on-failure -j 8
```

The example and numerical tests are built only with MPI, double precision, and
MUMPS. The example generates its own P2/P1 mesh, uses domain-wise reaction and
a nonzero mean, and explicitly prints `convection OFF`.

## Verification scope

Generated-mesh tests run on one, two, and four MPI ranks and cover:

- Exact full-matrix/RHS recovery and blockwise transpose actions, for all four
  legacy discretizations, both augmented and unaugmented layouts, 2D/3D
  simplex/tensor meshes, variable viscosity, and nonzero velocity lifting.
- P2 cases and curved P3 2D recovery; direct solution comparisons at beta = 0.
- Manufactured variable-viscosity symmetric-stress traction and nonzero means
  on a nonunit-volume domain; vector and component traction registration.
- Stationary/moving Brinkman targets, attribute replacement, borrowed penalty
  updates, positive reaction energy, and reaction removal of rigid modes without
  velocity constraints, including a signed high-order simplex quadrature rule.
- Pressure-driven channel flow, pressure sign/absolute level, component velocity
  closure, equivalence to explicit traction, stationary-target reset, and
  domain-force replacement.
- Rejection of unsupported drag stabilization/backends, unresolved or excessive
  pressure gauges, shared-corner conflicts, incompatible flux, unconstrained
  rigid modes, stale algebraic views, and incorrect output spaces.

These tests establish the small-problem MUMPS baseline, not penalty-contrast,
Reynolds-number, or mesh-independent iterative robustness. Nonlinear convection
is Stage C; reaction/convection residual stabilization is Stage D; iterative
presets/projection are Stage E; supplied-state adjoints are Stage F.

## Validation checkpoint: November 8, 2026

- The pre-change Stage A regression gate closes with 484/484 passing at `-j 4`.
- Both new executables and all six Stage A/legacy executables build successfully
  with the compiler/MPI environment from `set_up_local.sh`.
- All 63 new tests pass independently on one, two, and four MPI ranks in
  6.39 seconds. The final **547/547 Stokes-label regression tests pass** at
  `ctest -j 4` in 1699.38 seconds, including all 424 legacy tests and all 60
  Stage A contracts. The 484 pre-existing test commands/properties are unchanged.
- The earlier `-j 8` full run passes 546/547. The unchanged legacy
  `stokes_mms_3d_invalid_pressure_gauge_none` exits nonzero but produces no
  captured diagnostic. It then passes ten standalone repetitions and the final
  full run. No legacy error-capture helper or test definition was modified;
  the precise cause of the missing output is not established.
- C++17 `-Wall -Wextra -Wpedantic` syntax checks succeed without warnings in
  the new sources or modified Stokes source. Existing MFEM headers emit warnings
  under these extra flags. Tracked-diff and new-file whitespace checks pass.

Validation logs and the pre-change Stage A source snapshot are retained under
`build/stage-b-20261108/` in the workspace. In particular, `focused.log`,
`regression.log`, `diagnostic-retry.log`, `regression-conservative.log`, their
JUnit reports, and `run.sh` preserve the numerical checks. Source/executable
hashes, the incremental `stage-b-only.patch`, and `completion-check.log` record
the tested changes. A configure/build overlap during development was resolved
by rerunning configuration and all builds sequentially; the failed logs are
retained separately as `configure-overlap.log` and `build-all-overlap.log`.
