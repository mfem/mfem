# Stage D: experimental nonlinear PSPG and GLS

Stage D extends `SteadyNavierStokesSolver` with full momentum-residual PSPG and
Oseen-test GLS, including convection, Brinkman reaction, moving targets, and
variable-viscosity symmetric-stress terms. The new scales require explicit
experimental opt-in; they are **not production defaults or robustness claims**.
The Stage B `StokesBrinkmanSolver`, legacy Stokes policies, and Stage C default
rejection of nonlinear PSPG/GLS remain unchanged without that opt-in.
The later [Stage E](NAVIER_STOKES_STAGE_E.md) adds iterative linear solves and
pressure projection without removing this stabilization opt-in.

## Selecting the extension

```cpp
problem.SetStabilization(GLSOptions{});
SteadyNavierStokesOptions options;
options.linear.backend = MumpsOptions{};
options.experimental_residual = ExperimentalResidualOptions{};
SteadyNavierStokesSolver solver(problem, options);
auto report = solver.Solve(velocity, pressure);
```

`ExperimentalResidualOptions` exposes nonnegative, finite `advection_scale` and
`reaction_scale`, both initially one. Zero disables the respective frequency in
tau, not the physical convection or reaction. The option is rejected for
Taylor–Hood/CIP; all choices must agree across MPI ranks. MUMPS remains the only
supported backend. No pressure-trace closure, backflow model, iterative preset,
or new adjoint solve is added.

Setting `options.convection = false` removes physical convection and its tau
frequency, while retaining consistent reaction stabilization. Positive reaction
with PSPG/GLS is available through this explicitly experimental nonlinear-solver
interface, not through the unchanged Stage B adapter.

## Residual and test operators

Using density-normalized pressure and viscosity, define

```text
L_nu(w) = -div(2*nu*epsilon(w)),
R_m(u,p) = L_nu(u) + (u dot grad)u + grad(p)
           + beta*(u - u_target) - f.
```

The unscaled continuity convention is retained. PSPG adds

```text
sum_K integral_K tau_K * R_m dot grad(q).
```

GLS instead adds one combined residual term

```text
sum_K integral_K tau_K * R_m dot T_u(v,q),
T_u(v,q) = L_nu(v) + (u dot grad)v + beta*v + grad(q).
```

GLS does not add PSPG a second time. Its strong test is the stated Oseen test,
not the full Frechet derivative of the nonlinear momentum operator. There is
no implicit grad-div penalty. The source includes both `f` and `beta*u_target`.

`stokes::EvaluateMomentumBasis` and its geometry-preserving Hessian workspace
are reused. Thus `L_nu` includes both `nu*(Delta(w)+grad(div(w)))` and the
viscosity-gradient contribution. It is not replaced by a component Laplacian,
and discrete `grad(div(w))` is not assumed zero. The existing requirement for
supported viscosity-gradient data remains in force.

## Experimental tau and quadrature

Let `k` be velocity degree, and use the existing
`h_K = mesh.GetElementSize(element, 1)`. Signed quadrature weights define cell
averages using physical volume. Then

```text
tau_0 = c_tau*h_K^2 / (average_K(nu)*(k+1)^4),
A = advection_scale^2 * (k+1)^2 / h_K^2,
D = reaction_scale^2,
X = tau_0^(-2) + A*average_K(|u|^2) + D*average_K(beta^2),
tau = X^(-1/2).
```

`c_tau` is the selected PSPG/GLS `tau_constant`. `A` is zero when convection is
disabled. The velocity in the average includes the prescribed essential lift.
There is no transient time scale, absolute-value differentiation at zero
velocity, or arithmetic nodal average. Invalid sampled viscosity, negative
reaction, nonpositive geometry, and invalid cell averages are rejected.

The residual and all its averages use the resolved nonlinear stabilization
quadrature when convection is enabled, and the original Stokes quadrature otherwise. Explicit
`problem.SetIntegrationOrder(q)` overrides both. The automatic convection rule
is `max(Stokes order, 3*k + 2*dimension*(geometry_order-1))` for convection/PSPG.
GLS instead uses `max(Stokes order, 4*k + 2*dimension*(geometry_order-1))` to
cover its quartic velocity products. `GetStabilizationIntegrationOrder()` exposes
that choice; it does not change the weak convection rule. These are practical
overintegration choices, not exactness guarantees for arbitrary coefficients
or curved rational integrands. The caller should check quadrature convergence
for coefficients outside the tested cases.

With convection disabled and beta zero, tau is the Stokes tau and the assembled
blocks and loads recover the Stokes discretization to roundoff using identical
quadrature. The no-opt-in path retains the original assembly itself, including
the stronger existing bitwise recovery tests.

## Exact Newton versus Picard

The exact state derivative is

```text
dR_m = L_nu(du) + (u dot grad)du + (du dot grad)u
       + beta*du + grad(dp),
dT_u = (du dot grad)v,
d_tau = -tau^3 * A * average_K(u dot du),
d(tau*R_m dot T_u) = d_tau*R_m dot T_u
                     + tau*dR_m dot T_u + tau*R_m dot dT_u.
```

For PSPG the test is `grad(q)` and its derivative is zero. For GLS the final
term is essential, even at zero velocity when the momentum residual is nonzero.
The cell-average tau derivative couples all local velocity basis functions.
All three terms are included in exact Newton and in public Jacobian/block/
`FlowLinearization` requests, regardless of the selected iteration method.

Picard deliberately freezes tau and the strong test for each correction and
keeps only the advecting-velocity part of the convective derivative. It is a
lagged/quasi-Newton iteration, not an exact Jacobian. The full nonlinear residual
still controls line search and convergence. No lagged approximation is silently
returned as the public exact Jacobian.

Material and target coefficients remain prescribed and state-independent.
Tests compare residual dependence on drag, viscosity, and target values with
independent analytic parameter loads. This validates those residual
dependencies; it does not introduce a public parameter-sensitivity/adjoint API.
Those interfaces and general end-to-end sensitivity gates remain Stage F work.

## Assembly, constraints, and initialization

The shared Stokes assembly supplies Galerkin viscosity, divergence/pressure,
reaction, boundary loads, and gauge blocks. An internal deferred-stabilization
mode omits only its old PSPG/GLS blocks and loads; the new parallel block
nonlinear form supplies the full stabilization exactly once. The public Stage B
constructor cannot select this incomplete intermediate assembly, and its solve
rejects deferred use.

The Stage C essential-row convention remains: evaluate physical terms using
the prescribed velocity lift, zero their essential rows and derivative columns,
and retain the original single identity equation. Pressure and gauge blocks
remain explicit; they are not reconstructed from assumed transpose identities.
Closed-domain constant pressure remains a null mode of the physical equations,
with the existing rank-zero integral-mean augmentation handling the gauge.

Default Stokes initialization now solves the **complete convection-off,
reaction-stabilized** system for experimental PSPG/GLS. It does not factor the
unstabilized intermediate matrix or silently use an inconsistent pressure-only
surrogate. The initialization solve is checked using its true linear residual.
Provided-state initialization and last-accepted-iterate failure behavior remain
unchanged. Correction counts exclude the optional initialization, as in Stage C.

## Decision gate and tested scope

CIP is reused unchanged, including MPI shared-face ownership and its viscous
`gamma*h_F^3/nu_F` scale. No reaction-aware or convection-aware CIP face formula
is accepted in this stage. Existing CIP tests remain part of the full regression.

The new PSPG/GLS formulas are explicitly experimental because consistency tests
and small direct solves do not establish general convection/penalty robustness.
Tests cover:

- Curved 2D/3D simplex/tensor exact state derivatives, Taylor remainders,
  transposes, and reconstruction of every velocity/pressure/gauge block.
- Variable-viscosity manufactured Newton/Picard solves with nonzero Dirichlet
  lifts, open traction, moving targets, heterogeneous drag, and nonzero means.
- Identical-quadrature Stokes recovery and closed-domain pressure null modes.
- Zero-velocity derivatives at zero/nonzero drag, including nonzero GLS residuals.
- Independent drag, viscosity, and target residual-parameter checks.
- Smooth P2/Q2 refinement with approximately third-order velocity and
  second-order pressure expectations, rather than borrowing a CIP superconvergence
  threshold.
- Bounded 2D triangle/quadrilateral sweeps over degrees 2/3, 3/5 cells per
  direction, viscosity 0.5/2, velocity scale 0.2/1, and piecewise drag 0 or
  100/1000. These use explicit quadrature order 12 and report sampled local
  `|u|*h/nu`, `beta*h^2/nu`, residuals, pressure/velocity errors, and iterations.

The sweep is manufactured-solution consistency evidence, not a general flow
robustness envelope. Arbitrary high Reynolds numbers, unresolved layers, extreme
penalty contrast, disconnected pressure modes, unsupported meshes/spaces, and
iterative conditioning are not validated by it. Keep experimental opt-in until
the broader decision gate in the implementation plan is met.

The complete four-rank logs contain 128 manufactured case configurations. Their
sampled `|u|*h/nu` ranges from 0.0273999 to 2.38805 and `beta*h^2/nu` from zero
to 222.222. Newton takes two or three corrections; maximum reported true
residual, velocity L2 error, and pressure L2 error are respectively
`2.61334e-13`, `4.74998e-15`, and `9.30495e-13`. These polynomial-exact solutions
are especially favorable consistency cases; do not interpret these numbers as
accuracy or convergence guarantees for general flows in that parameter range.
The complete logs are saved separately because CTest truncates passing output.

## Build and verification

```bash
source ./set_up_local.sh
export HYDRA_LAUNCHER=fork HYDRA_RMK=user
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cmake --build build --target steady_navier_stokes_nonlinear_tests -j 12
ctest --test-dir build -L '^navier-stokes-stabilization$' --output-on-failure -j 4
```

All new tests use generated meshes on one, two, and four ranks. Validation
artifacts and the pre-change Stage C snapshot are in `build/stage-d-20261108/`.

## Validation checkpoint: November 9, 2026

- All ten affected executables build with the compiler/MPI installation supplied
  by `set_up_local.sh`.
- All **68/68 Stage D tests pass** on one, two, and four ranks in 47.89 seconds.
  The separate four-rank sweep logs preserve all 128 case configurations.
- The full regression passes **692/693** at `ctest -j 4` in 1656.24 seconds.
  Every Stage D test passes. The unchanged legacy
  `stokes_mms_2d_invalid_pressure_gauge_none` returns the expected nonzero exit
  but supplies no captured diagnostic; it then passes ten isolated repetitions.
  This resembles previously observed legacy error-capture failures, but the
  precise cause is not established. No clean 693/693 full run is claimed.
- All 625 prior test commands/properties and the legacy error-capture helper
  remain unchanged. No test was disabled or had its expected failure relaxed.
- C++17 `-Wall -Wextra -Wpedantic` syntax checks pass without Stokes miniapp
  source warnings; existing MFEM headers emit warnings. Changed/new-file and
  tracked-diff whitespace checks pass.

The artifact directory includes earlier test-development failures, the final
focused and full logs/JUnit, isolated diagnostic repetitions, complete sweep
logs, source/executable hashes, and `stage-d-only.patch`. The initial test
helper's per-element collective was corrected before the final validation;
it is distinct from the remaining legacy diagnostic-capture limitation.
