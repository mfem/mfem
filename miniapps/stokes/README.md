# Parallel symmetric-gradient Stokes

This CPU/full-assembly miniapp solves steady incompressible Stokes flow using
MPI-distributed continuous H1 velocity and pressure spaces. Select PSPG (the
unchanged default), pressure-gradient CIP, full momentum-residual GLS, or
Taylor–Hood. All methods support triangles/quadrilaterals in 2D and
tetrahedra/hexahedra in 3D, with constant or spatially varying positive viscosity.

## Build and run

From the workspace containing `mfem`, `build`, and `set_up_local.sh`:

```bash
source ./set_up_local.sh
cmake -S mfem -B build
cmake --build build --target stokes_pspg stokes_mms stokes_pspg_tests --parallel 8
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$PETSC_DIR/bin/mpirun" -launcher fork -rmk user -n 4 \
  build/miniapps/stokes/stokes_pspg --element quad --order 2 --cells 4 \
  --solver fgmres --variable-viscosity --shift 0.2
"$PETSC_DIR/bin/mpirun" -launcher fork -rmk user -n 4 \
  build/miniapps/stokes/stokes_pspg --element hex --order 3 --cells 2 \
  --solver mumps
```

The launcher flags above select local MPICH processes without a batch allocation.
Use the appropriate scheduler launcher settings for multi-node runs. Do not mix
MPI installations: the configured compilers and launcher come from the PETSc
prefix. PETSc and SLEPc may remain enabled in MFEM, but this solver uses MFEM,
HYPRE, and optionally MUMPS, not PETSc/SLEPc solver interfaces.

`--help` lists options. `--dim` optionally checks the dimension inferred from
`--element` or `--mesh`. The example applies the manufactured velocity on every
boundary; its manufactured pressure has zero mean on the unit square/cube.
Use that domain for quantitative MMS errors. `--paraview --output DIR` writes
parallel velocity and pressure fields. Visualization is off by default.

The executable remains `stokes_pspg` for compatibility. For example:

```bash
for method in taylor-hood cip gls; do
  "$PETSC_DIR/bin/mpirun" -launcher fork -rmk user -n 4 \
    build/miniapps/stokes/stokes_pspg --method "$method" --element quad \
    --order 2 --cells 4 --solver fgmres --variable-viscosity --shift 0.2
done
```

`--order k` specifies velocity degree. Pressure degree is `k` except for
Taylor–Hood, which uses `k-1`. An explicit `--pressure-order` must match that
pairing. All methods require `k>=2`. `--tau-constant` and `--gls-tau-scale`
default to one; `--cip-gamma` defaults to 0.01. These are optional positive
dimensionless multipliers, not raw stabilization coefficients. Element/face
weights are computed automatically from mesh size, viscosity and, for PSPG/GLS,
velocity degree. The executable help lists the formulas; setting a multiplier
overrides only that scale, not the mesh-dependent computation.
Only the selected method's parameter is used and validated. In particular,
Taylor–Hood ignores all stabilization parameters.

## Five-hole mesh MMS driver

Use `stokes_mms --mesh FILE` for the periodic and nonperiodic meshes
under `meshes/holes-2d` and `meshes/holes-3d`. The driver infers 2D or 3D from
the mesh and rejects 1D meshes and surfaces embedded in 3D. It uses an analytic,
divergence-free manufactured solution adapted to the box and hole boundaries,
not the unit-box solution in `stokes_pspg`. Use `--mesh` for one input mesh,
`-rs` / `--refine-serial` for refinements before partitioning, and
`-rp` / `--refine-parallel` for subsequent parallel refinements. Both counts
default to zero; solves run at parallel levels 0 through `-rp`, with level 0
already including all serial refinements. It accepts
`--solver` and `--preconditioner` and reports **outer iterations at every level**,
along with DOFs, errors, residual, convergence, assembly time, and total solve
time. Optional `--csv` writes one row per level on rank zero, with
`pressure_gauge`, `pressure_nullspace_detected`, and `pressure_gauge_active`
appended after `assembly_seconds`, followed by MMS-case, boundary-error, and
boundary-projection diagnostics and refinement rates.
Optional `--paraview --output DIR` saves velocity and pressure at every level
in `DIR/stokes_mms_2d/stokes_mms_2d.pvd` or
`DIR/stokes_mms_3d/stokes_mms_3d.pvd`. Visualization is disabled by default;
series timesteps denote refinement levels, and solve timings exclude output.

`--mms-case no-slip` preserves the default cutoff-based solution with stationary
hole walls. For stronger verification, use `--mms-case smooth`: it removes the
hole cutoffs, evaluates analytic velocity/forcing throughout the domain, and
prescribes a flux-compatible L2 projection of the nonzero analytic hole trace.
The outer-wall trace remains zero. This tests nonzero boundary lifting on
faceted and curved meshes without relaxing the solver's flux check. Run
`ctest --test-dir build -L stokes-smooth-mms --output-on-failure -j 8` for
boundary enforcement, analytic-field, accuracy, and refinement checks.

The `stokes_pspg` and `stokes_mms` executables report `assembly_seconds`, the maximum rank-local wall
time for the latest full-system `StokesSolver::Assemble()`. It includes geometry
validation, boundary lifting, stabilization weights, fine matrices (including
fine preconditioning matrices), RHS, and pressure-gauge data. It excludes
preconditioner setup (including directly assembled P1 coarse matrices),
factorization, iterations, error evaluation, and visualization. The returned
`StokesSolveInfo::assembly_seconds` retains the last assembly duration when a
solve reuses an assembled system, including after explicit `Assemble()` calls.
The MPI-max total time remains `solve_seconds` in the mesh MMS driver and `seconds`
in `stokes_pspg`; it includes assembly, preconditioner setup, and the solve.

Both drivers also report MPI-max `preconditioner_setup_seconds`,
`direct_setup_seconds`, `linear_solve_seconds`, and `finalization_seconds`.
Setup includes explicit HYPRE AMG hierarchy setup (not a discarded solve),
P1 coarse construction, gauge preconditioning, and Krylov object construction.
MUMPS setup combines construction, matrix conversion, analysis, and factorization;
its application is timed separately. The unused backend setup field is zero.
Linear time includes right-preconditioned recovery and all residual-correction
solves. Finalization includes backend destruction, final residual verification,
field distribution, and gauge diagnostics. The four local intervals are reduced
together after finalization; their MPI maxima need not sum to the inclusive total.
No inner-loop barriers are added. Report printing and the timing reduction are
inside the total but outside these phase intervals.

`assembly_performed` distinguishes assembly in the current `Solve()` call from
the retained historical `assembly_seconds` after reuse or explicit assembly.
Backend setup is rebuilt on every solve even when assembly is reused.
The MMS driver additionally reports `boundary_projection_seconds`, outside
the solver total and zero for the no-slip case. Existing CSV columns keep their
positions; these six fields are appended after the refinement rates. Timings
remain available for nonconverged last iterates, not for fatal setup errors.
The `stokes-timing` CTest label validates this contract without speed thresholds.

For reproducible 2D screening/refinement, use `benchmark_2d.py plan` / `run` / `report`.
The default matrix has 60 configurations and 240 sequential launches including
warm-ups. Add `--refinements 0 1 2` for 180 configuration-level cases and 720
independent cold launches with calibrated level-specific gates. It preserves
manifests, per-attempt logs, failures, and raw CSVs;
produces summary tables and SVG plots; and supports fingerprint-checked resume.
Multi-level reports include phase/iteration ratios, MMS rates, and total-time
and iteration curves versus DOFs, with direct references matched by level.
See [BENCHMARK_2D.md](BENCHMARK_2D.md) for the fixed baseline, accuracy gates,
resource limits, and exact commands. The driver now appends `absolute_residual`,
`order`, and `pressure_order` after its timing fields for runner validation.

See [MESH_MMS.md](MESH_MMS.md) for commands, the manufactured fields, boundary
treatment, high-order geometry, and prescribed MMS interface constraints on
`filled` meshes.

## Reusable interface

Include `stokes_solver.hpp` and link the solver, preconditioner, and integrator
sources with MFEM, as the local CMake targets do. The namespace is `mfem::stokes`.
All custom integrators, including PSPG and quadrature adapters, are consolidated
in `stokes_integrators.hpp` / `stokes_integrators.cpp`.

```cpp
H1_FECollection collection(2, mesh.Dimension());
ParFiniteElementSpace velocity_space(&mesh, &collection,
                                    mesh.Dimension(), Ordering::byVDIM);
ParFiniteElementSpace pressure_space(&mesh, &collection);
StokesSolverOptions options;
options.linear_solver = LinearSolverType::FGMRES;
options.preconditioner = StokesPreconditionerType::BlockUpperTriangularP1AMG;
options.p1_smoothing_steps = 2;
StokesSolver solver(velocity_space, pressure_space, 1.0, options);
for (int index = 0; index < mesh.bdr_attributes.Size(); ++index)
{
   solver.AddVelocityBC(mesh.bdr_attributes[index], -1, 0.0);
}
solver.SetVolForce(1.0, 0.0, 0.0);
StokesSolveInfo info = solver.Solve();
const ParGridFunction &velocity = solver.GetVelocity();
const ParGridFunction &pressure = solver.GetPressure();
```

The coefficient constructor takes `(velocity_space, pressure_space, mu, grad_mu,
options)`. Supply a `Coefficient` and a matching `VectorCoefficient` for its
physical gradient. A zero gradient is valid for constant or elementwise-constant
viscosity; it is not a substitute for the gradient of a spatially varying
coefficient. The caller is responsible for supplying a consistent gradient.

Set `options.method` to `StokesStabilizationMethod::PSPG`, `CIP`, `GLS`, or `TaylorHood`.
`StokesSolver` is the concrete solver class. The public API and its ownership,
collective-call, boundary, and solution-validity contracts are documented in
`stokes_solver.hpp`. The solver borrows externally created spaces; it never
silently replaces them. For Taylor–Hood, create a
separate `H1_FECollection(k-1, dim)` for the pressure space and keep both
collections/spaces alive for the solver's lifetime.

Spaces must share a connected, conforming `ParMesh`, use uniform H1 degrees
with the method's pairing, and have vector dimensions `dim` and one. Velocity **must**
use `Ordering::byVDIM`; this is required by MFEM's elasticity AMG. The same
contract applies to the MUMPS backend. Uniform polynomial H1/L2 coordinate
geometry of degree `g>=1` is supported, including curved and periodic meshes.
Coordinate degree is independent of solution degree; superparametric `g>k`
meshes are allowed. Mesh/space updates are rejected; construct a new solver
after refinement or repartitioning.

PSPG and GLS require a velocity basis with implemented reference Hessians.
Positive/Bernstein and serendipity velocity bases are rejected during solver
construction with an explicit diagnostic. This restriction does not apply to
the pressure basis or to CIP/Taylor–Hood, which do not use velocity Hessians.

The solver borrows external spaces and coefficients. They must outlive it.
Constant inputs and options are copied; solution fields and coefficient adapters
are owned by the solver. Setters invalidate assembly and the previously exposed
solution. `Solve()` assembles as needed, and `Assemble()` unconditionally rebuilds
all matrices and RHS data. After changing a borrowed coefficient's internal
state, explicitly call `Assemble()`. Solves and registrations are collective:
call them in the same order with the same attributes/options on every rank.

`Solve()` returns convergence, iteration count, true residuals, pressure mean,
boundary flux, and gauge information: `pressure_gauge` is the requested policy,
`pressure_nullspace_detected` reports the detected constant-pressure mode, and
`pressure_gauge_active` reports whether the mean constraint is present.
Getters expose the last iterate even when
that solve did not converge; check `info.converged` before using it as a solution.
Convergence requires finite solution entries and a finite true residual on every
rank. Nonfinite assembled matrices or right-hand sides are rejected collectively,
including overflow produced by otherwise finite input coefficients.
The miniapp returns nonzero on nonconvergence. Configuration errors use MFEM's
error mechanism collectively. No adjoint or arbitrary algebraic-RHS API is
claimed in this version.

### Forces and boundary conditions

`SetVolForce` accepts a `VectorCoefficient`, a copied constant `Vector`, or
`real_t` components `(force_x, force_y, force_z=0)`. The default is zero force.
Vector dimensions must match the mesh; a nonzero z force in 2D is rejected.

Boundary registration follows the attribute/component style of `mtop`:

- `AddVelocityBC(attribute, direction, real_t)` prescribes a constant component.
- `AddVelocityBC(attribute, direction, Coefficient&)` prescribes a scalar field.
- `AddVelocityBC(attribute, VectorCoefficient&)` prescribes the whole vector.
- Directions are `0=x`, `1=y`, `2=z`, and `-1=all active components`.
- Vector coefficients must have dimension two or three as appropriate. z BCs in
  2D are errors, not silently ignored.
- A later registration replaces the selected components on that attribute.
  Thus a scalar registration can override one component of a prior vector BC.
- Different attributes meeting at a shared DOF must prescribe consistent values.
  Conflicting or nonfinite projected values are rejected.
- `DelVelocityBC()` removes all scalar/vector registrations and their adapters.

For Bernstein velocity bases (CIP/Taylor–Hood), boundary data are first
interpolated in a same-degree Gauss–Lobatto H1 space and then converted to the
Bernstein basis. This reproduces representable polynomial traces rather than
treating point samples as Bernstein coefficients. Shared-DOF consistency and
component overrides use the converted coefficients; pressure bases are unchanged.

An unspecified component has **zero natural traction** in that Cartesian
direction, not zero velocity and not the vector-Laplacian Neumann condition.
Prescribing only normal Cartesian components on axis-aligned walls gives a
free-slip condition. Arbitrary normal/tangential constraints are not implemented.
Boundary constraints must eliminate all rigid translations/rotations.

## Equations and stabilization

The PDE and stress convention are

```
-div(2 mu epsilon(u)) + grad(p) = f
div(u) = 0
epsilon(u) = (grad(u) + grad(u)^T) / 2
```

The PSPG uneliminated true-DOF system is

```
[ A    -B^T ] [u] = [F]
[ B+C    S  ] [p]   [G]
```

Here `A=(2 mu epsilon(u),epsilon(v))`, `B=(div(u),q)`,
`C=sum_K (tau_K L(u),grad(q))`, `S=sum_K (tau_K grad(p),grad(q))`,
`F=(f,v)`, and `G=sum_K (tau_K f,grad(q))`. The full strong operator is

```
L(u) = -mu (Delta(u) + grad(div(u))) - 2 epsilon(u) grad(mu).
```

`ElasticityIntegrator(mu,0,1)` supplies `A`; `C` uses physical Hessians and
gradients, including non-affine mapping corrections. Its element-local columns
are component-blocked even though global velocity storage is interleaved
`byVDIM`. Both essential-velocity RHS shifts are applied before elimination:
`F -= A*u_D` and `G -= (B+C)*u_D`. The upper-right block is never `-(B+C)^T`.

The stabilization coefficient is shared by `C`, `S`, and `G`:

```
tau_K = C_pspg * h_K^2 / (average(mu)_K * (k+1)^4)
```

`h_K` is MFEM's minimum perfect-reference Jacobian singular value at the element
center (`GetElementSize(element,1)`). Viscosity averaging uses volume quadrature.
The default integration order is `2*k+2+2*dim*(g-1)`, where `g` is the
coordinate degree; `--quadrature-order` overrides it with a positive value.
This preserves `2*k+2` on degree-one geometry. The center-based `h_K` definition
is intentionally unchanged: equivalent coordinate representations use the
same stabilization scale, independent of the quadrature override. It is not
a bound on the smallest length scale throughout a strongly curved element.
The dimensionless `C_pspg` defaults to one. Neither the coefficient nor the
quadrature order is claimed universally optimal for distorted meshes, arbitrary
coefficients, high contrast, or arbitrarily high polynomial degrees.

### Additional discretizations

CIP uses `[A, -B^T; B, S_CIP]` and RHS `[F; 0]`, where

```
S_CIP(p,q) = sum_F gamma * h_F^3 / mu_F * integral_F jump(grad(p).n) jump(grad(q).n)
h_F = min(h_K1, h_K2)
1/mu_F = (1/average(mu)_K1 + 1/average(mu)_K2)/2
```

The same unit normal is used for both sides. Only interior faces are penalized;
there is no pressure-value jump or exterior-boundary penalty. `h_K` and volume
averages are the same as for PSPG. These element data are exchanged using
piecewise-constant parallel grid functions before shared-face integration, so
user coefficients are evaluated only on their owning elements. MFEM's
`ParBilinearForm` assembles local rows on shared faces and maps to true DOFs;
there is no extra manual face contribution or factor of one half. Face
quadrature uses the same geometry-aware default. Gamma has no hidden degree scaling and is a
tunable shape-regular-mesh baseline, not a universal hp-optimal parameter.

**CIP mesh limitation:** the gradient-jump penalty does not control every
equal-order pressure mode on very coarse grids. On the two-by-two square
(`Q2/Q2`), let `f(x)=(x-1/2)-(10/3)(x-1/2)*abs(x-1/2)`. The nonconstant,
mean-zero pressure `f(x)f(y)` is C1, has zero CIP energy, and is invisible to
all free velocity tests under full Dirichlet conditions. The tensor product
in 3D gives the same counterexample. Tests explicitly verify these kernels.
As a conservative guard, assembly rejects fully Dirichlet all-tensor meshes
with at most `2^dim` elements, with a refinement diagnostic. This is not a
general inf-sup test: other deficient mesh/degree combinations may exist.
The tested Cartesian tensor grids use at least three cells per axis. No PSPG
term, boundary pressure penalty, or extra pressure pin is silently added.

GLS adds `sum_K tau_K (L(u)+grad(p)-f, L(v)+grad(q))`:

```
[ A+H  -B^T+D^T ] [u] = [F+F_H]
[ B+D      S    ] [p]   [G    ]
H(u,v) = sum_K (tau_K L(u), L(v))
D(u,q) = sum_K (tau_K L(u), grad(q))
F_H(v) = sum_K (tau_K f, L(v))
```

`D`, `S`, and `G` reuse the PSPG definitions, with `gls_tau_scale` in place of
`pspg_constant`. Every residual term includes the viscosity gradient. Both
mixed blocks are assembled independently before transposition/elimination to
avoid unsafe sparse subtraction or incompatible off-diagonal maps. The GLS
upper-right block is **not** `-(B+D)^T`.

Taylor–Hood uses `[A, -B^T; B, 0]` with RHS `[F; 0]`. Its pressure block is
absent, not a small regularization. It does not assemble residual/Hessian
stabilization terms. All methods apply the lifting from the complete velocity
and continuity blocks, then eliminate prescribed velocity rows/columns. Gauge
moments and block offsets always use the supplied pressure space.

### Gauge and compatibility

Both drivers accept `--pressure-gauge auto|mean-zero|none` (short option
`-pg`). The default `auto` preserves automatic detection and treatment.
The API equivalent is `StokesSolverOptions::pressure_gauge`, using
`StokesPressureGauge::Auto`, `StokesPressureGauge::MeanZero`, or
`StokesPressureGauge::None`.

| Policy | Constant-pressure nullspace detected | Absolute pressure already determined |
| --- | --- | --- |
| `auto` (default) | Add a mean-zero constraint | Do not add a constraint |
| `mean-zero` | Add a mean-zero constraint | Reject the request |
| `none` | Reject the request | Do not add a constraint |

For example, explicitly require the mean-zero constraint on the closed unit
domain:

```bash
mpirun -n 4 build/miniapps/stokes/stokes_pspg --pressure-gauge mean-zero
```

Boundary conditions create or remove the nullspace; this option controls its
treatment, not its existence. The current backends solve a nonsingular,
possibly augmented system, not a projected singular system. `none` does not
enable a nullspace-aware Krylov method or pressure pinning. Forcing a mean-zero
constraint when natural traction determines pressure would change the equations
and is rejected. Detection and policy validation repeat on every assembly,
including after boundary-condition changes.

The solver detects a constant-pressure null mode from the free momentum
coupling. When a constraint is needed, it adds the pressure mass-moment vector
and one global multiplier to enforce `integral(p)=0` exactly. The multiplier belongs to rank
zero, with distributed couplings to every pressure partition. Prescribed net
flux must be compatible before solving; the multiplier cannot silently repair
incompatible boundary data, and no policy bypasses this check. With the
uneliminated divergence matrix and constant-pressure coefficient vector `c`,
the check uses `w=B^T*c`, `r=abs(B)^T*abs(c)`, and the prescribed lifting `u_D`:
it requires `abs(w^T*u_D) <= 1e-10*S + 128*epsilon*R`, where
`S=sum_i abs(w_i*u_D,i)`, `R=sum_i r_i*abs(u_D,i)`, and `epsilon` is machine precision.
The second term allows roundoff in cancellation-dominated fluxes, including
tangential traces. Both terms scale with boundary data and geometry, never
with body force or stabilization forcing. There is no unit absolute flux
floor; exactly zero lifting requires exactly zero flux.

If a natural boundary fixes absolute pressure, no gauge is added and pressure
is not shifted afterward. Pressure mass is used only for preconditioning, never
as a penalty in the physical system. Multiple disconnected components and
their independent gauges are outside the current connected-domain contract.

Solver summaries, driver result lines, and mesh-MMS CSV files report the selected
policy, detected nullspace, and active constraint separately. The CTest label
`stokes-pressure-gauge` covers all four methods, available backends, closed/slip
and natural boundaries, reassembly, invalid policies, flux compatibility, and
driver diagnostics on one/four ranks, including periodic 2D/3D mesh-MMS cases.

## Linear solvers

`--solver mumps` uses a distributed monolithic `HypreParMatrix` and nonsymmetric
MUMPS factorization. Requesting this backend without MFEM MUMPS support fails
explicitly. Successful factorization alone is not treated as convergence: all
backends are checked against the true distributed residual.

`StokesSolverOptions::preconditioner` selects a `StokesPreconditionerType`
independently of `linear_solver`. MUMPS ignores the preconditioner selection.
The default `BlockDiagonalAMG` (`--preconditioner block-diagonal-amg`) preserves
the original fixed block-diagonal inverse approximation:

- Velocity: one BoomerAMG cycle, configured with
  `SetElasticityOptions(&velocity_space, amg_elasticity_interp_refine)`.
  Elasticity interpolation refinement defaults to true and is toggled with
  `--amg-interp-refine` / `--no-amg-interp-refine`.
  The velocity block is `A+H` for GLS and `A` otherwise.
- Pressure: inverse diagonal of the sum of the inverse-viscosity-weighted mass
  matrix and the selected pressure stabilization matrix; mass only for
  Taylor–Hood.
- Gauge, when present: inverse scalar `m^T D_p^{-1} m`.

Three P1-AMG block structures are available, each with Galerkin or directly
assembled coarse operators:

- `BlockDiagonalP1AMG` (`--preconditioner block-diagonal-p1-amg`): applies a
  two-level inverse to **both** the velocity block `K=A` (or `A+H` for GLS) and
  pressure approximation `Q=M_(1/mu)+S` (mass only for Taylor–Hood).
- `BlockUpperTriangularP1AMG` (`--preconditioner block-upper-p1-amg`): uses
  the same diagonal inverses and back substitution with the actual constrained
  upper-right block `C=-B^T` (or `-B^T+D^T` for GLS). Without a gauge this is
  `z_p=Qinv*r_p`, `z_u=Kinv*(r_u-C*z_p)`.
- `BlockLowerTriangularP1AMG` (`--preconditioner block-lower-p1-amg`): uses
  the same diagonal inverses and forward substitution with the actual
  constrained continuity block `L=B+D` for PSPG/GLS or `L=B` for CIP/Taylor–Hood.
  Without a gauge this is `z_u=Kinv*r_u`, `z_p=Qinv*(r_p-L*z_u)`; `L` is not
  inferred by transposing the upper-right block.

Each two-level inverse owns a same-mesh degree-one H1 space: P1 on simplices,
Q1 on quadrilaterals/hexahedra, **not** a refined LOR mesh or high-order AMG.
Parallel true-DOF interpolation `P` defines the Galerkin coarse matrix
`P^T*K*P` or `P^T*Q*P` for the original variants. Component-wise velocity
essential conditions are imposed on both fine transfer rows and coarse columns;
coarse constrained rows are
identity. Velocity coarse AMG uses `SetElasticityOptions` on the **coarse**
vector space, honoring `amg_elasticity_interp_refine`. Pressure coarse AMG is
scalar. One zero-start BoomerAMG cycle supplies the coarse correction.

To assemble directly on P1/Q1 instead of forming distributed Galerkin products,
select one of these additional enum/CLI pairs:

| Enum | `--preconditioner` |
| --- | --- |
| `BlockDiagonalP1AMGAssembled` | `block-diagonal-p1-amg-assembled` |
| `BlockUpperTriangularP1AMGAssembled` | `block-upper-p1-amg-assembled` |
| `BlockLowerTriangularP1AMGAssembled` | `block-lower-p1-amg-assembled` |

These choices are available in `stokes_pspg` and `stokes_mms`.
Both coarse blocks are assembled using the same bilinear forms
as their fine counterparts: symmetric-gradient viscosity plus the GLS
momentum-residual term for velocity, and inverse-viscosity mass plus PSPG/GLS
diffusion or CIP jump stabilization for pressure. Taylor–Hood uses mass only.
GLS terms are not assumed to vanish on P1/Q1: variable viscosity and curved or
nonaffine maps can give nonzero momentum residuals even at degree one.

Direct assembly retains the fine-level quadrature, element sizes, cell-average
viscosities, and `tau` (including its **fine velocity degree** scaling). It does
not recompute stabilization with `k=1`. CIP includes MPI shared faces. With
these matched forms and weights, the coarse matrices agree with Galerkin
projection up to roundoff, including componentwise essential elimination.
The mesh geometry, transfers, fine smoothers, coarse elasticity AMG, triangular
couplings, and pressure-gauge treatment are unchanged. Coarse assembly happens
once per preconditioner construction, never inside `Mult`.

This avoids distributed `P^T*K*P`/`P^T*Q*P` products, but still assembles the
fine matrices for the Stokes operator and fine smoothing. Setup speed depends
on mesh, quadrature, method and MPI layout; this option does not guarantee
fewer outer iterations or faster solves on every problem.

Fine pre/post smoothing uses HYPRE's convergent l1-Jacobi, with absolute row-sum
scaling including off-rank entries. For an SPD block this scaling bounds the
largest scaled eigenvalue by one, so the stationary iteration converges.
`p1_smoothing_steps` / `--p1-smoothing-steps` sets the positive fixed number of
sweeps **on each side**, default 2. There are no tolerance-based inner solves;
the resulting action remains fixed and linear for ordinary GMRES as well as
FGMRES. Convergence of this smoother does not imply mesh/order-independent
outer iteration counts; smoothers and coarse hierarchies can be tuned later.
The residual tolerance is not a pressure-error bound: tight high-order pressure
comparisons can require smaller tolerances, as recorded in `VALIDATION.md`.

With a mean-pressure gauge, define `s=m^T*Qinv*m > 0`. The P1 diagonal variant
uses `1/s` for its extra scalar block. The upper variant retains the gauge
column and uses the negative gauge Schur sign: `z_g=-r_g/s`,
`z_p=Qinv*(r_p-m*z_g)`, then `z_u=Kinv*(r_u-C*z_p)`. The gauge scalar is owned
only on rank zero, but the column action is collective. The lower variant
first computes `z_u` and `z_p` as above, then applies the gauge row:
`z_g=-(r_g-m^T*z_p)/s`. This row action is also collective, including ranks
with no locally owned multiplier. No gauge is introduced
for natural boundaries that fix absolute pressure. These approximations do
not change the physical pressure block or regularize the Stokes equations.

Each solve constructs a fresh factorization/preconditioner; setters and explicit
reassembly therefore rebuild the hierarchy with current coefficients and BCs.
All integrators remain together in `stokes_integrators.hpp/.cpp`; two-level and
upper-triangular helpers are in `stokes_preconditioners.hpp/.cpp`.
The lower variant reuses MFEM's `BlockLowerTriangularPreconditioner`.

Both Krylov paths use right preconditioning. For GMRES this is implemented as
the operator `A*P^{-1}`, followed by recovering `x=P^{-1}y`; MFEM's built-in GMRES
preconditioner interface otherwise uses a left-preconditioned residual. This
keeps the stopping criterion in the physical RHS/residual norm for both paths.
If the Krylov residual estimate passes but the recomputed true residual does
not, at most two correction solves use that true residual as the RHS. These
reuse the same preconditioner, count against the original iteration budget,
and do not relax the requested tolerance. This addresses residual-estimate
drift in tight high-order CIP solves; iteration-limit failures are not retried
with an unreported extra budget.

## Validation

### High-order geometry

All four methods accept uniform-degree polynomial H1/L2 coordinates on the
supported tri/quad/tet/hex families. PSPG and GLS prepare an equivalent local
H1 nodal map of degree `g` once per element/integrator, then evaluate physical
Hessians with the full chain rule. This supplies derivatives missing from
MFEM's L2 coordinate elements without changing periodic coordinates or
linearizing curved geometry. The chain rule has no absolute curvature cutoff.
Coefficients and boundary data still use the original transformation.

Both P1-AMG constructions keep their P1/Q1 spaces on the **same curved mesh**.
Coarse functions need not reproduce physical rotations exactly; elasticity
AMG uses projected near-nullspace information. No coarse mesh is straightened.

Signed Jacobians are checked at assembly quadrature points, coordinate nodes,
and reference vertices before BC projection or assembly. These sampled
checks reject detected folded/singular maps but do not prove global validity.
NURBS/rational maps, variable coordinate order, nonconforming meshes,
prisms/pyramids, and embedded surfaces remain unsupported.

The automatic quadrature is a conservative starting rule, not an exactness
guarantee for nonpolynomial mapped integrands or arbitrary coefficients. Check
sensitivity with a higher `--quadrature-order` on strongly distorted meshes.
`StokesSolveInfo::geometry_order` and `integration_order` expose the coordinate
degree and resolved rule; the mesh MMS driver includes them in level/CSV reports.

`stokes_geometry_np1` and `_np4` check maps, Jacobians, physical Hessians at
coordinate scales `1e-6`, `1`, and `1e6`, degrees 1/2/3, H1/L2 coordinates,
curved CIP traces, and P1-AMG actions. The `curved-*` suites exercise genuinely
curved 3D elements, all solvers/preconditioners, BCs, quadrature sensitivity,
partition invariance, and refinement. Quadratic mesh tests include native,
periodic Gmsh, and intentionally constrained filled meshes.

### Solver validation

See `VALIDATION.md` for the extension's measured convergence tables,
parameter sensitivity, reproduction commands, and numerical limitations.

```bash
export HYDRA_LAUNCHER=fork HYDRA_RMK=user
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
ctest --test-dir build -R '^stokes_' --output-on-failure
```

The `stokes-p1-assembled` label tests direct coarse assembly on one and four
MPI ranks. Matrix checks compare against constrained Galerkin projection for
all four methods, tri/quad/tet/hex elements, coordinate degrees 1/2/3, variable
viscosity, and componentwise velocity constraints. They also verify fixed,
repeatable actions and that assembly occurs only during setup. Solver tests
cover all three block structures, GMRES/FGMRES, curved boundaries, partition
invariance, coefficient reassembly, and periodic high-order MMS meshes.
The new curved-boundary reproduction checks use a `1e-14` relative solve
tolerance while retaining the `1e-8` field-error bound: on the curved 3D
Taylor–Hood plug case, a `1e-12` algebraic residual tolerance is insufficient
to guarantee that pressure-error bound for every preconditioner.

```bash
ctest --test-dir build -L stokes-p1-assembled --output-on-failure -j 8
```

The tests include independently evaluated polynomial stresses/residuals,
distorted-element Hessian mappings, rigid rotations, both viscosity models,
component/vector/slip/open-boundary conditions, all force overloads, Krylov/direct
comparisons, partition invariance, coefficient reassembly, nonconvergence,
invalid configurations, quadrature sensitivity,
and manufactured refinement suites. The convergence suite prints machine-readable
`CONVERGENCE` rows with errors and observed slopes. Its 2D resolutions are 4, 8,
and 16 cells per direction; 3D resolutions are 2, 3, and 4, except CIP and
Taylor–Hood hexahedra, which use 3, 4, and 5. Error quadrature is higher than
assembly quadrature. For new methods the finest velocity rates must exceed
`k+0.5` in L2 and `k-0.5` in H1; Taylor–Hood pressure must exceed `k-0.5`.
These acceptance floors apply to the tested smooth solutions/mesh sequences,
not arbitrary meshes.

Each new method runs solver, boundary, sensitivity, partition, and reassembly
suites on one and four ranks. Additional operator tests cover CIP local face
symmetry/side reversal, constant and linear null actions, nonzero jump energy,
distributed symmetry and serial/MPI energy agreement; GLS polynomial residual
actions, nonzero velocity source correction, symmetry, positivity, and mixed
transpose identities for both viscosity models. Taylor–Hood solutions are
checked against independently assembled unstabilized Galerkin equations.
Polynomial Stokes solutions exercise nonzero lifting and both GLS force terms.
Open-boundary tests compare absolute pressure without mean subtraction.

Exact-polynomial boundary and viscosity-reassembly comparisons use a tighter
linear tolerance (`rtol=1e-12`, `atol=1e-14`) to separate discretization/API
correctness from pressure-error amplification by the saddle-point system.
The affine PSPG block-diagonal-AMG Krylov/direct comparisons exercise the default
solver tolerances. Curved PSPG comparisons use `rtol=1e-13`, `atol=1e-15` for
both reference and comparison solves, retaining the `1e-7` pressure-agreement
threshold. Public solver defaults are unchanged.
For the new methods, direct-solution accuracy comparisons use `rtol=1e-12`,
`atol=1e-14`; a small true residual at the default tolerance need not imply a
pressure coefficient error below `1e-7`. CIP coefficient/boundary comparisons
use `rtol=1e-14`, `atol=1e-16`, restart 200 and at most 3000 iterations because
its weakly controlled pressure modes amplify algebraic error. The basic block preconditioner can
require these larger budgets on high-order CIP systems; production runs must
check `converged` rather than assume the defaults suffice.

The initial Taylor–Hood constant-viscosity Q2/Q1 hex experiment had pressure
L2 errors `0.0632419`, `0.585707`, and `0.244848` at 2, 3, and 4 cells per
axis: pressure was nonmonotone on that very coarse sequence despite decreasing
velocity errors. The acceptance track therefore includes the finer 3/4/5
sequence; the coarse observation is not a claimed monotone convergence result.

Use `ctest -L stokes -LE slow` for the shorter suite. Failed MPI runs must not be
mistaken for numerical failures caused by the PDE: the MPI runtime requires
local sockets and therefore cannot execute inside a socket-denying sandbox.

### Recorded PSPG convergence

The following results were measured with MFEM 4.10.1, GCC 13.3.1, HYPRE 2.31.0,
MUMPS 5.7.3, double precision, and four MPI ranks. These are the finest levels
of the variable-viscosity cases, with `mu=1+x+y[+z]`, `C_pspg=1`, and a nonzero
constant velocity offset. Rates use the preceding resolution. The suite also
checks constant viscosity on every listed element family/order.

| Element | k | Cells/axis | Velocity L2 | Velocity H1 seminorm | Pressure L2 | L2 velocity rate | H1 velocity rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Triangle | 2 | 16 | 1.39778e-3 | 1.59886e-1 | 1.54161e-1 | 3.095 | 1.966 |
| Triangle | 3 | 16 | 4.39298e-5 | 7.66249e-3 | 5.77889e-3 | 4.130 | 3.013 |
| Quadrilateral | 2 | 16 | 7.72316e-4 | 8.01431e-2 | 5.29188e-2 | 2.990 | 1.995 |
| Quadrilateral | 3 | 16 | 1.78160e-5 | 2.66181e-3 | 6.18669e-4 | 3.983 | 2.996 |
| Tetrahedron | 2 | 4 | 2.35220e-1 | 5.88639 | 4.76226 | 2.869 | 1.720 |
| Tetrahedron | 3 | 4 | 3.49735e-2 | 1.27332 | 1.85963 | 3.877 | 2.721 |
| Hexahedron | 2 | 4 | 1.18987e-1 | 3.25964 | 2.43935 | 2.811 | 1.961 |
| Hexahedron | 3 | 4 | 1.10308e-2 | 4.30429e-1 | 2.22537e-1 | 3.885 | 2.949 |

The 3D meshes are deliberately coarse, especially for pressure; these numbers
are convergence evidence, not fine-grid accuracy or scalability claims.

Baseline validation before the extension: all 21 Stokes CTest cases passed, including 48
manufactured-solution solves. Existing MFEM Hessian/Laplacian tests pass (1,380
assertions), as do the selected parallel direct-solver, BoomerAMG, preconditioner
update, and matrix tests on one and four ranks (39 and 43 assertions). Command-line
help and four-rank ParaView output were also exercised.

Valgrind 3.27.1 could not run the memory check: it encounters an unsupported
AVX-512 instruction in MFEM's static `MemoryManager::Init`, before entering
the miniapp. No successful memory-check result is claimed. A separate generic-CPU
MFEM/dependency build would be needed for that check; the requested build
configuration was not changed to accommodate Valgrind.
