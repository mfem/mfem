# Parallel MMS refinement driver

`stokes_mms` solves a manufactured Stokes problem on the supplied five-hole
meshes, inferring 2D or 3D from the input mesh. It handles fully periodic and
fully nonperiodic boxes. It runs in MPI, constructs fresh velocity/pressure
spaces and a fresh `StokesSolver` at each level, and reports the **outer Krylov iteration count at
every level**. MUMPS reports zero outer iterations.

This executable replaces `stokes_mms_2d` and `stokes_mms_3d`; use the same
arguments with `stokes_mms`. No dimension flag is needed. One-dimensional
meshes and embedded surfaces are rejected. Dimension-specific ParaView
collection names are retained so existing output paths remain unchanged.

## Build and run

Run from the workspace containing `mfem`, `build`, and `meshes`:

```bash
set +u; source ./set_up_local.sh
export HYDRA_LAUNCHER=fork HYDRA_RMK=user OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cmake -S mfem -B build
cmake --build build --target stokes_mms --parallel 8
```

Use the MPI launcher from the configured PETSc installation, not a system MPI.
The `fork`/`user` settings are for local runs without a batch allocation; use
the appropriate scheduler settings for multi-node jobs.

### Serial and parallel refinement

Select one input file with `-m` / `--mesh`. Refinement proceeds in two stages:

1. `-rs N` / `--refine-serial N` uniformly refines the serial mesh `N` times
   before MPI partitioning. No solves or visualization output occur during
   this stage.
2. `-rp N` / `--refine-parallel N` solves the partitioned mesh at level 0,
   then uniformly refines in parallel and solves at levels 1 through `N`.

Both counts default to zero and must be nonnegative. Each run produces
`N+1` reports for `-rp N`, regardless of the serial refinement count.
Level 0 is the mesh **after all serial refinements**, and level numbers in
console output, CSV, and ParaView count only parallel refinements. Mesh
filenames are literal paths; generated mesh-level sequences are not loaded.

2D periodic triangles:

```bash
"$PETSC_DIR/bin/mpirun" -n 4 build/miniapps/stokes/stokes_mms \
  --mesh meshes/holes-2d/triangles/level-1/void/periodic.mesh \
  -rs 0 -rp 2 --method taylor-hood --order 2 \
  --solver fgmres --preconditioner block-lower-p1-amg \
  --variable-viscosity --csv build/mms-2d-periodic.csv
```

For 2D nonperiodic meshes, substitute `nonperiodic.mesh`. Use `quadrilaterals`
instead of `triangles` for the quadrilateral collection.

3D nonperiodic tetrahedra:

```bash
"$PETSC_DIR/bin/mpirun" -n 4 build/miniapps/stokes/stokes_mms \
  --mesh meshes/holes-3d/tetrahedra/level-1/void/nonperiodic.mesh \
  -rs 0 -rp 1 --method pspg --order 2 \
  --solver fgmres --preconditioner block-upper-p1-amg \
  --csv build/mms-3d-nonperiodic.csv
```

Substitute `periodic.mesh` for 3D periodic meshes, or `hexahedra` for the
hexahedral collection. Hexahedral meshes have substantially more DOFs.

### Combining serial and parallel refinement

For example, `-rs 1 -rp 1` first refines once before partitioning, solves at
parallel level 0, then refines once in parallel and solves at level 1. In 2D
these meshes have 4 and 16 times as many elements as the input.
Use `-rs 1 -rp 0` to solve only the serially refined mesh.

```bash
"$PETSC_DIR/bin/mpirun" -n 4 build/miniapps/stokes/stokes_mms \
  --mesh meshes/holes-2d/triangles/level-1/void/nonperiodic.mesh \
  -rs 1 -rp 1 --method taylor-hood --order 2 \
  --solver gmres --preconditioner block-diagonal-p1-amg \
  --relative-tolerance 1e-10 --csv build/mms-2d-uniform.csv
```

## Solver options and reports

The same options apply in 2D and 3D; `--help` lists all of them.

| Option | Choices / meaning |
| --- | --- |
| `-m`, `--mesh` | Required input mesh filename |
| `--mms-case`, `-case` | `no-slip` (default, hole cutoffs) or `smooth` (nonzero analytic hole velocity, no cutoffs) |
| `-rs`, `--refine-serial` | Uniform refinements before partitioning (default 0) |
| `-rp`, `--refine-parallel` | Parallel uniform refinements, with solves at levels 0 through this count (default 0) |
| `--solver` | `mumps`, `gmres`, `fgmres` (default) |
| `--pressure-gauge`, `-pg` | `auto` (default), `mean-zero`, or `none`; explicit policies must agree with detected pressure nullspace |
| `--preconditioner` | `block-diagonal-amg`, `block-diagonal-p1-amg`, `block-upper-p1-amg` (default), `block-lower-p1-amg`; append `-assembled` to any P1-AMG choice for direct P1/Q1 coarse assembly |
| `--method` | `pspg` (default), `cip`, `gls`, `taylor-hood` |
| `--order` | Velocity degree, at least 2; pressure degree is one less for Taylor–Hood and equal otherwise |
| `--p1-smoothing-steps` | Fixed pre/post smoothing sweeps for the P1-AMG preconditioners |
| `--relative-tolerance`, `--absolute-tolerance` | True-residual stopping tolerances |
| `--max-iterations`, `--restart` | Total outer iteration budget and Krylov restart dimension |
| `--mu`, `--variable-viscosity` | Positive viscosity scale (default 1); optionally spatially varying viscosity |
| `-tau`, `--tau-constant` | Positive dimensionless PSPG multiplier (default 1) for automatically computed element weights |
| `-cip`, `--cip-gamma` | Positive dimensionless CIP multiplier (default 0.01) for automatically computed face weights; no additional degree scaling |
| `-gls`, `--gls-tau-scale` | Positive dimensionless GLS multiplier (default 1) for automatically computed element weights |
| `--quadrature-order` | Assembly quadrature order; default `-1` selects `2*order+2+2*dim*(g-1)` for coordinate degree `g` |
| `--print-level` | Extra solver verbosity; level reports are always printed |
| `--csv` | Output file, overwritten at startup and flushed after each level |
| `--paraview`, `--no-paraview` | Enable/disable per-level velocity and pressure output (disabled by default) |
| `--output` | ParaView output directory (default `stokes-mms-output`); ignored without `--paraview` |

Stabilization weights remain mesh-dependent when a multiplier is supplied.
PSPG/GLS use `tau_K = C*h_K^2/(mean(mu)_K*(k+1)^4)`, with velocity degree `k`,
local mesh size metric `h_K`, and cell-average viscosity `mean(mu)_K`.
CIP uses `gamma*h_F^3*(1/mean(mu)_K+ + 1/mean(mu)_K-)/2`, with
`h_F = min(h_K+,h_K-)` for adjacent elements. Omitting the options uses the
default multipliers; for example, `-tau 2` doubles the automatic PSPG weights
rather than setting every element's coefficient to 2. Only the selected
method's multiplier is used; Taylor–Hood ignores all three.

The preconditioner selection is ignored by MUMPS. `--solver mumps` requires an
MFEM build with MUMPS. Krylov preconditioners use elasticity BoomerAMG on the
velocity block; `--amg-interp-refine` controls its interpolation refinement.
See the main README for the block and P1-AMG definitions.

For example, `--preconditioner block-lower-p1-amg-assembled` uses direct P1/Q1
assembly for both coarse diagonal blocks, retaining fine-level stabilization
weights, quadrature, and curved geometry. The existing names continue to use
Galerkin coarse projection. All three assembled choices support both GMRES and
FGMRES, periodic/nonperiodic meshes, and the serial/parallel refinement options.

Rank zero writes a line starting with `MMS_LEVEL` after every solve:

```text
MMS_LEVEL dim=2 level=1 periodic=1 internal_interface_bcs=0 mms_case=no-slip method=taylor-hood solver=fgmres preconditioner=block-lower-p1-amg pressure_gauge=auto pressure_nullspace_detected=1 pressure_gauge_active=1 order=2 pressure_order=1 elements=... velocity_dofs=... pressure_dofs=... outer_iterations=28 converged=1 residual=... velocity_l2=... velocity_h1=... pressure_l2=... divergence_l2=... solve_seconds=... assembly_seconds=...
```

`outer_iterations` includes true-residual correction solves, if used. It does
not count AMG cycles or smoothing steps separately. DOFs are global true DOFs
before boundary elimination and exclude the single pressure-mean multiplier.
`residual` is the relative algebraic residual. `velocity_h1` is the H1
**seminorm** error; pressure L2 error is measured modulo an additive constant.
Error integration uses four more quadrature orders than assembly.
`assembly_seconds` is the maximum rank-local elapsed wall time for full-system
assembly: geometry validation, boundary lifting, stabilization weights, fine
matrices (including fine preconditioning matrices), RHS, and pressure-gauge
data. It excludes preconditioner setup, including directly assembled P1 coarse
matrices, as well as factorization and iterations. Both console and CSV reports
include this field. CSV columns `pressure_gauge`, `pressure_nullspace_detected`,
and `pressure_gauge_active` follow it, reporting policy, detected nullspace, and
active mean constraint separately. `mean-zero` explicitly requires a pressure
nullspace; `none` rejects one rather than attempting a singular solve. The
standard closed/periodic MMS boundaries require a constraint, so use `auto` or
`mean-zero`. See the README's gauge and compatibility section for the full policy.

`solve_seconds` remains the maximum rank-local elapsed time for assembly, solver
setup, and solve, **not linear-solver-only time**. It excludes mesh loading,
error evaluation, and visualization output. `geometry_order` and
`integration_order` report the coordinate degree and resolved assembly
quadrature order.

The appended CSV/console fields `assembly_performed`,
`preconditioner_setup_seconds`, `direct_setup_seconds`, `linear_solve_seconds`,
`finalization_seconds`, and `boundary_projection_seconds` separate setup,
application, cleanup/diagnostics, and MMS trace preparation. Explicit AMG
hierarchy setup and P1 coarse assembly are charged to preconditioner setup;
MUMPS analysis/factorization are charged to direct setup. The unused setup field
is zero. Linear time includes true-residual correction solves and right recovery.
Finalization includes backend destruction and final residual/field/gauge processing.
Each time is an independently measured MPI-max wall interval, so phase maxima
are not additive. The boundary projection is excluded from `solve_seconds` and
is zero for `no-slip`. All driver solves are fresh (`assembly_performed=1`);
the API can instead return zero when using a previously assembled system while
retaining its historical assembly time. Existing CSV column positions are unchanged.

The driver also appends `absolute_residual`, `order`, and `pressure_order` after
the timing fields. `absolute_residual` is the true global constrained-system
residual norm; `residual` in the console and `relative_residual` in CSV retain
their existing meanings. These fields let the 2D benchmark runner validate
both residual tolerances and the selected spaces without guessing from DOFs.
See `BENCHMARK_2D.md` for sequential screening/refinement commands and
reproducibility rules. The runner's `--refinements 0 1 2` launches independent
cold solves with `-rs 0/1/2 -rp 0`; each raw driver's `level` is therefore zero,
while runner CSVs identify each mesh with `serial_refinements`.

A nonconverged solve is still reported and subsequent requested levels are
attempted. The final exit status is 2 if any solve did not converge, and 0 if
all did. Invalid inputs or unrecoverable assembly errors terminate the run.

## Optional ParaView visualization

Add `--paraview --output DIR` to either driver to save the computed `velocity`
and `pressure` fields at **every requested refinement level**, including the
serially refined level-0 mesh and nonconverged iterates. Visualization is
disabled by default; `--no-paraview` disables it explicitly. The short forms
are `-pv`, `-no-pv`, and `-out DIR`.

For example, after setting up the environment as above:

```bash
"$PETSC_DIR/bin/mpirun" -n 4 build/miniapps/stokes/stokes_mms \
  --mesh meshes/holes-2d/triangles/level-1/void/nonperiodic.mesh \
  -rs 0 -rp 1 --method taylor-hood \
  --paraview --output build/mms-2d-viz

"$PETSC_DIR/bin/mpirun" -n 4 build/miniapps/stokes/stokes_mms \
  --mesh meshes/holes-3d/tetrahedra/level-1/void/periodic.mesh \
  -rs 0 -rp 1 --method taylor-hood \
  --paraview --output build/mms-3d-viz
```

Open `build/mms-2d-viz/stokes_mms_2d/stokes_mms_2d.pvd` or
`build/mms-3d-viz/stokes_mms_3d/stokes_mms_3d.pvd` in ParaView 5.5 or newer.
Each collection retains all levels, with the displayed timestep equal to the
parallel refinement level, **not physical time**. Each level has a parallel
`.pvtu` descriptor and one `.vtu` piece per MPI rank. Binary high-order VTK output
uses degree `max(velocity order, geometry order)` to retain curved geometry.
Use a distinct output directory for each run to avoid overwriting earlier
results. The driver prints the collection path after the last level; it does
not launch a viewer or require a GLVis server.

## Manufactured fields and boundaries

### Default no-slip case

The domain lengths are `(2,1)` or `(2,1,1)`. Define a periodic scalar potential
`psi = product_i sin²(pi*x_i/L_i) * product_h chi_h`. The five centers are the
mesh-generator centers. Each radial cutoff is zero for radius at most 0.10
and one for radius at least 0.20. In between, with
`t = (radius² - 0.10²)/(0.20² - 0.10²)`, it is
`chi = 126*t⁵ - 420*t⁶ + 540*t⁷ - 315*t⁸ + 70*t⁹`.
The potential is C4 across the cutoff endpoints. The velocity is

- 2D: `(psi_y, -psi_x)`.
- 3D: `(3*psi_y - 2*psi_z, psi_z - 3*psi_x, 2*psi_x - psi_y)`.

This gives exactly divergence-free velocity, periodic across opposite box
faces and identically zero on the outer walls and the radius-0.09 faceted
hole surfaces. Pressure is `sin(pi*x)*cos(2*pi*y)` with an additional
`cos(2*pi*z)` factor in 3D. Variable viscosity is
`mu0*(1 + 0.25*product_i cos(2*pi*x_i/L_i))`. The vector force is evaluated
analytically as `-div(2*mu*epsilon(u)) + grad(p)`, including viscosity-gradient
terms. There is no numerical differentiation in assembly.

The driver detects periodicity from mesh face connectivity, not filenames.
Outer boundary attributes are 1–4 in 2D or 1–6 in 3D; the following five
attributes identify the holes/interfaces. In the default case, nonperiodic runs impose zero
velocity on all walls and holes. Periodic runs constrain only the
holes/interfaces, **not** the periodically identified outer faces. Native
MFEM `.mesh` and exported Gmsh `.msh` files are supported, including Gmsh files
that retain the periodic outer labels.

### Smooth nonzero-boundary verification

`--mms-case smooth` uses `psi = product_i sin²(pi*x_i/L_i)` without any hole
cutoffs. It uses the same curl formulas, pressure, and viscosity as the default
case, with all forcing derivatives evaluated analytically from this smooth
potential. Velocity is divergence-free and periodic, vanishes on the outer
box, and is generally nonzero on each hole. No change to the Stokes operator
or stabilization is made.

```bash
"$PETSC_DIR/bin/mpirun" -n 4 build/miniapps/stokes/stokes_mms \
  --mesh meshes/holes-2d-order2/triangles/level-1/void/periodic.mesh \
  --mms-case smooth --method pspg --variable-viscosity -rp 1 \
  --relative-tolerance 1e-12 --csv build/smooth-mms.csv
```

The finite-element trace must also have zero net flux. Nodal interpolation of
an analytically divergence-free velocity does not guarantee this: the initial
3D mesh checks rejected such traces. Instead, the smooth case uses a constrained
L2 projection on the marked hole/interface boundaries of the actual mesh.
With boundary mass matrix `M`, exact-trace load `b`, and
`w = B^T*1` restricted to hole-boundary true DOFs, compute
`g0 = M^-1*b` and, when flux correction is needed,
`z = M^-1*w`, `g = g0 - z*(w^T*g0)/(w^T*z)`.
This enforces `w^T*g = 0` while minimizing the boundary L2 error. Outer-wall
DOFs stay zero; periodic outer faces receive no Dirichlet conditions.
The pressure multiplier does not repair incompatible data, and the production
solver's compatibility test is unchanged. Projection setup is performed before
the timed Stokes solve; it is not included in `assembly_seconds` or `solve_seconds`.

The trace is a finite-element approximation, not exact pointwise boundary data.
Console and CSV output distinguish:

- `boundary_l2_error`: computed velocity versus the analytic trace on prescribed boundaries.
- `boundary_dof_error`: maximum true-DOF mismatch against the prescribed finite-element trace.
- `hole_velocity_l2`: computed velocity norm on hole/interface boundaries; nonzero in smooth tests.
- `boundary_flux` and `gauge_multiplier`: independent compatibility/continuity diagnostics.
- `projection_flux_before`: flux of the unconstrained L2 projection, not of nodal interpolation.
- `projection_correction_l2`: L2 size of the flux correction; zero when it is unnecessary.

These columns follow `mms_case` at the end of the existing CSV fields, followed
by `velocity_l2_rate`, `velocity_h1_rate`, `pressure_l2_rate`, and
`boundary_l2_rate`. Rates are `log2(previous_error/current_error)` for successive
uniform parallel refinements; they are zero at level 0 or for zero errors. The
`stokes-smooth-mms` label covers all four mesh collections, both simplex and
tensor elements, periodic/nonperiodic boundaries, one/four-rank solves, all four
methods, and GMRES/FGMRES/MUMPS where available. Refinement tests require strict
decreases in velocity L2/H1, pressure L2, and analytic boundary L2 errors, with
rates above 2.5, 1.7, 1.5, and 2.0 respectively for the tested degree-two spaces,
as well as absolute accuracy bounds. Every smooth case also checks nonzero hole
velocity, accurate boundary-DOF enforcement, negligible net flux, and a negligible
gauge multiplier. Coarse 3D pressure errors can still be large; finer-level
accuracy is checked separately rather than assuming the coarsest mesh is asymptotic.

The momentum integrators use an equivalent element-local H1 nodal
representation of the coordinate map to evaluate coordinate Hessians,
including the mixed derivatives of nonaffine Q1 maps and higher-order curved
maps. The full Hessian chain rule supplies the derivatives required by
PSPG/GLS even when periodic coordinates use L2 elements without `CalcHessian`.
Mesh coordinates, discontinuous periodic traces, and periodic connectivity
remain unchanged; no input files are modified.

`filled` meshes intentionally impose MMS velocity constraints (zero in `no-slip`,
generally nonzero in `smooth`) on the marked
internal inclusion interfaces (`internal_interface_bcs=1`). They are not an
unconstrained material-transmission benchmark. These constraints also remove
the constant-velocity nullspace on fully periodic filled meshes. Pressure is
continuous and has the solver's global zero-mean gauge.

## Limitations and validation

- Use conforming triangle, quadrilateral, tetrahedral, or hexahedral meshes
  with uniform polynomial H1/L2 coordinate degree. Both the degree-one
  collections and `holes-2d-order2`/`holes-3d-order2` are supported. The solver
  checks positive Jacobians at sampled points; this is not a proof of global
  validity of a curved map. See the main README for the geometry contract.
- The driver is tailored to these five-hole boxes and boundary tags, not
  arbitrary domains. Partially periodic boxes are rejected. Periodicity must
  already be encoded in the mesh; the driver does not construct identifications.
- The default no-slip cutoff layers have large derivatives. Coarse meshes may have
  large errors; increase quadrature and refinement to resolve them. Do not
  infer asymptotic high-order rates from the coarsest two generated levels;
  the potential's C4 regularity also limits very-high-order convergence.
- `stokes_mesh-mms-fields_np1` and `_np4` check analytic gradients, force,
  divergence, periodicity, and both no-slip/nonzero traces. The `stokes-mms` CTest label
  additionally checks dimension inference from actual meshes, level/iteration reports, CSV output,
  serial/parallel refinement element counts, zero-refinement defaults,
  solver/preconditioner choices, and failure handling, including unsupported
  dimensions and embedded surfaces. Removed sequence and
  refinement options are rejected rather than silently accepted.
  Mesh-dependent tests are registered when the local fixtures are present.
- The `stokes-visualization` label checks 2D/3D output on one and four MPI
  ranks, retention of every level, both field arrays and all rank pieces,
  explicitly disabled output, and output after nonconvergence.

```bash
cmake --build build --target stokes_pspg_tests --parallel 8
ctest --test-dir build -R '^stokes_(mesh-mms-fields|mms_)' --output-on-failure -j 8
```
