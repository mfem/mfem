# Mesh-driven channel flow with a Brinkman inclusion

`channel_flow` reads a three-dimensional channel mesh and solves Stokes–Brinkman
or steady Navier–Stokes–Brinkman using the existing `mfem::flow` implementation.
There is no separate numerical backend. The default is Taylor–Hood degree 2/1,
FGMRES with upper-triangular P1-AMG, and convection disabled. Execution is MPI,
CPU, double precision, and full assembly.

For Navier–Stokes, the built-in velocity preconditioner includes the current
convective Newton/Picard terms; Stokes and its initialization retain the SPD
viscous/Brinkman proxy. See [convection-aware preconditioning](NAVIER_STOKES_PRECONDITIONER.md).

The pressure Schur approximation defaults to `--schur mass`. For an experimental
least-squares commutator alternative, use `--schur lsc --lsc-solver mumps` with
`--backend fgmres`. This factors only the auxiliary pressure Laplacian, not the
full flow system. Alternatively, `--lsc-solver amg --lsc-amg-cycles 2` uses a
fixed number of BoomerAMG cycles for each of the two pressure Laplacian solves.
The LSC middle operator includes the current Newton/Picard velocity block,
including convection and Brinkman reaction. The summary records the chosen
Schur approximation and LSC inner solver. This option does not guarantee
Reynolds-independent convergence.

Pressure convection–diffusion is available through `--schur pcd --pcd-solver amg
--pcd-amg-cycles 2`, or `--pcd-solver mumps` for direct mass/Laplacian inverses.
It uses the current velocity, viscosity, and Brinkman coefficient in a scalar
pressure convection–diffusion operator. See [PCD preconditioning](PCD_PRECONDITIONER.md)
for the composition order, auxiliary inlet/outlet boundary treatment, and limitations.

## Mesh and boundary contract

The supplied files are `meshes/channel-3d/channel_hex.mesh` (80 hexahedra) and
`meshes/channel-3d/channel_tet.mesh` (435 tetrahedra), both on
`[0,10] x [0,1] x [0,1]`. The driver also accepts other first-order conforming
axis-aligned rectangular channel meshes with the same attribute convention:

| Attribute | Location | Condition |
| --- | --- | --- |
| Boundary 1 | Minimum x | Prescribed inlet velocity |
| Boundary 2 | Maximum x | Natural pressure-derived traction |
| Boundaries 3, 4 | Minimum/maximum y | No slip |
| Boundaries 5, 6 | Minimum/maximum z | No slip |
| Domain 1 | Entire box | Fluid plus spatial Brinkman coefficient |

The driver checks dimensions, orientation, positive quadrature Jacobians,
domain attributes, total volume, boundary locations, and boundary areas before
partitioning. `--serial-refinements N` applies N uniform refinements before
partitioning. `--min-elements-per-rank N` adds serial refinements until every
partition has **strictly more** than N elements, checking the actual smallest
partition rather than just the average. Zero disables this threshold. If a
partition is too small, the driver refines the serial mesh and repartitions.
`--refinements` still controls additional parallel refinements. All refinement
precedes construction of the spaces; the replicated serial mesh is released
after partitioning. Reports distinguish input/serial element counts, applied
serial refinement count, and minimum/maximum pre-parallel-refinement partition
sizes.
Curved, nonconforming, periodic, perforated, and non-box meshes are outside this
driver's scope; this is narrower than the solver library's geometry support.

The inlet shape is proportional to
`36 * U_mean * y_hat*(1-y_hat) * z_hat*(1-z_hat)`, where the transverse coordinates
are normalized to `[0,1]`. It vanishes on all inlet-wall intersections. Its
finite-element interpolant is normalized to give the requested **discrete mean
inlet speed**, including on P2 tetrahedra where the quartic shape is not exactly
representable. The normalization is reported. This is not the exact developed
square-duct profile.

`--outlet-pressure` supplies `-p_out*n` traction, not essential pressure values.
No pressure-mean constraint is imposed. The computed mean outlet pressure may
differ from `p_out` because the natural condition includes viscous stress.
No body force is applied; density is normalized to one and `--viscosity` is
kinematic viscosity. Reynolds number uses mean inlet speed and hydraulic diameter.

## Inclusion and resolution

The stationary sphere defaults to center `(5, 0.5, 0.5)`, radius `0.2`, penalty
`100`, and transition half-width `0.05`. Let `d = |x-center| - radius` and
`s = d/width`. Its compact smooth indicator is

```text
chi = 1                         when d <= -width
chi = (1-s-sin(pi*s)/pi)/2       when |d| < width
chi = 0                         when d >= width
alpha(x) = penalty * chi(x)
```

The momentum equation includes `alpha(x)*u`. The mesh is not cut or retagged,
and a constant penalty is **not** applied throughout domain attribute 1.
`--penalty 0` disables drag but retains the passive indicator and leakage
diagnostics, allowing comparisons against the unobstructed solution.
The entire transition region must lie strictly inside the box.

Default quadrature order is `max(8, 3*order, 2*order+4)`; `--quadrature` overrides it for
assembly and diagnostics. An active inclusion missed entirely by assembly
quadrature is rejected. The global maximum element size is a conservative
resolution indicator, using the largest singular value of the element Jacobian
relative to the ideal reference element (the longest edge for Cartesian hexes).
Warnings flag fewer than four cells across the diameter,
fewer than two across the transition, or element size exceeding the Brinkman
penetration length `sqrt(viscosity/penalty)`.

The original hex mesh has only two cells across the channel. The default one
refinement is a demonstration, **not a resolved obstacle calculation**. Vary
mesh resolution, integration order, transition width, and penalty separately.
Increasing penalty alone may worsen conditioning without improving accuracy.
Weighted-mass/P1-AMG performance is not guaranteed at arbitrary contrast or
Reynolds number. MUMPS is useful for small reference problems, not as a default
for large three-dimensional refinements.

## Build and run

From the workspace root, after loading the local dependency environment:

```bash
source ./set_up_local.sh
cmake -S mfem -B build -DPETSC_CURRENT=NO
cmake --build build --target channel_flow -j 4
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

mpiexec -n 4 build/miniapps/stokes/channel_flow \
  --mesh meshes/channel-3d/channel_hex.mesh --refinements 1 \
  --output results/channel-stokes

mpiexec -n 4 build/miniapps/stokes/channel_flow \
  --mesh meshes/channel-3d/channel_tet.mesh --convection \
  --viscosity 1 --inlet-speed 1 --penalty 100 \
  --output results/channel-navier-stokes
```

Use the site's MPI launcher settings. For local MPICH/Hydra fork-based tests
in this workspace, `HYDRA_LAUNCHER=fork HYDRA_RMK=user` avoids requesting Slurm
placement; it is not a multi-node launch recipe.

`--help` lists all options. Important controls include `--radius`, `--center-x`,
`--center-y`, `--center-z`, `--interface-width`, `--order`, `--backend mumps`,
`--nonlinear-method picard`, `--damping`, and separate linear/nonlinear residual
tolerances and iteration limits. Navier–Stokes initializes from the corresponding
Stokes–Brinkman problem, retaining the inclusion. `--load-steps N` ramps inlet
speed through N equal positive fractions of its requested final value. The first
step uses Stokes initialization; subsequent steps use the previous fields and
notify the problem of the changed inlet coefficient. This continuation is only
available with convection and is not a guarantee of a steady solution.

## Results and failures

Rank zero prints `channel_config`, `channel_mesh`, `channel_step`, `channel_solve`,
and, after convergence, `channel_diagnostics` records. `--summary path.txt` saves
the final report; its parent directory must already exist. Each step reports its
residual; final iteration counts/timings describe the last step except
`total_seconds`, which sums all attempted solver calls. Requested/completed step
counts make incomplete continuation explicit.

`mesh_seconds` covers input validation, refinement, and partitioning.
`channel_resources` reports wall time through diagnostics (excluding visualization)
and Linux peak RSS in MiB: maximum rank RSS and the sum of rank high-water marks.
The latter is a conservative memory estimate, not a simultaneous resident-memory
measurement. It includes the earlier replicated serial mesh.

Diagnostics include outward signed inlet/outlet/wall fluxes, relative mass
imbalance, divergence norm, end-plane area-mean pressures and their difference,
velocity L2 norm, `sqrt(integral(chi*|u|^2)/integral(chi))` inclusion leakage,
Brinkman power `integral(alpha*|u|^2)`, and force on the stationary inclusion
`integral(alpha*u)`. The force on the fluid has the opposite sign. Leakage and
mask volume include the smooth transition, not just the solid core.

ParaView output is under `OUTPUT/channel/`. Open `channel.pvd`. It contains
high-order velocity and pressure plus **cell-averaged** `inclusion_indicator`
and `brinkman_penalty`, computed with assembly quadrature so a sampled inclusion
does not vanish merely because it misses element centers. These visualization
averages are not fed back into assembly. `--no-visualization` skips output.
Use distinct output/summary paths for separate runs; existing results at the
same paths may be overwritten.

Exit status is 0 for success, 1 for solver nonconvergence, and 2 for handled
invalid input or output errors. MFEM fatal mesh/solver errors can abort MPI.
Failure reports are printed/saved, but nonconverged runs do not write new
ParaView fields. A successful iterative flag is insufficient on its own:
the shared solver independently checks the true residual.

## Validation

When the channel meshes are available, CMake registers the
`stokes-channel` CTest label. It exercises both meshes, both physics modes,
penalty-free/penalized flow, one/four ranks, direct/iterative agreement when
MUMPS is enabled, continuation, Picard, outlet pressure shifts, visualization,
refinement/quadrature/order controls, the low-Reynolds-number limit, invalid
inputs, and failure exit/output behavior. These are bounded regression checks,
not an obstacle-boundary convergence study. CMake launches the MPI cases and
the C++ `stokes_driver_checks` helper validates floating-point comparisons and
visualization fields; no interpreter is required. Build `stokes_package` before
running the tests.

```bash
ctest --test-dir build -L '^stokes-channel$' --output-on-failure -j 4
```

## Exact-size serial subdivision

`refine_channel_serial` is a serial utility for uniform Cartesian, domain-1 hex
channels with the same boundary convention. It validates the original grid and
subdivides it by independent integer factors in x/y/z. It constructs the equivalent
refined Cartesian connectivity, preserving original grid lines, geometry, domain
attribute, and boundary tags; element numbering and ordering may change. It is
not a general unstructured mesh refiner. Existing output files are rejected.

The supplied 20 x 2 x 2 grid reaches exactly **64,000 cells** with factors 8 x 10 x 10:

```bash
cmake --build build --target channel_flow refine_channel_serial -j 4
build/miniapps/stokes/refine_channel_serial \
  --mesh meshes/channel-3d/channel_hex.mesh --output channel-64000.mesh \
  --factor-x 8 --factor-y 10 --factor-z 10
mpiexec -n 32 build/miniapps/stokes/channel_flow \
  --mesh channel-64000.mesh --refinements 0 --min-elements-per-rank 1000 \
  --no-visualization --summary channel-64000-stokes.txt
```

This grid has 160 x 20 x 20 cells, with spacings 0.0625 x 0.05 x 0.05, and an
average of 2,000 elements per rank. The driver checks the actual minimum exceeds
1,000. Factors 12 x 15 x 15 and 16 x 20 x 20 give geometrically similar 216,000-
and 512,000-cell grids. Each is a serial subdivision of the original mesh;
the 216k grid is not nested in the 64k grid. Add `--convection` for Navier–Stokes,
holding geometry, penalty/interface width, viscosity, inlet speed, discretization,
quadrature, and solver tolerances fixed in a mesh-refinement study. Full-assembly
memory must be checked before increasing the problem size.
