# Projection ghost penalty (LEGACY assembly)

This integrator adds a componentwise projection-residual penalty on
selected pairs of complete background cells. It supports serial `BilinearForm`
and MPI `ParBilinearForm`, including processor-local and shared faces.

## Build

The integrators are compiled automatically whenever `MFEM_USE_LAPACK=ON`;
no separate ghost-penalty option is needed. With `MFEM_USE_LAPACK=OFF`, their
API, implementation, demos, and tests are disabled. MFEM requires BLAS/LAPACK
to be found when LAPACK is enabled; merely installing LAPACK does not enable
it. PETSc, MUMPS, and SLEPc do not substitute for this LAPACK requirement.
The projection uses MFEM's local `DenseMatrixSVD`, not a distributed SLEPc
eigensolver. Applications still add the integrators to their forms explicitly.

For this workspace, run from the directory containing `mfem` and `build`:

```bash
source ./set_up_local.sh
cmake -S mfem -B build -DMFEM_USE_LAPACK=ON
cmake --build build --parallel 16 --target projection_ghost_penalty_serial \
  projection_ghost_penalty_parallel ghost_penalty_tests pghost_penalty_tests \
  ghost_penalty_failure
```

The existing configuration uses the MPI wrappers and `mpiexec` under
`$PETSC_DIR/bin`, and LAPACK from `$PETSC_DIR/lib/libopenblas.so`.
Source the environment script before both compilation and execution.
The same feature can be built with `MFEM_USE_MPI=OFF`; the serial test and
demo targets do not require MPI. Omit `pghost_penalty_tests` and
`projection_ghost_penalty_parallel` in that build. The latter target exists
only when both LAPACK and MPI are enabled.

GNU Make builds both examples without linking the unrelated mtop solver helpers:

```bash
make -C mfem/miniapps/mtop MFEM_BUILD_DIR="$PWD/build" projection_ghost_penalty_serial
make -C mfem/miniapps/mtop MFEM_BUILD_DIR="$PWD/build" projection_ghost_penalty_parallel
```

The serial example belongs to `SEQ_MINIAPPS`; the MPI example belongs to
`PAR_MINIAPPS`. Both participate in `all` and the corresponding Make tests when
their dependencies are enabled.

GNU Make also supports these tests. The guarded numerical translation unit is
part of the shared unit-test objects and becomes empty without LAPACK. The
failure harness has its own `main()` and is never included in the shared objects;
with `MFEM_USE_LAPACK=YES`, `make -C tests/unit ghost_penalty_failure` builds it
separately. It is also built by `all`, but is not run as an ordinary unit test:
it requires a failure-scenario argument. The configuration regression checks
GNU Make source selection and compilation through LAPACK OFF/ON/OFF transitions.

## Mathematical definition

For the scalar element evaluation matrices `A1`, `A2`, form a block-diagonal
`A`. Evaluate the selected common polynomial basis on both cells, collected
in the evaluation matrix `P`. With positive physical quadrature weights `W`,
construct

```text
Q = sqrt(W) P,          H = sqrt(W) A,
Q = U Sigma V^T,        C = V Sigma^-1 U^T H,
R = H - Q C,            G = (gamma / h_F^2) R^T R.
```

The implementation accumulates individual residual outer products and mirrors
the computed triangle. It never forms `Q^T Q`, a mass-matrix inverse, or the
cancellation-prone difference `D - B^T M^-1 B` in production.

`h_F = max(diam(K1), diam(K2))`, where each first-order affine cell diameter is
the maximum distance between its physical vertices. On the curved/higher-order
path, the diameter is sampled as described below. The polynomial origin is the
average of the two cell centroids; its separate normalization length is
`(volume(K1) + volume(K2))^(1/d)`. These quantities are side-independent.
For fixed coefficients and gamma, scaling the physical mesh by `s` scales
`G` by `s^(d-2)`.

Default quadrature is a positive collapsed-coordinate Gauss rule on each
simplex, or a tensor-product Gauss-Legendre rule on each quad/hex. The target
order on the affine path is `2*max(p_i,r) + quadrature_extra`, interpreted per
coordinate on quads/hexes. Curved P projection includes the geometry degree and
determinant degree, as described below. Simplex rules additionally account for
the collapsed-map Jacobian.
The kernel also accepts independent caller-supplied
volume rules with adequate advertised order; every physical weight is checked
for positivity and finiteness. Supplied rules must have truthful order metadata.

All singular values must pass the relative threshold. An `svd_rtol` of zero
selects `32*max(nq,m)*epsilon(real_t)`; a positive value overrides it. A failure
reports the patch context, FE orders/geometries, dimensions, rank, threshold,
and spectrum. There is no truncation, regularization, or silent degree change.
`gamma=0` produces a zero matrix but still validates geometry and projection.
The explicit floating-point matrix may have tiny negative eigenvalues from
roundoff; the defining residual energy is nonnegative.

## Projection-space selection

`ProjectionGhostPenaltyOptions::projection_space` selects:

- `ProjectionGhostPenaltySpace::AUTO` (default): physical total-degree `P_r`
  on first-order affine simplex pairs, and tensor-product `Q_r` on compatible
  first-order affine quad/hex pairs. New geometry paths require explicit P.
- `ProjectionGhostPenaltySpace::TOTAL_DEGREE`: physical `P_r` on either
  supported cell family. There are `binomial(r+d,d)` basis functions.
- `ProjectionGhostPenaltySpace::TENSOR_PRODUCT`: `Q_r` on quad/hex pairs,
  with `(r+1)^d` basis functions. Explicit Q on simplices is rejected.

The common Q frame is determined by the canonical cell's normalized affine
Jacobian columns, with deterministic signs and ordering. It need not be
orthogonal: common sheared meshes are supported. The second cell's axes must
match up to signs, permutation, and independent scaling. Coordinates in this
frame are centered at the patch origin and divided by `basis_length`.
Incompatible tensor frames cause an error in Q and AUTO modes; there is no
silent fallback. Select P explicitly to handle such affine neighbors.

On Cartesian Q1 cells, Q projection reproduces `xy` in 2D and `xyz` in 3D,
while degree-one P projection generally penalizes these modes. On rotated or
sheared cells, Q reproduction refers to the common tensor coordinates, not
arbitrary monomials in global Cartesian coordinates. Reproduction applies to
functions also representable by both FE spaces. `projection_degree=-1`
continues to select `min(p1,p2)`; explicit nonnegative degrees are permitted.
Diagnostics and verbose output identify the resolved P/Q space. Q projection
has more columns and therefore a higher dense SVD cost at high order.

## Supported spaces and geometry

- Conforming polynomially mapped triangles/quads in 2D and tetrahedra/hexes in 3D.
- Arbitrary supported H1 solution order, independent of geometry order.
  AUTO/Q retain the first-order affine restriction; quad/hex
  centered vertex positions must agree with the centered affine map, to a
  tolerance of `256*epsilon(real_t)*cell diameter`. Tensor frames compare unit
  directions with a dimensionless tolerance of `512*epsilon(real_t)`.
  Absolute coordinate offsets never enlarge either tolerance.
  MFEM's `OrderJ()` bound alone does not identify affine tensor cells.
- Same-type selected patches only; selected triangle-quad interfaces fail.
- Uniform-order continuous `H1_FECollection` spaces for assembled operators.
- Scalar `vdim=1`, or independent vector components with arbitrary `vdim>=2`.
- Both global `Ordering::byNODES` and `Ordering::byVDIM` layouts.
- Unequal cell shapes and sizes; standalone kernels also accept unequal FE
  orders and DOF counts. The default common degree is `min(p1,p2)`.

Only functions in the common polynomial space are reproduced. In particular,
selecting a projection degree below the H1 degree also penalizes higher-degree
global polynomials. This is algebraic support, not an hp-optimality claim.

NC/hanging faces, variable-order assembled spaces, embedded geometry, curved Q,
NURBS/rational or non-polynomial maps, H(div), H(curl), DG, and elasticity terms
are not supported. Degenerate cells and ambiguous multiple-face element-pair
keys are rejected. Periodic identifications are not an advertised geometry
policy. FULL extension assembly, ELEMENT, PA, matrix-free and GPU assembly,
static condensation and hybridization are outside this implementation.

## Curved polynomial geometry

Use `ProjectionGhostPenaltySpace::TOTAL_DEGREE` (`--projection-space p`) for
non-affine bilinear/trilinear cells and high-order polynomial geometry. AUTO and
Q reject these maps without fallback, including on curved simplices and on
high-order representations of geometrically straight cells. The projection is
onto one **physical** total-degree space over the complete two-cell patch; it is
not onto independent reference-element spaces or a shared nonlinear Q chart.

Geometry uses polynomial `IsoparametricTransformation` maps with Pk simplex or
Qk tensor geometry FEs. Assembled nodal geometry must be uniform-order continuous
H1. The standalone kernel can use different geometry orders when the physical
shared faces agree. Mesh geometry must be injective and orientation-preserving.
At the center, sampled geometry points, and integration points the implementation
checks finite coordinates/Jacobians and
`det(J/||J||_F) > 64*dimension*epsilon(real_t)`. This MFEM version's `Weight()`
is already signed for square Jacobians; the explicit normalized checks also
detect poor Jacobian quality and validate points outside the integration rule.
**Finite sampling is not a certificate** of
positive Jacobians everywhere or global injectivity; mesh validity remains the
caller's responsibility. Extremely ill-conditioned sampled cells are rejected.

For a map of polynomial degree `k`, a physical P_r polynomial pulls back to
degree at most `k*r` (total degree on simplices, per-coordinate on tensors).
Therefore, `p_i >= k_i*r` on both sides is sufficient for polynomial reproduction,
but is not required to assemble the operator. The nullspace contains only common
physical polynomials also representable in both FE spaces. In particular,
interpolating a physical polynomial into an insufficient-order FE space need not
produce zero penalty. The default projection degree remains `min(p1,p2)`; there
is no silent degree reduction. Constants are always reproduced.

The curved path requires quadrature order
`2*max(p_i,k_i*r) + w_i`, where `w_i` is the polynomial determinant bound supplied
by the supported mapping family (`OrderW()`): `d*(k_i-1)` on simplices and
`d*k_i-1` per coordinate on tensors. This bounds FE-FE, FE-polynomial, and
polynomial-polynomial products including the Jacobian determinant. Generated
rules add `quadrature_extra`; supplied rules must meet the minimum without it
and have finite positive reference and physical weights. Large order/allocation
arithmetic is checked; no underintegration or SVD rank truncation is substituted.
Increasing geometry order can substantially increase integration and dense SVD
cost. The optimized first-order affine integration path is retained.

Cell volumes and physical centroids use a separate positive geometry-only rule
of order `k + w`, independent of the solution/projection order and quadrature
extra. Centroid moments are accumulated relative to a nearby anchor. Curved-cell
diameters use the maximum distance between mapped points of a closed-uniform
reference lattice with `2*k` subdivisions, including boundary and interior
points. This sampled size assumes shape-regular cells, is not a certified bound
on the true diameter, and is independent of penalty quadrature settings. It
preserves rotation/translation invariance and the `s^(d-2)` matrix scaling law.

Geometry inspection evaluates a cell-centered copy of the coordinate field.
Assembly evaluates both cell maps relative to a shared patch offset before
forming Jacobians, integration weights, and polynomial coordinates. The input
mesh is not changed. This avoids cancellation from large absolute positions;
it cannot recover geometric detail already lost when input coordinates were
rounded to the working precision. Reported centroids remain physical coordinates.

Diagnostics expose `geometry_orders`, `required_orders`, `quadrature_orders`,
`volumes`, `centroids`, and `minimum_jacobians`, all in the caller's first/second
cell order. MPI preparation exchanges geometry nodes before accessing neighbor
transformations. Rebuild selections and forms after geometry updates, including
raw node edits not detected by sequence counters.

## API and ownership

The public API is in `fem/ghost_penalty.hpp`, included by `mfem.hpp` when the
feature is enabled. All declarations are in namespace `mfem`.

```cpp
ProjectionGhostPenaltyOptions options;
options.gamma = 1.0;
options.projection_degree = -1;
options.quadrature_extra = 2;
options.svd_rtol = 0.0;
options.projection_space = ProjectionGhostPenaltySpace::AUTO;

auto selection = std::make_shared<GhostFaceSelection>(fes, active, cut, 0);
BilinearForm form(&fes);
form.SetAssemblyLevel(AssemblyLevel::LEGACY);
form.AddInteriorFaceIntegrator(
   new ScalarProjectionGhostPenaltyIntegrator(selection, options));
form.Assemble();
form.Finalize();
```

Use `VectorProjectionGhostPenaltyIntegrator` for `vdim>=2`. For MPI, `fes`
is a `ParFiniteElementSpace` and the form is a `ParBilinearForm`. After
`Assemble()` and `Finalize()`, use an owned `HypreParMatrix` returned by
`ParallelAssemble()` to operate on true DOFs.

Call `Assemble()` through the concrete `ParBilinearForm` type in MPI mode.
In this MFEM version it is not virtual: calling through a `BilinearForm&`
or `BilinearForm*` skips the parallel neighbor exchange and shared-face loop.
Generic assembly helpers must preserve the concrete type (for example with
a function template) or explicitly dispatch to `ParBilinearForm`.

`GhostFaceSelection(FiniteElementSpace&, active, cut, extension_layers=0)`
copies the selected-face lookup, not the flag arrays. Each flag array must
contain exactly one zero/one entry per LOCAL element. The FE space and mesh
must outlive the selection and forms. Integrators share immutable selection
ownership; the form owns the integrator itself.

The selector verifies mesh/space sequence numbers before every face lookup.
Rebuild the selector and forms after mesh refinement, repartitioning, changes
to geometry or flags, or FE-space updates. Raw coordinate/flag edits that do
not advance MFEM sequence counters remain the caller's responsibility.

For standalone testing or other local use:

```cpp
ProjectionGhostPenaltyDiagnostics diagnostics;
DenseMatrix patch;
ProjectionGhostPenaltyKernel::Assemble(
   fe1, fe2, transformation1, transformation2, options, patch, &diagnostics);
```

Diagnostics contain singular values, numerical rank, relative threshold,
condition number, quadrature count, `face_length` and `basis_length`.
`ComputeProjection(Q,H,rtol,C,diagnostics,context)` also exposes the rectangular
weighted least-squares operation without assembling a patch matrix.
SVD data and quadrature workspaces are local to each call. Transformations are
mutable MFEM objects; concurrent calls must not share transformation objects.

## MPI selection and assembly

Construct the selection collectively on the FE-space communicator. By default
a face is selected when both adjacent elements are active and at least one is
cut. Each additional `extension_layers` step grows the cut band by one active
cell adjacency before face selection. This supports explicit connectivity
bands without pretending the default establishes cut-independent conditioning.

Two scalar L2/P0 fields exchange active and current-band markers through
`ParGridFunction::ExchangeFaceNbrData()`. A remote element is accessed using
the shifted `Elem2No` in the P0 grid function. No metadata MPI communication
occurs in `AssembleFaceMatrix()`.

The kernel lexicographically orders each cell's sorted physical vertex list
and uses those lists to canonicalize the two cells. This is independent of
MPI rank IDs and local/neighbor labeling. FE columns, quadrature rows, and
the final returned matrix are permuted consistently. The two ranks evaluate
the same physical patch, then return MFEM's required local-first ordering.

Keep `ParBilinearForm::KeepNbrBlock(false)`, its default. MFEM inserts the local
element rows with both local and remote columns; both ranks must process the
face. Owner-only face processing loses contributions. `ParallelAssemble()`
then maps these contributions to true DOFs. Shared faces count as two local
visits but only one physical face; the demo sums local counts plus half the
shared-face visit counts.

Use identical projection options on every rank. The selector checks uniform
order and extension-layer agreement. Errors use MFEM's configured error
handler; its default MPI behavior aborts rather than leaving peer ranks
blocked. A supplied MFEM exception policy requires collective handling by
the application.

## Reading the examples

Start with `miniapps/mtop/projection_ghost_penalty_serial.cpp`, then compare it with
`miniapps/mtop/projection_ghost_penalty_parallel.cpp`. Each is self-contained:
there is no generic serial/parallel driver or shared example framework. Both
`main()` functions follow the same eight numbered stages:

1. Read options and create the complete background mesh. The parallel example
   initializes MPI/Hypre and partitions identical geometry on every rank.
2. Construct the H1 collection and a concrete `FiniteElementSpace` or
   `ParFiniteElementSpace`.
3. Supply active/cut flags for the locally owned elements. The helper below
   `main()` implements only a sampled toy level-set classifier; applications
   should substitute their own geometry classification.
4. Set `ProjectionGhostPenaltyOptions` explicitly, independently of FE order.
5. Construct a shared `GhostFaceSelection`, with optional extension layers.
   Parallel construction is collective, including inactive/empty ranks, and
   prepares neighbor flags and geometry before integrator callbacks.
6. Register `ScalarProjectionGhostPenaltyIntegrator` for `vdim=1`, or
   `VectorProjectionGhostPenaltyIntegrator` for independent component penalties.
   The concrete form uses `AssemblyLevel::LEGACY`.
7. Assemble and finalize. The parallel example calls `ParBilinearForm::Assemble()`
   directly, retains `KeepNbrBlock(false)`, and uses `ParallelAssemble()` to obtain
   an owned `HypreParMatrix` on true DOFs.
8. Interpolate a smooth field and measure its penalty energy. The parallel
   example extracts true DOFs and reduces energies and physical face counts.

The declaration order makes ownership visible: mesh and FE collection outlive
the FE space, which outlives the selection and form. The form owns its integrator;
the integrator shares ownership of the selection, which borrows the space.
Rebuild the selection and form after mesh, geometry, FE-space, or flag changes.
The parallel true-DOF matrix is held by `std::unique_ptr`.

`ProjectionGhostPenaltyKernel` and `ProjectionGhostPenaltyDiagnostics` are the
lower-level per-patch APIs described above. Normal assembled applications use
the selection and integrators; they do not need to assemble patches manually
or repeat kernel calls to obtain the global operator. Use `--verbose` for the
integrators' patch diagnostics.

These are stabilization examples, not complete CutFEM solvers. Physical-domain
bilinear/linear forms, boundary conditions, and inactive-DOF treatment must be
added by the application; solving the ghost operator alone would leave its
polynomial nullspace untreated.

## Demos and tests

`projection_ghost_penalty_serial` is always serial.
`projection_ghost_penalty_parallel` always initializes MPI, even on one rank.
The old `--serial` and `--parallel` switches have been removed: migrate by
choosing the executable, not a mode flag. Other numerical options and defaults
are unchanged.

```bash
source ./set_up_local.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
build/miniapps/mtop/projection_ghost_penalty_serial --order 3
mpiexec -n 4 build/miniapps/mtop/projection_ghost_penalty_parallel \
  --dimension 3 --order 2 --vdim 3 --ordering byVDIM --verbose
mpiexec -n 4 build/miniapps/mtop/projection_ghost_penalty_parallel \
  --element-type tensor --dimension 3 --projection-space auto \
  --order 2 --vdim 3
build/miniapps/mtop/projection_ghost_penalty_serial --element-type tensor \
  --dimension 2 --projection-space p --order 2
ctest --test-dir build -L ghost-penalty --output-on-failure
```

For local-only runs on this host, PETSc's Hydra launcher may need
`mpiexec -launcher fork -rmk user -n 4 ...`, or the equivalent
`HYDRA_LAUNCHER=fork HYDRA_RMK=user` environment settings. Use scheduler/site
settings instead for multi-node runs. MPI initialization needs local sockets;
it cannot run inside a sandbox that prohibits them.

Both demos accept `--mesh`, `--order`, `--dimension`, `--vdim`, `--ordering`,
`--gamma`, `--projection-degree`, `--svd-rtol`, `--quad-extra`, `--layers`,
`--test-level-set`, `--element-type simplex|tensor`, `--projection-space auto|p|q`,
`--geometry-order`, `--warp`, and `--verbose`/`--quiet`.
Geometry order defaults to 1 and warp to 0; neither changes the solution order.
These controls apply to generated meshes before MPI partitioning. Loaded meshes
retain their geometry and reject nondefault generation controls. Warp is a
polynomial deformation with amplitude in `[-0.1,0.1]`, adding `warp*x*y` to the
first two coordinates and `warp*y*z` in 3D. Order >= 2 also adds
`warp*y*(1-y)`, `0.3*warp*x*x`, and (in 3D) `0.2*warp*x*x` to the respective
coordinates; order >= 3 adds `0.1*warp*y^3` to the first coordinate.
Element type defaults to
simplex and controls generated meshes only; `--mesh` supplies its own cell types.
The deterministic test level set defines `x[0] < threshold`, with cut flags
from vertex extrema on unwarped generated affine cells. Otherwise the demo uses
the deterministic `2*k` geometry lattice: these flags are approximate, not
certified intersection tests. The demo reports geometry order, selected-face count,
true DOFs and an interpolated smooth field's penalty energy, not a PDE solve.
The true-DOF count includes the whole background FE space, not just active
elements. The parallel example uses deterministic element-index partitioning
for reproducibility, rather than an optimized load-balancing strategy.

Curved and non-affine examples (run from the workspace root):

```bash
source ./set_up_local.sh
build/miniapps/mtop/projection_ghost_penalty_serial --element-type tensor \
  --dimension 2 --geometry-order 1 --warp 0.05 --projection-space p
mpiexec -launcher fork -rmk user -n 4 \
  build/miniapps/mtop/projection_ghost_penalty_parallel --element-type tensor \
  --dimension 3 --geometry-order 2 --warp 0.05 --order 2 \
  --projection-space p --projection-degree 1 --vdim 3
```

The targeted tests cover mixed-order kernels, P1 through P5 common degrees
in 2D, 3D kernels, polynomial nullspaces, symmetry, eigenvalue roundoff,
scaling, independent quadrature/energy, analytical projection coefficients,
both vector layouts, MPI empty/remote-only cut bands and connectivity
extension, entirely inactive ranks, and precomputed sparsity. Distributed
tests compare every matrix entry against a serial
reference using coordinate/component DOF identities, and check transpose
actions, energies, constant nullspaces and nonzero off-process couplings.
Failure subprocesses require both an unsuccessful exit and the expected
diagnostic; MPI failure testing includes an error on only one rank.

Configuration tests also check MPI/LAPACK target gating.

This is a stabilization contribution only. It uses entire background cells,
not physical-domain or cut-cell quadrature. A full CutFEM solver must supply
diffusion, boundary terms, and elimination/constraints for inactive DOFs.
The ghost operator alone is semidefinite and is not an SPD PDE system.

## Make and coordinate-offset regressions (2026-11-02)

- The standalone failure harness is excluded from GNU Make's shared objects and
  linked separately when LAPACK is enabled. The shared aggregate depends on the
  Makefile so source-list changes invalidate stale objects containing a `main()`.
- Both ghost test translation units are guarded by `MFEM_USE_LAPACK`. GNU Make
  source-selection/compilation checks pass through OFF/ON/OFF configuration
  transitions; Clang syntax checks against the LAPACK-disabled headers also pass.
- A GNU Make unit runner containing the ghost numerical tests and the separate
  failure harness both link successfully. The runner passes the offset tests;
  the harness reports the expected rank-deficiency failure.
- New 2D/3D tests cover translated affine, non-affine, higher-order, and
  incompatible-frame cases. Offsets of `1e12` (double precision) do not bypass
  AUTO/Q rejection; valid P/Q cases retain matrix/metric and overintegration
  agreement within roundoff tolerances. Six failure cases and 13,024 numerical
  assertions specifically cover large offsets.
- **88/88 MPI-enabled Release** and **63/63 serial Debug** ghost-penalty CTest
  cases pass. The serial numerical suite checks 4,463,567 assertions in 13 cases.
- Logs: `build/ghost-review-mpi-tests.log`, `build/ghost-review-debug-tests.log`,
  `build/ghost-review-make-configuration.log`, `build/ghost-review-make-link.log`,
  `build/ghost-review-make-relink.log`, `build/ghost-review-clang-off.log`, and
  `build/ghost-review-offsets.log`. This is not a full unrelated MFEM-suite run.

## Initial curved-geometry validation (2026-11-02)

- MPI-enabled Release: **82/82 ghost-penalty CTest cases passed**, including
  curved true-DOF matrix comparisons and MTOP demos on 1/2/4 ranks.
- MPI-disabled Debug: **57/57 cases passed**; the serial numerical suite checks
  **4,450,543 assertions in 12 cases**. Five dedicated curved local cases check
  367,850 assertions, in addition to curved extensions of existing reference tests.
- Coverage includes order-1 non-affine tensor cells, order-2/order-3 geometry,
  physical P1/P2 reproduction with sufficient solution order, nonrepresentable
  interpolants, integrated metrics, hidden vertex-invisible curvature, sampled
  size sensitivity, mixed local geometry orders, reference permutations, scaling,
  PSD, independent integration, vector layouts, and inactive/empty MPI selections.
- Failure cases check curved AUTO/Q rejection, insufficient rules, bad weights,
  sampled inverted/singular/folded geometry, nonfinite data, integer overflow,
  rational/embedded geometry, and validation with zero penalty strength.
- Filtered Doxygen has no warnings; `git diff --check` passes. The pinned
  Artistic Style 3.1 executable is unavailable, so its formatter check was not run.
- Logs: `build/ghost-curved-mpi-tests.log`, `build/ghost-curved-debug-tests.log`,
  `build/ghost-curved-local-debug.log`, and `build/ghost-curved-doc.err` (empty).
  The unrelated complete MFEM suite was not run.

## Original affine quad/hex validation (2026-11-02)

- MPI-enabled Release: **48/48 ghost-penalty CTest cases passed**, including
  quad/hex true-DOF matrix comparisons on 1/2/4 ranks and tensor miniapps.
- MPI-disabled Debug: **35/35 cases passed**, including MFEM dimension checks.
- The serial numerical suite passes **3083061 assertions in seven cases**.
  Coverage includes rotated/sheared and unequal-size cells, reference-axis
  permutations, P/Q nullspaces, symmetry/PSD, independent mass/energy references,
  high-order Q5/Q6 quad and Q3/Q4 hex kernels, and explicit P on incompatible
  tensor frames. The high-order Q projection ranks are 36 and 64, respectively.
- Negative cases cover non-affine quad/hex maps, incompatible Q/AUTO frames,
  Q on simplices, selected mixed-type pairs, degeneracy, invalid mode, and
  insufficient/invalid quadrature. LAPACK OFF/ON/OFF configuration checks pass.
- Targeted Doxygen generation using MFEM's actual LaTeX input filter is clean.

Logs: `build/ghost-tensor-mpi-tests.log`,
`build/ghost-tensor-debug-tests.log`, and `build/ghost-tensor-doc.err`.
The complete pre-existing MFEM test suite was not run for this extension.

## Original simplex validation (2026-11-02)

Toolchain: MFEM 4.10.1, GCC 13.3.1, C++17, double precision, MPICH 4.3.0,
HYPRE 2.31.0 and OpenBLAS 0.3.28 for BLAS/LAPACK. The main configuration
retains PETSc 3.22.5, SLEPc 3.22.2, MUMPS 5.7.3 and METIS support.

- MPI-enabled Release build: **25/25 CTest cases passed**, including all
  1/2/4-rank comparisons and expected failures.
- The configuration regression toggles LAPACK OFF/ON/OFF in the same build
  directory and checks automatic enablement and removal of ghost-penalty tests
  and demos, without a separate feature option.
- MPI-disabled Debug build: **18/18 CTest cases passed**. Configure it with:

```bash
source ./set_up_local.sh
cmake -S mfem -B build/ghost-serial-debug \
  -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER="$(command -v g++)" \
  -DMFEM_USE_MPI=OFF -DMFEM_USE_METIS=OFF \
  -DMFEM_USE_LAPACK=ON \
  -DBLAS_LIBRARIES="$PETSC_DIR/lib/libopenblas.so" \
  -DLAPACK_LIBRARIES="$PETSC_DIR/lib/libopenblas.so"
cmake --build build/ghost-serial-debug --parallel 16 \
  --target ghost_penalty_tests ghost_penalty_failure projection_ghost_penalty_serial
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ctest --test-dir build/ghost-serial-debug -L ghost-penalty --output-on-failure
```

The original serial suite executed 103276 assertions in five cases. Representative
full-rank diagnostics: 2D P5/P6 gives rank 21 and condition number about 5238;
3D P3/P4 gives rank 20 and condition number about 297.

The P2, three-component demo agrees across serial and 1/2/4 ranks with both
orderings: five selected faces and energy about `2.23489157344e-5` in 2D;
eight faces and energy about `1.09900218350e-6` in 3D. Comparison is to
floating-point tolerance, not bitwise identity.

Logs relative to the workspace root: `build/ghost-final-tests.log`,
`build/ghost-serial-debug-tests.log`, and `build/ghost-demo-comparison.log`.
The complete pre-existing MFEM test suite was not run.

Optional Valgrind validation was attempted but could not run: the installed
native OpenBLAS `dgemm` contains an AVX-512 instruction that Valgrind 3.27.1
does not recognize. It terminates before `main`; this is not a successful
memory/leak check. The diagnostic is in `build/ghost-valgrind.log`.
