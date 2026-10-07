# Extensible cut quadrature design

This note defines the contracts implemented by MFEM's backend-neutral cut
quadrature API.  The first backend uses Algoim commit
`da1d81499608e1d499695d255f0233140b8c81e8`.

## Building and running the miniapp

The MFEM Algoim interface uses header-only `algoim/quadrature_general.hpp`.
It requires C++17 and does not require building Algoim or installing Blitz++.
Other enabled MFEM dependencies, such as RAJA or Umpire, may require C++20.

With `mfem/` and `algoim/` as sibling source directories, obtain the supported
Algoim revision from their parent directory:

```sh
git clone https://github.com/algoim/algoim.git algoim
git -C algoim checkout da1d81499608e1d499695d255f0233140b8c81e8
```

For an existing Algoim checkout, only the checkout command is needed. The
`ALGOIM_DIR` setting must name the checkout root containing
`algoim/quadrature_general.hpp`, rather than the nested header directory.

From the same parent directory, configure MFEM with Algoim enabled and build
the miniapp:

```sh
cmake -S mfem -B build -DMFEM_USE_ALGOIM=ON -DALGOIM_DIR="$PWD/algoim"
cmake --build build --target cut-quadrature --parallel 4
```

Include the remaining options and dependency paths needed by your normal MFEM
configuration, or set them in `mfem/config/user.cmake`. The commands also work
with an existing `build/` directory configured for this MFEM source tree.
`MFEM_ENABLE_MINIAPPS=ON` is not needed when building the named target explicitly.
For an MFEM library configured with GNU make, build the tool from
`mfem/miniapps/tools` with `make cut-quadrature`; enable `MFEM_USE_ALGOIM=YES`
and set `ALGOIM_DIR` when configuring the library.

The source is `miniapps/tools/cut-quadrature.cpp`. For the CMake layout above:

```sh
./build/miniapps/tools/cut-quadrature -h
./build/miniapps/tools/cut-quadrature -no-vis
./build/miniapps/tools/cut-quadrature -r 2 -no-vis
./build/miniapps/tools/cut-quadrature -r 5 -no-vis
./build/miniapps/tools/cut-quadrature -r 1 -o 3 -qo 8 -c 0.3 -no-vis
```

The tool uses MFEM's `OptionsParser` and prints the selected settings:

| Option | Purpose | Default |
| --- | --- | --- |
| `-h`, `--help` | Print usage and exit | |
| `-m`, `--mesh` | Read a nonempty planar quadrilateral mesh | Generated two-element unit square |
| `-r`, `--refine` | Number of uniform refinements, at least zero | `0` |
| `-o`, `--order` | Finite-element and coefficient interpolation degree, from 1 through 64 | `2` |
| `-qo`, `--quadrature-order` | Target integration order, from 0 through 19 | `6` |
| `-c`, `--cut-position` | Set the level set to `phi(x,y) = x - c` | `0.45` |
| `-d`, `--device` | Configure MFEM's device; Algoim construction still runs on the host | `cpu` |
| `-vis`, `--visualization` / `-no-vis`, `--no-visualization` | Enable or disable initial level-set visualization in GLVis | Disabled |
| `-p`, `--visualization-port` | GLVis server port on localhost | `19916` |

For visualization, start a GLVis server separately and use
`./build/miniapps/tools/cut-quadrature -vis -p 19916`. GLVis is optional and is
not needed for construction or integration.

The reported measures are totals over all elements, rather than values for
element 0. With the default mesh and cut position, refinement should leave
total volume near `0.45` and total interface length near `1`. Deformation scales
the mesh by `1.2` in x and `0.8` in y while reusing the reference rules, giving
deformed total volume near `0.432` and interface length near `0.8`. Element and
quadrature-point counts change with refinement. Interfaces coincident with
shared element faces are counted per element; the tool does not deduplicate
those faces.

## API decisions

`CutMeasure` is a scoped enum.  MFEM supplies explicit `operator|` and
`operator&` overloads, so combinations remain type safe and expressions such
as `CutMeasure::Volume | CutMeasure::Interface` compile in C++17.  Requests
also carry an explicit `CutExecutionMode`; this implementation supports host
execution and rejects device execution rather than silently falling back.

Level-set extraction and quadrature generation are separate operations.
`ElementLevelSetExtractor::GetElementLevelSet` returns only `Success`,
`UnsupportedSourceBasis`, or `InvalidLevelSet`.  The first failure means that
the source finite-element representation cannot be converted exactly.  In
contrast, `UnsupportedPolynomialBasis` is produced by a constructor that cannot
consume an otherwise well-formed `ElementLevelSet`; extractors never return it.
Batch callers record both a per-element descriptor and extraction status.
`extraction_status` is a closed set containing only those three extractor
outcomes.  Any other value makes the whole call `InvalidBatch`; it is never
passed through as an element result.  Descriptors of successful extractions
must match the declared descriptor, otherwise the call is
`HeterogeneousBatch`. Each coefficient matrix column is one element and each
row one fixed descriptor coefficient. `DenseMatrix` stores columns contiguously,
so all coefficients for an element are adjacent in memory.

Scalar calls check `InvalidRequest` and `UnsupportedExecutionMode`, in that
order, before inspecting element data.  Batch calls check `InvalidRequest`,
`UnsupportedExecutionMode`, `InvalidBatch`, and `HeterogeneousBatch`, in that
order.  A whole-call failure after generation starts is `ExecutionFailure` and
leaves batch outputs unpopulated; tests use deterministic mock fault injection,
not resource exhaustion.  Scalar return values always equal `result.status`.
`ElementLevelSetDescriptor` and `CutQuadratureRequest` have hand-written,
field-wise equality operators because MFEM supports C++17.

## Geometry and failure semantics

The negative and positive volume phases are the strict open sets `phi < 0`
and `phi > 0`.  This convention applies to regular, codimension-one zero sets.
A zero set of positive cell measure violates the regularity precondition and
deliberately returns `Degenerate`/`DegenerateVolume`, even though literal
strict-set evaluation could produce an ordinary classification.  Classification
uses the coefficient-dependent convex-hull bounds of the Bernstein polynomial,
not a fixed absolute tolerance.  A sign-definite bound gives `Full` or `Empty`;
otherwise generation proceeds as a candidate `Cut` cell.

Interface presence is checked independently on the closed element.  In
particular, a boundary-aligned interface may be nonempty when the requested
open volume phase is `Empty` or `Full`, and may coexist with an interior
interface in a `Cut` cell. Identically zero face restrictions are integrated
using reference-face rules for every classification. For `Cut` cells, their
boundary factors are removed from a copy of the Bernstein polynomial before
Algoim generates volume rules and the remaining interface. The removed factors
are `x_d` and `1-x_d`, which are strictly positive inside the reference cell,
so both open volume sign regions are preserved. Removing them from the volume
path also avoids Algoim root searches on identically zero face restrictions
during dimension reduction. Coefficient deflation and degree elevation preserve
the tensor layout and assign each boundary component to
the face rule exactly once. Classification, interface normals, and gradient
degeneracy checks continue to use the original polynomial. A repeated boundary
factor with a vanishing original gradient is still a degenerate interface.
This is an element-local rule only;
ownership or deduplication across neighboring elements is out of scope.

Algoim diagnoses volume degeneracy when the Bernstein coefficient norm is
exactly zero after finite-value validation.  This scale-invariant test avoids
misclassifying a small but valid rescaling of a level set.  For a requested
interface, it diagnoses interface-only degeneracy when sampled zero-set points
and every generated interface point have gradient norm at most
`64 epsilon * order * max(abs(coefficient))`.  The former always pairs
`Degenerate` with `DegenerateVolume`; the latter retains `Cut` and returns
`DegenerateInterface`.  A volume-only request never reports interface
degeneracy.  A combined request is deliberately all-or-nothing: on
`DegenerateInterface`, its volume output is also unusable.  Callers needing
independent volume reliability make a separate volume-only request.

Status alone governs output readability.  A non-`Success` element's rules are
never consumed.  `classification` is diagnostic after classification has run,
but is `Unclassified` for pre-classification failures.  Extractor-owned
`UnsupportedSourceBasis` reaches that state only through batch passthrough;
constructor-owned `UnsupportedPolynomialBasis` can occur in scalar and batch
calls.

Algoim's verified native range is `1 <= qo <= 10`.  With
`qo = ceil((target_order + 1)/2)`, capabilities therefore report MFEM target
orders 0 through 19 and reject other values without clamping.

Polynomial degree is a separate capability: `min_polynomial_degree` and
`max_polynomial_degree` bound `ElementLevelSet::order`, and `Supports()` checks
both polynomial degree and target quadrature order. The Algoim adapter accepts
tensor Bernstein degrees 0 through 64 in each coordinate. This explicit cap
bounds tensor storage, derivative preparation, and evaluation costs; it is not
an accuracy guarantee for arbitrary cuts. Negative degrees are `InvalidLevelSet`;
degrees above 64 are `UnsupportedPolynomialDegree`. Scalar calls check the degree
before coefficient count validation or evaluation. Unsupported target orders
remain `UnsupportedOrder`. A well-formed batch reports unsupported polynomial degrees
per element, preserving failed extraction statuses and empty rule ranges.
The miniapp validates its interpolation degree before constructing a mesh or
finite-element space; it requires at least degree one.

The legacy `AlgoimIntegrationRules` wrapper retains its historical minimum of
one for both target order and level-set projection degree. Its constructor and
setters check the backend capabilities, currently allowing target orders 1--19
and projection degrees 1--64. Unsupported values are rejected immediately with
an `MFEM_VERIFY` range diagnostic instead of failing later during rule generation.
Rejected setter calls leave the existing configuration intact. Values are not
clamped. These limits are specific to the Algoim wrapper; the shared
`CutIntegrationRules` base does not impose them on other backends. Callers needing
status-returning validation should use the backend-neutral cut-quadrature API.

The adapter retains the Bernstein basis throughout evaluation. Values use
tensor de Casteljau evaluation, and derivatives use Bernstein coefficient
differences, all in Algoim's `real` type (double in the supported revision).
There is no conversion to monomials or integer binomial calculation. Exactly
constant coordinate directions are removed from internal tensors without
approximating the polynomial. Evaluation buffers are reused within each
generation call; the adapters do not share mutable buffers between callers.
The interface degeneracy check skips sampling when Bernstein derivative bounds
certify a gradient component stays above its degeneracy tolerance in magnitude.
Otherwise it samples only dependent coordinate directions and stops as soon as
a sampled zero has a nondegenerate gradient. These shortcuts preserve the
original sample-grid diagnosis without approximating the polynomial or reducing
the sampling resolution in dependent directions.

Algoim interval evaluations use centered Taylor enclosures with Hessian bounds
from the convex hull of Bernstein derivative
coefficients on the reference element. Gradients use the same enclosure
construction. This avoids the repeated interval dependency of directly
evaluating a high-degree Bernstein polynomial with interval coordinates.
Intervals extending outside the reference element use interval de Casteljau
evaluation instead, since the reference-element convex-hull bounds no longer
apply there. MFEM coefficient storage and returned rules still use `real_t`.

## Extractors, retention, and concurrency

The `GridFunction` extractor converts supported scalar tensor H1 elements
exactly to Bernstein coefficients.  The `Coefficient` extractor samples the
coefficient at tensor H1 nodes at a caller-selected order and documents this
as an element-local interpolation.  Both expose a caller-controlled revision;
pointer identity and `GridFunction::GetSequence()` are not value revisions.

Retained results are reusable only when extractor `Id()`, element identity (or
ordered batch identity), extractor revision, and exact request equality all
match.  Extractor IDs come from an atomic, never-decremented process counter.
Extractors are non-copyable and non-movable so two live objects never share an
ID and an ID is never transferred ambiguously.

The important failure mode is silent stale reuse: after changing a
`GridFunction` value or a `Coefficient`'s behavior, the application **must bump
the extractor revision**.  If it forgets, all keys still match and stale rules
are reused without an error.  Applications should couple field updates and
revision increments in the same operation.

Constructors, capabilities, and extractors are safe for concurrent calls as
shared const objects, provided the wrapped source is not concurrently mutated
and its read access is itself thread safe.  Each thread must use its own
workspace; workspaces are intentionally not thread safe.  Host batch generation
is serial within one workspace, while callers may parallelize chunks using one
workspace per thread.

## Rules, mapping, and future backends

Packed points and optional normals are column-major matrices (`dim` by
total point count), with one point or normal per column; offsets delimit
elements. Reference rules never change
when a mesh deforms.  Volume consumers multiply each reference weight by
`Tr.Weight()` exactly once.  Surface consumers additionally multiply by
`norm(J^{-T} n_ref)` and obtain the unit physical normal by normalizing that
same vector.  Positive-phase selection may negate the Algoim evaluation
adapter, but stored coefficients and output normals always use the original
gradient, oriented from negative to positive.

The runtime backend interface keeps dependencies and templates out of public
headers and supports inspection, batching, and retained rules.  Compile-time
policies may be useful internally but would expose dependencies; arbitrary
callbacks cannot provide Algoim interval evaluation; an integrator-only API
would prevent reuse and inspection; and an external-only prototype would
duplicate conversion and mapping.  A persistent packed `CutQuadratureSpace`
can build on this API later.

The neutral descriptors reserve a simplex Bernstein basis.  Future simplex
implementations may use direct moment rules, a simplex-specific backend, or
simplex-to-tensor decomposition (with extra mapping, interface, and accuracy
costs).  Moment fitting can also add generated or fixed candidate nodes and
signed or nonnegative policies; infeasible nonnegative constraints must return
`WeightConstraintInfeasible`.  GPU count/scan/fill generation and consumption
belong in MFEM's kernel execution layer and will use the existing execution-mode
field and packed layout.
