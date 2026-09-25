# Extensible cut quadrature design

This note defines the contracts implemented by MFEM's backend-neutral cut
quadrature API.  The first backend uses Algoim commit
`da1d81499608e1d499695d255f0233140b8c81e8`.

## API decisions

`CutMeasure` is a scoped enum.  MFEM supplies explicit `operator|` and
`operator&` overloads, so combinations remain type safe and expressions such
as `CutMeasure::Volume | CutMeasure::Interface` compile in C++17.  Requests
also carry an explicit `CutExecutionMode`; this implementation supports host
execution and rejects device execution rather than silently falling back.

Level-set extraction and quadrature generation are separate operations.
`ElementLevelSetProvider::GetElementLevelSet` returns only `Success`,
`UnsupportedSourceBasis`, or `InvalidLevelSet`.  The first failure means that
the source finite-element representation cannot be converted exactly.  In
contrast, `UnsupportedPolynomialBasis` is produced by a generator that cannot
consume an otherwise well-formed `ElementLevelSet`; providers never return it.
Batch callers record both a per-element descriptor and extraction status.
`extraction_status` is a closed set containing only those three provider
outcomes.  Any other value makes the whole call `InvalidBatch`; it is never
passed through as an element result.  Descriptors of successful extractions
must match the declared descriptor, otherwise the call is
`HeterogeneousBatch`.  Coefficients use a structure-of-arrays layout: each
matrix column is one element and each row one fixed descriptor coefficient.

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
open volume phase is `Empty` or `Full`.  This is an element-local rule only;
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
but is `Unclassified` for pre-classification failures.  Provider-owned
`UnsupportedSourceBasis` reaches that state only through batch passthrough;
generator-owned `UnsupportedPolynomialBasis` can occur in scalar and batch
calls.

Algoim's verified native range is `1 <= qo <= 10`.  With
`qo = ceil((target_order + 1)/2)`, capabilities therefore report MFEM target
orders 0 through 19 and reject other values without clamping.

## Providers, retention, and concurrency

The `GridFunction` provider converts supported scalar tensor H1 elements
exactly to Bernstein coefficients.  The `Coefficient` provider samples the
coefficient at tensor H1 nodes at a caller-selected order and documents this
as an element-local interpolation.  Both expose a caller-controlled revision;
pointer identity and `GridFunction::GetSequence()` are not value revisions.

Retained results are reusable only when provider `Id()`, element identity (or
ordered batch identity), provider revision, and exact request equality all
match.  Provider IDs come from an atomic, never-decremented process counter.
Providers are non-copyable and non-movable so two live objects never share an
ID and an ID is never transferred ambiguously.

The important failure mode is silent stale reuse: after changing a
`GridFunction` value or a `Coefficient`'s behavior, the application **must bump
the provider revision**.  If it forgets, all keys still match and stale rules
are reused without an error.  Applications should couple field updates and
revision increments in the same operation.

Generators, capabilities, and providers are safe for concurrent calls as
shared const objects, provided the wrapped source is not concurrently mutated
and its read access is itself thread safe.  Each thread must use its own
workspace; workspaces are intentionally not thread safe.  Host batch generation
is serial within one workspace, while callers may parallelize chunks using one
workspace per thread.

## Rules, mapping, and future backends

Packed points and optional normals are structure-of-arrays matrices (`dim` by
total point count); offsets delimit elements.  Reference rules never change
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
