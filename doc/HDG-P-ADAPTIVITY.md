# `p`-adaptivity for HDG — the contained route

Roadmap §7 carries the measurements and why this is the shape it is. In one
line: **the element spaces are already `p`-adaptive and the trace space is the
entire remaining job**, because the trace order sets the convergence rate and
`dim M` — the only globally coupled unknowns — does not move when the elements
alone are refined.

This plans the *contained* route of §7's two: keep the trace space a uniform
`FiniteElementSpace` at `p_max`, and use each face's dof slots only up to that
face's degree `p_f`. Nothing in `fem/fespace.cpp` changes. The low-order face is
not a subspace of the high-order one — there is no hierarchical basis in MFEM to
make it one — but it does not need to be: it is a *different basis in the same
storage*, and the HDG trace is discontinuous face to face, so nothing outside
that face ever looks at it.

## Already done

* Element orders through `DarcyHybridization`: works untouched, measured.
* The HDG face quadrature takes the trace element's order into account, so a
  face may legally be richer than its elements (commit "The HDG face quadrature
  never saw the trace element").
* `DarcyOperator` survives the NC mesh that variable order requires (commit
  "DarcyOperator dereferenced a null prolongation…").

## Steps

**1. A per-face order map, behind two accessors.**
`DarcyHybridization::SetTraceOrders(const Array<int> &face_order)`, stored and
not derived — the *caller* applies whatever rule it wants. Two private helpers,

```
const FiniteElement *TraceFE(int f) const;
void TraceVDofs(int f, Array<int> &vdofs) const;
```

falling through to `c_fes.GetFaceElement(f)` / `c_fes.GetFaceVDofs(f, …)` when
no map is set, so every existing path is byte-identical. Then substitute at the
call sites: 37 of `c_fes.GetFaceElement(...)` and 15 of `GetFaceVDofs`, every
one of them already a single face index, so the substitution is mechanical.
`fec->GetFE(geom, p_f)` is the whole of `TraceFE` and it caches per order.

*This adds a data member, so it is the class-layout trap: `make clean` in both
trees, not a rebuild. Budget for it; a parameter cannot carry a persistent map.*

**2. Retire the surplus slots.**
A face with `p_f < p_max` leaves `nt(p_max) - nt(p_f)` slots unused. Union them
into the essential list through the existing `SetEssentialVDofs()`; `ComputeH`'s
`DIAG_ONE` already gives such dofs a unit row and `Mult()` already zeroes them.
No new elimination path, and the *effective* system is the `p`-adaptive one.

**3. A knob to drive it.**
`convdiff -pref` setting element orders from a rule, plus `-nc`, plus a face
rule of `min` or `max` over the two neighbours. `-nc` is also the flag that
pins the `DarcyOperator` NC fix, which currently has none because reaching it
needs a flag the miniapp does not have.

**4. Parallel.**
`dim M` per rank is unchanged and `Dof_TrueDof` is untouched — that is the point
of this route. The one new requirement is that the two ranks either side of a
shared face agree on `p_f`, which needs the neighbour's element order: one
exchange, or a rule computable from data both sides already hold.

**5. The demonstrator, and the indicator.**
Every `convdiff` problem is analytic and uniform `p` already converges
exponentially on them, so the case has to be `anisodiff -p 6` (steady peak) or
`-p 5` (boundary layer). `HDGErrorEstimator` says *where*; nothing says *`h` or
`p`*. Cheapest candidate for the smoothness half is the postprocessing gap
`‖u*_h − u_h‖_K`, which the tree already computes — to be measured against a
coefficient-decay indicator rather than assumed.

## Acceptance

1. **Null test**: every `p_f` equal to the uniform order reproduces every
   existing answer bit-for-bit. If this is not exact, stop.
2. A mesh carrying two element orders converges at the rate its trace orders
   set, and reaches a given error at fewer global dofs than uniform `p_max`.
3. **`min` against `max` at a genuine `p`-interface.** The investigation showed
   a trace richer than *both* neighbours is exactly redundant; whether it earns
   its dofs where the neighbours differ is open, and this is the first thing
   that can answer it. Measure it; do not pick a rule by argument.
4. Rank-count independence on `pconvdiff` at 1, 2, 3, 4 ranks.

## What this route does not do

Storage stays `O(p_max)` per face, so a mesh with a few high-order faces pays
for them everywhere. Making the trace space genuinely minimal is §7's other
route — one variant per entity inside `FiniteElementSpace` — which is the
upstream-quality answer and is not needed to find out whether any of this is
worth having.
