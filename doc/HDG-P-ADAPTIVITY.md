# `p`-adaptivity for HDG — the contained route

**The element spaces are already `p`-adaptive and the trace space is the entire
remaining job.** `FiniteElementSpace::SetElementOrder()` on the L2 flux and
potential spaces, `Update()`, and `DarcyHybridization` runs untouched — every
offset it builds is per entity already, and `CanBatchLocalFactor()` was written
to notice when the blocks are *not* all the same size. The space has to sit on
an NC mesh (`Construct()` refuses variable order otherwise, even for L2, which
needs no prolongation at all; `EnsureNCMesh()` is the workaround and is
bit-for-bit a no-op on the answer).

And that buys nothing globally, because the trace order sets the rate. Rates
over `nx` = 4, 8, 16, 32 on `convdiff -p 1 -dg -hb`:

| element / trace | dim M at nx=32 | flux | potential |
|---|---|---|---|
| 2 / 2 | 6336 | → 2 | 3.9 |
| 3 / 2 | 6336 | 1.98 | 3.03 |
| 4 / 2 | 6336 | 2.00 | 2.99 |
| 3 / 3 | 8448 | 2.96 | 4.65 |

`dim M` never moves when only the elements are refined, and raising the element
order above the trace order changes the constant (12x at `nx = 32`) and not the
rate. So the whole of it is a per-face trace order.

**And it is not MFEM's variable-order machinery.** That derives edge/face orders
from element orders and keeps a *variant* per incident order, which is the
`hp`-conformity mechanism; the HDG trace is discontinuous face to face and wants
exactly one order per face. Measured on a `DG_Interface` space:
`SetElementOrder()` does change the dof count (120 → 134 on 4x4),
`GetFaceElement()` then refuses outright ("not implemented"), and
`GetFaceVDofs()` aborts in `FindDofs` because 2D `GetFaceDofs()` looks the edge
variant up by the *base* order's dof count. Teaching `FiniteElementSpace` a
single-variant layout for trace collections is the upstream-quality answer and
is **not** what this plans.

This plans the *contained* route instead: keep the trace space a uniform
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

*And one place outside `DarcyHybridization` reads the trace space directly:*
`DarcyForm::ReconstructTotalFlux()` and `ReconstructFluxAndPot()` use
`fes_tr->GetFaceElement(f)` and `fes_tr->GetFaceVDofs(f, ...)` (six sites in
`darcyform.cpp`). With per-face degrees set those would read `p_max` elements
against a solution occupying only `p_f` slots. `GetTraceOrders()` is public so
the form can consult the hybridization; until it does, **reconstruction and a
per-face trace must not be used together**, and that has to be a guard rather
than a note.

**2. Retire the surplus slots. DONE.**
`Finalize()` unions the unused slots into `ess_tdof_list`, rebuilding it from
the caller's own list each time so a second `Finalize()` after `Reset()` with
different degrees does not inherit the first one's. `ComputeH`'s `DIAG_ONE`
gives them a unit row and `Mult()` zeroes them; no new elimination path.

The trap this step actually turned on was elsewhere. `Init()` runs from
`EnableHybridization()`, so C, E, G and H were already built at the *uniform*
degree before any caller could state a per-face one -- the dof count came out
exactly right and the system was wrong. `SetTraceOrders()` therefore rebuilds
them and calls `Reset()`, and must be called straight after
`EnableHybridization()` and before `Assemble()`.

**3. A knob to drive it. DONE.**
`convdiff -pref n [-prefx x] [-pmax|-pmin]` raises the element order on a
region and derives the face degrees through
`DarcyHybridization::FaceOrdersFromElementOrders()`; `-nc` puts the mesh in
nonconforming mode on its own. Three references, and they discriminate:
stripping `-pref` fails, and so does swapping `-pmax` for `-pmin` -- so at a
genuine `p`-interface the rule is **not** a no-op, unlike the uniform case
where `max` was measured to be exactly redundant. Which of the two is *better*
still needs a convergence study.

`regression_test.py` had to learn the new options, since it rebuilds each
command from a fixed list rather than from the recorded line. Its parameter
reader greps unanchored, so `--p-refine` also matches `--p-refine-x`; the new
options use an anchored reader and the old ones are left exactly as they were.

`-nc` is the flag that finally pins the `DarcyOperator` null-prolongation fix.
Its reference does not discriminate on the *answer* -- an NC mesh with no
hanging nodes is bit-for-bit the conforming one, which is the whole point --
but it is the configuration that used to segfault, so the reference catches a
crash rather than a number.

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

## Driving it

The hybridization stores degrees and derives nothing, so everything below is
the caller's. Written out because "set some orders" hides four separate
decisions and only the first is obvious.

**1. The ceiling, chosen before anything else.** `p_ceiling` is fixed when the
constraint space is built and faces can only go below it, so it is the highest
degree the run will ever reach -- not the degree it starts at. Element degrees
are capped there too: above the ceiling they buy nothing, because the trace
order sets the rate.

**2. Element degrees**, on the L2 flux and potential spaces:
`SetElementOrder(e, p_e)` then `Update()`, on a mesh that has had
`EnsureNCMesh()` called. This part needs no library change and is measured.

**3. Face degrees from element degrees**, which is the step with a real choice
in it:

    p_F = rule(p_K1, p_K2)   interior,   p_F = p_K1   boundary,   capped at p_ceiling

`min` is safe and is what a first driver should use. `max` is the literature's
rule; it needs the face-quadrature fix (now on the trunk) and is measured to
be *exactly redundant* where both neighbours agree, so it can only pay at a
genuine `p`-interface. **Which rule wins there is the open question this whole
branch exists to answer**, and both must be available for it to be answerable.

Worth a helper rather than a miniapp loop, because it is the same three lines
every driver needs and it is testable on its own:

    void FaceOrdersFromElementOrders(const Mesh &, const Array<int> &elem_order,
                                     Array<int> &face_order, Rule rule, int cap);

**4. The indicator, which is the part that does not exist.**
`HDGErrorEstimator` gives an element error, so it says *where*. Nothing in the
tree says *`h` or `p`* -- that needs a smoothness estimate, and the candidates
are the postprocessing gap `‖u*_h − u_h‖_K`, which the tree already computes,
and the decay of the local projection coefficients across degrees, which it
does not. **A first driver should not choose**: mark on the estimator and
raise the degree, `p` only, no `h`. That isolates the machinery from a
question that deserves its own measurement.

**5. Parallel.** Element degrees are rank-local and a shared face needs the
neighbour's, so `min`/`max` both require one exchange of element degrees over
face neighbours. `dim M` per rank and `Dof_TrueDof` are untouched, which is
the point of this route.

**6. The demonstrator.** Every `convdiff` problem is analytic and uniform `p`
already converges exponentially on them, so the case has to be
`anisodiff -p 6` (steady peak) or `-p 5` (boundary layer). A `convdiff -pref`
flag is still worth having first as a *mechanism* test -- prescribed degrees,
no adaptation -- because it puts the machinery under the regression suite
before any indicator exists.

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

## The cost of branching from the trunk, and how to settle it

`gf-hdg-linearise-first` has **four `c_fes` call sites this branch does not** —
48 here against 52 there — two of them in `NPCReduce()` and `NPCRecover()`.
Step 1's substitution cannot convert what is not here, so those four have to be
converted when the two features meet in `meq-integration`.

The optimistic reading is that an unconverted site is *loud*: the per-face and
the uniform trace element would disagree on block size. **That is a guess and
it has not been measured.** Settle it by building this branch, merging it with
`gf-hdg-linearise-first` in a scratch integration, and running the `*_npc.txt`
references: if it aborts, the gap is safe to leave until then; if those
references pass with different numbers, it is not, and the four sites have to
be converted before the merge rather than after.

## What this route does not do

`p_max` is a **ceiling fixed at construction**, so faces can only be coarsened
below the degree the constraint space was built with -- there is no enrichment
past it. A driver that means to raise degrees builds the constraint space at
the highest degree the run will ever reach and starts below it. Row count and
trace-vector length are then `O(p_max)` per face whatever the degrees are,
though the local blocks follow `p_f` because they are sized from `TraceFE()`;
whether the factorization follows the active size as well is expected but
unmeasured. Making the trace space genuinely minimal means the other
route — one variant per entity inside `FiniteElementSpace` — which is not
needed to find out whether any of this is worth having. Upstream has stale
history for it on `origin/hpfem-var-order-space` (Dylan Copeland, 2021), whose
`GetFaceElement` still carries the same `MFEM_VERIFY(!IsVariableOrder())`.
