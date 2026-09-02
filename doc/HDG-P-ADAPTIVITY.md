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

*And on reconstruction, of which there are two kinds and only one is a
problem.* `DarcyForm::Reconstruct()` solves a mixed local problem driven by a
total flux built from the traces, and reads the trace space directly at six
sites in `darcyform.cpp`; with per-face degrees it would read `p_max` elements
against a solution occupying `p_f` slots, so `-pref` refuses `-rec`.

`HDGPotentialPostprocessor` -- the classic local postprocessing, Nguyen,
Peraire & Cockburn eq (25) -- has no such problem: it reads the flux and the
potential on the element it is working on and nothing else, never the trace
space and never a neighbour, so what degree the faces carry cannot reach it.
Its `Compute()` already took `GetFE(z)` per element from all three spaces; the
only thing that was uniform was the enriched space it builds by default, which
now follows the potential element by element. That is `convdiff -pp`, it works
under `-pref`, and it is what a `p`-adaptive run should use.

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

**5. The demonstrator, and the indicator. DONE.**
`anisodiff -p 5 -ks 1e2 -hb -dg -dorf -hp`, with `HDGErrorEstimator` saying
where and `PerssonPeraireSmoothness` saying `h` or `p`. The table is in the
miniapp's header comment; the short of it is 48x fewer globally coupled
unknowns than uniform refinement and 2.3x fewer than `h`-adaptivity at a
relative error of 1e-4, and three further decades where neither of the others
can be run. What it cost was three defects in the estimate, all recorded on the
methods that now carry the fixes.

## Driving it — now in the code, not here

All six pieces this section used to specify exist, so what it said belongs
where the change happened rather than in a plan:

| what a driver has to know | where it says so |
|---|---|
| the ceiling is fixed at construction and faces only go below it | `SetTraceOrders()` doxygen |
| call it straight after `EnableHybridization()` and before `Assemble()` | same |
| a hanging-node family runs at the ceiling, and why | same |
| the `min`/`max` rule, and that `max` is redundant where neighbours agree | `TraceOrderRule` doxygen |
| how the face rule handles a nonconforming mesh | `FaceOrdersFromElementOrders()` doxygen |
| element degrees before the mesh, and two `Update()`s not one | `anisodiff.cpp`, the refine block |
| what the estimator has to be told under a per-face degree | `SetHybridization()` doxygen |
| the `h`-or-`p` rule, and the sensor's threshold | `anisodiff.cpp` header, `PerssonPeraireSmoothness` |

The helper signature, since this section had it wrong:

    static void FaceOrdersFromElementOrders(const Mesh &mesh,
                                            const Array<int> &elem_order,
                                            TraceOrderRule rule, int cap,
                                            Array<int> &face_order);

## What is left

The demonstrator exists and steps 1 to 5 are done. What follows is what it
turned up and did not settle, most-open first.

### The one open question, now characterised

**The anisotropic split cannot be used with the postprocessed estimate**, and
four measurements say what that is and is not. The write-up is on
`HDGErrorEstimator::SetAnisotropic()`; the short of it:

* **Not the degree gap.** `--postprocessed-projected-down` projects the
  postprocessed potential back onto the potential's own degree, removing the
  gap against the trace entirely. Every one of the six flags stays at `x` and
  eta moves by 0.5%. `TraceComparison::Projected`'s motivating theory is now
  dead twice.
* **Not the anisotropy.** The loop stalls the same way at `-ks 1` and on
  problem 6, a radially localised peak, both isotropic.
* **Not the marking.** Under isotropic refinement the two estimates select the
  *same elements* for two cycles, agreeing to every printed digit.
* **It is the directional energy, by up to seven orders.** Summed over the
  mesh, `d_0`/`d_1` is 1.94 / 4.2e-4 / 2.4e-5 on the computed potential at
  `-ks` 1 / 100 / 10000 and 3.57 / 1.6e5 / 1.9e7 on the postprocessed one. At
  `-ks 100` the postprocessed total is nothing but that one component: eta² is
  11.09 and `d_0` alone is 11.07.

What is left is a decision rather than an investigation. `p̂ - λ` on the
computed potential is the scheme's own stabilization term; on a superconverged
potential it is essentially λ's own error -- a real quantity, but not the
element's, and the geometric attribution sends it to the direction NORMAL to
the face, which is not the direction that would reduce it. Two things follow
and neither has been tried:

1. **The flag rule is a hard threshold and converts a systematic bias into a
   wrong answer.** A direction is refined when it holds more than
   `0.15*3/dim` of the element's energy, 0.225 in 2D. At `-ks 1` the
   postprocessed estimate's `y` share is 0.219 -- it misses by 3% -- where the
   computed potential's is 0.34. A rule relative to the largest component
   rather than to the sum would not be tipped by a uniform bias. Worth
   measuring; the constant carries a `TODO: reorientation with the element`
   already.
2. **Or accept that the two fields answer different questions** and use the
   computed potential for direction and the postprocessed one for magnitude,
   which is one estimator object each and costs one extra face loop.

Until one of those is measured, `--anisotropic-estimate` defaults off under
`--hp-adaptivity`, which is what the demonstrator's numbers were taken with.

### Mechanism, and what it caps

**A coarsened boundary face cannot carry an essential datum, and the fix has
two halves.** `RetireSurplusTraceDofs()` now refuses the combination rather
than being wrong by 21x -- measured by sweeping the ceiling over a fixed mesh
where every face sits at the element degree, so the answer cannot legitimately
move: the weak datum gives 0.0225002 at every ceiling from 2 to 8, identical to
every printed digit, and the essential one gives 0.0124, 0.0926, 0.196, 0.259.
Closing it needs the surplus slots forced to zero regardless of what the
caller's vector holds -- those dofs are this route's, not the caller's -- and
the datum projected face by face at the face's own degree, which needs an
entry point the constraint space does not offer. A ceiling equal to the element
degree reproduces the essential answer exactly and is the way round it today.

**A hanging-node family has to run at the ceiling**, which is where coarsening
stops. The reason is in `SetTraceOrders()`: the constraint space's conforming
prolongation interpolates in the ceiling basis, and this route's convention --
coarse coefficients followed by retired zeros -- is a different function in
that basis. Removing the restriction means constraining the surplus slots
rather than retiring them, so that a face's coefficient vector is the coarse
function *expressed at the ceiling*. That is a different mechanism, not an
adjustment, and it is what would make `h` and `p` compose without a penalty at
every hanging node.

**The route is shaped for coarsening and has only ever been driven upwards.**
A face can go below the degree the constraint space was built at and never
above, so a driver that starts uniform at `p_max` and *coarsens* where the
sensor says the element is over-resolved uses the mechanism the way it is
built, needs no ceiling raise, and starts with hanging-node families already
at the degree they are stuck at. Nothing in the tree does this and it is the
cheapest unexplored direction.

**Parallel is absent, not incomplete.** `FaceOrdersFromElementOrders()` refuses
a `ParMesh` with shared faces and non-uniform degrees rather than guessing, and
closing that needs one exchange of element degrees over face neighbours -- but
beyond the library, neither `pconvdiff` nor `panisodiff` carries a single
p-adaptivity flag, so there is nothing to drive it with either.

**`DarcyForm::Reconstruct()`** still reads the trace space directly at six
sites, so `-pref` refuses `-rec`. `HDGPotentialPostprocessor` is the
reconstruction that a per-face degree does not disturb and is what the
demonstrator uses.

### Measurements not taken

**Essential against weak trace boundary conditions. SETTLED, and the earlier
claim here was wrong.** It said `--trace-ess-bc` is "about three times cheaper
at fixed error" and quoted 7.3e-4 against 2.7e-3 at M = 1272 and 2.1e-4
against 6.8e-4 at M ~ 1130. Those pairs are at matched **M**, so 3.7x and 9.8x
are ERROR ratios, not dof ratios; `t_err` falls like `M^-2` on these curves, so
they are a **1.6x** dof saving. Read properly at matched error over three
values of `ks`, four methods and thirty interpolated points, the ratio is
**1.0 to 1.7, it decays with error, and it dips below 1 in six of the thirty**.

It is also **not an adaptivity effect**: the same ratio appears on a uniform
mesh. The uniform error ratio peaks at 1.9-2.0 exactly where `h` reaches the
layer thickness and decays either side of that -- 1.23, 1.56, 1.92, 1.91,
1.57, 1.27 over nx = 8 to 256 at `ks = 1e2`, layer 1/31 -- so it is a
transient of the layer-resolution regime, and the dof saving underneath it is
just the pinned boundary trace dofs, `1 + 1/nx`, which is 1.004 by nx = 256.

So the table stays with the weak datum, and next to hp itself this is
second-order: at `ks = 1e2` and `t_err = 1e-5`, hp needs M ~ 3700 where
h-adaptivity needs M ~ 20700, a factor of 5.6, against `--trace-ess-bc`'s 1.1
at the same point.

**Orders and dimensions. DONE, and the headline is that the CEILING, not the
sensor, is making the hp decision.**

`hp` works at orders 1 to 4, with the largest gains away from order 2 -- 432x
and 228x better than uniform at comparable `M` at orders 1 and 2, and 130x and
42x better on four times fewer dofs at orders 3 and 4. Order 0 is a special
case rather than a failure: the sensor reports the least-smooth value at
degree 0 by design, so `0 < 0` is false, `p` is never chosen in any cycle, and
`--hp-adaptivity` degenerates to plain h-AMR worth 2 to 3x.

**But `spend_on_p` requires `p < p_max`, and that clause is doing most of the
work.** Varying only the ceiling, the number of h-refinements collapses by 45
to 100x -- 812 to 18 at order 2 as the ceiling goes from `K+1` to `K+5`, 600 to
6 at order 3 -- and at matched `M = 1200` a generous ceiling is uniformly
better per dof, by 53x at order 2 and 214x at order 4. Meanwhile `--hps` is
nearly inert: the threshold sits **below the entire sensor distribution over
marked elements**, so stricter hurts badly (10-35x at matched `M`) and more
lenient does nothing at all at order 3, with `+1` and `+2` byte-identical. So
the decision being taken is effectively *"p unless at the ceiling"*, `-4
log10(p)` is not badly chosen but is not discriminating either, and **order 2
is the only order measured so far at which the sensor discriminates at all**.

Two things follow. The default ceiling `order+3` is conservative and the
measurement says raise it -- held pending the wall-clock answer, since the
ceiling costs `(pmax+1)/(order+1)` on *built* trace storage and whether that
costs time is the open half. And the sensor deserves a problem that exercises
it, because this one does not.

**Three dimensions works, and getting there found a silent wrong answer that
was not ours.** `anisodiff` set problem 5's Dirichlet faces by 2D attribute
index; `Mesh::Make2D` numbers them 1=y0, 2=x1, 3=y1, 4=x0 and `Mesh::Make3D`
numbers them 1=z0, 2=y0, 3=x1, 4=y1, 5=x0, 6=z1, so the 2D pair landed on
`z = 0` and `x = 1` and the layer faces got no condition -- and it ran to
completion returning 0.986 as though nothing were wrong. Set by geometry now.
With that, 3D hp reaches 2.6e-2 at `M = 21289` against uniform's 7.6e-2 at
`M = 117504`: 2.9x the accuracy on 5.5x fewer dofs.

**And simplices were unreachable for one default argument.**
`Mesh::EnsureNCMesh()` leaves simplex meshes conforming unless told otherwise,
so `FiniteElementSpace::Construct()` refused every variable order on a
triangle or tetrahedron mesh -- in 2D as much as 3D. `EnsureNCMesh(true)` fixes
it, and the side effect is worth knowing: `--hp-adaptivity` now works on
tetrahedra while plain `--amr-ref-levels` on the same mesh still aborts in
`Mesh::LocalRefinement` wanting `Finalize(true)`, because hp needs the
nonconforming representation and conforming tet refinement is what is broken.
That h-only abort is upstream of this branch and left alone.

**Nothing measures time.** Every curve is error against `dim M`, which is the
right axis for a hybridized method and is not the whole cost: the ceiling makes
the trace vector `nt(p_max)` per face whatever the degrees are, and whether a
sparse direct solve follows the *active* size is the reasonable expectation and
has never been checked.

### Coverage

**No regression reference for the `hp` loop at all** -- `regression_test.py`
drives `convdiff` only, so the demonstrator is measured and not pinned. Either
teach the script `anisodiff`, or add a `convdiff` case that runs the loop.

**No `[Parallel]` p-adaptivity unit test**, which follows from parallel being
absent.

**The `h`-or-`p` junction has no test.** `HDGErrorEstimator` and
`PerssonPeraireSmoothness` each have cases; the rule joining them lives only in
`anisodiff` and is checked only by the demonstrator converging.

**The estimator's caller-side setup is per-miniapp and easy to get wrong.**
`SetExcludedBoundary()`, `SetHybridization()` and `SetTraceComparison()` all
default to the old behaviour, so a caller that forgets one gets a quietly wrong
estimate rather than an error -- which is exactly how all three were found.
Only `anisodiff` sets them.

### Deliberately not planned

RT and broken-RT flux spaces, by standing instruction. And the other route --
one variant per entity inside `FiniteElementSpace` -- which is what a genuinely
minimal trace space needs and is not required to find out whether any of this
is worth having.

## Acceptance

1. **Null test**: every `p_f` equal to the uniform order reproduces every
   existing answer bit-for-bit. DONE, and it caught two defects that nothing
   else would have -- the raised ceiling perturbing the error estimate, and
   `SubDofOrder()` answering at the collection's degree.
2. A mesh carrying two element orders converges at the rate its trace orders
   set, and reaches a given error at fewer global dofs than uniform `p_max`.
   DONE; see the table in `anisodiff.cpp`.
3. **`min` against `max` at a genuine `p`-interface. DONE**, and `max` wins:
   21-27% of the dofs at fixed potential error in the hp loop, and `min` is
   measured to get *worse* as the degree jump grows. `anisodiff --p-face-rule`
   now defaults to `max` under `--hp-adaptivity` for that reason, and
   `TraceOrderRule`'s doxygen carries both studies and why they only look
   contradictory. The original note read: The investigation showed
   a trace richer than *both* neighbours is exactly redundant -- **on a
   conforming mesh, and that qualifier turned out to matter**: across a hanging
   node the master sees several fine elements which between them do reach the
   higher modes, so the extra degrees are determined rather than annihilated
   and the answer changes, measured 0.118 against 0.098 for the worse. Whether
   `max` earns its dofs at a genuine `p`-interface is still open. Measure it;
   do not pick a rule by argument.
4. Rank-count independence on `pconvdiff` at 1, 2, 3, 4 ranks.

## The cost of branching from the trunk

Settled, and it is **`doc/HDG-P-ADAPTIVITY-MEQ-MERGE.md`** now rather than
here. What this section used to say -- that meeting `gf-hdg-linearise-first`
in `meq-integration` costs "four `c_fes` call sites", and that an unconverted
one would be loud -- was right about the count, wrong about the work, and
never measured on the second point. The trial merge says the port is six named
substitutions in five named functions, that 51 of this branch's 53 conversions
merge cleanly, and that the conflict is two hunks where lf restructured
`ComputeH`. Whether an unconverted site is loud is still a guess, and that
file says how to find out.

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
