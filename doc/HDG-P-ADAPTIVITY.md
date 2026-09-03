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

### The open question, answered -- and it moves the blocker

**The anisotropic split's two jobs come from different fields.** `p̂ - λ` on
the computed potential is the scheme's own stabilization term and its
directional split is right; on the postprocessed potential the same difference
is essentially λ's own error, real but not the element's, and the geometric
attribution sends it to the direction NORMAL to the face rather than the one
that would reduce it. Four measurements ruled out everything else -- not the
degree gap (`--postprocessed-projected-down` removes it entirely and every flag
stays put), not the anisotropy (the same stall at `-ks 1` and on problem 6),
not the marking (both estimates select the same elements for two cycles) -- and
the directional energy differs by up to seven orders. It is written up on
`HDGErrorEstimator::SetAnisotropic()`.

**The fix is to take each from the field that answers it**, and it needs no
library change: a second estimator on the computed potential supplies
GetAnisotropicFlags() while the postprocessed one supplies GetLocalErrors().
`anisodiff --anisotropic-estimate 2`, now the default off hp. It works:

| h-adaptive | M at 1.1e-3 | at 1e-4 | at 3.2e-5 |
|---|---|---|---|
| before, both jobs from one field | 1704 | 4812 | 9330 |
| direction from the computed potential | 1146 | 3501 | 6258 |

-- and it turns `--postprocessed-estimate` from something that bought only
cycles into something worth 1.4x in dofs, which is a controlled comparison
since both rows above take the direction from the same place.

**But hp still cannot use it, and the reason is the ceiling, not the estimate.**
A hanging-node family has to run at the ceiling degree, which enriches the
trace across every hanging node, and anisotropic refinement makes hanging
nodes prolifically. Holding everything fixed and moving only the ceiling: at
`--max-order` equal to `--order` the hp loop reproduces the non-hp one **to
every printed digit**, and one degree above it stalls at 0.078. Every higher
ceiling stalls, and so does the run with p-refinement disabled altogether, and
so does the run with the direction taken from the computed potential. So it is
the ceiling at the hanging nodes and nothing else.

**FOUND, and fixed: the excess is a magnitude, not a direction.** A
per-element dump on one identical hanging-node mesh, changing only the ceiling
from 2 to 3, puts the whole difference on the twelve elements next to a
hanging node and entirely in `d₀`:

| | Σd₀ at ceiling 2 | at 3 | Σd₁ at 2 | at 3 |
|---|---|---|---|---|
| next to a hanging node | 1.11e-4 | **5.45e-2** | 6.55e-3 | 4.51e-3 |
| everything else | 2.91e-5 | 2.83e-5 | 6.93e-3 | 3.60e-3 |

A factor of 490. The master trace at the ceiling fits the several fine
elements better than the one coarse element, so the coarse element's
`|p̂ - λ|` genuinely grows -- right as a magnitude, since it *is* the
mismatched element, and exactly wrong as a direction. Refining in `y` puts
hanging nodes on **vertical** faces, whose energy the geometric split
attributes to `x`, so the neighbour is split in `x` when another `y` is what
would match it, and the loop alternates forever. Four elements in the layer
flip `y` to `x` at seventeen times their estimate.

`HDGErrorEstimator::SetSkipEnrichedDirection()` keeps such a face's magnitude
and drops its direction. Anisotropic refinement then works under hp, at 1.5 to
1.9 times fewer unknowns than the isotropic loop -- 1.05e-4 at M = 921 against
1351, 1.8e-6 at M = 1302 against 2473.

**Two other repairs were tried and measured to fail**, and both are recorded
next to the fix so nobody spends them again. Dropping the face altogether
stalls the hp loop at 1.7e-3 against 1.4e-6, because it discards the part of
`p̂ - λ` the element *can* see. And projecting λ down to the element's own
degree -- which removes exactly the modes it cannot represent -- moves eta by
2% and changes no flag: the excess is not in λ's high modes, it is in where λ
sits, and λ sits where the fine side puts it.

**And the plateau that left behind was the same term at a `p`-interface, which
is now closed too.** Comparing the estimate against the TRUE per-element error
-- the diagnostic that should have been reached for first, three times -- the
stalled loop was marking a cluster of degree-2 elements next to degree-5 ones,
in the middle of the domain, nowhere near the layer:

| cycle | η on the marked cluster | true error there | ratio |
|---|---|---|---|
| 22 | 4.7e-6 | 1.1e-8 | 443 |
| 24 | 9.3e-6 | 5.4e-9 | 1700 |
| 25 | 1.2e-5 | 3.8e-9 | 3000 |

Wrong by three orders and getting worse, while the elements actually carrying
the error -- five times more of it -- went unmarked. And self-feeding:
splitting them in `x` makes them narrower, `τ ~ 1/h` on their vertical faces
grows, and η grows with it, so the refinement the estimate triggers is what
makes the estimate bigger.

`HDGErrorEstimator::SetCapTraceAtElement()` compares such an element against λ
projected down to its own degree. **The two halves together -- direction
skipped, magnitude capped -- are what a face richer than its element needs**,
and neither alone is enough: at a hanging-node family the cap moves eta by 2%
and only the direction matters, at a `p`-interface the cap is the whole of it.

With both, the `max` face rule is usable and is the default under hp again:
about 10% of the dofs at every matched error and an order deeper in the same
cycle budget, 4.5e-10 at M = 3264 against `min`'s 3.0e-9 at M = 3022. That is
consistent with what a prescribed interface says about the rule on its own,
where `min` gets worse as the degree jump grows.

**So every question about the estimate is closed.** The table at the defaults:

| ‖t−t_ex‖ | uniform M | h-adapt M | hp M |
|---|---|---|---|
| 1.0e-3 | 24960 | 1146 | 717 |
| 1.0e-4 | 99072 | 3501 | 952 |
| 3.0e-5 | — | 6258 | 1054 |
| 1.0e-7 | — | — | 1847 |
| 4.5e-10 | — | — | 3264 |

104 times fewer globally coupled unknowns than uniform refinement at 1e-4 and
3.7 times fewer than `h`-adaptivity, and four decades past where either can be
run.

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

Two things follow. **The default ceiling is now `order+5`**, the wall-clock
answer having come back and said the ceiling is nearly free -- see below. And
the sensor deserves a problem that exercises it, because this one does not.

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

**Wall clock. DONE, and it does not rank the methods the way dofs do.**

*The ceiling is nearly free, and the direct solve does follow the active size.*
Holding the mesh and every face degree fixed and moving only the ceiling from 2
to 7 -- a 2.67x storage ratio, with `M` and the error identical to every
printed digit at all five mesh sizes, which is what says the probe isolates it
-- assembly comes out 0.98-1.07x, the preconditioner 0.94-1.23x, the trace
solve 1.03-1.19x, and peak RSS at most 1.15x. Only the hybridization's own
setup scales, 1.46-1.64x, and it is about 5% of a run. So the ceiling costs
roughly 2% of wall clock for its 2.67x of storage, and the default is now
`order+5`: on the demonstrator that takes the dofs at 1e-7 from 13191 to 4189
and the wall clock for 26 cycles from 10.8 s to 6.4 s, at a better error.

*But hp does not win in seconds until about 1e-5.* An adaptive loop pays for
every intermediate solve, and hp takes more cycles to reach a given error than
h-adaptivity does. At 1e-4: h-adaptive 0.51 s, hp 0.81 s, uniform 2.22 s. hp
overtakes below 1e-5 and then wins outright -- 4.8x faster than uniform at
7e-6, 11x at 2e-6 -- where h-adaptivity cannot reach at all, dying on
direct-solver memory at `M` around 1.4 million. **The dof ranking is not the
time ranking, and a table quoting only dofs oversells hp at loose tolerances.**
Both are in the miniapp's header now.

### Coverage

**No regression reference for the `hp` loop at all** -- `regression_test.py`
drives `convdiff` only, so the demonstrator is measured and not pinned. Either
teach the script `anisodiff`, or add a `convdiff` case that runs the loop.

**No `[Parallel]` p-adaptivity unit test**, which follows from parallel being
absent.

**The `h`-or-`p` junction has no test.** `HDGErrorEstimator` and
`PerssonPeraireSmoothness` each have cases; the rule joining them lives only in
`anisodiff` and is checked only by the demonstrator converging.

**The estimator's caller-side setup is per-miniapp and easy to get wrong**, and
this is the one piece of it that got worse rather than better. Five of the six
things a `p`-adaptive caller must ask for default to the old behaviour, so
forgetting one gives a quietly wrong estimate rather than an error -- which is
how every one of them was found. Two are now implied by `SetHybridization()`,
since they can only bite where per-face degrees exist and are measured inert
otherwise; `SetExcludedBoundary()`, `SetTraceComparison()`, `SetAnisotropic()`
and the choice of which field supplies the direction are still the caller's,
and only `anisodiff` gets them all right. A driver-side helper that sets them
together is the obvious answer and does not exist.

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
3. **`min` against `max` at a genuine `p`-interface. DONE**, and `max` wins,
   but only once the estimate stops charging an element for modes it cannot
   represent -- without that, `max` is what creates such faces and the loop
   plateaus. With both halves handled it is worth about 10% of the dofs at
   every matched error and reaches an order deeper in the same cycle budget,
   and `min` is separately measured to get *worse* as the degree jump grows.
   `anisodiff --p-face-rule` defaults to `max` under `--hp-adaptivity`.
   `TraceOrderRule`'s doxygen carries the studies. The original note read:
   The investigation showed
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
