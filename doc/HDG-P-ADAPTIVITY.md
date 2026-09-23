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

*And on reconstruction.* **`-pref` no longer refuses `-rec`, and every reason
this paragraph used to give for the refusal was wrong.** It said the local
problem's shapes "assume one trace degree per element" and pointed at six
direct reads of the trace space in `darcyform.cpp`. A backtrace puts the abort
in `DarcyHybridization::ReconstructTotalFlux`, not in `DarcyForm` at all; that
routine never sees a per-face degree, `TraceFE()` being the CEILING's element
for every face by construction; and it fired with **zero** elements refined
and again with every face at the ceiling, both uniform traces, so "per-face"
was never the trigger.

What it was: `DarcyForm::ReconstructTotalFlux()` built the total flux space
from the **flux** collection's order where it has to match the **trace**'s.
The two agree in every configuration without `-pref` and differ the moment the
constraint space is built a degree up. One line, and all three `-prefx` arms
then run, mixed element degrees included.

**But the reconstruction's answer FOLLOWS THE CEILING, and that is new and
undocumented.** Its local problem is built from the constraint space's
collection -- `ReconstructFluxAndPot()` clones it one degree up for the
enriched trace, and the total flux now follows the trace too -- so a higher
ceiling is a genuinely richer postprocessing of the same discrete solution.
Measured with the discrete problem held fixed (`-o 2 -rec -pref n -prefx 0.0
-nx 8`): the primary flux error is 5.13313e-04 at every ceiling against the
uniform arm's 5.13312e-04, while `t_hs` goes 3.507e-05, 3.503e-05, 3.132e-05,
2.363e-05 as the ceiling rises. **So a caller choosing a ceiling for the TRACE
is also choosing the reconstruction's richness.** Pinned by "Reconstruction
runs under a constrained trace and is the coarse one" in
`tests/unit/fem/test_darcy_reconstruction.cpp`, whose first draft asserted the
reconstructed fields MATCH the uniform arm and failed at 3.9e-03.

Two smaller things the same chase turned up, both fixed: the contract
`ut` must satisfy was an `MFEM_ASSERT`, compiled out of a release build, so a
mismatched space aborted forty lines later inside `DenseMatrixInverse::Factor`
with a message naming neither space -- it is an `MFEM_VERIFY` now and says
which face and which two counts. And `ut`'s RT **interior** DOFs were never
initialised: the face loop writes only face DOFs and `GridFunction::SetSpace()`
sizes without zeroing. It reads as zero deterministically here, a fresh `mmap`
being zero-filled, which is the allocator's luck and not a contract.

`HDGPotentialPostprocessor` -- the classic local postprocessing, Nguyen,
Peraire & Cockburn eq (25) -- has no such problem: it reads the flux and the
potential on the element it is working on and nothing else, never the trace
space and never a neighbour, so what degree the faces carry cannot reach it.
Its `Compute()` already took `GetFE(z)` per element from all three spaces; the
only thing that was uniform was the enriched space it builds by default, which
now follows the potential element by element. That is `convdiff -pp`, it works
under `-pref`, and it is what a `p`-adaptive run should use.

**2. Constrain the surplus slots. DONE, and it replaced retiring them.**
A face's slots hold the CEILING basis's coefficients of a function of the
face's own degree, and a per-face `E` says so; the reduced system is in the
constrained unknowns, so it is the sum of `nt(p_f)` and carries no unit rows
at all. Retiring them into `ess_tdof_list` came first and is gone -- it is
what the three closed limits below were all about.

The trap this step actually turned on was elsewhere. `Init()` runs from
`EnableHybridization()`, so C, E, G and H were already built at the *uniform*
degree before any caller could state a per-face one -- the dof count came out
exactly right and the system was wrong. `SetTraceOrders()` therefore rebuilds
them and calls `Reset()`, and must be called straight after
`EnableHybridization()` and before `Assemble()`. Under the present sizing that
rebuild is a no-op; the contract is kept because taking the local blocks back
down to `p_f` would need it back.

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
cycle budget, 4.5e-10 at M = 3264 against `min`'s 3.0e-9 at M = 3022 -- both
measured before hanging-node families were freed from the ceiling, which cost
`max` 1 to 9 per cent of its dofs. That is
consistent with what a prescribed interface says about the rule on its own,
where `min` gets worse as the degree jump grows.

**So every question about the estimate is closed.** The demonstrator's table
lives in `miniapps/hdg/anisodiff.cpp`'s header, where it is maintained; the
copy that used to sit here went stale the moment the hp column moved, which is
the whole reason for the rule about where findings live.

### Mechanism, and what it caps

**ALL THREE OF THE LIMITS THIS SECTION USED TO DESCRIBE ARE CLOSED**, and
what closed them is one change: the trace surplus is CONSTRAINED rather than
retired, so a face's slots hold the ceiling basis's coefficients of a function
of the face's own degree instead of a coarser basis's coefficients followed by
zeros. A hanging-node family can sit below the ceiling, an essential datum can
sit on a coarsened boundary face, and a shared face can be coarsened; the
measurements that justified each refusal are turned round in
`DarcyHybridization::SetTraceOrders()`'s doxygen, next to where the refusal
used to be. `doc/HDG-P-ADAPTIVITY-CONSTRAIN.md` is what is left of the plan.

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

**The `hp` loop now has an acceptance test**, `miniapps/hdg/hp_acceptance.py`,
run by `make hp-acceptance`. It is not a stored answer, because the thing worth
defending is a RELATION between three runs rather than one number: hp must
reach 1e-9 at all, and must need at most two thirds of `h`-adaptivity's
globally coupled unknowns and a fifth of uniform refinement's at each of two
tolerances. Currently 9.9e-10, and ratios of 0.540 and 0.276 against `h`,
0.0073 and 0.0098 against uniform. It is shown to be able to fail rather than
assumed to be: `HP_ARGS=... -no-captr` takes the loop to 1.02e-9 and the reach
check trips. That margin used to be three orders -- the cap was worth 8.7e-7
against 5.9e-8 -- and shrank when hanging-node families stopped being forced
to the ceiling, which was where most faces richer than their elements came
from. The check still discriminates, and only just; it is worth watching.
Serial only, for the reason above.

**The `[Parallel]` p-adaptivity unit tests are two.** One checks the derived
face degree against an INDEPENDENT computation -- the degrees are a function of
the element centre, so each rank works out what its neighbour must have had
without being told -- at 2, 3 and 4 ranks, both rules. The other checks that
the constrained trace size summed over ranks equals the SERIAL answer for the
same mesh and degrees, which is the shared-face refusal turned round; it is run
at 1, 2, 3 and 4 ranks and `pconvdiff --p-refine` agrees to five digits at all
four.

**The `h`-or-`p` junction has a test now, and getting one meant moving the
rule.** It was three lines inside `anisodiff`, where nothing could reach it;
it is `PerssonPeraireSmoothness::SpendOnP()` next to the `Threshold()` it
calls, and the miniapp calls that -- which is what makes the case worth
anything. A static member, so no layout moves.

The case pins the STRUCTURE and not a tuned number, because the structure is
what this branch measured as mattering: the ceiling clause is a hard gate that
comes first and does most of the work, the sensor decides below it at the
paper's strict threshold, the shift moves that boundary the documented way,
and `Threshold()` is monotone in `p` so an element cannot be enriched forever.
Falsified by dropping the ceiling clause and by relaxing `<` to `<=`; both
fail it. `make hp-acceptance` reproduces the recorded run to every figure
after the move -- 9.92e-10, ratios 0.540 and 0.276 -- so the junction decides
exactly what it decided before.

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

## The cost of branching from the trunk — settled, and the merge is done

`doc/HDG-P-ADAPTIVITY-MEQ-MERGE.md` is **deleted**: the merge it planned has
happened, `meq-integration` carries this branch
(`git merge-base --is-ancestor` against it is yes), and by that file's own
opening criterion — "it goes when the merge is done and its findings are in
the code" — it had outlived itself. Its durable halves went where they belong:
the merge rules into the operational notes, and the one real defect it found
into the code (`e764526794` — `anisodiff` built the estimator's `amr_bfi` as a
bare `HDGDiffusionIntegrator` while the potential mass form got
`SetStabilization(*stab)`, so with `--tau-floor > 0` the estimate measured a
different stabilization than the solve used; the `-tf 0` column is the control
that makes the fix believable, being inert to every printed digit).

Five of its six semantic questions were answered by the CONSTRAIN redesign
rather than by the merge, and it is worth knowing why: under that design
`TraceFE()` and `TraceVDofs()` are plain passthroughs, so there is no lazy
`var_orders` cache to race, no unconverted site to be loud about, and
`CanBatchLocalFactor()` refuses on the flux/potential offsets — which are
trace-independent, so variable ELEMENTS refuse and a varying TRACE correctly
still batches. The planned "six substitutions" became cosmetic.

**That question has been run, the blocker it found is repaired, and the
question itself is still one run.** The matrix-free gradient is `-gm 2`, "do
not assemble, apply it and solve unpreconditioned"; `-gm 1` is the assembled
operator with a Gauss-Seidel preconditioner, and this section used to name the
wrong one. It is honoured only where the solver asks for a gradient at all,
which is the nonlinear route under Newton — and that route did not run under
`-pref` in either gradient mode, so the arm that was meant to be the control
is what convicted. `convdiff -p 1 -o 2 -dg -hb -nl -nld -nls 3 -pref 1` died
inside `malloc()` with `-gm 0` and with `-gm 2` alike, and on this branch,
which has no `GradientMode` at all, with no `-gm` on the command line.

### What was wrong, and it was three call sites rather than the loop

`MultNL()` addresses the trace by `TraceVDofs()` throughout, so it can only be
handed a vector in the CONSTRAINT SPACE's VDOFs — while the vector a nonlinear
solver hands the operator is in the trace's true unknowns, which are fewer
once a face carries less than the ceiling. Every other route already converts:
`ReduceRHS()` accumulates at the ceiling and applies the prolongation's
transpose, `ComputeSolution()` prolongs before the local solves, `ComputeH()`
does it as a RAP, and `ParMultNL()` — whose name is the only parallel thing
about it — is the wrapper that prolongs, runs the loop and restricts.

`Mult()`, `GetGradient()` and the matrix-free `Gradient` called the element
loop **directly**, which was right for exactly as long as the trace
prolongation was the space's own conforming one and therefore null in a serial
build. Under a per-face trace it is not null, and those three then indexed a
vector of constrained unknowns with the ceiling's numbers: valgrind names an
invalid read in `Vector::GetSubVector` and an invalid write in
`Vector::AddElementVector`, both eight bytes past the 138-double block
`RestrictTrace()` allocates on a 4x4 run the miniapp reports as `138 of 160
trace DOFs active`.

They go through the wrapper now, and two more defects of the same family went
with them:

* `ParMultNL()` sized its own trace output from
  `c_fes.GetRestrictionOperator()`, which is null together with the
  prolongation at a uniform trace and *not* once a face is coarsened — so the
  alias it builds claimed `c_fes.GetVSize()` entries of a vector that is only
  `ctr_offsets.Last()` long. Latent, because until now nothing reached it in
  serial with a per-face trace. It is guarded on the prolongation now.
* `ParOperator` and `ParGradient` announced `c_fes.GetTrueVSize()` where the
  operator's size is `GetTraceTrueVSize()`. `Finalize()` had already made that
  distinction for the serial operator and said why in a comment; the two
  parallel classes were written before it.

And the invariant is a check rather than a convention now: `MultNL()` opens
with `MFEM_VERIFY(x.Size() == c_fes.GetVSize())`, one comparison a residual
evaluation, so a fourth site cannot be added the same way.

### What pins it

Three cases in `tests/unit/fem/test_darcy_padapt.cpp`:

* **"A nonlinear solve under a per-face trace is the coarse problem"** — the
  nonlinear counterpart of the linear equivalence above. A ceiling of
  `order+gap` with every face constrained to `order` must give the same
  recovered fields as a trace space actually at `order`. Not a fall-through:
  the constrained arm goes through `E` and `ctr_PE` and the other does not.
* **"A genuinely non-uniform trace carries a nonlinear solve"** — alternate
  faces one degree BELOW their elements, which has no uniform twin, so what is
  asserted is that it solves, that the system really did shrink, and that the
  coarsening *moves* the answer. The direction matters and the first draft had
  it backwards: with the elements at `order` and the ceiling at `order+1`,
  coarsening half the faces back to `order` moves the potential by **1.8e-14
  on 1.2e+00**, because a trace richer than both its neighbours is exactly
  redundant — which this branch had already measured and which the draft was
  answered by.
* **"The reduced nonlinear gradient and residual are in one numbering"** — the
  sharp one, and the one a half-repair fails. `ComputeH()` restricted the
  gradient and always did; the residual had to be taught. Prolong one and not
  the other and they are operators on different spaces, which a solve would
  show only as Newton converging badly. A central difference of the residual
  along a fixed direction settles it directly.

  It compares only the rows the residual depends on, and that exclusion is
  the thing to read before the number. This fixture puts no boundary face
  integrator on `B`, which is how the hybridization's constraint reaches a
  boundary face at all — so a boundary face's trace rows are empty, and
  `ComputeH()`'s `EliminateZeroRows()` gives them a unit diagonal to keep the
  matrix invertible while the residual leaves them zero. Newton is right
  either way, its correction there being zero whichever is used. **The first
  run of this check reported a relative error of 0.358 and I nearly read it as
  the repair being broken**; the arm with no per-face trace at all reports
  0.348, which is what said the number was about the fixture and not about the
  numbering. With a boundary constraint installed, every mode and order agrees
  to between 4.6e-11 and 8.3e-10.

`GradientMode` is `gf-hdg-linearise-first`'s, so `-gm 2` under `-pref` is
still a run that belongs in `meq-integration`, where both exist. It is now a
run that can be made.

### The three-site routing is TRUNK material, and this branch is not where it belongs

The per-face trace is the second way to reach that defect. The first has been
on every branch all along: **`DG_Interface_FECollection` derives from
`RT_FECollection` and therefore reports `GetContType() == NORMAL`, not
`DISCONTINUOUS`** — so `FiniteElementSpace::BuildConformingInterpolation()`
does not take its early exit, and on a nonconforming mesh with hanging nodes
the trace space has a real conforming prolongation. Measured on
`data/amr-quad.mesh` with one uniform refinement, order 3: `VSize = 1056`,
`TrueVSize = 928`, `cP` **NONNULL**. `Mult()` then hands `MultNL()` a 928-long
true-dof vector and indexes it by VDOFs drawn from 1056.

It reproduces on `gf-hdg-linearise-first`, which has no per-face trace at all:

```
convdiff -no-vis -m ../../data/amr-quad.mesh -nx 0 -ny 0 -r 1 -o 1          -dg -hb -nl -nld -nls 3
```

dies on `IsFinite(norm)`, and valgrind names **five invalid reads eight bytes
past a 3712-byte block**, from `MultNL()` called by `Mult()` and by
`ReducedGradient()` — a fourth site, and NPC's.

**No reference anywhere covers the combination.** Grouping the `_nc_`
references by their options: every nonlinear one is NOT hybridized, and every
hybridized one is linear. Nonconforming + hybridized + nonlinear has never
been run.

The trunk-material part is small — route the three sites through
`ParMultNL()`, plus the `MultNL()` invariant — and lifting it is a separate
operation across five branches, with `ReducedGradient()` to add on the two
that have it. Not done here.

### What the repair uncovered: a wrong Jacobian, and it is TRUNK material

**FIXED.** `convdiff -p 1 -dg -hb -nl -nls 3 -pref 1 -prefx 0.5` took **26**
Newton iterations at order 2 where every uniform configuration takes 1, and
diverged to 3.3e+112 at order 3. It now takes **one** at both, on the same
answer. The defect is two lines in `DarcyHybridization::AssembleHDGGrad()`,
both overloads: they strode E and G by **their own element's** potential dof
count where `AllocEG()` sizes the face and `GetEFaceMatrix()` reads it by
**element 1's**. The two are equal whenever the neighbours carry the same
degree, which is every configuration in this tree but a `p`-adaptive one.

The account is on the routine. The pin is "The reduced nonlinear gradient and
residual are in one numbering on MIXED ELEMENT DEGREES" in
`tests/unit/fem/test_darcy_padapt.cpp`, which reports a relative Jacobian
error of **0.152 at order 1 and 0.167 at order 2** without the fix and under
1e-5 with it.

**It is not p-adaptivity's defect and the two lines are not this branch's.**
`AssembleHDGGrad()` predates every descendant and carries the same expression
on `gf-hdg-dev`, `gf-hdg-subdomains-dev` and `gf-interp-hdg-dev`, and
`gf-hdg-linearise-first` inherits it a third time in `SeedLinearEG()`, whose
comment says so ("Exactly AssembleHDGGrad()'s offset"). Checked with
`git show <branch>:fem/darcy/darcyhybridization.cpp`, not reasoned about.
This branch is merely the only one that can REACH it, mixed element degrees
being what it exists to produce. **The lift is owed and is not done here** --
the documented method applies: cherry-pick onto `gf-hdg-dev`, then merge out,
never rebase.

**The attribution that stood here was wrong in both halves and is withdrawn.**
It read "so it is the nonlinear FLUX on a variable-order element space", from
`-nld` being untouched. Two arms it never ran say otherwise:

| arm | what it puts on the nonlinear form | mixed elements | uniform, no `-pref` |
|---|---|---|---|
| `-nlp` | the potential mass, LINEAR content | **26 / diverges** | 1 |
| `-nlu` | the flux mass, LINEAR content | NaN | **NaN** |
| `-nld` | `MixedConductionNLFIntegrator` | 1 | 1 |

`-nlp` reproduces the whole thing on its own, so the flux was never necessary;
and `-nlu` fails on a plain uniform mesh with **no `-pref` at all**, so that
arm cannot have been about variable element degrees either. It is a separate,
pre-existing failure -- problem 1 with `-dg -hb`, a configuration no reference
covers -- and it is NOT chased here; `p2_o2_dg_hb_upwind_nlu_newton` passes,
so it is problem-dependent rather than a broken route.

**`-nlp` and `-nlu` are the sharp form of the reproduction because they are
arithmetically INERT**: both put a LINEAR integrator on a nonlinear form, so
the discrete problem does not move and only the route changes. The answer is
the linear route's to six digits in every arm that completes. A defect that
survives an inert knob is a defect in the route, which is what took this
from "a nonlinear solve on variable element degrees" to two lines of
addressing.

**And the residual was never wrong.** LBFGS never calls `GetGradient()`, and
it reaches the same answer to five digits at order 2 and the correct answer at
order 3 where Newton diverges. The miniapp's own residual history said so
before any code was read: 1.216e-05, 6.926e-06, 3.923e-06, 2.232e-06,
1.265e-06 is a **fixed linear rate of 0.567**, which is a systematically wrong
step direction and not a hard problem.

The standalone probe agrees from the other side: with uniform elements and a
genuinely non-uniform trace, Newton converges in **5 iterations to 1e-15** at
orders 1, 2 and 3 -- that axis was always sound, which is why the trace was
the wrong place to look.

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
