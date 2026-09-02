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

**`min` against `max` at a genuine `p`-interface**, which the demonstrator is
now the place to measure.

**Essential against weak trace boundary conditions.** The table in
`anisodiff.cpp` was taken with the weak datum, which is the miniapp's default.
`--trace-ess-bc` is about **three times cheaper at fixed error** on the same
problem -- h-adaptive 7.3e-4 at M = 1272 against 2.7e-3, and hp 2.1e-4 at
M = 1139 against 6.8e-4 at M = 1311. Worth knowing which the table should be
taken with, and it is a flag on the command line rather than a change of
default: moving the miniapps onto the essential-trace route is not ours to do.

**One order, one dimension.** Everything measured is 2D at `--order 2`. Orders
0, 1 and 3 and a 3D case are unexercised by the loop, and the sensor's
threshold `-4 log10(p)` is a 1D argument.

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
3. **`min` against `max` at a genuine `p`-interface.** The investigation showed
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
