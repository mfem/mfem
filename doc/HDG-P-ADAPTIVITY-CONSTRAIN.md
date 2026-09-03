# Constrain the surplus instead of retiring it

Scratch, like everything in `doc/`. It goes when the work is done and its
findings are in the code.

Three things this branch refuses or works around are one thing. This is what
that thing is, what the repair is, which of two shapes it should take, what
each piece costs, and the order to build it in.

## 1. The one cause

A face of degree `p_f` in a constraint space built at ceiling `p_max` owns
`nt(p_max)` slots. The route stores its function as **`p_f`-basis coefficients
in the first `nt(p_f)` of them** and retires the rest as essential -- "a
different basis in the same storage". That is well defined for anything inside
`DarcyHybridization`, which knows the degrees. It is wrong for every reader
that does not, and the space's own machinery is exactly such a reader, because
it assumes the ceiling basis:

| symptom | where | measured cost |
|---|---|---|
| a hanging-node family cannot be coarsened | the conforming prolongation interpolates master to slave in the ceiling basis | families pinned at the ceiling; the error goes 0.284, 1.06, 3.67 if they are not |
| an essential datum cannot sit on a coarsened boundary face | the caller projects the datum in the ceiling basis | 21x, from 0.0124 to 0.259, from a parameter that must be inert |
| a shared face cannot be coarsened | the two ranks order its dofs by their own view of the orientation | 144 retired true dofs on one rank, 152 on two, 162 on three; error 5.9e-4 to 0.56 |

All three are refused in code today, with the measurements next to the
refusals. Each refusal is a real capability given up: coarsening stops at every
hanging node, `--trace-ess-bc` cannot be combined with a raised ceiling, and
`p`-adaptivity does not run on more than one rank at all. Two more things sit
downstream of the same cause and are worked around rather than refused:
`DarcyForm`'s reconstruction reads the trace space directly at six sites (so
`-pref` refuses `-rec`), and `HDGErrorEstimator` had to be handed the
hybridization so that it would read the face element at the face's degree.

## 2. The repair

Store the same function as its **ceiling-basis** coefficients, constrained to
the degree-`p_f` subspace. Two small dense matrices per (geometry, `p_f`) say
how, and they are interpolations in opposite directions:

    E(j, i) = phi_i^lo( node_j^hi )      nt(p_max) x nt(p_f)   embed
    R(i, j) = phi_j^hi( node_i^lo )      nt(p_f) x nt(p_max)   restrict

`R E = I` exactly -- interpolating a degree-`p_f` polynomial at the ceiling's
nodes and then reading it back at the coarse nodes returns it -- and that is
the whole algebra. `E` is the *solution* map, `R` the *data* map, `E^T` the
*residual* map, `R^T` the *scatter* map. In the uniform case both are the
identity and every one of those is a no-op, which is what keeps the null path
byte-identical.

**They are keyed by geometry and degree, not by face.** The rows of `E` are the
ceiling face element's nodes in the face's own reference ordering, which is
exactly the ordering `GetFaceVDofs()` returns; orientation never enters,
because the only places orientation matters -- the nonconforming transfer and
the parallel ldof-to-tdof map -- both act at the ceiling, where the space's own
machinery already handles it. So the cache is a handful of matrices, not one
per face.

Everything downstream that assumes the ceiling basis is then right, because the
stored coefficients *are* ceiling-basis coefficients:

    x_vdofs  =  cP . P_E . c            serial, nonconforming
    x_ldofs  =  Dof_TrueDof . P_E . c   parallel

where `P_E` is block diagonal over faces with `E_f` in each block.

### Why it cannot change the discretisation, and this is measured

A degree-`p_f` polynomial *is* a degree-`p_max` polynomial, so
`phi_i^lo = sum_j E(j,i) phi_j^hi` **pointwise**, and therefore any face matrix
assembled against the coarse trace equals the one assembled against the ceiling
trace restricted by `E` -- for a linear form exactly, and for a nonlinear one
exactly too provided the same quadrature rule is used on both, since the
identity holds at every point rather than under the integral.

That is an argument, so it is also a test:
`"A coarse trace basis is an exact combination of the ceiling's"` in
`tests/unit/fem/test_darcy_padapt.cpp` assembles
`NormalTraceJumpIntegrator` both ways on a real interior face and checks
`M_hi E == M_lo` to `1e-12` relative, over degrees 0, 1, 2 and gaps of 1, 2, 3.
It also checks `E^T E` is invertible, which is the other half: the constrained
face carries exactly `nt(p_f)` unknowns, neither fewer -- which would lose the
function -- nor more. **It needs one line adding: `R E == I`.**

**The reduced system gets smaller, not the same size.** The plan's earlier
draft said the dof count does not move. That was wrong: today's route carries
`nt(p_max)` rows per face of which `nt(p_max) - nt(p_f)` are unit rows, and the
constrained route carries `nt(p_f)`. The *active* count is what does not move.

## 3. Two shapes of the same repair, and the arithmetic that separates them

The choice is where `E` is applied, and it decides what the local blocks cost.

**Shape A -- promote.** `TraceFE()` returns the ceiling element for every face.
`C`, `E`, `G`, `H`, `Ct` are all ceiling-sized; the element-local static
condensation runs at the ceiling; `E` appears in exactly one place, the
prolongation. Nothing inside the assembly changes at all.

**Shape B -- constrain at the boundary.** The local blocks stay at `p_f`, as
today. Every gather from the global trace vector goes through `R`, every
scatter through `R^T`, and the global matrix carries `R^T H_lo R` per face.
Reduced: `E^T R^T H_lo R E = (R E)^T H_lo (R E) = H_lo`, which is today's
matrix exactly.

Shape A is one line on top of the shared infrastructure. Shape B is thirteen
`TraceVDofs()` sites, each gaining a small dense multiply. What buys the extra
work is local cost, and it is worth counting rather than timing first
(the trace-dependent element-local flops, `n_el` element dofs, `n_c` trace dofs
per element):

    A^-1 Ct :  n_el^2 . n_c        H_local = Ct^T A^-1 Ct :  n_el . n_c^2

so they grow as `r` and `r^2` with `r = nt(p_max)/nt(p_f)`. For the
demonstrator's own configuration -- 2D quads, element degree 2, `p_max = 7`,
`r = 8/3 = 2.67`, the storage ratio already measured:

| | today | shape A |
|---|---|---|
| `n_el` | 27 | 27 |
| `n_c` | 12 | 32 |
| `n_el^2 n_c` | 8,748 | 23,328 |
| `n_el n_c^2` | 3,888 | 27,648 |
| total | 12,636 | **50,976 (4.0x)** |

against a saving on the trace solve of **at most 1.19x**, because the unit rows
were already measured nearly free (1.03-1.19x solve at a 2.67x storage ratio,
peak RSS at most 1.15x). So shape A is a predicted *loss* at an extreme
ceiling, and the prediction has a threshold like the quadrature one did: at
`p_max = order + 1`, which is all a coarsening-only driver needs, `r = 4/3` and
the local growth is 1.4x, which nothing will notice.

**That is the staging.** Shape A first: it unlocks all three capabilities, it
is cheap at the ceilings those capabilities actually need, and it is small
enough to be verified against today's answers. Then measure the demonstrator.
Then shape B, which is a pure optimisation of shape A -- same reduced matrix,
different rounding -- and can therefore be validated against shape A's numbers
rather than argued for.

## 4. What to build

### 4.1 The constrained numbering

`ctr_offsets`, one entry per face plus one, prefix-summed: `nt(p_f) * vdim` for
a face that **owns** constrained dofs, zero otherwise. A face owns them iff its
ceiling slots are true dofs here, which is one uniform rule covering all three
cases -- serial conforming (every face), serial nonconforming (every face that
is not a slave), parallel (every owned face). Assert the all-or-nothing
property it relies on: a `DG_Interface` face's dofs are face-interior, so they
are all independent or all dependent, all owned or all not.

`GetTraceTrueVSize()` returns `ctr_offsets.Last()`.

### 4.2 The two matrices

`TraceEmbedding(geom, p_f)` and `TraceRestriction(geom, p_f)`, cached by key.
Built by evaluating one element's shape functions at the other's nodes, which
is what the existing unit test does.

**Verify `R E == I` numerically, once per distinct key, at configuration
time.** That is the cheap runtime check that the ceiling collection really is
nodal and really does contain the coarse space -- a modal or non-nested
collection fails it loudly instead of silently producing a different
discretisation. `SetTraceOrders()` already checks its degree coincidence this
way and the precedent is the right one.

### 4.3 The prolongation, and its restriction

    const Operator      *GetTraceProlongation() const;       // ctdof -> vdof
    const SparseMatrix  *GetTraceProlongationMatrix() const; // serial
    HypreParMatrix      *GetParTraceProlongation() const;    // parallel
    void ProlongTrace(const Vector &X_c, Vector &x_vdof) const;
    void RestrictTrace(const Vector &x_vdof, Vector &X_c) const;

`GetTraceProlongation()` returns **exactly what the eleven sites read today**
when `tr_order` is empty -- the same pointer, including null when it would be
the identity -- so the null path is unchanged by construction rather than by
test.

`RestrictTrace` is `R` composed with the space's own restriction, **not** the
transpose of the prolongation. That distinction is the same one MFEM already
draws between `cP` and `cR`, and it is what makes the boundary datum
ceiling-independent: `R` interpolates at the coarse nodes, so restricting the
ceiling interpolant of `g` gives the coarse interpolant of `g` whatever the
ceiling is. A least-squares pseudo-inverse would not have that property, and
that is why `E^+` is the wrong choice here even though it is also a left
inverse.

Building `P_E`: for each face owning constrained dofs, for each field, emit
`E`'s entries at rows given by the face's ceiling true dofs and columns
`ctr_offsets[f] + k*nt(p_f) + i`. Faces at `p_f == p_max` emit an identity
block. Negative vdofs carry a sign, as `RetireSurplusTraceDofs()` already has
to handle.

Serial: the vdof-to-tdof lookup comes from `GetConformingRestriction()` (one
entry per row) or is the identity. Parallel: `GetLocalTDofNumber()`, `-1`
meaning "not mine, skip". Then `OperatorHandle::MakeRectangularBlockDiag()`
wraps the local `P_E` -- the same idiom `ComputeParH()` already uses for
`MakeSquareBlockDiag()` -- and `mfem::ParMult(Dof_TrueDof, P_E)` composes it.
Column starts by a scan over `GetTraceTrueVSize()`.

Cached; cleared by `SetTraceOrders()` and by anything that invalidates the
space.

### 4.4 The essential dofs

`ess_tdof_user` keeps its job -- the caller's list, in ceiling true dofs -- and
`ess_tdof_list` becomes the same list mapped into the constrained numbering.
The map is per face: a face's constrained dofs are essential iff its ceiling
dofs are, which is again all-or-nothing. `RetireSurplusTraceDofs()` goes
entirely, and with it the surplus that made `ess_tdof_user` necessary in the
first place; the member stays because the mapping has to be redone when the
degrees change.

Five sites read `ess_tdof_list` (`darcyhybridization.cpp:2013`, `2042-2044`,
`4013`, `4038-4043`, and `EliminateTraceTrueDofs()` from `Finalize()`); all are
already at trace-true-dof level, so they need no change beyond the numbering
they are handed.

## 5. The sites

Eight regions in `fem/darcy/darcyhybridization.cpp` read a trace prolongation,
and each becomes a read of §4.3 instead:

| line | what |
|---|---|
| 289 | `TraceVDofsToTDofs()`, the boolean transpose behind `SetEssentialVDofs()` |
| 1839-1846 | the serial reduction, `RAP(cP, H, cP)` |
| 1873 | the parallel reduction, `pP.ConvertFrom(Dof_TrueDof_Matrix())` |
| 2338-2356 | the matrix-free trace operator, prolong |
| 2430-2440 | the matrix-free trace operator, restrict |
| 3284-3300 | `ReduceRHS()`, sizing the reduced right-hand side |
| 3406-3425 | `ReduceRHS()`, the transpose |
| 3490-3504 | `ComputeSolution()`, prolonging before recovery |

Three in `miniapps/hdg/darcyop.cpp`: 566 (reduce the RHS -- and the parallel
branch above it, which currently calls `ParLinearForm::ParallelAssemble()` and
must instead apply the composed transpose), 677 and 682 (prolong the solution).
The `cR` read at 572 becomes `RestrictTrace()`.

Twelve `TraceVDofs()` sites (614, 639, 654, 1750, 1815, 2131, 2328, 3399,
3456, 3568, 3738, 3799) are untouched by **shape A** and are the whole of
**shape B**.

## 6. What goes away

* `RetireSurplusTraceDofs()`, and its three refusals: the shared face, the
  coarsened boundary datum, and -- in `FaceOrdersFromElementOrders()` -- the
  hanging-node family held at the ceiling.
* The `MFEM_VERIFY` guarding `SubDofOrder()` in `AssembleNCSlaveFaceMatrix()`.
  Under shape A the transfer matrix is built between two ceiling elements, so
  the permutation is at the collection's own degree and the overrun that
  presented as `malloc(): unaligned tcache chunk detected` cannot arise.
  Under shape B it is built between two coarse elements again and the guard
  comes back -- or, better, shape B routes the transfer through the ceiling as
  `R_s . I_hi . E_m`, which is the same coarse transfer computed where the
  ordering is right. **Decide that when shape B is written, not before.**
* `HDGErrorEstimator::SetHybridization()`'s *basis* role. The ceiling element
  becomes the right element to read, so the estimator needs no hybridization to
  read the trace correctly. What it still needs the degrees for is the two
  physical flags that method implies -- a face richer than its element is still
  a real thing -- so the method survives as the driver-side helper for those,
  which is separately on the list as "the one part of this branch that got
  worse rather than better". **Re-measure both flags after the port**; their
  measurements were taken on the retired route.
* The `-pref` refuses `-rec` guard, if `DarcyForm`'s six reconstruction sites
  are then correct -- which they should be, since they read a genuine
  ceiling-basis trace. Check, do not assume.
* The `SetTraceOrders()` contract could relax under shape A (`C`, `E`, `G`, `H`
  no longer depend on the degrees, so it need not rebuild them or call
  `Reset()`). **Do not relax it**: shape B needs it back, and a contract that
  tightens again later is worse than one that never loosened.

## 7. Risks, and what settles each

| risk | settles it |
|---|---|
| shape A costs more than it saves at a large ceiling | the count in §3 predicts 4.0x local against at most 1.19x saved; time the demonstrator at `-pmax 7` and at `-pmax 3` before deciding shape B is needed |
| the three `p1_o2_dg_hb_pref*` references move | they were generated in **this** build (`GMRES+UMFPack`, 1 iteration), so the iteration count cannot drift and only the L2 errors are compared, at 1e-4. The real check is tighter: the errors must agree to 8 figures, not 4. If they do not, it is a bug in the port, not arithmetic |
| a nonlinear problem's face residual genuinely changes | it does, wherever the quadrature rule changes with the trace degree, and shape A raises that rule. The identity is pointwise, so a *fixed* rule gives identical answers; the difference is entirely "which rule", and shape A's is the richer one. Pin it by running a `-nlu` case both ways |
| the coarse element is not nested in the ceiling | the `R E == I` check at configuration time (§4.2), which fails loudly instead of silently discretising something else |
| a hanging-node family with members at different degrees | only the master carries an unknown, so the family's degree is the master's; the slaves' local blocks (shape B) must be at least it. Keep `FaceOrdersFromElementOrders()`'s one-degree-per-family rule, but for this reason rather than the ceiling one |
| the two ranks of a shared face disagree on `p_f` | keep the face-neighbour exchange. Under shape A only the owner builds `E`, so a disagreement would be harmless; under shape B the local blocks are sized per rank and it would not be |

## 8. Order of work

Each step is a commit and each has something that fails if it is wrong.

1. **`R`, and the identity test extended.** `R E == I` added to the existing
   case, over the same degrees and gaps. Nothing else changes; the suite must
   be unmoved but for the new assertions.
2. **The numbering and the two matrices**, with the configuration-time check.
   Still nothing reads them. A unit test builds `ctr_offsets` on a mixed-degree
   mesh and checks it against an independent count -- the same shape as the
   parallel shared-face test that already exists.
3. **`P_E` and the composed prolongation**, serial. Test: with `tr_order`
   empty, `GetTraceProlongation()` returns the same pointer the space does.
   With degrees set, `Pi^T H_ceiling Pi` equals the matrix assembled at `p_f`
   directly, to round-off -- this is the acceptance test for the whole
   redesign and it is buildable before anything is ported.
4. **Shape A, serial**: `TraceFE()` to the ceiling, the eight regions onto
   §4.3, the essential map, `RetireSurplusTraceDofs()` deleted, `darcyop.cpp`.
   Acceptance: the three `pref` references to 8 figures; the `[PAdapt]` null
   tests unchanged; the hanging-node refusal lifted and the 0.284 / 0.118 /
   0.091 sequence back.
5. **The boundary datum.** `RestrictTrace()` in the miniapp's essential path.
   Acceptance: the ceiling sweep that measured 0.0124, 0.0926, 0.196, 0.259
   returns 0.0124 at every ceiling -- the 21x turned round, as a test.
6. **Parallel.** `MakeRectangularBlockDiag` + `ParMult`, the `ParallelAssemble`
   site. Acceptance: the constrained dof count is the same at 1, 2 and 3 ranks
   (it is 144, 152, 162 today), and `pconvdiff --p-refine` gives the same
   answer at 1, 2, 3 and 4 ranks -- which is the acceptance item the main plan
   has never been able to run.
7. **The estimator and the reconstruction.** Drop the basis role of
   `SetHybridization()`, re-measure the two flags, lift the `-rec` refusal if
   it is genuinely lifted.
8. **Measure the demonstrator**, and only then decide on shape B.
9. **Shape B**, if step 8 says so: twelve sites, validated against step 4-7's
   answers to round-off rather than against an argument.

## 9. Acceptance for the whole

1. **Byte-identical on the uniform path.** Every reference that does not set a
   per-face degree, and the `[PAdapt]` null tests. Not "to tolerance" --
   `GetTraceProlongation()` returns the same pointer, so the arithmetic is the
   same arithmetic.
2. **Eight figures on the `p`-refined path.** The two routes are the same
   discretisation, so anything looser is a bug in the port rather than a
   difference of method.
3. `make hp-acceptance` still passes and its ratios do not worsen.
4. **The three refusals become capabilities**, each with the measurement that
   currently justifies the refusal turned round (steps 4, 5, 6 above).
5. **Rank-count independence** of `pconvdiff --p-refine` at 1, 2, 3, 4 ranks.
6. Serial unit 512+, serial regressions 2 / 134 with 49 skipped, parallel unit
   85+, parallel regressions 15 / 98.
7. A time for the demonstrator at both ceilings, whichever way it comes out.

## 10. What this is not

It is **not** the other route -- one variant per entity inside
`FiniteElementSpace`, upstream's `hpfem-var-order-space`. That would make the
trace space genuinely minimal, storing `nt(p_f)` per face rather than
`nt(p_max)`, and would mean building a variable-order space's nonconforming
prolongation and its parallel dof-to-true-dof map by hand. This keeps the
ceiling's storage for the *vector* and constrains it, which is the contained
route's whole bargain -- and note what shape B does to that bargain: the local
blocks follow `p_f`, the ldof vector follows `p_max`, and the reduced system
now follows `p_f` as well, which is one better than today.
