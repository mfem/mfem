# Constrain the surplus instead of retiring it

Scratch, like everything in `doc/`. It goes when the work is done and its
findings are in the code.

Three things this branch refuses or works around are one thing. This is what
that thing is, why the repair cannot change any answer, and what it costs.

## The one cause

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
`p`-adaptivity does not run on more than one rank at all.

## The repair

Store the same function as its **ceiling-basis** coefficients, constrained to
the degree-`p_f` subspace. The constraint is a per-face matrix `E_f`, whose
columns are the coarse basis functions written in the fine one:

    E_f(j, i) = phi_i^{p_f}( node_j^{p_max} )

so a face's `nt(p_max)` slots are `E_f c` for `nt(p_f)` free unknowns `c`. That
is a prolongation, and it composes with the ones already there:

    x_vdofs  =  cP . P_p . c          serial, nonconforming
    x_ldofs  =  Dof_TrueDof . P_p . c parallel

Every reader that assumes the ceiling basis is then right, because the stored
coefficients *are* ceiling-basis coefficients. Orientation stops mattering:
the ceiling basis is the space's own, so its dof ordering, its prolongation and
its true-dof numbering all apply unchanged.

## Why it cannot change any answer, and this is measured

A degree-`p_f` polynomial *is* a degree-`p_max` polynomial, so
`phi_i^{p_f} = sum_j E_f(j,i) phi_j^{p_max}` exactly, and therefore the face
matrix assembled against the coarse trace equals the one assembled against the
ceiling trace restricted by `E_f`. The two routes are the same discretisation
in two bases.

That is an argument, so it is also a test:
`"A coarse trace basis is an exact combination of the ceiling's"` in
`tests/unit/fem/test_darcy_padapt.cpp` assembles
`NormalTraceJumpIntegrator` both ways on a real interior face and checks
`M_hi E_f == M_lo` to `1e-12` relative, over degrees 0, 1, 2 and gaps of 1, 2,
3. It also checks `E_f^T E_f` is invertible, which is the other half: the
constrained face carries exactly `nt(p_f)` unknowns, neither fewer -- which
would lose the function -- nor more.

**The dof count does not move either.** The retired route's *active* count is
already `nt(p_f)` per face; the constrained route makes that the true count
rather than a count with holes in it.

## The work

Eleven sites read a trace prolongation and each needs `P_p` composed in.

`fem/darcy/darcyhybridization.cpp`:

| line | what |
|---|---|
| 289 | `TraceVDofsToTDofs()`, the boolean transpose that maps vdofs to tdofs |
| 1715 | the serial reduction, `RAP(cP, H, cP)` |
| 1873 | the parallel reduction, `pP.ConvertFrom(Dof_TrueDof_Matrix())` |
| 2356, 2439 | the matrix-free trace operator, both directions |
| 3284, 3421 | the two local-recovery paths |
| 3504 | the solution prolongation before recovery |

`miniapps/hdg/darcyop.cpp`: 566 (reduce the RHS), 677 and 682 (prolong the
solution).

And within the hybridization:

* `RetireSurplusTraceDofs()` **goes**, with its three refusals -- the shared
  face, the coarsened boundary datum, and (in
  `FaceOrdersFromElementOrders()`) the hanging-node family at the ceiling.
* `TraceVDofs()` returns the face's whole slot list again; the truncation moves
  into `P_p`.
* `TraceFE()` keeps its meaning for sizing the LOCAL blocks, which still follow
  `p_f` -- that is what makes the route cheap and it is unaffected.
* `HDGErrorEstimator::SetHybridization()` and the two options it implies can go
  too, since a face no longer outruns its element in any basis the estimator
  sees. **Check that rather than assume it**: the enriched-face terms were real
  and the measurements are on `SetSkipEnrichedDirection()`.

## Acceptance

1. **Bit-for-bit on everything that works today.** Every `p1_o2_dg_hb_pref*`
   reference, and the `[PAdapt]` null tests, must be unchanged -- the two
   routes are the same discretisation, so anything else is a bug in the port,
   not a difference of method.
2. `make hp-acceptance` still passes, and its ratios do not worsen.
3. **The three refusals become capabilities**, each with the measurement that
   currently justifies the refusal turned round: a hanging-node family below
   the ceiling gives 0.0818 rather than 1.06; a coarsened boundary face with an
   essential datum reproduces the ceiling-2 answer at every ceiling; and the
   retired-dof count is 144 at one, two and three ranks.
4. **Rank-count independence** of `pconvdiff --p-refine` at 1, 2, 3, 4 ranks,
   which is acceptance item 4 of the main plan and cannot be run at all today.
5. Serial unit 511+, serial regressions 2 / 134 with 49 skipped, parallel unit
   85+, parallel regressions 15 / 98.

## What this is not

It is **not** the other route -- one variant per entity inside
`FiniteElementSpace`, upstream's `hpfem-var-order-space`. That would make the
trace space genuinely minimal, storing `nt(p_f)` per face rather than
`nt(p_max)`. This keeps the ceiling's storage and constrains it, which is the
contained route's whole bargain: row count and trace-vector length stay
`O(p_max)` per face while the local blocks and now the true dofs follow `p_f`.
Whether the ceiling's storage is worth paying is already measured -- at a 2.67x
storage ratio the solve costs 1.03 to 1.19x and peak RSS at most 1.15x -- so it
is, and this is the right route to finish rather than to replace.
