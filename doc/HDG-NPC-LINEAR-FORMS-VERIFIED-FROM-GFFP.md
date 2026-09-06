# The ten-gate NPC fix: two defects gone, one still there, and now localised

**From gffp, 2026-09-06.** A verification report against `meq-integration` at
`7004a19d51`, which merges `gf-hdg-linearise-first` carrying `0680889329`
("NPC could not run on a linear DarcyForm: ten gates, not the three reported")
and `sundials-ida-integration` carrying the `IDA` component.

Thank you for the seven gates we missed, and for `ReduceRHS` in particular —
the note that it "fails AFTER the two reported, so a patch carrying only those
gets an abort whose stack names neither" describes exactly the state gffp was
in, and explains why our local three-gate scaffolding was never going to be
enough.

## What is verified fixed

**The crashes are gone.** gffp calls `EnableNPC()` on a `DarcyForm` carrying no
nonlinear integrator and then drives it as an IDA residual — the case that
segfaulted twice, first in `LocalNLOperator::AddMultA` on an empty
`Af_lin_data` and then at `0x0` in `MultNL`'s trace row on an unallocated
element-wise `H`. Against the ten-gate build the whole gffp suite constructs,
assembles, takes gradients and marches with no fault of any kind.

**The IDA linkage is fixed**, as already retired: `-lsundials_ida` is in the
install's link line alongside `-lsundials_idas`, additively, exactly as
reported.

**Verification hygiene, because a stale install would have faked this.** The
install was rebuilt from `7004a19d51` and checked header-by-header against the
branch (`darcyhybridization.hpp`, `darcyform.hpp`, `sundials.hpp` all current),
and gffp was rebuilt **from clean** afterwards, since `bnpc` changes the class
layout. Numbers below are from that build.

## What is still there

**§3 of the original report — the disagreement between the NPC and
condensation routes — survives the complete fix, with bit-identical numbers.**

That is itself informative. §3 named two candidate causes: a fourth defect, or
an incompleteness in our own three-gate patch, since that patch was what made
`H` reach NPC at all. **The second is now excluded**: the numbers do not move
between a three-gate and a ten-gate build.

One HDG convection-diffusion problem, 128 wedges, order 1, `kappa = 1e-3`,
periodic in the streaming direction. One assembly per route (they are mutually
exclusive on a single one, as your commit notes). `L2` error of the potential
against the same manufactured solution:

| face potential integrator | velocity | condensation | NPC | relative gap |
|---|---|---|---|---|
| none | — | 113.688 | 113.688 | ~1e-12 |
| `HDGConvectionCentered` | `( 0, 0, w )` | 0.356519 | 0.356519 | ~1e-12 |
| `HDGConvectionUpwinded` | `( 0, 0, 1 )` | 3.00482 | **3.00475** | **2.1e-5** |
| `HDGConvectionUpwinded` | `( 0, 0, w )` | 0.360644 | **11.7979** | **32** |

**Both upwinded rows disagree.** One Newton step drives the NPC residual to
`1e-16` on every row, so the rows that genuinely agree do so to about `1e-12`.
The uniformly-signed upwinded row is out by `2.1e-5` — seven orders above that
floor, small enough to read as convergence noise and far too large to be it.

**What the fourth row buys is the localisation.** gffp's drift is `( 0, 0, w )`,
so on the `l` faces the upwind side is `sign( w )` and **changes across the
mesh**; a constant `( 0, 0, 1 )` fixes it everywhere. Varying the sign turns a
subtle wrong answer into an obvious one, but does not create it.

**Not the boundary term.** With `( 0, 0, w )` and `l` periodic, `v.n = 0` on
every boundary face, so the boundary constraint integrator contributes nothing.

**A correction to our first report, which we got wrong.** §3 said the
disagreement "tracks `H` exactly". We never measured `H` — that was inferred
from which configurations agreed, and the inference does not survive this
round: your new tests put `HDGDiffusionIntegrator` on the interior and boundary
faces, which also populates the trace-trace block, and they pass. The measured
discriminator is the **upwinded convection face integrator**, not `H` being
non-zero. We are sorry for sending you after the wrong object.

## The second, independent route to it: IDA cannot initialise

The parity test above compares two solves. This one does not compare anything —
it asks SUNDIALS IDA to compute a consistent initial condition for the DAE whose
residual is the NPC operator, which is what gffp actually needs the pathway for.

```
free streaming to t = 0.25, k = 1, flux centred
  n = 3   err = 0.510016    n = 6  err = 0.148088   n = 12  err = 0.0399538
  rates: 1.78409 1.89004
free streaming to t = 0.25, k = 2, flux centred
  rates: 2.75549 3.01583
free streaming to t = 0.25, k = 1, flux upwinded
  [ERROR] ida_ic.c:728 [IDAICFailFlag] Newton/Linesearch algorithm failed to converge
  --> error in IDACalcIC()
```

**Centred reaches design order in both unknowns; upwinded cannot get past
`IDACalcIC` at the coarsest mesh in the sweep** — `n = 3`, `k = 1`, 1854 dofs.
The two differ in nothing but the face integrator: same mesh, same tolerances,
same operator, same initial state.

Two things worth having from this. **It is cheap to reproduce** — the failure is
at the smallest configuration, not at the end of a refinement study. And **the
failure mode has changed for the better**: before the ten gates this was a
segfault at `0x0`, and it is now a clean non-convergence with a diagnosable
message. That is the crash half of the report being genuinely fixed, with the
residual half showing through underneath it.

`IDACalcIC` solves the algebraic blocks — flux and trace — against the
differential one, so it is a direct probe of the rows the parity test says are
wrong, reached without any comparison to condensation at all.

## Where it is NOT: the integrator itself is excluded

**This is a code reading, not a measurement, and it is labelled so deliberately
— the last thing we inferred rather than measured was wrong.** But it is a
reading of three routines against each other, and it is checkable.

`HDGConvectionUpwindedIntegrator` is reached by three entry points, and the two
routes do not use the same ones. Condensation assembles through
`AssembleHDGFaceMatrix(trace_el, el1, el2, ...)` — `bilininteg_hdg.cpp:484`,
the two-sided form, visiting each interior face **once**. NPC's local residual
goes through `AssembleHDGFaceVector` — `:753`, per element, visiting each
interior face **twice**, once from each side.

Compared block by block, they agree:

| block | two-sided, one visit | one-sided, visit from el1 / from el2 |
|---|---|---|
| elem-elem | `(b+a)` on el1, `(b-a)` on el2 | `(b+a)` / `(b'+a') = (b-a)` |
| constraint | `(b+a)` on el1, `(b-a)` on el2 | `(b+a)` / `(b-a)` |
| trace | `(b-a)` on el1, `(b+a)` on el2 | `-(b-a)` / `-(b'-a') = -(b+a)` |
| face (H) | `2b` once | `(b-a)` + `(b+a)` = `2b` |

using `nor.Neg()` on the second visit, so `un' = -un`, `a' = -a`, `b' = b`. The
boundary special case matches too: the one-sided form's
`(Elem2No >= 0) ? (b-a) : 2b` is `2b` on a boundary face visited once, which is
what the two-sided form writes unconditionally.

**So the integrator is consistent with itself, and the defect is in the caller.**
`bilininteg_hdg.cpp` is not where to look.

One asymmetry in `darcyhybridization.cpp` we noticed and could NOT convict, in
case it is a lead for someone who knows the invariant. `LocalNLOperator::AddMultA`
(`:4912`) and the `D` term of `AddMultDE` (`:4933`) each have a nonlinear branch
and a **linear fallback** — `else if (!dh.A_empty)` at `:4921` and
`else if (!dh.D_empty)` at `:4938`, reading `Af_lin_data` / `Df_lin_data`. The
`E` term of the same function (`:4945`) has only `if (dh.c_nlfi_p)` and no
fallback, **and the gradient side repeats the pattern exactly**: `:5083` and
`:5101` are the two linear fallbacks, `:5107` the unguarded `if (dh.c_nlfi_p)`.

We traced that the linear `E` is supplied instead by `MultNL`'s `bp - E x`
(`:2098`, and `:5243` on the other path), so the term does arrive and this is
not by itself the bug. But the two halves of the potential row now get their
linear contribution from two different places, only one of which sits next to
the code documenting the convention — and the block that goes missing in that
arrangement would be a **face** block, which is the family the failing rows are
in.

## Why the new tests do not catch it

`test_darcy_npc.cpp`'s two new cases are the right shape — a form with no
nonlinear integrator, and a parity check against condensation — and they are
what §5 asked for. What they do not reach is the **integrator**: both use
`HDGDiffusionIntegrator`, which is symmetric. gffp's failing configuration adds
`HDGConvectionUpwindedIntegrator`, which is not, and whose face contribution
depends on the sign of `v.n`.

A row with an upwinded convection face term and a **sign-varying** velocity
would fail today. A row with a uniformly-signed one would fail by `2e-5`, which
is the more dangerous of the two: **a patch validated only on a uniformly-signed
velocity could report that gap as convergence noise and be believed.**

## What gffp is doing meanwhile

The gffp-side parity test carries all four rows above, with the two upwinded
ones red and the reason recorded, so it will report the day this changes.
Upwinding is gffp's production choice — the branch's own comment says a centred
flux with Dirichlet data and hybridization gives a diverging system — so this is
not a configuration we can route around.

---

# Reply, 2026-09-06

**Found, fixed, and it is not the upwinded integrator.** One line in
`DarcyForm::AssemblePotHDGFaces()`. Your localisation was right about the half
that mattered most -- *the defect is in the caller, not in
`bilininteg_hdg.cpp`* -- and the block-by-block reading that got you there
holds up.

## What it is

`Mesh::MakePeriodic` identifies the two ends of the mesh, so the faces there
become **interior**. It does **not** remove the boundary *elements* that sat on
them: `GetNBE()` still counts them, and `GetBdrElementFaceIndex()` hands back
the now-interior face. Every other boundary loop in `darcyform.cpp`, and
`DarcyHybridization::ConstructC` twenty lines away, drops those by calling
`Mesh::GetBdrFaceTransformations()` and skipping the null it returns
(`mesh.cpp:1241` -- `FaceIsTrueInterior(fn) || faces_info[fn].NCFace >= 0`).

**The potential-mass boundary loop did not.** It called
`ComputeAndAssemblePotBdrFaceMatrix()` on those faces, which takes `Elem1No`
only, builds a **one-element** `E`, `G` and `H`, and writes them with
`DenseMatrix::CopyMN` -- which **assigns**. On an interior face the slot holds
**two** elements' blocks and the interior pass has already filled it, so the
boundary write destroyed element 1's half of `E` and `G` and overwrote `H`.

Traced, 2-D triangles, 4x4, periodic in `y`, order 1:

```
[bdr] bface 0 face 3  ndof 3 c_dof 2 slotE 12  |elmat| 0  |Epre| 0.788675
[bdr] bface 4 face 3  ndof 3 c_dof 2 slotE 12  |elmat| 0  |Epre| 0
[bdr] bface 8 face 2  ndof 3 c_dof 2 slotE  6  |elmat| 0  |Epre| 0
```

Faces 3, 7, 11, 15 appear **twice** in the boundary-element list and carry
`slotE 12` -- two elements' worth, i.e. they are interior. `|Epre| 0.788675`
is the live block a moment before it is overwritten with zeros.

## Why it looked like the upwinded integrator, and why that reading was wrong

**The integrator's value is irrelevant. Only its presence matters**, because
the damage is an assignment, not an accumulation.

Your bisection changed two things at once between the agreeing and the
disagreeing rows, and the second is not in your table:
`HDGConvectionDiffusion.cpp:206-221` adds `AddBdrFaceIntegrator` in the
**upwinded** branch and deliberately not in the centred one. So "upwinded"
was a proxy for "has a boundary face integrator on the potential mass form",
and the periodic mesh supplied the other half. Both your upwinded rows fail
for that reason, and the sign structure only scales what gets destroyed --
which is why varying it made an obvious failure out of a subtle one without
creating it.

The control that settles it: a boundary face integrator with a **zero
velocity**, so every block it writes is exactly zero.

| 2-D triangles, periodic, natural BCs | `max|S-H|` | `|u|` cond | `|u|` NPC |
|---|---|---|---|
| no boundary integrator | 0.00e+00 | 2.4833045454 | 2.4833045454 |
| **+ an identically-zero boundary integrator** | **2.00e+00** | **1.9087291089** | **4.2005388795** |
| + zero boundary integrator, after the fix | 0.00e+00 | 2.4833045454 | 2.4833045454 |

An integrator contributing exactly nothing moved the answer by 23% on one
route and 69% on the other. That is not arithmetic, and no reading of
`AssembleHDGFaceMatrix` against `AssembleHDGFaceVector` could have found it.

**It is also not an NPC defect.** The condensation route is corrupted too --
2.4833 to 1.9087 above. The two routes *disagree* only because the `H` block
has different destinations (`H_data` under NPC, the assembled sparse `H`
otherwise), so they are damaged differently; `E` and `G` are shared and both
routes eat that identically. Gating the three writes separately confirmed it:
skipping only the `H` write restored `max|S-H| = 0` while both routes stayed
wrong together.

**So your `IDACalcIC` failure and the parity gap are one defect**, and neither
needed the residual/gradient machinery to explain it.

## Our own claim to withdraw

Our reply of earlier today said the disagreement had to be "the reduced
right-hand side", on the grounds that the flux and potential residual blocks
were round-off and only the trace block was not. The block pattern was right
and the inference from it was wrong: `max|S-H|` is **2.25e+00** in the failing
configuration, so it was the operator all along. Our own suggested diagnostic
pointed away from the cause. We should have measured `S - H` on **your**
configuration rather than reusing a number taken on ours -- ours had essential
trace dofs, which mask this entirely.

## The fix

`fem/darcy/darcyform.cpp`, in `AssemblePotHDGFaces()`'s boundary loop:

```cpp
if (!mesh->GetBdrFaceTransformations(f)) { continue; }
```

Making one routine agree with its neighbours, not a design change. Nothing
non-periodic can reach it: on a mesh with no periodic identification every
boundary element's face is a true boundary face and the guard never fires.

## What this does NOT fix, measured rather than assumed

A **3-D periodic** mesh (hexes and wedges alike) with the natural boundary
route and a boundary integrator on the potential mass form still reduces to an
exactly zero right-hand side and solves to zero. It is unrelated to this
report: both routes agree to `max|S-H| = 0.00e+00` and both return the same
zero, it predates this change, and the 2-D analogue is unaffected. We have not
chased it. If your `IDACalcIC` sweep still misbehaves in 3-D after this fix,
that is the thing to look at and we would want to know.
