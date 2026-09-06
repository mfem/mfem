# NPC on a **linear** `DarcyForm` cannot run: two segfaults and a wrong answer

**From gffp, 2026-09-06.** A defect report, not a request. Branch:
`gf-hdg-linearise-first`. Line numbers are that branch's.

Three defects, all in the same place: **`IsNonlinear()` gates things NPC needs**,
so a `DarcyForm` with no nonlinear integrator cannot use the NPC pathway. The
first two are null-pointer segfaults with nothing in the stack naming the
cause. The third is still open and is the reason this report exists rather than
a patch.

## 0. Why a linear form wants NPC at all

**Any DAE integrator needs it.** gffp drives an HDG convection-diffusion
operator as
`R(t, y, y') = 0` for SUNDIALS IDA, whose unknown is the whole `(q, u, lambda)`
vector. `DarcyHybridization`'s condensation route solves a local nonlinear
problem per element and then iterates on the trace alone, so **the vector it
iterates on is not the vector the integrator integrates** and there is no
residual to hand IDA. NPC is the only route with a residual and a gradient over
the full state.

IDA also evaluates `R` at states no solve has produced — predictor states,
finite-difference probes, consistent-IC iterations. So a **linear** problem
needs the full NPC residual exactly as much as a nonlinear one does.

**The suite already has a test named for this case, and it does not cover it.**
`test_darcy_npc.cpp:1575`, *"One NPC step is exact on a linear problem"*, runs
`PedestalHDG P(8, 1, 0.05, 0.0)` — `amp = 0`, and the fixture's own comment
says it "makes the whole problem linear without changing anything". That is
true of the **problem** and not of the **form**: the fixture still populates
`GetPotentialMassNonlinearForm()` (`:1208`), so `m_nlfi_p` is non-null,
`IsNonlinear()` is true, and every gate below passes. The distinction the three
defects turn on is *structural* — whether the `DarcyForm` carries a nonlinear
integrator at all — not whether the mathematics is linear. **No test on the
branch builds a `DarcyForm` with no nonlinear integrator and asks NPC for a
residual**, which is why all three survive a green suite. The parallel twin at
`:2084` has the same gap for the same reason.

## 1. `Finalize()` leaves the local blocks factored, with no backup

`darcyhybridization.cpp:2434`

```cpp
if (!IsNonlinear())
{
   ComputeH(ComputeHMode::Linear, H);   // factors local A and D IN PLACE
   ...
}
```

`ComputeH()` factors each element's `A` and `D` in place and keeps no copy —
reasonable, when the only thing that will ever be asked of the hybridization is
one reduced solve. NPC's contract is the other one: it evaluates the residual
and the gradient at **arbitrary states**, so it needs the blocks and not their
factorisations, and reads them from `Af_lin_data` / `Df_lin_data`
(`LocalNLOperator::AddMultA`, `:4881`; `AddMultDE`, `:4898`), which the linear
route leaves empty.

**Symptom:** null-pointer `DenseMatrix` inside `DenseMatrix::AddMult` ←
`LocalNLOperator::AddMultA` ← `NPCResidual`.

## 2. The element-wise `H` is never allocated, and never assembled

Three gates, all `IsNonlinear()`:

| where | what it gates |
|---|---|
| `:254` | `AllocH()` in `Init()` — so `H_data` is never allocated |
| `:493` | interior face: `H` into `H_data` (NPC) **vs** into the global sparse `H` |
| `:595` | boundary face: the same choice |

The second and third are not merely allocation. They decide **where the face
`H` goes**: element-wise into `H_data`, which NPC reads through
`GetHFaceMatrix` (`:1853`, `H.Reset(&H_data[H_offsets[f]], ...)`), or straight
into the global sparse `H`, which only the reduced solve reads. A linear form
takes the second branch, so `H_data` stays empty even if allocated, and
`GetHFaceMatrix` returns a `DenseMatrix` over a null pointer.

**Symptom:** `memory access violation at address: 0x0` inside
`DarcyHybridization::MultNL`'s trace row, at `H.AddMult(x_f, y_l)`. It only
appears once §1 is fixed, and only when the face potential integrator makes `H`
non-zero — with a centred convection flux `H` is zero and nothing faults.

`MFEM_ASSERT` does not help here: a release build (`MFEM_DEBUG=NO`) compiles
every one of them out, which is the normal configuration for a consumer.

## 3. The minimal patch, which is necessary and **not sufficient**

What gffp applied locally to get past §1 and §2 — offered as a description of
the shape, not as a proposed commit:

* an opt-in `DarcyHybridization::EnableNPC()` setting a `bnpc` flag, with
  `NPCEnabled()` returning `bnpc || IsNonlinear()`;
* `Finalize()`'s linear gate becomes `if (!IsNonlinear() && !bnpc)`, so a
  linear form with the flag is finalized as a fully nonlinear one — local
  blocks kept, no reduced `H`, which is correct because `NPCGradient()`
  assembles its own;
* the `PotNL` and `FluxNL` sub-branches gain an `IsNonlinear() &&` conjunct.
  Each leaves one of `Af_lin_data` / `Df_lin_data` empty on the guarantee that
  the corresponding nonlinear integrator supplies that block — a guarantee
  `IsNonlinear()` provided and the new flag does not — so a linear form must
  fall through to `FullNL`;
* the three gates in §2 become `NPCEnabled()`;
* `EnableNPC()` allocates `H` itself when `Init()` has already run, because it
  normally *has*: the hybridization does not exist until
  `DarcyForm::EnableHybridization` has made it, and that is what calls `Init()`.
  `Ct_data.Size()` is `Init()`'s own "already run" test and serves as the
  signal.

Opt-in rather than unconditional because the kept blocks are the largest thing
the hybridization owns, and no reduced-route caller should pay for them.

**With all of that, NPC runs and is self-consistent — and still gets a
different answer from condensation.** One exact Newton step from zero on a
linear problem drives `|R|` to `2.2e-15`, so the residual and the gradient
agree with each other; the resulting field does not agree with the condensation
solve of the same `DarcyForm`.

### The bisection

Identical mesh, spaces, `tau` and coefficients; one variable changed. HDG
convection-diffusion on 128 wedges, order 1, `kappa = 1e-3`, periodic in the
streaming direction. `L2` error of the potential against the same manufactured
solution, by each route:

| face potential integrator | `H` | condensation | NPC | agree? |
|---|---|---|---|---|
| none (`velocity = nullptr`) | — | 113.688 | 113.688 | **yes** |
| `HDGConvectionCenteredIntegrator` | zero | 0.356519 | 0.356519 | **yes** |
| `HDGConvectionUpwindedIntegrator` | non-zero | 0.360644 | **11.7979** | **no** |

(The 113.688 row is a large error by construction — the manufactured source is
built for the advection-diffusion equation, so with the velocity removed
neither route is solving the problem that solution belongs to. The point of the
row is that the two routes agree to six figures.)

**So the disagreement tracks `H` exactly.** It is not the boundary integrator:
the drift is `(0, 0, w)` with the streaming direction periodic, so `v·n = 0` on
every boundary face and the boundary term contributes nothing. It is the
interior upwinded face term, which is what makes the trace–trace block
non-zero.

### A positive control: the same march reaches design order with `H` zero

The steady bisection above is a comparison of two routes. This is a comparison
of one route against a **closed form**, and it isolates the defect from
everything else in the stack.

Free streaming `d_t u + w d_l u = 0` on a mesh periodic in `l`, integrated to
`t = 0.25` with SUNDIALS IDA over the full `(q, u, lambda)` state, against the
exact translate `u_0(w, mu, l - w t)`. Identical meshes, tolerances
(`1e-11 / 1e-13`), stabilisation and step count; the ONLY difference is the
face potential integrator, and therefore whether `H` is zero:

| flux | `H` | `k` | `L2` errors, `n = 3, 6, 12` | observed rate |
|---|---|---|---|---|
| centred | zero | 1 | 0.510, 0.148, 0.0400 | 1.78, **1.89** |
| centred | zero | 2 | 0.0975, 0.0144, 0.00179 | 2.76, **3.02** |
| upwinded | non-zero | 1 | — | `IDACalcIC` fails at `n = 3` |

**So the time integration, the consistent initialisation, the tolerances, the
mesh and the spatial discretisation are all sound**: with `H` zero the same
code reaches `k+1` in both degrees. With `H` non-zero, `IDACalcIC` cannot even
make the algebraic blocks consistent with the potential — which is the expected
place for it to surface first, since that is precisely a solve of the flux and
trace rows against a fixed potential.

That is the control the bisection lacked, and it is why this is reported as a
defect in the `H` path rather than as gffp holding NPC wrongly.

**The upwinded failure is not uniform, and that is worth knowing before you
reproduce it.** A separate, coarser check on the same problem -- that the
solution translates in `+l` for `w > 0` -- passes with upwinding at `n = 8`,
agreeing with the exact translate to 1.2%. So an upwinded run can complete and
look plausible. It is not evidence against the defect: that check has a 5%
tolerance and cannot separate a wrong field from a right one, whereas the
steady parity at `n = 4` is 11.80 against 0.36. Expect the symptom to appear as
a failed `IDACalcIC` or a wrong answer depending on the mesh, rather than
always as a crash.

**What is not yet established** is whether the remaining discrepancy is a
fourth defect in the linear `c_bfi_p` path through NPC, or an incompleteness in
§3's patch — the patch is exactly what makes `H` reach NPC, so it is a
candidate for its own symptom. Ruled out so far:

* **not double-counted assembly.** `ComputeAndAssemblePotFaceMatrix` is called
  once per interior face (`darcyform.cpp:2413`, a loop over faces skipping
  non-interior ones), so `CopyMN` into `H_data[H_offsets[face]]` and
  `H->AddSubMatrix` are equivalent in multiplicity;
* **not the trace row applying `H` twice.** `MultNL` guards it with
  `if (el1 == el)`, so once per face across the two element visits;
* **not a right-hand-side convention.** Both routes negate the potential RHS
  under `bsym` — `MultNL` at `:2035`, the condensation path at `:3633` and
  `:4125`;
* **not the `bsym` sign of the potential mass block.** Measured rather than
  assumed: differencing the NPC residual between two assemblies that differ
  only in a mass coefficient gives `+M u` exactly, and the agreeing rows above
  use the same measurement.

A worthwhile next probe, which needs the branch's own harness rather than
gffp's: assemble `NPCGradient`'s reduced trace operator and `ComputeH()`'s `H`
on the same upwinded form and difference them. If they differ, the reduced
operator is the place; if they agree, it is the residual.

## 4. Related: the two routes are mutually exclusive on one assembly

`DarcyForm::FormLinearSystem` aborts on a form with NPC enabled, because
`Finalize()` skipped `ComputeH()` and there is no reduced `H` to hand back. That
is defensible — they are different methods — but it is worth stating on
`EnableNPC()`, because it forecloses the obvious way to compare the two routes:
solving both from a single assembly. The bisection in §3 therefore uses two
assemblies of the same problem, which is weaker evidence than it needed to be.

## 5. Suggested test

A regression that would have caught §1 and §2 needs no new physics, and the
existing fixture is nearly it. `PedestalHDG` with `amp = 0` is already
mathematically linear; what it needs is a variant that puts the same face terms
on the **linear** `GetPotentialMassForm()` instead of
`GetPotentialMassNonlinearForm()`, so that `IsNonlinear()` is false. Then
require the NPC solve to match the condensation solve of the same form.

With a face potential integrator that makes `H` non-zero — upwinded convection
is the natural one — the same test would also catch §3.

Two properties are worth asserting separately, because they fail separately:
that NPC **runs** on such a form (§1, §2 are crashes), and that it gets the
**same answer** as condensation (§3 is a wrong answer with a clean residual).

---

# Reply, 2026-09-06

**Sections 1 and 2 accepted and fixed. Section 3 does not reproduce here, and
the probe you suggested rules out the operator.** Everything below was
reproduced before anything was changed.

## 1. Reproduced, with your stack

A `DarcyForm` whose HDG face terms sit on the linear `GetPotentialMassForm()`
segfaults in `NPCResidual`, with exactly the chain you name:

```
DenseMatrix::AddMult <- LocalNLOperator::AddMultA <- LocalNLOperator::Mult
  <- LocalResidual <- MultNL <- NPCResidual
```

`Af_lin_data` is populated at exactly one site, inside the `else` branch the
linear route never reaches, so the read is of an empty array. Your section 0 is
right about the test gap too: `PedestalHDG` populates
`GetPotentialMassNonlinearForm()` unconditionally, so `m_nlfi_p` is non-null
and `IsNonlinear()` is true even at `amp = 0`. The distinction really is
structural rather than mathematical.

## 2. The fix is yours, and there were ten gates rather than three

`EnableNPC()` / `NPCEnabled()`, `Finalize()`'s gate widened, the three `H`
gates widened, and `IsNonlinear() &&` added to the `PotNL` and `FluxNL`
conditions so a form with no nonlinear integrator falls through to `FullNL` --
the only branch that backs up *both* `Af_lin_data` and `Df_lin_data`. Your
reasoning for that last point is what we implemented; it is right.

**Seven more gates needed the same widening**, and one of them is not
cosmetic:

| site | why |
|---|---|
| `ReduceRHS` | **fills `darcy_rhs`.** Gated out, `MultNL` gets an unsized `BlockVector` and corrupts the heap |
| `EliminateVDofsInRHS`, `EliminateTrueDofsInRHS` | save the fields for the local solve |
| `EliminateTraceTrueDofs`, `EliminateTraceTrueDofsInRHS` | there is no assembled `H` to eliminate from |
| `ComputeSolution` | dispatches to the nonlinear path |
| `ComputeH`'s assert | `Linear` mode must now also refuse a `bnpc` form |

`ReduceRHS` is worth calling out because it fails *after* your two, so a patch
carrying only sections 1 and 2 gets a heap corruption whose stack names
neither. With all ten, NPC runs and one Newton step is exact on a linear
problem: `|R|` 2.15 -> 1.7e-14, zero local nonlinear iterations.

## 3. Your suggested probe, run: the operator is not the place

You asked for `NPCGradient`'s reduced trace operator and `ComputeH()`'s `H`
assembled on the same upwinded form and differenced. Done, on 2-D triangles
with an upwinded convection term making `H` non-zero:

```
max |S - H| over all NON-essential rows and columns  =  0.000000e+00
max |S - H| on essential rows/columns                =  3.8e+00
```

Bit-identical, with and without the upwinded term. The essential-row
difference is the two routes' conventions -- one eliminates, the other writes a
unit row -- and is not a disagreement about the discretisation. By your own
criterion, that puts it in the residual.

## 4. And in the residual it was a right-hand side that reached one route only

**`DarcyForm` does not fold `GetPotentialRHS()` into the block `b` on the
hybridized path.** Not into `b`, and not into the reduced vector either: with a
homogeneous Dirichlet datum and a live source, the condensation route's reduced
right-hand side comes back **exactly zero**. The caller has to add the
potential load to `b` itself, and if it reaches one route and not the other the
two disagree by precisely that load.

That was our own driver's bug, and it produced a disagreement that looked
structural: the residual at the condensation solution was exactly proportional
to the source (6.53e-2, 1.31e-1, 2.61e-1 at source 1, 2, 4) and vanished at
source 0. With both routes given the same load:

| face potential integrator | `H` | reduced RHS | end to end |
|---|---|---|---|
| `HDGDiffusionIntegrator` only | zero | rel 1.0e-15 | rel 9.5e-15 |
| plus `HDGConvectionUpwindedIntegrator` | non-zero | rel 1.0e-15 | rel 7.4e-15 |

and the condensation solution is a root of the NPC residual in every block
(2.2e-16 / 1.3e-14 / 1.8e-14).

**So we cannot reproduce section 3.** That is not a claim it is not real --
your case is 3-D wedges and periodic and ours is 2-D triangles -- but the
"fourth defect in the linear `c_bfi_p` path" hypothesis is dead, because the
operator is bit-identical. The load path is the first thing worth checking at
your end: if your two assemblies get the potential load by different routes,
the disagreement is exactly that.

One diagnostic that separates them cheaply, and which is what we should have
run first: evaluate `NPCResidual` **at the condensation solution**. If the flux
and potential blocks are round-off and only the trace block is not, the
operator is fine and it is the reduced right-hand side. If a local block is
non-zero, it is the local rows. Sweeping the source amplitude then says
immediately whether it is the load -- ours was exactly linear in it.

## 5. Your section 4 and section 5

Section 4 is documented on `EnableNPC()`: the routes are mutually exclusive on
one assembly, so a comparison needs two. Section 5 is now two test cases in
`tests/unit/fem/test_darcy_npc.cpp`, asserted separately as you asked -- that
NPC **runs** on such a form, and that it **agrees** with condensation, the
latter stated as the condensation answer being a root of the NPC residual.
Both would have crashed rather than failed before the fix; `MFEM_ABORT` aborts
rather than throws in this build, so a "must refuse" case cannot be a unit test
here either way.
