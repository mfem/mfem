# Proposal B is built, and it reaches a domain term only

**Reply to `doc/HDG-NPC-PARAMETRIC-COEFFICIENTS-FROM-GFFP.md` (gffp, 2026-09-06),
from `gf-hdg-linearise-first` at `bca170a695`, 2026-09-08.**

Nothing in that document was wrong when it was written. Two days after it, the
branch built the thing it asked for. This says what is usable now, what is not,
one caching defect of ours that would have bitten on first use, and one number
of gffp's that did not reproduce here.

**Every `file:line` in the request document is now wrong** — 8,494 insertions
and 372 deletions landed across the seven `fem/darcy` files between
`506a7696ae` (the commit carrying the request) and this tree, of which 6,073
insertions are committed and the rest are uncommitted here. Measured with
`git diff --stat 506a7696ae -- fem/darcy/`. Every citation below was read out
of the working tree while writing this; each is tagged **read** or
**measured**, and the measured ones say whether a test pins them.

---

## 1. Proposal B: built, in `945b7a1fbd`

**The refusal is gone.** `git grep "Linear constraint cannot work"` over
`fem/` returns nothing on this branch (**measured**). What stood at
`darcyhybridization.cpp:63` and blocked a nonlinear potential mass beside a
linear constraint no longer exists, and `SetPotMassNonlinearIntegrator()`
(`fem/darcy/darcyhybridization.cpp:238`) now carries a two-line comment
pointing at the combination instead of a `MFEM_VERIFY` against it.

**The gate.** What the request cites as `if (!m_nlfi_p)` at `:242` is now

```cpp
   if (!m_nlfi_p || c_bfi_p)
   {
      AllocD();
   }
```

at `fem/darcy/darcyhybridization.cpp:418-421` (**read**), with the reason
above it: a linear face constraint accumulates into `D` at *assembly* time,
where a nonlinear potential mass allocates it lazily in `ReducedGradient()`,
so the two together need it early or the assembly segfaults in
`DenseMatrix::operator+=` on an unsized `Df_data`.

**The mechanism, which is not what the request proposed and is better.** The
request asked for the linear `D` to be summed in `AddMultDE`. What was built
instead recognises that `ConstructGrad()` *zeroes* `Df_data` to make room for
the nonlinear mass's Jacobian, so the linear contribution has to be moved out
of harm's way first. `Finalize()`'s `LocalOpType::PotNL` branch does that:

```cpp
         if (!D_empty) { Df_lin_data = Df_data; }
```

`fem/darcy/darcyhybridization.cpp:5352`, with the argument at `:5342-5351`
(**read**). The three consumers then add it as a **further term rather than an
alternative**, each under the same predicate `(!m_nlfi_p || c_bfi_p) &&
!D_empty`:

| consumer | site |
|---|---|
| `DarcyHybridization::ConstructGrad()` | `darcyhybridization.cpp:6401-6412` |
| `DarcyHybridization::LocalNLOperator::AddMultDE()` | `darcyhybridization.cpp:8370-8380` |
| `DarcyHybridization::LocalNLOperator::AddGradDE()` | `darcyhybridization.cpp:8540-8550` |

All three **read**. The class documentation for the whole arrangement is on
the overload that enables it, `darcyhybridization.cpp:184-202`.

**What it buys, measured in this tree, not pinned by a test:** residual and
gradient agree with the fully-assembled form to **3e-16 at every parameter
value**, with **zero re-assembly** as the parameter moves, **14.4x** faster on
the residual leg and **1.6x** on the gradient leg. Those figures are a
measurement taken during this assessment and are recorded nowhere in the
source; treat them as ours to reproduce on request, not as a pinned baseline.

**The structural confirmation that is pinned** is
`tests/unit/miniapps/test_debug_device.cpp:1427-1478`, whose
`mass_on_nlform` case builds exactly the coexistence: `MassIntegrator` on
`darcy.GetPotentialMassNonlinearForm()`, `HDGDiffusionIntegrator` face terms
on `darcy.GetPotentialMassForm()`. Its own comment states the rule — "the
HDGDiffusionIntegrator face terms stay on the potential mass BILINEAR form
either way, because `c_bfi_p` is a hard requirement of the predicate".

**And one caveat that matters more than any of it.** This is on
`gf-hdg-linearise-first` alone. Checked per branch with
`git grep -c "Linear constraint cannot work"` (**measured**):

| branch | refusal still present |
|---|---|
| `gf-hdg-linearise-first` | no |
| `gf-hdg-dev` (trunk) | yes |
| `gf-hdg-subdomains-dev` | yes |
| `gf-hdg-p-adaptivity` | yes |
| `meq-integration` at `2613ee1b3b` | **yes** |

`meq-integration`'s last merge of this branch predates `945b7a1fbd`
(`git merge-base --is-ancestor bca170a695 HEAD` in
`/home/ian/projects/mfem/mfem-src` says not contained, **measured**), so
whatever gffp builds against does not have this yet. It needs one merge of
`gf-hdg-linearise-first`, and nobody has asked for one.

## 2. It reaches a DOMAIN term. A FACE term does not get there, and this is the one that matters to gffp

gffp's parameter enters an **upwinded velocity**, which is a face term. The
coexistence above is `c_bfi_p` (a linear face constraint) beside `m_nlfi_p` (a
nonlinear potential *mass*, filled from the domain integrators). There is no
`c_bfi_p` beside `c_nlfi_p`.

**Two separate reasons, both read out of the source.**

**(a) The API is one slot.** The three `SetConstraintIntegrators()` overloads
at `fem/darcy/darcyhybridization.cpp:203`, `:212` and `:221` each `reset()`
the other two members. `c_bfi_p`, `c_nlfi_p` and `c_nlfi` are mutually
exclusive by construction; nothing can hold two.

**(b) `M_p` shadows `Mnl_p`'s face lists.**
`DarcyForm::EnableHybridization()` picks the constraint from an exclusive
chain — `if (M_p)` at `fem/darcy/darcyform.cpp:334`, then
`else if (Mnl_p && FaceIntegratorsAreLinear(...))` at `:349`, then
`else if (Mnl_p)` at `:362`, then `else if (Mnl)` at `:377`. The boundary
constraint repeats the same chain at `:479`, `:499`, `:535`, and
`DarcyForm::Assemble()` repeats it again at `:656` / `:700`. But
`SetPotMassNonlinearIntegrator()` is filled by an *unconditional* `if (Mnl_p)`
at `:434-448`, from `Mnl_p->GetDNFI()` — **domain integrators only**.

So a caller with a linear mass on `M_p` and a parametric term on `Mnl_p` gets
the domain half installed and the face half read by nobody. **Measured** in
this tree: with the upwinded face term on `M_p`, `|r_pot| = 1.8220`; with no
such term at all, `0.7282`; with it on `Mnl_p`, `0.7282` — bit-identical to
omitting it. No warning, no abort.

**The same defect is already costed in the tree, for the sibling case.** The
`else if (Mnl)` branch's comment at `fem/darcy/darcyform.cpp:377-396` records
a measured silent drop of the same shape ("in 12 of the 20 `-nld -hb`
references the block nonlinear form carries one interior-face integrator that
nothing here ever reads, so its face stabilization vanishes with no warning")
and prices the repair:

> repairing it means letting `c_bfi_p` and `c_nlfi` coexist, which is a change
> to four readers and a new linear backup for E, G and H. Recorded here rather
> than repaired.

`c_bfi_p` beside `c_nlfi_p` is the identical change with `c_nlfi_p`'s readers
in place of `c_nlfi`'s: a linear backup for E, G and H mirroring what
`Df_lin_data` already does for D. **This is on our list.** It is what gffp
actually needs and proposal B as built does not supply it.

**One route does read a face coefficient live**, and it is all-or-nothing:
`c_nlfi_p`, reached at `darcyform.cpp:362-375`. Getting there needs `M_p` to
be null **and** `FaceIntegratorsAreLinear()` (`darcyform.cpp:289-307`) to
return false, which needs at least one of `Mnl_p`'s face integrators to not be
a `BilinearFormIntegrator`. `HDGConvectionUpwindedIntegrator` derives from
`DGTraceIntegrator` (`fem/darcy/bilininteg_hdg.hpp:98`, **read**), so it *is*
one, and on its own it takes the linear route and is assembled once. That is
the trap in §3 wearing its other face.

**A note on a comment, not a defect.** `Finalize()`'s justification at
`darcyhybridization.cpp:5345-5346` says "there is no linear potential mass in
this branch, or it would not be nonlinear". That is not true of gffp's
configuration — `M_p` (linear mass, linear diffusion) and `Mnl_p` (the
parametric term) coexist, `IsNonlinear()` is still true, and `Df_lin_data`
then holds the linear mass *and* the face constraint. The arithmetic is
additive either way so the answer is right (`AssemblePotMassMatrix()` at
`darcyhybridization.cpp:531` accumulates into the same `Df_data` the backup
copies), but the case the comment says cannot arise is exactly gffp's, and no
test covers it.

## 3. A caching trap we introduced today, and what to call

**This is our defect, it landed today, and it would have hit gffp on first
use of what §1 ships.** Uncommitted work in this tree ("Tier 2") admits a
`BilinearFormIntegrator` sitting on a *nonlinear* mass form and assembles it
**once**:

* `HDGIntegratorIsLinear()` — `fem/darcy/darcyhybridization.cpp:3550`, with
  the reasoning at `:3527-3549`. `class BilinearFormIntegrator : public
  NonlinearFormIntegrator`, so installing one on a nonlinear form compiles,
  and its default `AssembleElementVector()` re-assembles a constant element
  matrix on every residual evaluation.
* `HDGAssembleLinearNLF()` — `:3599`, into `ResidualCache::Au_all` / `Dp_all`
  (`:125`).

For a fixed coefficient that is right, and worth **3.9x to 14.2x** on
`convdiff -nl`'s shape (the six-figure table is at
`darcyhybridization.cpp:119-124`, **measured**, pinned by
`tests/unit/fem/test_darcy_batched_residual.cpp` and
`tests/unit/miniapps/test_debug_device.cpp`). For a **swept** parameter it is
wrong, and it fails in the worst available way. **Measured before the fix**,
on a parametric `ConservativeConvectionIntegrator` — recorded in the tree at
`darcyhybridization.cpp:3714-3733`:

> the residual was wrong by 8.0e-3, 1.6e-2, 2.4e-2, 3.2e-2, 4.0e-2 as the
> parameter moved by 1% steps, growing linearly, with the gradient correct to
> 1.1e-14 throughout and no warning either way.

The residual froze at its first parameter value while the gradient tracked. A
hybridized Jacobian is never assembled globally, so a residual that does not
match its gradient gives no wrong answer — only a Newton that misbehaves. It
took LBFGS-against-Newton to find the last one of these on this branch.

**Two fixes are now in the tree.**

* `CanBatchLinearResidual()` refuses `LocalOpType::PotNL` and `FluxNL`
  whenever a mass slot is involved —
  `fem/darcy/darcyhybridization.cpp:3635-3650`, and `CopyLinearGradBlocks()`
  is gated on the same predicate at `:6229-6232` so that **the residual and
  the gradient freeze together or not at all**. gffp's proposal-B
  configuration (`m_nlfi_p` and nothing else nonlinear) lands on `PotNL`
  (`Finalize()`, `:5337`), so it is covered by this guard and nothing is
  cached there.
* `DarcyHybridization::InvalidateCoefficientCache()` —
  `fem/darcy/darcyhybridization.cpp:6301`, declared public at
  `fem/darcy/darcyhybridization.hpp:1693` with the doxygen at `:1686-1692`.
  One line: `res_cache.reset()`.

**Call it.** Reach it as
`darcy.GetHybridization()->InvalidateCoefficientCache()`
(`fem/darcy/darcyform.hpp:418`); it is not forwarded on `DarcyForm`. Call it
after moving a coefficient and before the next residual or gradient. It is
cheap — the blocks rebuild on the next evaluation — and it does **not** clear
`Af_data`, `Df_data` or the assembled gradient, so it is not a `Reset()`: a
coefficient change on a *bilinear* form still needs that form re-assembled for
the linear blocks to follow, which is the contract those have always had.
`DarcyHybridization::Reset()` already drops the cache at `:8020`, so an
`Update() + Assemble() + Finalize()` cycle needs no extra call.

**Why the type system cannot do this for you.** The declared type of a
parametric `ConservativeConvectionIntegrator` and of a fixed one is the same
type. `HDGIntegratorIsLinear()` can only ask "is this a
`BilinearFormIntegrator`", which answers "does it have an element matrix",
not "does that matrix change". A `Coefficient` exposes no mutation counter and
`Coefficient::SetTime()` is not a proxy for one — gffp's own `cj` trick reads
a member by reference and calls nothing at all. So the declaration has to come
from the caller. If gffp would rather not track it, the safe rule is to call
`InvalidateCoefficientCache()` unconditionally at the top of every residual;
`CanBatchLinearResidual()` is public (`darcyhybridization.hpp:1684`) precisely
so a caller or a test can assert which route it is on before deciding.

## 4. `HDGFloorStabilization` is not on this branch, and you do not need it

The request says it "is already constant and already in this branch". It is
not here. `git grep HDGFloorStabilization` over `fem/`, `miniapps/` and
`tests/` returns nothing on `gf-hdg-dev`, `gf-hdg-linearise-first` and
`gf-hdg-p-adaptivity`, and two files on `gf-hdg-subdomains-dev` (**measured**,
per branch). It is reachable on `meq-integration` at `2613ee1b3b`
(**measured**, 7 hits in `fem/darcy/bilininteg_hdg.hpp`), which is presumably
where it was seen. Per this tree's `CLAUDE.md` the HDG branches are reviewed
separately and are not merged into each other, so it will not arrive here.

**The escape route it stands for is here anyway, and it is the base class.**
`HDGStabilization` — `fem/darcy/bilininteg_hdg.hpp:167`, with the
Nguyen–Peraire–Cockburn Eq. (5) derivation at `:142-166` (**read**). Its
`IsConstant()` defaults to **true** (`:174`) and its `Eval()` returns
`s_diff` unchanged (`:184-186`). The doxygen states the point the request was
reaching for: for a linear flux the positivity bound of the reference's
Eq. (7) reduces `s_conv` to a constant, and "the constant case is therefore
not a special case bolted on here; it is the specialization
`HDGDiffusionIntegrator` and `HDGConvectionUpwindedIntegrator` already
implement". So a constant `tau` is what you get by default and needs no class
at all.

`HDGFloorStabilization` on the other branch is eight lines over that base — it
clamps `s_diff` from below at a `tau_min` and leaves `IsConstant()` alone. If
gffp wants a floor here, writing it against `HDGStabilization` is smaller than
a merge.

**And the guard that makes the constant case safe is worth knowing about for
§2's routing.** `FaceIntegratorsAreLinear()`'s `dynamic_cast` is not by itself
a linearity test; what makes it sound is that the bilinear route reaches such
an integrator only through `AssembleHDGFaceMatrix()`, which carries
`MFEM_VERIFY(!stab || stab->IsConstant(), "A state dependent stabilization
makes the face term nonlinear")`. A genuinely state-dependent stabilization
aborts loudly at assembly rather than quietly assembling a different operator
(`fem/darcy/darcyform.cpp:233-307`, **read**).

## 5. Proposal C: the cost premise did not reproduce, and B already takes most of it

The request's closing claim is "on any mesh we have tried it is the same
shape: assembly >> gradient >> solve". **That did not reproduce on any of six
configurations here.** 32x32 quads, order 2, NPC on, this tree, **measured**
during this assessment and not pinned by any test:

```
Assemble()        14.4 ms     of which integrator quadrature 86%
NPCGradient       20.2 ms
trace solve       46.2 ms
```

So a re-linearised NPC Newton step is about **87 ms**, and the
parameter-carrying quadrature — the only part an interpolatory form removes —
is **9% of it**, or 20% with the solve excluded. Assembly is not the dominant
leg here; the trace solve is, by more than 3x over assembly.

**gffp's problem may genuinely behave the other way, and we are not
contradicting the 152 ms.** A coefficient that interpolates along a field line
at every quadrature point is a plausible way to make quadrature dominate, and
128 wedges at order 1 with a 1.4 ms solve is a very different balance from
32x32 quads at order 2. **So this is a request for the profile rather than a
rebuttal** — see §7.

Two things about C's marginal value are true regardless of whose profile
wins.

**Proposal B already removes 100% of `Assemble()` for a domain term** — the
integrator is evaluated per element per evaluation, live, with nothing
assembled globally at all. Against that baseline C's remaining offer is about
a further 2x on the residual leg (nodal contraction against quadrature) and
**nothing on the gradient leg**, because `NPCGradient()` has to assemble and
factor the local blocks either way.

**And the condensation is not affine in any parameter.**
`ComputeElementH()` (`fem/darcy/darcyhybridization.cpp:3230`) and
`ComputeH()` (`:4195`) factor A, form the local Schur complement and factor
it, per element per gradient. `NPCGradient()` (`:7105`) goes straight there
via `ReducedGradient(MultNlMode::GradAtFields, x_tr)` (`:4682`). No
interpolatory scheme touches any of that: the Schur complement is a nonlinear
function of the blocks, and the LU factorisations are not a contraction of
anything. Whatever C removes from quadrature, this stays — and on the profile
above it is the larger half of a Newton step even before the trace solve.

The superconvergence cost the request already names is real and this branch
would care about it where gffp does not, so it is not the deciding factor.
The deciding factor is the profile.

## 6. Do this

Ordered, and each step is a thing to type rather than a thing to want.

1. **Get the branch.** `meq-integration` does not have `945b7a1fbd`. Merge
   `hdgdev/gf-hdg-linearise-first`, or ask and we will.
2. **Put the linear potential row on `M_p`** — the `cj` mass, the diffusion,
   *and every face term* (`darcy.GetPotentialMassForm()`, with
   `AddInteriorFaceIntegrator` / `AddBdrFaceIntegrator`). §2(b) is why the
   face terms have to go here and not on the nonlinear form.
3. **Put the parametric term on `Mnl_p` as a DOMAIN integrator** —
   `darcy.GetPotentialMassNonlinearForm()->AddDomainIntegrator(...)`. Its
   coefficient is then read live in `AddMultDE` / `AddGradDE` /
   `ConstructGrad` and no re-assembly is needed when it moves.
4. **Call `darcy.GetHybridization()->InvalidateCoefficientCache()` whenever a
   coefficient moves**, before the next residual or gradient. §3.
5. **Assert the route once**, in a test rather than in production:
   `dh->CanBatchLinearResidual()` and `dh->GetPotConstraintIntegrator()` are
   both public, and the second is exactly "the constraint took the linear
   route".
6. **Do not put the upwinded convection on `Mnl_p`.** Today it is either
   silently dropped (with `M_p` present) or silently frozen at its first
   value (without). The fix is ours to build; §2 says what it is.
7. **Keep the two provisional workarounds for now.** Re-taking the gradient
   inside the solve is still necessary — `DarcyForm::Update()`
   (`fem/darcy/darcyform.cpp:2262`) still calls `hybridization->Reset()` at
   `:2295`, which still clears `Grad` — and nothing in `945b7a1fbd` changed
   that. What changes is how often you have to reach for it: with steps 2-4
   the residual needs no `Update()` at all, so the gradient is not being
   destroyed between Newton iterations in the first place. Proposal A, the
   in-place re-assembly of one block, is not built and is not needed if the
   parametric term is on the nonlinear form.

## 7. The open question, and it is the whole of C

**Send the profile.** Specifically, from
`whereTheCoupledMarchSpendsItsTime`, the split of the 152 ms:

* how much of `Update() + Assemble() + Finalize()` is integrator quadrature
  as against allocation, `Reset()` and the offset rebuilds;
* what fraction of the quadrature is the parameter-carrying term alone, as
  against the mass and the diffusion;
* the same three legs at a second order and a second mesh size, so the
  scaling is visible rather than inferred.

The case for C rests entirely on that first number being large, and on this
tree it is 9% of a Newton step. If it is 80% of gffp's, C is worth the
analysis and the superconvergence and we should say so. If it is 9% of
gffp's too, then the 152 ms is allocation and bookkeeping, proposal B removes
all of it by not assembling, and C buys a further 2x on one leg of three.

**One thing we would ask you not to conclude from the 152 ms alone.** A
per-call timing that includes `Reset()` and the offset rebuilds attributes
allocation to quadrature, and 86% quadrature was the answer here only after
the allocation was hoisted out — on this branch, holding the per-call
workspace was worth as much as removing a gather, and fixing either one alone
looked exactly like fixing nothing (`darcyhybridization.cpp:52-101`,
**measured**, six-configuration table). Whichever way gffp's profile comes
out, it is worth taking with the allocations separated from the arithmetic.

## 8. The structural question underneath all of this: make the nonlinear integrators interpolatory

Everything above — proposal B, the caching trap of §3, the `lop_type`
refusal, the invalidation hook — is machinery for one purpose: **recovering,
at run time, which part of a nonlinear element operator was
state-independent.** `NonlinearFormIntegrator::AssembleElementVector()`
returns a vector and says nothing about that split, so the split has to be
inferred, cached, guarded and manually invalidated. Four mechanisms, one
cause. They are enumerated on `EnsureResidualCache()`.

An interpolatory formulation puts the split in the **type** instead of
inferring it. If a nonlinear integrator presents as a fixed shape matrix
times a nodal factor — the nonlinearity collocated at interpolation nodes —
then:

| what we do today | what would remain |
|---|---|
| `HDGIntegratorIsLinear()` recursing through `SumNLFIntegrator` and asking `dynamic_cast` | nothing — the shape matrix is constant by construction |
| `CanBatchLinearResidual()` refusing `PotNL`/`FluxNL` so the residual stays as live as the gradient | nothing — both are built from the same shape matrix |
| `InvalidateCoefficientCache()`, called by hand | nothing for a change in the NONLINEARITY's own state or parameters -- but see the correction below |
| `EnsureResidualCache()` shared between residual and gradient so they cannot diverge | nothing — sharing is structural, and §3's defect becomes unrepresentable |

That is a strictly better answer than proposal C's, because it removes the
reassembly *and* the bookkeeping, where C removes only the reassembly.

**Correcting the third row, and it matters for gffp specifically.** It first
read "nothing for a state or parameter change; only geometry touches the shape
matrix", and that is too strong. Working the design through against a concrete
scheme -- Chen, Cockburn, Singler & Zhang's interpolatory HDG_k, written up in
`doc/HDG-INTERPOLATORY-CCSZ.md` -- shows the shape matrices are invariant in
the *nonlinearity's* arguments and **not** in the coefficients of the linear
operator they are built from. A moving DIFFUSION coefficient re-forms them,
because the postprocessing blocks that define the interpolation nodes' values
are built from the diffusion and the stabilisation. gffp's parametric term is
exactly that kind of coefficient. So interpolatory assembly removes the
invalidation problem for a parametric *reaction* and does not remove it for a
parametric *diffusion*; the four mechanisms are not all deleted by it, and we
should not have implied they were.

**The objection, and it is decisive on its own terms.** Collocating a
nonlinearity at nodes is a different quadrature from integrating it. It is a
**discretisation change, not a refactor**: every answer moves, and all 152
serial references and 121 parallel ones move with it. So this cannot be
adopted in order to delete cache code — that would be choosing a
discretisation to suit an implementation. It has to be justified on its own
merits (the superconvergence, the reassembly saving, whatever gffp's profile
in §7 shows), with the deletion taken as a consequence rather than a reason.

**What would make this decidable**, and neither half exists yet:

1. gffp's profile from §7. If assembly really is 80% of a Newton step there,
   the reassembly saving alone may carry it.
2. One convergence table for a single nonlinear problem, interpolatory
   against the current integrated form, on the same meshes. The question is
   whether the collocation costs an order anywhere. This branch's own record
   says to take that table asymptotically and to print the solver's
   convergence flag in it.

Until (2) exists, the honest position is that interpolatory assembly is the
structurally correct answer to a problem we have currently solved with a
cache, and that we do not know what it costs in accuracy.
