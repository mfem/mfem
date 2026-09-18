# HDG capabilities still wanted in `fem/darcy`

**This file, and every other `.md` here, is scratch.** It is a to-do list and
nothing else, and it is expected to be deleted before this branch becomes a PR.
Anything worth keeping lives in doxygen, in a source comment, or — where it is
about how a miniapp is used — in `miniapps/hdg/README.md`. Nothing in the code
depends on a markdown file for its meaning, and a section that is finished is
cut down to a pointer rather than left here describing itself.


## Section numbers do NOT agree across the branch family

This used to say "Sections keep the numbers they had, so earlier commit
messages citing §4 still point somewhere sensible. Where a section is gone it
says why." **That is false for §9 and §10**: they are not gone, they were
REUSED for what the trunk calls Optional B and Optional A, with no note saying
so. Neither scheme can be renumbered now without breaking commit messages on
its own branches, so here is the concordance instead. **Check which branch a
commit message is on before following a `§` in it.**

| number | here and `gf-interp-hdg-dev` / `gf-hdg-linearise-first` | `gf-hdg-dev`, `gf-hdg-subdomains-dev`, `gf-hdg-p-adaptivity` |
|---|---|---|
| §3 | Whether the degenerate order loss is asymptotic | Genuinely general Darcy-like problems (this is its §3(d)) |
| §4 | Postprocessing for a system | Systems of coupled nonlinear problems (this was part of it) |
| §9 | Superconvergence at `k = 0` | A driver, attempted and withdrawn (this is their Optional B) |
| §10 | Interpolatory evaluation | Three loose ends, swept (this is their Optional A) |
| §11 | NPC | — |

§1, §2, §5, §6, §7 and §8 mean the same thing in both.

## What this branch family is FOR, and it is narrower than this file reads

**The job on every `gf-*` branch is to make classic NPC HDG work well.** That
is the Nguyen-Peraire-Cockburn method on the spaces this branch's users
actually run -- a discontinuous L2 flux, an L2 potential, a `DG_Interface`
trace, hybridized -- and the solver story around it.

**Fixing the original Darcy pathways is not an obligation.** Clearly inherited
and clearly not owed: the RT and broken-RT flux spaces, the two reductions, the
rich reconstruction (`ReconstructFluxAndPot`), and the `H1_Trace` (EDG) trace
space. They are not what this work is for, and a defect found in one of them is
*recorded* rather than owed. This file has repeatedly grown entries that are
true, interesting, and nobody's job here; a finding about an inherited pathway
belongs in a comment on that pathway, and the entry here shrinks to a pointer.

**Where the line falls elsewhere has not been drawn and should not be guessed.**
§8's time integration in particular has had real work on this branch, and
nothing here assumes it either way.

Two consequences worth stating once. A section marked "not ours" is not
thereby *wrong* or unimportant -- it is a note for whoever owns `fem/darcy`,
which is why the measurement goes into the code where they will meet it. And
directed work still overrides this: when the caller asks for one of them, it
gets done, as §4's first two pieces were.

## The branch topology, because four sections turn on it

```
gf-hdg-dev  (trunk)
  |- gf-hdg-subdomains-dev     extension/lifting       -> its own PR
  |- gf-hdg-linearise-first    NPC  <-- this branch    -> its own PR
  |    `- gf-interp-hdg-dev    interpolatory HDG       -> its own PR
  `- gf-hdg-p-adaptivity       per-face trace order    -> its own PR
```

`gf-interp-hdg-dev` is the one DESCENDANT here rather than a sibling, and that
is forced: its work is built on this branch's `LocalOpType`, batched local
routes and block-nonlinear-integrator path. So §10 is the one section whose
machinery lives downstream rather than sideways.

**These are reviewed by upstream separately and are not merged into each
other**, so a section whose machinery lives on a sibling is not blocked work —
it is work belonging to that branch's PR. Integration for `meq` happens in
`/home/ian/projects/mfem/mfem-src` on `meq-integration`, which carries all
four branches plus upstream master and is what `meq` builds against.

## 1. Extension and lifting — solving on a subdomain of the true domain

**Built, on `gf-hdg-subdomains-dev`, not here.** `fem/darcy/extension_hdg.{hpp,cpp}`
(nine classes, including `HDGExtensionIntegrator` and
`TransferredDatumCoefficient`), `miniapps/hdg/extension.cpp`, and 27 unit test
cases in `tests/unit/fem/test_darcy_extension.cpp`.

Not a merge task — see the topology above. What follows for the sections that
want §1's machinery (§3's `τ` floor, §5's `anisodiff -p 11`, §7's `η₅`) is that
they cannot be done **on this branch**, and are actionable on `meq-integration`,
which has everything.

Two artefacts of §1 sit here either way: the doxygen at
`fem/darcy/darcyform.hpp:174` refers the reader to `extension_hdg.hpp`, which
does not exist here, and `AssembleFluxMassBdrFaces()` exists solely to serve
§1.

## 2. Coupling at a distance to an exterior boundary-integral solve

**FULLY OPTIONAL, and the plan lives on `gf-hdg-subdomains-dev` as
`doc/HDG-BEM-COUPLING.md`** -- there because it builds on §1's machinery, which
is there. It used to be described here as "the largest item by a wide margin";
nobody is waiting for it.

**Choosing the artificial boundary to be a circle in 2-D or a sphere in 3-D
makes the exterior operator exact and diagonal and removes the boundary
integral machinery entirely** -- Gatica & Hsiao, *The uncoupling of boundary
integral and finite element methods for nonlinear boundary value problems*,
J. Math. Anal. Appl. **189** (1995) 442-461. **meq has that implemented and
working** (`src/meq/ExteriorDtN.hpp`: an exact diagonal Dirichlet-to-Neumann
map on a semicircle in a Gegenbauer basis, one number per mode, no layer
potentials); **gffp has it derived and relied upon, not yet coded**
(`PHYSICS-NOTES.md` §4a, diagonal in Legendre modes on a sphere, and their own
note calls it "a verification oracle, not a method"). Both known consumers of
this tree avoid the coupling by choosing the boundary.

The transmission integrator the request did want was written by meq and merged
(`ExtensionBoundaryQuadrature()`); only the auxiliary globally-coupled
unknowns are unbuilt, and they are an optimisation rather than a prerequisite.

## 3. Whether the degenerate order loss is asymptotic

The practical answer is known — floor the stabilisation — but whether the loss
is asymptotic or pre-asymptotic was never settled, and two things stand in the
way.

* **The measurement cannot answer it as written.** `Rates()` in
  `tests/unit/fem/test_darcy_degenerate.cpp` runs three meshes from n = 4 to
  n = 16 and overwrites `rate_p` at each refinement, so it computes two rates
  and reports the last. Settling the question needs a deeper sweep (n → 64 or
  128) that keeps the whole rate sequence.
* **The floor is not in the library on this branch.** `HDGFloorStabilization`
  is on `gf-hdg-subdomains-dev`; here the only floor is a test-local
  `class FloorTau` in that same file. So §3 belongs to that branch's PR.

The loss itself and the floor's repair are already pinned by regressions there.

## 4. Postprocessing for a system

**Done for the linear diffusion path.** Both reconstructions are general in
`vdim` — `DarcyForm::Reconstruct` and the `ReconstructTotalFlux` under it —
alongside the classic `HDGPotentialPostprocessor`, which always was. The
measurement, the closure-row argument and the ordering argument are in the
doxygen on those two methods and in `tests/unit/fem/test_darcy_reconstruction.cpp`,
whose four `[System]` cases are the pins.

One piece is left, and two are done:

* ~~The flux functionals refuse a system~~ — **done.** The API decision went to
  a `Vector` of one value per field, overloaded beside the `real_t` entry
  points, which now *are* the `vdim == 1` case of it and refuse a system for
  the reason they always did: one number cannot answer for several fields. The
  per-field read is on `AddFaceNormalFlux()`; the conservation identity is
  pinned field by field, under both `Ordering`s, in
  `tests/unit/fem/test_darcy_functionals.cpp`.
* ~~The nonlinear branches of the rich reconstruction are unexercised at
  `neq > 1`~~ — **done, and running them found a gap.** The coupled nonlinear
  manufactured problem now goes through `Reconstruct()` at two fields and gets
  `k+1 → k+2` in both, at `k = 1` and `k = 2`; the `k = 0` DG row is flat,
  which is the known CCSZ restriction reproduced rather than a defect of the
  system path. **The gap: an H(div) flux at `neq > 1` segfaulted** in the local
  solve, because the frozen law's coefficient is `neq*dim` square and
  `VectorFEMassIntegrator` reads a `dim`-square one. Now a loud refusal, with
  what a real repair would take — a coupled vector-FE mass, which the tree does
  not have — recorded on `ReconstructFluxAndPot()`.
* **A divergence-form term is not covered by the closure argument — and this
  is NOT ours.** The closure is unconditional and is correct only while the
  lifted local operator keeps the per-field constant in its null space; the
  conservative form does not, which is measured and written up at the closure
  itself in `darcyform.cpp`, along with the two reasons it is not yet shown to
  break anything. `ReconstructFluxAndPot()` is an original Darcy pathway, so
  this is recorded rather than owned. See the scope note at the top of this
  file.

  One thing to know before anyone reaches for it:
  **`HDGPotentialPostprocessor` is immune, structurally** — its matrix is the
  Neumann stiffness whatever the PDE is, with the physics only in the
  right-hand side. So the sentence this entry used to carry, that the closure
  is "what stands between this and postprocessing `miniapps/hdg/navierstokes`",
  probably names the wrong obstacle: the classic postprocessing is
  `vdim`-general and needs only the computed flux and potential. Whether its
  `q = -K grad p` assumption suits the Navier-Stokes viscous flux is unchecked.

Separately and smaller, and it is §8's rather than §4's:
`miniapps/hdg/darcyop.cpp:370` and `:396` refuse `vdim > 1` for the H(div) flux
time mass. The DG path handles `vdim` already.

## 5. `τ` for problems that are convection- and diffusion-dominated at once

**Measured; the tables and the mechanism are in the header comment of
`miniapps/hdg/navierstokes.cpp`.** The short version is that the
direction-aware `S = λ_max(û,n) I` is 2.0–3.6× *worse* than the best constant
`τ` in the flux and the pressure, better than any constant at keeping Newton
alive on coarse meshes at high `Re`, and that its accuracy level is set by `β`
— a free parameter of the formulation — rather than by the flow.

**What is left is a problem, not a method.** Both of that miniapp's exact
solutions put their sharp structure across the flow and little or none along
it, so the along-flow faces — the only ones where `λ_max` differs from `√β` —
are exactly where the solution is easiest to represent. Kovasznay cannot repair
that on its own window, its decay rate `λ → −4π²/Re` flattening the along-flow
structure at exactly the `Re` that makes it convective. **A genuinely
two-directional exact solution is what would settle the general question**, and
`anisodiff -p 11` on `gf-hdg-subdomains-dev` is the linear-diffusion shape of
it — so this half belongs there.

A library constraint bounding how far this can go:
`MixedConductionNLFIntegrator`'s HDG face stabilization for more than one
equation is `face_w * TauVar(e)`, one constant per equation through
`SetVariableStabilization()`. It cannot express a stabilization depending on
the state or the face normal. The Navier-Stokes driver sidesteps it by carrying
the convective stabilization on the `NumericalFlux`; a *viscous* stabilization
varying with direction could not.

## 6. Functionals of the solution — DONE

Nothing left. `fem/darcy/functionals_hdg.hpp` carries what it does and does
not. The number is kept only so commit messages citing "§6" land somewhere.

## 7. Adaptive refinement: `hp`, and the estimator's fifth term

**`h` is done and tested. `p` is `gf-hdg-p-adaptivity`'s**, where steps 1–3 of
the plan are built (a per-face trace order behind two accessors, the surplus
constrained, `convdiff -pref`) along with an `hp` demonstrator, a smoothness
sensor and the parallel port. **The plan lives on that branch** —
`doc/HDG-P-ADAPTIVITY.md` plus `HDG-P-ADAPTIVITY-CONSTRAIN.md` and
`HDG-P-ADAPTIVITY-MEQ-MERGE.md` — and not here, so read it there.

What the scoping established, since it is why that is a separate branch at all:
the element spaces are **already** `p`-adaptive and need no library change —
every offset in `DarcyHybridization` is built per entity — but it buys nothing
on its own, because **the trace order sets the rate.** Rates over
`nx` = 4, 8, 16, 32 on `convdiff -p 1 -dg -hb`:

| element / trace | dim M at nx=32 | flux | potential |
|---|---|---|---|
| 2 / 2 | 6336 | → 2 | 3.9 |
| 3 / 2 | 6336 | 1.98 | 3.03 |
| 4 / 2 | 6336 | 2.00 | 2.99 |
| 3 / 3 | 8448 | 2.96 | 4.65 |

Raising the element order above the trace order changes the constant and not
the rate; the global system is `dim M` and never moves. Two prerequisites were
paid for here and are on the trunk: the HDG face quadrature now sees the trace
element's order, and `DarcyOperator` survives a hanging-node-free NC mesh.

**`η₅` of the SSC estimator is open, and blocked twice over.**
`HDGErrorEstimator` has exactly two terms (`Type::{Residual, Energy}`) and
takes an integrator rather than a coefficient, so it needs an adapter or a
second entry point; and `TransferredDatumCoefficient`, the thing `η₅` would be
built from, is §1's and not on this branch.

## 8. Time integration of the DAE

**The integrators work and problem 4 is verified.** `DarcyOperator` is a
`TimeDependentOperator(IMPLICIT)` with `ImplicitSolve`; `convdiff` has four ODE
solvers behind `-ode` (backward Euler and three SDIRK, formally orders 1–4) and
four transient problems. Observed temporal orders 1, 2.00, 3 and 4, and order
4 = `k+1` in space at `k = 3`; the table is in the header comment of
`miniapps/hdg/convdiff.cpp`, which is where it belongs. Two defects found and
fixed on the way: `convdiff` never called `SetTime()` on the exact-solution
coefficients, so every transient error it had ever printed compared against
`t = 0`; and problem 4's exact solution spread as `2σ² + 4kt·π/4` where the PDE
requires `2σ² + 4kt`, so it solved no equation the miniapp poses.

**Correcting the first of those two, which this entry got wrong.** The error
was not compared against `t = 0`. `gcoeff` is built unconditionally as
`ProductCoefficient(-1., tcoeff)` and `ProductCoefficient::SetTime` propagates
to its operands, so `tcoeff` was always being advanced — to the time
`DarcyOperator` last set, which is the **stage** time `t + c_i·dt`. That
distinction is the content of the defect, because it says which solvers are
hit: backward Euler's one stage is at `c = 1` and is exactly unaffected, while
`-ode 2`, `3` and `4` land at `c = 0.707`, `0.211` and `1 - a < 0`. And
`pconvdiff` never had the fix at all until the trunk merge that brought this
note.

**Problems 7 and 9 now carry transient references** — four each per arm,
sixteen in all, one per ODE solver, at `-tf 0.5 -nt 2`, coarse enough that the
four `-ode` answers are separated by far more than the suite's `1e-4`. They are
the right two: `u = (exp(t) − 1)·ux·uy` is identically zero at `t = 0`, so the
initial condition is exact in every space, and their data is homogeneous and
time-independent.

What is left:

* **Problem 5 is not a temporal study and never was** — `GetQFun` returns zero
  for `KovasznayFlow`, so `q_err` prints `inf`, and its `GetTFun` ignores `t`.
  Problem 4 is checked and has no reference; it is the only one with
  time-varying boundary data.
* **The DAE questions proper**: index, consistent initialisation of the
  algebraic trace block — which problems 7 and 9 dodge by starting from zero
  rather than answering — and stage-order reduction on the constraint under a
  DIRK method. The last of these is measured: `convdiff.cpp`'s header comment
  carries the self-convergence table showing the flux capped at `min(p, 2)`.
* ~~The `vdim == 1` refusal in the H(div) time mass~~ — H(div), so not ours;
  `doc/HDG-HDIV-OPTIONAL.md` §3 has it.

**ARKODE is present and not usable here**, which is worth knowing before
anyone tries to wire it. `ARKStepSolver` offers `IMPLICIT` and `IMEX` DIRK
methods, but its implicit path drives `RHS1` — the operator's `Mult`, i.e. an
explicit `f(t, y)` — plus `LinSysSetup`/`LinSysSolve`, and runs its own Newton.
`DarcyOperator` defines **no `Mult` at all**, only `ImplicitSolve`, and cannot
meaningfully define one, the trace block having no time derivative. Reaching
ARKODE means either its mass-matrix/DAE facilities or a reformulation — the DAE
questions above, not a wiring job. MFEM's own SDIRK methods already give orders
1 through 4.

## 9. Superconvergence at `k = 0` — the HHO-inspired methods

Optional. Cheaper than it reads: two of the three ingredients are already
available. `τ ~ 1/h` is the built-in default scaling, and unequal
flux/potential/trace orders are unconstrained — nothing in `fem/darcy` ties the
spaces' orders together, so flux in `[P^k]^d`, potential in `P^{k+1}`, trace in
`P^k` is constructible today. Missing is the third: a stabilisation acting on
the **L2 projection of the potential onto the trace space** rather than on the
potential itself. `HDGStabilization` is a scalar hook that can rescale `τ` but
cannot change what `τ` multiplies, so this needs a new face integrator.

## 10. Interpolatory evaluation of the nonlinear coefficient

**Built, on `gf-interp-hdg-dev`, not here**, and the plan lives on that branch
— `doc/HDG-INTERPOLATORY-CCSZ.md`, which used to sit here and does not any
more, a plan document belonging on the branch doing the work. Chen, Cockburn,
Singler & Zhang, *J. Sci. Comput.* **81** (2019) 2188–2212.

What is there: `HDGPostprocessBlocks` splitting the classic postprocessing
into per-element blocks that can be applied rather than re-solved;
`fem/darcy/reaction_hdg.{hpp,cpp}` with an interpolatory reaction integrator
and a quadrature one to measure it against; the Jacobian's (1,0) block, which
this branch's hybridization did not have and which a term evaluated at the
postprocessed potential needs; and
`BlockNonlinearFormIntegrator::GetBlockRowMask()`, so a term writing one block
row keeps the flux mass factored once.

Two things to know before merging any of it back this way. The mask adds a
**virtual to `BlockNonlinearFormIntegrator`**, which is a layout change to
every integrator deriving from it — `make clean`, both trees. And the entry
below, that "nothing in `fem/darcy` interpolates a coefficient", is true HERE
and false there; it is a statement about this branch, not about the family.

Optional, and *purely* so — the secondary payoff this entry used to claim, that
it is what makes the classic local postprocessing general in `vdim`, has been
overtaken, that postprocessing already being general. Nothing in `fem/darcy`
interpolates a coefficient or holds a `QuadratureFunction`.

## 11. NPC — Newton on the full system

**Built**: `DarcyHybridization::NPCResidual/NPCGradient/NPCReduce/NPCRecover`,
wrapped as `DarcyNPCOperator` + `DarcyNPCSolver`, serial and parallel, with
`[NPC]` cases in `tests/unit/fem/test_darcy_npc.cpp` — including this tree's
first `[Parallel]` Darcy test. `NPCReduce()` and `NPCRecover()` have blocked
forms taking one column per right-hand side, and `DarcyNPCSolver::ArrayMult()`
composes them, for a bordered or adjoint solve applying one factored Jacobian
to several right-hand sides known at once; `doc/HDG-ORDERING-API.md` §3.2a is
the contract. `miniapps/hdg/navierstokes.cpp` is driven by it,
and `convdiff`/`pconvdiff` expose it as `-npc`. **The mechanism and every
measurement are in the code**, on `NPCResidual()`; `doc/HDG-ORDERING-API.md` §3
is the API reference for a caller.

The reference set exists — 23 serial and 23 parallel `*_npc.txt`. (The suite
is 160 + 129 on this branch; the count used to be quoted here as 152 + 121 and
was overtaken by the sixteen transient references, which is what a total
written into prose does.) They compare the local nonlinear iteration count
as well as the solver, the Krylov count and the two error norms, without which
an NPC reference would pass even if `-npc` became a no-op, both routes reaching
the same discrete solution. NPC runs no local nonlinear solve, so the count is
identically zero and the check fails loudly if the flag stops taking effect.

What is left:

Settled, and no longer open: **the H(div) refusal**, which is measured and
stands for a better reason than the one it used to give, with
`BrokenRT_FECollection` as the H(div)-shaped space that does work and is
covered. It is also not ours by the scope note. The numbers are on
`NPCCheck()`.

* ~~A trace-assembled load still has no slot~~ — **built.**
  `DarcyForm::GetTraceRHS()` is the third load beside `GetFluxRHS()` and
  `GetPotentialRHS()`, and **both routes carry it with no caller wiring**:
  `DarcyHybridization::ReduceRHS()` adds `P^T b_λ` to the reduced right-hand
  side and `NPCResidual()` subtracts it from the trace block, which is the
  same convention read off `r = A x - b`. The sign, the API and the reason
  the registration happens at construction rather than at `Assemble()` are on
  the accessor; the pin is "A load on the skeleton reaches both routes" in
  `tests/unit/fem/test_darcy_npc.cpp`, which compares VECTORS against the
  hand-added answer because a wrong sign converges.
* ~~A regression case is on offer and has not been taken~~ — **taken**, in
  `55465de4e9`: "A transport barrier diverges without going non-finite",
  `tests/unit/fem/test_darcy_npc.cpp:2441`. This entry outlived the commit
  that closed it by two sessions, which is the reason for the scope note at
  the top of this file about what markdown is for.
* ~~A face constraint whose COEFFICIENT moves has to be re-assembled~~ —
  **built**: `DarcyForm::SetFaceConstraintMode(FaceConstraintMode::Live)`
  keeps the nonlinear potential mass form's face integrators on the
  hybridization's live slot while the linear form's stay frozen beside them,
  and `DarcyHybridization` keeps the frozen half of E, G and H so the live
  pass can be added to it — the same device `Df_lin_data` is for D. NPC only,
  refused elsewhere. The pin is "A live face constraint reads its coefficient
  at every residual" in `tests/unit/fem/test_darcy_npc.cpp`, whose null
  section (live and frozen are the same operator when nothing moves) is what
  checks the seeding and whose third section is the freeze it repairs.
  Asked for by gffp, who measured 92.7 ms of Update() + Assemble() +
  Finalize() against 8.9 ms for the gradient it was performed to move — 90%
  of a coupled Newton step spent assembling in order to move one coefficient.

## 12. A flux that carries fewer directions than the mesh has

**Built**, on the NPC path only: `RestrictedVectorDivergenceIntegrator` and
`RestrictedNormalTraceJumpIntegrator` in `fem/darcy/bilininteg_hdg.hpp`, with
`tests/unit/fem/test_darcy_restricted_flux.cpp` serial and parallel. Asked for
by gffp, whose collision operator diffuses in the velocity coordinates only, so
the parallel direction has a structural zero diffusion at every point and
permanently — `VectorMassIntegrator(1/kappa)` asks for an infinite coefficient
there and no floor is a limit, only a window. **The mechanism, the slice
identity and every measurement are in the code**, on the two classes and on the
guards in `AssembleDivMatrix()`, `ConstructC()` and `AssembleFluxMassMatrix()`.

`vdim = 0` — no flux unknown at all, i.e. pure HDG advection — needed nothing
new and is pinned in the same file.

What is left:

* ~~No miniapp and no regression reference~~ — **built.** `convdiff`/
  `pconvdiff` problem **11**, `-fc` and `-ofl`, with six references per arm.
  `p = x y + y^2` on the unit square, `c = (1,0)`, diffusion in `y`: degree 2
  and so exact in the discrete spaces, `d_xx p = 0` so a flux that also
  carries `x` solves the same problem, and NOT zero on the boundary, which is
  what every other problem in the file fails to test. The arms are `xy` and
  `y` (exact), `x` (the falsification — restricting to the direction the
  problem does not diffuse in, 0.99 against 7e-16), `none` with `-k 0` (no
  flux unknown at all, pure advection, exact) and the two `-ofl` arms below.
  The problem is numbered 11 and not 10 because `gf-interp-hdg-dev` uses 10
  and both branches merge into `meq-integration`.
* **Only an axis-aligned subset of directions.** A general injection matrix is
  a linear combination of directional derivatives rather than a column slice,
  so it is not a wrapper. Nobody has asked for it.
* ~~The weak Dirichlet route does not carry the restriction~~ — **built**:
  `RestrictedVectorBoundaryFluxLFIntegrator`, the third member of the family.
  `VectorBoundaryFluxLFIntegrator` sizes its element vector from the MESH
  dimension while the space owns `|S|*dof`, and `Vector::AddElementVector()`
  reads the first `|S|*dof` entries and returns, so the load landed in the
  wrong direction with nothing to notice — 9.3e-01 where the restricted class
  gives 2.1e-15. `DarcyForm::Assemble()` now refuses the stock class by name
  when the flux space carries fewer components than the mesh. This is the
  same `dim`-versus-`vdim` confusion the constraint had, in a third place.
* **The reduced route refuses it**, at `DarcyHybridization::Mult()`. Out of
  scope rather than known broken: the same blocks would be eliminated by the
  same algebra and nothing has run it. One measurement would settle whether
  the refusal is load-bearing or policy.
* ~~A one-sided physical inflow/outflow trace is unbuilt~~ — **measured, and
  it needed no library change for the case this section is about.** With no
  flux component along the outflow normal the one-sided constraint row is the
  upwinded convective one alone, which reads `uhat = u_h`: the transmissive
  outflow, for free. `-fc y -ofl` is bit-identical to `-fc y`, because a datum
  there contributes nothing anyway. A flux that DOES reach the outflow has no
  such luck — `-fc xy -ofl` is 1.6 and does not converge with the mesh — and
  the repair is the prescribed numerical flux on `DarcyForm::GetTraceRHS()`,
  which already exists and is measured exact. Pinned by "A non-essential
  boundary trace with no datum imposes zero flux" in
  `tests/unit/fem/test_darcy_npc.cpp`, which is a SCALAR problem and so
  answers the question `pnavierstokes.cpp`'s note left open: that note is
  about the hybridization and not about the artificial-compressibility system.

## Deliberately not being done here

The miniapps still default to the weak route for DG. Moving `convdiff` and its
siblings onto the essential-trace route is the branch author's call, not ours,
and it would move their regression references; it is being raised with them.
The same goes for the `-trbc` gap, which the library fix has closed but which
nothing in the suite exercises. `doc/HDG-HDIV-OPTIONAL.md` §4 has the seven
guards and the mechanism.

## References

Only those an *open* section still needs. The rest were moved into the doxygen
of the code that implements them.

* **CS-Extensions** — Cockburn & Solano, on solving problems posed on curved
  domains by extension from a polyhedral subdomain. §1.
* **CSS-Coupling** — Cockburn, Sayas & Solano, on coupling an HDG interior
  solve to an exterior boundary-integral representation across an unmeshed
  interface, with **CSS-Analysis** its companion, including the relaxed
  iteration and the contraction estimate. §2.
* **CCSZ-I** — Chen, Cockburn, Singler & Zhang, *Superconvergent interpolatory
  HDG methods for reaction diffusion equations I: an HDGk method*, J. Sci.
  Comput. **81** (2019) 2188–2212. The interpolatory idea is §10.
* **CCSZ-II** — *… II: HHO-inspired methods*, Commun. Appl. Math. Comput. **4**
  (2022) 477–499. Its Table 1 classifies the three variants; (A) and (B) are
  superconvergent from `k = 0`, (C) only from `k = 2`, and all three take
  `τ ~ 1/h`. §9.
* **Lehrenfeld–Schöberl** — HDG+, the same object as CCSZ-II's HDG (A), with
  **Oikawa**, *A hybridized discontinuous Galerkin method with reduced
  stabilization*, J. Sci. Comput., arriving at it independently. §9.
* **NPC-Stokes** — Nguyen, Peraire & Cockburn, *A hybridizable discontinuous
  Galerkin method for Stokes flow*, Comput. Methods Appl. Mech. Engrg. **199**
  (2010) 582–597. §3.2 is the augmented-Lagrangian reduction to the velocity
  trace alone, §4.1 the stabilisation sweep. §9.
* **Persson & Peraire**, modal-decay smoothness sensor — the standard choice
  for the `hp` criterion of §7.
