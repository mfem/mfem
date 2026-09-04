# HDG capabilities still wanted in `fem/darcy`

**This file, and every other `.md` here, is scratch.** It is a to-do list and
nothing else, and it is expected to be deleted before this branch becomes a PR.
Anything worth keeping lives in doxygen, in a source comment, or — where it is
about how a miniapp is used — in `miniapps/hdg/README.md`. Nothing in the code
depends on a markdown file for its meaning, and a section that is finished is
cut down to a pointer rather than left here describing itself.

Sections keep the numbers they had, so earlier commit messages citing "§4"
still point somewhere sensible. Where a section is gone it says why.

## The branch topology, because three sections turn on it

```
gf-hdg-dev  (trunk)
  |- gf-hdg-subdomains-dev     extension/lifting       -> its own PR
  |- gf-hdg-linearise-first    NPC  <-- this branch    -> its own PR
  `- gf-hdg-p-adaptivity       per-face trace order    -> its own PR
```

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

Untouched, and nothing in the tree touches it — no boundary-element machinery
exists anywhere in MFEM, so this is a from-scratch build of the exterior
representation rather than an HDG task. By a wide margin the largest item here.
`doc/HDG-BEM-COUPLING-FROM-MEQ.md` is the request, revised down to one
integrator meq will write themselves plus an optimisation nobody needs yet.

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

Three pieces are left:

* **The flux functionals**, `ComputeOutwardFlux` and `ComputeBoundaryFlux` in
  `fem/darcy/functionals_hdg.hpp`. They *refuse* a system rather than silently
  returning field 0's flux, which is what `GetVectorValue` gives them. A
  per-field version wants an API decision: another argument, or a `Vector` of
  one value per field.
* **The nonlinear branches of the rich reconstruction** — the frozen flux law
  and the lifted `Mp_nl` gradient. They are written per field and compile, but
  nothing exercises them with `neq > 1`; every case is linear.
* **A hyperbolic system is not covered by the closure argument.** The closure
  drops one equation per field because the lifted local operator annihilates
  per-field constants (`darcyform.cpp:1673-1690`). A lifted
  `HyperbolicFormIntegrator` Jacobian does not, so the local problem is then
  over-determined in exactly the way the scalar path already is with such a
  term. That is what stands between this and postprocessing
  `miniapps/hdg/navierstokes`, which does not call `Reconstruct` at all today.

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

What is left:

* **Problems 5, 7 and 9 are unchecked.**
* **No transient regression reference**, and one is now possible for the first
  time: all 273 references (152 serial + 121 parallel) pass `--ntimesteps 0`.
* **The DAE questions proper**: index, consistent initialisation of the
  algebraic trace block, and stage-order reduction on the constraint under a
  DIRK method.
* The `vdim == 1` refusal in the H(div) time mass, noted under §4.

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

Optional, and *purely* so — the secondary payoff this entry used to claim, that
it is what makes the classic local postprocessing general in `vdim`, has been
overtaken, that postprocessing already being general. Nothing in `fem/darcy`
interpolates a coefficient or holds a `QuadratureFunction`.

## 11. NPC — Newton on the full system

**Built**: `DarcyHybridization::NPCResidual/NPCGradient/NPCReduce/NPCRecover`,
wrapped as `DarcyNPCOperator` + `DarcyNPCSolver`, serial and parallel, with
`[NPC]` cases in `tests/unit/fem/test_darcy_npc.cpp` — including this tree's
first `[Parallel]` Darcy test. `miniapps/hdg/navierstokes.cpp` is driven by it,
and `convdiff`/`pconvdiff` expose it as `-npc`. **The mechanism and every
measurement are in the code**, on `NPCResidual()`; `doc/HDG-ORDERING-API.md` §3
is the API reference for a caller.

The reference set exists — 23 serial and 23 parallel `*_npc.txt`, which is what
takes the suite to 152 + 121. They compare the local nonlinear iteration count
as well as the solver, the Krylov count and the two error norms, without which
an NPC reference would pass even if `-npc` became a no-op, both routes reaching
the same discrete solution. NPC runs no local nonlinear solve, so the count is
identically zero and the check fails loudly if the flag stops taking effect.

What is left:

* **H(div) flux — attempted, measured, and the refusal stands for a better
  reason.** Not sign conventions: NPC iterates on the broken state and a
  conforming space has no room for it, so the trace row is annihilated for
  every conforming state and lambda is never driven. `BrokenRT_FECollection`
  is the H(div)-shaped space that does work and is now covered.
  `doc/HDG-HDIV-OPTIONAL.md` §1; the numbers are on `NPCCheck()`.
* **A trace-assembled load still has no slot** in `DarcyForm`, on either
  route. Where the caller puts it instead is measured and pinned; adding a
  real slot would move the reduced route too, and nobody has asked.
* **A regression case is on offer and has not been taken.** meq's transport
  barrier is the case that used to throw out of `NewtonSolver::Mult`'s
  `IsFinite` check at iteration zero, and nothing here reproduces that fault —
  the only evidence the divergence guard fixed it is theirs. Worth taking
  before the guard is ever touched.

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
