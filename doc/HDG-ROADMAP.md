# HDG capabilities still wanted in `fem/darcy`

**This file, and every other `.md` here, is scratch.** It is a to-do list and
nothing else, and it is expected to be deleted before this branch becomes a PR.
Anything worth keeping lives in doxygen, in a source comment, or — where it is
about how a miniapp is used — in `miniapps/hdg/README.md`. Nothing in the code
depends on a markdown file for its meaning, and a section that is finished is
cut down to a pointer rather than left here describing itself.

Sections keep the numbers they had, so earlier commit messages citing "§4"
still point somewhere sensible. Where a section is gone it says why.

## 1. Extension and lifting — solving on a subdomain of the true domain

**Built, on `gf-hdg-subdomains-dev`, not here.** `fem/darcy/extension_hdg.{hpp,cpp}`
(nine classes, including `HDGExtensionIntegrator` and
`TransferredDatumCoefficient`), `miniapps/hdg/extension.cpp`, and 27 unit test
cases in `tests/unit/fem/test_darcy_extension.cpp`.

**This is not a merge task, and an earlier version of this entry said it
was.** The two branches are meant to be reviewed by upstream *separately*, so
neither is waiting on the other and the divergence between them (32/13, and
growing) is not a debt to pay down. Merging them is done elsewhere and for a
different purpose: `meq-integration` in a separate working tree
(`/home/ian/projects/mfem/mfem-src`) carries **all three** HDG branches on
upstream master and is what `meq` builds against. So the machinery below is
not unavailable in an absolute sense — it is unavailable *here*, and present
there.

What follows for the sections that want §1's machinery — §3's `τ` floor, §5's
`anisodiff -p 11`, §7's `η₅` — is that they cannot be done **on this branch**
at all, because `HDGFloorStabilization`, that driver and
`TransferredDatumCoefficient` live on the other one. They are not blocked
work; they are work that belongs to the other branch's PR, or to the
integration tree. Nothing here should wait for them.

One artefact remains on this branch either way: the branch is named
`gf-hdg-linearise-first` after an ordering that has since been deleted.

One artefact of §1 sits on this branch: `fem/darcy/darcyform.hpp:174` refers
the reader to `extension_hdg.hpp`, which does not exist here, and
`AssembleFluxMassBdrFaces()` exists solely to serve §1.

**Three other sections want machinery that lives there**: §3's remedy, §5's
remaining problem and §7's `η₅`. See each. None of them is actionable here.

## 2. Coupling at a distance to an exterior boundary-integral solve

Untouched, and nothing in the tree touches it — no boundary-element machinery
exists anywhere in MFEM, so this is a from-scratch build of the exterior
representation rather than an HDG task. By a wide margin the largest item here.

## 3. Whether the degenerate order loss is asymptotic

The practical answer is known — floor the stabilisation — but whether the loss
is asymptotic or pre-asymptotic was never settled, and two things stand in the
way that the entry did not previously name.

* **The measurement cannot answer it as written.** `Rates()` in
  `tests/unit/fem/test_darcy_degenerate.cpp` runs three meshes from n = 4 to
  n = 16 and overwrites `rate_p` at each refinement, so it computes two rates
  and reports the last. Settling the question needs a deeper sweep (n → 64 or
  128) that keeps the whole rate sequence.
* **The floor is not in the library on this branch.** `HDGFloorStabilization`
  is on `gf-hdg-subdomains-dev`; here the only floor is a test-local
  `class FloorTau` in that same test file. So §3 belongs to the other branch's
  PR, not to this one.

The loss itself and the floor's repair are already pinned by regressions there.

## 4. Postprocessing for a system

**Done for the linear diffusion path; three pieces are left, all named below.**
Both reconstructions are now general in `vdim` — `DarcyForm::Reconstruct` and
the `DarcyHybridization::ReconstructTotalFlux` under it — alongside the classic
`HDGPotentialPostprocessor`, which always was. The measurement, the closure-row
argument and the ordering argument are in the doxygen on those two methods and
in `tests/unit/fem/test_darcy_reconstruction.cpp`; the four new `[System]`
cases there are the pins.

Two claims this entry used to make were wrong and are withdrawn rather than
edited away. `ReconstructTotalFlux` was **not** already vdim-general: its face
solve inverted the scalar face mass against an `neq`-times-too-long right-hand
side, and its element-interior pass took the interior dofs as the tail of the
whole vdof list. And the contiguous-tail problem was never in
`ReconstructFluxAndPot` at all — that is where the entry said it was.

What is left:

* **The flux functionals**, `ComputeOutwardFlux` and `ComputeBoundaryFlux` in
  `fem/darcy/functionals_hdg.hpp`. They now *refuse* a system rather than
  silently returning field 0's flux, which is what `GetVectorValue` gives them.
  A per-field version is the piece to write, and it wants an API decision:
  another argument, or a `Vector` of one value per field.
* **The nonlinear branches of the rich reconstruction** — the frozen flux law
  and the lifted `Mp_nl` gradient. They are written per field and compile, but
  nothing exercises them with `neq > 1`; every new case is linear.
* **A hyperbolic system is not covered by the closure argument.** The closure
  drops one equation per field because the lifted local operator annihilates
  per-field constants. A lifted `HyperbolicFormIntegrator` Jacobian does not,
  so the local problem is then over-determined in exactly the way the scalar
  path already is with such a term. That is what stands between this and
  postprocessing `miniapps/hdg/navierstokes`, which does not call `Reconstruct`
  at all today.

Separately and smaller: `miniapps/hdg/darcyop.cpp:370` and `:396` refuse
`vdim > 1` for the H(div) flux time mass. The DG path handles `vdim` already.
That is §8's, not §4's.

## 5. `τ` for problems that are convection- and diffusion-dominated at once

**Measured; see the header comment of `miniapps/hdg/navierstokes.cpp`, which
carries the tables and the mechanism.** The short version is that the
direction-aware `S = λ_max(û,n) I` is 2.0–3.6× *worse* than the best constant
`τ` in the flux and the pressure, better than any constant at keeping Newton
alive on coarse meshes at high `Re`, and that its accuracy level is set by `β`
— a free parameter of the formulation — rather than by the flow.

**What is left is a problem, not a method.** Both of that miniapp's exact
solutions put their sharp structure across the flow and little or none along
it, so the along-flow faces — the only ones where `λ_max` differs from `√β` —
are exactly where the solution is easiest to represent. Kovasznay cannot repair
that on its own window: its decay rate `λ = Re/2 − √(Re²/4 + 4π²) → −4π²/Re`,
so the parameter that makes it convective flattens its along-flow structure.
**A genuinely two-directional exact solution is what would settle the general
question.** `anisodiff -p 11` on `gf-hdg-subdomains-dev` is the linear-diffusion
shape of it, and it is on the other branch — so this half of §5 belongs
there too.

A library constraint that bounds how far this can go:
`MixedConductionNLFIntegrator`'s HDG face stabilization for more than one
equation is `face_w * TauVar(e)`, one constant per equation set through
`SetVariableStabilization()`. It cannot express a stabilization depending on
the state or the face normal. The Navier-Stokes driver sidesteps it by carrying
the convective stabilization on the `NumericalFlux`; a *viscous* stabilization
varying with direction could not.

## 6. Functionals of the solution — DONE

Nothing left. `fem/darcy/functionals_hdg.hpp` carries what it does and what it
does not. The number is kept only so that commit messages citing "§6" land
somewhere.

## 7. Adaptive refinement: `hp`, and the estimator's fifth term

`h`-adaptivity is done and tested. **`p` is untouched** — nothing in the tree
sets an element order or computes a smoothness indicator, so this is both the
`p` mechanism and the `h`-versus-`p` decision.

`η₅` of the SSC estimator is also open, and is blocked twice over.
`HDGErrorEstimator` has exactly two terms (`Type::{Residual, Energy}`) and
takes an integrator rather than a coefficient, so it needs an adapter or a
second entry point — that much the entry already said. What it did not say is
that **`TransferredDatumCoefficient`, the thing `η₅` would be built from, is
not on this branch**: it is §1's, and so is this.

## 8. Time integration of the DAE

**Not untouched, and the reason it is unverified is now known rather than
guessed.** `DarcyOperator` is a `TimeDependentOperator(IMPLICIT)` with
`ImplicitSolve`; `btime_u`/`btime_p` lift a `1/dt` mass onto either block;
`convdiff` has four ODE solvers behind `-ode` (backward Euler, and
`SDIRK23Solver` at two options plus `SDIRK34Solver`, formally orders 1, 2, 3
and 4) and four transient problems.

**One defect found and fixed.** `convdiff` never called `SetTime()` on the
exact-solution coefficients used to compute the error — only `gcoeff`,
`fcoeff` and `qtcoeff` are handed to `DarcyOperator`, which is the only thing
that called it. So every transient error the miniapp has ever printed compared
the evolving solution against the exact one **frozen at t = 0**, and problem
4's exact solution is a Gaussian rotating with `cos(4 c t π/4)`. Fixed; no
steady reference moves, their exact solutions ignoring the argument.

**The time integrators themselves work.** Measured on problem 4 with the
spatial error made negligible, final-time potential error against `nt`:

| `nt` | `-ode 1` | `-ode 2` | `-ode 3` | `-ode 4` |
|---|---|---|---|---|
| 16 | 0.00925 | 0.01262 | 0.01456 | 0.01243 |
| 32 | 0.01081 | 0.01230 | 0.01226 | 0.01223 |
| 64 | 0.01152 | 0.01222 | 0.01221 | 0.01220 |
| 128 | 0.01186 | 0.01221 | 0.01220 | 0.01220 |

All four converge, and the higher-order ones get there sooner — `-ode 4` is
converged by `nt = 32` where backward Euler is still climbing at 128. That is
the first evidence in this tree that the time-stepping does anything correct.

**~~But they converge to the wrong answer.~~ Fixed: problem 4's exact solution
was wrong.** It spread as `2σ² + 4kt·π/4`, and the PDE requires `D' = 4k`, so
`2σ² + 4kt`. The `π/4` belongs to the *rotation*, which carries it legitimately
and where it recurs as a `4·X·π/4` idiom; in the diffusion it made the exact
solution solve no equation the miniapp poses. The exact solution and the exact
flux were mutually consistent — `q` really was `−k∇u` of that `u` — which is
why it did not look wrong locally, and the source is zero so nothing absorbed
it. Hence a limit independent of both `dt` and `h`, and an error vanishing as
`t → 0` where the two denominators agree.

**Problem 4 is now the tree's one verified transient problem, and §8's first
task is done.** Observed temporal orders 1, 2.00, 3 and 4 for `-ode 1…4`, and
order 4 = `k+1` in space at `k = 3`. The table is in the header comment of
`miniapps/hdg/convdiff.cpp`, which is where it belongs. Problems 5, 7 and 9
remain unchecked.

What is still open here is the DAE theory rather than the verification: index,
consistent initialisation of the algebraic trace block, and stage-order
reduction on the constraint under a DIRK method. A transient *regression
reference* is now possible for the first time and has not been taken — every
one of the 273 references still passes `--ntimesteps 0`.

**ARKODE is present and not usable here.** `ARKStepSolver` in
`linalg/sundials.hpp` offers `IMPLICIT` and `IMEX` DIRK methods, but its
implicit path drives `RHS1` — the operator's `Mult`, i.e. an explicit
`f(t, y)` — plus `LinSysSetup`/`LinSysSolve` for `I - γJ`, and runs its own
Newton. `DarcyOperator` defines **no `Mult` at all**, only `ImplicitSolve`,
which is MFEM's own DIRK interface; and it cannot meaningfully define one,
the trace block having no time derivative. Reaching ARKODE means either
ARKODE's mass-matrix/DAE facilities or a reformulation — which is exactly the
"DAE questions proper" below, not a wiring job. MFEM's own SDIRK methods
already give orders 1 through 4 and are what the table above uses.

The DAE questions proper remain untouched: index, consistent initialisation of
the algebraic trace block, and stage-order reduction on the constraint under a
DIRK method. The `vdim == 1` refusal in the H(div) time mass, noted under §4,
is also still there.

## 9. Superconvergence at `k = 0` — the HHO-inspired methods

Optional, and it subsumes §4's remaining motivation if built. Cheaper than it
reads: two of the three ingredients are already available. `τ ~ 1/h` is the
built-in default scaling, and unequal flux/potential/trace orders are
unconstrained — nothing in `fem/darcy` ties the spaces' orders together, so
flux in `[P^k]^d`, potential in `P^{k+1}`, trace in `P^k` is constructible
today. Missing is the third: a stabilisation acting on the **L2 projection of
the potential onto the trace space** rather than on the potential itself.
`HDGStabilization` is a scalar hook that can rescale `τ` but cannot change what
`τ` multiplies, so this needs a new face integrator.

## 10. Interpolatory evaluation of the nonlinear coefficient

Optional, and now *purely* optional: the secondary payoff this entry used to
claim — that its step 2 is what makes the classic local postprocessing general
in `vdim` — has been overtaken, since that postprocessing is already general.
Nothing in `fem/darcy` interpolates a coefficient or holds a
`QuadratureFunction`.

## 11. NPC — Newton on the full system

**Built**: `DarcyHybridization::NPCResidual/NPCGradient/NPCReduce/NPCRecover`,
wrapped as `DarcyNPCOperator` + `DarcyNPCSolver`, serial and parallel, with
`[NPC]` cases in `tests/unit/fem/test_darcy_npc.cpp` — including this tree's
first `[Parallel]` Darcy test. `miniapps/hdg/navierstokes.cpp` is driven by it.
**The mechanism and every measurement are in the code**, on `NPCResidual()`;
`doc/HDG-ORDERING-API.md` §3 is the API reference for a caller.

What is left:

* **~~`convdiff` and `pconvdiff` still use `DarcyOperator`.~~ Done.**
  `DarcyOperator::SetNPC()` adds an NPC branch to `ImplicitSolve`, and both
  miniapps expose it as `-npc`; `navierstokes` bypasses `DarcyOperator`
  entirely. The trace right-hand side rides in the outer solver's own
  `b`, and the solver stack is reused as-is — `solver` was already the outer
  Newton and `prec` the trace solve, which is NPC's pairing — so `-gm` keeps
  its meaning.

  **The NPC reference set now exists**: 22 serial and 22 parallel `*_npc.txt`,
  taking the suite from 129 + 98 to 151 + 120. They compare the local
  nonlinear iteration count as well as the solver, the Krylov count and the
  two error norms — without that an NPC reference would pass even if `-npc`
  became a no-op, since both routes reach the same discrete solution. NPC runs
  no local nonlinear solve, so the count is identically zero and the check
  fails loudly if the flag stops taking effect.
* **H(div) flux.** Refused rather than attempted: the local rows would be a
  conforming scatter with RT sign conventions that have not been checked.
* **~~`ComputeSolution()`~~ checked.** It reconstructs the fields from the
  trace, which under NPC is redundant — the fields are already Newton state —
  but redundant is not wrong, and it was simply untested. "ComputeSolution
  reproduces the fields NPC already holds" in
  `tests/unit/fem/test_darcy_npc.cpp` pins the agreement, and the reason it
  must hold: NPC converges when the FULL residual vanishes, and that
  residual's local rows are exactly the problem `ComputeSolution()` solves
  given the trace.
* **A regression case is on offer and has not been taken.** The caller's
  transport barrier is the case that used to throw out of
  `NewtonSolver::Mult`'s `IsFinite` check at iteration zero, and nothing here
  reproduces that fault — the only evidence the divergence guard fixed it is
  theirs. They have offered to extract it from
  `tests/convergence/PedestalConvergence.cpp`. Worth taking before the guard is
  ever touched.

## Deliberately not being done here

The miniapps still default to the weak route for DG. Moving `convdiff` and its
siblings onto the essential-trace route is the branch author's call, not ours,
and it would move their regression references; it is being raised with them.
The same goes for the `-trbc` gap, which the library fix has closed but which
nothing in the suite exercises.

## References

Only those an *open* section still needs. The rest were moved into the doxygen
of the code that implements them, which is where they belong now.

* **CS-Extensions** — Cockburn & Solano, on solving problems posed on curved
  domains by extension from a polyhedral subdomain, reducing the boundary
  treatment to line integrals along transferring paths. §1.
* **CSS-Coupling** — Cockburn, Sayas & Solano, on coupling an HDG interior
  solve to an exterior boundary-integral representation across an unmeshed
  interface, with **CSS-Analysis** its companion, including the relaxed
  iteration and the contraction estimate. §2.
* **CCSZ-I** — Chen, Cockburn, Singler & Zhang, *Superconvergent interpolatory
  HDG methods for reaction diffusion equations I: an HDGk method*, J. Sci.
  Comput. **81** (2019) 2188–2212. Table 1 is the study §4 would compare
  against; the interpolatory idea is §10.
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
