# HDG capabilities still wanted in `fem/darcy`

**This file, and every other `.md` here, is scratch.** It is a to-do list and
nothing else, and it is expected to be deleted before this branch becomes a PR.
Anything worth keeping must live in doxygen or in a source comment — the
findings that used to be written up here have been moved into the code that
they are about, and nothing in the code depends on a markdown file for its
meaning.

Sections keep the numbers they had, so earlier commit messages citing "§4"
still point somewhere sensible. Where a section is gone it says why.

## 1. Extension and lifting — solving on a subdomain of the true domain

**Built, on `gf-hdg-subdomains-dev`, not here.** `fem/darcy/extension_hdg.{hpp,cpp}`
(nine classes, including `HDGExtensionIntegrator` and
`TransferredDatumCoefficient`), `miniapps/hdg/extension.cpp`, and 27 unit test
cases in `tests/unit/fem/test_darcy_extension.cpp`.

What is left here is a **branch merge**, not development: the two branches are
21/13 diverged and neither has been merged into the other. Three artefacts of
§1 already sit on this branch and one of them is a bug:

* `fem/darcy/darcyform.hpp:174` refers the reader to `extension_hdg.hpp`, which
  does not exist here. `AssembleFluxMassBdrFaces()` exists solely to serve §1.
* `miniapps/hdg/extension` — an **18 MB ELF binary is committed**, added by a
  docs-only commit. That is an accident and should be removed whatever is
  decided about the merge.

**Two other sections are blocked on this merge and the dependency was not
previously recorded**: §3's remedy and §7's `η₅`. See both.

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
  `class FloorTau` in that same test file. So §3 is blocked on §1's merge.

The loss itself and the floor's repair are already pinned by regressions there.

## 4. Postprocessing for a system

**Smaller than it reads, and its motivation is largely retired.** The *classic*
local postprocessing is already general in `vdim` —
`HDGPotentialPostprocessor` in `fem/darcy/postprocess_hdg.hpp`, tested over
`neq = 1, 2, 3` — and so is `DarcyHybridization::ReconstructTotalFlux` and the
vector `VectorDivergenceGridFunctionCoefficient` the plan asked for. A
two-equation superconvergence study is therefore possible today.

What remains is the **rich reconstruction only**: `DarcyForm::ReconstructFluxAndPot`,
whose kernel has some forty distinct sites assuming `vdim == 1` — the closure
row replaces exactly one equation of the local system, and the interior-dof
count assumes a contiguous tail that `byNODES` does not give for `vdim > 1` —
plus the enriched *potential* and *trace* spaces (`darcyform.cpp:1178, :1201`),
which are built with no `vdim` argument while the flux space already takes one.
The gates are `darcyform.cpp:1016` and `:1140`.

Two consumers wait on it: the flux functional of the retired §6 below, which
would read out of bounds if handed a system, and the Navier-Stokes miniapp,
which cannot postprocess. A third thing would retire it entirely — §9.

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
shape of it, which makes this a second thing waiting on §1's merge.

A library constraint that bounds how far this can go:
`MixedConductionNLFIntegrator`'s HDG face stabilization for more than one
equation is `face_w * TauVar(e)`, one constant per equation set through
`SetVariableStabilization()`. It cannot express a stabilization depending on
the state or the face normal. The Navier-Stokes driver sidesteps it by carrying
the convective stabilization on the `NumericalFlux`; a *viscous* stabilization
varying with direction could not.

## 6. Functionals of the solution, evaluated from the numerical trace — DONE

`ComputeOutwardFlux` and `ComputeBoundaryFlux` in
`fem/darcy/functionals_hdg.{hpp,cpp}`, with two unit test cases over
`dim = 2, 3` and `order = 0, 1, 2`. The header carries the reasoning, including
why the implementation integrates the reconstructed total flux rather than the
pointwise `q̂` this entry used to name — that expression is single-valued for a
constant `τ` and **not** for a solution-dependent one.

Its limitations are `vdim == 1` (which is §4) and conforming matching meshes,
and `ComputeBoundaryFlux` returns a rank-local sum with no `MPI_Allreduce`.
Nothing outside its own tests calls it.

## 7. Adaptive refinement: `hp`, and the estimator's fifth term

`h`-adaptivity is done and tested. **`p` is untouched** — nothing in the tree
sets an element order or computes a smoothness indicator, so this is both the
`p` mechanism and the `h`-versus-`p` decision.

`η₅` of the SSC estimator is also open, and is blocked twice over.
`HDGErrorEstimator` has exactly two terms (`Type::{Residual, Energy}`) and
takes an integrator rather than a coefficient, so it needs an adapter or a
second entry point — that much the entry already said. What it did not say is
that **`TransferredDatumCoefficient`, the thing `η₅` would be built from, is
not on this branch**: it is §1's. Third item waiting on that merge.

## 8. Time integration of the DAE

**Not untouched, as this entry used to claim — the machinery exists and runs.**
`DarcyOperator` is a `TimeDependentOperator(IMPLICIT)` with `ImplicitSolve`;
`btime_u`/`btime_p` lift a `1/dt` mass onto either block, into the linear or
the nonlinear form, rebuilding the hybridization where it must; `convdiff` has
four ODE solvers behind `-ode` and four transient problems, one of them
nonlinear.

What is missing is everything that would make it trustworthy:

* **No verification at all.** All 129 regression references use
  `--ntimesteps 0` and exercise only steady problems, so the four transient
  problems have no reference. There is no unit test touching `DarcyOperator`,
  `btime_u`, `btime_p` or any ODE solver, and no temporal convergence table
  exists anywhere in the tree. **A temporal convergence study on `convdiff -p 4`
  is available today and is the cheapest genuinely new result on this list.**
* **The DAE questions proper are untouched** — index, consistent initialisation
  of the algebraic (trace/constraint) block, and stage-order reduction on the
  constraint under an SDIRK method. That is the hard part and none of it is
  addressed.
* The `vdim == 1` refusal in the H(div) time mass, noted under §4.

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
