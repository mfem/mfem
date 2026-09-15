# HDG capabilities still wanted in `fem/darcy`

**This file, and every other `.md` here, is scratch.** It is a to-do list and
nothing else, and it is expected to be deleted before this branch becomes a PR.
Anything worth keeping lives in doxygen, in a source comment, or — where it is
about how a miniapp is used — in `miniapps/hdg/README.md`. Nothing in the code
depends on a markdown file for its meaning, and a section that is finished is
cut down to a pointer rather than left here describing itself.

**This used to be a requirements document**, written outside the repository as
`HDG-REQUIREMENTS.md` before any of the work, and it grew to 2004 lines because
every section had a natural place to accrete its own status. A to-do list and a
requirements list are different documents and this is now only the first; the
descendants were pruned the same way, `gf-hdg-subdomains-dev` first. What left
here went into the code, not into another file.

## Section numbers do NOT agree across the branch family

Two schemes are in use and neither can be renumbered without breaking commit
messages on its own branches, so here is the concordance instead. **Check which
branch a commit message is on before following a `§` in it.**

| number | here, `gf-hdg-subdomains-dev`, `gf-hdg-p-adaptivity` | `gf-hdg-linearise-first`, `gf-interp-hdg-dev` |
|---|---|---|
| §3 | Genuinely general Darcy-like problems | Whether the degenerate order loss is asymptotic (was §3(d) here) |
| §4 | Systems of coupled nonlinear problems | Postprocessing for a system (was part of §4 here) |
| §9 | A driver, attempted and withdrawn | Superconvergence at `k = 0` (is Optional B here) |
| §10 | Three loose ends, swept | Interpolatory evaluation (is Optional A here) |
| §11 | — | NPC |

§1, §2, §5, §6, §7 and §8 mean the same thing in both.

## What this branch family is FOR, and it is narrower than a requirements list

**The job on every `gf-*` branch is to make classic NPC HDG work well** — the
Nguyen–Peraire–Cockburn method on a discontinuous L2 flux, an L2 potential and
a `DG_Interface` trace, hybridized, and the solver story around it.

**Fixing the original Darcy pathways is not an obligation.** Clearly inherited
and clearly not owed: the RT and broken-RT flux spaces, the two reductions, the
rich reconstruction (`ReconstructFluxAndPot`) and the `H1_Trace` (EDG) trace
space. A defect found in one of them is *recorded where the code is* and does
not become a to-do here. This file repeatedly grew entries that were true,
interesting and nobody's job; check a candidate against this before adding one.

## The branch topology, because most sections turn on it

```
gf-hdg-dev  (trunk)  <-- this branch
  |- gf-hdg-subdomains-dev     extension/lifting       -> its own PR
  |- gf-hdg-linearise-first    NPC                     -> its own PR
  |    `- gf-interp-hdg-dev    interpolatory HDG       -> its own PR
  `- gf-hdg-p-adaptivity       per-face trace order    -> its own PR
sundials-ida-integration       an IDASolver, off master -> its own PR
direct-solver-symbolic-reuse   off master, PR open
```

**These are reviewed by upstream separately and are not merged into each
other.** A section whose machinery lives on a descendant is not blocked work —
it is work belonging to that branch's PR. The trunk carries what is common:
a defect fixed here is merged OUT to the descendants, which is how the face
quadrature, the null prolongation, the periodic-boundary guard, the transient
fixes, `-prec` and the local reference baseline all reached them.

Integration for `meq` happens in `/home/ian/projects/mfem/mfem-src` on
`meq-integration`, which carries the HDG branches plus upstream master.

## 1. Extension and lifting — solving on a subdomain of the true domain

**Built, on `gf-hdg-subdomains-dev`, not here.** `fem/darcy/extension_hdg.{hpp,cpp}`,
`miniapps/hdg/extension.cpp`, and 29 unit cases. Cockburn & Solano's method:
a Dirichlet datum on the true boundary is transferred to the computational one
along paths, so the design order survives `dist(Γ_h, Γ) = O(h)`.

Not a merge task. What follows for the sections wanting its machinery (§7's
`η₅`) is that they are actionable there and on `meq-integration`, not here.

Left there: the third dimension beyond the geometric control, whose blocker is
a 3-D trace preconditioner rather than the method.

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

## 3. Genuinely general Darcy-like problems

**(a)–(c) and (f) are built.** A full varying conduction tensor, a reaction
term, a convective term in the mixed form, and their composition. What is on
this branch is the machinery; the driver that *composes* all of it in one
operator is `anisodiff -p 11` and lives on `gf-hdg-subdomains-dev`, which is
why §3 reads DONE there and not here.

**(d) — whether the degenerate order loss is asymptotic — is the one piece
still open**, and it is a measurement rather than a build. The practical answer
is known: floor the stabilisation. `Rates()` in
`tests/unit/fem/test_darcy_degenerate.cpp` runs n = 4 to 16 and overwrites its
rate at each refinement, so it reports the last of two; settling the question
needs a deeper sweep that keeps the whole sequence. `HDGFloorStabilization` is
`gf-hdg-subdomains-dev`'s, so the repair is pinned there.

## 4. Systems of coupled nonlinear Darcy-like problems, with exact Jacobians

**Built; the coupling is through the hyperbolic flux, not the diffusive one.**
`VectorBlockDiagonalIntegrator` replicates one integrator down the diagonal and
cannot express an off-diagonal block, so every cross-equation term comes from
the hyperbolic integrator. That, and the exact-Jacobian work under it, is in
the doxygen of the integrators concerned.

**The rich reconstruction is scalar on this branch** —
`MFEM_VERIFY(fes_p->GetVDim() == 1)` in `darcyform.cpp` — and making it general
in `vdim`, along with the per-field flux functionals, is done on
`gf-hdg-linearise-first`. It is an inherited pathway either way; see the scope
note. The hyperbolic closure question that work turned up is recorded at the
closure itself on that branch.

## 5. `τ` for problems that are convection- and diffusion-dominated at once

**Measured, on `gf-hdg-linearise-first`**, whose `miniapps/hdg/navierstokes.cpp`
header carries the tables and the mechanism: the direction-aware
`S = λ_max(û,n) I` is 2.0–3.6× *worse* than the best constant `τ` in accuracy,
and better than any constant only at keeping Newton alive on coarse meshes at
high `Re`.

**What is left is a problem, not a method**: both exact solutions there put
their structure across the flow, which is where `λ_max` does not differ from
`√β`. A genuinely two-directional exact solution would settle it, and
`anisodiff -p 11` is the linear-diffusion shape of one — so that half belongs
to `gf-hdg-subdomains-dev`.

## 6. Functionals of the solution

**Built here**, `fem/darcy/functionals_hdg.hpp`, which carries what it does and
does not. The per-field (`vdim > 1`) read is `gf-hdg-linearise-first`'s.

## 7. Adaptive refinement

**`h` is done and tested here. `p` is `gf-hdg-p-adaptivity`'s**, where the plan
and its five completed steps live; read it there. Two prerequisites for it were
paid for on this branch and are why it could start at all: the HDG face
quadrature now sees the trace element's order, and `DarcyOperator` survives a
hanging-node-free NC mesh.

The finding that made `p` a branch rather than a patch: the element spaces are
**already** `p`-adaptive and need no library change, but it buys nothing on its
own, because **the trace order sets the rate**.

**`η₅` of the SSC estimator is open and is not actionable here.**
`HDGErrorEstimator` has exactly two terms (`Type::{Residual, Energy}`) and takes
an integrator rather than a coefficient; and the datum it would be built from is
§1's. Built on `gf-hdg-subdomains-dev` as `HDGDatumErrorEstimator`.

## 8. Time integration of the DAE

**The integrators work and the miniapp side is verified — that happened on this
branch.** `DarcyOperator` is a `TimeDependentOperator(IMPLICIT)` with
`ImplicitSolve`; `convdiff` and `pconvdiff` reach four ODE solvers through
`-tf`, `-nt` and `-ode`. Two defects were fixed on the way: the error was taken
against the exact solution at the last STAGE time rather than the end of the
step — so the higher the formal order, the worse the miniapp made the method
look, with backward Euler exactly unaffected as the control — and problem 4's
exact solution spread as `2σ² + 4kt·π/4` where the PDE requires `2σ² + 4kt`.

**Problems 7 and 9 carry transient references**, four each per arm, sixteen in
all, one per ODE solver, at `-tf 0.5 -nt 2` — coarse enough that the four `-ode`
answers are separated by far more than the suite's `1e-4`, so each discriminates
against the other three rather than merely existing.

What is left:

* **The DAE questions proper**: index, consistent initialisation of the
  algebraic trace block — which problems 7 and 9 dodge by starting from zero
  rather than answering — and stage-order reduction on the constraint under a
  DIRK method, predicted from the tableaux and measured on
  `gf-interp-hdg-dev`'s `convdiff.cpp` header but not repeated here.
* **Problem 4 is the only one with time-varying boundary data and has no
  reference.** Problem 5 is not a temporal study and never was: `GetQFun`
  returns zero for `KovasznayFlow`.
* **ARKODE is present and not usable here.** Its implicit path drives `RHS1`,
  an explicit `f(t, y)`, and `DarcyOperator` defines no `Mult` at all and cannot
  meaningfully define one, the trace block having no time derivative.

**An `IDASolver` now exists, on `sundials-ida-integration`** — off
`origin/master`, with `linalg/sundials.{cpp,hpp}`, a unit test file and a design
note in `doc/SUNDIALS-IDA.md`. This section used to name "write an `IDASolver`
against MFEM's `ODESolver`/`SundialsSolver` interfaces" as one of three
hypothetical ways out and told the reader to decide early; the decision has been
taken and built, and nothing here has been wired to it.

## 9. A driver, attempted and withdrawn — DONE

Nothing left. The number is kept so commit messages citing "§9" land somewhere.
Note that `§9` means something else on `gf-hdg-linearise-first` and
`gf-interp-hdg-dev`; see the concordance above.

## 10. Three loose ends, swept — DONE

Nothing left. The same warning about `§10` applies.

## Optional A. Interpolatory evaluation of the nonlinear coefficient

**Built, on `gf-interp-hdg-dev`, not here**, where the plan
(`doc/HDG-INTERPOLATORY-CCSZ.md`) and all seven of its stages live. Chen,
Cockburn, Singler & Zhang, *J. Sci. Comput.* **81** (2019) 2188–2212.

Nothing in `fem/darcy` on THIS branch interpolates a coefficient or holds a
`QuadratureFunction`; that sentence is false on `gf-interp-hdg-dev` and is a
statement about this branch, not about the family.

## Optional B. Superconvergence at `k = 0` — the HHO-inspired methods

**Open, optional, and the largest remaining build that is ours** — which is not
saying much: two of the three ingredients already exist. `τ ~ 1/h` is the
built-in default scaling, and unequal flux/potential/trace orders are
unconstrained, so flux in `[P^k]^d`, potential in `P^{k+1}`, trace in `P^k` is
constructible today. Missing is the third: a stabilisation acting on the **L2
projection of the potential onto the trace space** rather than on the potential
itself. `HDGStabilization` is a scalar hook that can rescale `τ` but cannot
change what `τ` multiplies, so this needs a new face integrator.

## Deliberately not being done here

The miniapps still default to the weak route for DG. Moving `convdiff` and its
siblings onto the essential-trace route is the branch author's call, not ours,
and it would move their regression references; it is being raised with them.
The same goes for the `-trbc` gap, which the library fix has closed but which
nothing in the suite exercises.

## References

Only those an *open* section still needs.

* **CS-Extensions** — Cockburn & Solano, on solving problems posed on curved
  domains by extension from a polyhedral subdomain. §1.
* **CSS-Coupling** — Cockburn, Sayas & Solano, on coupling an HDG interior
  solve to an exterior boundary-integral representation, with **CSS-Analysis**
  its companion. §2.
* **CCSZ-II** — Chen, Cockburn, Singler & Zhang, *… II: HHO-inspired methods*,
  Commun. Appl. Math. Comput. **4** (2022) 477–499. Its Table 1 classifies the
  three variants; (A) and (B) are superconvergent from `k = 0`, (C) only from
  `k = 2`, and all three take `τ ~ 1/h`. Optional B.
* **Lehrenfeld–Schöberl** — HDG+, the same object as CCSZ-II's HDG (A), with
  **Oikawa** arriving at it independently. Optional B.
