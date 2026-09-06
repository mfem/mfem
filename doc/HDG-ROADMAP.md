# HDG capabilities still wanted in `fem/darcy` — the subdomains branch

**This file, and every other `.md` here, is scratch.** It is a to-do list and
nothing else, and it is expected to be deleted before this branch becomes a PR.
Anything worth keeping lives in doxygen, in a source comment, or — where it is
about how a miniapp is used — in that miniapp's header comment. Nothing in the
code depends on a markdown file for its meaning, and a section that is finished
is cut down to a pointer rather than left here describing itself.

**This file was 2604 lines and is now this.** It began as a requirements
document written outside the repository (`HDG-REQUIREMENTS.md`), and every
section accreted its own status, measurements and withdrawals until the
document was mostly a record of work already done — which is precisely what the
rule above says does not belong here. What was cut is not lost: it is in the
doxygen of the code it describes, in the miniapps' header comments, and in git.
Sections keep the numbers they had, so earlier commit messages citing "§4"
still point somewhere sensible.

## What this branch family is FOR, and it is narrower than a requirements list

**The job on every `gf-*` branch is to make classic NPC HDG work well** — the
Nguyen-Peraire-Cockburn method on the spaces this branch's users actually run,
a discontinuous L2 flux, an L2 potential and a `DG_Interface` trace,
hybridized, and the solver story around it.

**Fixing the original Darcy pathways is not an obligation.** Clearly inherited
and clearly not owed: the RT and broken-RT flux spaces, the two reductions, the
rich reconstruction (`ReconstructFluxAndPot`), and the `H1_Trace` (EDG) trace
space. A defect found in one of them is *recorded where the code is* and does
not become a to-do here. Directed work overrides this: when the caller asks for
one of them, it gets done.

**And this branch's own subject is §1.** Everything else below is either done,
someone else's, or a note.

## The branch topology, because several sections turn on it

```
gf-hdg-dev  (trunk)
  |- gf-hdg-subdomains-dev     extension/lifting  <-- this branch
  |- gf-hdg-linearise-first    NPC
  `- gf-hdg-p-adaptivity       per-face trace order
```

**These are reviewed by upstream separately and are not merged into each
other**, so a section whose machinery lives on a sibling is not blocked work —
it is work belonging to that branch's PR. Integration for `meq` happens in
`/home/ian/projects/mfem/mfem-src` on `meq-integration`, which carries all four
plus upstream master.

**Check what a branch contains with `git merge-base --is-ancestor` or
`git ls-tree`, never by reasoning about it.** That has been wrong here
repeatedly. This branch is 30 commits ahead of the trunk and 2 behind. Both
of those are no-ops in content: `0c3410ad51` is a CMake test-list fix whose
four lines this branch's own `09c1761e61` already has, and `14a69fdaad` deletes
a document this branch does not carry. **Content-neutral is not conflict-free,
though** — `git merge-tree` says the merge collides on
`tests/unit/CMakeLists.txt`, because this branch adds `test_darcy_extension.cpp`
and `test_darcy_singular.cpp` in the region the trunk edited. Resolve as ours;
the trunk lists no test file this branch lacks.

## 1. Extension and lifting — solving on a subdomain of the true domain

**Built, and this branch is where it lives.** `fem/darcy/extension_hdg.{hpp,cpp}`
(nine classes), `miniapps/hdg/extension.cpp`, and 29 unit cases in
`tests/unit/fem/test_darcy_extension.cpp`. The method is Cockburn & Solano's:
a Dirichlet datum given on the true boundary `Γ` is transferred to the
computational boundary `Γ_h` by line integrals along a family of paths, so the
design order survives a distance `dist(Γ_h, Γ) = O(h)` where earlier techniques
needed `O(h^{k+1})`.

**What it achieves, and where that is written**: the miniapp's header
comment carries the method, the three reproduced experiments and how to run
them; `extension_hdg.hpp`'s doxygen carries the contracts, the tiling property
the vertex-first construction exists for, and the `TransformBack` trap — that
`ElementTransformation::TransformBack()` *clamps*, so a point outside the
element comes back as a boundary point rather than as a failure. None of that
is repeated here.

Two things are left, and neither is what this list used to say:

* ~~The aerofoil's flux order~~ — **there is nothing to repair: the dip is
  pre-asymptotic and the rate recovers.** At the reference's own tail, order 1,
  the flux rate reads 2.08, 1.46, 1.53, **2.50** as `n` runs 32 to 256, coming
  back and overshooting to catch up the deficit once the mesh resolves a tail
  thinner than a mesh width. The blunt tail reads 2.01, 2.03, 2.01, 2.01
  throughout and the two error curves converge; the potential holds 2.00 in
  both the whole way. The measurements and the five controls that made each
  candidate innocent are in `miniapps/hdg/extension.cpp`'s header comment and
  on `VertexConePath`.

  Two real things came out of chasing it. `ExtensionRegionQuadrature()`'s
  weight is now signed, which took the tiling residual from 1.13e-2 to
  −2.29e-04 and is the same defect meq found in
  `ExtensionBoundaryQuadrature()`. And the cone `C(x)` is built but **off by
  default**: it closes nothing here, and meq reported that it costs the far
  face's quadrature — coverage stays exact, but the foot map roughens and a
  fixed-order rule under-resolves it. Their report and our reply were
  `doc/HDG-CONE-TILING-FROM-MEQ.md`, closed and deleted at `50124ac48d`; what
  the exchange established is on `VertexConePath`.

* **Three dimensions: RUN, and the dimension-generic half holds up.** A ball
  carved from a tetrahedral background mesh, `ClosestPointPath` onto the
  sphere, `p = sin x sin y sin z`. It compiles, assembles, solves and
  converges with no change to any library file. **The geometric control is the
  result worth having**: the swept regions tile `D_h^c` to 3.7e-11 / 2.0e-11 /
  3.5e-11 / 7.5e-11 at `n` = 8, 16, 24, 32, against 2-D's 1.6e-10 floor, so
  `ExtensionRegionQuadrature` with a TRIANGLE face rule is as exact as with a
  segment. Without the extension the flux rate is 0.94; with it, 1.81 — so the
  method earns its keep in 3-D. Order 2 climbs to k+1 (flux 1.63, 2.57;
  potential 1.36, 2.85).

  What is LEFT is two things:

  - **The order-1 flux rate is not settled and looks wrong.** Flux 1.81, 1.59,
    1.28 over `n` = 4, 8, 16, 24 while the potential converges properly at
    1.71, 1.75, 1.89. Verified solver-converged — a direct trace solve
    reproduces the iterative numbers to every printed digit. It is NOT
    concluded: this branch's own aerofoil went 2.08, 1.46, 1.53, 2.50 and
    recovered, so a degrading three-increment sequence is exactly the shape
    that needs the next refinement before it means anything.
  - **And that next refinement is blocked on the trace solve.** At `n` = 32
    (407k trace dofs) GMRES + Gauss-Seidel does not converge in 5000
    iterations, and a 3-D direct factorisation at that size is out of reach
    here. **A 3-D trace preconditioner is what stands between this and an
    asymptotic answer**, and it is the actual next task.

  Generalising the vertex search — `VertexConePath`, the only refusal in the
  whole of `extension_hdg`, at `extension_hdg.cpp:206`, written in `atan2` and
  half-circles — is not needed for any of the above and is now clearly
  optional rather than blocking.

## 2. Coupling at a distance to an exterior boundary-integral solve

**Untouched, and the largest item here by a wide margin** — no boundary-element
machinery exists anywhere in MFEM, so it is a from-scratch build of the
exterior representation rather than an HDG task.

It builds directly on §1 and so belongs on this branch rather than a sibling:
Cockburn, Sayas & Solano's `Σ_h`, `E_h(q_h)` and `L_h(g)` are `TransferPath`,
`ElementExtension` and `TransferredDatumCoefficient` term for term. A revised
request from `meq` lives on `gf-hdg-linearise-first` as
`doc/HDG-BEM-COUPLING-FROM-MEQ.md`; it asks for one integrator they will write
themselves plus an optimisation nobody needs yet, and **nothing in this tree
has to change for them to start**.

## 3. Genuinely general Darcy-like problems — DONE

**(a), (b), (c) and (f) are built and composed.** `anisodiff -p 11` is the
composing driver — a full varying conduction tensor, a convective term along
the strong direction and a volumetric sink in one operator — and it converges.

**(d) is settled, and half of what it used to say is withdrawn.** The
*degenerate* order loss is asymptotic and `HDGFloorStabilization` does recover
it; the sequences are on "HDG: a tau floor recovers the order a degeneracy
costs", pinned to `n = 128`. The *anisotropic* half — which claimed the same
floor bought back 1.49 → 2.00 — is **wrong, and was measured over a window
that was entirely pre-asymptotic**. Taken to `n = 256`, floored and unfloored
are identical to every printed digit from `n = 128` on, and the flux rate is
`k` either way. The withdrawal, the mechanism (a floor cannot change the
scaling of `τ`, only lift faces where the coefficient collapses, and that set
shrinks with `h` here) and the replacement rate table are on
`HDGFloorStabilization` and `HDGDiffusionIntegrator`.

The table is the durable part: the built-in `O(1/h)` `τ` gives flux `k` and
potential `k+2`, an `O(1)` `τ` gives `k+1` in both and is the theoretical
best, and anisotropy costs a further half order in the flux that no `τ` here
repairs.

**(e) is settled too, and it turned out to be (d)'s mechanism rather than a
second one.** The criterion for a singular reaction coefficient is whether the
*solution* meets the singularity, not whether the coefficient is integrable —
a non-integrable `γ/x²` against a boundary attains the best-approximation rate
while an integrable `γ/r` at a vertex loses two orders, and the same `γ/r` is
harmless once the solution vanishes on it. The report, the table and what to
do about it are doxygen on `DarcyForm::GetPotentialMassForm()`, where a caller
installs such a term; the four rows are pinned by
`tests/unit/fem/test_darcy_singular.cpp`. The change of variable the entry
recommended is confirmed as advice and its stated justification withdrawn: it
is worth an order because it makes the *solution* smooth, not because the
discretisation finds the singular chart harder — it is optimal in both.

The number is kept so commit messages citing "§3" land somewhere.

## 4. Systems of coupled nonlinear Darcy-like problems, with exact Jacobians

**Built; the coupling is through the hyperbolic flux, not the diffusive one** —
`VectorBlockDiagonalIntegrator` replicates one integrator down the diagonal and
cannot express an off-diagonal block, so every cross-equation term comes from
the hyperbolic integrator. That, and the exact-Jacobian work under it, is in
the doxygen of the integrators concerned.

What is left here is **not this branch's**, and both halves say so:

* The **rich reconstruction is still scalar on this branch** —
  `MFEM_VERIFY(fes_p->GetVDim() == 1)` at `darcyform.cpp:1016` and `:1140` —
  and making it general in `vdim` is done on `gf-hdg-linearise-first`, along
  with the per-field flux functionals and the coupled nonlinear system case.
  It is an inherited pathway either way; see the scope note.
* The **hyperbolic closure question** that work turned up is recorded at the
  closure itself on that branch and is not owed here.

## 5. `τ` for problems that are convection- and diffusion-dominated at once
— ANSWERED

**No, one scalar cannot serve both, and the obstruction is symmetry rather
than magnitude.** `Eval()` on `HDGStabilization` returns one number per
quadrature point, which the integrator applies to both sides of a face, so
every `τ` reachable through that hook is symmetric in the sign of `u·n`.
Upwinding is exactly the antisymmetric half, `±½α(u·n)`. That is a property of
the interface and is now written on it.

The measurements are on `anisodiff -p 11`'s convection block (what the centred
and upwinded fluxes are each worth, by order and by `c`) and on
`HDGConvectiveFloorStabilization` (the `β_c` sweep showing magnitude is not
the missing ingredient). `-up`, `-vs` and `-tc` are the knobs that reach them.
None of it is repeated here.

**What is left, and it is a different question from the one this section
asked.** The split is 9.5x better where convection dominates and 30% *worse*
where diffusion does, so the choice is per face rather than per problem, and
nothing selects it per face today. A convection integrator whose upwind
strength varies with the local balance is the shape of that, and it is a new
integrator rather than a stabilization hook — which is precisely what the
symmetry finding above says. Not started, and not obviously owed: the scope
note's line is that the *classic* NPC method is the job, and NPC section 2.4
prescribes the plain split.

## 6. Functionals of the solution — one thing left

`fem/darcy/functionals_hdg.hpp` carries what it does and does not, for one
field, which is what this branch's callers run.

**One claim here was wrong and is withdrawn.** This entry said the scalar entry
points "refuse a system loudly, which is correct behaviour rather than a gap".
There is no `vdim` check anywhere in `functionals_hdg.{hpp,cpp}` on this branch,
so nothing refuses. `FaceNormalFlux()` reads the flux with
`GridFunction::GetVectorValue()`, and at `vdim > 1` that lands on one of two
size mismatches depending on the space's range type: a vector-valued element
takes the documented `GetVectorValue()` defect, `vshape.MultTranspose(loc_data,
val)` applying a `dof`-row `vshape` to a `dof*vdim` `loc_data`; a scalar-valued
one returns `val` at length `vdim` to meet a `nor` of length `dim` in the dot
product here. **Both are guarded by `MFEM_ASSERT` only, and this tree configures
`MFEM_DEBUG = NO`** — so in the build anyone here actually runs, both are
compiled out and a system gets a number rather than a diagnostic.

The per-field version *and* the `MFEM_VERIFY` are on `gf-hdg-linearise-first`
(`functionals_hdg.cpp:97` there); neither was ever here.

**What is NOT claimed**: that anything reaches this. No caller on this branch
builds a total flux at `vdim > 1` — that is why the refusal was written on the
sibling, where systems exist, and why this is a one-line carry-back rather than
a defect report. Recording it that way is this file's own rule about "X is
unguarded" being two claims. Carrying the refusal back is the only thing left in
this section. The number is kept so commit messages citing "§6" land somewhere.

## 7. Adaptive refinement, and the estimator's fifth term

**`h` is done and tested. `p` is `gf-hdg-p-adaptivity`'s.**

**`η₅` is built.** `HDGDatumErrorEstimator` in `fem/darcy/estimators_hdg.hpp`,
pinned by "the estimator's boundary-datum term" in
`tests/unit/fem/test_darcy_extension.cpp`. It is a class of its own rather than
a third `Type` because the other two are built from an HDG face integrator and
this compares a field against a *coefficient*; why it matters, and the two
ordering constraints the transferred datum imposes on any caller, are on the
class.

Nothing is left in this section beyond what a caller does with it: assembling
the five terms of the SSC estimator into one indicator and driving a refiner
with it is an application's business, not the library's.

## 8. Time integration of the DAE

**The integrators work.** `DarcyOperator` is a `TimeDependentOperator(IMPLICIT)`
with four ODE solvers behind `-ode` and observed temporal orders 1 to 4; the
table is in `miniapps/hdg/convdiff.cpp`'s header comment.

What is left is verification and theory — unchecked transient problems, no
transient regression reference, and the DAE questions proper (index, consistent
initialisation of the algebraic trace block, stage-order reduction on the
constraint under a DIRK method). **Whether this is ours has not been decided**;
it has had real work on the sibling branch and the scope note does not assume
either way.

**ARKODE is present and not usable here**, which is worth knowing before anyone
tries to wire it: its implicit path drives an explicit `f(t, y)` and
`DarcyOperator` defines no `Mult` at all, the trace block having no time
derivative.

## 9. A driver, attempted and withdrawn — DONE

A Stokes-shaped driver was built here and removed; §3's composition is served
by `anisodiff -p 11` instead. The number is kept for commit messages.

## 10. Three loose ends — DONE

The flux-mass boundary pass, the essential-trace route for RT, and the constant
null mode. All three settled, and two of them said something other than the
entry claimed. The findings are in the code and in
`tests/unit/fem/test_darcy_nullmode.cpp`; the number is kept for commit
messages.

## Optional A. Interpolatory evaluation of the nonlinear coefficient — WITHDRAWN

**Measured, and not worth building.** The measurement lives on the two rule
selections in `MixedConductionNLFIntegrator`, `fem/nonlininteg_mixed.cpp`,
which is what it is about; the two `//<---` markers that used to flag those
lines as uncertain are answered and gone.

The short of it. The entry's payoff was the over-integration an interpolant
lets you skip, and **nothing here over-integrates** — bumping the rule by 16 on
convdiff's nonlinear diffusion takes 193k quadrature points to 2.6M and moves
neither L2 error in any printed digit. Nor would an interpolant evaluate the
flux law less often: this rule carries `(k+1)^d` points on a tensor-product
element and an L2 space of order `k` has `(k+1)^d` dofs, so the ratio is 1.00
at every order. On the Gauss-Lobatto basis the miniapps build with, it is 4x
to 260x *less* accurate than the rule it would replace; on the L2 default
Gauss-Legendre basis it is bit-identical, the nodes being the quadrature
points. The one structural win — an `O(nq)` cheaper gradient — is capped at
6-11% of a solve, the two element routines being 9-21% of one.

**CCSZ-I is not wrong**; its cost model assumes an over-integration this code
does not do. The number is kept so commit messages citing "Optional A" land
somewhere.

## Optional B. Superconvergence at `k = 0` — the HHO-inspired methods

Optional. Cheaper than it reads: two of the three ingredients are already here.
`τ ~ 1/h` is the built-in default scaling, and nothing in `fem/darcy` ties the
flux, potential and trace orders together, so flux in `[P^k]^d`, potential in
`P^{k+1}`, trace in `P^k` is constructible today. Missing is the third: a
stabilisation acting on the **L2 projection of the potential onto the trace
space** rather than on the potential itself. `HDGStabilization` can rescale `τ`
but cannot change what `τ` multiplies, so this needs a new face integrator.

## Deliberately not being done here

The miniapps still default to the weak route for DG. Moving `convdiff` and its
siblings onto the essential-trace route is the branch author's call, not ours,
and it would move their regression references; it is being raised with them.
The same goes for the `-trbc` gap, which the library fix has closed but which
nothing in the suite exercises.

## References

Only those an *open* section still needs. The rest moved into the doxygen of
the code that implements them.

* **CS-Extensions** — Cockburn & Solano, on solving problems posed on curved
  domains by extension from a polyhedral subdomain. §1, and §2.4.1 is the cone
  restriction §1 still wants.
* **CSS-Coupling** — Cockburn, Sayas & Solano, coupling an HDG interior solve
  to an exterior boundary-integral representation across an unmeshed
  interface, with **CSS-Analysis** its companion. §2.
* **SSC** — Sánchez-Vizuet, Solano & Cerfon; eq. (20) is the estimator whose
  fifth term §7 wants.
* **CCSZ-I** — Chen, Cockburn, Singler & Zhang, *Superconvergent interpolatory
  HDG methods for reaction diffusion equations I*, J. Sci. Comput. **81**
  (2019) 2188. Its Table 1 is the `k = 0` limit Optional B is about, which is
  why it is still here; Optional A, which it was the reference for, is
  withdrawn.
* **CCSZ-II** — *… II: HHO-inspired methods*, Commun. Appl. Math. Comput. **4**
  (2022) 477. Optional B.
