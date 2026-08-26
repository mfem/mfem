# Defects in `fem/darcy`, found by an outside user

Four defects in the HDG implementation, found while building a hybridized
solver (`../meq`) on top of this branch. Nothing in this tree has been changed
in response to any of them; the application worked around all four, and this
file records what it worked around and why, so that the fixes can be made here
rather than rediscovered there.

**Companion to `HDG-ROADMAP.md` and written to the same rules.** Each defect is
stated as a property of the discretisation or of the API, in terms any HDG user
would recognise, so that a fix can be judged without reference to the
application. Each says explicitly whether it was **measured** or is a
**reading of the tree**, because two of the four are only readings and a
reviewer should be able to tell which.

**Written against `gf-hdg-subdomains-dev`.** §1 and §2 are in `darcyform.cpp`
and `bilininteg_hdg.cpp`, which both branches share, so they apply to
`gf-hdg-dev` unchanged. §3 and §4 concern `fem/darcy/extension_hdg.{hpp,cpp}`,
which exists only on the subdomains branch — on `gf-hdg-dev` those two sections
describe files that are not there. The line numbers throughout are from
`527cb4c74a`.

**Status, added by the branch.** §1 and §2 are fixed on `gf-hdg-dev`, each
with a regression that fails without the fix; **§3 is withdrawn — it is not a
defect**, and the paragraph at the end of it has the measurement; what was measured is in
`HDG-ROADMAP.md` §4 ("Three defects the reconstruction had, found from
outside") and §7. §4 is untouched here and belongs on
`gf-hdg-subdomains-dev`, which is where the file it concerns lives. Two things
came out differently from the reading below and are recorded where they are
described: §1 is worse than stated — the local problem loses its face
constraint as well as its mass, which is what makes it singular rather than
merely wrong — and §3's fix could not be the loud one, because refusing would
break the extension miniapp that relies on reconstructing — and then §3
turned out not to be a defect at all, the drop it reports being required
rather than tolerated, which the extension miniapp's new postprocessing pass
measures. A fifth defect,
not reported from outside, came out of writing §3's regression and is added at
the end; it is worse than any of the four.

**Ordered by how badly they fail, not by how hard they are to fix.** §1 returns
wrong numbers with no diagnostic and is the one that matters. §2 is the same
shape but narrower. §3 is measured harmless and is listed because "harmless on
one benchmark" is not "correct". §4 is a missing capability rather than a bug,
but it defeats a published error estimator, so it is here rather than in the
roadmap.

**The common thread is worth stating once.** Three of the four are cases where
a code path was extended for one configuration — a nonlinear potential mass, a
user supplied stabilization, a weakly imposed boundary datum — and a sibling
path that needed the same extension did not get it. In each case the untouched
path does not fail: it produces numbers. §1 produces numbers wrong by twenty
orders of magnitude.

---

## 1. `ReconstructFluxAndPot()` ignores the nonlinear potential mass form

**`fem/darcy/darcyform.cpp:1096`**, the block beginning `if (M_p)` at
**:1259**. **Measured.**

The local post-processing reads only `M_p`, the *linear* potential mass form:

```cpp
if (M_p)
{
   BilinearForm *Mp_s = reconstruction->GetPotentialMassForm();
   auto Mp_dbfi = *M_p->GetDBFI();
   ...
}
```

There is no `else if (Mnl_p)`. When the whole potential block lives on the
nonlinear form — which is what a Newton path on a solution-dependent source
requires, since a nonlinear potential mass and a linear one cannot be mixed —
`Mp_s` receives no integrators at all and `M_p_src` is left null. The local
problem `ReconstructFluxAndPot()` then builds has neither a potential mass nor
the constraint that would otherwise pin the element average, and it is solved
anyway.

**The same function knows better one screen earlier.**
`ReconstructTotalFlux()` at **:987** is written

```cpp
if (M_p && M_p->GetDBFI()) { ... }
else if (Mnl_p && Mnl_p->GetDNFI()) { ... }
```

so the asymmetry is internal to `DarcyForm`, not a limitation imposed from
outside it.

**What it costs.** On a benchmark with an exact solution, solved twice — once
through a linear form and once through Newton, with the two `p_h` agreeing to
six significant figures — the post-processed potential came back as

| | linear path | Newton path |
|---|---|---|
| mesh 1 | 3.8e-6 | **9.9e14** |
| mesh 2 | 2.4e-7 | **8.4e15** |
| mesh 3 | 1.5e-8 | **3.9e14** |

No abort, no warning, no `MFEM_VERIFY`. A caller that feeds `Reconstruct()`
into an error estimator — which is the principal reason to call it, since the
post-processed potential is what keeps the estimator of Sánchez-Vizuet, Solano
& Cerfon at optimal order — gets an indicator built on 1e15 and a refinement
pattern with no relation to the solution.

**Fix.** Mirror the `else if (Mnl_p)` branch that `ReconstructTotalFlux()`
already has. Failing that, and as a separate change worth making regardless,
refuse: an `MFEM_VERIFY` that the potential block is representable on the
enriched space turns twenty orders of silent error into a message. The
application currently throws from its own wrapper, which is a fix only for
that application.

**Fixed, and the reading above understates it.** The missing mass is not what
made the numbers 1e15. `DarcyHybridization` keeps a non-linear potential
constraint in `c_nlfi_p` and a linear one in `c_bfi_p`, and the local problem
read only the second — so with the potential block on the non-linear form it
had **no face constraint either**, and the matrix was singular. That is why the
in-tree reproduction returns `nan` rather than a large number: same defect,
different arithmetic on the way out. Both blocks are now taken as the Jacobian
frozen at the computed potential, which reduces to the `M_p` branch term for
term when the integrators are bilinear. The trace is not available to the
reconstruction, so a face term whose gradient would notice it — `convdiff
-nlc` — is refused with a message rather than linearised about a state that is
not the computed one. `convdiff -nx 8 -ny 8 -hb -dg -rec -anal -o 1 -nlp` went
from `nan` to agreeing with its linear twin to six figures.

---

## 2. `ComputeHDGFaceEnergy()` ignores the installed `HDGStabilization`

**`fem/darcy/bilininteg_hdg.cpp:1843`**. **A reading of the tree**, not a
measurement — the application has its own estimator and does not call this.

`HDGDiffusionIntegrator` routes every stabilization through `StabValue()`
(`bilininteg_hdg.hpp:199`), which is what gives a `SetStabilization()` hook its
effect: with a hook installed it divides out the quadrature weight, calls
`HDGStabilization::Eval()`, and multiplies the weight back. The call sites are

| function | line | routes through `StabValue()` |
|---|---|---|
| `AssembleFaceMatrix` | 1174, 1202, 1228, 1232 | yes |
| `AssembleHDGFaceMatrix` | 1400 | yes |
| `AssembleHDGFaceVector` | 1596 | yes |
| `AssembleHDGFaceGrad` | — | yes, by delegating when `IsConstant()` |
| **`ComputeHDGFaceEnergy`** | **—** | **no** |

`ComputeHDGFaceEnergy()` instead forms `w * (ni * nor)` directly from the
built-in expression, so it reports the energy of the `{h⁻¹Q}`-scaled
stabilization whatever hook is installed. With a constant `τ` — the choice both
Grad–Shafranov papers make, and the one an application installs a hook to get
in the first place — the two differ by the local mesh size and by the diffusion
coefficient, so the discrepancy grows under refinement rather than staying a
fixed factor.

**Where it surfaces.** `HDGErrorEstimator` (`fem/darcy/estimators_hdg.cpp`)
calls it for `Type::Energy` at **:174** and **:179**. `Type::Residual`, in the
same `switch`, goes through `AssembleHDGFaceVector()` and is therefore correct.
So one estimator class has two modes, one of which honours the hook and one of
which does not, and nothing says so.

**Fix.** Route the energy through `StabValue()` as the other four do. The
signature already carries `elfun` and `trfun`, so the state a non-constant hook
would need is in hand; the constant case needs nothing beyond the call.

**Fixed as stated**, state-dependent case included. With no hook the energy is
the same number to the last bit. The one thing the reading does not mention is
the anisotropic split: its direction weights are geometry and sum to `ni·nor`,
so normalising by that hands the split exactly the energy the hook produced,
whatever it was.

---

## 3. `ReconstructFluxAndPot()` lifts only domain integrators

**`fem/darcy/darcyform.cpp:1096`**, at **:1201** and **:1259**. **Measured, and
measured harmless** — which is why it is third rather than second.

Building the enriched form, the flux mass is copied as

```cpp
auto Mu_dbfi = *M_u->GetDBFI();
```

— domain integrators only. The potential mass gets `GetDBFI()` and `GetFBFI()`
but not `GetBFBFI()`. So a boundary-face integrator on either form is dropped
on the way to the enriched space.

That is exactly how `HDGExtensionIntegrator` is installed. `extension.cpp:355`
adds it with `AddBdrFaceIntegrator` on the flux mass form, because the
solution-dependent half of the transferred datum is element-local and must not
reach the hybridization. The reconstruction therefore solves its local problems
without the term that carries the boundary datum.

**And it does not seem to matter.** On the extension benchmark the
post-processed potential still converges at `k+2`: rates 2.62 and 3.00 at
`k = 1`, 3.46 and 3.90 at `k = 2`. The local problem is driven by the
reconstructed total flux and by the element average of `p_h`, and both of those
already carry the extension, which is apparently enough. This came out the
opposite way from the prediction, and is recorded because of that rather than
in spite of it.

**Fix, and the argument for making it anyway.** A silent drop that happens to
be benign on the cases tried is a defect waiting for a case where it is not,
and there is no diagnostic to tell those apart. Either copy the boundary-face
integrators, or refuse to reconstruct when the form carries integrators the
lift cannot represent. The second is cheap and is the one that fails loudly.

**Not a defect. Fixed, measured, and withdrawn.** Carrying the term was
implemented, and then the extension miniapp was given a postprocessing pass to
check it — `extension -rec` — which is the measurement this section always
needed and did not have. It is the drop that is required. At `k = 2` on
problem 1, the disc, where the whole computational boundary is transferred:

| | `‖p−p*‖` at `n=64` | rate | `‖u−u*‖` | rate |
|---|---|---|---|---|
| dropped | 1.58e-9 | **3.80** | 3.22e-8 | 3.63 |
| lifted | 8.57e-5 | **1.27** | 2.43e-4 | 1.25 |

`k+2` kept or lost, and a factor of 5e4 in the error. At `k = 1`, 2.43e-7 at
rate 2.82 against 1.07e-4 at rate 1.35.

**Why the reading was wrong.** The local problem is not the assembled problem
restricted to an element. Its trace unknown is free on *every* face, boundary
faces included, and is determined by the `⟨u_t·n, μ⟩` equation; the boundary
condition reaches it through the reconstructed total flux and the element
average — which is exactly the mechanism this section identified when it
recorded the drop as harmless, and it is the whole mechanism, not a lucky
part of one. `⟨L_e(u_h), v·n⟩` is one half of a boundary condition; the other
half, `⟨g∘a, v·n⟩`, is a linear-form term the local problem has no way to
know. One half imposes half a condition against a trace free to answer it,
and both halves would double-count the boundary flux against that same free
trace. The rates above are what half a condition costs.

Two things came out of the section anyway. The **argument for making it
anyway** — that "harmless on the cases tried" cannot be told from the cases
where it is not — is right in general and was the reason to build the
measurement; it just came back the other way. And the drop is now stated in
the class documentation of `DarcyForm`, with the numbers, rather than left to
be read as an oversight. A unit case pins it: one solve, reconstructed with
and without such a term installed, must give the same answer to the last bit.

**One thing does stand, and it turned out to be worse than this.**
`DarcyForm::Assemble()` builds the hybridized flux mass from
`ComputeElementMatrix()` too, so a boundary-face term on the flux mass needs a
pass of its own — `AssembleFluxMassBdrFaces()`. The claim made here, that
`gf-hdg-dev` has no equivalent, was wrong when it was written: the commit
withdrawing §3 carried that pass over with it. **What it did not carry was the
half that makes it work.** `AssembleFluxMassBdrFaces()` reaches the
hybridization through `DarcyHybridization::AssembleFluxMassMatrix()`, called a
second time for the element owning the face, and that routine *assigned*. So
the term did not fail to reach the solve; it reached it and replaced the
element's whole flux mass block, on every element touching the boundary,
silently. The hybridized flux differed from the monolithic one by more than
5% in the max norm. Fixed, with a
regression, in §10 of the roadmap.

---

## 4. The transferred datum `φ_h` is not reachable after a solve

**Not addressed on `gf-hdg-dev`: the files do not exist there.** It belongs on
`gf-hdg-subdomains-dev`.

**`fem/darcy/extension_hdg.{hpp,cpp}`.** **Measured consequence**, on the
application's estimator. A missing capability rather than a bug, but it defeats
a published error estimator, which is why it is here.

The extension method imposes

```
φ_h = g∘a + L_e(u_h)
```

on a face of `Γ_h`. The two halves have very different visibility:

* **`g∘a` is reachable.** `PathTraceCoefficient` (`extension_hdg.hpp:343`) is
  exactly that term and can be evaluated anywhere a `Coefficient` can.
* **`L_e(u_h)` is not.** It is formed inside
  `HDGExtensionIntegrator::AssembleFaceMatrix` (`extension_hdg.cpp:422`) as the
  local matrix `L`, contracted straight into `elmat`, and never stored or
  exposed. After the solve there is no way to ask what value was imposed on a
  given face.

**Why that is not merely inconvenient.** Any face-based indicator that compares
a computed trace against the datum actually imposed cannot be formed on `Γ_h`.
In eq. (20) of Sánchez-Vizuet, Solano & Cerfon that is `η₅`, and on the
extension path the trace unknown there is pinned rather than free, so the term
compares the post-processed potential against **zero**. The difference is then
`O(dist(Γ_h, Γ)) = O(h)` and swamps everything else: measured at `k = 2`,
`η = 4.09e-1` against `η₁ = 2.12e-3`, converging at about one half. Unmitigated,
an adaptive loop built on it runs, produces plausible pictures, and refines the
wrong elements.

The application excludes those faces from `η₅`. That restores `k+1` for the
total, and it is an **omission rather than a repair** — the term genuinely
carries information on `Γ_h` and it is being discarded.

**Fix.** The loop in `AssembleFaceMatrix` already computes `L` at each face
quadrature point. A method that contracts it against the element's flux dofs
and returns `L_e(u_h)` at a face integration point — or a `Coefficient` doing
the same, to sit beside `PathTraceCoefficient` — would make `φ_h` evaluable
with no new quadrature and no new geometry.

---

## A fifth, found here rather than there

**`DarcyHybridization::ReconstructTotalFlux()` writes into the stored
constraint blocks.** Not in this report, and worse than anything in it. The
face loop uses one `DenseMatrix` for the constraint block; on an interior face
`GetCtFaceMatrix()` *aliases* it onto `Ct_data`, and on a boundary face the
constraint integrator assembles into the same variable, which keeps the
aliased pointer whenever the shape already matches — always, on a uniform
mesh. The call's own answer is right and every miniapp number is unchanged,
but the object it was called on is left corrupt: a second `Reconstruct()`
moves `u_t` by half its norm, and `RecoverFEMSolution()` afterwards moves the
solution by more than the solution. Invisible to anything that reconstructs
once, wrong for anything that reconstructs in a loop.

**Found by trying to write the regression for §3**, which needs the same
solution reconstructed twice with and without the boundary term, and could not
be written until this was fixed. `HDG-ROADMAP.md` §4 has the measurements.

---

## Not defects, listed so they are not reported as such

**`DarcyForm::GetOffsets()` returns three entries, not four.** It never learns
about the trace space, and the four-entry version lives in
`DarcyOperator::ConstructOffsets()` in `miniapps/hdg/darcyop.hpp`, which is not
in `libmfem.a`. A caller outside the miniapps builds its own. That is a
documentation question at most.

**`DarcyHybridization::SetEssentialBC` with a nonlinear reduced operator.**
This *was* broken — `EliminateTraceTrueDofsInRHS` returned early for nonlinear
problems and the essential BC was silently ignored — and it is fixed, with the
reasoning commented in `darcyhybridization.cpp`. It is named here only because
`CLAUDE.md` records that no regression covers the combination, and a Dirichlet
problem solved by Newton is precisely that combination. An application hitting
a converged-but-wrong answer near the boundary would look here first, and
should not have to rely on a comment to know the path is uncovered.

---

## Where these came from

`../meq`, a fixed-boundary Grad–Shafranov solver built on this branch. It is a
hybridized Dirichlet problem, made nonlinear by a solution-dependent source and
solved by Newton over `DarcyForm::GetGradient`, on a curved boundary reached
through `fem/darcy/extension_hdg`. That combination — essential trace BC,
nonlinear reduced operator, user-supplied constant stabilization, extension —
is thinly covered here, which is why an outside user found four things in one
pass rather than none.

The measurements above are reproducible from that tree's convergence suite:
§1 from `tests/convergence/EstimatorConvergence.cpp`, §3 from
`ExtensionConvergence.cpp`, §4 from both. Every rate quoted is a measured
order of accuracy over a dyadic mesh sequence, not a tolerance.
