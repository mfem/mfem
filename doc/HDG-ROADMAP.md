# HDG capabilities still wanted in `fem/darcy`

What is left. Everything this file used to record about work already done has
been removed; the code and its doxygen carry that now, and the history is in
git. Sections are numbered as they were, so earlier commit messages that cite
"§4" still point somewhere sensible.

## 1. Extension and lifting — solving on a subdomain of the true domain

Untouched. Sits on its own branch of the dependency graph: nothing in the
completed work blocks it.

## 2. Coupling at a distance to an exterior boundary-integral solve

Untouched, and independent of §1 in the same way.

## 3. Whether the degenerate order loss is asymptotic

The practical answer is already known — floor the stabilisation — but whether
the loss is asymptotic or pre-asymptotic was never settled.

## 4. Postprocessing for a system

The reconstruction is scalar-only, so a two-equation study cannot be
postprocessed and its superconvergence table is a single field. Making it
general in `vdim` needs: the enriched spaces built with a `vdim`;
`GetElementVDofs` throughout the kernel; a vector
`DivergenceGridFunctionCoefficient`; a `vdim`-aware
`DarcyHybridization::ReconstructTotalFlux` with a vector-valued callback; and a
linearised potential constraint for a system stabilized by the nonlinear face
integrator.

Two things qualify it. The *classic* local postprocessing — the small
per-element solve of CCSZ, not the branch's flux-and-potential reconstruction —
is a loop over equations away from being general in `vdim`. And §9 below would
remove the need entirely for the quantity that matters, since HDG (A) is
superconvergent as solved.

## 5. `τ` for problems that are convection- and diffusion-dominated at once

**Measured, on the Navier-Stokes driver, and the headline answer is negative.**
937 runs of `miniapps/hdg/navierstokes.cpp` — Kovasznay and plane Poiseuille,
orders 1-3, `Re` 10 to 1000, cell aspect ratios 1/4 to 4, `τ` 0.125 to 10 and
`β` over a factor of 16. The tables are in `doc/HDG-NAVIER-STOKES.md` under
*The §5 measurement*; the three results are:

1. **The direction-aware `S = λ_max(û,n) I` is 2.0-3.6x worse than the best
   constant `τ` in the flux and the pressure, and indistinguishable from it in
   the potential.** Over-stabilization also costs the flux its rate: 2.60 down
   to 2.18 at `k = 2` as `τ` goes 0.5 to 5.
2. **The mechanism, established by a `β` sweep rather than by inspection.**
   `β` cannot change the steady answer — the continuity row is `β ∇·v = s_0` —
   but it sets `λ_max = √β` on every face where `v·n = 0`. `λ_max`'s error
   tracks the constant `√β` to within 5-16% and *moves with `β`*, while a fixed
   `τ` is `β`-independent to 0.02%. So `λ_max` is a constant `√β` in disguise
   on the faces that do the work, its level is set by an arbitrary parameter of
   the formulation rather than by the physics, and the extra weight it carries
   on the along-flow faces is a penalty.
3. **It wins the other half.** `λ_max` converged on every one of ~300 cases;
   every constant `τ ≤ 1` diverges somewhere, on *coarse* meshes at high `Re`,
   recovering under refinement. Cold on plane Poiseuille at `Re = 100`, `λ_max`
   converges in 9 iterations where `τ ∈ [0.25, 2]` all fail and only `τ ≥ 5`
   recovers — so it is not the *amount* of stabilization but where it is put.
   And the accuracy optimum sits exactly at that robustness boundary: at
   `Re = 400` the best converging `τ` is 0.375 at 48x32 and 0.5 at 24x16.

**The mesh aspect ratio changes none of it** — `λ_max`'s penalty is if anything
largest on cells stretched along the flow.

**What is still open, and why.** Both of the driver's exact solutions put their
sharp structure across the flow and little or none along it, so the along-flow
faces — the only ones where `λ_max` differs from `√β` — are exactly the faces
where the solution is easiest to represent. Kovasznay cannot repair this on its
own window, because its decay rate `λ = Re/2 − √(Re²/4 + 4π²) → −4π²/Re` means
the parameter that makes it convective is the parameter that flattens its
along-flow structure: 94x of variation at `Re = 10`, **1.16x at `Re = 400`**,
with the measured consequence that errors are identical to four digits over a
16x range in `nx`. A longer window at moderate `Re` is a partial repair and is
what the aspect-ratio table above uses. **A genuinely two-directional solution
is what would settle the general question, and this miniapp has none.**
`anisodiff -p 11` on `gf-hdg-subdomains-dev` is the linear-diffusion shape of
it.

A constraint on how far the existing library can go:
`MixedConductionNLFIntegrator`'s HDG face stabilization for more than one
equation is `face_w * TauVar(e)`, one constant per equation set once through
`SetVariableStabilization()`. It cannot express a stabilization depending on
the state or the face normal. The Navier-Stokes driver sidesteps it by carrying
the convective stabilization on the `NumericalFlux`; a *viscous* stabilization
that varies with direction would not be able to.

One methodological point survives from a withdrawn table because it cost a day:
**rates must be taken asymptotically.** The same configurations read 1.6 rather
than 2.5 on the coarsest pair of meshes, which is enough to condemn a correct
`τ`.

## 6. Functionals of the solution, evaluated from the numerical trace

Independent of everything else and small. Compute a surface integral of the
numerical flux `q̂ · ν` over a prescribed internal or boundary surface as a
first-class quantity. `q̂ = q_h + τ(u_h - λ)ν` is single-valued on faces by
construction — that is what hybridization is — so the integral is consistent
with the discrete conservation statement rather than an after-the-fact
diagnostic. For a problem whose answer *is* a small flux, that is the
difference between an accurate result and catastrophic cancellation.

## 7. Adaptive refinement: `hp`, and the estimator's fifth term

`hp` is open. So is `η₅` of the SSC estimator: it is buildable on `Γ_h` from
`TransferredDatumCoefficient`, but `HDGErrorEstimator` takes an integrator, not
a coefficient, so it needs an adapter or a second entry point.

## 8. Time integration of the resulting DAE

Untouched, in its entirety.

## 9. Superconvergence at `k = 0` — the HHO-inspired methods

Optional, and it subsumes §4's motivation if built.

## 10. Interpolatory evaluation of the nonlinear coefficient

Optional. Step 2 of it is also what makes the classic local postprocessing
general in `vdim`; see §4.

## Deliberately not being done here

The miniapps still default to the weak route for DG. Moving `convdiff` and its
siblings onto the essential-trace route is the branch author's call, not ours,
and it would move their regression references; it is being raised with them.
The same goes for the `-trbc` gap, which the library fix has closed but which
nothing in the suite exercises.

## References

Cited by the short labels used above. Full bibliographic detail is given only
where this file recorded it at the time; the rest are identified by author and
subject.

* **NPC-1** — Nguyen, Peraire & Cockburn, *An implicit high-order hybridizable
  discontinuous Galerkin method for linear convection–diffusion equations*,
  J. Comput. Phys. **228** (2009) 3232–3254. §3.6 gives the stabilisation
  `s = s_d + s_c` with `η_c = |c·n|` and `η_d = κ/ℓ`, `ℓ` a fixed problem length
  scale; Table 1 is the convergence study §5 reproduces; **§4 is the two-pass
  reconstruction the branch implements**, and is what `DarcyHybridization`'s
  header cites.
* **NPC-2** — Nguyen, Peraire & Cockburn, *An implicit high-order hybridizable
  discontinuous Galerkin method for nonlinear convection–diffusion equations*,
  J. Comput. Phys. **228** (2009) 8841–8855. Eq. (5) is the numerical flux with a
  solution-dependent `s`; Eq. (7) the positivity bound; Eq. (15)–(16) the Newton
  linearisation and its block structure.
* **CCSZ-I** — Chen, Cockburn, Singler & Zhang, *Superconvergent interpolatory
  HDG methods for reaction diffusion equations I: an HDGk method*, J. Sci.
  Comput. **81** (2019) 2188–2212. The nonlinear term is interpolated
  elementwise and evaluated at the postprocessed solution, so the HDG matrices
  assemble once; Table 1 is the convergence study §4 compares against, and
  Theorem 3.19 the `k ≥ 1` hypothesis.
* **CCSZ-II** — Chen, Cockburn, Singler & Zhang, *… II: HHO-inspired methods*,
  Commun. Appl. Math. Comput. **4** (2022) 477–499. Table 1 there classifies
  three variants: (A), the Lehrenfeld–Schöberl / HDG+ method with the scalar in
  `P^{k+1}`, and (B), with an HHO stabilisation acting on the postprocessed
  trace, both superconvergent from `k = 0`; (C) only from `k = 2`. All three
  take `τ ~ 1/h`.
* **CSZ-Interpolatory** — Cockburn, Singler & Zhang, *Interpolatory HDG method
  for parabolic semilinear PDEs*, J. Sci. Comput. **79** (2019) 1777–1800. The
  interpolatory method without the postprocessed argument — optimal rates, no
  superconvergence, which is the loss CCSZ-I repairs.
* **CDE-Bridge** — Cockburn, Di Pietro & Ern, *Bridging the hybrid high-order
  and hybridizable discontinuous Galerkin methods*, ESAIM Math. Model. Numer.
  Anal. **50** (2016) 635–650. Defines the HDG (ABC) family and the
  reconstruction `𝔭^{k+1}` that Optional B uses.
* **Lehrenfeld–Schöberl** — the HDG+ method, the same object as CCSZ-II's
  HDG (A): flux in `[P^k]^d`, potential in `P^{k+1}`, trace in `P^k`, and a
  stabilisation `h^{-1}` acting on the projected trace. **Oikawa**, *A
  hybridized discontinuous Galerkin method with reduced stabilization*, J. Sci.
  Comput., is the same idea arrived at independently. Optional B.
* **NPC-Stokes** — Nguyen, Peraire & Cockburn, *A hybridizable discontinuous
  Galerkin method for Stokes flow*, Comput. Methods Appl. Mech. Engrg. **199**
  (2010) 582–597. The velocity–pressure–gradient formulation §9 follows; §3.2
  is the augmented-Lagrangian reduction to the velocity trace alone, §4.1 the
  stabilisation sweep §9 reproduces.
* **PNC-NS** — Peraire, Nguyen & Cockburn, *A hybridizable discontinuous
  Galerkin method for the compressible Euler and Navier-Stokes equations*,
  AIAA 2010-363. Eq. (3) is the numerical flux `F(û)·n + S(u,û)(u−û)`, Eq. (5)
  the eigen-decomposition stabilisation and Eq. (6) the local Lax-Friedrichs
  one `S = λ_max I`; Eq. (8) the inflow/outflow boundary flux `B̂`; Eq. (13)-(14)
  the Navier-Stokes first-order system and its HDG formulation. §5's driver,
  `miniapps/hdg/navierstokes.cpp`, follows it; see `doc/HDG-NAVIER-STOKES.md`.
* **CS-Extensions** — Cockburn & Solano, on solving problems posed on curved
  domains by extension from a polyhedral subdomain, reducing the boundary
  treatment to line integrals along transferring paths. §1.
* **CSS-Coupling** — Cockburn, Sayas & Solano, on coupling an HDG interior solve
  to an exterior boundary-integral representation across an unmeshed interface,
  with **CSS-Analysis** its companion analysis, including the relaxed iteration
  and the contraction estimate. §2.
* **Persson & Peraire**, modal-decay smoothness sensor, cited in §7 as the
  standard choice for an `hp` criterion.
