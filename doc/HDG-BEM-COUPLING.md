# Coupling HDG at a distance to an exterior operator — FULLY OPTIONAL

Split out of roadmap §2, which called this "the largest item by a wide margin"
and read as though someone were waiting for it. Nobody is. This file is the
plan if it is ever wanted; the roadmap now carries a pointer.

## Read this before costing it: the case that matters is already solved, twice

**Choosing the artificial boundary to be a circle in 2-D or a sphere in 3-D
makes the exterior operator exact and diagonal, and removes the boundary
integral machinery entirely.** That is not a simplification of the coupling —
it is a different and much cheaper route to the same answer, and it is the
route both known consumers of this tree actually take.

**Gatica & Hsiao (1995)** is the reference: *The uncoupling of boundary
integral and finite element methods for nonlinear boundary value problems*,
J. Math. Anal. Appl. **189** (1995) 442–461, doi:10.1006/jmaa.1995.1029. On a
circle the adjoint double layer collapses — `K' σ = −(1/4πr)⟨1, σ⟩` is **rank
one**, and zero on the zero-mean Neumann space — so the Calderón identity
`W V = I/4 − (K')²` degenerates and the integral operators invert in closed
form, leaving one weakly singular term. They report the coding and the
computational work more than halved and the quadrature much easier. Their §2
quotes the 3-D sphere analogue from Hsiao & Zhang.

**meq: implemented and working.** `src/meq/ExteriorDtN.hpp` in that tree is the
exact Dirichlet-to-Neumann map on a semicircle for the axisymmetric
Grad–Shafranov operator, and it is **diagonal**: separating `ψ = ρ^α f(μ)` gives
the Gegenbauer equation of order `−1/2`, so the angular functions are
`C_n(μ) = (P_{n−2}(μ) − P_n(μ))/(2n − 1)` and the map is one number per mode.
No layer potentials, no singular quadrature, no elliptic integrals, no `O(N²)`
kernel evaluations. Two consequences their header records, both of which delete
work rather than add it: every mode vanishes on the axis identically, so the
flat side of the half-disc needs no separate treatment in the exterior at all;
and the exterior modes all decay with no admissible constant mode, so the
coupling paper's undetermined `u_∞` and its compatibility condition — its two
fiddliest pieces — do not arise. The axisymmetric problem is *cleaner* than the
Laplace one it is modelled on, which is not the usual direction.

**gffp: derived and relied upon, and not yet code.** `PHYSICS-NOTES.md` §4a in
that tree has the exterior DtN exactly diagonal in Legendre modes on a
spherical interface, and is explicit that it is at present "a verification
oracle, not a method" — no `.cpp`, `.hpp` or `.py` there implements it. Their
reference note adds the detail worth having: for a gyrotropic distribution the
trace expands in Legendre polynomials, `V` and `W` are diagonal on a sphere,
and Gatica & Hsiao's 3-D formula reduces to `σ_l = −(l+1) u_l / R` — *exactly*
the elementary exterior DtN, reached by two routes with nothing in common.
**Stated as their document states it**; this tree has not run their code.

### So what is this plan for

Only the cases the choice of boundary cannot buy:

* a `Γ` that cannot be a circle or a sphere, because the geometry forbids it;
* an exterior operator that does not separate on such a `Γ`;
* a caller who wants Galerkin BEM matrices on `Γ` for their own reasons.

None of those is in front of us. **Build nothing here until one is.**

## The shape of the coupling, in this branch's terms

The interior problem is what `DarcyForm` with `EnableHybridization` already
solves, on a polygonal `D_h` inside a smooth artificial boundary `Γ`, with the
Dirichlet datum on `Γ_h` transferred along the paths of §1. In a coupled
problem the datum `g` is an **unknown**, expanded in a small basis on `Γ` (`N`
of order 20 to 40), and the exterior supplies `N` more equations — the
transmission condition `E_h(q_h)·ν + λ = 0` tested against that same basis:

```
[  K    B  ] [ Λ ]     [ F ]          K = the hybridized trace matrix
[  T    D  ] [ a ]  =  [ G ]          B = ∂(trace residual)/∂a
                                      T = the transmission rows
                                      D = the exterior operator, N × N
```

`B` and `T` are **sparse** — both touch only the elements owning `Γ_h` faces —
and `D` is opaque to the library. **`D` is where the paragraph above lands**:
on a circle or a sphere it is diagonal and the caller writes it in a few lines.

This is expressible because most of Cockburn, Sayas & Solano, *Coupling at a
Distance HDG and BEM*, SIAM J. Sci. Comput. **34** (2012) A28–A47 is already
here: that paper's reference [5] is Cockburn & Solano, which is
`fem/darcy/extension_hdg.*` on this branch, and its `Σ_h`, `E_h(q_h)` and
`L_h(g)` are `TransferPath`, `ElementExtension` and
`TransferredDatumCoefficient` term for term.

## 1. The transmission integrator — DELIVERED, not open

The transmission condition needs the extended flux evaluated on `Γ` — the far
end of the paths — paired with the basis over the induced measure there: the
`t = 1` face of `y(ξ,t) = x(ξ) + t(a(x(ξ)) − x(ξ))`, with the surface measure
on `Γ` and the normal `ν` there rather than on `Γ_h`.

**meq wrote it and sent it back, and it is merged**:
`ExtensionBoundaryQuadrature()` in `fem/darcy/extension_hdg.{hpp,cpp}` on this
branch. It arrived with the finding that an unsigned weight integrates a swept
image *with multiplicity*, so a backtracking foot map overcounts `|Γ|` by
`O(h)` — and the region sweep beside it had the same defect, which is how the
aerofoil's tiling residual went `+1.13e−02 → −2.29e−04`.

The paper collocates this condition instead (its eq (4.1)), which needs only
point evaluation and gives up the variational structure. The Galerkin version
is the one that was built; collocation falls out of it as a special case of the
rule.

## 2. Auxiliary globally-coupled unknowns — the only unbuilt piece

`DarcyHybridization` has no concept of a global unknown that is not a trace
dof, so `B`'s column cannot be carried through the element-local elimination
the way `C` and `E` are. What would be asked for:

```cpp
void SetNumAuxiliaryUnknowns(int M);   // reduced system becomes [K B; T D]
void AssembleAuxFluxMatrix(int el, const DenseMatrix &B_el);
void AssembleAuxPotMatrix (int el, const DenseMatrix &B_el);
void SetAuxBlock(const DenseMatrix &D);   // the caller owns this entirely
```

Nothing of this exists on any branch — checked, not assumed.

**It is not a free-boundary feature, and that is the argument for it whatever
meq does.** An auxiliary unknown coupled to a hybridized system is the shape of
any global constraint: a mean-value condition, a total-flux condition, a
Lagrange multiplier pinning a pure Neumann problem (which
`ReconstructFluxAndPot()` already solves element-locally with the element
average as the closure), any boundary integral operator, and any parameter the
solution is constrained by.

**What it buys is an *assembled* border rather than a differenced one**, which
removes `M` residual evaluations per step and removes a real fragility: meq's
differenced border once read `1.6e5` where it should read about `1`, because
the element-local Newtons were seeded from a vector captured at
`FormLinearSystem()` time and never refreshed, so on a hard problem they hit
their iteration cap and returned something that was not a function of the trace
at all. Perturbing the parameter by `9e−6` moved the recovered maximum from
`0.896` to `3.84`, and the iteration looked exactly like a singular Jacobian.

**And the reason it is still not worth building.** meq had a design for the
caller's half, and the entry points it asked for by name — an application of the
local elimination in each direction — already exist as `NPCReduce()` and
`NPCRecover()`, both public, and both now have blocked forms taking several
right-hand sides at once (`gf-hdg-linearise-first`, `DarcyNPCSolver::ArrayMult`).
So the caller can already drive the elimination it needs, and the differenced
border is `K` applications of a routine that now blocks them.

## What is not being asked for, and was not

* **No BEM.** See the top of this file. `D` is opaque; a caller who genuinely
  needs Galerkin BEM matrices on `Γ` can supply them the same way. Flagged
  because "BEM" in the paper's title is otherwise read as a request for
  integral operators in this tree, and it never was one.
* **No change to the extension technique**, the path families, the lifting or
  the stabilisation. §1 sat beside them and used them.
* **No change to `PathTraceCoefficient`** — it stays the right thing for a
  *given* `g`, which is what the fitted and prescribed-datum cases want.
* **A caller-supplied basis, not a fixed one.** What the basis *is* belongs to
  the exterior problem: meq's is Gegenbauer functions on a semicircle, the
  paper's trigonometric polynomials on a circle, gffp's Legendre polynomials on
  a sphere.
* **No rectangular datum integrator.** Withdrawn by meq as their own error: the
  datum's data half is an **essential trace value**, `ψ̂|_{Γ_h} = P a`, whose
  columns are `ProjectBdrCoefficient` against the existing
  `PathTraceCoefficient`. It does not enter weakly, so no `⟨φ_n ∘ a, v·n⟩_e`
  form is wanted.

## References

* **Gatica & Hsiao**, *The uncoupling of boundary integral and finite element
  methods for nonlinear boundary value problems*, J. Math. Anal. Appl. **189**
  (1995) 442–461, doi:10.1006/jmaa.1995.1029. The reason this plan is
  optional.
* **Hsiao & Zhang**, the 3-D sphere analogue, quoted in Gatica & Hsiao §2.
* **CSS-Coupling** — Cockburn, Sayas & Solano, *Coupling at a Distance HDG and
  BEM*, SIAM J. Sci. Comput. **34** (2012) A28–A47, with **CSS-Analysis** its
  companion, including the relaxed iteration and the contraction estimate.
* **CS-Extensions** — Cockburn & Solano, the extension-from-subdomains method
  this branch implements, and CSS-Coupling's reference [5].
