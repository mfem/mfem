# Coupling HDG at a distance to an exterior operator

A request from meq, written 2026-08-29 and **revised down twice since**.
Nothing in this tree has been changed for it. The request as originally filed
is `ccb079c4a3`; this is what survives of it.

It asks for what `refs/CouplingAtADistance.pdf` — Cockburn, Sayas & Solano,
*Coupling at a Distance HDG and BEM*, SIAM J. Sci. Comput. 34 (2012) A28–A47 —
needs in order to be expressible. **Most of it is already here**: that paper's
reference [5] is Cockburn & Solano, which is `fem/darcy/extension_hdg.*` on
`gf-hdg-subdomains-dev`, and the paper's `Σ_h`, `E_h(q_h)` and `L_h(g)` are
`TransferPath`, `ElementExtension` and `TransferredDatumCoefficient` term for
term.

**Nothing in this tree needs to change for meq to start**, and that is the
result of the two revisions:

* **The rectangular datum integrator is withdrawn — meq's error, not ours.**
  The datum's data half is an **essential trace value**, `ψ̂|_{Γ_h} = P a`, whose
  columns are `ProjectBdrCoefficient` against the existing
  `PathTraceCoefficient`. It does not enter weakly, so no
  `⟨φ_n ∘ a, v·n⟩_e` form is wanted.
* **Auxiliary unknowns are an optimisation, not a prerequisite** (§2 below).
  meq's `ψ_ax` bordered Newton already works by differencing the reduced
  residual, and under NPC that difference is well defined — the fields are
  Newton state, so the border is a derivative of a residual that is
  *evaluated*, not of one reconstructed from a linearisation. An earlier
  version claimed the fallback returned exactly zero under one ordering; that
  was read out of a doxygen summary rather than the code, was wrong, and the
  ordering it hedged against is deleted.

## The shape of the coupling, in this branch's terms

The interior problem is the one `DarcyForm` with `EnableHybridization` already
solves, on a polygonal `D_h` inside a smooth artificial boundary `Γ`, with the
Dirichlet datum on `Γ_h` transferred along the paths. **In a coupled problem
the datum `g` is an unknown**, expanded in a small basis on `Γ` (`N` of order
20 to 40), and the exterior supplies `N` more equations — the transmission
condition `E_h(q_h)·ν + λ = 0` tested against that same basis:

```
[  K    B  ] [ Λ ]     [ F ]          K = the hybridized trace matrix
[  T    D  ] [ a ]  =  [ G ]          B = ∂(trace residual)/∂a
                                      T = the transmission rows
                                      D = the exterior operator, N × N
```

`B` and `T` are **sparse** — both touch only the elements owning `Γ_h` faces —
and `D` is opaque to the library.

## 1. The one integrator still wanted: `⟨ E_h(v_i)·ν, φ_n ⟩_Γ`

The transmission condition needs the extended flux evaluated **on `Γ`** — the
far end of the paths — paired with the basis over the induced measure there.
`ExtensionQuadrature` already sweeps `K^ext_e` and `ExtensionPoint` already
carries `xbar = a(x)`; what is missing is the *boundary* piece, the `t = 1`
face of `y(ξ,t) = x(ξ) + t(a(x(ξ)) − x(ξ))`, with the surface measure on `Γ`
and the normal `ν` there rather than on `Γ_h`.

**meq will write one and come back with it**, with the tiling check attached —
summing these boundary weights over the faces must give `|Γ|`, the way the
volume weights must give `|Ω| − |D_h|`, and that is what says the path family
covers `Γ` exactly once. A version that has been used is a better request than
this one. `TransferPath::Endpoint`, `ElementExtension::TransformBack` and a
central difference along the face reach it in about forty lines of caller code.

The paper collocates this condition instead (their eq (4.1)), which needs only
point evaluation and gives up the variational structure. **The Galerkin version
is the one worth building**; collocation falls out of it as a special case of
the rule.

## 2. Auxiliary globally-coupled unknowns — worth doing, not needed yet

`DarcyHybridization` has no concept of a global unknown that is not a trace
dof, so `B`'s column cannot be carried through the element-local elimination
the way `C` and `E` are. What would be asked for is

```cpp
void SetNumAuxiliaryUnknowns(int M);   // reduced system becomes [K B; T D]
void AssembleAuxFluxMatrix(int el, const DenseMatrix &B_el);
void AssembleAuxPotMatrix (int el, const DenseMatrix &B_el);
void SetAuxBlock(const DenseMatrix &D);   // the caller owns this entirely
```

**It is not a free-boundary feature**, which is the argument for it whatever
meq does. An auxiliary unknown coupled to a hybridized system is the shape of
any global constraint (a mean-value condition, a total-flux condition, a
Lagrange multiplier pinning a pure Neumann problem — which
`ReconstructFluxAndPot()` already solves element-locally with the element
average as the closure), of any boundary integral operator, and of any
parameter the solution is constrained by.

**What it buys is an *assembled* border** rather than a differenced one, which
removes `M` residual evaluations per step and removes a real fragility: meq's
differenced border was reading `1.6e5` where it should read about `1`, because
the element-local Newtons were seeded from a vector captured at
`FormLinearSystem()` time and never refreshed, so on a hard problem they hit
their iteration cap and returned something that was not a function of the trace
at all. Perturbing the parameter by `9e−6` moved the recovered maximum from
`0.896` to `3.84`, and the iteration looked exactly like a singular Jacobian.

**And the reason not to build it yet**: meq had a design for the caller's half,
and the entry points it asked for by name — an application of the local
elimination in each direction — turned out to exist already as `NPCReduce()`
and `NPCRecover()`, both public. The caller can already drive the elimination
it needs.

## What is not being asked for

* **No BEM.** meq's `D` is a diagonal matrix it forms itself — the exterior
  Dirichlet-to-Neumann map on a sphere is diagonal in a Gegenbauer basis for
  the axisymmetric operator — and there are no layer potentials anywhere in it.
  Nothing above assumes that: `D` is opaque, and a caller who does need
  Galerkin BEM matrices on `Γ` can supply them the same way. Flagged only so
  that "BEM" in the paper's title is not read as a request for integral
  operators in this tree.
* **No change to the extension technique**, the path families, the lifting or
  the stabilisation. §1 sits beside them and uses them.
* **No change to `PathTraceCoefficient`** — it stays the right thing for a
  *given* `g`, which is what the fitted and prescribed-datum cases want.
* **A caller-supplied basis, not a fixed one.** What the basis *is* belongs to
  the exterior problem: meq's is Gegenbauer functions on a semicircle, the
  paper's trigonometric polynomials on a circle.
