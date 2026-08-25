# HDG capabilities wanted in `fem/darcy`

What the HDG implementation in this tree does not yet do, what has since been
built or measured on top of it, and what remains open. The implementation is
Nikl's, in the Nguyen–Peraire–Cockburn lineage, and arrived on `darcy-hdg-dev`.

**Commit messages on this branch cite the numbered sections below.** The document
was written outside the repository, under the name `HDG-REQUIREMENTS.md`, and
older messages call it that; the numbering has not changed.

**This file is deliberately physics-free.** Every requirement is stated as a
property of the discretisation, in terms any HDG user would recognise. It was
written that way from the start so that it could be upstreamed: a requirement
phrased in terms of one application's needs is worth nothing to a reviewer, and
these are features of the method rather than of any application.

**Nothing here was checked against a compiler when it was first written. Much of
it has been since**, and sections carrying measurements say so explicitly. Where
a claim is still only a reading of the tree or of a paper, it is left as such.
Several expectations recorded here were contradicted by measurement; the
corrections are in §3(d), §4, §5 and in the mesh section, and they are worth
more than the confirmations. Papers are cited by the short labels defined under
**References** at the foot of this file.

**The target formulation is Nguyen, Peraire & Cockburn: element spaces that are
fully discontinuous for both the flux and the scalar, coupled only through a
trace space and a stabilisation `τ`. No Raviart–Thomas space appears anywhere in
those papers.** The distinction matters when reading the measurements below,
because a hybridized *mixed* RT method is a different scheme that this branch
also supports, and one which has no `τ` at all. Results obtained in an RT
configuration are real but partial: they say nothing about any requirement whose
content is the stabilisation. Each measurement below states which configuration
produced it.

## Where the work described here lives

The implementation is `fem/darcy/` — `DarcyForm`, `DarcyHybridization`,
`DarcyReduction`, `bilininteg_hdg`, `estimators_hdg`, `tmop_hdg`, and parallel
counterparts — together with `fem/nonlininteg_mixed.{hpp,cpp}` for the nonlinear
flux laws and `miniapps/hdg/` for the drivers.

The measurements quoted below come from the unit suite unless a section says
otherwise. The tests that hold them:

| section | tests |
| --- | --- |
| 3 hybridization, constant scalar `κ` | `tests/unit/fem/test_darcy_hybridization.cpp` |
| 3(d) degenerate coefficients | `tests/unit/fem/test_darcy_degenerate.cpp` |
| 4 systems | `tests/unit/fem/test_darcy_system.cpp` |
| 4 nonlinear, Jacobians, per-variable `τ` | `tests/unit/fem/test_darcy_nonlinear.cpp` |
| 4 nonlinear system, manufactured solution | `tests/unit/fem/test_darcy_nonlinear_mms.cpp` |
| 5 `τ` and the stabilisation interface | `tests/unit/fem/test_bilininteg_hdg.cpp` |
| 3(f) cross-field block placement | `tests/unit/fem/test_darcy_system.cpp` |
| 7 estimators | `tests/unit/fem/test_estimators_hdg.cpp` |

The Darcy tests run polynomial orders 0, 1 and 2 by default. Order 2 used to
sit behind `--all` in several of them; the one case still gated there is order
2 in 3D, which is minutes rather than seconds. Widening the rest costs the
Darcy set about fifty seconds.

§3 is closed by a driver that composes its terms and measures rates, and there
is not one yet: the first attempt was withdrawn and §9 records what it
established.

`miniapps/hdg/regression_test.py` is the miniapp-level regression suite;
its baseline at the time of writing is 2 failed / 49 skipped of 129, and any
change here should leave that untouched.

## What the branch already provides

Advertised on PR #4350 and visible in the tree: hybridization from the mixed form
of convection–diffusion, a nonlinear diffusion framework taking lambda functions,
nonlinear convection tied to the hyperbolic framework, MPI, superconvergent
reconstruction, broken Raviart–Thomas spaces needing no stabilization, and
`miniapps/hdg/` with `convdiff`, `anisodiff` and `anisodiff-hr`. `fem/darcy/`
carries `DarcyForm`, `DarcyHybridization`, `DarcyReduction`, `bilininteg_hdg`,
`estimators_hdg` and parallel counterparts.

One line of that list is the one to watch: **systems of equations are supported
"linear systems so far"**. See §4.

## A constraint on everything below: no tensor-product assumption

**The mesh this work is written against is neither structured nor fully
unstructured.** It is a **product of a 1-D unstructured mesh in one coordinate
with a 2-D unstructured (triangulated) mesh in the other two** — a triangulation
extruded into layers of arbitrary thickness, giving **prisms (wedges)**.
Unnumbered because it is a constraint on how every feature below is written, not
a feature to add. Readers whose mesh is simpler can take everything below as
covering their case too; the wedge is the demanding one.

**Measured, not merely checked: hybridized Darcy works on wedges.** Equivalence
against the monolithic solve and the order of accuracy both hold at orders 0, 1
and 2, in a mixed RT configuration. The expectation was that a wedge's mixed face
geometry — two triangular and three quadrilateral faces on one element — would be
the hardest thing in §3; it is not. Nothing in `fem/darcy` had ever run in three
dimensions at all before this, so hexahedra were covered first to keep a wedge
failure attributable to the element rather than to the dimension.

**Checked against the pinned tree, and this element choice is much better
supported than a general tet mesh would be.** Four facts, all of which pushed the
earlier draft of this section in the wrong direction:

* **Anisotropic refinement of prisms is supported, in exactly the two useful
  modes.** `NCMesh` refines a wedge either by an **XY split** — splitting the
  triangular cross-section into four wedges — or by a **Z split** along the
  extrusion axis, selected by `ref_type` (`< 4` is normalised to
  `Refinement::XY`; `4` is Z). So the two directions can be refined
  independently, which is precisely what the anisotropic flags from §7's
  estimator express. The earlier claim in this section that anisotropic
  refinement would have no consumer was true of tetrahedra and is **false for
  this mesh**. `prism_deref_table` exists too, so derefinement works.
* **The finite element spaces are all there on wedges**: `H1_WedgeElement`,
  `L2_WedgeElement`, and — the one that was not obvious — `RT_WedgeElement`,
  registered in `RT_FECollection` with its own DOF count. That matters because
  the branch's broken-Raviart–Thomas option would otherwise have been closed.
  (Nédélec on wedges appears absent, which nothing here needs.)
* **`Extrude2D(mesh, nz, sz)` only makes uniform layers.** It takes a layer
  *count* and a total size, not a spacing. Arbitrary layer thicknesses therefore
  need either an externally generated mesh or a uniform extrusion followed by
  repositioning the nodes — the topology is what the utility supplies, and the
  geometry is a `Nodes` `GridFunction`. A small meshing-side task, worth knowing
  about before it is discovered.
* **Quadrature and node-placement tricks that assume a *structured* element still
  do not transfer** — in particular the device of a one-sided (Radau) rule on the
  element touching a degenerate boundary, so no node sits on it. §3(d) should be
  met by the weak form being integrable, which is more robust and element-type
  agnostic.

### Varying the cross-section along the extrusion direction

A separate question, and the answer is three-tiered. Wanting a *different*
cross-sectional mesh at different positions along the extrusion axis is
reasonable — the region where the solution is supported may shrink or move along
it — and how much it costs depends on how different.

* **Different geometry, same topology — free, works today.** Move the
  cross-sectional node positions as a function of the extrusion coordinate. An
  MFEM mesh carries its geometry as a `Nodes` `GridFunction`, so the wedges
  simply become slanted or graded rather than right prisms, and layer interfaces
  stay flat, so **faces still match and nothing non-conforming arises**. Good for
  smooth grading, tracking a region that contracts.

  **The cost is the tensor-product factorisation.** `H1_WedgeElement` and
  `RT_WedgeElement` are literally tensor products — the RT header says so:
  `L2Triangle × H1Segment + RTTriangle × L2Segment`. On *right* prisms with an
  extrusion-independent cross-section, an operator with **no derivative in the
  extrusion direction** (§3(f) produces several) acts only on the triangle factor,
  so its discrete system decouples per segment basis index. Slant the prisms and
  that decoupling is gone. Anything relying on it must therefore treat "right
  prisms" as a checkable precondition.
* **Different resolution, common ancestry — free, works today.** Nonconforming XY
  refinement of individual wedges refines one layer's cross-section and not its
  neighbour's. This is ordinary AMR (§7) and `DarcyHybridization` already has a
  nonconforming path.
* **Genuinely unrelated cross-sections per layer — we implement it.** Two layers
  meeting with independently generated triangulations share a surface on which
  the face partitions do not align at all. Nothing in MFEM assembles a system
  across that. What exists and does *not* solve it: `SubMesh`/`TransferMap`
  requires shared entities with a parent; `FindPointsGSLIB` gives pointwise
  interpolation, useful for diagnostics and initialisation but not a variational
  coupling.

  **What does most of the hard part is `fem/moonolith/`.** `MortarAssembler`
  "allows to perform quadrature in the intersection of elements of two separate
  and unrelated meshes", generating intersection quadrature rules exact to machine
  precision. That element-intersection machinery is exactly what a non-matching
  interface needs. Four caveats: it is an **optional external dependency**
  (`MFEM_USE_MOONOLITH`, off by default, needs ParMoonolith); it assembles a
  *transfer* operator between two spaces rather than a coupling block inside one
  system; **curved elements are not supported**; and it is only "experimental"
  with `RT_FECollection`, which matters if the broken-RT option is taken.

  **Hybridization makes this much more tractable than it would be for a
  continuous method**, and that is worth saying plainly because it is the reason
  to consider it at all: all inter-element coupling already passes through a trace
  space on faces. A non-matching interface therefore needs only (a) a trace space
  on the interface — one side's partition, or the common refinement — and (b) each
  side's local solves integrating against it over face-to-face intersections,
  which is what the intersection quadrature supplies. It is an implementation task
  with a known shape, not an open problem.

  It also **preserves the factorisation that the first option destroys**: each
  layer keeps its own clean cross-sectional mesh, so an operator with no
  derivative in the extrusion direction stays exactly block-diagonal per layer.
  The two options trade off in opposite directions, which is the main reason to
  decide deliberately rather than by whichever is easier first.

**The discipline all three constrain.** A product mesh *does* factorise, and a
solver may exploit that — sub-meshes at fixed values of the extrusion coordinate,
directional solves, per-layer assembly. But every option above breaks the
factorisation somewhere: adaptivity breaks it locally, slanting breaks it within
an element, and only the third preserves it while removing face matching. So any
such exploitation must be an **optimisation guarded by a check**, never an
assumption baked into a data structure.

## 1. Extension and lifting — solving on a subdomain of the true domain

**Capability.** Solve on a polyhedral subdomain `D_h` strictly inside the true
domain `Ω`, with boundary data given on the true boundary `Γ` rather than on the
mesh boundary `Γ_h`, and retain the method's full order of accuracy.

The construction is Cockburn & Solano's (**CS-Extensions**), and its appeal is that it reduces curved or otherwise inconvenient boundaries to the
evaluation of **line integrals**. Four pieces:

* **Subdomain selection.** Given a background mesh and a level set `φ`, take
  `D_h` to be the elements lying entirely inside. Cheap for convex domains
  (check the vertices); needs care otherwise.
* **A family of paths `Σ_h`** joining each point of `Γ_h` to a point of `Γ`. In
  general this is the fiddly part: per-vertex cone and half-space construction,
  `N` rays, bisection on `φ`, then tangent interpolation along edges, arranged so
  paths do not cross before reaching `Γ`. **When `Γ` is an analytic surface with a
  closed-form closest-point map, all of it collapses to one line** — for a sphere,
  `a(x) = c + R(x-c)/|x-c|`. Worth supporting that case directly rather than only
  as a fast path through the general code.
* **An extension operator `E_h`.** Evaluate an element's own polynomial *outside*
  the element: `E_h(q_h)|K_ext(x) := q_h|K(x)`. Trivial mathematically; the
  question is whether the library will do it without complaint.
* **A lifting `L_h`,** the line integral along the path,
  `L_h(g)(y) = g(a(x)) + ∫_σ C E_h(q_h)·m`, whose restriction to `Γ_h` is the
  Dirichlet datum the local solves are given.

**Accuracy this buys**, measured rather than asserted: optimal `k+1` for both
scalar and vector unknowns over the whole of `Ω` for `k ≥ 0`, and `k+2` for the
postprocessed scalar at `k ≥ 1`, **even when `dist(Γ_h, Γ)` is only `O(h)`**. That
last clause is the whole point — earlier techniques needed `O(h^(k+1))`. `τ = 1`
throughout, with sensitivity reported only at extreme `τ`.

**Library-side questions.** Can an FE function be evaluated outside its element
cheaply? Where do per-face path data live? Is there a quadrature abstraction for
integrating along a path that is not a mesh entity?

## 2. Coupling at a distance to an exterior boundary-integral solve

**Capability.** Couple the HDG system on a bounded interior to a boundary-integral
representation of an unbounded exterior, across an artificial interface `Γ` that
**is not meshed and need not touch `Γ_h`**. Builds on §1: the interface data is
transferred by the same lifting.

Cockburn, Sayas & Solano (**CSS-Coupling**, with its analysis
**CSS-Analysis**). Pieces:

* **The transmission condition**, continuity of the normal flux across `Γ`,
  imposed at collocation points between the extended interior flux and the
  exterior Neumann trace.
* **Boundary-integral operators** on `Γ`, assembled by quadrature with proper
  treatment of the weak singularity. On a circle they diagonalise in Fourier
  modes and on a sphere in spherical harmonics, but that form is available only
  for those two geometries and only for a harmonic exterior, so **the general
  quadrature path is the one that has to work**.
* **The constants.** A zero-integral condition on the exterior Neumann datum, an
  additive constant at infinity carried as an unknown, and the correction that
  forces each iterate to satisfy the zero-integral condition. All three are easy
  to omit and fatal if omitted.
* **Solver structure.** The natural alternating iteration — interior Dirichlet
  solve, exterior Neumann solve, repeat — **is not a contraction in general**.
  The afternote to **CSS-Analysis** proves convergence only of a *relaxed* iteration
  `g^n = ω g̃ + (1-ω)g^(n-1)` for `ω` in an interval, and gives the `ω`
  minimising the contraction constant. It also records that a **monolithic**
  solve of both systems together is possible and more efficient, the alternating
  form having been kept only because two separate codes already existed. **A new
  implementation should be monolithic**, and doubly so if the interior problem is
  nonlinear, since a relaxation parameter inside a Newton loop is a bad place to
  live.
* **Exterior operators beyond Laplace.** The literature covers a harmonic
  exterior. A **fourth-order (biharmonic) exterior problem** is also needed, whose
  representation carries four boundary densities rather than two — and where some
  of those densities may be known from a previously solved second-order problem
  rather than being unknowns. No paper found covers the combination of this with
  coupling at a distance; it is the largest genuinely open item in this file.

**Library-side reality.** MFEM has no boundary element method at all. This is new
code rather than an extension, and the interface between it and `DarcyForm` is
the design question.

## 3. Genuinely general Darcy-like problems

**Capability.** `DarcyForm` / `DarcyHybridization` should accept the full second-
order operator, not the pure-diffusion special case. Specifically:

* **(a) Full, spatially varying matrix coefficients.** Not a scalar, not a
  constant tensor: a symmetric positive-definite matrix whose entries vary over
  the element, with nonzero off-diagonal entries. `miniapps/hdg/anisodiff.cpp` is
  the entry point to read; whether it takes a general `MatrixCoefficient` is the
  first thing to check.
* **(b) Zeroth-order (reaction) terms.** `div q + c u = f` rather than
  `div q = f`. Needed for essentially any operator obtained by differentiating
  another one.
* **(c) First-order (convective) terms in the mixed form**, coexisting with (a).
  The branch handles convection–diffusion, so this may already be present; what
  matters is that it composes with a full tensor and a reaction term rather than
  being a separate code path.
**Audited, and (a)–(c) are in a different state from what this file implied.**
The nav table above used to credit them to
`tests/unit/fem/test_darcy_hybridization.cpp`; that harness builds
`ConstantCoefficient k(1.0)`, and no Darcy test anywhere constructs a
`MatrixCoefficient` or a convective term. What exists is in the drivers:

| | implemented | driven by | unit test | rate study |
| --- | --- | --- | --- | --- |
| (a) full varying tensor | yes | `anisodiff` | **none** | **none** |
| (b) reaction | yes | `anisodiff` | constant `a`, equivalence only | **none** |
| (c) convection | yes | `convdiff`, scalar `κ` | — | `convdiff` |
| (a)+(b)+(c) together | — | **nothing** | **none** | **none** |
| (f) derivative source | **yes** | — | `test_darcy_system.cpp` | withdrawn, §9 |

So the gap is not the individual terms — `anisodiff` carries a
`MatrixFunctionCoefficient` conductivity and a `MassIntegrator` reaction, and
`convdiff` carries convection — but the **composition**, which is exactly what
the requirement asks for: that they compose rather than being separate code
paths. Nothing in the tree builds an operator with all three at once, and
nothing measures rates for (a) or (b) at all.

* **(d) Degenerate coefficients.** Operators of the form `∂_x( w(x) ∂_x · )` with
  `w` vanishing on part of the boundary. These are well posed in the natural
  weighted space, and the requirements were expected to be practical rather than
  theoretical: weighted quadrature; a node distribution that **does not place a
  node on the degenerate locus** — Gauss–Legendre–Radau on the affected element
  rather than Lobatto is the standard trick; and a `τ` that does not misbehave as
  `w → 0`.

  **Measured, in a mixed RT configuration, and the expectation was wrong in our
  favour.** The mixed form assembles `w⁻¹`, so a vanishing `w` was expected to be
  the singular case (e) in disguise — and by an integrability argument it is:
  vanishing at a *point* leaves an integrand going like `1/r` against an area
  element `r dr` and the entries stay finite, while vanishing along a *line or
  face* leaves `1/y` with nothing to compensate and the entries diverge
  logarithmically. **Both retain the design order `k+1` in the potential and the
  flux, at orders 0, 1 and 2, indistinguishable from a constant coefficient run
  through the same harness**, in 2-D on quadrilaterals and in 3-D on hexahedra and
  wedges. The reason is that the exact flux vanishes with the coefficient, so the
  solution sits in the weighted space the degenerate operator is well posed in and
  the discrete solution follows it; the large entries cost conditioning rather
  than accuracy. No weighted quadrature and no node-placement trick were needed.

  **The `τ` half is now measured, and it changes what this requirement asks
  for.** The accuracy loss is real, but neither of the remedies proposed above
  is the right one.

  With the stabilisation the integrator builds — `τ = td·κ(x)/h`, so `τ`
  vanishes wherever `κ` does — the potential loses order, and the loss grows
  with `k`: at `k = 2`, 2.18 for a point degeneracy and 2.36 for a whole face,
  against a clean control of 2.99 on the same meshes.

  Three candidates were measured:

  | candidate | effect |
  | --- | --- |
  | nodal basis off the degenerate locus | **none** — bit-identical |
  | quadrature order raised by 14 | **none** — 2.176 → 2.179 |
  | a `τ` that does not vanish with `κ` | **full recovery** |

  | `k` | locus | `τ = κ(x)` | `τ = 1` | `τ = 4` |
  | --- | --- | --- | --- | --- |
  | 1 | point | 1.71 | 1.88 | 1.95 |
  | 1 | face | 1.91 | 1.99 | 2.03 |
  | 2 | point | 2.18 | **3.06** | **3.22** |
  | 2 | face | 2.36 | **3.13** | **3.18** |

  with the errors themselves falling five- to sevenfold.

  **So the misbehaviour to fear is `τ → 0`, not `τ → ∞`**, and the fix is a
  floor on the stabilisation rather than weighted quadrature or a node
  distribution. The Gauss–Legendre–Radau trick proposed above cannot help here
  and should not be attempted: the bases span the same space and MFEM's
  quadrature does not depend on the basis nodes, so moving nodes is a change of
  basis and not of method. It is a real device in collocation settings, where
  the nodes *are* the quadrature points, which is where Hardman uses it.

  Two consequences. Wherever a coefficient vanishes on part of the boundary, a
  floored `τ` is a requirement rather than a tuning option. And
  `HDGDiffusionIntegrator` cannot express one, since it derives `τ` from the
  coefficient; supplying an `HDGStabilization` is how it is done.

* **(e) Singular coefficients — wanted, but check first whether they are
  avoidable.** A zeroth-order coefficient behaving like `1/x²` near a boundary is
  much harder to accommodate than a degeneracy, and **a change of independent
  variable can sometimes remove it entirely**. The application that motivated
  this file is a worked example: the same operator, differentiated in one chart,
  produces a singular reaction term, and in another produces a bounded
  first-order term.
  **The general principle worth recording: prefer a degenerate formulation to a
  singular one, and look for a change of variable before asking the
  discretisation to cope.** If that search succeeds, this requirement reduces to
  (c) plus (d) and nothing exotic is needed.
* **(f) Sources that are derivatives of another solved field — built, and
  demonstrated end to end.** The requirement was stated as a sequencing problem:
  the fields must be assembled in a fixed order and the derivative taken
  consistently with the space it came from. That framing was wrong, and the
  right one is better. A derivative of another field is a **first-order
  cross-field coupling**, and the place for it is the `B` block, which couples
  flux to potential — not a source computed after the fact. `DarcyForm` forms
  `±Bᵀ` itself, so placing one such coupling delivers its adjoint for free.
  Nothing has to be assembled in a particular order and no derivative of a
  discontinuous field is ever taken.

  What was missing was an integrator that writes into a chosen *(row-field,
  column-field)* block, since `VectorBlockDiagonalIntegrator` can only
  replicate down the diagonal. `VectorBlockIntegrator` does that, with element,
  mixed-element and face overloads, and
  `VectorBlockDiagonalHDGIntegrator` handles the HDG face terms of a system
  whose potential and trace spaces carry different numbers of fields. Both are
  in the library and unit-tested against every block of a rectangular layout.

  It **has** been driven end to end — a Stokes-shaped driver reached `k+1` in
  velocity, stress and pressure with the pressure gradient and the
  incompressibility constraint as one such coupling and its transpose — but
  that driver has since been withdrawn, so what survives is the tested
  capability rather than a standing demonstration. §9 records what the attempt
  established.

## 4. Systems of coupled nonlinear Darcy-like problems, with exact Jacobians

**"Systems of equations, linear systems so far" now has a precise meaning, found
by reading and by experiment: systems on this branch couple through the
*hyperbolic* flux, not the diffusive one.** `VectorBlockDiagonalIntegrator`
replicates one integrator down the diagonal and cannot express an off-diagonal
block; in `miniapps/plasma/braginskii_hdg.cpp` every diffusive term is wrapped in
it and all cross-equation coupling comes from the hyperbolic integrator. A
diffusive coupling is therefore not a gap that can be avoided by choosing a
different existing integrator.

Measured or built since:

* **`N` fields as one system works** — spaces of `vdim = neq`, each equation
  solved as it would be solved alone, hybridization still exact, design order.
* **Linear cross-equation coupling works**, through a matrix coefficient on the
  potential mass block, which is the only route the branch offers for a dense
  `neq × neq` block.
* **Nonlinear coefficients do accept an analytic Jacobian** — the framework takes
  the derivative alongside the coefficient — and the Jacobian is right: `J dy`
  agrees with a central difference of the residual, with a control confirming a
  wrong derivative is caught.
* **`MixedConductionNLFIntegrator` carried a scalar potential** in every assembly
  path, so a nonlinear law could not couple equations at all. All of it — element
  and face terms alike — is now general in `num_equations`.

  A correction to what this document said before: the earlier claim that the face
  terms "refuse rather than assembling the first equation silently" was true only
  of *half* of them. The integrator has four face methods, an LDG pair used
  without hybridization and an HDG pair used with it, and only the LDG pair was
  guarded. The HDG pair — the one hybridization actually calls — would have
  assembled equation 0 and dropped the rest.

* **`τ` for a system is a scalar per variable.** A matrix over the variables was
  the other option, and the reason it is not needed is worth stating: the flux
  function already handles the spatial directions, so the only structure a matrix
  could carry is over the *variable* index, and nothing in the face term asks for
  it. The term is a penalty `w (p − û)` against both test spaces and its gradient
  is four rank-one updates — NPC Eq (15) with `∂s = 0` — so a scalar per variable
  leaves every face block diagonal in the variables. A test checks that
  diagonality directly rather than assuming it. The default is `τ = 1`.

  One equation keeps its existing route entirely, deriving the stabilization from
  the inverse flux Jacobian and ignoring the `τ` vector; a test requires the
  residual to be unchanged bit for bit, and the `convdiff` regression suite is
  unmoved.

**The hybridized Jacobian is now complete.** `DarcyHybridization::ConstructGrad`
and `LocalNLOperator::GetGradient` both set the local Jacobian's `(0,1)` block to
`±Bᵀ`, the transpose of the linear divergence form, and never asked the
integrator for `∂(flux residual)/∂p`. For a flux law `q = D(p) u` that term is
exactly the `J_u` the flux function supplies, and
`MixedConductionNLFIntegrator` assembles it on request. Both now ask, and
`ComputeH` needs only one correction to `AiBt` — the Schur complement and the
`C A⁻¹ Bᵀ + G` product are both built from it. On `convdiff`'s own nonlinear
hybridized cases:

| case | before | after |
|---|---|---|
| `p8_o1_hb_nld_newton` | 9 Newton iterations | 4 |
| `p8_o1_dg_hb_nld_newton` | 8 | 4 |

with the converged answers unchanged to six figures.

This is the requirement below about exact Jacobians, met — and it is worth
recording how nearly it was mis-diagnosed. The first evidence offered for the
defect was a residual "exactly linear in `ε` over six decades", presented as
proof that Newton stalled one order into the nonlinearity. It was worthless.
The probe behind it had no boundary condition, so 71 of its 160 trace dofs sat
in the operator's null space — the residual is identically zero there while the
gradient is not — and Newton was wandering in that null space. And `r₁ ∝ ε` is
not a stall signature at all: it is what *correct* quadratic convergence looks
like when the nonlinear part of the residual carries a factor `ε`. **A nonlinear
solver's convergence history says nothing about a Jacobian until the problem is
known to be well posed.** Differencing the operator against its own gradient
finds the null space in one step, and is the check to run first; it is what the
test now does, with the boundary traces pinned so that no dof has to be excused.

**The fully discontinuous formulation now carries the whole of the above.** The
system harness runs in both, and the DG system converges at the design order in
both variables:

| | RT `p` | RT `u` | DG `p` | DG `u` |
|---|---|---|---|---|
| `k=0` | 0.99 | 1.00 | 1.10 | 1.00 |
| `k=1` | 2.00 | 2.00 | 1.99 | 1.99 |

with `τ = κ/L` fixed. The one thing that does *not* carry over is
**hybridized ≡ monolithic**: `DarcyForm::Assemble` routes the potential faces
through `AssemblePotHDGFaces` when hybridization is on and `AssemblePotLDGFaces`
when it is off, so for a DG flux those are not the same operator and the
equivalence is a property of the hybridized mixed method only.

### Solved end to end, against a manufactured solution

Everything above is checked pointwise — a Jacobian against a difference of its
own residual, one Newton step against an exact linear answer. None of it says
the discretization converges to the right thing, because the Jacobian of a
wrong residual is still its own Jacobian.
`tests/unit/fem/test_darcy_nonlinear_mms.cpp` closes that: a two-equation
nonlinear system with a known solution, hybridized, in both formulations.

Constructing the solution needs one turn. `ComputeDualFlux` returns `D(p) u`
and the flux equation is `D(p) u + ∇p = 0`, so `D` is a *resistivity*, and
stating the solution against it would need `u = −D(p)⁻¹ ∇p` and the divergence
of that. Choosing the **conductivity** `K(p)` instead and handing the solver
`D = K⁻¹`, inverted analytically, makes `u = −K(p) ∇p` explicit and its
divergence elementary. `K` is symmetric, nonlinear in both potentials, with
both off-diagonal entries nonzero, and its determinant is positive for every
`p`, so the law is uniformly elliptic and Newton cannot walk out of its domain.

Rates over the 8×8 to 16×16 pair, `τ = 2`, five Newton iterations at every
order, form and mesh:

| | RT `p` | RT `u` | DG `p` | DG `u` |
|---|---|---|---|---|
| `k=0` | 0.99 | 1.00 | 0.98 | 0.93 |
| `k=1` | 2.00 | 2.01 | 1.89 | 1.83 |
| `k=2` | 3.00 | 3.01 | 2.91 | 2.86 |

The control is the same study with the terms that exist in the source *only*
because `K` depends on `p` removed — differentiating as though the conductivity
were locally frozen, which is the classic manufactured-solution mistake. It
flattens the rate to 0.02. A separate test checks the manufactured solution
itself, away from the mesh and the assembly: that the law handed to the solver
inverts the `K` the exact flux was built from, that the source really is minus
the divergence of that flux, and that the analytic flux Jacobian differentiates
the law. A wrong source is the easiest thing here to get wrong and shows up as
rate zero with nothing to say why.

**`τ` for the DG form**, measured over 16×16 to 32×32:

| `τ` | `k=0`: `p`, `u` | `k=1`: `p`, `u` |
|---|---|---|
| 0.5 | 1.00, 0.67 | 1.83, 1.76 |
| 1 | 1.00, 0.83 | 1.91, 1.84 |
| 2 | 1.01, 0.93 | 1.96, 1.90 |
| 4 | 1.05, 0.94 | 1.98, 1.96 |
| 8 | 0.91, 0.86 | 1.99, 1.98 |

`τ` of the size of `κ`, which is NPC-1's `η_d = κ/ℓ` with `ℓ = 1` and `K`
between 1 and about 2.5 — and the flux is what pays for getting it wrong in
either direction, too small and it never reaches its order, too large and `k=0`
begins to lock. That is the same trade the linear study found. A per-variable
`τ` tracking each equation's own conductivity, `(1, 2)` against the scalar `1`,
moved nothing measurably: on this problem the magnitude matters and the ratio
does not.

**Two defects, both found by this study and by nothing else.**

* `MixedConductionNLFIntegrator`'s HDG face pair evaluated the element basis at
  `Trans.Elem1`'s integration point whatever side it was assembling. Element 2's
  basis at element 1's reference point is a different function; constant shapes
  hid it at `k=0`, and at `k=1` the DG rate read **0.68 where 2 was wanted**.
  `HDGDiffusionIntegrator`'s side-aware overload already does this correctly and
  is what the fix follows.
* `DarcyHybridization::AssembleHDGGrad`'s `BlockNonlinearFormIntegrator`
  overload **accumulated** into `E` and `G`. Only `H` is reset between gradient
  evaluations, because it takes a contribution from each side of a face; `E` and
  `G` hold one block per face and side and every other writer in that file
  overwrites its own. So `GetGradient` depended on how many times it had been
  called. Newton calls it exactly once per step, which is precisely why no
  Jacobian check finds this — every one of them evaluates the gradient once, and
  the first one is correct. What it produced was a good first step and garbage
  after it: **divergence from a 5% nonlinearity on a 2×2 mesh**. A test now
  calls `GetGradient` twice at the same point and requires the same matrix.

Neither is reachable from any example on the branch. Instrumenting both sites
and running `convdiff`'s nonlinear hybridized DG cases shows neither is entered
— with hybridization on, the miniapp supplies the stabilization as a linear
`HDGDiffusionIntegrator` on the potential mass form and never calls the
nonlinear integrator's HDG face methods at all. That is why the 129-case serial
regression suite and the 98-case parallel one are byte-for-byte unmoved by both
fixes, and it is the reason a convergence study was worth the effort: two live
defects in code that every existing test walked past.

### Postprocessing, and what it does and does not fix

The branch reconstructs the normally continuous total flux and then flux and
potential one order higher. Applied to the same manufactured problem — see
below for why with one field rather than two — over the 8×8 to 16×16 pair:

| `k` | form | `p` | `u` | `u_t` | `p_s` | `u_s` |
|---|---|---|---|---|---|---|
| 0 | RT | 0.99 | 1.00 | 1.00 | **2.00** | 1.00 |
| 0 | DG | 0.98 | 0.92 | 0.90 | **0.98** | 0.90 |
| 1 | RT | 2.00 | 2.01 | 2.01 | **3.09** | 2.06 |
| 1 | DG | 1.89 | 1.83 | 1.82 | **2.83** | 1.80 |
| 2 | RT | 3.00 | 3.01 | 3.01 | **4.12** | 3.07 |
| 2 | DG | 2.91 | 2.86 | 2.85 | **3.89** | 2.87 |

The postprocessed potential gains a full order everywhere except the fully
discontinuous form at `k=0`, which is a known restriction and not a defect:
the local postprocessing needs the solved potential to be superconvergent in
its element averages, and for an L2 flux that holds from `k=1`. **CCSZ-I**
Table 1 reports 0.97 at `k=0` and 3.01 at `k=1` for the same `HDG_k` method,
and its Theorem 3.19 carries the hypothesis `k ≥ 1` explicitly. The hybridized
mixed form has it at `k=0` as well, and shows 2.00 — the same split
**CCSZ-II** draws between the `HDG_k` methods and the HHO-inspired ones.

**This answers the question the convergence study left open.** The DG form's
flux lags its potential, and postprocessing is not what was missing — `u_s`
tracks `u_h` to within a few hundredths of an order in every row, as it should,
since the reconstructed flux is not superconvergent and was never claimed to
be. Two things narrow the lag down further. It is **not about systems**: the
single field measured here lags by the same amount the two equations do, 0.88
against 1.00 at `k=0` and 1.87 against 1.95 at `k=1`. And it **closes as `τ`
grows** — 1.76, 1.84, 1.90, 1.96, 1.98 at `τ` = 0.5, 1, 2, 4, 8 for `k=1` —
so it is the size of the stabilization relative to `κ`, not a defect, and the
price of pushing `τ` up is `k=0` beginning to lock. That is a `τ` question,
§5's, rather than an open question about the discretization.

**Two limitations, one of them fixed here.**

* `DarcyForm::ReconstructFluxAndPot` copied the integrators of the *linear*
  flux mass form onto the enriched space, and dereferenced it unconditionally.
  With a solution-dependent law there is no such form and it **segfaulted** —
  reachable straight from the miniapp as `convdiff -rec -nld`. There is a
  natural thing to do instead, and it is now done: linearise about the
  converged potential, which for `q = D(p) u` means a flux mass with the
  matrix coefficient `D(p_h)`. `FrozenDualFluxCoefficient` in
  `fem/nonlininteg_mixed.hpp` is that coefficient. A nonlinear form carrying
  linear integrators — `convdiff -nlu` — has its mass reused as it stands.
* **Postprocessing remains scalar-only**, so the two-equation system cannot be
  postprocessed and the table above is a single field. This is not one assert
  to relax. `ReconstructFluxAndPot` builds the enriched potential, trace and
  total-flux spaces with no `vdim` argument and indexes them with
  `GetElementDofs` rather than `GetElementVDofs`;
  `DarcyHybridization::ReconstructTotalFlux` takes a callback with a scalar
  potential; and the source term uses `DivergenceGridFunctionCoefficient`,
  which is scalar. Listed in what is still open. The fourth item on that list
  — a stabilization living on the nonlinear face integrator leaving the local
  problem with no potential constraint — is no longer one of them; it is the
  first of the three defects below.

### Three defects the reconstruction had, found from outside

**All three were found by a user of this branch rather than by this work**,
building a fixed-boundary Grad–Shafranov solver on it, and reported in
`doc/HDG-DEFECTS-FROM-MEQ.md`. Each is the same shape: a path was extended for
one configuration and its sibling was not, and the sibling does not fail — it
produces numbers. They are recorded here with what was measured, because the
first of them is the strongest argument in this document for not trusting a
postprocessing pass that has no twin to check against.

**(a) The lift read the potential mass off the linear form only.** The block
began `if (M_p)` with no `else if (Mnl_p)`, so a potential block living
entirely on the nonlinear form — which a Newton path on a solution-dependent
source forces, since a nonlinear potential mass and a linear one cannot be
mixed — produced an enriched form with no integrators at all. Worse than the
missing mass: the hybridization keeps a nonlinear potential constraint in
`c_nlfi_p` rather than `c_bfi_p`, and the local problem read only the latter,
so it had **no face constraint either**. The matrix was singular and was
factored and solved anyway.

Measured here, on the same problem solved twice with the potential mass on
each form, `p_h` agreeing to solver tolerance:

| | `‖p_s − p‖` linear | `‖p_s − p‖` nonlinear |
|---|---|---|
| before | as the table above | **7.8e16** |
| after | as the table above | same to 8 figures |

and from the miniapp, `convdiff -nx 8 -ny 8 -hb -dg -rec -anal -o 1`, where
`-nlp` returned `nan` for both postprocessed fields and now returns
0.00510265 and 0.00185785 against the linear path's 0.00510268 and 0.00185788.
The reporting application measured 9.9e14 against 3.8e-6 on its own benchmark.

The fix is to take the potential block as **the Jacobian frozen at the
computed potential**, lifted onto the enriched space — the treatment the flux
mass already gets one screen earlier, and one that reduces to the `M_p` branch
term for term when the integrators are bilinear, since
`BilinearFormIntegrator::AssembleElementGrad` is its element matrix. The
enriched space contains the original one, so the embedding of `p_h` that the
freezing point needs is exact and adds nothing of its own. The face constraint
comes from `c_nlfi_p->AssembleHDGFaceGrad`, which lays its block out exactly as
`AssembleHDGFaceMatrix` does when the whole mask is asked for.

**What it refuses, and why.** That gradient is taken at the computed potential
and at a **zero trace**, because the reconstruction is never handed the trace
solution — `Reconstruct()` passes `sol_r` to the total-flux pass and not to
this one. So the lift verifies that every interior-face integrator on the
nonlinear potential mass is bilinear, and that any `HDGDiffusionIntegrator`
among them carries no state-dependent stabilization; a genuinely nonlinear face
term, `convdiff -nlc`, is refused with a message naming the reason instead of
returning `nan`. Making that case work needs the coarse trace threaded into
`ReconstructFluxAndPot`, which is an API question rather than a bug.

**(b) The lift took domain integrators only.** A boundary-face term on the flux
mass was dropped on the way to the enriched space. That is exactly how the
extension work on `gf-hdg-subdomains-dev` installs the solution-dependent half
of a transferred boundary datum — on the flux mass with `AddBdrFaceIntegrator`,
deliberately, so that the element-local term does not reach the hybridization —
and the reconstruction was solving its local problems without it. The reporting
application measured this one **harmless**: `p_s` still converged at `k+2` on
its benchmark, because the local problem is driven by the reconstructed total
flux and the element average, and both already carry the extension.

It is fixed rather than merely documented, because "harmless on the cases tried"
has no diagnostic to tell it from the cases where it is not. The boundary-face
integrators are lifted with their markers and assembled face by face into the
local flux block. An **interior**-face term on the flux mass is refused
instead: it couples two elements and the local problem is one element at a
time, so there is nothing to lift it onto.

**A second thing surfaced while pinning this.** `DarcyForm::Assemble()` builds
the hybridized flux mass from `M_u->ComputeElementMatrix()`, which is domain
integrators only — so on this branch a boundary-face term on the flux mass
never reaches **the solve** either. `gf-hdg-subdomains-dev` added
`AssembleFluxMassBdrFaces()` for exactly that and this branch has no
equivalent, which is why the test here holds the discrete solution fixed and
measures only what the postprocessing does with it. Anyone bringing the
extension work back to `gf-hdg-dev` needs that pass as well as this fix.

**(c) `ComputeHDGFaceEnergy()` ignored an installed `HDGStabilization`.** See
§7 below; it is an estimator question rather than a postprocessing one.

### A fourth, found by trying to reconstruct twice

**Not in the outside report, and worse than anything in it.**
`DarcyHybridization::ReconstructTotalFlux()` walks the faces with a single
`DenseMatrix` for the constraint block. On an interior face
`GetCtFaceMatrix()` **resets** that matrix onto the stored `Ct_data` — it
aliases, it does not copy — and on a boundary face the constraint integrator
assembles into the same variable. A `DenseMatrix` that already has the right
shape keeps the pointer it was reset to, so the assembly **lands in
`Ct_data`**. On a uniform mesh every face has that shape, so every boundary
face overwrote the stored block of whichever interior face preceded it. The
parallel shared-face branch does the same thing through `CopyMN`.

**The call returns the right answer**; the corruption is behind it. Every
number the miniapps print is unchanged by the fix, to the last digit —
`convdiff -nx 8 -ny 8 -hb -dg -rec -anal -o 1` gives 0.00615473, 0.00510268
and 0.00185788 before and after. What is damaged is the object it was called
on, and the damage is total:

| | measured |
|---|---|
| second `Reconstruct()`, `u_t` | moves by 0.40 on a field of norm 0.79 |
| second `Reconstruct()`, `u_s` | moves by 1.58 |
| `RecoverFEMSolution()` after a `Reconstruct()` | moves the solution by 3.24, on a solution of norm 2 |

The reduced operator itself does **not** notice, because by then it is an
assembled `SparseMatrix`; it is the recovery of flux and potential from the
trace that reads `Ct_data` again. So this is invisible to a driver that solves
once, reconstructs once and stops — which is every driver in this tree — and
wrong for anything that reconstructs inside a loop: a time step, a Newton
iteration, an adaptive pass that estimates and then solves again. §7's
adaptive loop is exactly that shape, which is how it would have been found
eventually.

The fix is that nothing may assemble into a matrix `GetCtFaceMatrix()` has
aliased. Two tests: reconstructing twice must agree to the last bit, and
recovering the solution after a reconstruction must return the solution.

The requirement as originally stated:

* **`N` coupled fields, each a Darcy-like problem of the §3 kind**, coupled
  through both coefficients and sources, solved as one system rather than by
  outer iteration.
* **Nonlinear coefficients with an analytically supplied Jacobian.** The branch's
  nonlinear diffusion framework takes lambda functions; the question is whether it
  will accept a derivative alongside, or insists on differencing.
* **Why this is worth insisting on.** In a hybridized method the Jacobian is not
  assembled globally, so **an error in it does not produce a wrong answer — only
  slow Newton convergence**. That failure mode is nearly invisible: it survives a
  passing regression suite indefinitely. Exact Jacobians plus a test that
  finite-differences the residual and requires `J dy = g` is the only thing that
  catches it.

## 5. `τ` for problems that are convection- and diffusion-dominated at once

The literature treats problems that are convection-dominated in some *regions*
and diffusion-dominated in others. The
requirement here is different: a problem convection-dominated in one *coordinate
direction* and diffusion-dominated in another, everywhere at once.

Whether a single scalar `τ` can serve, or whether it must become
direction-aware, is open — and it is the question to answer before trusting any
order-of-accuracy study, because a badly chosen `τ` degrades the rate without
producing an obviously wrong answer.

**Corrected by measurement: it degrades *a* rate, not *the* rate.** The
sentence above implies one number. What the sweep below shows is that the
potential and the flux behave differently under `τ`, and that only the flux
suffers. The distinction matters wherever the quantity of interest is itself a
flux (§6), which is the case this file was written for.

### What the nonlinear paper actually specifies

Read from **NPC-2**. Its Eq. (5) defines the numerical flux as

```
q̂_h + F̂_h = q_h + F(û_h) + s(u_h, û_h)(u_h − û_h) n        on E_h
```

so **`τ` is a function of the solution and of its own trace**, not a coefficient.
Four consequences, all of them structural rather than cosmetic.

**It is additive, and that is what keeps the existing code path.** §2.4 splits
`s(u_h, û_h) = s_diff + s_conv(u_h, û_h)` with `s_diff = κ/ℓ` constant. For a
linear flux the positivity bound (7), `s ≥ ½ sup_{J(u,û)} |F'(s)·n|`, collapses to
`½|c·n|`, which is exactly the `τ_± = β|u·n| ± ½α(u·n)` the branch's upwinded
convection integrator already implements. **The two integrators in the branch are
the constant-`s` specialisation of Eq. (5)**, not a different scheme, so a
single-valued scalar `τ` can and should keep its current path untouched.

**It makes the face term nonlinear even for a linear PDE.** `s(u,û)(u−û)` is not
bilinear, so a solution-dependent `τ` cannot live on a bilinear-form assembly
path, which never sees `u_h` or `λ`. It belongs on the residual/gradient pair.

**Newton needs both partials.** Eq. (15) linearises with `∂₁s` and `∂₂s`
explicitly, and they enter the blocks of Eq. (16) as

| block | coefficient |
| --- | --- |
| E (potential row, trace column) | `F'(û)·n + ∂₂s·(u−û) − s` |
| H (trace row, trace column) | `F'(û)·n + ∂₂s·(u−û) − s` |
| G (trace row, potential column) | `∂₁s·(u−û) + s` |

which is **the block naming the branch already uses** in its face-gradient
assembly. The framework is already the right shape; what is missing is the
dependence. Note that with `∂s = 0` and no convection this gives `E = −s`,
`G = +s`, which is the negative-transpose relation between the trace-flux and
constraint blocks that a unit test on the branch measures independently.

**Admissibility is checkable.** The energy identity of Proposition 1 is positive
only if (6) holds, for which (7) is sufficient. A violated positivity condition
does not raise an error; it produces a bad solve. It is cheap to assert per
quadrature point in a debug build and worth doing.

The monotone-flux choices — Godunov (9a), Engquist–Osher (9b), Lax–Friedrichs
(9c), combined into `s_conv` by (10a) — are implementations of one interface
rather than special cases inside the integrator.

**What none of this settles is the question at the top of this section.** The
paper's `s` is a scalar function of two scalars for a scalar conservation law.
For a *system* it must become a matrix over the equations, and combined with
solution dependence the derivative is `∂τ_ij/∂u_k`. That is the open item
blocking §4's face terms.

### What was measured, and a correction

**An earlier version of this section carried a table showing the flux losing an
order as `τ` grew, and concluded that the branch's default sat on a knee. That
was wrong, and the error was in the measurement, not in the method.**

`HDGDiffusionIntegrator` does not take `τ`. Its parameter `td` enters as

```
τ = td · κ / h
```

with `1/h = |nor|/det(J)` — the source says so in its own comment. So holding
`td` fixed while refining makes `τ` grow like `1/h`. The sweep that produced the
withdrawn table varied the *coefficient of a 1/h-scaled stabilisation* and then
read the result as the effect of `τ`.

NPC-1 §3.6.3 is explicit that this is not their choice: they take
`s = s_d + s_c` with `η_c = |c·n|` and `η_d = κ/ℓ`, where **`ℓ` is a
representative diffusive length scale of the problem, not the element size**. So
their `τ` is `O(1)` and does not move under refinement.

**With `τ` held fixed, the branch reproduces NPC-1's convergence study.** Run on
the branch's own `convdiff -p 2 -dg -hb`, its steady advection–diffusion
problem, four meshes from 8×8 to 64×64:

| `p` | scaling | rate, flux | rate, scalar |
| --- | --- | --- | --- |
| 1 | `τ` fixed | 2.10 | 2.12 |
| 1 | `td` fixed | 1.01 | 2.52 |
| 2 | `τ` fixed | 3.11 | 3.14 |
| 2 | `td` fixed | 2.01 | 3.55 |

against NPC-1 Table 1 (centered scheme), which reports `2.00 / 2.00` at `p = 1`
and `3.00 / 3.00` at `p = 2`. **Optimal `p+1` in both variables, as the paper
has it.** A manufactured-solution sweep on a pure diffusion problem gives the
same answer independently: with `τ` fixed the flux rate is `k+1` for every `τ`
in `[0.5, 4]`, and at `k = 0` it *improves* with larger `τ` rather than
degrading.

**The `1/h` scaling is a different method, not a broken one.** Holding `td`
fixed gives the flux at `p` and the scalar at about `p+1.5` — the scalar
superconverges while the flux drops an order. That is the LDG-like regime, and
it is a legitimate choice; it is simply not the one NPC analyse, and it is not
the one to measure against their tables.

Three practical consequences.

* **`td` is not `τ`, and the distinction is invisible at a single mesh
  resolution.** Every convergence study on this branch has to say which is being
  held fixed. A study that fixes `td`, as the miniapp defaults do, is measuring
  the `1/h` method.
* **`HDGDiffusionIntegrator` cannot express `κ/ℓ` directly**, because the `1/h`
  is built into the assembly. Reproducing NPC's stabilisation means either
  scaling `td` with the mesh, which is what was done above, or supplying an
  `HDGStabilization` object that returns `κ/ℓ` and ignores the value the
  integrator computed. The latter is what that interface is for.
* **The question at the top of this section is still open.** Everything here is
  a diffusion-dominated or mildly convective problem with a single `τ`. Nothing
  measured so far speaks to a `τ` serving convection in one coordinate direction
  and diffusion in another simultaneously.

What survives from the withdrawn table is one methodological point, worth
keeping because it cost a day: **rates must be taken asymptotically**. The same
configurations read 1.6 rather than 2.5 on the coarsest pair of meshes, which is
enough to condemn a correct `τ`.

## 6. Functionals of the solution, evaluated from the numerical trace

**Capability.** Compute a surface integral of the numerical flux `q̂ · ν` over a
prescribed internal or boundary surface, as a first-class quantity.

The point is that `q̂ = q_h + τ(u_h - λ)ν` is single-valued on faces by
construction — that is what hybridization is — so such an integral is
consistent with the discrete conservation statement rather than being an
after-the-fact diagnostic. For a problem whose answer *is* a small flux, this is
the difference between an accurate result and a catastrophic cancellation.

**Corrected: that is true for a constant `τ` and false for a solution-dependent
one.** **NPC-2** states it immediately after its Eq. (5): because `s`
is nonlinear, the trace equation "cannot force the normal component of the total
flux to be single valued on all interior faces; it only forces the L²-projection
of the normal component into `M_h(0)` to be single valued". That is still enough
for local conservativity, which is what the method needs — but it is not enough
for the wording above. **The functional must therefore be computed as the pairing
against the trace space, not as a pointwise evaluation of `q̂·ν`**, or the
identity this section calls the sharpest available test of the whole assembly
will not hold to round-off. Wherever the quantity of interest *is* a flux, this
is a requirement on the diagnostic and not a footnote.

Two things follow. The identity "integrated flux through a surface equals the
rate of change of the integral inside it" should hold **to round-off**, and is
therefore the sharpest available test of the whole assembly. And superconvergent
postprocessing, which the branch already advertises, is exactly what a functional
of the solution benefits from — with goal-oriented adjoint error estimation as
the natural next step, noted as a direction rather than a requirement. §4
measures what it delivers: `k+2` for the potential wherever the theory offers
it, and nothing extra for the flux, which is the expected answer and not a
shortfall. It is scalar-only, which for a functional of a system's flux is the
same limitation recorded there.

## 7. Adaptive refinement for a solution-dependent internal layer

**Revised twice, and the second revision is mostly good news.** An earlier version
asked for meshes aligned to a prescribed internal surface with jump conditions
across it; that came from a *reduced* model in which the surface was a genuine
internal boundary, and is withdrawn — in the unreduced formulation it is not one,
and §6's flux integral is over an ordinary domain boundary. The replacement
requirement was adaptive resolution of a layer that is solution-dependent and
therefore moves as the calculation converges. **Most of that is already
implemented in the pinned branch**, which is a much better position than the
previous wording implied.

**What is already there.** `fem/darcy/estimators_hdg.hpp` defines
`HDGErrorEstimator : public AnisotropicErrorEstimator`, purpose-built for
hybridized Darcy-like systems, in two flavours:

* `Type::Residual` — the residual of the potential constraint `|G p + H λ|`
  integrated over faces. General; needs only `AssembleHDGFaceVector` from the
  integrator.
* `Type::Energy` — an energy-like norm `~ (p̂-λ)ᵀ τ (p̂-λ)`, generalised so the
  product can be evaluated **component-wise in reference space**, which is what
  produces the anisotropic flags. Needs `ComputeHDGFaceEnergy` with its
  `d_energy` parameter.

**The energy mode reported the wrong `τ` when one was installed.** Every other
path through `HDGDiffusionIntegrator` takes its stabilization from
`StabValue()`, which is what gives `SetStabilization()` its effect: with a hook
set it divides the quadrature weight out, calls `HDGStabilization::Eval()` and
puts the weight back. `ComputeHDGFaceEnergy()` formed the built-in `{h⁻¹Q}`
expression directly instead, so `Type::Energy` measured the energy of a
stabilization that was not the one being solved with, while `Type::Residual` —
the same class, the same `switch` — honoured the hook all along. With the
constant `τ` both Grad–Shafranov papers use, and which is the usual reason to
install a hook at all, the two differ by the local mesh size and the diffusion
coefficient, so the discrepancy **grows under refinement** rather than staying a
fixed factor. Reported from outside as §2 of
`doc/HDG-DEFECTS-FROM-MEQ.md`; the energy now goes through `StabValue()` like
the other four, and with no hook it is the same number to the last bit. The
anisotropic split follows it: the direction weights are geometry and sum to
`ni·nor`, so normalising by that distributes exactly the energy added, whatever
the stabilization was.

The indicator is the trace jump `|p̂ - λ|` — **the same quantity the scheme uses
for stabilisation**, which is elegant and cheap, and the header argues that
refining on it therefore supports convergence of the scheme. Note this differs
from the more common HDG indicator built from the postprocessed solution
`‖u* - u_h‖`; this one needs no postprocessing pass.

`DarcyHybridization` also already branches on `mesh->Nonconforming()` in several
places and walks the NC face list including slave faces, so **hybridization on a
nonconforming mesh is implemented rather than hypothetical**. MFEM's underlying
mesh supports anisotropic refinement through the `ref_type` bitmask for quads,
hexes **and prisms** — for a prism, an XY split of the triangular cross-section
or a Z split along the extrusion axis. Under the mesh constraint above that is
exactly two independent directions, so **the estimator's anisotropic flags have a
consumer**.

**Measured, and there is a precondition on using the estimator at all.** The
indicator is `η_K² = Σ_F ∫_F τ (p̂ − λ)²`, the stabilisation energy, which is
what the header claims. Run against the branch's usual DG boundary
arrangement — faces stabilised on the interior only, the Dirichlet datum
entering weakly through the flux equation — it does not converge:

| `n` | max local | total | true L² error |
|---|---|---|---|
| 4 | 1.4542 | 2.133883 | 5.77e-3 |
| 8 | 1.0855 | 2.133017 | 1.53e-3 |
| 16 | 0.7880 | 2.132926 | 3.84e-4 |

**And the reason is arithmetic, not the estimator.** That arrangement leaves the
boundary trace unknowns dead — empty row, empty column, left at zero, as §4
records — so on a boundary face the jump being integrated is `p_h` itself. With
`τ = 1` the total is then `‖p‖_{L²(∂Ω)}`, which for `p = eˣ sin y` on the unit
square is `√4.549402 = 2.132933`: **the measured value to six figures**, and
unchanged at `k = 2`, which on its own settles that it is not measuring a
discretisation error.

The consequence is worse than a stopping criterion that does not work. Boundary
elements carry `η ~ √h` and interior ones `η ~ h^{k+1.5}`, so the ratio diverges
like `h^{-(k+1)}` — **marking would select boundary elements and nothing else.**

**Constrain the boundary traces and it behaves.** With the essential-trace route
(`SetEssentialBC` and the datum projected onto the trace, which is what
`convdiff -trbc` does), against what the scaling predicts:

| `k` | local pred | local meas | total pred | total meas | L² pred | L² meas |
|---|---|---|---|---|---|---|
| 1 | 2.5 | 2.42 | 1.5 | 1.41 | 2 | 1.90 |
| 2 | 3.5 | 3.41 | 2.5 | 2.46 | 3 | 2.96 |

The total falls half an order slower than the L² error, which is not a defect —
it is an energy-norm quantity and a different norm. Both cases are now pinned in
`tests/unit/fem/test_estimators_hdg.cpp`, the broken one included, so the trap
is documented rather than rediscovered.

**This is the third thing to trip over the same boundary arrangement**, after
§4's unconstrained traces in the Jacobian probe and §Optional B's local problem
going singular the moment the potential space is enriched. The sweep that
followed is below.

### The boundary treatment, swept

**Outcome: the fully discontinuous spaces now take the Dirichlet datum on an
essential trace by default. Raviart–Thomas and broken RT are untouched.**

**Why the traces were dead is one line.** `DarcyHybridization::ConstructC`
loops over faces calling `GetInteriorFaceTransformations(f)` and skips anything
that returns null, so **`C` is assembled on interior faces only** unless a
boundary flux constraint integrator is registered. A boundary trace unknown
therefore had no constraint row *and* no `Cᵀ` column: nothing determined it,
and it influenced nothing.

**Corrected in §9: registering one is easier than this section first said.**
`DarcyForm::EnableHybridization` walks the *divergence form's* boundary face
markers and registers the trace-jump integrator as a boundary flux constraint
on each, so **adding a boundary face integrator to `B` is what gives `C` a
boundary block** — which the DG harnesses do, and which is why their essential
traces inject through `Cᵀ` and not, as this section originally claimed, through
the stabilisation.

**Which route is available depends on where the datum can enter, and the two
spaces differ.**

* **RT and broken RT keep the weak route.** Measured, going through the motions
  of the essential route leaves the datum nowhere and the potential comes out
  right up to a constant, a fixed error of 0.26 at every order and every mesh.
  The reason given here at first — that a boundary flux constraint integrator
  cannot be registered in time — is **wrong**, and §9 says why: `B`'s boundary
  face markers are what register it, and the RT harnesses add no `B` face
  integrators at all, so none is registered. Whether adding one is the fix for
  RT is untested and is the open item below. The weak route is in any case the
  classical hybridized mixed Dirichlet condition and there is nothing wrong
  with it; the price is that `λ` on a boundary face is meaningless and must not
  be read.
* **DG takes the essential route.** The boundary faces join the divergence
  form and the stabilisation, the datum is projected onto the trace, and `λ`
  then means the same thing everywhere. The estimator above needs exactly that,
  and so would an enriched potential space (§Optional B). It is joining the
  *divergence form* that matters: that is what registers the boundary flux
  constraint and gives `C` its boundary block. §9 found the stabilisation's boundary
  contribution to be inert — a shape bug in `VectorBlockDiagonalIntegrator`,
  since fixed.

**What it cost in rates.** A different method, landing a little lower at the
same orders. §4's nonlinear system study and its postprocessing table are both
recorded under the new default; the DG rows moved by three to six hundredths.
`test_darcy_system.cpp`'s linear study did not move at all, and could not have:
its manufactured solution is `A Π sin(π x_d)`, which **vanishes on the whole
boundary of the unit square**, so both routes inject nothing there. That is
worth remembering — a comparison of boundary treatments against homogeneous
data is vacuous, which is why the estimator uses `eˣ sin y` and §4 an offset
sine.

**One library gap had to be closed first.**
`DarcyHybridization::EliminateTraceTrueDofsInRHS` opened with
`if (IsNonlinear()) { return; } // not implemented`, so `SetEssentialBC` was
**silently ignored on the nonlinear path** — no warning, no error, just an
unconstrained trace. The constraint is now carried
the way `NonlinearForm` carries one: the values ride in `x`, `Mult` zeroes the
residual on those rows, `GetGradient` puts a unit diagonal there, and the
reduced right-hand side is zeroed to match, so Newton leaves them alone. Rows
only, not rows and columns — the correction is zero on those dofs so their
columns contribute nothing, and `EliminateRowCol` would demand a structurally
symmetric matrix, which the reduced gradient is not. **The change is inert
unless `SetEssentialBC` has been called, which no RT path does.** A test
asserts the values come out of Newton bit-identical to what went in.

**That fix reaches the miniapps without the miniapps being touched.**
`convdiff -trbc` with any nonlinear option was, before it, quietly solving a
different problem — the trace it asked to be essential simply was not. It now
behaves. **No regression covers the combination**: all three `-trbc` reference
cases are `-p 3` with `--no-nonlinear`, which is why the suite is unchanged
either side of the fix, and also why the defect survived. Flagged for the
branch author rather than acted on here.

**So the h-adaptive requirement, anisotropic included, is largely "verify and
use", not "build".** What has to be checked: that nonconforming refinement of
*prisms* composes with `DarcyHybridization`'s NC handling end to end, since its
NC code path is most likely exercised on hexes in the branch's own examples; that derefinement works,
since a moving layer needs coarsening behind it as much as refinement ahead; and
that any integrator written for §3 supplies `AssembleHDGFaceVector` and
`ComputeHDGFaceEnergy`, since the estimator is only as good as the integrator's
cooperation. Transfer of the
solution across a remesh is the other piece, and is a harder problem than it
looks for a multistep time integrator.

**`p` and `hp` are the real gap.** `fem/darcy/` contains **no variable-order
awareness at all** — no `GetElementOrder`, no `IsVariableOrder`, no `elem_order`
— so `DarcyHybridization` assumes a uniform degree. MFEM has variable-order
machinery in `FiniteElementSpace`, but its convenience layer is H1-centric
(`IsVariableOrderH1`) and no example or miniapp in the tree calls
`SetElementOrder`, so it is lightly exercised. Per-element degrees for a
hybridized method would be new work.

**Which lever is worth the work is a measured question, and there is evidence.**
In a sibling 1-D HDG code, raising the *global* polynomial degree beat adaptive
`h` **by seven orders at matched degrees of freedom on every smooth benchmark**,
and `h`-adaptivity did not pay on any of them once the cost of the extra solves
was counted — which is why that code implemented global degree adaptation and
still has not implemented `h`. But its one benchmark with **limited regularity**
capped the `p` gain at a factor of ~19, matching the predicted
`~ 1/(k+1)²` behaviour.

That is the classical `hp` criterion, and it is the reason both levers are wanted
here rather than just the cheaper one: **`p` where the solution is smooth, `h`
where regularity is limited or a layer is unresolved** — and a thin internal
layer is the second case until `h` resolves it, after which it becomes the first.
Choosing between them needs a **smoothness sensor**, for which the standard
choice is the Persson & Peraire modal-decay ratio; that is a separate small
requirement, and cheap wherever a nodal basis is built from a modal one.

## 8. Time integration of the resulting DAE

A hybridized formulation with auxiliary fields carries a large number of
algebraic constraints: the traces, and every field of §3(f) that is defined by an
elliptic problem rather than by a time derivative.

**A fully implicit DAE solve is the intended route.** Explicit multiscale
schemes that accelerate the approach to equilibrium by rescaling a fast operator
and masking part of the domain are prior art here and are **not** being adopted.
Three reasons, none of them specific to a particular application. Such schemes
slow the *fast* operator, which does not help when the stiffness that binds is a
diffusive limit in the operator they leave alone. They assume a residual
evaluation is cheap, whereas with a chain of auxiliary elliptic solves behind
every residual the lever is fewer steps rather than cheaper ones. And they
converge to the equilibrium of a *modified* equation, where an implicit solve
converges to that of the discrete system itself and keeps time-accurate
transients.

**MFEM wraps CVODE, CVODES, ARKStep and KINSOL, and no IDA** — `IDA` appears
nowhere in the tree, on any branch checked, including those carrying the SUNDIALS
v7 work. Those are ODE and nonlinear solvers, not DAE ones. The
three ways out are: write an `IDASolver` against MFEM's `ODESolver` /
`SundialsSolver` interfaces; drive IDA directly and use MFEM only for assembly;
or eliminate the constraints locally so the global system is an ODE. This
determines the shape of the residual, so decide it early.

**The known objection, and the answers to it.** Implicit methods are avoided for
advection-dominated problems for good reasons: non-symmetric, indefinite
operators, iterative solvers tuned for elliptic and parabolic problems performing
badly on hyperbolic ones, and a global reduction per iteration that hurts GPU and
large distributed machines. That objection is now answered in the literature for
a closely related problem, and the answer is specific enough to be a requirement
rather than a hope. **Measured elsewhere: time steps `2.5×10⁴` larger than the
explicit stability limit, for a `~2500×` net speedup.** The ingredients:

* **A Jacobian-free Newton–Krylov outer solve**, so the Jacobian never has to be
  assembled — which suits a hybridized method, where it is not assembled anyway.
* **Algebraic multigrid preconditioning using AIR** (Approximate Ideal
  Restriction). Standard AMG assumes symmetric positive definite systems; AIR is
  what extends it to the non-symmetric indefinite systems that advection
  produces, and is the reason the objection above is no longer decisive.
* **A preconditioner built from a *lower-order* discretisation than the
  operator** — first-order upwind preconditioning a fifth-order scheme in the
  reported case. The point is sparsity: the low-order matrix is where AMG
  performs robustly, and it stays effective at large time steps. **This is a
  requirement on the code's structure, not just on the solver**: the assembly
  must be able to produce a cheap low-order surrogate of the same operator, which
  is far easier if designed in than bolted on.
* **A cheaper preconditioner that omits some coupling is acceptable.** In the
  reported case the preconditioner neglects the self-consistent field entirely
  and still costs only ~25% against the fully coupled problem.

The two structural answers remain and compose with the above: **static
condensation already reduces the global system to the traces alone**, which is
the smallest object any iterative method would have to work on, and it is
precisely the low-order-surrogate trick that makes a trace-space preconditioner
cheap to form.

## 9. A driver, attempted and withdrawn

**A Stokes-shaped driver was built here and has been removed.** It was meant to
close §3 by composing (a), (b) and (c) in one operator with a manufactured
solution, and it never got that far: it spent its life as a testbed for the
system machinery instead, which is not what §3 needs. A fresh one will be
defined later. What it established is worth keeping, and divides into what is
now in the library and what the next driver should know.

**In the library, tested, and independent of any driver.** Two integrators that
did not exist and could not be expressed:

* `VectorBlockIntegrator` writes one integrator into a chosen *(row-field,
  column-field)* block of a rectangular layout — element, mixed-element and
  face overloads. This is §3(f)'s enabler.
* `VectorBlockDiagonalHDGIntegrator` replicates an HDG face integrator down the
  diagonal of a system whose potential and trace spaces carry **different**
  numbers of fields. `VectorBlockDiagonalIntegrator` infers one multiplicity
  for every space it touches and cannot describe that.

Both are exercised in `tests/unit/fem/test_darcy_system.cpp` against every
block of a rectangular layout, so §3(f) is built and unit-tested whatever
happens to the drivers.

**Three defects were found by running a convergence table**, and all are fixed:
the boundary HDG face block that `VectorBlockDiagonalIntegrator` built in the
wrong shape and the caller then silently discarded (§7); the same wrapper's
inability to describe unequal field counts; and, before those, the two in §4
that only a manufactured solution could have caught. That is the argument for
having a driver at all — **a convergence table finds things a unit test does
not**, because it exercises the whole path against an answer that is known.

**What the next driver should know.**

* **The couplings a system needs are `B` blocks.** A derivative of another
  solved field is a first-order cross-field coupling; the place for it is the
  block that couples flux to potential, and `DarcyForm` forms `±Bᵀ` itself, so
  one placement delivers its adjoint. Nothing needs assembling in a particular
  order and no derivative of a discontinuous field is ever taken.
* **A field with no flux of its own does not fit as another Darcy field.** The
  first attempt gave it a dummy flux to keep the wrappers square, and its
  potential came out under-determined; the near-singular solve then damaged the
  fields that were fine. The rectangular arrangement is the one that works.
* **Diagnose by applying the assembled blocks to the exact solution.** Two sign
  guesses failed to find that a row was taking a sum where it needed a
  difference; one run comparing two blocks at the exact solution found it, and
  said which sign, because the two agreed to 7e-18 and only their combination
  was in question.
* **A constant null mode has to be pinned, and it disguises itself.** Pinning
  it with a small mass leaves the conditioning; unpinned, it surfaces as
  `SparseMatrix::EliminateRowCol #4` with an essential trace and as a NaN in
  GMRES without one, neither of which reads as "your system is singular". An
  hour went into reading the first of those as a structural property of the
  coupling. It was not.
* **Errors are only meaningful modulo the null mode.** Whatever carries the
  undetermined constant — for Stokes the pressure, and the stress that carries
  it on its diagonal — has to have the mean removed before its error means
  anything. Removing it from one and not the other made a converging quantity
  look stalled.

## Optional A. Interpolatory evaluation of the nonlinear coefficient

**Not a requirement.** This and §Optional B are candidate work that came out of
§4's postprocessing measurements, written up so the decision can be taken on
numbers rather than on impressions. Neither has any test in the suite yet.

The reference is **CCSZ-I**, with **CCSZ-II** extending it. Both are for the
scalar reaction-diffusion problem `∂ₜu − Δu + F(u) = f`, and the technique is
older than HDG — it is the group finite element method, or interpolated
coefficients, or product approximation, depending on who is writing.

### What the method is

Two pieces that separate cleanly.

**(a) Interpolate the nonlinearity.** Replace `F(u_h)` by `I_h F(u_h)`, its
elementwise Lagrange interpolant at the nodes of a polynomial space. Then

    (I_h F(u_h), v_h)_K  =  A₉ · [F(γ₁), …, F(γ_N)]ᵀ

with `A₉ = [(χ_j, φ_i)]` assembled **once**, and `γ` the nodal values. The
Jacobian is `A₉ diag(F′(γ)) · (nodal evaluation map)` — also assembled once
apart from a diagonal. No quadrature in the iteration loop, and the Jacobian is
written down rather than assembled.

**(b) Evaluate it at the postprocessed solution.** Use `I_h F(u*_h)` with
`u*_h ∈ P^{k+1}` the elementwise postprocessing, not `u_h ∈ P^k`.

**The point to hold on to is that (b) exists to repair damage (a) does.** The
authors' earlier paper did (a) alone, kept optimal rates, and *lost*
superconvergence. CCSZ-I adds (b) and proves it comes back. So the interpolatory
approach is a **cost optimisation**, and CCSZ-I is the recipe for taking it
without paying accuracy for it.

That matters for us because §4 measured the branch's nonlinear problem
postprocessing at `k+2` already — 3.09 and 2.92 at `k=1`, 4.12 and 3.94 at
`k=2`. There is nothing to restore. What is on offer here is speed.

### The mismatch, and what it costs

Their nonlinearity is a **reaction term in the potential equation** with linear
diffusion. Ours is a **coefficient in the flux equation**, `D(p) q`. The
consequence is not cosmetic:

* For them the whole matrix `A` is constant in time and only a small separate
  block changes, so "assemble the HDG matrices once before the time
  integration" is nearly total.
* For us the nonlinearity is *inside* the flux mass block. The analogue is the
  product approximation `I_h[D(p*)] q_h`: precompute, per element, the tensor
  `T_{ijl} = (χ_l φ_j, φ_i)_K` and contract it with the nodal `D_l` each
  iteration. Quadrature disappears from the loop, but the local block is still
  rebuilt and refactorised every time.

So the structural win is smaller for us than for them. The remaining win is real
but narrower: **it is the constitutive law's own evaluation cost.**

### Measured: how much redundant evaluation there is

Counted on §4's own two-equation DG study at `k=1`, over three meshes:

| | evaluations per element, per outer Newton step |
|---|---|
| as it stands — quadrature re-run for every residual and Jacobian assembly, including the element-local Newton | **82–89** |
| interpolatory — one per node of `P^{k+1}`, doubled if value and derivative are separate calls | **9–18** |

**Four to nine times fewer calls into the constitutive law.** Whether that is
worth anything depends entirely on what the law costs. For the 2×2 matrix
inverse in the unit test it is worth nothing — the local dense factorisation
dominates. For a collision operator it plausibly dominates instead. **That
measurement, against a representative coefficient, is the first task and the
decision criterion; everything below is conditional on it.**

### Does hybridization survive? Yes

This is the part that could have killed it and does not. The postprocessing is
elementwise: `u* = B₁₁ α + B₁₂ β` with `B₁₁`, `B₁₂` block diagonal and computed
once. Evaluating the law at `p*` therefore keeps `LocalNLOperator` element
local, and the element-local solve stays element local. CCSZ-I static-condense
for the same reason.

The Jacobian gains terms through the chain rule on `p*`, in **both** columns of
the local Jacobian. The `(0,1)` column is the `Bnl_data` / `grad_Aup` path
restored in `fd028d151b`, so that machinery exists. What is new is an extra
`(0,0)` contribution from `∂p*/∂q`, which the branch has no hook for.

### What would have to be built

1. **A representative-cost profile.** See above. Stop here if the law is cheap.
2. **The classic local postprocessing, general in `vdim`.** CCSZ-I's `u*` is a
   *different and much smaller object* than the branch's
   `ReconstructFluxAndPot`; the next subsection sets the two side by side.
   Generalising CCSZ-I's to `vdim` is a loop over equations. This corrects what
   "What is still open" said: postprocessing for a system is only hard if it has
   to be the branch's superconvergent flux-and-potential reconstruction.

   (Do not confuse CCSZ-I's `u*`, built from `q_h`, with CCSZ-II's
   `𝔭^{k+1}(u_h, û_h)` of §Optional B, built from the potential and its trace.
   They are different operators serving different methods.)
3. **An interpolatory variant of `MixedConductionNLFIntegrator`.** It needs a
   per-element cache of `T_{ijl}`, keyed on the element, invalidated on mesh
   change; the nodal `D_l` and `∂D_l/∂p`; and the contraction. `MixedFluxFunction`
   already exposes exactly what is required — `ComputeDualFlux` and
   `ComputeDualFluxJacobian` — and `FrozenDualFluxCoefficient`, added for the
   postprocessing, is the same evaluation in a different wrapper.
4. **The `∂p*/∂q` hook in the local Jacobian**, alongside the existing `(0,1)`
   one.
5. **Re-run §4's study.** The manufactured solution and the control are already
   there and are exactly the harness this needs: rates must not move.

### The risk to state plainly

**The theory does not cover our nonlinearity.** CCSZ-I and II prove their
results for `F(u)` alone. CCSZ-II's own concluding paragraph says that
"superconvergent HDG methods for equations with the more general nonlinear term
`F(∇u, u)` constitutes a subject of ongoing work" — and a diffusion coefficient
multiplying the flux is exactly that case. The authors elsewhere name the
p-Laplacian and nonlinear elasticity as places the technique "can be applied",
but do not do it.

So: the implementation transfers, the proof does not, and step 5 is not a
formality. If rates degrade, CCSZ-I's own remedy — interpolate at `p*` rather
than at `p_h` — is the first thing to try, and it is why step 2 is in the list
at all.

### The two postprocessings are not the same object

Worth spelling out, because the branch's is the heavier of the two by a wide
margin and the difference decides how much of step 2 is real work.

**CCSZ-I (4a)–(4b)** — the classical local postprocessing. On each element,
find the one scalar field `u* ∈ P^{k+1}(K)` with

    (∇u*, ∇z)_K = −(q_h, ∇z)_K   for all z ∈ [P^{k+1}(K)]^⊥
    (u*, w)_K   = (u_h, w)_K      for all w ∈ P^0(K)

Its only data are `q_h|_K` and the element average of `u_h`. No trace, no
enriched flux space, no constraint integrators, no coupling to neighbours. In
the Lagrange-multiplier form (7a)–(7b) it is one dense solve of size
`dim P^{k+1}(K) + 1` — ten by ten for a `k=1` quadrilateral. Its output is a
superconvergent potential and nothing else.

**The branch** follows **NPC-1** §4 instead — that is the paper
`DarcyHybridization`'s header cites for it, the *linear* convection–diffusion
one, not the nonlinear NPC-2 — and it is two passes:

* `ReconstructTotalFlux` uses the constraint equation to project the total flux
  onto the face restriction of an RT space, filling the interior degrees of
  freedom by integral projection. Its output `q̂` is single-valued and in
  H(div) — a first-class quantity, and the one §6 needs.
* `ReconstructFluxAndPot` then re-solves the **whole local mixed problem** on
  spaces one degree higher, with the total flux as the element source: the full
  `[M_u −Bᵀ Cᵀ; B M_p E; C G H]` block over enriched flux, potential *and*
  trace. For a `k=1` DG element in 2D that is `18 + 9 + 12 = 39` unknowns
  against CCSZ-I's ten — of order sixty times the work per element, on top of
  the first pass. It also needs the *linear* forms `M_u`, `B`, `M_p` and the
  constraint integrators `c_bfi`, `c_bfi_p` to exist, which is exactly why it
  had never been run against a nonlinear law and dereferenced a null pointer
  when it was, and why it is entangled with `vdim`.

**What the extra buys, measured.** §4's table gives `u_s` tracking `u_h` to
within a few hundredths of an order in all six rows, and `u_t` likewise. The
potential rate — the one thing CCSZ-I's ten-by-ten also delivers — is `k+2` in
both. So on that problem the second pass's extra output, a flux in a richer
space, is not converging any faster than the solved flux.

**The caveat on that reading.** The manufactured problem is pure diffusion, so
the total flux `qt = q + c·u` degenerates to `q` and the first pass is only
projecting `q_h` into H(div) — for RT it returns `q_h` unchanged, which is why
`u_t` and `u_h` agree to every digit in those rows. Nothing here exercises what
`ReconstructTotalFlux` is *for*. The honest conclusion is narrower than it
looks: for a superconvergent **potential**, CCSZ-I's postprocessing appears to
be all that is needed and is a fraction of the cost; the branch's first pass
remains the right machinery for §6 and should not be judged on this problem.

**Decision, taken: keep the branch's reconstruction and do not replace it.**
The extra work it does is real but it is paid once per solve, not once per
Newton step, and it produces the total flux §6 asks for as a by-product. There
is no measured cost complaint against it — only the observation that a cheaper
route to the same potential rate exists. Revisit only if one of two things
happens: postprocessing turns up in a profile as a material share of a run, or
a system needs a superconvergent potential and §Optional B has not been done,
in which case CCSZ-I's ten-by-ten is the cheap way round the `vdim` problem.
Until then this is recorded, not scheduled.

## Optional B. Superconvergence at `k = 0` — the HHO-inspired methods

**Not a requirement**, and no test in the suite yet. §4 measured the
postprocessed potential at `k+2` for every case except the fully discontinuous
form at `k=0`, where it reads 1.01. That is the known limit of the `HDG_k`
method — **CCSZ-I** Table 1 reports 0.97 there and its Theorem 3.19 assumes
`k ≥ 1`. **CCSZ-II** is the way out, and it is a different method rather than a
tuning of this one.

### The three methods

All three keep `V_h = [P^k]^d` for the flux and `M_h = P^k` for the trace, and
differ in the potential space `W_h = P^ℓ` and in what the stabilization
penalises. Writing `𝔭^{k+1}(u_h, û_h)` for the HHO reconstruction — the element
of `P^{k+1}(K)` with

    (∇𝔭, ∇z)_K = −(u_h, Δz)_K + ⟨û_h, n·∇z⟩_∂K   for z ∈ [P^{k+1}_ℓ(K)]^⊥
    (𝔭, w)_K   = (u_h, w)_K                        for w ∈ P^ℓ(K)

— the numerical flux trace is

    q̂·n = q·n + r*_∂K [ h_K^{-1} r_∂K(u_h − û_h) ],   r_∂K(u_h − û_h) = Π^∂_k u*_h − û_h

with `u* = 𝔭^{k+1}(u_h, û_h)`, so the penalty acts on the **projected trace of
the reconstruction** rather than on `u_h − û_h`, and `τ ~ 1/h`.

| | `W_h` | `u*` | `τ` | `q` | `u*` |
|---|---|---|---|---|---|
| **(A)** Lehrenfeld–Schöberl / HDG+ | `P^{k+1}` | `u_h` itself | `1/h` | `k+1` | **`k+2`, `k ≥ 0`** |
| **(B)** | `P^k` | `𝔭^{k+1}(u_h, û_h)` | `1/h` | `k+1` | **`k+2`, `k ≥ 0`** |
| **(C)** | `P^{k−1}` | `𝔭^{k+1}(u_h, û_h)` | `1/h` | `k+1` | `k+2`, `k ≥ 2` |

CCSZ-II Table 2 confirms each numerically. Note the `τ ~ 1/h` scaling, which is
the *opposite* of what §5 and §4 settled on for the equal-order method — a
reminder that `τ` is not separable from the choice of spaces.

### (A) is the one to want, for a reason beyond `k = 0`

For (A), `ℓ = k+1` forces `𝔭 = u_h` and the reconstruction is vacuous: **the
potential is superconvergent as solved, and there is no postprocessing step at
all.** For a system that is decisive. §4 records that the branch's
reconstruction is scalar-only and lists five distinct pieces of work to widen
it; (A) sidesteps the whole of it, because there is nothing to reconstruct.

(B) is the more faithful HHO method but is structurally worse for this branch.
Its `u*` depends on the element's potential **and on all of its face traces at
once**, so the stabilization is an element-level operator coupling one element's
potential dofs to every one of its traces. `DarcyHybridization` assembles HDG
faces per (face, side) — `AssembleHDGFaceGrad` produces `D`, `E`, `G`, `H` for
one face — and there is no element-level hook. (B) would need one. (A) needs
none: with `u* = u_h`, the penalty is face-local in exactly the shape the
existing hook produces.

(C) buys nothing here: `k ≥ 2` only, and CCSZ-II notes that the `k=1` rate
claimed for it in the earlier literature is wrong.

### What (A) actually changes in the assembly

Very little, and this is the encouraging part. Write `M_F = ⟨ψ_a, ψ_b⟩_F` for
the trace mass on a face and `C = ⟨ψ_a, φ_i⟩_F` for the mixed trace–element
mass. The existing penalty `τ⟨u_h − û_h, v_h − v̂_h⟩_F` assembles as

    (el,el) τ⟨φ_i, φ_j⟩_F     (el,tr) −τ Cᵀ     (tr,el) −τ C     (tr,tr) τ M_F

and (A)'s `h^{-1}⟨Π^∂_k u_h − û_h, Π^∂_k v_h − v̂_h⟩_F`, with `Π^∂_k u_h|_F =
M_F^{-1} C u_el`, assembles as

    (el,el) h^{-1} Cᵀ M_F^{-1} C   (el,tr) −h^{-1} Cᵀ   (tr,el) −h^{-1} C   (tr,tr) h^{-1} M_F

**Only the `(el,el)` block differs**, and only by one face-mass inverse;
everything else is what `HDGDiffusionIntegrator` already produces with a fixed
`td`, which is precisely `τ = td·κ/h`. So (A) is: a potential space one degree
up, a fixed `td`, and one changed block in one integrator.

### Measured: how far the branch already gets

Probed on the single-field nonlinear manufactured problem of §4, DG flux,
hybridized, with the potential space raised one degree and the **plain**
penalty at `τ ~ 1/h` — that is (A) without the `Π^∂_k` projection. Rates over
the 16×16 to 32×32 pair:

| | `p` | `u` |
|---|---|---|
| `k=1`, `W = P^k` (as now) | 2.12 | 1.12 |
| `k=1`, `W = P^{k+1}` | **3.22** | **2.21** |
| `k=0`, `W = P^k` | −0.12 | −0.03 |
| `k=0`, `W = P^{k+1}` | 0.07 | 0.34 |

At `k=1` the potential reaches `k+2 = 3` **as solved, with no postprocessing** —
which is (A)'s claim, arrived at without the projection. At `k=0` it locks, and
so does the equal-order method at this `τ`. That split is what one would expect
if the projection is the essential ingredient: the plain penalty forces all of
`u_h|_F ∈ P^{k+1}(F)` to match `û_h ∈ P^k(F)`, one constraint per face too many,
and `Π^∂_k` is exactly the relaxation of it. Proportionally that over-constraint
is worst at `k=0`, which is where it locks.

**What the probe found that has to be dealt with first.** With the
potential space raised, the local problem on a boundary element is
under-determined and the solve diverges — from the coarsest mesh, and for the
*linear* problem too, so it is not the nonlinear solver. Stabilizing the
boundary faces fixes it outright. The branch's DG arrangement, copied from
`convdiff`, stabilizes interior faces only and imposes Dirichlet data weakly
through the flux equation, leaving the boundary traces dead — §4 records that.
That arrangement does not survive a richer potential space, so **(A) needs its
boundary condition rethought**, most likely as the essential-trace route
`convdiff` already has behind `-trbc`.

### What would have to be built

1. **A boundary treatment that constrains the boundary traces** — probably
   `DarcyHybridization::SetEssentialBC` with the trace projected from the
   Dirichlet data, which exists and is exercised by `convdiff -trbc`. Needed
   before anything else, and worth having on its own account.
2. **A projected variant of `HDGDiffusionIntegrator`** — one block, as above,
   plus the face mass factorisation. It is also the natural place to check the
   claim that the projection is what unlocks `k=0`.
3. **Unequal-order plumbing.** `DarcyForm` takes the two spaces separately and
   the probe assembled and solved with them unequal, so the framework tolerates
   it. What has *not* been checked is the rest: `MixedConductionNLFIntegrator`'s
   HDG face methods set `ndof_u = el_p.GetDof()`, harmless today but an
   equal-order assumption written down; the reconstruction; and the estimators.
4. **A convergence study**, orders 0 to 2, both forms, against §4's manufactured
   solution — the harness exists and already reports `p`, `u`, `u_t`, `p_s`,
   `u_s`.

### Why this may be worth more than Optional A

(A) delivers `k+2` on the *solved* potential, for a system, with no
postprocessing and therefore none of the scalar-only limitation §4 records. It
does so by making the potential space richer, which costs local dofs but not
global ones — the trace space, and hence the globally coupled system, is
unchanged. Against that, it is a change to the discretisation rather than to the
implementation of one, and every measurement in §4 and §5 would have to be
re-taken against it, `τ` included.

## Dependencies, and a sensible order

```
3(a) full tensors ─┐
3(c) convection    ├─→ 4 systems + exact Jacobians ─→ 8 time integration
3(d) degenerate   ─┘
3(e) singular  ──── may vanish; check the change of variable first
1 extension/lifting ─→ 2 coupling at a distance ─→ 2(biharmonic exterior)
6 trace functionals   (independent; needs only a working trace)
7 h-adaptivity ── largely present; verify. hp ── new work, needs 3 first
5 τ ──── calibrate once 3 and 4 exist
```

§3 is the critical path and the most likely to be partly there already. §2's
biharmonic case is the one with no prior art and should be settled on paper
early, since it may change what §1 has to supply.

**Where the implementation actually stands, as against the testing.** Every
operator term §3 asks for exists — a full varying tensor and a reaction in
`anisodiff`, convection in `convdiff`, a floored `τ` through
`HDGStabilization::SetStabilization`, and now (f) through the block placement
integrators. §4 is complete. What §3 still lacks is a driver that *composes*
(a), (b) and (c) and measures rates against them — work in a miniapp rather
than in the library, and still to be defined; §9 records what the first attempt
learned. **Nothing in §3 or §4 is a prerequisite for
§1**, which sits on the other branch of the graph above; the remaining
implementation items — a `vdim`-general postprocessing, §6's flux functional,
`hp`, §8 — are each independent of it, and of each other.

The two optional sections sit off this graph, and in one order if both are
wanted: **Optional B before Optional A.** B changes the discretisation, so
every calibration in §4 and §5 would have to be re-taken after it; A is an
implementation of a fixed discretisation and would have to be re-measured
against whichever one is settled on. B also makes A's hardest dependency — a
`vdim`-general postprocessing — unnecessary, since HDG (A) has nothing to
postprocess. Neither is on the critical path, and A is gated on a single
measurement that may well close it.

## First questions to answer in the branch — answered

Kept with their answers rather than deleted, because the answers are the content.

1. **Full matrix coefficient?** Yes. `HDGDiffusionIntegrator` has a
   `MatrixCoefficient` overload, and one combining a `VectorCoefficient` with it.
2. **Zeroth-order term, composing with convection and a tensor?** Yes; the
   potential mass form takes it, and the three compose.
3. **Degenerate coefficient?** Works, at design order, with no quadrature or
   node-placement intervention — see §3(d), including what that measurement does
   *not* cover.
4. **Analytic Jacobian rather than differencing?** Accepted, and correct: the
   framework takes the derivative alongside the coefficient, and `J dy` matches a
   differenced residual.
5. **What does "linear systems so far" exclude?** Cross-equation coupling through
   anything but the hyperbolic flux — see §4.
6. **Evaluating an FE function outside its element?** Essentially free.
   `InverseElementTransformation` with the Newton solver does not restrict
   iterates, and the reference point is written before it is classified, so
   out-of-element coordinates survive an `Outside` return. Nothing in the tree
   uses this. Relevant to §1.
7. **Is `braginskii_hdg` worth reading first?** Yes, and it should have been read
   first. It is the only place in the tree that assembles a system, and its
   arrangement — `vdim = neq` spaces with `VectorBlockDiagonalIntegrator` — is
   what §4 above is written against.

## What is still open

1. **§1 and §2**, untouched. Nothing in §3 or §4 blocks them: they sit on their
   own branch of the dependency graph.
2. **§7's `hp`**, and §8 in its entirety.
3. **Whether the degenerate order loss is asymptotic** — recorded in §3(d),
   where the practical answer is already known: floor the stabilisation. The
   estimator's flat total, which used to sit here, is settled in §7 — it was
   the boundary arrangement, not the estimator.
4. **Postprocessing for a system.** The reconstruction is scalar-only, for the
   several reasons §4 lists, so the two-equation study cannot be postprocessed
   and the superconvergence table there is a single field. Making it general in
   `vdim` needs the enriched spaces built with a `vdim`, `GetElementVDofs`
   throughout the kernel, a vector `DivergenceGridFunctionCoefficient`, a
   `vdim`-aware `DarcyHybridization::ReconstructTotalFlux` with a
   vector-valued callback, and a linearised potential constraint for a system
   stabilized by the nonlinear face integrator.

   Two things qualify this. The *classic* local postprocessing — the small
   per-element solve CCSZ use, not the branch's flux-and-potential
   reconstruction — is a loop over equations away from being general in `vdim`;
   §Optional A step 2. And §Optional B would remove the need entirely for the
   quantity that matters, since HDG (A) is superconvergent as solved.
5. **The essential-trace route for RT.** `C` gets a boundary block from the
   *divergence form's* boundary face markers, and the RT harnesses add no `B`
   face integrators, so nothing registers one — whether adding one is the fix
   is untested. Matters only where `λ` on a boundary face is read. The inert
   boundary stabilisation that sat alongside this is fixed; see §9.
6. **The miniapps still default to the weak route for DG, and are being left
   that way deliberately.** The sweep changed the unit-test harnesses, not the
   drivers. Moving `convdiff` and its siblings is the branch author's call,
   not ours, and it would move their regression references; it is being raised
   with them rather than done here. The same goes for the `-trbc` gap above,
   which the library fix has already closed but which nothing in the suite
   exercises.
7. **A constant null mode needs pinning properly**, for any problem that has
   one. Pinning it with a small mass works but leaves the conditioning: the
   withdrawn driver's `k=2` finest mesh took 2178 GMRES iterations and the rate
   collapsed. A mean-zero constraint, or a preconditioner that knows about the
   mode, is the real answer. **This subsumes the item that used to sit here** —
   essential traces looked unavailable alongside a first-order cross-field
   coupling, and that was this same mode, not anything to do with essential
   conditions. Withdrawn; see §9.

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
* **CS-Extensions** — Cockburn & Solano, on solving problems posed on curved
  domains by extension from a polyhedral subdomain, reducing the boundary
  treatment to line integrals along transferring paths. §1.
* **CSS-Coupling** — Cockburn, Sayas & Solano, on coupling an HDG interior solve
  to an exterior boundary-integral representation across an unmeshed interface,
  with **CSS-Analysis** its companion analysis, including the relaxed iteration
  and the contraction estimate. §2.
* **Persson & Peraire**, modal-decay smoothness sensor, cited in §7 as the
  standard choice for an `hp` criterion.
