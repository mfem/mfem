# Coupling HDG at a distance to an exterior operator

A request from meq, written 2026-08-29. Nothing in this tree has been changed.

This asks for what `refs/CouplingAtADistance.pdf` — Cockburn, Sayas & Solano,
*Coupling at a Distance HDG and BEM*, SIAM J. Sci. Comput. 34 (2012) A28–A47 —
needs in order to be expressible on this branch. **Most of it is already here**:
that paper's reference [5] is Cockburn & Solano, which is this branch's
`fem/darcy/extension_hdg.*`, and the paper's `Σ_h`, `E_h(q_h)` and `L_h(g)` are
`TransferPath`, `ElementExtension` and `TransferredDatumCoefficient` term for
term. What is missing is the *coupling*: the artificial boundary's datum is an
unknown, and there is currently no way for a hybridized system to carry an
unknown that is not a trace dof.

The request is in two parts of very different size, and **the small one is
useful on its own** — it is asked for first for that reason.

meq's use is a free-boundary Grad–Shafranov equilibrium, where the exterior is
the vacuum region outside a tokamak. The full plan is `meq/FREE-BOUNDARY-PLAN.md`
in the sibling tree; nothing below depends on reading it.

---

## 1. What the coupling looks like, in this branch's terms

The interior problem is the one `DarcyForm` with `EnableHybridization` already
solves, on a polygonal `D_h` inside a smooth artificial boundary `Γ`, with the
Dirichlet datum on `Γ_h` transferred along the paths:

```
φ_h(x) = g( a(x) ) + ∫_{σ(x)} C E_h(u_h) · m ds          x ∈ Γ_h
```

Today `g` is data — `PathTraceCoefficient` takes a `PositionFunction` — and
`HDGExtensionIntegrator` carries the second, solution-dependent term into the
element's flux mass block, where the hybridization never has to see it.

**In a coupled problem `g` is an unknown.** It is expanded in a small basis on
`Γ`,

```
g = Σ_{n=1..N} a_n φ_n         N of order 20 to 40,
```

and the exterior supplies `N` more equations, which in the paper are the
transmission condition `E_h(q_h)·ν + λ = 0` on `Γ` tested against that same
basis. The coupled system is then

```
[  K    B  ] [ Λ ]     [ F ]          K = the hybridized trace matrix
[  T    D  ] [ a ]  =  [ G ]          B = ∂(trace residual)/∂a
                                      T = the transmission rows
                                      D = the exterior operator, N × N
```

`B` and `T` are **sparse**: `g` reaches the datum on a `Γ_h` face only through
`g(a(x))` for `x` on that face, and the transmission integrand reads `E_h(q_h)`
on the far end of the paths from that same face. So both blocks touch only the
elements owning `Γ_h` faces. `D` is `N × N` and dense at worst; in meq's case it
is diagonal, for reasons that are meq's geometry and not this branch's business.

**A distinction §2 and §3 turn on.** The integrals above are the *raw* blocks, on
the flux dofs of an element. `B` as written in the system is those blocks after
the element-local elimination, which is this library's to perform. §2 asks for
the raw blocks; §3 asks for them to be carried through the elimination. Neither
is useful without the other, but they are separable pieces of work and only §2
has to happen first.

---

## 2. The small ask: two rectangular integrators

Both are element-local in exactly the sense `HDGExtensionIntegrator`'s own
comment argues — a face of `Γ_h` belongs to one element, and the extension beyond
it is that element's own polynomial — so neither disturbs the hybridization, the
constraint, or the static condensation.

### 2.1 `⟨ φ_n ∘ a, v·n ⟩_e` — the block `B`

The datum's data half, as a rectangular form against a basis rather than as a
coefficient for one fixed `g`. Something of the shape

```cpp
/** @brief The data half of the transferred datum, against a basis of boundary
    functions: the rectangular block  <phi_n o a, v.n>_e  on a face of Gamma_h.

    PathTraceCoefficient computes g o a for a given g. This is the same
    quantity for a family { phi_n }, assembled as a rectangular element matrix
    from the flux space's dofs on the owning element to the N coefficients, so
    that g may be an unknown rather than data. */
class PathTraceMixedIntegrator : public BilinearFormIntegrator
{
public:
   /// @param path_   the transferring paths, the same family the extension
   ///                integrator was given.
   /// @param basis_  phi_n evaluated at a point of Gamma; n = 0 .. N-1.
   PathTraceMixedIntegrator(const TransferPath &path_,
                            std::function<void(const Vector &xbar,
                                               Vector &values)> basis_,
                            int N_, real_t sign_ = 1., int ir_order_ = -1);

   void AssembleFaceMatrix(const FiniteElement &flux_fe,
                           FaceElementTransformations &FTr,
                           DenseMatrix &elmat) override;   // ndof_flux x N
};
```

Everything it needs exists: `TransferPath` supplies `a(x)`, and the face
quadrature and normal are what `HDGExtensionIntegrator` already sets up. The only
new thing is that the result is rectangular.

**A caller-supplied basis rather than a fixed one**, because what the basis *is*
belongs to the exterior problem and therefore to the caller. meq's is a set of
Gegenbauer functions on a semicircle; the paper's is trigonometric polynomials on
a circle; neither is this branch's concern.

### 2.2 `⟨ E_h(v_i)·ν, φ_n ⟩_Γ` — the block `T`

The transmission condition needs the extended flux evaluated **on `Γ`** — the far
end of the paths — and paired with the same basis over the induced measure there.

`ExtensionQuadrature` already sweeps `K^ext_e`, the region between a `Γ_h` face
and `Γ`, and `ExtensionPoint` already carries `xbar`, the end `a(x)` of each
path. What is missing is the *boundary* piece: a quadrature over the image
`a(e) ⊂ Γ` of one `Γ_h` face, with the correct surface measure on `Γ` and the
normal `ν` there rather than on `Γ_h`. In the notation of the existing header,
that is the `t = 1` face of the map `y(ξ,t) = x(ξ) + t( a(x(ξ)) − x(ξ) )`, whose
Jacobian the sweep already forms.

The same tiling caveat applies and is worth restating in whatever is written:
summing these boundary weights over the faces must give `|Γ|`, in the way summing
the volume weights must give `|Ω| − |D_h|`. **That is the check that says the
path family covers `Γ` exactly once**, and it is as cheap as the existing one.

The paper collocates this condition rather than integrating it (their eq (4.1),
at `2n` points). Collocation would need only point evaluation of `E_h(q_h)` at
`a(x)`, which is strictly less than the above — but it gives up the variational
structure, and for a Newton method on a nonlinear interior problem the
variational form is what keeps the Jacobian symmetric with the residual. **The
Galerkin version is the one worth building**; collocation falls out of it as a
special case of the rule.

---

## 3. The structural ask: auxiliary globally-coupled unknowns

**`DarcyHybridization` has no concept of a global unknown that is not a trace
dof, and that is what stops §2's blocks from reaching the reduced system.**

`B` is a rectangular block on an element's flux dofs. Static condensation
eliminates that element's flux and potential in terms of its trace, and `B`'s
column must be carried through that elimination the way `C` and `E` are — but its
other index is not a trace dof, so there is nowhere for it to go. The same is
true of `T` in the other direction.

What is being asked for is the ability to declare `M` extra unknowns, coupled to
the element-local blocks, eliminated along with them, so that

* `Mult()` returns a residual of length `n_trace + M`;
* `GetGradient()` returns the bordered matrix `[ K B ; T D ]`;
* `ComputeSolution()` recovers flux and potential given both halves.

Something like

```cpp
/** @brief Declare M auxiliary unknowns coupled to the element-local blocks.

    The reduced system becomes [ K B ; T D ] of size n_trace + M. Blocks are
    assembled per element the way the trace blocks are: AssembleAuxFluxMatrix
    for the flux coupling, AssembleAuxPotMatrix for the potential coupling,
    and SetAuxBlock for the M x M part, which the caller owns entirely. */
void SetNumAuxiliaryUnknowns(int M);
void AssembleAuxFluxMatrix(int el, const DenseMatrix &B_el);
void AssembleAuxPotMatrix (int el, const DenseMatrix &B_el);
void SetAuxBlock(const DenseMatrix &D);
```

### 3.1 Why this is worth doing whatever meq does

**It is not a free-boundary feature.** An auxiliary unknown coupled to a
hybridized system is the shape of every one of these:

* **Any global constraint** — a mean-value condition, a total-flux or total-flow
  condition, a Lagrange multiplier pinning a pure Neumann problem. The last is
  interesting because `ReconstructFluxAndPot()` already solves exactly that
  problem element-locally, with the element average as the closure.
* **A boundary integral operator**, which is what this document is about, and
  which is a rank-`M` nonlocal term however it arises.
* **A parameter the solution is constrained by.** meq has one of these already
  and it is the cleanest illustration: `ψ` on the magnetic axis is a functional of
  the solution and enters the source through it, so meq's Newton carries it as an
  extra unknown with a bordered Jacobian. That works today entirely *outside*
  this library — one extra backsolve per Newton step, and both border entries
  obtained by **differencing the residual**, because the derivative of the
  condensed residual with respect to a source parameter is not something this
  branch exposes.

That last one is worth dwelling on, because it is what a caller does when this
capability is missing, and it is not cheap or safe. meq's border is differenced,
and differencing a condensed residual turned out to be a much sharper instrument
than it looks: the element-local Newtons are seeded from a vector captured at
`FormLinearSystem()` time and never refreshed, so on a hard problem they were
hitting their iteration cap and returning something that was not a function of
the trace at all. Perturbing the parameter by `9e−6` moved the recovered maximum
from `0.896` to `3.84`, the border read `1.6e5` where it should read about `1`,
and the iteration stalled looking exactly like a singular Jacobian. meq works
around it by re-forming the system each step. **With auxiliary unknowns the
border would be assembled and none of that would arise.**

(The seed itself is a separate matter and is not what this document asks for.
It is recorded in `meq/CLAUDE.md` under *Newton, and the obligation it creates*
if it is ever of interest; meq is not blocked on it.)

### 3.2 What §2 gives without §3, and it is less than it looks

§2's integrators produce the **raw** blocks — integrals over `Γ_h` faces and over
`Γ`. What the outer solver needs is `∂R/∂a`, those blocks pushed through the
element-local elimination, and that is inside this library either way. So the
fallback is not "assemble `B` and eliminate outside"; it is:

* evaluate the reduced residual `R` at `M` perturbed values of `a`, one per
  auxiliary unknown, and difference;
* then solve the bordered system by block elimination against one factorisation,
  `M` backsolves, which `SetReuseSymbolic()` already makes affordable.

**CORRECTION, 2026-08-30.** An earlier version of this section claimed that the
fallback exists under `CondenseThenLinearise` and **not** under
`LineariseThenCondense`, on the grounds that an auxiliary unknown enters only
through a retained `r_lin` and so a difference of the reduced residual would
return exactly zero. **That was read out of this header's own summary of the
ordering and is not what the code does.** `darcyhybridization.hpp`'s description
of `SetNonlinearOrdering()` prints the substitution as

```
(q, u)(L) = (q, u)_lin + M^-1( -r_lin - [C; E](L - L_lin) )
```

with `r_lin` among the quantities `GetGradient()` refreshes and retains.
`Relinearise()` states the opposite in its own comment — the local residual at
the linearisation point is *deliberately not retained*, because retaining it made
it enter the substitution twice and cost the gradient its exactness — and
`MultInvLin()`'s correction loop calls `LocalResidual()`, which constructs a
`LocalNLOperator` and evaluates it at the current fields. Anything the source
depends on is therefore read afresh on every residual evaluation.

**So the difference is not zero, and it was measured before this was rewritten.**
meq's `ψ_ax` bordered Newton, border still obtained by differencing, run under
both orderings at `k = 2` on an 8×8 mesh:

| case | condense-first | linearise-first |
|---|---|---|
| ν = 2, A = 1 | 3.058984e−01, 4 it | 3.058984e−01, 5 it |
| ν = 2, A = 100 | 3.036075e+00, 4 it | 3.036075e+00, 5 it |
| ν = 4, A = 1 | 2.834510e−01, 8 it | 2.834510e−01, 9 it |
| ν = 4, A = 10 | 8.643745e−01, 11 it | 8.643745e−01, 13 it |

Identical to every digit printed, self consistent to between 2e−16 and 7e−12,
one or two iterations more. meq's refusal of that ordering has been lifted.

**What that leaves of the fallback**, under either ordering: evaluate the reduced
residual at `M` perturbed values of the auxiliary unknown and difference, then
solve the bordered system by block elimination against one factorisation.
`M` residual evaluations and `M` backsolves per Newton step. It works. It is
fragile in the way §3.1 describes, because differencing a condensed residual is
only as good as the local solves under it.

**So §3 is an improvement on a route that already works, not a prerequisite.**
That is a weaker claim than this document used to make and it is the correct one.
What §3 buys is an *assembled* border — `∂R/∂a` expressed through the factors the
elimination already forms, rather than differenced — which removes the fragility,
removes `M` residual evaluations per step, and lets the caller build one bordered
sparse matrix and solve it once. Under `LineariseThenCondense` that expression is
particularly clean, because the local elimination is a linear solve with the
stored `M` and there is no converged nonlinear local state to differentiate at.

meq has a design for the caller's half of that at
`meq/NORMALISED-LINEARISE-FIRST.md`, which names the entry points it would want
— an application of the local elimination in each direction — and argues they are
smaller than the auxiliary-unknown machinery §3 describes. **Read that before
acting on §3**, which may be more than is needed.

---

## 4. What is not being asked for

* **No BEM.** meq's artificial boundary is a semicircle centred on the symmetry
  axis, and for the axisymmetric operator the exterior Dirichlet-to-Neumann map on
  a sphere is diagonal in a Gegenbauer basis — so meq's `D` is a diagonal matrix
  it forms itself and there are no layer potentials anywhere in it. **Nothing in
  §2 or §3 assumes that**: `D` is opaque to the library, and a caller who does
  need Galerkin BEM matrices on `Γ` can supply them the same way. This is
  flagged only so that "BEM" in the title of the paper is not read as a request
  for integral operators in this tree.
* **No change to the extension technique**, the path families, the lifting or the
  stabilisation. §2 sits beside them and uses them.
* **No change to `PathTraceCoefficient`** — it stays the right thing for a
  *given* `g`, which is what the fitted and prescribed-datum cases want.

---

## 5. Status of this document

A request, not a finding, and not a report against work in progress. Nothing in
this tree has been measured to be wrong; §2 and §3 are capabilities that do not
exist rather than behaviour that misbehaves.

The one measurement quoted, in §3.1, is meq's own and is about meq's workaround.
It is included because it is the argument for §3, and it should be read as "this
is what a caller does without the capability", not as a defect report against
`DarcyHybridization`.

meq has not started FB-1, the first stage that would exercise any of this. The
request is being filed now because the split between what belongs here and what
belongs in meq is clearer before the code exists than after.
