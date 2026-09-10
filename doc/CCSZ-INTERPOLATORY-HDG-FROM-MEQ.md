# Interpolatory HDG needs a (1,0) gradient block and the postprocessing's own matrices

**A request from MEQ, written 2026-09-09. Nothing in this tree has been changed
for it, and nothing here is a defect claim** — both pieces of machinery it
builds on are correct for what they do today, and the request is that they be
reachable for one more thing.

It asks for what Chen, Cockburn, Singler & Zhang, *Superconvergent Interpolatory
HDG Methods for Reaction Diffusion Equations I: An HDG_k Method*, J. Sci. Comput.
**81** (2019) 2188–2212, https://doi.org/10.1007/s10915-019-01081-3, needs in
order to be expressible on `fem/darcy/`. Its predecessor is Cockburn, Singler &
Zhang, J. Sci. Comput. **79** (2019) 1777–1800,
https://doi.org/10.1007/s10915-019-00911-8, and the HHO-flavoured sequel is
Chen, Cockburn, Singler & Zhang, Commun. Appl. Math. Comput. **4** (2022)
477–499, https://doi.org/10.1007/s42967-021-00128-3. Only the first is being
asked about: its `HDG_k` is `V_h`, `W_h`, `M_h` all at degree `k` with `τ`
elementwise constant and `O(1)`, which is this branch's method and MEQ's, where
the sequel needs different spaces and `τ = 1/h`.

**Two asks, and they are tiered.** §2 is the one MEQ cannot work around. §1 is
the one MEQ can do by hand and would rather not.

**Every line number below is against `meq-integration` at `2613ee1b3b`**, which
is the merge MEQ builds from, and `fem/darcy/darcyhybridization.cpp` on
`gf-hdg-linearise-first` is over a thousand lines away from it — so read the
function names, which are stable, and treat the numbers as a hint.

## What the method is, in four lines

The semilinear term `(F(u_h), v_h)` is replaced by `(I_h F(u*_h), v_h)`, where
`u*` is the classic local postprocessing and `I_h` is elementwise Lagrange
interpolation onto the nodes of `Z_h = P^{k+1}`. Because `u*` is *linear* in
`(q_h, u_h)` — the local problem is the Neumann stiffness driven by the flux and
closed by the element average, with `F` nowhere in it — the `Z_h` coefficients
are `γ = B11 α + B12 β` with `B11`, `B12` block diagonal and assembled **once**.
The load becomes `A9 𝓕(γ)` with `A9 = [(χ_j, φ_i)]` fixed and `𝓕` a pointwise
evaluation, and the Jacobian is `A9 diag(𝓕'(γ)) B11` against the flux and
`A9 diag(𝓕'(γ)) B12` against the potential (paper §2.2, the equations after
(9)). Interpolating into `W_h` instead is the predecessor, which proves optimal
rates and loses the postprocessing's superconvergence; evaluating at `u*` is
what restores it, and that is the whole content of Remark 2.1.

**MEQ's interest is not the parabolic saving.** MEQ solves a steady problem, so
"assembled once before the time integration" reads as "no quadrature of `F` in
any Newton step". Paper I has **no steady theorem** — every estimate is
`L^∞(0,T; L^2)` for `∂_t u − Δu + F(u) = f` — so MEQ's acceptance is a measured
rate ladder and not a check against a proof. That is MEQ's problem, and it is
recorded here so that nobody reads the asks below as resting on a theorem that
covers the case.

## 1. `HDGPotentialPostprocessor` exposing `B11` and `B12`

`fem/darcy/postprocess_hdg.cpp` already forms both and discards them. Per
element it builds `A` from `AddMult_a_AAt(w, dshape_s, A)` (line 164), replaces
row `i_c` with the mass row `(χ_j, 1)_K` (lines 204–205), factors, and puts
`(u_h, 1)_K` into that entry of the right-hand side (line 215). In that
factorisation

```
B11^e = -Ai * A2~        A2~ = [ (phi_j, grad chi_i) ] with row i_c zeroed
B12^e =  (Ai * e_{i_c}) * mass_p^T                     -- rank one
```

so the extraction is `Ai` applied to two right-hand-side *operators* rather than
to one vector. The row-replacement closure is not the paper's bordered eq (7),
and it does not need to be — the two are equivalent in exact arithmetic, since
the rows of the `P^{k+1}` Neumann stiffness sum to zero for a partition-of-unity
nodal basis and `Σ_i (q, ∇χ_i)_K = 0` makes the right-hand side consistent — but
the blocks that come out are this system's, which is worth saying because
MEQ's sibling MaNTA implements the bordered form and the two would be compared.

What is asked for:

```cpp
/// Factor every element's local matrix once; Compute() then applies.
void Assemble();

/** @brief The per-element blocks of gamma = B11 alpha + B12 beta.

    B11 is nd_s x (flux dofs on the element), B12 is nd_s x nd_p and has
    rank one: u* sees u_h only through its element average. */
void GetLocalBlocks( int el, DenseMatrix &B11, DenseMatrix &B12 ) const;
```

**Why not caller side.** The algebra is thirty lines and MEQ could write them.
What MEQ would have to re-own is everything the class already does around them:
reading the flux layout out of its space rather than assuming it, the `ik`/`iK`
diffusion-inverse hook — which is how MEQ's axisymmetric `r` reaches the local
problem at all — the `neq` blocking, and the enriched-space construction. A
duplicate of those in a caller is the kind of thing that agrees for a year and
then does not.

**One requirement rather than a complaint.** If this machinery is to be reached
from inside `DarcyHybridization::MultNL()`'s threaded element loop, which is
where a residual that needs `u*` would call it from, it needs the
caller-allocated `GetElementTransformation( i, IsoparametricTransformation * )`
overload rather than the shared one at `postprocess_hdg.cpp:126`. The shared one
is correct where the class is used today — a serial post-processing pass — and
MEQ raises it only because the new path is not that. MEQ has hit this six times
in its own code and it never fails loudly.

## 2. A solution-dependent (1,0) gradient block — the one MEQ cannot work around

`A9 diag(𝓕'(γ)) B11` is `d(potential residual)/d(flux)`. It is
`d_dofs × a_dofs`, **exactly `Bf_data`'s shape and orientation**, and it has to
be added to `B` wherever `B` acts as the (1,0) block of the Jacobian.

`darcyhybridization.cpp:3208` sets `grad_arr(1,0) = NULL`, and the comment above
it says why:

> Block (1,0) stays NULL: the divergence form is linear, so B is already exact
> in `Bf_data`.

That is true of every problem this branch has met. It is not true of paper I,
because `u*` is built from `q_h` as well as `u_h`, so the reaction term reaches
the flux. **This is not a Grad–Shafranov feature.** It is the shape of any
method whose potential equation depends on the flux: the predecessor paper's
general class is `F(∇u, u)`, the sequel's postprocessing is
`p^{k+1}(u_h, û_h)`, and the mirror case — a flux law `q = D(p) u`, so the flux
equation depends on the potential — is already supported here, as `Bnl_data`,
whose own header comment states it in as many words.

The three sites, all private:

| site | role |
|---|---|
| `ComputeElementH()`, `:1446` | `S = D + B·AiBt`, and `BAiCt = B·AiCt` in the trace loop |
| `MultInv()`, `:3140` | the eliminated Jacobian applied, under the existing `with_bnl` flag; reached from `NPCReduce()`, `NPCRecover()` and `MultNL(GradMult)` |
| `LocalNLOperator::B`, `:4763`, used at `:5203` | `grad.SetBlock(1, 0, &B)` |

**The elimination needs no new algebra.** It is a 2×2 block Schur complement
that already keeps the two off-diagonal blocks distinct — `AiBt` carries the
(0,1) block and `B` the (1,0) one — and the (0,1) side is already
solution-dependent. Because `MultInv()` is the single funnel for applying the
eliminated Jacobian, one change there covers `GradientMode::MatrixFree` as well
as `Assembled`; the comment at `:2181` records that getting the (0,1) case wrong
at that exact site *"made the matrix-free gradient disagree with the assembled
one"*, which is both the precedent and the regression to copy.

What is asked for is symmetry with what exists:

```cpp
// In ConstructGrad() and LocalNLOperator::AddGradBlock():
//   grad_arr( 1, 0 ) read rather than forced NULL.
// Stored per element in Bf_data's own d_dofs x a_dofs orientation.
// Added to B at the three sites above, gated by the existing with_bnl.
```

**And a question rather than an ask.** MEQ installs its source on `m_nlfi_p`,
which selects `LocalOpType::PotNL` (`:2463`) — the branch where `InvertA()`
factors the flux block once and `ComputeElementH()` then skips re-factoring it
(`:1440`). Any `m_nlfi` at all falls through to `FullNL` (`:2485`) and gives
that up: one dense LU of `A` per element per Newton step. MEQ has **not
measured** what that costs and it may be nothing. What MEQ would rather have is
a slot contributing the potential residual and the (1,1) and (1,0) gradient
blocks while leaving the flux block linear. Whether that is worth a fourth
`LocalOpType` is entirely this tree's call, and MEQ is not asking for it — only
recording that the block integrator is a wider hammer than the term needs.

## What MEQ will do itself

All of it except the two above, and none of it belongs in a general library.

* **`I_h`.** Nothing new is needed: `L2_TriangleElement` is a
  `NodalFiniteElement`, so the interpolation is evaluation at `GetNodes()`. MEQ
  will build `Z_h` with an **open** basis rather than cloning its own
  `BasisType::GaussLobatto` spaces, because MEQ's integrand is `F/r` and a
  Lobatto triangle puts dofs on vertices — one of which sits at `r = 0` on any
  domain reaching the symmetry axis. Quadrature never meets this; nodal
  interpolation does. `Compute()` already accepts a caller-supplied space, which
  is all MEQ needs for that.
* **The axisymmetric weight and the explicit `(r, z)` dependence.** Outside the
  paper's `F(u)`, and notational: MEQ interpolates the whole integrand
  `F(r, z, ψ*)/r`, which keeps `A9 = [(χ_j, φ_i)]` a pure mass coupling.
  Keeping `1/r` under the quadrature instead would make `A9 = [(χ_j/r, φ_i)]`,
  which diverges on an element touching the axis.
* **The cut elements.** MEQ's source can be confined to the plasma, whose edge
  is a level set of the solution cutting through elements. `F` is a genuine jump
  there only in one limit of MEQ's profile family; otherwise it is a kink. An
  interpolant is the wrong object in either case, so MEQ will interpolate on
  uncut elements and keep the quadrature where the edge cuts. The papers assume
  a smooth `F` under a local Lipschitz condition and say nothing about this, and
  it is MEQ's geometry rather than anybody's discretisation.
* **The residual, today, with no change here.** `m_nlfi` is a
  `BlockNonlinearFormIntegrator` receiving `(u_l, p_l)` and writing to both
  blocks (`AddMultBlock`, `:4846`), and under NPC the flux is Newton state — so
  MEQ can already compute `u*` locally, interpolate, and write `A9 𝓕(γ)` into
  the potential residual. The (1,1) Jacobian block `A9 diag(𝓕'(γ)) B12` goes
  into `grad_arr(1,1)`, which is read. Only (1,0) has nowhere to go.
* **Therefore: the rate evidence first, and MEQ will come back with it.** A
  converged answer does not depend on which Jacobian reached it, so the
  `k+1` / `k+1` / `k+2` ladder can be established with (1,0) missing, and the
  cost of its absence measured as an iteration count rather than predicted.
  MEQ's own rule is that a version that has been used is a better request than
  one that has not, and this file is filed early only because §2 is a hole in an
  API rather than something a prototype could discover.

## What is not being asked for

* **No interpolation operator, no `Z_h`, no `A9`, no source machinery.** The
  interpolatory *method* is the caller's. What is asked for is the two places a
  caller cannot reach: the postprocessing's blocks and the (1,0) gradient block.
* **No change to `DarcyForm::Reconstruct()`.** MEQ builds `ψ*` from it today and
  will move to `HDGPotentialPostprocessor` for this, because the richer
  reconstruction lifts the nonlinear potential integrators as a Jacobian frozen
  at the computed potential (`darcyform.cpp:1370-1404`) and its output is
  therefore not a linear function of the unknowns — feeding it back into the
  source would make each element's reconstruction an implicit local fixed point.
  The two answer different questions, `postprocess_hdg.hpp` says so, and the
  smaller one is the one this method wants. Nothing about the richer one needs
  to change.
* **Nothing about `τ`.** Paper I's `τ` is elementwise constant and `O(1)`, which
  is what MEQ runs and what `HDGDiffusionIntegrator` supplies.
* **No claim that anything here is broken.** The (0,1) facility, the
  postprocessing class and the `with_bnl` plumbing are all recent and all
  correct for the problems they were written for. This is a request to widen
  two of them by one block each.
