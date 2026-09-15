# Interpolatory HDG (CCSZ-I): what is left

Scratch. **This file was 1774 lines and all seven of its stages are done.**
Chen, Cockburn, Singler & Zhang, *Superconvergent interpolatory HDG methods for
reaction diffusion equations I: an HDGk method*, J. Sci. Comput. **81** (2019)
2188-2212.

What was built, and where its write-up lives -- not here, per the rule that
markdown is a to-do list:

| | |
|---|---|
| `HDGPostprocessBlocks`, the classic postprocessing split into per-element blocks that can be APPLIED rather than re-solved | `postprocess_hdg.hpp` doxygen |
| `HDGInterpolatoryReactionIntegrator` and `HDGQuadratureReactionIntegrator`, the second existing to measure the first against | `reaction_hdg.hpp` doxygen |
| the Jacobian's (1,0) block, which a term evaluated at the postprocessed potential needs | `DarcyHybridization`'s gradient doxygen |
| `BlockNonlinearFormIntegrator::GetBlockRowMask()`, so a term writing one block row keeps the flux mass factored once | `nonlininteg.hpp` doxygen |
| `convdiff`/`pconvdiff` problem 10 with `-rx`, `-tau0`, `-pp`, and its references (five serial, three parallel) | **`convdiff.cpp`'s header comment**, which carries the formulation, the flag meanings and the `k+1/k+1/k+2` ladder |

The headline result: the ladder reproduces on uniform triangulations -- rates
of u / q / u\* of 1.00/1.00/1.00 at k=0, 2.02/2.01/3.00 at k=1, 3.04/3.05/4.00
at k=2, 4.03/4.01/5.01 at k=3 -- including the k=0 row that must NOT
superconverge, and the `kappa/h` arm reproduces the postprocessed error growing
rather than falling. Those numbers are in `convdiff.cpp`'s header and in
`CLAUDE_MEASUREMENTS.md`.

**Two sections of this file are deliberately gone rather than moved.** §5's
staged plan described work that is finished, and §7, "Claims I could not
verify", was a PRE-WORK caveat list ending "Nothing here was compiled or run"
-- true when written, false since stage 0, and actively misleading now. The
items in it that mattered were settled by the stages themselves: the `B11`/`B12`
extraction was checked by stage 0's rank-one and zero-mean tests, the enriched
space's degree was confirmed there too, and the `Mnl` reachability concern was
answered by the domain-integrator route problem 10 actually takes.

**What is left is not implementation. It is what the METHOD does not give us**,
which is §6 below, kept verbatim because none of it has changed.

## 6. What CCSZ-I does not give us, and the open questions

### 6.1 Nonlinear diffusion: no. Not even a diffusion coefficient.

Eq (1) is `∂_t u − Δu + F(u) = f`. The diffusion is the identity Laplacian and
there is no coefficient anywhere in the paper. The nonlinearity is a **reaction**
`F(u)`, a scalar function of a scalar.

The predecessor (ref [16]) handles a general `F(∇u, u)` and proves **optimal**
rates with **no** superconvergence — §1, p. 2189, first paragraph, and §5,
p. 2208. §5's closing sentence is explicit: "We are also considering how to
guarantee that superconvergence property holds for semilinear PDEs with a
general nonlinearity `F(∇u, u)`." **So superconvergence for a flux-dependent or
diffusion-dependent nonlinearity is stated as open by the authors.**

That is the sharpest limitation for this tree, whose nonlinear problems are
*flux laws* — `MixedConductionNLFIntegrator` with a `MixedFluxFunction`
(`fem/nonlininteg_mixed.hpp:23`, `:109`) is `q = q(u, ∇u)`, which is exactly the
class CCSZ-I does not cover.

**AMENDED: "outside the theory" is not "does not work", and a neighbouring
code has run it on flux laws.** `../MaNTA/` — a 1-D nonlinear reaction-diffusion
and *transport* HDG code, integrated as an index-1 DAE by SUNDIALS IDA — has
CCSZ-I behind a `Superconvergent = true` configuration flag, citing this same
paper. Read out of that tree's `docs/superconvergence.rst`, not from memory:

* **It runs on transport systems, and converting cost nothing at the physics
  level**: "No physics case needs changing to run under the flag, in C++,
  Python or JAX — the batched hooks loop over however many points they are
  given."
* **It hit the same new coupling this branch did.** "The only genuinely new
  coupling is that `u*` on a cell depends on that cell's `q` as well as its
  `u`" — the Jacobian's (1,0) block, which is why this branch had to build one.
* **Their measured orders**: `u*` reaches `k+2` and `u_h` keeps `k+1`, for a
  linear constant-`kappa` case and for a nonlinear reaction `u^3 - u`, at
  `k = 1` and `k = 2`.

**And their limits section is the part to quote, because it is narrower than
the flag suggests**: "A general nonlinear flux `sigma(u, q)` is outside the
papers' theory — their conclusion names `F(grad u, u)` as open. **The Jacobian
is verified for such a flux, but no order study asserts `k+2` for one.**"

So the honest position for this section is three-way rather than two-way: the
theory is open, the practice is **fine** — it runs, the Jacobian is right, and
nothing degenerates — and the superconvergence RATE on a genuine flux law is
unmeasured anywhere, here or there. The sentence above ("the class CCSZ-I does
not cover") is true of the theory and was being read as if it were true of the
method; it is not.

**One anomaly of theirs worth carrying, because it contradicts the stated
mechanism.** With their flag OFF, `u*` superconverges at `k = 2` but not at
`k = 1`, and that is true whether or not the source is nonlinear. The papers
attribute the loss to `I_h F(u_h)` evaluating `F` at an `O(h^{k+1})`-accurate
`u_h`, which predicts the nonlinear rows differ from the linear ones. **They do
not.** So whatever caps their `k = 1` rate is not the papers' mechanism, and
their `k = 2` rows say the interpolatory method is not universally losing
superconvergence. Their partial explanation is that nodal interpolation of a
*known* smooth source at the Chebyshev nodes leaves an error very nearly
orthogonal to `P^k`.

**What this does NOT license for a caller with a gated source.** Interpolation
is weakest exactly where the integrand is non-smooth, and a free-boundary
Grad-Shafranov source is `{psi > 0}`-gated with a per-element connectivity
test, so `F` is DISCONTINUOUS in the unknown on any element the plasma boundary
crosses. meq already carries an `extraOrderIn = 4` on its `SourceIntegrator`
"to keep the integration of an exponential in psi from being what limits a
measured rate", i.e. it is already paying to fight integration error on this
term, and interpolation is the opposite move. **That element is the probe to
run before converting anything**, and it is not covered by MaNTA's evidence:
their sources are smooth.

### 6.2 Systems (`vdim > 1`): no theory

`u : Ω → ℝ` throughout. Example 4.2's Schnakenberg system is two equations and
the paper says it "does not satisfy the assumptions of the convergence theory
established here" (p. 2204); Figures 1–3 are pattern plots with no rates.

This tree supports `vdim > 1` throughout — the postprocessing
(`postprocess_hdg.hpp:37-45`), `Reconstruct()`
(`darcyform.hpp:556-564`) and the flux layouts are all `neq`-general. So the
*implementation* extends naturally; the **theory does not.** Two concrete
consequences for the design:

* `diag(𝓕'(γ))` becomes **nodewise `neq x neq` blocks**, not a diagonal, once
  `F` couples the equations (Schnakenberg's `C_a²C_i` does). The interface in
  §3.2 already reflects that: `EvalJacobian()` returns a `DenseMatrix`.
* the equation-outermost layout of `gamma` (§3.1) has to interleave against
  `A9`'s per-equation blocking, and getting that wrong is the exact shape of the
  `HyperbolicFormIntegrator` defect this branch fixed. A `neq = 3` case where
  the equations genuinely couple is the pin, and a `neq = 1` case cannot
  substitute — that defect passed every `neq = 1` assertion.

### 6.3 Steady problems: no theorem

Every estimate is `L^∞(0,T; L^2)`. §1.4. Stage 2(a) is therefore a measured
ladder and not a check against a proof, and the write-up must say so.

### 6.4 Simplices and `P^k`: the theory's mesh

§2.1 says "a collection of disjoint simplices"; §3.1 says "for each simplex
`K ∈ T_h`"; every space is `P^k`. On quadrilaterals and hexahedra MFEM's L2 is
`Q^k`, `Z_h` is `Q^{k+1}`, and Lemma 3.3's counting changes. The implementation
is dimension- and geometry-generic; the theorem is not. Run the ladder on
triangles for the comparison against Table 1 and on quads separately.

### 6.5 The node set of `Z_h` is a free choice with an unmeasured effect

CCSZ say "the finite element nodes for the postprocessing space `Z_h`" and
nothing about which nodes. MFEM's `L2_FECollection` defaults to
`BasisType::GaussLegendre` (`fem/fe_coll.hpp:386-389`), an **open** set;
`GaussLobatto` is closed and puts dofs on vertices. `I_h` differs between them
and so, in principle, does the answer at fixed `h`.

meq's request gives a concrete reason to care:
meq's integrand is `F/r` and a Lobatto triangle puts a dof at `r = 0` on any
domain reaching the symmetry axis — "Quadrature never meets this; nodal
interpolation does." So the node set is a caller choice, which is why
`HDGPostprocessBlocks` takes `fes_s` as an argument (§3.1) rather than building
it. **Unmeasured**: whether the two node sets give the same rates and how far
apart the answers are at fixed `h` is a sweep nobody has run.

### 6.6 `τ` — a premise to fix before running the ladder

§2.5. Paper I needs `τ` elementwise constant and **O(1)**; this tree's
`HDGDiffusionIntegrator` default is **O(1/h)**. The `HDGStabilization` hook
supplies the paper's `τ`. This is a correction to the meq request doc's premise
and it changes what stage 2 has to run, not what the design has to be.

### 6.7 What "part II" would be for

**The paper does not describe a part II.** Its title is "I: An HDG_k Method" and
§5, p. 2208, states only the intention: "We are interested in extending our
results to methods closely related to the HHO methods; see [8]" — ref [8] being
Cockburn, Di Pietro & Ern, *Bridging the HHO and HDG methods*, ESAIM M2AN **50**
(2016) 635–650.

meq's request identifies the sequel as Chen,
Cockburn, Singler & Zhang, Commun. Appl. Math. Comput. **4** (2022) 477–499, and
says it needs **different spaces and `τ = 1/h`**. **I have not read it.**

On that description the sequel is roadmap **§9**'s territory — "Superconvergence
at `k = 0` — the HHO-inspired methods", `doc/HDG-ROADMAP.md:244-254`. That entry
already records what is present and what is missing for it: `τ ~ 1/h` is the
built-in default (which §2.5 above confirms), unequal flux/potential/trace
orders are unconstrained, and the missing third ingredient is "a stabilisation
acting on the **L2 projection of the potential onto the trace space** rather
than on the potential itself", which `HDGStabilization` cannot express — it
rescales `τ` but cannot change what `τ` multiplies. **So a sequel that needs
`τ = 1/h` and different spaces needs §9's new face integrator, and paper I does
not.** Doing paper I first is the right order for that reason as well as for the
smaller diff.

The other thing a sequel would be needed for is `k = 0`, where paper I offers
nothing: `min{k,1} = 0` and Table 1's `k = 0` `u*` column is 0.97.

### 6.8 Open questions this design does not settle

1. **Whether the collocation costs an order anywhere.** Stage 2 answers it.
   Until then, the reply doc's honest position stands: "interpolatory assembly
   is the structurally correct answer to a problem we have currently solved with
   a cache, and we do not know what it costs in accuracy."
2. **Whether the four cache mechanisms actually die.** They do not die because
   an interpolatory integrator exists; they die when nothing installs a
   non-interpolatory nonlinear mass integrator. Convdiff's `-nl` route installs
   `VectorMassIntegrator` on `GetFluxMassNonlinearForm()`, which is a *flux
   mass*, not a reaction, and CCSZ-I says nothing about it. **So the deletion of
   items 1–4 is not a consequence of this work as scoped.** It would be a
   consequence of interpolatory *flux laws*, which is §6.1's open problem.
   Do not sell this design as deleting the cache.
3. **A moving diffusion coefficient still forces re-assembly** of `B11`/`B12`,
   because `iK` is a right-hand-side operator of the local problem (§3.1). The
   reply doc's table row "nothing for a state or parameter change; only geometry
   touches the shape matrix" is right about the *reaction*'s parameters and
   wrong about a diffusion parameter. gffp's case is the second kind.
4. **Whether `Bg_data` should be a fourth `LocalOpType`'s business** rather than
   a flag. §3.4 / stage 5.
5. **Cut elements.** meq's source is confined to a region whose edge is a level
   set of the solution cutting through elements, where `F` has a kink or a jump.
   An interpolant is the wrong object there; the papers assume a smooth `F`
   under a local Lipschitz condition and say nothing about it. meq's plan is to
   interpolate on uncut elements and keep quadrature where the edge cuts, which
   would need the integrator to admit a per-element opt-out. **Not designed for
   here** — flagged because it is a caller requirement that would change
   `HDGInterpolatoryReactionIntegrator`'s interface if it arrived late.

---
