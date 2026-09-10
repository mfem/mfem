# Interpolatory HDG_k and its superconvergent postprocessing — a design, not a build

**Status: nothing is implemented. This file is a to-do.** It specifies what
Chen, Cockburn, Singler & Zhang, *Superconvergent Interpolatory HDG Methods for
Reaction Diffusion Equations I: An HDG_k Method*, J. Sci. Comput. **81** (2019)
2188–2212 (**CCSZ-I** below; PDF at `/home/ian/projects/meq/refs/SuperconvergentHDG-I.pdf`)
would need in order to be expressible on `fem/darcy/`, what interface the
interpolatory integrator should present, and what each stage's falsifying
measurement is.

It is the design half of a pair. The *request* half is
`doc/CCSZ-INTERPOLATORY-HDG-FROM-MEQ.md`, written by meq the same day; it asks
for two API widenings and independently identified the same missing (1,0)
gradient block. Where this file and that one differ, the differences are called
out (§2.5, §6.6) — one of meq's premises about `τ` does not hold in this tree.

**Line numbers are against the WORKING TREE of `gf-hdg-linearise-first` at
`bca170a695` with the uncommitted offload changes in place.** Every one was
read, not remembered. Function and class names are the stable part; if a number
is off by a few dozen lines, grep the name.

**No source file, test or makefile was touched, and nothing was compiled or
run, in producing this.** Every number quoted from this tree is a line number
or a count of grep hits; every number quoted as a measurement is CCSZ-I's own
and is attributed. There are no new measurements here — §5 is a list of
measurements to *take*.

---

## 1. What CCSZ-I's method actually is

### 1.1 The problem, the spaces, the method

The PDE is eq (1), p. 2189:

    ∂_t u − Δu + F(u) = f   in Ω × (0,T),   u = 0 on ∂Ω × (0,T),   u(·,0) = u_0

on a Lipschitz polyhedral Ω ⊂ ℝ^d, d = 2,3. **The diffusion is the identity
Laplacian.** There is no diffusion coefficient anywhere in the paper. The only
nonlinearity is the reaction `F(u)`, a scalar function of a scalar `u`.

The spaces (§2.1, p. 2190), on a mesh `T_h` of **disjoint simplices**:

| space | definition | role |
|---|---|---|
| `V_h` | `[P^k(K)]^d`, discontinuous | the flux `q = −∇u` |
| `W_h` | `P^k(K)`, discontinuous | the scalar `u` |
| `Z_h` | `P^{k+1}(K)`, discontinuous | **postprocessing only** |
| `M_h` | `P^k(F)`, `μ|_{F_h^∂} = 0` | the trace `û` |

All three of `V_h`, `W_h`, `M_h` are at degree `k` — that is what "HDG_k"
names, and it is this branch's method. `τ` is nonnegative, **elementwise
constant and O(1)** (§1, p. 2189; Lemma 3.2 additionally needs
`τ_max = max τ|_{∂K} > 0`).

The semidiscrete method is eqs (2a)–(2d), p. 2190–2191, with the numerical flux
eq (3):

    (q_h, r_h) − (u_h, ∇·r_h) + ⟨û_h, r_h·n⟩              = 0                (2a)
    (∂_t u_h, v_h) − (q_h, ∇v_h) + ⟨q̂_h·n, v_h⟩
                                 + (I_h F(u*_h), v_h)     = (f, v_h)         (2b)
    ⟨q̂_h·n, v̂_h⟩_{∂T_h \ F_h^∂}                            = 0                (2c)
    u_h(0) = Π u_0                                                            (2d)
    q̂_h·n = q_h·n + τ(u_h − û_h)                                              (3)

**Every term except `(I_h F(u*_h), v_h)` is the standard HDG_k method and is
already what `DarcyForm`/`DarcyHybridization` assembles.** The whole content of
the paper is that one term.

### 1.2 What is interpolated, and at which nodes

`I_h` is **elementwise Lagrange interpolation onto the finite element nodes of
`Z_h = P^{k+1}`** (§2.1, p. 2190, final paragraph): "Let `I_h` be the
elementwise interpolation operator with respect to the finite element nodes for
the postprocessing space `Z_h`. Therefore, for any function `g` that is
continuous on each element we have `I_h g ∈ Z_h`."

The argument is the **postprocessed** scalar `u*_h ∈ Z_h`, not `u_h`. That is
Remark 2.1, p. 2191, and it is the entire difference from the predecessor
(Cockburn, Singler & Zhang, J. Sci. Comput. **79** (2019) 1777–1800, ref [16]),
which used `I_h F(u_h)` with `I_h` mapping into `W_h`, proved optimal rates, and
**observed no superconvergence after postprocessing**. Swapping `u_h` for `u*_h`
restores it. Nothing else changes.

`u*_h` is the classic HDG local postprocessing, eqs (4a)–(4b), p. 2191, solved
independently on each `K`:

    (∇u*_h, ∇z_h)_K = −(q_h, ∇z_h)_K   for all z_h ∈ [P^{k+1}(K)]^⊥          (4a)
    (u*_h, w_h)_K   =  (u_h, w_h)_K    for all w_h ∈ P^0(K)                  (4b)

where `[P^{k+1}(K)]^⊥ = {z ∈ P^{k+1}(K) : (z, w_0)_K = 0 ∀ w_0 ∈ P^0(K)}`.
Equivalently, and this is the form the implementation uses, eqs (7a)–(7b),
p. 2192, with a Lagrange multiplier `η^n_h ∈ P^0(K)`:

    (∇u*_h, ∇z_h)_K + (η_h, z_h)_K = −(q_h, ∇z_h)_K   ∀ z_h ∈ P^{k+1}(K)     (7a)
    (u*_h, w_h)_K                  =  (u_h, w_h)_K    ∀ w_h ∈ P^0(K)         (7b)

Remark 2.2, p. 2192: `w_h ∈ P^ℓ(K)` with `ℓ = 0, 1, …, k−1` also works, in the
analysis and in the experiments. The paper uses `ℓ = 0`.

### 1.3 Which integrals become fixed matrices, and what the shape matrices are

**`u*_h` is a LINEAR function of `(q_h, u_h)`.** `F` does not appear in (4)/(7):
the local problem is the `P^{k+1}` Neumann stiffness driven by the flux and
closed by the element average of `u_h`. That is the structural fact the whole
method rests on.

With the bases of eq (8), p. 2192 — `V_h = span{φ_j}_1^{N1}` (bold, vector),
`W_h = span{φ_j}_1^{N2}`, `Z_h = span{χ_j}_1^{N3}`, `M_h = span{ψ_j}_1^{N4}`,
and coefficient vectors `α, β, γ, ζ` for `q_h, u_h, u*_h, û_h` — eq (8) defines

| matrix | definition | shape | this tree's block |
|---|---|---|---|
| `A1` | `[(∇χ_j, ∇χ_i)]` | `N3 × N3` | none (Z_h stiffness) |
| `A2` | `[(φ_j, ∇χ_i)]` bold φ | `N3 × N1` | none |
| `A3` | `[(φ_j, φ_i)]` bold | `N1 × N1` | `Af_data` (the flux mass `A`) |
| `A4` | `[(φ_j, ∇·φ_i)]` | `N1 × N2` | `Bf_data`, transposed, `±` per `bsym` |
| `A5` | `[⟨ψ_j, φ_i·n⟩]` | `N1 × N4` | `Ct_data` |
| `A6` | `[(τφ_j, φ_i)_{∂T_h}]` | `N2 × N2` | part of `Df_data` (`D`) |
| `A7` | `[⟨τψ_j, φ_i⟩]` | `N2 × N4` | `E_data` / `G_data` |
| `A8` | `[⟨τψ_j, ψ_i⟩]` | `N4 × N4` | `H_data` |
| `A9` | `[(χ_j, φ_i)]` | `N2 × N3` | **new** — the `Z_h → W_h` mass coupling |
| `M`  | `[(φ_j, φ_i)]` | `N2 × N2` | the potential mass |
| `b1` | `[(χ_j, 1)]` | `1 × N3` row | the `Z_h` mean row |
| `b2` | `[(φ_j, 1)]` | `1 × N2` row | the `W_h` mean row |
| `b3` | `[(f^n, φ_i)]` | `N2` | `DarcyForm::GetPotentialRHS()` |

Every one is block diagonal or block sparse with element-local blocks; the
paper says so explicitly after eq (8).

Element `ℓ` of eq (7) reads (p. 2192, bottom):

    ⎡ A1^ℓ  (b1^ℓ)^T ⎤ ⎡ γ_ℓ ⎤   ⎡ −A2^ℓ   0    ⎤ ⎡ α_ℓ ⎤
    ⎣ b1^ℓ     0     ⎦ ⎣ η_ℓ ⎦ = ⎣   0    b2^ℓ ⎦ ⎣ β_ℓ ⎦

so, inverting (p. 2193, top),

    γ^n = B11 α^n + B12 β^n                                                   (§2.2)

with `B11` (`N3 × N1`) and `B12` (`N3 × N2`) block diagonal and **assembled
once**. `B12` has **rank one per element**: `u*` sees `u_h` only through its
element average.

Then, and this is the payoff:

    [(I_h F(u*_h), φ_i)] = A9 𝓕(γ) = A9 𝓕(B11 α + B12 β)

with `𝓕(γ) = [F(γ_1), F(γ_2), …, F(γ_{N3})]^T` — a **pointwise evaluation at
the nodal values**, no quadrature. And the Jacobian (p. 2193, the two displays
after eq (9)):

    A10 = A9 diag(𝓕'(γ)) B11        the (potential row, FLUX column) block
    A11 = A9 diag(𝓕'(γ)) B12        the (potential row, potential column) block

Remark 2.3, p. 2194: "we only need to assemble the HDG matrices and the HDG
postprocessing matrices `B11` and `B12` once before the time integration."

### 1.4 What is a theorem and what is numerical observation

**Theorems.** Under the standing assumptions of §3 (Ω Lipschitz polyhedral;
the solution unique and smooth on `[0,T]`; the semidiscrete solution existing
and unique; the mesh uniformly shape regular with `h ≤ 1`; and the dual-problem
regularity eq (10) / eq (35), satisfied if Ω is convex):

* **Theorem 3.14** (p. 2202) — the error estimates, under **Assumption 3.4**
  (global Lipschitz: `|F(u) − F(v)| ≤ L|u − v|` for all real `u, v`).
* **Corollary 3.15** (p. 2202) — if in addition `u`, `q`, `F(u)` are
  sufficiently smooth on `[0,T]`, then for all `0 ≤ t ≤ T`

      ‖q − q_h‖_{T_h}  ≤ C h^{k+1}
      ‖u − u_h‖_{T_h}  ≤ C h^{k+1}
      ‖u − u*_h‖_{T_h} ≤ C h^{k+1+min{k,1}}

  so `u*` is `h^{k+2}` for `k ≥ 1` and only `h` for `k = 0`. The text on
  p. 2195 states it plainly: "As in the linear case [2], superconvergence is
  only obtained for `k ≥ 1`."
* **Theorem 3.19** (p. 2204) — the same conclusions under **Assumption 3.16**
  (local Lipschitz on `[−M, M]`, `M` from eq (28)), for `k ≥ 1`, a
  **quasi-uniform** mesh, and `h` small enough.
* Supporting: Lemma 3.2 (the HDG_k projection eq (11) is uniquely solvable and
  bounded, eqs 12a–12b), Lemma 3.3 (standard interpolation and projection
  bounds, eqs 14a–14c — note 14a needs `w ∈ C(K̄) ∩ H^{k+2}(K)`).
* Remark 3.1 (p. 2195): the linear-case result of ref [2] carries a `√log κ`
  factor in `L^∞(L^2)`; CCSZ use Wheeler's duality argument (ref [39]) instead
  and avoid the factor, **at the cost of requiring higher regularity of the
  solution than the linear theory needs.**

**All of the above is `L^∞(0,T; L^2)` for a PARABOLIC problem. There is no
steady theorem and no fully discrete theorem.** Eqs (5a)–(5d), p. 2191–2192,
are the backward-Euler scheme, and §2.2 opens by saying the implementation is
described "using the Newton method to solve the nonlinear system at each time
step. The extension to other time-marching methods is straightforward" — no
estimate accompanies it.

**Numerical observation, §4.**

* **Example 4.1**, p. 2204 — Allen–Cahn / Chaffee–Infante. Ω = (0,1)²,
  `F(u) = u³ − u`, exact `u = sin(t) sin(πx) sin(πy)`, `f` chosen to match,
  `T = 1`. `k = 0` with backward Euler and `Δt = h`; `k = 1` with
  Crank–Nicolson and `Δt = h²`. Table 1 reports errors at `T = 1` over
  `h/√2 = 2^{-1} … 2^{-5}`. Observed rates:

  | | `‖q−q_h‖` | `‖u−u_h‖` | `‖u−u*_h‖` |
  |---|---|---|---|
  | `k = 0` | 0.87, 0.99, 1.00, 1.00 | 0.82, 0.88, 0.94, 0.97 | 0.84, 0.86, 0.94, 0.97 |
  | `k = 1` | 1.90, 1.98, 2.00, 2.00 | 1.82, 1.94, 1.98, 2.00 | **2.95, 3.02, 3.02, 3.01** |

  "The observed convergence rates match the theory." Note the rates are
  **still climbing** on the coarse end in every column — a point §5.3 returns to.
* **Example 4.2**, p. 2204–2207 — the Schnakenberg model, a **two-equation**
  reaction-diffusion system with **zero-flux Neumann** boundary conditions,
  `κ = 100, a = 0.1305, b = 0.7695, D1 = 0.05, D2 = 1`, `k = 1`,
  Crank–Nicolson, `Δt = 0.001`, on a square (2048 elements), a disc (7168) and
  a narrow rectangle (2048). The paper states it "does not satisfy the
  assumptions of the convergence theory established here". **Figures 1–3 only:
  no rates, no errors, no theorem.** It is a demonstration of applicability.

---

## 2. The map onto this tree

### 2.1 `DarcyForm` — where the term is registered

`fem/darcy/darcyform.hpp:110-115` holds the six form slots:

    std::unique_ptr<BilinearForm>       M_u;    // flux mass          A3
    std::unique_ptr<BilinearForm>       M_p;    // potential mass     M, A6
    std::unique_ptr<NonlinearForm>      Mnl_u;
    std::unique_ptr<NonlinearForm>      Mnl_p;
    std::unique_ptr<MixedBilinearForm>  B;      // flux divergence    A4
    std::unique_ptr<BlockNonlinearForm> Mnl;

with accessors at `darcyform.hpp:263-301`. `Mnl_p` is a `NonlinearForm` on
`fes_p` alone; `Mnl` is a `BlockNonlinearForm` over `(fes_u, fes_p)`.

**The CCSZ-I term does not fit `Mnl_p`.** `A9 𝓕(B11 α + B12 β)` reads the flux
coefficients `α`. `Mnl_p`'s integrators are called as
`AssembleElementVector(*fe_p, *Tr, p_l, Dp)`
(`darcyhybridization.cpp:8525`) — the flux is not an argument and cannot be.
So the term belongs on `Mnl`, the block slot.

`DarcyForm::EnableHybridization()` (`darcyform.cpp:342`) hands the forms'
integrators to the hybridization: `SetFluxMassNonlinearIntegrator()` at
`darcyform.cpp:497`, `SetPotMassNonlinearIntegrator()` at `:514`, and the block
form at `:518` onwards. The corresponding setters are declared at
`darcyhybridization.hpp:2617-2624` and the slots at `darcyhybridization.hpp:321-326`.

Two contracts to respect, both documented traps in `CLAUDE.md`:
`EnableHybridization()` reads the forms' face integrator lists **at call time**,
so the integrator must be installed before it; and `DarcyForm::Update()`
(`darcyform.cpp:2360`) calls `hybridization->Reset()` at `:2393`, which drops
`res_cache` at `darcyhybridization.cpp:8140` — so a mesh change is already a
signal any new cache can hang off.

### 2.2 `DarcyHybridization` — the local operator, and the hole

The NPC entry points are declared at `darcyhybridization.hpp:2561-2586`, with
the group doc at `:2357-2552`. One Newton step is

    NPCResidual(b, x, x_tr, r, r_tr)      F(q, u, λ)
    S = NPCGradient(x, x_tr)              factor J; S = H − C' M^-1 [C; E]
    NPCReduce(r, r_tr, b_tr)
    solve S dtr = b_tr
    NPCRecover(r, dtr, dx)

Under NPC the flux and the potential are **Newton state**, which is exactly what
CCSZ-I needs: `γ = B11 α + B12 β` is a function of the current iterate, and the
residual evaluation has both blocks in hand.

`LocalNLOperator` (`darcyhybridization.hpp:622-713`) is the per-element
operator. Its members `grad_A`, `grad_D`, `grad_Aup`, `grad_Bt`
(`:661-664`) are the (0,0), (1,1), (0,1) blocks and the dense sum of (0,1) with
`±B^T`. **There is no (1,0) member.** The block integrator is called at:

| routine | file:line | what it reads / writes |
|---|---|---|
| `AddMultBlock` | `darcyhybridization.cpp:8395` | reads `u_l`, `p_l`; writes `bu` **and** `bp` (`:8440-8442`) |
| `AddGradBlock` | `darcyhybridization.cpp:8594` | sets `grad_arr(0,0)`, `(0,1)`, `(1,1)` at `:8615-8617`; `grad_arr = NULL` at `:8614` |
| `ConstructGrad` | `darcyhybridization.cpp:6415-6431` | `grad_arr(1,0) = NULL` **explicitly**, at `:6428` |

and `ConstructGrad()` carries the reason in a comment at `:6421-6422`:

> Block (1,0) stays NULL: the divergence form is linear, so B is already exact
> in Bf_data.

That is true of every problem this branch has met and false for CCSZ-I. So:

> **The residual is already expressible today with no library change. The
> gradient is not.** `A11 = A9 diag(𝓕') B12` lands in `grad_arr(1,1)`, which is
> read. `A10 = A9 diag(𝓕') B11` lands in `grad_arr(1,0)`, which is forced NULL.

The mirror case is already supported: a flux law `q = D(p) u` makes the (0,1)
block solution dependent, and `Bnl_data` (`darcyhybridization.hpp:410`, doc at
`:395-409`) holds it, written from `grad_Aup` at `darcyhybridization.cpp:6450`.
So the facility exists in one direction only.

The sites where `B` acts as the **(1,0)** block — every one of which a nonlinear
addend would have to reach:

| routine | file:line | expression |
|---|---|---|
| `ComputeElementH` | `:3281-3282` | the view `B(&Bf_data[Bf_offsets[el]], nd, na)` |
| | `:3318` | `AddMult(B, AiBt, D)` — the Schur complement |
| | `:3350` | `AddMult(B, AiBt, S_el)` — the `FluxNL` Schur |
| | `:3383` | `Mult(B, AiCt, BAiCt)` — the trace loop |
| `FactorElementsBatched` | `:3052`, `:3069` | `AddMult(B, AiBt_all, S_v)` |
| `ComputeElementsHBatched` | `:3153-3155` | the chunked `DenseTensor` view |
| `LinearResidualBatched` | `:3910`, `:3952` | `Mult(B, u_all, rp_all)` |
| `MultInv` | `:6136-6137`, `:6146` | `B.Mult(u, p)` |
| `MultInvBatched` | `:6226`, `:6254` | `BatchedLinAlg::AddMult(B, u, p, …)` |
| `LocalNLOperator::Mult` | `:8825` | `B.Mult(u_l, bp)` |
| `LocalNLOperator::GetGradient` | `:8861` | `grad.SetBlock(1, 0, &B)` |
| `LocalFluxNLOperator` | `:8904`, `:8980` | `B.AddMult(u_l, p_l, −1.)`, `B.Mult(u_l, bp)` |
| `LocalPotNLOperator` | `:8919`, `:8965` | `B.MultTranspose`, `B.AddMultTranspose` |

Sites where `B` acts **transposed** are the (0,1) block and already carry the
`Bnl` addend (`:6155` + `:6163-6166`; `:6262` + `:6274-6276`; `:3038-3046`;
`:8864-8871`); they need nothing.

`LocalOpType` (`darcyhybridization.hpp:585`) is chosen at
`darcyhybridization.cpp:5421-5463`. **Any `m_nlfi` at all forces `FullNL`**
(`:5421` excludes it from the `PotNL` branch, `:5446` from `FluxNL`), which
gives up `InvertA()`'s once-per-assembly flux factorisation at `:5444` and costs
one dense LU of `A` per element per Newton step. That is the price of using the
block slot and §5.6 says how to measure it.

### 2.3 `bilininteg_hdg` — the face integrators and their rules

`HDGDiffusionIntegrator` (`bilininteg_hdg.hpp:645`) supplies `τ`; its class doc
at `:629-644` gives the form
`τ_± = (β ± ½α(u·n)/|u·n|) {h^{-1} Q}`. The quadrature rule is
centralised: `GetHDGFaceIntRule()` in two overloads (`bilininteg_hdg.hpp:698`
two-sided, `:710` one-sided; bodies at `bilininteg_hdg.cpp:101` and `:114`) and
both return `IntRules.Get(geom, 2*max(el orders, trace_el.GetOrder()))`. The
declaration's own comment (`:691-697`) says why it exists — the batched face
assembly has to integrate at *exactly* the per-face loop's rule or the two build
different operators. **That is the precedent to copy for `A9`**: one function
returning the rule, called by every route.

The `trace_el` is in the `max` because of the rank argument recorded in
`CLAUDE.md`: a degree-`2*p_el` rule on a face carries `p_el + 1` points, and a
Gram matrix from `n` points has rank at most `n`, so at `p_trace = p_el + 1` the
trace-trace block is rank-deficient by exactly one per face.

**The same counting applies to `A9` and to the postprocessing stiffness**, and
it is the one place a wrong rule would be silent:
`A1 = [(∇χ_j, ∇χ_i)_K]` needs degree `2k`;
`A2 = [(φ_j, ∇χ_i)_K]` needs `k + k`;
`A9 = [(χ_j, φ_i)_K]` needs `(k+1) + k = 2k+1`.
The existing postprocessing takes
`2 * fe_s->GetOrder() + T->OrderW()` (`postprocess_hdg.cpp:137-139`), which is
`2(k+1) + OrderW` — generous enough for all three. Reuse it rather than
recomputing.

### 2.4 The existing postprocessing: what exists, what is `vdim`-general

**Two postprocessings exist and they answer different questions.** The class doc
at `postprocess_hdg.hpp:47-53` says so.

**(a) `HDGPotentialPostprocessor`** — `fem/darcy/postprocess_hdg.hpp:54`,
implementation `postprocess_hdg.cpp:24-225`. **This is CCSZ-I's eqs (4)/(7)
already built.** Its doc at `:22-35` names it as Nguyen–Peraire–Cockburn eq (25),
which is the same local problem. Verified against the source:

* the local stiffness `A1` is `AddMult_a_AAt(w, dshape_s, A)` at
  `postprocess_hdg.cpp:164`;
* the right-hand side `−A2 α` is the loop at `:191-196`
  (`rhs_all(i,e) -= w * dshape_s(i,d) * giq(d)`), with `giq = iK q_e`;
* the mean rows `b1` and `b2` are `mass_s` and `mass_p`, accumulated at
  `:167` and `:169`;
* the closure is **row replacement**, not the bordered eq (7): row `i_c = 0` of
  `A` is overwritten by `b1` at `:204-206`, and `rhs(i_c) = mass_p * p_e` at
  `:215`. The comment at `:200-203` says why the mean constraint is the
  definition and not one option among several.
* `neq` blocks are looped at `:209-223`, the factorisation reused across them
  (`:141-143`, `Ai.Factor(A)` at `:207`) — the stiffness and the mean row do not
  depend on the equation, only the right-hand side does.

It **is** `vdim`-general: `neq` comes from the potential's `vdim`
(`postprocess_hdg.cpp:35`) and both flux layouts are read from the space rather
than assumed (`:37-49`, `GetFluxBlock()` at `:52-82`) — `neq*dim` on a
scalar-range space and `neq` on H(div). The enriched space is built if the
caller does not supply one (`:92-110`), `ParFiniteElementSpace` in an MPI build.

**Three properties block its use inside an element loop, all verified:**

1. It is constructed from two `GridFunction`s (`postprocess_hdg.hpp:75-76`) and
   `Compute()` reads them through `GetSubVector` (`:132`, `:134`). An
   element-local residual has `u_l` and `p_l`, not grid functions.
2. `Compute()` uses the **shared** `mesh->GetElementTransformation(z)`
   (`postprocess_hdg.cpp:126`), which is not thread safe.
   `ConstructGrad()` already uses the caller-allocated overload
   (`darcyhybridization.cpp:6408`, `fes.GetMesh()->GetElementTransformation(el, &ws.elem)`)
   and that is the pattern to follow.
3. It **factors and discards**: `Ai.Factor(A)` at `:207` happens once per
   element per `Compute()` call, and `B11`/`B12` are never formed. They are
   implicit in `Ai` applied to two right-hand-side *operators* rather than to
   one vector.

**(b) `DarcyForm::ReconstructTotalFlux()` / `ReconstructFluxAndPot()` /
`Reconstruct()`** — `darcyform.hpp:565`, `:597`, `:605`. A richer mixed local
problem on spaces one order higher, with a free trace on every face. Doc at
`darcyform.hpp:76-97` and `:548-596`. Also `vdim`-general (`:556-564`,
`:574-580`).

**`Reconstruct()` is the wrong object for CCSZ-I and must not be used for it.**
Not a defect — a different question. It lifts the nonlinear potential
integrators as a Jacobian frozen at the computed potential (`Mp_nl_lift`,
`darcyform.hpp:153-159`; `Mu_nl_coeff`, `:143-148`), so its output is **not a
linear function of the unknowns**. Feeding it into the source would make each
element's postprocessing an implicit local fixed point, and `B11`/`B12` would not
exist. CCSZ-I's `u*` must be the smaller, linear one.

### 2.5 What CCSZ-I needs that is genuinely absent

| piece | status |
|---|---|
| eqs (4)/(7), the local postprocessing | **built** — `HDGPotentialPostprocessor` |
| `B11`, `B12` as matrices | **absent** — formed and discarded |
| `A9 = [(χ_j, φ_i)_K]` | **absent** — a `MassIntegrator::AssembleElementMatrix2` between `fes_p` and the enriched space |
| `I_h`, i.e. evaluation at `Z_h`'s nodes | **available with no new code** — every L2 element is nodal (`L2_SegmentElement`, `L2_QuadrilateralElement`, `L2_HexahedronElement` are `NodalTensorFiniteElement`; `L2_TriangleElement`, `L2_TetrahedronElement` are `NodalFiniteElement` — `fem/fe/fe_l2.hpp:22,45,79,109,139`), so `FiniteElement::GetNodes()` (`fem/fe/fe_base.hpp:476`) is the node set |
| a reaction term `F(u)` of any kind | **absent from the whole tree.** `grep -rn "Reaction\|reaction" --include=*.hpp --include=*.cpp fem/ miniapps/hdg/` returns **nothing**. There is no `(F(u), v)` integrator to compare against, interpolatory or not. |
| a (1,0) gradient block | **absent** — §2.2 |
| `τ` elementwise constant and **O(1)** | **needs the hook, not the default** — see below |

**A correction to `doc/CCSZ-INTERPOLATORY-HDG-FROM-MEQ.md`.** That file says
"Paper I's `τ` is elementwise constant and O(1), which is what MEQ runs and what
`HDGDiffusionIntegrator` supplies." The second half does not hold as written.
`HDGDiffusionIntegrator`'s built-in stabilization carries `1/h` **intrinsically**:
`bilininteg_hdg.cpp:1201` forms `wn = ip.weight / Trans.Elem1->Weight()`, and the
comment at `:1218-1227` states the intent — "in the jump term, we use
`1/h1 = |nor|/det(J1)`". So the default `τ` is **O(1/h)**, i.e. the LDG-style
scaling, which is the *sequel's* choice and not paper I's.

An O(1) elementwise-constant `τ` is reachable, and cleanly, through the
`HDGStabilization` hook (`bilininteg_hdg.hpp:167`): `StabValue()` at
`bilininteg_hdg.hpp:716-723` divides the quadrature weight out
(`s_diff = wq*ba/face_w`) and multiplies `face_w` back in, and `face_w` is the
physical face weight `ip.weight * nor.Norml2()` (`bilininteg_hdg.cpp:1265`). So
an `HDGStabilization` whose `Eval()` ignores `s_diff` and returns a constant
`τ0` contributes exactly `∫_F τ0 (·)(·)` — paper I's `τ`. `IsConstant()` stays
true, so it costs one query per face (`bilininteg_hdg.hpp:159-160`).

**This matters for the acceptance measurement, not for the design.** A rate
ladder run at the default `τ ~ 1/h` is not a test of CCSZ-I's theorem, whose
Lemma 3.2 and Theorem 3.14 both use O(1) `τ`. §5.3 says to run both.

### 2.6 Names proposed

| new thing | where it sits | derives from / sits beside |
|---|---|---|
| `HDGPostprocessBlocks` | `fem/darcy/postprocess_hdg.{hpp,cpp}` | new class **beside** `HDGPotentialPostprocessor`; the latter is refactored to use it |
| `NodalReactionFunction` | `fem/darcy/postprocess_hdg.hpp` or a new `fem/darcy/reaction_hdg.hpp` | new abstract base, in the spirit of `MixedFluxFunction` (`fem/nonlininteg_mixed.hpp:23`) |
| `HDGInterpolatoryReactionIntegrator` | new `fem/darcy/reaction_hdg.{hpp,cpp}` | `BlockNonlinearFormIntegrator` (`fem/nonlininteg.hpp:196`) |
| `HDGQuadratureReactionIntegrator` | same file | `BlockNonlinearFormIntegrator` — the control, §5.2 |
| `Bg_data` + `GetGradBMatrix()` | `DarcyHybridization` private | beside `Bnl_data` / `GetBnlMatrix()` (`darcyhybridization.hpp:410`, `:416`) |
| `BlockNonlinearFormIntegrator::GetBlockRowMask()` | `fem/nonlininteg.hpp` | new virtual with a conservative default, §3.4 |

---

## 3. The interface — the point of the exercise

`DarcyHybridization::EnsureResidualCache()`
(`darcyhybridization.cpp:3811`, doxygen `:3742-3810`) enumerates four
mechanisms that exist for one purpose: recovering, at run time, which part of a
nonlinear element operator was state-independent. Its own summary, `:3783-3787`:

> All four follow from one fact: `NonlinearFormIntegrator::AssembleElementVector()`
> hands back a vector and says NOTHING about which part of it was
> state-independent. The cache is a run-time reconstruction of information the
> interface discarded, and every item above is a place where the reconstruction
> can be wrong.

and `:3789-3801` says an interpolatory formulation would make all four
unnecessary rather than easier.
`doc/HDG-NPC-PARAMETRIC-COEFFICIENTS-REPLY-TO-GFFP.md` §8 carries the same table
and the same objection.

The three properties that has to buy, and the interface that buys them.

### 3.1 The fixed matrices, assembled once and visibly constant

```cpp
/** @brief The FIXED element-local blocks of the classic HDG postprocessing.

    On each element the nodal values of `u*` are

        gamma_e = B11_e u_e + B12_e p_e

    where B11_e is (ns x na), B12_e is (ns x nd) and RANK ONE -- u* sees the
    potential only through its element average -- and ns, na, nd are the
    enriched, flux and potential dof counts of the element, per equation.

    **Constant in the state by construction.** Only the geometry and the
    diffusion inverse enter, so Assemble() runs once and NOTHING invalidates
    it when the solution, or a parameter of the reaction, moves. That is the
    property EnsureResidualCache() reconstructs at run time and cannot have.

    The one thing that DOES invalidate it is a moving DIFFUSION coefficient,
    because iK is a right-hand-side operator of the local problem. Stated
    rather than guarded: GetSequence() records the Mesh::sequence Assemble()
    saw, GetCoefficientStamp() the caller's own counter, and Apply() aborts on
    a mismatch. A stale read is not silent. */
class HDGPostprocessBlocks
{
public:
   /** @param fes_q flux space; layout read from its range type, exactly as
                    HDGPotentialPostprocessor does (postprocess_hdg.cpp:37-49)
       @param fes_p potential space; neq = fes_p.GetVDim()
       @param fes_s the enriched space, Z_h. The caller supplies it -- see
                    HDGPotentialPostprocessor::Compute()'s own construction
                    (postprocess_hdg.cpp:92-110) for the default to copy, and
                    note that the NODE SET of this space is a free choice with
                    an unmeasured effect on the answer (sec 6.5). */
   HDGPostprocessBlocks(const FiniteElementSpace &fes_q,
                        const FiniteElementSpace &fes_p,
                        const FiniteElementSpace &fes_s);

   void SetDiffusionInverse(Coefficient &c);        // the ik hook
   void SetDiffusionInverse(MatrixCoefficient &c);  // the iK hook
   void SetIntegrationOrder(int order);             // negative: derive it

   /// Form and store every element's B11 and B12. Idempotent.
   void Assemble();

   /// The Mesh::sequence and the caller's coefficient stamp Assemble() saw.
   long GetSequence() const;
   long GetCoefficientStamp() const;
   /// Bump when a diffusion coefficient has moved; the next Apply() aborts
   /// unless Assemble() has been re-run.
   void CoefficientsMoved();

   int GetNumEquations() const;
   int NumNodes(int el) const;   ///< ns for element el, per equation

   /// Views into storage this object owns. Invalid after the next Assemble().
   void GetBlocks(int el, DenseMatrix &B11, DenseMatrix &B12) const;

   /** @brief gamma from an element state, per equation, allocation-free.

       @a gamma is sized neq*NumNodes(el) and laid out EQUATION OUTERMOST,
       as the rest of the branch lays blocks out (postprocess_hdg.hpp:41-45).
       Reads no GridFunction and no shared ElementTransformation, so it is
       callable from inside MultNL()'s threaded element loop. */
   void Apply(int el, const Vector &u_l, const Vector &p_l,
              Vector &gamma) const;
};
```

`HDGPotentialPostprocessor::Compute()` is then `Apply()` per element plus a
scatter, so there is **exactly one copy of the algebra**. That is not a
nicety: `postprocess_hdg.cpp:141-223` is thirty lines that would otherwise exist
twice, and a duplicate of it "agrees for a year and then does not" — meq's
phrase, and this branch's own record of the `HDGDiffusionIntegrator` rule copied
into a batch driver (`bilininteg_hdg.hpp:683-685`).

**Extraction, for the implementer.** With `A` the row-replaced stiffness of
`postprocess_hdg.cpp:204-206` and `Ai = A^{-1}`:

    B11_e = -Ai * A2t          A2t = [(phi_j, grad chi_i)]_K with row i_c zeroed
    B12_e =  (Ai * e_{i_c}) * mass_p^T                       -- outer product

so `B11` is `Ai` applied to `na` right-hand sides and `B12` is one solve
followed by an outer product. `A2t` is the loop already at `:191-196` with
`giq` replaced by each flux basis function in turn. **The row-replacement
closure is not the paper's bordered eq (7)**; the two are equivalent in exact
arithmetic (the rows of the `P^{k+1}` Neumann stiffness sum to zero and
`Σ_i (q, ∇χ_i)_K = 0` makes the right-hand side consistent), but the blocks that
come out are this system's. Say so where they are formed, because a sibling
implementing the bordered form would be compared against them.

### 3.2 Residual and gradient from the SAME matrices

```cpp
/** @brief A pointwise reaction law, evaluated at nodal values.

    Nodal, not quadrature: this is the interface that makes the method
    interpolatory. It is asked for a value at ONE node and knows nothing about
    integration. */
class NodalReactionFunction
{
public:
   virtual ~NodalReactionFunction() = default;
   virtual int NumEquations() const { return 1; }
   /// F(u) at one node. @a x is that node's physical coordinate.
   virtual void Eval(const Vector &x, const Vector &u, Vector &F) const = 0;
   /// dF/du at the same node, (neq x neq). NOT optional -- see below.
   virtual void EvalJacobian(const Vector &x, const Vector &u,
                             DenseMatrix &J) const = 0;
};

/** @brief CCSZ-I's interpolatory reaction term (I_h F(u*_h), v)_K.

    The residual is  A9 F(gamma)  and the gradient is
    A9 diag(F'(gamma)) [B11  B12], with gamma = B11 u_e + B12 p_e. All three
    of A9, B11, B12 are assembled once; F and F' are nodal evaluations.

    **The residual and the gradient cannot be of different operators**, and
    that is structural rather than guarded: both entry points below call the
    same private Prepare(), which is the only code that forms gamma, and both
    read the same A9 and the same blocks. This branch shipped exactly the
    opposite defect twice -- HyperbolicFormIntegrator::AssembleHDGFaceGrad()
    indexing its blocks equation-outermost while AssembleHDGFaceVector() wrote
    them group-outermost, and ConstructGrad()'s unguarded `A += grad_A` double
    counting after CopyLinearGradBlocks() had already written A
    (darcyhybridization.cpp:6473-6492) -- and in both the residual was
    bit-identical while the gradient was wrong. There is nothing here to get
    out of step. */
class HDGInterpolatoryReactionIntegrator : public BlockNonlinearFormIntegrator
{
public:
   /// @a blocks is BORROWED and must outlive this object; one instance may be
   /// shared by any number of reaction integrators.
   HDGInterpolatoryReactionIntegrator(const NodalReactionFunction &F,
                                      HDGPostprocessBlocks &blocks);

   /// Form A9 for every element. Idempotent. The ONLY place geometry or a
   /// quadrature rule is read. Calls blocks.Assemble() if it has not run.
   void Assemble();

   /// A9 F(gamma) into elvec[1]. elvec[0] is left at size zero -- this term
   /// has no flux row.
   void AssembleElementVector(const Array<const FiniteElement*> &el,
                              ElementTransformation &Tr,
                              const Array<const Vector*> &elfun,
                              const Array<Vector*> &elvec) override;

   /// A9 diag(F'(gamma)) B12 into elmats(1,1) and
   /// A9 diag(F'(gamma)) B11 into elmats(1,0). (0,0) and (0,1) untouched.
   void AssembleElementGrad(const Array<const FiniteElement*> &el,
                            ElementTransformation &Tr,
                            const Array<const Vector*> &elfun,
                            const Array2D<DenseMatrix*> &elmats) override;

   /// This term writes the potential row only; see sec 3.4.
   int GetBlockRowMask() const override { return 1 << 1; }

private:
   /// gamma and the nodal F / F'. The one place gamma is formed.
   void Prepare(int el, const Vector &u_l, const Vector &p_l) const;
};
```

The element index is `Tr.ElementNo` (`fem/eltrans.hpp:91`) — a public member of
every `ElementTransformation`, including the caller-allocated
`IsoparametricTransformation` that `ConstructGrad()` fills at
`darcyhybridization.cpp:6408`. So the integrator finds its precomputed blocks
without a new argument on the virtual, which matters: widening
`BlockNonlinearFormIntegrator`'s signature would touch every implementor.

**`EvalJacobian()` is required, not optional.** The `HDGStabilization` doc says
why in as many words (`bilininteg_hdg.hpp:192-195`): in a hybridized method the
Jacobian is never assembled globally, so omitting a piece of it "gives no wrong
answer, only slow Newton convergence — a failure that survives a passing
regression suite."

### 3.3 Where the nodal values live, and who owns them

**`gamma` is not stored.** It lives in the integrator's per-thread scratch —
members under `#ifndef MFEM_THREAD_SAFE`, method locals otherwise, which is
this tree's convention at `bilininteg_hdg.hpp:757` and
`fem/nonlininteg_mixed.hpp:128` — and it is recomputed by `Prepare()` in *both*
`AssembleElementVector()` and `AssembleElementGrad()`.

That is deliberate and it is the whole point. A `gamma` cached between the two
calls is a fifth mechanism of exactly the kind §8 of the reply doc is trying to
delete: it would need invalidating when the state moves, and the state moves
every Newton step. The recompute is one `(ns x (na+nd))` GEMV, which is cheaper
than the quadrature of `F` it replaces. If a later measurement says otherwise,
**the place to cache it is the caller** — `MultNL(GradAtFields)`
(`darcyhybridization.cpp:4893`, the mode "whose fields survive the loop") has
the state in hand and already batches two other things on exactly that
argument — not the integrator.

`B11`, `B12` and `A9` are owned by `HDGPostprocessBlocks` and by the integrator
respectively, both allocated in `Assemble()` and never written again. The
integrator holds `HDGPostprocessBlocks` **by reference and does not own it**, so
one instance serves several reaction terms and the geometry pass happens once.

**Threading.** `Apply()` and `Prepare()` read only owned, const storage plus the
caller's `u_l`/`p_l`; nothing touches a shared `ElementTransformation`. That is
what `postprocess_hdg.cpp:126` prevents today and it is the one requirement meq
flags. The physical node coordinates `x` that `NodalReactionFunction::Eval()`
takes are precomputed per element in `Assemble()` — `Z_h`'s nodes are fixed on
the reference element, so their images are geometry and belong with the
matrices, not in the hot loop.

### 3.4 Not paying `FullNL` for a term that writes one row

Installing anything on `Mnl` forces `LocalOpType::FullNL`
(`darcyhybridization.cpp:5421`), and `FullNL` gives up `InvertA()` (`:5444`) —
one dense LU of the flux block per element per Newton step. `PotNL` exists
precisely to avoid that, and it is refused here only because a block integrator
*might* write the flux row.

**Propose a declaration in the type instead of an inference at run time**, which
is the same move §8 is asking for one level up:

```cpp
// on BlockNonlinearFormIntegrator, fem/nonlininteg.hpp
/** @brief Which block ROWS this integrator ever writes, as a bitmask.

    The default is every row, so an existing implementor is unaffected. An
    integrator that returns a narrower mask is PROMISING not to write the
    others, and DarcyHybridization's LocalOpType selection reads the promise:
    a block integrator writing the potential row only can keep the flux mass
    linear and factored once, instead of falling through to FullNL. */
virtual int GetBlockRowMask() const { return ~0; }
```

Then `darcyhybridization.cpp:5421` widens from `!m_nlfi` to
`(!m_nlfi || m_nlfi->GetBlockRowMask() == (1 << 1))`. One virtual, a
conservative default, no `dynamic_cast`, and the discovery is a statement rather
than a reconstruction — which is the distinction `EnsureResidualCache()`'s item
1 is about.

**This is optional and is stage 5.** It is a performance choice and the
measurement that decides it is in §5.6. Do not bundle it with the correctness
work.

### 3.5 The (1,0) plumbing, concretely

Follow `Sf_data`'s precedent (`darcyhybridization.cpp:5457`) rather than
`Bnl_data`'s. `Bnl_data` is an *addend* applied at four sites; a (1,0) addend
would have thirteen (§2.2's table), and `B` there is a **view onto `Bf_data`**,
so it cannot be added into in place.

```cpp
/** @brief The JACOBIAN's (1,0) block: B plus whatever a block integrator's
    grad_arr(1,0) supplied. Bf_data's own (d_dofs, a_dofs) orientation and
    Bf_offsets indexing.

    Empty -- @a Bg_empty true -- whenever no integrator writes that block,
    which is every problem this branch has met, and GetGradBMatrix() then
    hands back Bf_data's own view. So nothing moves by default and the check
    for that is byte identity of the reference outputs, not a pass.

    Written by ConstructGrad() as B + grad_Apu; the mirror of Bnl_data, which
    holds the (0,1) block's nonlinear addend and whose doc at :395-409 states
    the same asymmetry from the other side. */
mutable Vector Bg_data;
mutable bool Bg_empty{true};

/** @brief The (1,0) block of element @a el: the Jacobian's when @a gradient
    and Bg_data is live, and the linear B otherwise. ONE funnel, so a site
    that forgets the distinction cannot exist. */
void GetGradBMatrix(int el, bool gradient, DenseMatrix &B) const;
```

Every site in §2.2's table that reads `&Bf_data[Bf_offsets[el]]` becomes a call
to `GetGradBMatrix(el, gradient, B)`, and the two batched routes take a
`DenseTensor` over `Bg_data` when it is live. `ComputeElementH()`'s existing
`gradient` flag (`darcyhybridization.cpp:3263`) and `MultInv()`'s `with_bnl`
(`:6100`) are already exactly the predicate needed; neither needs widening.

The regression to copy is named in the source: the comment at
`darcyhybridization.cpp:6159-6161` records that getting the (0,1) case wrong at
`MultInv()` "made the matrix-free gradient disagree with the assembled one".
Because `MultInv()` is the single funnel for applying the eliminated Jacobian,
one change there covers `GradientMode::MatrixFree` as well as `Assembled`.

---

## 4. What it costs in accuracy, and what moves

**Interpolating a nonlinearity is a different quadrature from integrating it.
This is a discretisation change, not a refactor.** That sentence is already in
the tree at `darcyhybridization.cpp:3802-3808` and it is the governing
constraint.

### 4.1 When the two forms agree exactly

`I_h` is a projection reproducing `Z_h`, so `I_h F(u*) = F(u*)` **iff
`F(u*) ∈ Z_h`**. Working through it:

| `F` | `I_h F(u*)` vs `F(u*)` |
|---|---|
| `F(u) = a + c u`, `a, c` **constants** | **identical.** `I_h(a + c u*) = a + c u*`, both being in `Z_h`. |
| `F(u) = a(x) + c(x) u`, spatially varying | **differ.** `a(x) ∉ Z_h`, and `c(x) u* ∉ Z_h` even when `c` is a polynomial of low degree. |
| `F` polynomial of degree `m ≥ 2` in `u` | **differ.** `F(u*)` has degree `m(k+1) > k+1`. |

So the only bit-identity-preserving configuration is **affine `F` with constant
coefficients**, and there the interpolatory term is a *linear* operator, which
§5.1 and §5.5 both exploit.

Two things this does **not** say:

* It does not say the interpolatory method agrees with the standard HDG method
  `(F(u_h), v)` for affine `F`. It does not, for any nonzero `F`: the argument
  changed from `u_h` to `u*`, and that is a different discrete problem
  independently of any quadrature. Remark 2.1 is precisely about this swap.
* It does not say `A9` is exact. `A9 = [(χ_j, φ_i)_K]` needs a rule of degree
  `2k+1`; the postprocessing's existing choice
  `2*fe_s->GetOrder() + T->OrderW()` (`postprocess_hdg.cpp:137-139`) covers it,
  and a cheaper rule would not.

### 4.2 What moves in the 152 serial + 121 parallel references

**Nothing, and the reason is stronger than "it is opt-in".**
`grep -rniE "class .*reaction|ReactionIntegrator" fem/ miniapps/` returns
nothing: **there is no reaction integrator anywhere in the tree.** No
reference installs a term of CCSZ-I's shape, interpolatory or integrated, so
there is no existing answer for the interpolatory form to move.

> **Correcting this entry's own first draft**, which claimed
> `grep -rn "Reaction\|reaction" fem/ miniapps/hdg/` "returns nothing". It
> does not -- it returns `fem/darcy/darcyform.cpp:1633`, and that hit is
> **material to this design rather than incidental**, which is why the
> narrower grep above replaces it rather than the claim being dropped. See
> §4.2.1.

#### 4.2.1 The postprocessing already has a stated position on reaction terms

`DarcyForm`'s reconstruction deliberately **omits** reaction-like domain
integrators from the local problem it solves
(`fem/darcy/darcyform.cpp:1627-1645`): it copies only `ConvectionIntegrator`
and `ConservativeConvectionIntegrator` out of `M_p`'s domain integrators into
the postprocessing form, on the reasoning that "NPC eq (25) is a pure Neumann
problem in the enriched potential, driven by the total flux and closed by the
element average; it carries the diffusion and the stabilisation and nothing
else."

**That is consistent with CCSZ-I and worth being explicit about, because it
looks like a conflict and is not.** CCSZ-I's `u*` is the solution of a local
problem in which the reaction does not appear either; the reaction is
evaluated *at* `u*`'s nodes, downstream of it. So the existing exclusion is
the behaviour this design wants, and `HDGPostprocessBlocks` must reproduce it
rather than "fix" it.

**Two things to check before stage 1, neither of which was verified here:**

* That exclusion list is a whitelist of two convection integrators, so a
  CCSZ-I reaction integrator on `M_p` would be dropped from the
  postprocessing *silently* -- the same silent-drop shape this branch has now
  paid for three times. Whether that matters depends on whether anything
  should ever put one there; state the answer in the class rather than leave
  it to the whitelist.
* The reconstruction reads `M_p`'s DOMAIN integrators. A CCSZ-I reaction term
  is a `BlockNonlinearFormIntegrator` on `Mnl`, which this loop never
  consults, so the exclusion may be vacuous for the proposed design. Confirm
  which form the integrator actually lands on before relying on either
  reading. The nonlinear
terms the suite does exercise are `MixedConductionNLFIntegrator`
(`fem/nonlininteg_mixed.hpp:109`, a flux law) and `HyperbolicFormIntegrator`
(`miniapps/hdg/convdiff.cpp:862`, a convective flux); neither depends on `u*`.

**No default changes are proposed.** The reaction integrator is a new class
nothing constructs; `HDGPostprocessBlocks` is a new class nothing constructs;
`GetBlockRowMask()` has a conservative default.

**The one thing that CAN move is the (1,0) plumbing**, because it touches
`ComputeElementH()`, `MultInv()`, `MultInvBatched()`, `FactorElementsBatched()`,
`ComputeElementsHBatched()`, `LinearResidualBatched()` and
`LocalNLOperator` — every hot path in the class. §5.4's falsifying check is
therefore **byte identity of the reference outputs**, not a pass count. A pass
count would not catch a change in the tenth digit, and this branch has twice
paid for a "green suite is evidence about the tests" mistake
(`CLAUDE.md`, the Tier 1 / Tier 2 lesson).

### 4.3 What it costs, and what it saves

**Saves.** No quadrature of `F` or `F'` in any residual or Jacobian
evaluation — Remark 2.3's claim. On a steady problem that reads as "no
quadrature of `F` in any Newton step", which is meq's interest and not the
paper's. The reply doc §7 records that on *this* tree assembly is 9% of a
Newton step, so the saving is bounded above by that on the problems here; §5.6
says to measure it rather than quote it.

**Costs, per element per evaluation.** One `(ns x (na+nd))` GEMV for `gamma`,
`ns` scalar (or `neq x neq`) evaluations of `F`, one `(nd x ns)` GEMV for the
residual — against a quadrature loop over roughly `(2k+2)^d` points. Two more
GEMVs for the gradient. Plus, if the block slot is used without §3.4's mask,
one dense LU of `A` per element per step from `FullNL`.

**Storage.** `B11` is `ns*na*NE`, `B12` is `ns*nd*NE`, `A9` is `nd*ns*NE`, and
`Bg_data` is `na*nd*NE` — the last being the same order as `Bnl_data` already
allocates when live. `B12` is rank one and could be stored as two vectors per
element; do not bother until it is measured to matter.

---

## 5. A staged plan, each stage with the check that would fail if it were wrong

Standing rules from `CLAUDE.md` that apply to every stage: `make -j6`, never a
bare `make -j`; `MKL_THREADING_LAYER=GNU` on every run; `make style` **before**
the build it is meant to bless; `make clean` after touching
`darcyhybridization.hpp` or any header under `fem/darcy/` (the class-layout trap,
paid for six times); run `unit_tests` from `tests/unit` in a subshell; `-no-vis`
in every batch miniapp run.

### Stage 0 — `HDGPostprocessBlocks`, with `HDGPotentialPostprocessor` on top of it

Extract the element algebra of `postprocess_hdg.cpp:141-223`, form `B11`/`B12`,
and rewrite `Compute()` as `Apply()` plus a scatter.

**Falsifying measurement.** `tests/unit/fem/test_darcy_postprocess.cpp` has three
cases (`:140`, `:179`, `:213`) covering one field, several fields treated
independently, and an H(div) flux. Rerun them and require the postprocessed
potential **bit-identical** to the pre-refactor values, not merely within a
tolerance. A refactor that changes the answer in the last digits has changed the
order of operations somewhere it should not have. Additionally: a direct check
that `Apply(el, u_l, p_l)` reproduces `Compute()`'s element block for a random
`(u_l, p_l)` — which fails if `B11` or `B12` picked up the row-replacement's
`i_c` row wrongly, the single most likely error in the extraction.

**And a check that the check can fail**: assert that `B12` has rank one by
verifying `B12 * v` is parallel to `B12 * w` for two random `v, w`, and that
perturbing `p_l` by anything with zero element mean leaves `gamma` unchanged to
round-off. Both are properties of the *method*, not of the code, so they fail
loudly on a mis-extraction and are not restatements of the implementation.

### Stage 1 — the two reaction integrators, and the null test

Build **both** `HDGInterpolatoryReactionIntegrator` and
`HDGQuadratureReactionIntegrator`, the latter evaluating `F(u*)` at quadrature
points and integrating against `W_h`. The control is not optional: it is the
only reference the tree has, there being no reaction integrator at all (§4.2),
and it is also the comparison the reply doc's item (2) asks for.

**The null test.** With `F(u) = a + c u` and `a, c` **constants**, the two
integrators must agree to round-off — residual *and* gradient, on a random state,
element by element. §4.1 says why: the interpolant is exact there.

**And the discriminating half, without which the null test is worthless.** With
`F(u) = u³` the two must **disagree**, by an amount that (i) is well above
round-off on a coarse mesh and (ii) **decreases like `h^{k+2}`** under
refinement. A null test that passes for a first draft returning zero for every
input is this branch's most recently paid-for mistake
(`HDGDatumErrorEstimator`, recorded in `CLAUDE.md`), and the fix there was the
same: a section whose answer is arithmetic rather than a comparison.

Stage 1 needs **no** library change: the residual is expressible on `Mnl` today
(§2.2), and the gradient's (1,1) block is read. So stage 1 is a complete,
testable increment that ends with a *wrong Jacobian* — deliberately, and stage 3
is where that is fixed and detected.

### Stage 2 — the convergence table

Two ladders, because the theorem covers only one of them.

**(a) Steady, first.** `−Δu + F(u) = f` on `Ω = (0,1)²`, `F(u) = u³ − u`,
`u = sin(πx) sin(πy)`, `f` manufactured. This isolates the spatial
discretisation and needs no time integrator — which matters, because
`CLAUDE.md` records roadmap §8's time integration as **unverified**: all 152
serial references use `--ntimesteps 0`. **No theorem covers the steady case**
(§1.4); the expected ladder is the elliptic analogue of Corollary 3.15 and that
is a guess, stated as one.

**(b) Transient, against Table 1.** CCSZ Example 4.1 exactly:
`F(u) = u³ − u`, `u = sin(t) sin(πx) sin(πy)`, `T = 1`, `k = 0` with backward
Euler and `Δt = h`, `k = 1` with Crank–Nicolson and `Δt = h²`. This is the only
run that is a check against the paper, and it inherits §8's unverified time
integration as a confound — say so in the table.

Mesh sequence: uniform triangulations with `h/√2 = 2^{-1} … 2^{-5}` (`n = 2, 4,
8, 16, 32`), CCSZ's own, **plus `2^{-6}`**. Orders `k = 0, 1, 2, 3`.
`τ`: run at O(1) via the `HDGStabilization` hook (§2.5) **and** at the built-in
`O(1/h)`, because only the first is the paper's method and only the second is
this tree's default.

**Falsifying measurement.** The `u*` column at `k = 1` must reach `k + 2 = 3`
and the `k = 0` column must **not** superconverge — CCSZ's Table 1 gives 3.01
and 0.97 respectively, and `min{k,1}` is why. A `k = 0` column that
superconverges is as much a refutation as a `k = 1` column that does not.

**Three procedural rules, every one of them already paid for on this branch:**

* **Rates must be taken asymptotically.** CCSZ's own `k = 1` `u*` column reads
  2.95, 3.02, 3.02, 3.01 — climbing from below on the coarse end. This branch
  has recorded two cases of stopping too early: the aerofoil flux rate went
  2.08, 1.46, 1.53, **2.50** and recovered, and the lesson written down was "I
  never ran the next refinement". Run `2^{-6}`.
* **Print the solver's convergence flag IN the table, not to a log.** The 3-D
  extension study reported `err_u = 4.17e-01` at `n = 32` — worse than the
  coarsest mesh — with a quiet "[GMRES did not converge]", and that point would
  have entered a rate table had the harness not printed the flag. Print it as a
  column.
* **Check the converged points too.** Re-run the coarsest three with a
  **direct** trace solve and require the iterative numbers to every printed
  digit. Roadmap §5 records a GS-preconditioned solve leaving the answer 13% off
  with the relative test satisfied.

### Stage 3 — the (1,0) gradient block

`grad_arr(1,0)` read at `darcyhybridization.cpp:6428` and `:8614`; `Bg_data` and
`GetGradBMatrix()` per §3.5; every site in §2.2's table routed through the
funnel.

**Falsifying measurement, and it is a threshold rather than a trend.** Pose
`F(u) = a + c u` with constant `a, c`. §4.1 says the interpolatory term is then a
**linear** operator, so the whole discrete problem is linear and **a correct
Jacobian must converge in exactly ONE Newton step.** A Jacobian missing `A10`
cannot: `A10 = c · A9 B11` is not zero, so the step is wrong by a fixed
non-vanishing amount. This branch has the pattern already — "Linear problem: one
step exact, 6.96e-01 to 6.22e-15" is the `633003aeba` acceptance signal for NPC.

The point is that **the residual is bit-identical either way**, so the
discriminating quantity is the residual *history*, not the answer. That is the
same trap as `ConstructGrad()`'s double count, where "the RESIDUAL was
bit-identical to the element loop — 2.9217004681959e+00 to every digit — while
`|S v|` came back 1.4826943370698e+01 against the loop's 2.5716073704950e+01"
(`darcyhybridization.cpp:6484-6486`).

**Two more, both from this branch's hard-won rule that a residual-only
comparison cannot catch a wrong gradient:**

* **LBFGS against Newton on the same operator.** LBFGS never calls
  `GetGradient()`, so it is blind to `A10`. Run `F(u) = u³` with both. Before
  stage 3: LBFGS converges, Newton degrades from quadratic to linear (or
  diverges). After stage 3: both converge and Newton is quadratic. That split is
  what found the `HyperbolicFormIntegrator` indexing defect and what the 20
  broken Tier 2 references announced for free — the 8 that diverged were Newton
  and `-npc`, the 12 that merely drifted were LBFGS.
* **A finite-difference directional derivative of `NPCResidual()` against
  `NPCGradient()`'s action**, with a **floor**. The two agree to every printed
  digit while the residual is meaningful and differ by 2.5e-04 relative on the
  last iterate where both are 2.25e-14 — `CLAUDE.md`'s own note: "An equality
  test between two solvers must not compare round-off relatively."

**And the no-regression check.** Both suites rerun with `Bg_data` present and
never written, requiring **byte-identical** output files (§4.2). Also
`GradientMode::MatrixFree` against `Assembled` on a problem that *does* write
it, per the precedent named at `darcyhybridization.cpp:6159-6161`.

### Stage 4 — parallel

`fem/darcy/pdarcyform.{hpp,cpp}` and the `ParOperator`/`ParGradient` wrappers
(`darcyhybridization.hpp:561`, `:574`). The postprocessing blocks are element
local, so nothing crosses a rank boundary; the risk is entirely in `Bg_data`
reaching the parallel gradient assembly.

**Falsifying measurement: rank-count independence.** `CLAUDE.md` records this as
"the sharp check on parallel NPC", and it is cheap — the same problem at 1, 2, 3
and 4 ranks must give identical error norms. `pconvdiff ... -npc` gave
0.000144657 / 0.000117035 at every rank count. Add the reaction term and require
the same.

### Stage 5 — `GetBlockRowMask()`, and only if it earns it

**Falsifying measurement, and it is the one that decides whether to do this at
all.** Time a Newton step on the stage-2 steady problem three ways: (i) the
reaction on `Mnl` with `FullNL`, (ii) the same problem with the reaction rewired
to depend on `p` only so it can sit on `Mnl_p` and take `PotNL`, (iii) `Mnl`
with the mask honoured. If (i) and (ii) are within noise, `InvertA()` is not
worth a virtual and this stage does not happen. Separate the allocations from
the arithmetic when timing it — the reply doc §7 records that on this tree
"86% quadrature was the answer only after the allocation was hoisted out".

---

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

`doc/CCSZ-INTERPOLATORY-HDG-FROM-MEQ.md` gives a concrete reason to care:
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

`doc/CCSZ-INTERPOLATORY-HDG-FROM-MEQ.md` identifies the sequel as Chen,
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

## 7. Claims I could not verify, listed so they are not read as facts

1. **The sequel's content.** Chen, Cockburn, Singler & Zhang, CAMC **4** (2022)
   477–499 — I have not read it. Everything in §6.7 about it comes from
   `doc/CCSZ-INTERPOLATORY-HDG-FROM-MEQ.md`.
2. **The predecessor's content.** Cockburn, Singler & Zhang, J. Sci. Comput.
   **79** (2019) 1777–1800 (ref [16]) — not read. What §1.2 and §6.1 say about
   it is CCSZ-I's own description of it (§1 p. 2189, Remark 2.1 p. 2191, §5
   p. 2208).
3. **Every performance number in §4.3 is an operation count, not a
   measurement.** No timing was taken. The 9%-of-a-Newton-step figure is quoted
   from `doc/HDG-NPC-PARAMETRIC-COEFFICIENTS-REPLY-TO-GFFP.md` §7, which
   attributes it to this tree.
4. **The `B11`/`B12` extraction formulae in §3.1 were derived by reading
   `postprocess_hdg.cpp:141-223`, not by running anything.** In particular the
   claim that the row-replacement closure and the bordered eq (7) agree in exact
   arithmetic is an argument (the `P^{k+1}` Neumann stiffness rows summing to
   zero, and `Σ_i (q, ∇χ_i)_K = 0`), not a computation. Stage 0's rank-one and
   zero-mean checks are what would falsify it.
5. **`Mnl`'s reachability.** `darcyform.cpp:446-463` records that the `else if
   (Mnl)` branch of `EnableHybridization()` is "REACHED BY NOTHING IN THIS
   TREE", measured over the 152 serial references, and that 12 of the 20
   `-nld -hb` references silently drop a block interior-face integrator. I read
   that comment; I did not re-run the measurement. It matters here because
   CCSZ-I's term goes on `Mnl`, so the design lands on a lightly-exercised
   branch, and the *face* part of it drops silently. The CCSZ-I term is a
   **domain** integrator, which is registered separately at
   `darcyform.cpp:518-520` onwards — but I did not read that block far enough to
   confirm it has no analogous gap. **Check it before stage 1.**
6. **`postprocess_hdg.cpp:95`'s `coll->Clone(coll->GetOrder() + 1)`.** The
   `GetOrder()` convention trap recorded in `CLAUDE.md` is specific to
   `DG_Interface_FECollection`, which returns `p+1`. I did not verify that L2
   and H1 collections return `p`, so I did not verify that this line builds
   `P^{k+1}` rather than `P^{k+2}`. It is almost certainly right — the existing
   postprocessing tests would fail otherwise — but the enriched space's degree
   is load bearing for CCSZ-I in a way it is not for a `Compute()` whose output
   is only compared against an exact solution. **Confirm the degree at stage 0.**
7. **`ns`, `na`, `nd` uniformity.** §4.3's storage figures assume one block size
   per mesh. `CanBatchLocalFactor()` was written to notice when the blocks are
   *not* all the same size (variable-order L2 works today, per `CLAUDE.md`), so
   the offsets are per entity. I did not check what a variable-order potential
   space would do to `Z_h`'s node count per element or to `A9`'s shape.
8. **Nothing here was compiled or run.** No claim about behaviour is a
   measurement. Every file:line was read in the working tree; every rate,
   error and parameter attributed to CCSZ-I was read off the PDF.
