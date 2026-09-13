# Interpolatory HDG_k and its superconvergent postprocessing — a design, not a build

**Status: CHOSEN, and nothing is implemented yet. This file is a to-do.** It is
to be built in preference to porting the element integrators to the device and
in preference to extending the element-local caches; §4.4.4 carries the
reasoning and the measurements behind it. It specifies what
Chen, Cockburn, Singler & Zhang, *Superconvergent Interpolatory HDG Methods for
Reaction Diffusion Equations I: An HDG_k Method*, J. Sci. Comput. **81** (2019)
2188–2212 (**CCSZ-I** below; PDF at `/home/ian/projects/meq/refs/SuperconvergentHDG-I.pdf`)
would need in order to be expressible on `fem/darcy/`, what interface the
interpolatory integrator should present, and what each stage's falsifying
measurement is.

It is the design half of a pair. The *request* half came from meq the same
day, asking for two API widenings and independently identifying the same
missing (1,0) gradient block. Where this file and that one differ, the differences are called
out (§2.5, §6.6) — one of meq's premises about `τ` does not hold in this tree.

**Line numbers are against the WORKING TREE of `gf-hdg-linearise-first` at
`bca170a695` with the uncommitted offload changes in place.** Every one was
read, not remembered. Function and class names are the stable part; if a number
is off by a few dozen lines, grep the name.

**Sections 1-3 touched no source file and ran nothing**; every number quoted
from this tree there is a line number or a count of grep hits, and every number
quoted as a measurement is CCSZ-I's own and is attributed. **§4.4 is the
exception and does carry new measurements** — quadrature point counts read out
of `IntRules`, and arithmetic intensities derived from the block shapes. §5 is
still a list of measurements to *take*.

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

**A correction to meq's request.** It says
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
The reply to gffp's parametric-coefficient proposal carries the same table
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

**Correction, found before implementing: "every site" is wrong, and following
it would corrupt the (0,1) block.** `LocalNLOperator` holds ONE `B` view
(`darcyhybridization.cpp:8759`) and uses it in BOTH roles — non-transposed as
(1,0) at `:9326`, `:9407`, `:9493`, and **transposed** as (0,1) at `:9319`,
`:9426`, `:9474`, plus `TransposeOperator Bt(B)`
(`darcyhybridization.hpp:675`) handed to `grad.SetBlock(0, 1, &Bt)` at
`:9377`. `Bt` is a live view of `B`, not a copy. So writing the (1,0) addend
into that object silently adds it to the (0,1) block as well — which already
carries `Bnl` and must not get it twice.

§2.2 says "sites where `B` acts transposed … need nothing", and that is true of
the *sites* and false of the *object* they share. The design is therefore:
**leave `B` as the linear `Bf` view so every transposed use stays correct, and
introduce a separate accessor for the (1,0) role only.** Concretely three dense
sites take it — `ComputeElementH` (`:3395`), `MultInv` (`:6582`) and
`LocalNLOperator`'s non-transposed uses — and the four batched routes
(`:3053`, `:3154`, `:4222`, `:6672`) take a `DenseTensor` over `Bg_data` when
it is live.

Also worth knowing before starting: only **two** places construct the dense
view from `&Bf_data[Bf_offsets[el]]` and one more is the `LocalNLOperator`
member initialiser; §2.2's table of thirteen counts USES, not views. `ComputeElementH()`'s existing
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

**Both of these were checked before stage 1, and both resolve the same way:
the concern is about a DIFFERENT postprocessing and does not reach CCSZ-I.**

* The whitelist is in `DarcyForm::ReconstructFluxAndPot()` -- the rich mixed
  reconstruction, which §2.4 already rules out for CCSZ-I on separate grounds.
  It reads `M_p->GetDBFI()`, i.e. BILINEAR form domain integrators.
* `HDGPotentialPostprocessor` and `HDGPostprocessBlocks` read **no integrator
  list at all** -- `grep -cE "GetDBFI|GetFBFI|AddDomainIntegrator|
  BilinearFormIntegrator" fem/darcy/postprocess_hdg.{hpp,cpp}` is **0 and 0**.
  They take the flux, the potential and `iK`, and nothing else. So there is no
  whitelist for a CCSZ-I term to be silently dropped from, and the silent-drop
  shape this branch has paid for three times does not arise on this path.
* And the second reading holds too: a CCSZ-I reaction term is a
  `BlockNonlinearFormIntegrator` on `Mnl`, which that loop never consults. The
  exclusion is vacuous for the proposed design twice over.

Stated on `HDGPostprocessBlocks` rather than left here, per this section's own
instruction. The nonlinear
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

### 4.4 The device case, and it is STRUCTURAL rather than arithmetic

**This is the section that decided the method gets built.** It is also the
section that withdraws the two arguments people reach for first, because both
are measured here and both come out at about 1.

#### 4.4.1 Machine balance kills the caching argument, on device

Arithmetic intensity — flops done per byte moved — is the test of whether
storing a result beats recomputing it. A machine pays off a cached read only
above its own balance, peak FLOP/s over peak bandwidth:

| part (FP64) | peak flop/s / bandwidth |
|---|---|
| RTX 2070 SUPER (the card in this workspace) | **0.63** flop/byte |
| A100 80GB | 5.01 |
| H100 SXM | 10.15 |
| MI300X | 15.42 |

**The element-local condensation cache has intensity `na/8`, exactly, and that
is structural rather than a property of any mesh.** Each cached product is the
*result* of its own flops, so flops/reals is the same factor in every one of
them:

| product | flops | reals | ratio |
|---|---|---|---|
| `A^-1 B^T` | `na^2 nd` | `na nd` | `na` |
| `A^-1 C^T` | `nf na^2 nc` | `nf na nc` | `na` |
| `B A^-1 C^T - E` | `nf nd na nc` | `nf nd nc` | `na` |
| `C^T A^-1 B^T + G` | `nf nc na nd` | `nf nc nd` | `na` |
| `C^T A^-1 C^T` | `nf^2 nc na nc` | `nf^2 nc^2` | `na` |

So the whole cache is `na/8` flop/byte independent of `nd`, `nc`, `nf` and of
the mesh, and it pays on device only above `na = 8 x balance`: **`na > 40` on
an A100, `na > 81` on an H100**. meq runs `na = 12`, which is 1.5 flop/byte —
a factor of 3.3 below an A100 and 6.8 below an H100.

**And the card here is the one machine that would say otherwise.** Its FP64 is
1/32 of its FP32, so its balance is 0.63 and the cache clears it above
`na = 5`, i.e. at every order anyone runs. A device measurement taken in this
workspace would report the cache viable and be wrong about every machine the
code would be deployed on.

**CCSZ-I's own steady-state work is worse on this axis, not better.** §4.3's
cost is GEMVs against the fixed matrices, and a GEMV is the lowest-intensity
dense operation there is: each matrix entry is read once for one multiply-add,
**0.25 flop/byte** in double precision. That is below every balance in the
table, including this card's.

The conclusion is not "do not build CCSZ-I". It is that **its performance case
is a HOST case** — where the recompute it removes is dominated by call
overhead and serial-dependent small solves rather than by peak FLOP/s — and
that anyone proposing it as a device throughput win should be shown the 0.25.

#### 4.4.2 The quadrature-count argument is measured, and it is ~1

The obvious remaining argument is that CCSZ-I evaluates `F` at `ns`
interpolation nodes instead of `nq` quadrature points. **Measured against
MFEM's own rules** — `IntRules.Get(geom, 2k+2)`, which is what the HDG
integrators in this tree ask for, against `dim Z_h = P^{k+1}` / `Q^{k+1}`:

| k | tri `ns` | tri `nq` | ratio | quad `ns` | quad `nq` | ratio |
|---|---|---|---|---|---|---|
| 1 | 6 | 6 | 1.00 | 9 | 9 | 1.00 |
| 2 | 10 | 12 | **1.20** | 16 | 16 | **1.00** |
| 3 | 15 | 16 | 1.07 | 25 | 25 | 1.00 |
| 4 | 21 | 25 | 1.19 | 36 | 36 | 1.00 |
| 5 | 28 | 33 | 1.18 | 49 | 49 | 1.00 |

**On tensor-product elements the ratio is exactly 1 at every order, and it is
arithmetic rather than luck**: a Gauss rule exact to degree `2k+2` needs `k+2`
points per dimension, and `dim Q^{k+1}` is `(k+2)^d`. On simplices MFEM's
symmetric rules are efficient enough that the saving never reaches 1.2.

So **even an infinitely expensive `F` buys 1.20x on meq's configuration
(triangles, k=2) and exactly nothing on quads.** Any estimate that puts this
ratio near 2 has sized `nq` as the dimension of the exactly-integrated space
instead of asking `IntRules`; that bound is loose by roughly a factor of two
on triangles.

#### 4.4.3 What `F` actually is here, and why it sits outside both arguments

For meq the relevant `F` is **the evaluation of a 1-D HDG grid function** — a
plasma profile in `psi` — so one evaluation is a data-dependent **search** for
the 1-D element, an **indexed gather** of that element's coefficients, and a
few flops of polynomial evaluation.

That is latency-bound and, on a GPU, divergence-bound. It is not described by
arithmetic intensity at all, and neither §4.4.1 nor §4.4.2 speaks to it.

**It is also the hard part of porting the integrators**, which
`doc/HDG-DEVICE-OFFLOAD.md` step 2 does not mention: a device integrator for
this problem has to carry a 1-D search into a kernel, where the threads of a
warp land in different 1-D elements and the coefficient fetch does not
coalesce.

**And this is where CCSZ-I earns its place, independent of every count above.**
It separates *evaluating* `F` from *integrating* it: form the nodal values
`gamma = B11 u + B12 p`, apply `F` at `ns` nodes, integrate with a GEMV. The
expensive operation becomes **a flat, batchable array of `ns x NE` independent
evaluations**, which can be sorted by 1-D element, coalesced, or held resident.
Classic HDG interleaves the same evaluations with integration inside a
per-element quadrature loop, where none of that is available.

**Worth doing regardless, and it needs no CCSZ-I**: if the 1-D `psi` mesh is
uniform the element index is `floor((psi - psi_0)/h)` and the search — with its
divergence — disappears outright. If it is not uniform, the boundary array is
small enough to sit in shared memory. That question is meq's and costs them
one look at their profile mesh.

#### 4.4.4 The decision

**CCSZ-I is to be built, and it is to be built INSTEAD of two things**: instead
of porting the element integrators to the device (`HDG-DEVICE-OFFLOAD.md`
step 2, the largest remaining item there), and instead of extending the
element-local caches any further.

The reason is §4.4.3 and not §4.4.1 or §4.4.2. CCSZ-I does not make the
arithmetic faster; it **removes the integrator that would otherwise have to be
written as a device kernel**, and leaves behind two things a device already
does well — a batched GEMV, which `BatchedLinAlg` supplies today, and a flat
array of `F` evaluations that can be organised for coalescing. The caches go
the other way: each one adds state with a lifetime to track, buys less than
the last, and trades the resource a device has spare for the one it does not.

`HDGPostprocessBlocks` also has the lifetime property the caches cannot: its
matrices are constant **by construction**, since only the geometry and the
diffusion inverse enter them, with a sequence and coefficient stamp that
aborts on a stale read rather than a rule someone has to remember.

---

## 5. A staged plan, each stage with the check that would fail if it were wrong

Standing rules from `CLAUDE.md` that apply to every stage: `make -j6`, never a
bare `make -j`; `make style` **before**
the build it is meant to bless; `make clean` after touching
`darcyhybridization.hpp` or any header under `fem/darcy/` (the class-layout trap,
paid for six times); run `unit_tests` from `tests/unit` in a subshell; `-no-vis`
in every batch miniapp run.

### Stage 0 — `HDGPostprocessBlocks`, with `HDGPotentialPostprocessor` on top of it

Extract the element algebra of `postprocess_hdg.cpp:141-223`, form `B11`/`B12`,
and rewrite `Compute()` as `Apply()` plus a scatter.

**BUILT.** `HDGPostprocessBlocks` is in `fem/darcy/postprocess_hdg.{hpp,cpp}`
and `HDGPotentialPostprocessor::Compute()` is `Apply()` plus a scatter. Three
cases in `tests/unit/fem/test_darcy_postprocess.cpp`.

**The falsifying measurement this section asked for was the wrong one, and it
is withdrawn.** It required the refactored `Compute()` to be **bit-identical**
to the pre-refactor values, on the grounds that "a refactor that changes the
answer in the last digits has changed the order of operations somewhere it
should not have". It has to change the order of operations, and that IS the
refactor: the old loop accumulated the right-hand side over quadrature points
from the flux **values**, and the cached route contracts a precomputed matrix
with the flux **coefficients**. Same sum, different association, and the whole
point of caching is to do it in the second order. Measured across nine
configurations (orders 1-3, 2-D and 3-D, `neq` 1-3, H(div) flux, a
non-diagonal x-dependent `iK`): **2.3e-15 to 3.5e-14 relative**, and zero of
the nine bit-identical.

**What replaced it is stronger in kind, because its answer is arithmetic.**
With a flux that is exactly `-grad P` for `P` of degree `k+1`, the local
Neumann problem has `P` as its solution and the mean row pins the constant, so
`u*` must BE `P` -- an answer that does not depend on any implementation
agreeing with any other. Measured: **1.7e-15 to 1.3e-13 relative** over the
same sweep. A comparison against the route being replaced cannot serve as the
acceptance here for the reason above; it is kept as a **separate** case, at
round-off rather than bit-identity, where what it actually pins is that the
scatter puts each equation's block where the enriched space expects it.

The `i_c` row is indeed the most likely error, and it is what the mutation
arms below exercise.

**And the checks were checked, by mutation rather than by reading.** Three
deliberate defects were gated on an environment variable inside `Assemble()`
(`env -u` for the control arm, per `CLAUDE.md`'s note that `getenv("X")` is
non-null for the empty string): (1) not clearing row `i_c` of the flux block,
(2) dropping the sign on `B11`, (3) putting the mean constraint's unit vector
on the wrong row. **All three fail**, 14 assertions each, and they are caught
by the polynomial case AND independently by the pre-existing convergence case
`"Local postprocessing improves the potential"`.

The two property checks this section proposed -- `B12` rank one, and a
zero-element-mean move of `p_l` leaving `gamma` alone -- are built and pass,
but **be clear about what they can and cannot see**: `B12` is STORED as its
two rank-one factors (`c12` and `mass_p`) rather than as a matrix, so rank one
is structural and none of the three mutations can break it. They are contract
tests against a future refactor that stores a dense block, not acceptance for
this one. Said here because a reader would otherwise count them as part of the
acceptance, which is the shape of mistake this file records elsewhere.

### Stage 1 — the two reaction integrators, and the null test

Build **both** `HDGInterpolatoryReactionIntegrator` and
`HDGQuadratureReactionIntegrator`, the latter evaluating `F(u*)` at quadrature
points and integrating against `W_h`. The control is not optional: it is the
only reference the tree has, there being no reaction integrator at all (§4.2),
and it is also the comparison the reply doc's item (2) asks for.

**BUILT.** `NodalReactionFunction`, `HDGReactionIntegratorBase`,
`HDGInterpolatoryReactionIntegrator` and `HDGQuadratureReactionIntegrator` in
`fem/darcy/reaction_hdg.{hpp,cpp}`; three cases in
`tests/unit/fem/test_darcy_reaction.cpp`. Both entry points go through one
`Prepare()`, so the residual and the gradient cannot be of different operators.

**The null test passes** — affine `F`, residual and gradient, to 1e-12
relative — **but only after the comparison was made capable of failing, and as
specified it was not.**

#### The finding: on a Gauss-Legendre box the two integrators are IDENTICAL

`L2_FECollection` is nodal at the **Gauss-Legendre** points. On a
tensor-product element the enriched space's `k+2` points per dimension are
therefore exactly the points of the rule the postprocessing already uses, and
`A9` assembled with that rule gives `(chi_j, phi_i) = w_j phi_i(x_j)`. The
interpolatory term is then the quadrature term **identically, for every `F`** —
interpolation has degenerated to collocation.

Measured at order 2, `n = 4`, `F = u^3 - u`, relative residual gap:

| element | enriched basis | nodes vs rule points | default rule | raised rule |
|---|---|---|---|---|
| quad | Gauss-Legendre | 16 vs 16 | **4.3e-16** | 3.9e-05 |
| quad | Gauss-Lobatto | 16 vs 16 | 6.5e-03 | 6.5e-03 |
| triangle | Gauss-Legendre | 10 vs 12 | 1.2e-03 | 1.1e-03 |
| triangle | Gauss-Lobatto | 10 vs 12 | 1.0e-02 | 1.0e-02 |

So **the null test as specified could not fail**: run at the default rule on a
box it passes for `u³` as readily as for `a + c u`, and would pass for an
integrator that ignored `F` entirely. The cases now raise the control's rule
deliberately, and a third case pins the collocation itself — it is a real
property and the next person to compare the two forms will hit it.

**This bears on the method and not only on the test.** Where this branch's
miniapps run — quadrilaterals and hexahedra with the default L2 basis —
CCSZ-I's interpolatory term *is* standard HDG evaluated with the `(k+2)`-point
Gauss rule. The paper's own meshes are simplices, where it is not.

#### And the gap converges at `h^{k+4}`, not the `h^{k+2}` predicted here

Measured on a Gauss-Legendre box with a non-collocating control, `n = 4..32`:

| k | rates | Gauss-Lobatto basis |
|---|---|---|
| 1 | 4.96, 4.99, **5.00** | 2.88, 2.98, **2.99** |
| 2 | 5.97, 5.99, **6.00** | 3.97, 3.99, **4.00** |
| 3 | 6.95, 6.99, **7.00** | — |

Two orders better than `||I_h g - g|| = O(h^{k+2})` suggests. The mechanism:
the leading interpolation error at Gauss points is the degree-`k+2` Legendre
polynomial, L2-orthogonal to `P^{k+1}`; the next term contributes
`<omega, (x-x_c) phi>`, and `(x-x_c) phi` is still in `P^{k+1}` for `phi` in
`W_h`, so it vanishes too. **The explanation was tested rather than asserted** —
the Lobatto column is the test, and it recovers `h^{k+2}` exactly.

#### Two more things the build turned up

* **The (1,1) gradient gap is EXACTLY zero at k = 1 on a box**, and that is the
  same orthogonality rather than a defect. `B12` is rank one and its column is
  the constant function, so that block probes `<I_h J - J, phi>` in one
  direction; at k = 1 the error is the degree-3 Legendre polynomial times a
  linear factor, orthogonal to `phi` in `Q1`. Measured 4.8e-16 against 2.4e-06
  at k = 2 and 8.9e-04 on a triangle at k = 1. The test therefore discriminates
  on the residual and the (1,0) block, and says why.
* **MFEM's simplex quadrature loses accuracy above its tabulated range.** A
  first draft of the control ran at degree 30, where a triangle rule jumps from
  126 points to **816**. The symptom was the control's own gradient disagreeing
  with a finite difference of its own residual by 1.1e-07, where every rule from
  degree 8 to 25 gives 4e-12 — i.e. it looked like a defect in the integrator
  and was a defect in my choice of rule. The control now runs at `4k+6`, exact
  for `F(u*) phi` and well inside the range. The branch
  `intrules-triangle-high-order` is about this.

**Each gradient is also checked against a finite difference of its OWN
residual**, so "the two agree" cannot mean "both are wrong the same way" —
4e-12 relative for both.

Stage 1 needs **no** library change: the residual is expressible on `Mnl` today
(§2.2), and the gradient's (1,1) block is read. So stage 1 is a complete,
testable increment that ends with a *wrong Jacobian* — deliberately, and stage 3
is where that is fixed and detected.

### Stage 2 — the convergence table

Two ladders, because the theorem covers only one of them.

**(a) RUN, and the falsifying measurement passes.** `Delta u - F(u) = g` on
`(0,1)^2`, `F(u) = u^3 - u`, `u = sin(pi x) sin(pi y)`, `g` manufactured;
uniform triangulations, direct trace solve, local nonlinear `rtol` 1e-13.

**tau = O(1), paper I's stabilization, interpolatory:**

| k | u | q | u* | CCSZ Table 1 |
|---|---|---|---|---|
| 0 | 1.00 | 1.00 | **1.00** | 0.97 — must NOT superconverge |
| 1 | 2.02 | 2.01 | **3.00** | 3.01 |
| 2 | 3.04 | 3.01 | **4.01** | — |
| 3 | 4.03 | 4.01 | **5.00** | — |

Both halves of the falsifier hold: `k = 1` reaches `k+2` and `k = 0` does not
superconverge, which is `min{k,1}`. Taken to `n = 64` for `k = 0,1` and `n = 32`
for `k = 2,3`, flat at the end, so the rates are asymptotic; the convergence
flag is a COLUMN and reads yes throughout.

**tau = 1/h, this tree's default, destroys it at every order** — and at `k = 0`
the error stops decreasing and starts GROWING:

| k | u | q | u* |
|---|---|---|---|
| 0 | −0.10 | 0.07 | **−0.25** |
| 1 | 2.03 | 1.14 | **1.96** |
| 2 | 3.03 | 2.12 | **3.08** |

So §2.5's correction is not a footnote: a ladder run at the default `tau` is not
a test of CCSZ-I's theorem, and would have reported the method as failing.

**Interpolating costs nothing measurable.** Against the quadrature control at
`tau = O(1)`, same meshes: `u*` agrees to the **fifth** digit at `k = 1`
(7.26499e-06 against 7.26462e-06 at `n = 32`) and the **sixth** at `k = 2`.
Same rates, same constants. That is Remark 2.3 borne out here, and it is the
`h^{k+4}` gap of stage 1 arriving in a solve.

**Every row was produced with a DIRECT trace solve**, which satisfies the third
procedural rule in the stronger direction — there is no iterative number to
cross-check because none was used.

### What the ladder cost to get right, none of which was in the plan

* **`q = -grad u`.** Measured, not reasoned: comparing against `+grad u` pinned
  the flux error at 4.44 on every mesh, which is exactly `2||grad u||`.
* **The source sign is `g = Delta u - F(u)`.** Both signs were run; `-1` gives
  the linear control's rates exactly and `+1` collapses `u*` to rate 0.
* **The local nonlinear solve's default `rtol` is 1e-6 and it caps the outer
  Newton.** Anything on `Mnl` forces `LocalOpType::FullNL`, which replaces the
  direct local inverse with an ITERATIVE element solve. With the reaction
  identically zero — a linear problem — Newton fell to 9.5e-06 in one step and
  then oscillated there for 40 iterations. `SetLocalNLSolver(..., rtol=1e-13)`
  fixes it. **A ladder run without this reports errors polluted at 1e-5.**

**(b) Transient, against Table 1 — NOT RUN.** It needs the time integration
that roadmap §8 records as unverified, and (a) already settles the spatial
question against CCSZ's own `k = 0` and `k = 1` columns.


### Stage 3 — the (1,0) gradient block

**BUILT, and the falsifier passes.** `Bg_data` / `Bg_empty` and
`GetGradBMatrix(el, gradient, B)` on `DarcyHybridization`; `ConstructGrad()`
passes `grad_arr(1,0)` instead of NULL and writes `B + grad_Apu`. One case in
`tests/unit/fem/test_darcy_reaction.cpp`.

**The two predicates were already exactly right**, as this section hoped:
`ComputeElementH`'s `gradient` and `MultInv`'s `with_bnl`. `with_bnl` was
CHECKED rather than taken on trust — it is true at `MultNL`, `NPCReduce` and
`NPCRecover`, the three Jacobian applications, and false at `ReduceRHS` and
`ComputeSolution`, the two linear ones.

**The falsifier, on the affine law where the discrete problem is exactly
linear:**

| addend | Newton steps | answer |
|---|---|---|
| none (pre-stage-3) | 5, at a constant factor 4.2e-3 | 5.94876e-02 |
| **`+grad_Apu`** | **1** | 5.94876e-02 |
| `-grad_Apu` | 6 | 5.94876e-02 |

All three return the SAME answer, which is the point: the residual is
identical either way and the discriminating quantity is the iteration history.
The cubic ladder's errors are byte-for-byte stage 2's, with Newton down to a
uniform **4** from 7, 6, 5, 5, 4.

### Three defects in the implementation, none of them where this section looked

* **"Every site that reads `Bf_data` becomes a call to the funnel" is wrong**,
  and §3.5 now carries the correction. `ComputeElementH` and `MultInv` EACH
  read `B` in both roles — transposed via `AiBt.Transpose(B)` and
  `B.MultTranspose` to build the (0,1) block, which already carries `Bnl`.
  Substituting one view for both put the (1,0) addend into the (0,1)
  application and Newton's first step came back `inf`. Each now keeps a linear
  `B` for the transposed role and takes `Bg` only for the non-transposed
  products. **The aliasing was found and written down for `LocalNLOperator`
  first, and then walked into in the two functions that had not been audited.**
* **`Bg_data` must be seeded with `Bf_data`, not with zero.** `Bnl_data` can be
  zero-filled because it is an ADDEND — an unwritten element contributes
  nothing. This array REPLACES `B` at its readers, and `Bg_empty` goes false as
  soon as the first element writes, which exposes every element the loop has
  not reached. Zero-filled, those read `B = 0`, the Schur complement loses its
  divergence form, and the step is `inf`. **A replacing cache and an adding
  cache have different initialisation obligations**, and that is the
  generalisable part.
* **The finite-difference check of the reduced gradient proposed below is not
  valid for this operator** and was discarded rather than concluded from: it
  reported ~100% disagreement with stage 3 DISABLED, i.e. against code that had
  converged for years. `A->GetGradient()` on the reduced trace operator is not
  `d(A->Mult)/dx` in the naive sense. The calibrated falsifier is the one-step
  Newton above.

**How the second and third were separated**, and it is the technique this file
already records: two environment gates inside the routine — one disabling the
READ side (`GetGradBMatrix` always returns `Bf`), one scaling the addend by
0/±1 — with `env -u` for the control arm. Read side off gave 5 steps, i.e. the
pre-stage-3 arithmetic exactly, which exonerated the routing in one run and
pointed at the data. Both gates are removed.

**The four batched routes are DONE**, which is §3.5's remaining half. They
refused first — `MFEM_VERIFY(Bg_empty, ...)` naming the fix — on the grounds
that a visible gap beats a wrong answer on a device path; that refusal is now
replaced by the thing it was standing in for.

* `FactorElementsBatched()` and `ComputeElementsHBatched()` each use `B`
  non-transposed only (`S += B·A⁻¹Bᵀ`, and `B·A⁻¹Cᵀ − E`), so each takes one
  tensor over the (1,0) store.
* `MultInvBatched()` uses it in **both** roles and now holds **two** tensors,
  exactly as the dense `MultInv()` holds two views and for the same reason.
* The batched **residual** route's guard was the one mistake here, and it was
  a correct refusal for a wrong reason: a residual never reads the Jacobian's
  (1,0) block at all. Its `B` stays linear, with the reason written on it.

A file-local `GradBStore()` picks the store, so the three sites state the
distinction rather than re-derive it — the batched mirror of
`GetGradBMatrix()`.

**Pinned, and the pin was checked to fail**: "The batched local routes carry
the (1,0) gradient block" runs the affine problem under `LocalFactorMode`
`Serial` and `Batched` and requires both to take ONE Newton step and to agree
to 1e-12. It also requires `CanBatchLocalFactor()` and `CanBatchLocalSolve()`
to be true in the batched arm, because this branch has three recorded cases of
a mode that never fired while every test passed. Gating `GradBStore()` back to
the linear store takes the batched arm from **1 step to 6** while the dense arm
stays at 1.

### §4.2's "byte identity, not a pass count" — done properly, and it found something else

`regression_test.py` compares the two printed error norms with
**`tol = 1e-4` relative** (`equal()` at `:17-19`); only the iteration count is
exact. So "4 / 152, unmoved" is a statement at 1e-4 and NOT the byte identity
§4.2 demands — a change in the sixth digit passes it. The check was redone
with `equal()` replaced by `a == b`, which at the six significant digits the
references store is exact.

**Result: Stage 3 on and Stage 3 off are IDENTICAL across all 152 cases** —
`diff` of the per-case verdicts is empty, with the (1,0) block gated back to
NULL by an environment variable in one arm and `env -u` in the other. That is
the byte-identity check, and it passes.

**And it exposed something that is NOT ours: 14 of the 152 references are
stale.** Under exact comparison the suite reports 4 DIFFERS + **14 FAILING** +
85 SUCCESS instead of 4 + 0 + 99, in **both** arms. They agree to 1e-4 and not
to the printed digits. Every one is a nonlinear hybridized case:
`p2_o2_hb_nl`, `p1_o2_hb_nl_nld_newton`, `p2_o2_dg_hb_h1_nl`,
`p2_o2_hb_upwind_nl`, and ten more. Pre-existing, reproducible, and unrelated
to this work — but it means **§4.2's acceptance criterion as written is not
achievable against this reference set**, and anyone who takes "byte identity of
the reference outputs" literally will think they broke something. The
achievable form is the two-arm comparison above.

### And then the references were rebuilt, on the caller's instruction

Local only, not pushed. **18 serial references regenerated** — the 4 DIFFERS
and the 14 FAILING — after which the suite is `SUCCESS: all tests finished
succesfully! (49 / 152 skipped)` **under the exact comparison**. So §4.2's
criterion as written is achievable again, against this reference set, and a
future change to the interpolatory path can be held to byte identity rather
than to 1e-4.

Two things that had to be got right, and neither is obvious:

* **The 49 skips are not stale, they are the OTHER build's references.** Every
  skipped case has a twin — `p1_o3.txt` records `GMRES+GS`, `p1_o3_umfpack.txt`
  records `GMRES+UMFPack`, and both reconstruct the SAME command line.
  A build with SuiteSparse reproduces one and skips the other. Regenerating a
  skipped reference would overwrite the non-UMFPack arm with a duplicate of the
  UMFPack one and quietly delete half the set's coverage.
* **`regression_test.py` rebuilds the command from a FIXED option list**, so a
  reference recording anything outside that list at a non-default value would
  be silently rewritten as a different case. The regeneration therefore refuses
  unless the old option block is a SUBSET of the new one. It cannot be equal:
  `convdiff` has gained fourteen options since these were written
  (`--ref-levels`, `--trace-DG`, `--newton-rtol`, `--npc`, `--gradient-mode`,
  …), so a first draft comparing the blocks for equality refused all 18. Subset
  is the right invariant — every recorded setting reproduced, anything new at
  its default.

**What the regeneration absorbed**, which is the interesting part: the 14
FAILING moved only in the 6th–7th significant digit at an UNCHANGED iteration
count (8.2e-07 to 7.8e-05 relative), while the 4 DIFFERS moved their iteration
count — 123→119, 173→198, 107→111, 140→139 — with the error norms **identical
to every printed digit** in three of the four. So the set was drifting in two
different ways, and the 1e-4 tolerance was hiding one of them completely.


### Stage 4 — parallel

**RUN, and it needed NO code change** — which was checked rather than hoped
for. `pdarcyform.cpp` has **zero** references to `Bf_data`, `ConstructGrad`,
`ComputeElementH` or `MultInv`; `ParOperator` and `ParGradient`
(`darcyhybridization.hpp:640`, `:653`) are nested in `DarcyHybridization` and
hold a reference to it, so they delegate to the same element-local machinery
and `Bg_data` reaches them by construction.

**Falsifying measurement: rank-count independence**, on the steady cubic
problem with the interpolatory term, `n = 8`, order 1, triangles:

| ranks | `err_u` | `err_u*` | Newton |
|---|---|---|---|
| 1 | 1.39666989016496e-02 | 4.66538176925364e-04 | 4 |
| 2 | 1.39666989016496e-02 | 4.66538176925401e-04 | 4 |
| 3 | 1.39666989016496e-02 | 4.66538176925475e-04 | 4 |
| 4 | 1.39666989016496e-02 | 4.66538176925430e-04 | 4 |

`err_u` is identical to all fifteen digits; `err_u*` agrees to twelve
significant digits, the residue being the order of a parallel reduction. The
1-rank row reproduces the SERIAL ladder's `n = 8` entry exactly.

**And stage 3's own falsifier holds in parallel**: the affine law converges in
ONE Newton step at 1, 2 and 4 ranks, so the (1,0) block is reaching the
parallel gradient assembly and not merely the serial one.

**Parallel baselines unmoved**: 101 cases / 70,946 assertions on 2 ranks,
identical to before any of this work. The parallel tree needed
`make config` (a library source file was ADDED, which `make clean` does not
fix) and then `make clean` (a member was added to `DarcyHybridization`, which
is a layout change).


### Stage 5 — `GetBlockRowMask()`: MEASURED, and it earns it

**The measurement was taken differently from the proposal above, and the
proposal was confoundable.** (i) against (ii) compares a block integrator on
`Mnl` against a *different nonlinearity* on `Mnl_p`, so the two arms differ in
the discretisation as well as in `lop_type` — this branch's own "check that the
two configurations differ in that parameter ONLY". What was run instead varies
`lop_type` and nothing else: a temporary gate in `Finalize()`'s selection
forces the general `FullNL` branch on a problem that would otherwise take
`PotNL`, with `env -u` for the control arm. Both arms then solve the **same**
discrete problem, which the printed errors confirm digit for digit.

`convdiff -p 3 -dg -hb -nlp -nlc -hdg 4 -nls 3` on quads, `Solver took` (the
Newton solve, trace solves included), median of five:

| order | n | Newton | PotNL | FullNL | FullNL / PotNL |
|---|---|---|---|---|---|
| 1 | 20 | 2 | 0.178 | 0.176 | **0.99** |
| 1 | 40 | 4 | 1.425 | 1.408 | **0.99** |
| 2 | 20 | 3 | 0.448 | 0.480 | **1.07** |
| 2 | 40 | 4 | 2.545 | 2.669 | **1.05** |
| 3 | 20 | 4 | 1.002 | 1.262 | **1.26** |
| 3 | 40 | 4 | 4.031 | 4.827 | **1.20** |

`err_t` is identical to every printed digit in both arms of every row.

**Order 1 is noise and order 3 is not**, so the answer is not "within noise"
and the stage happens. A fourth-order point at `n = 10` gives 0.447 against
0.929 — but the two arms took 4 and 3 Newton steps there, so it is quoted as
evidence of the TREND and not as a ratio.

Two things sharpen the reading in opposite directions, and both are worth
stating:

* **The denominator includes the global trace solves**, which `InvertA()` does
  not touch. So 20–26% of the whole step at order 3 understates the saving on
  the element-local work.
* **The nonlinearity here is a `HyperbolicFormIntegrator` on domain and
  faces**, which is dearer than CCSZ's reaction — one nodal `F` per node. A
  cheaper nonlinearity makes the flux factorisation a LARGER share, so these
  are a lower bound for the interpolatory case, not an upper one.

#### The stage is bigger than "one virtual", and §3.4 understates it

§3.4 says the change is `GetBlockRowMask()` plus widening the predicate at
`darcyhybridization.cpp:5421`. That is the SELECTION. The `PotNL` machinery
then has to be able to run a block integrator at all, and today it cannot:

* **`LocalPotNLOperator::Mult()` never calls `AddMultBlock()`**, which is the
  only site that evaluates `m_nlfi`. It calls `AddMultDE()`, which knows
  `m_nlfi_p`, the linear `D` and `c_nlfi_p`. So under `PotNL` a block
  integrator on `Mnl` is silently not evaluated — the residual would simply
  lose the reaction.
* **`LocalPotNLOperator::GetGradient()` builds `B A⁻¹ Bᵀ + ∂R_p/∂p` with the
  LINEAR `B` in the left factor.** CCSZ's reaction writes only the potential
  row but READS both — `γ = B11 α + B12 β` — so its eliminated gradient is
  `(B + ∂R_p/∂u) A⁻¹ Bᵀ + ∂R_p/∂p`, and `∂R_p/∂u` is exactly stage 3's
  `Bg`. The `PotNL` path needs the same two-role split the `FullNL` one
  already has.

**The mask is still the right promise.** "Writes the potential row only" is
exactly the condition for `A` to stay linear and factorable once; reading the
flux in that row is fine *provided* the chain rule is carried, which is what
`Bg` is. The work is four sites, not one, and this section used to say
otherwise.

#### BUILT, and what the pin does and does not cover

`BlockNonlinearFormIntegrator::GetBlockRowMask()` with a `~0` default;
`HDGReactionIntegratorBase` returns `1 << 1`, so both reaction integrators
inherit it. `Finalize()` widened. `ConstructGrad()` NULLs the flux-row entries
of `grad_arr` under `PotNL` — **asking for them is what would destroy the
factorisation**, because a well-behaved implementor handed a non-NULL `(0,0)`
writes a zero matrix into the array that holds `A`'s LU factors, and the
existing `else { A = 0.; }` would do it even if the integrator wrote nothing.
`LocalPotNLOperator::Mult()` gained the block integrator's potential row;
`LocalPotNLOperator::GetGradient()` takes the (1,0) block in its left factor
and gained `AddGradBlockPot()` for the (1,1). One new public predicate,
`FluxMassIsPrefactored()`, so a caller and a test can see which regime is in
force without the enum becoming public.

**The pin is a control, because a performance declaration must change
nothing**: "GetBlockRowMask() keeps the flux mass factored once" runs the same
problem twice, the second time through a test-local subclass that overrides
the mask back to `~0`. Both must be one Newton step and agree to 1e-12, and
`FluxMassIsPrefactored()` must be true in one arm and false in the other — no
environment gate anywhere.

**Two of the three ways this can break are covered and one is not**, and the
uncovered one is said so in the test rather than left to look covered:

| break | caught by | measured |
|---|---|---|
| the residual loses the reaction | outer Newton count | 1 → **10** |
| `ConstructGrad()` writes `A` | would destroy the LU factors | not separately gated |
| the gradient's left factor is the LINEAR `B` | **nothing here** | answer, outer count and the local total (96) ALL unchanged |

The third was gated and measured rather than assumed, and
`LocalPotNLOperator::GetGradient()` was confirmed reached by aborting inside
it — so the substitution IS exercised and simply does not show: the local
solve converges to the same root from an inexact Jacobian. That half rests on
the analytic argument and on `ComputeElementH()`/`MultInv()`, where the same
substitution returns `inf` on the first step.

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

## 7. Claims I could not verify, listed so they are not read as facts

1. **The sequel's content.** Chen, Cockburn, Singler & Zhang, CAMC **4** (2022)
   477–499 — I have not read it. Everything in §6.7 about it comes from
   meq's request.
2. **The predecessor's content.** Cockburn, Singler & Zhang, J. Sci. Comput.
   **79** (2019) 1777–1800 (ref [16]) — not read. What §1.2 and §6.1 say about
   it is CCSZ-I's own description of it (§1 p. 2189, Remark 2.1 p. 2191, §5
   p. 2208).
3. **Every performance number in §4.3 is an operation count, not a
   measurement.** No timing was taken. The 9%-of-a-Newton-step figure is quoted
   from the reply to gffp's parametric-coefficient proposal, §7, which
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
