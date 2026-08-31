# The two nonlinear orderings of `DarcyHybridization`

A technical reference for
`DarcyHybridization::SetNonlinearOrdering(NLOrdering)`: what
`CondenseThenLinearise` and `LineariseThenCondense` each do, what they cost,
and where `LineariseThenCondense` breaks an assumption an MFEM caller is
entitled to make about an `Operator`.

**Which source this describes.** Everything below is read from branch
`gf-hdg-linearise-first` at commit `50c5d75def` ("Move the findings out of the
markdown and into the code they are about"). Every `file:line` citation is a
line number *at that commit*; read them with
`git show gf-hdg-linearise-first:<path>`. The working tree was moved onto
`gf-hdg-dev` while this was being written, and `gf-hdg-dev` does not carry the
ordering at all — `NLOrdering` does not exist there.

`doc/HDG-LINEARISE-THEN-CONDENSE.md` is the *plan*, not the design. The
implementation deliberately departs from it, and in three places the plan is
now simply stale; §7 says where. Where the plan and the code disagree, the code
is what is described here.

---

## 1. The setup

`DarcyHybridization` hybridizes a mixed system in a flux `u` (called `q` in the
HDG literature and in the miniapps), a potential `p`, and a trace `λ` living on
the mesh skeleton. The full system is
(`fem/darcy/darcyhybridization.hpp:63-69`):

```
| Mu  ±Bᵀ  Cᵀ | | u |   | bu |
| B    D   E  | | p | = | bp |
| C    G   H  | | λ |   | br |
```

`Mu` is the flux mass, `B` the (generalised) divergence, `D`/`E`/`G`/`H` the
potential mass and the HDG stabilization contributions, `C` the flux
constraint. `bsym` (set in the constructor,
`fem/darcy/darcyhybridization.hpp:690-696`) chooses the sign convention, and
`±Bᵀ`/`∓B` follow from it.

Static condensation eliminates `u` and `p` **element by element** — the (0,0)
and (1,1) blocks are block diagonal once the spaces are broken — leaving a
system in `λ` alone (`fem/darcy/darcyhybridization.hpp:78-92`):

```
                 | Mu ±Bᵀ |-1 | Cᵀ |
H  ←  H - [C G]  | B   D  |   | E  |
                 | Mu ±Bᵀ |-1 | bu |
br ←  br - [C G] | B   D  |   | bp |
```

Write `M` for that 2×2 local block on one element. The reduced operator is a
Schur complement `H − [C G] M⁻¹ [Cᵀ; E]`.

**Where the nonlinearity sits.** Any of five integrator slots can be
nonlinear, and `IsNonlinear()` is their disjunction
(`fem/darcy/darcyhybridization.hpp:557`):

| slot | member | set by |
|---|---|---|
| flux mass | `m_nlfi_u` | `SetFluxMassNonlinearIntegrator()` |
| potential mass | `m_nlfi_p` | `SetPotMassNonlinearIntegrator()` |
| coupled flux+potential | `m_nlfi` | `SetBlockNonlinearIntegrator()` |
| potential constraint | `c_nlfi_p` | `SetConstraintIntegrators()` |
| coupled constraint | `c_nlfi` | `SetConstraintIntegrators()` |

`B` is always linear — it is the divergence form — so the (1,0) block of the
local Jacobian is always `Bf_data`
(`fem/darcy/darcyhybridization.cpp:3183-3185`). The (0,1) block is **not**
always `±Bᵀ`: for a flux law `q = D(p) u` the flux residual depends on the
potential, and the extra term `d(flux residual)/dp` is assembled into
`Bnl_data` (`fem/darcy/darcyhybridization.hpp:319-338`). That block is the
subject of §5.4.

If nothing is nonlinear, `Finalize()` assembles `H` once
(`fem/darcy/darcyhybridization.cpp:2333-2339`) and both `Mult()` and
`GetGradient()` return it immediately
(`fem/darcy/darcyhybridization.cpp:1757-1761`, `1779`). **The ordering is inert
on a linear problem.**

`Finalize()` also classifies the local problem into a `LocalOpType`
(`fem/darcy/darcyhybridization.hpp:455-456`,
`fem/darcy/darcyhybridization.cpp:2340-2374`): `PotNL` when only the potential
is nonlinear (`A` is inverted up front), `FluxNL` when only the flux mass is
(`D` is inverted up front), `FullNL` otherwise. §7 records a hole in the
`FluxNL` case.

---

## 2. `CondenseThenLinearise` — the default

The unknown handed to the outer solver is the trace alone, and the *reduced
residual* is a genuinely nonlinear function of it: eliminating `u` and `p` on
an element means solving that element's nonlinear equations.

### One call to `Mult(x, y)`

`DarcyHybridization::Mult` (`fem/darcy/darcyhybridization.cpp:1753-1772`)
delegates to `MultNL(MultNlMode::Mult, darcy_rhs, x, y)` and then zeroes the
essential trace rows. Per element, `MultNL`
(`fem/darcy/darcyhybridization.cpp:1819-2214`):

1. loads `bu_l`, `bp_l` from the stored right-hand side `darcy_rhs`, negating
   `bp_l` when `bsym` (`:1896-1918`);
2. gathers the element's trace block `x_l` face by face (`:1940-1946`);
3. forms `bu_l − Cᵀ x_l` and `bp_l − E x_l` (`:1947-1968`);
4. **runs a local nonlinear solve**: `MultInvNL(el, bu_l, bp_l, x_l, u_l, p_l)`
   (`:2034`), starting from `darcy_u`/`darcy_p` if those were filled by the
   essential-BC elimination, otherwise from zero (`:2004-2029`);
5. accumulates the trace row `Cᵀᵀ u_l + G p_l + H x_l` plus any nonlinear face
   contribution into `y` (`:2070-2196`).

`MultInvNL` (`fem/darcy/darcyhybridization.cpp:2621-2785`) builds a
`LocalNLOperator` (or `LocalFluxNLOperator` / `LocalPotNLOperator` depending on
`lop_type`), builds an `LBFGSSolver`, `LBBSolver` or `NewtonSolver` per
`SetLocalNLSolver()`, solves, and adds the iteration count to
`num_local_nl_iters` (`:2764`). **This happens once per element per residual
evaluation of the outer solver.** The local operator reads the *linear* form
data `Af_lin_data` / `Df_lin_data`, not the Jacobian storage
(`fem/darcy/darcyhybridization.cpp:4448`, `:4465`), which is why a residual
evaluation is unaffected by whatever the last `GetGradient()` wrote.

### One call to `GetGradient(x)`

`GetGradient` (`fem/darcy/darcyhybridization.cpp:1775-1817`) allocates
`D`/`E`/`G`/`H` if needed, runs `MultNL(MultNlMode::Grad, ...)` — which repeats
steps 1–4 above (another local nonlinear solve) and then calls
`ConstructGrad(el, faces, x_l, u_l, p_l)` (`:2055`) to assemble the local
Jacobian at the converged local fields — and then `ComputeH(Gradient)` to
factor `A`, form and factor the Schur complement, and assemble the reduced
matrix (`fem/darcy/darcyhybridization.cpp:1485-1613`). Essential trace rows get
a unit diagonal via `EliminateRow` (`:1802-1805`).

So the reduced operator really is `F(λ)` with `F` a nonlinear function of the
trace, and `GetGradient` really is `dF/dλ` — obtained by differentiating
through the converged local solve.

---

## 3. `LineariseThenCondense` — new, off by default

This is Newton on the full `(q, u, λ)` system with the resulting *linear*
system hybridized: Nguyen, Peraire & Cockburn, JCP 228 (2009) 8841–8855,
eqs (14)–(18) (`fem/darcy/darcyhybridization.hpp:197-203`). No element ever
runs a nonlinear solve; every local operation is a linear solve.

The mechanism is **not** the plan's. The plan wanted an accumulated iterate,
which needs a "step accepted" signal, which means
`NewtonSolver::ProcessNewState` — a virtual on the *solver* — and `KINSolver`
never calls it. Instead the hybridization retains a **linearisation point** and
refreshes it only in `GetGradient()`:

```
mutable Vector lin_trace, lin_u, lin_p;        // hpp:376-388
mutable Vector lin_u_next, lin_p_next;         // hpp:389-390 (scratch)
mutable bool   lin_valid{false};               // hpp:391
```

with the *factored local Jacobian* left in place in the ordinary block storage
`Af_data`, `Df_data` (holding the factored Schur complement) and `Bnl_data`.
The local residual at the linearisation point is **deliberately not retained**
(`fem/darcy/darcyhybridization.hpp:383-386`,
`fem/darcy/darcyhybridization.cpp:3085-3089`); §7 notes that the doxygen of
`SetNonlinearOrdering()` still says otherwise.

### One call to `Mult(x, y)`

Same `MultNL` walk as above, except that step 4 becomes
`MultInvLin(el, faces, x_l, bu_l, bp_l, u_l, p_l, 1)`
(`fem/darcy/darcyhybridization.cpp:1999-2001`), which is a **linear** solve
plus **exactly one** frozen-Jacobian correction
(`fem/darcy/darcyhybridization.cpp:2973-3081`):

1. form `−[Cᵀ; E](x − lin_trace)` — *and nothing else*; the linearisation's own
   residual must not be applied here (`:2983-3011`);
2. `MultInv(el, ..., with_bnl = true)` for the increment `(du, dp)`, using the
   stored factors of `A` and of the Schur complement (`:3026`);
3. `u_l = lin_u + du`, `p_l = lin_p + dp` (`:3033-3037`);
4. one local Newton correction: evaluate `LocalResidual` at those fields, solve
   `M δ = −r`, add (`:3072-3081`).

Step 4 is the whole method. Without it the fields solve the linearised local
equations exactly, the trace row is linear in the fields, so `F` is **affine in
the trace**: the outer Newton lands on its root in one step and stops, at the
solution of the first linearisation. It reads as convergence
(`fem/darcy/darcyhybridization.cpp:3038-3053`). The term is the
`−[C' E'] M⁻¹ r_local` of NPC eq (18), applied to the fields rather than to the
right-hand side.

**A claim that used to stand here is withdrawn.** It said this must be the
*only* correction when the operator is evaluated, because with two
`d(residual)/dx` picks up `M⁻¹ (J(fields) − M) M⁻¹ [C; E]`. The algebra is
right and the conclusion was inferred, not measured; measured, zero, one, two
and three corrections in an evaluation agree to four digits, and the quantity
that actually governs the gradient is how completely the *retained* fields
solve the local problem — see the next section. What must not happen is the
linearisation's own residual being applied in step 1 as well as in step 4,
which is a different error and is the one that was fixed.

### One call to `GetGradient(x)`

Two cases, decided by `LinearisedAt(x)` — a **bitwise** `memcmp` of the trace
against `lin_trace` (`fem/darcy/darcyhybridization.cpp:2943-2948`):

* **already there** (`relinearise == false`): the retained fields are used as
  they stand, without substitution
  (`fem/darcy/darcyhybridization.cpp:1974-1988`). Taking a substituted step
  here would apply a local Newton step per `GetGradient()` call, and on a stiff
  local problem those unglobalised steps run away — "the residual at one trace
  grew from 1.9e+01 to 4.2e+03 between two calls"
  (`fem/darcy/darcyhybridization.cpp:1983-1985`).
* **advancing**: `MultInvLin(..., corrections = -1)`
  (`fem/darcy/darcyhybridization.cpp:1999-2001`), which **iterates** the
  correction to the tolerance `SetLocalNLSolver()` carries, capped by its
  iteration count, keeping the best iterate and stopping if the frozen-Jacobian
  step diverges.

  This was a fixed two, and **that count was the accuracy of the gradient**.
  `GetGradient()` is the Schur complement of the Jacobian at the retained
  fields, so it is the derivative of `Mult()` only as far as those fields solve
  the local problem; what is left over enters as `d(trace row)/d(fields)`
  evaluated at the wrong point. Against a central difference on a stiff
  pedestal source, and independent of the difference step across four decades:

  | corrections | `σ² = 0.05` | `0.02` | `0.01` |
  |---|---|---|---|
  | 1 | 1.98e-05 | 1.48e-04 | 1.13e-03 |
  | **2** (what shipped) | **2.99e-06** | **2.66e-05** | **3.06e-04** |
  | 4 | 5.63e-08 | 1.29e-06 | 6.16e-05 |
  | 8 | 2.44e-11 | 6.25e-09 | 4.14e-06 |
  | 16 | 1.22e-11 | 1.30e-11 | 3.56e-08 |

  against 1e-10 or better for `CondenseThenLinearise` throughout. The guard is
  not decoration: eight steps taken blindly at `σ² = 0.005` drove the same
  measurement to 1.6e+27.

Either way `Relinearise(el, ...)`
(`fem/darcy/darcyhybridization.cpp:3083-3104`) writes the fields into the
`lin_*_next` scratch and calls `ConstructGrad()`; the point is committed only
after every element has been through, because in a conforming flux space a
later element's dofs overlap an earlier one's
(`fem/darcy/darcyhybridization.cpp:2198-2213`). That commit is also the only
place `evals_since_advance` resets (`:2208`).

### The cold start

`lin_valid` is false until the first `GetGradient()`, but `NewtonSolver` asks
for a residual first. So a `Mult` or `Sol` pass with no linearisation runs the
gradient pass **twice** inside itself before returning anything
(`fem/darcy/darcyhybridization.cpp:1852-1888`): once to get factors at the
caller's raw initial guess, once more to correct the fields and relinearise
there. The measurement that forced the second pass, gradient against a central
difference at a cold linearisation, `(c p², w)` at `c = 100`: **3.2e-03 without
it, 1.1e-11 with it, against 8.9e-12 for the other ordering**
(`fem/darcy/darcyhybridization.cpp:1881-1883`). It costs one extra local
assembly and factorisation per *solve*, not per iteration.

---

## 4. Sizes

Notation, all read off the code rather than assumed:

| symbol | meaning | source |
|---|---|---|
| `NE` | elements | `fes.GetNE()` |
| `NF` | faces (`Mesh::GetNumFaces()`) | `cpp:1140` |
| `N_u` | flux L-dofs, `fes.GetVSize()` | `cpp:3092` |
| `N_p` | potential L-dofs, `fes_p.GetVSize()` | `cpp:3093` |
| `N_λ` | trace L-dofs, `c_fes.GetVSize()` | `fem/hybridization.cpp:33` |
| `a_e` | **free** flux dofs of element `e` | `Af_f_offsets[e+1]-Af_f_offsets[e]`, `cpp:189-202` |
| `h_e` | *all* flux vdofs of element `e` | `hat_offsets[e+1]-hat_offsets[e]`, `cpp:120-131` |
| `d_e` | potential dofs, `fes_p.GetFE(e)->GetDof() * fes_p.GetVDim()` | `cpp:228` |
| `c_f` | trace dofs of face `f`, `c_fes.GetFaceElement(f)->GetDof() * c_fes.GetVDim()` | `cpp:1009` |

`a_e = h_e` unless flux dofs were declared essential in `Init()`; the
difference `e_e = h_e − a_e` is what the eliminated blocks carry.

### 4.1 Global

| object | size | where |
|---|---|---|
| monolithic mixed system | `(N_u + N_p) × (N_u + N_p)` | never assembled by this class |
| reduced (trace) system `H` | `N_λ × N_λ`, sparse | `ComputeH()`, `cpp:1502` |
| `Operator::Height()` of `DarcyHybridization` | `N_λ` | `fem/hybridization.cpp:33` |
| `Operator::Height()` of `ParOperator` | `c_fes.GetTrueVSize()` | `hpp:434` |
| reduced RHS `b_r` | conforming/true trace size | `ReduceRHS()`, `cpp:3469-3494` |

Hybridization pays off when `N_λ < N_u + N_p`, which is the usual HDG
situation: `N_λ` counts trace dofs on `NF` faces while `N_u + N_p` counts
`neq·(dim+1)` volume fields. The class does not check this, and the base-class
doxygen says as much (`fem/hybridization.hpp:47-49`).

### 4.2 Per element and per face

Flat arrays, all `Array<real_t>`, indexed by prefix-sum offset arrays. Note
that `Bnl_data` is indexed by `Bf_offsets` but read in the **transposed**
orientation (`fem/darcy/darcyhybridization.hpp:329-331`), and that `G_data`
shares `E_data`'s offsets by reference (`hpp:356`).

| data | offsets | block shape | count | note |
|---|---|---|---|---|
| `Af_data` | `Af_offsets` | `a_e × a_e` | `Σ a_e²` | `Mu` block, overwritten by the Jacobian's (0,0) and then LU-factored in place |
| `Af_ipiv` | `Af_f_offsets` | `a_e` | `Σ a_e` | pivots |
| `Af_lin_data` | `Af_offsets` | `a_e × a_e` | `Σ a_e²` | backup of the **linear** flux mass |
| `Ae_data` | `Ae_offsets` | `e_e × h_e` | `Σ e_e·h_e` | eliminated flux rows |
| `Bf_data` | `Bf_offsets` | `d_e × a_e` | `Σ a_e·d_e` | the linear divergence form |
| `Bnl_data` | `Bf_offsets` | `a_e × d_e` | `Σ a_e·d_e` | `d(flux residual)/dp`; empty unless a flux law depends on `p` |
| `Be_data` | `Be_offsets` | `e_e × d_e` | `Σ e_e·d_e` | eliminated divergence rows |
| `Df_data` | `Df_offsets` | `d_e × d_e` | `Σ d_e²` | `D`, then the **Schur complement**, LU-factored in place |
| `Df_ipiv` | `Df_f_offsets` | `d_e` | `Σ d_e` | pivots |
| `Df_lin_data` | `Df_offsets` | `d_e × d_e` | `Σ d_e²` | backup of the **linear** potential mass |
| `Ct_data` | `Ct_offsets` | `a_{el1} × c_f` and `a_{el2} × c_f` | `Σ_f c_f(a_{el1}+a_{el2})` | `GetCtFaceMatrix(f, side, ·)`, `cpp:1647-1667` |
| `E_data` | `E_offsets` | `d_{el} × c_f` per side | `Σ_f c_f(d_{el1}+d_{el2})` | `cpp:1669-1688` |
| `G_data` | `E_offsets` (alias) | `c_f × d_{el}` per side | same | `cpp:1690-1709` |
| `H_data` | `H_offsets` | `c_f × c_f` | `Σ_f c_f²` | face-diagonal only; `cpp:1711-1716` |

The local Jacobian `M` of §3 is therefore never a separate object: it is
`Af_data` (0,0), `Bf_data` (1,0), `±Bf_dataᵀ + Bnl_data` (0,1) and `Df_data`
(1,1), the last two folded into the Schur complement by `ComputeElementH()`
(`fem/darcy/darcyhybridization.cpp:1292-1356`). Its per-element size is
`(a_e + d_e)²` conceptually and `a_e² + a_e d_e + d_e²` in storage.

An element's block of the reduced matrix is `T_e × T_e` with
`T_e = Σ_{f ∈ ∂e} c_f` (`GetElementTraceSize()`, `cpp:1265-1273`); in
`AssemblyMode::Threaded` a chunk of those blocks is buffered before the serial
scatter (`cpp:1531-1572`).

### 4.3 What `LineariseThenCondense` adds

| object | size | where |
|---|---|---|
| `lin_trace` | `N_λ` | assigned in `MultNL`, `cpp:2209` |
| `lin_u`, `lin_u_next` | `N_u` each | `cpp:3092`, `3100` |
| `lin_p`, `lin_p_next` | `N_p` each | `cpp:3093`, `3101` |

Total `N_λ + 2(N_u + N_p)` reals. Against the base scheme's `Σ a_e²` +
`Σ d_e²` + `Σ a_e d_e` — which is `O(p^{2d})` per element against the
linearisation point's `O(p^d)` — this is small: for order `k` in `d`
dimensions it is smaller by roughly the local dof count. **It adds no second
copy of the factored Jacobian**; the factors live in the same `Af_data` /
`Df_data` the other ordering uses.

The `_next` duplicates exist because the commit must be atomic across the
element loop (`cpp:3095-3099`). The extra *work* per element is the correction
loop on a `Grad` pass, which iterates to `SetLocalNLSolver()`'s tolerance
rather than taking a fixed count — each step a residual evaluation and a
back-substitution against the one factorisation already held.

**It is not free, and the count is measured rather than assumed**: 4.2 to 12.1
corrections per element per linearisation across the pedestal cases, against
the fixed two it replaced. End to end that puts this ordering at 0.9 s against
condense-first's 0.7 s on `n = 24, σ² = 0.02`, 1.3 s against 0.8 s at
`σ² = 0.005`, and level at 1.7 s on `n = 32, σ² = 0.003` — the case it used not
to solve at all. So the pitch for this ordering is not speed on a stiff
problem; `MultInvNL`'s nonlinear iteration disappears, but the corrections that
replace it are of the same order. What it buys is a local problem that is
always a linear solve, which is the uniform, batchable workload
`HDG-ELEMENT-LOCAL-PARALLELISM.md` wants.

---

## 5. What changes from standard MFEM assumptions

### 5.1 Is `Mult` a function of its argument?

**`CondenseThenLinearise`: yes**, up to the local solver's tolerance. The
local initial guess is fixed (`darcy_u`/`darcy_p`, written once by the
essential-BC elimination at `cpp:2394-2395` and `cpp:2491-2492`, otherwise
zero), the local operator reads only the linear form data, and nothing a
`GetGradient()` writes is read back. The caveat is accuracy, not purity: the
default local tolerance is `rtol = 1e-6`
(`fem/darcy/darcyhybridization.hpp:716`), so the reduced residual is only
correct to that, and the tests that use this ordering as a round-off control
tighten it explicitly to `1e-14`/`1e-30`
(`tests/unit/fem/test_darcy_linearise_first.cpp:485-489`). A local solve that
fails to converge prints to `mfem::out` and continues
(`fem/darcy/darcyhybridization.cpp:2773-2781`).

**`LineariseThenCondense`: only conditionally.** Precisely:

* **At a trace the linearisation is already at, `Mult` is exact and
  repeatable — bit for bit.** `GetGradient()` at that trace does not move the
  fields, so `Mult`, `GetGradient`, `Mult`, `GetGradient`, `Mult` returns the
  identical vector three times. Pinned by "The reduced operator is a function
  of the trace", which requires `BitwiseEqual`
  (`tests/unit/fem/test_darcy_linearise_first.cpp:414-423`) at
  `c = 1, 1e2, 1e4, 1e5`.
* **Across a linearisation that *advances* onto the trace, it is not.** This is
  every Newton step after the first: the retained fields move, and the residual
  is evaluated at fields substituted from them. Note what this does *not* mean
  any more. `Mult()` now performs the advance itself when the linearisation is
  not at its argument (§5.2), so the residual and the gradient an outer solver
  obtains at one iterate are taken about the *same* linearisation; the
  impurity below is a dependence on the path taken to reach a trace, not the
  mismatch between residual and Jacobian that used to break `NewtonSolver`. The doxygen states the measured
  size: *"The gap was measured at 5.0e-10, 4.8e-06 and 1.1e-02 as the
  nonlinearity grew, and it cannot be closed within this ordering: exactness
  there needs the local problem solved exactly, which is
  `CondenseThenLinearise`"* (`fem/darcy/darcyhybridization.hpp:760-767`).

  The test that pins it, "The reduced residual survives the linearisation
  advancing", does `Mult(x0)`, `GetGradient(x0)`, `Mult(x1)`,
  `GetGradient(x1)`, `Mult(x1)` and requires the two `x1` residuals to agree to
  `1e-7` relative
  (`tests/unit/fem/test_darcy_linearise_first.cpp:641-653`). It generates
  `c = 1` and `c = 10` only, so it pins the `5.0e-10` end; the `4.8e-06` and
  `1.1e-02` figures are outside its range (see §7).

  That the gap is *second order* rather than first is itself load-bearing, and
  it holds only because the linearisation is formed with a correction applied,
  including the very first one: *"retaining the caller's raw initial guess
  instead put this at 3.3e-05 rather than 5.0e-10"*
  (`tests/unit/fem/test_darcy_linearise_first.cpp:619-624`).

* `GetGradient()` **is** the derivative of `Mult()` at the retained trace, to
  round-off. "The reduced gradient is the derivative of the reduced residual"
  compares it against a central difference with the essential rows masked and
  requires `< 1e-8` at `c` up to `1e3`, in both gradient modes and for both
  orderings (`tests/unit/fem/test_darcy_linearise_first.cpp:505-601`). With the
  double-application defect present, that comparison read **3.2e-03 at
  `c = 100`, independent of `h` across four decades** — which is what
  identified it as a Jacobian error rather than a differencing artefact
  (`tests/unit/fem/test_darcy_linearise_first.cpp:522-527`).

### 5.2 There is no contract: `Mult()` linearises at its own argument

**This section previously described a requirement on the solver, and that
requirement is gone.** It is worth stating what it was, because the change is
recent and because `meq` and this document were both written against the old
behaviour.

The linearisation used to advance only in `GetGradient()`, so an outer
iteration had to ask for a gradient once per accepted iterate; a solver that
did not had to call `AdvanceLinearisation(trace)` by hand, and
`SetMaxEvalsWithoutAdvance()` guarded the requirement by aborting past a count
of residual evaluations without an advance. All three methods, the counter
behind them and its accessor have been **removed**, not deprecated.

What replaced them is one condition. `MultNL()` establishes a linearisation
inside `Mult()` when there is not one **at the argument** — `!LinearisedAt(x)`
— where it used to ask only whether there was one anywhere. The predicate was
already present and already used by the gradient path
(`fem/darcy/darcyhybridization.cpp:1905-1909`).

That also repaired `NewtonSolver`, which had not been able to take a Newton
step in this mode: it evaluates the residual before it asks for the gradient,
on every iterate and not only the first, so it read `r` at `x_k` about the
linearisation retained at `x_{k-1}` and then `J` at `x_k`. On a stiff
semilinear pedestal, three of seven benchmark configurations that converge
under `CondenseThenLinearise` did not converge in sixty iterations under this
ordering, landing at traces of norm 24.0, 26.5 and 54.8 against true values of
11.3, 14.7 and 13.5. Two of the three now converge, in 9 and 8 iterations
against the exact ordering's 8 and 10; every case that converged before now
matches or beats it; one remains, stalling at 1.7e-03 where it used to diverge
to 2.0e+03.

It costs nothing in a plain Newton loop — the advance happens in `Mult()`
instead of `GetGradient()`, which then finds the linearisation already at `x`
and reuses it, so it is one advance per iterate either way. A line search pays
one advance per trial point, which is the price of the trial residual being
the residual.

The test that used to pin the obligation now pins its absence: a hand-rolled
JFNK loop that never asks for a gradient reaches the reference to **2.45e-15**,
where it previously required the residual below 1e-11 *and the answer wrong by
more than 1e-7*.

### 5.3 Consequences for solvers

| solver | `CondenseThenLinearise` | `LineariseThenCondense` |
|---|---|---|
| `NewtonSolver` | works | **works, and needs to know nothing.** It did not, until `Mult()` began linearising at its own argument -- see §5.2 |
| Newton + line search | works | works — between two `GetGradient()` calls `F` is an ordinary function of the trace (`hpp:753-757`) |
| lagged Jacobian | works | works, same reason |
| `LBFGSSolver`, `LBBSolver` | works | works. Previously **unsound** -- never asks for a gradient, so the linearisation never advanced -- and the miniapps carried a warning, now withdrawn to a comment beside the code |
| `KINSolver`, matrix-based | works | works, via `KINSolver::LinSysSetup` → `oper->GetGradient()` (`linalg/sundials.cpp:1945-1955`) — but see below |
| `KINSolver::SetJFNK(true)` | works | works. Previously needed `SetMaxSetupCalls(1)` and a registered preconditioner against KINSOL's default of ten, on which *"the residual falls to 2e-15, the iteration reports convergence, and the answer is wrong in the fifth digit"* |
| any JFNK / matrix-free outer solve | works | works, and needs to know nothing |

`SetLocalNLSolver()`'s **iteration cap and tolerances are no longer inert**
under `LineariseThenCondense`: they bound the correction loop that forms the
linearisation point, and they set the gradient's accuracy. The solver *type*
and `SetLocalNLPreconditioner()` remain inert — the correction is a Newton step
on factors `M` already holds, so there is nothing to choose — and `MultInvNL`
is still never reached (`fem/darcy/darcyhybridization.cpp:2031-2036`).
`GetNumLocalNLIterations()` staying at zero is still the acceptance signal that
the ordering changed (`fem/darcy/darcyhybridization.hpp:891-898`); it counts
local *nonlinear solves*, and there are none — every local step is a solve with
one factorisation.

### 5.4 `MultInv(..., with_bnl)` and the gradient modes

`MultInv()` applies `M⁻¹` with the stored factors
(`fem/darcy/darcyhybridization.cpp:3106-3161`). Its `with_bnl` argument decides
whether the (0,1) block is the Jacobian's — `∓Bᵀ` **plus** `Bnl_data` — or the
linear `∓Bᵀ` alone (`fem/darcy/darcyhybridization.hpp:664-672`). The Schur
complement in `Df_data` must have been built the same way, which is what
`ComputeH(ComputeHMode::Gradient)` does
(`fem/darcy/darcyhybridization.cpp:1319-1327`).

Current callers:

| call site | `with_bnl` |
|---|---|
| `MultNL(GradMult)` — the matrix-free gradient application, `cpp:2067` | `true` |
| `MultInvLin` affine step, `cpp:3026` | `true` |
| `MultInvLin` local correction, `cpp:3077` | `true` |
| `ReduceRHS` (linear path), `cpp:3562` | default `false` — correct, `Bnl` is empty |
| `ComputeSolution` (linear path), `cpp:3772` | default `false` — same |

**`MFEM_DARCY_HYBRIDIZATION_GRAD_MAT` no longer exists.** It was unconditionally
defined, so its non-default path was never built; commit `c849adffd5` deleted
it in favour of a run-time choice,
`SetGradientMode(GradientMode::Assembled | MatrixFree)`, defaulting to
`Assembled` (`fem/darcy/darcyhybridization.hpp:228-251`, `:823-837`). That
commit also closed the gap the plan document still describes:
`ConstructGrad()` used to carry its own factorisation, reached only when the
macro was off, built from the linear `∓Bᵀ` alone — so the two modes were
different operators whenever the flux law depended on the potential. It is
deleted (`fem/darcy/darcyhybridization.cpp:3322-3328`) and both modes now take
`ComputeH()`'s first half. **So there is no longer any refusal of the
matrix-free gradient by `LineariseThenCondense`, and the gap is closed for
`CondenseThenLinearise` too** — the tests sweep `GradientMode` against both
orderings (`tests/unit/fem/test_darcy_linearise_first.cpp:531-537`,
`:624-627`, `:882-886`).

What `MatrixFree` costs, in exchange: `GetGradient()` returns an `Operator`
with no matrix, so `GSSmoother`, `UMFPackSolver` and the algebraic
preconditioners abort (`fem/darcy/darcyhybridization.hpp:827-831`). It is also
**refused for `LocalOpType::FluxNL`** — `Df_data` there holds the factored
linear potential mass, so there is nothing for `MultInv()` to read back
(`fem/darcy/darcyhybridization.cpp:1342-1347`). And two things the assembled
matrix gets for free have to be applied by hand: the unit row on essential
trace dofs, and `SetDiagIdentity()`'s regularisation of rows nothing
contributed to (`fem/darcy/darcyhybridization.cpp:4180-4205`, and
`MarkEmptyTraceRows()` at `:2787-2851`). Without the latter the two modes
differed by **0.497 on 64 of 160 trace rows** on a problem whose constraint has
no boundary face term (commit `c849adffd5`); the case is
`tests/unit/fem/test_darcy_linearise_first.cpp:761-798`.

### 5.5 Other places the default expectation is wrong

These apply to the nonlinear hybridized operator generally; the first two bite
harder under `LineariseThenCondense`.

1. **The reduced right-hand side is zero, and the load lives inside the
   operator.** For a nonlinear problem `ReduceRHS()` stores the mixed RHS in
   `darcy_rhs` and returns a zeroed `b_r`
   (`fem/darcy/darcyhybridization.cpp:3445-3495`); `Mult()` returns the full
   residual, load included. `NewtonSolver::Mult(B, X)` then subtracts a zero
   `B`. `EliminateTraceTrueDofsInRHS()` for the nonlinear case only zeroes the
   essential rows (`fem/darcy/darcyhybridization.cpp:2587-2599`). A caller
   reasoning "`A x = b` with the `b` I was handed" will be surprised.
2. **`Mult()` is `const` and mutates a great deal**, including — on the first
   call under `LineariseThenCondense` — running two full gradient passes and
   creating the linearisation point
   (`fem/darcy/darcyhybridization.cpp:1852-1888`). The first residual
   evaluation of a solve is therefore far more expensive than the rest, and it
   is where the linearisation is born.
3. **`GetGradient()` overwrites the local block storage.** After it,
   `Af_data`/`Df_data` hold the last Jacobian and its Schur complement, not the
   forms. Under `CondenseThenLinearise` this is invisible (the residual reads
   `Af_lin_data`/`Df_lin_data`); under `LineariseThenCondense` it is the
   mechanism.
4. **The returned gradient does not outlive the next call.** `Assembled` resets
   `Grad` each time (`cpp:1793`); `MatrixFree` returns a `Gradient` that reads
   whatever is currently in `Af_data`/`Df_data`, so holding a reference across
   a later `GetGradient()` silently applies the *new* operator.
5. **Essential trace dofs are carried the `NonlinearForm` way, not the
   `BilinearForm` way**: the values ride in `x`, the residual is zeroed on those
   rows, the Jacobian gets a unit row, and only rows (not columns) are
   eliminated (`fem/darcy/darcyhybridization.cpp:1766-1772`, `:1802-1805`).
   Any finite-difference check of the gradient must mask them or it is
   meaningless — the accessor `GetEssentialTrueDofs()` exists for exactly that
   and says so (`fem/darcy/darcyhybridization.hpp:1098-1104`).
6. **`ComputeSolution()` does not solve the local problem exactly under
   `LineariseThenCondense`.** It runs `MultNL(MultNlMode::Sol, ...)`
   (`fem/darcy/darcyhybridization.cpp:3666-3673`), which for this ordering
   substitutes from the linearisation and takes **one** correction, not a local
   nonlinear solve. At convergence that agrees with the other ordering to
   `1e-9` relative on the potential
   (`tests/unit/fem/test_darcy_linearise_first.cpp:298-313`), but it is a
   different computation.
7. **`SetNonlinearOrdering()` discards the linearisation** when the ordering
   actually changes (`fem/darcy/darcyhybridization.cpp:2906-2919`), so it is
   safe to call mid-life; it is a no-op when the ordering is unchanged.

---

## 6. When to use which

`CondenseThenLinearise` is the default and every caller written before this
gets it (`fem/darcy/darcyhybridization.hpp:736-738`). Reach for
`LineariseThenCondense` when the local nonlinear solves are the cost, or when
they are the thing that fails.

**The measured comparison.** A two-equation nonlinear DG system, hybridized,
4×4 quadrilaterals, outer Newton residual per iteration. These numbers are from
the commit message of `2e1752717f` ("An ordering that condenses the Jacobian
instead of eliminating nonlinearly") — see §7, they are *not* in any comment or
test:

| | residual history |
|---|---|
| order 0, condense-then-linearise | 1.11e-01 1.52e-04 5.56e-10 1.01e-16 |
| order 0, linearise-then-condense | 1.11e-01 1.52e-04 3.91e-07 7.61e-17 |
| order 1, condense-then-linearise | 7.38e-02 2.24e-05 4.00e-12 7.02e-17 |
| order 1, linearise-then-condense | 7.38e-02 2.24e-05 9.08e-09 7.39e-17 |

Same step count and the same first two residuals — the first Newton step of the
full system is the same either way. **Local nonlinear iterations: 192 and 304
in the old ordering, zero in the new one.** The two solutions agree to
round-off: trace 4.5e-16, potential 7.4e-16, flux 4.7e-16; on a linear problem
both land in one step and agree to 3.4e-16.

What the tests in the tree assert about the same comparison
(`tests/unit/fem/test_darcy_linearise_first.cpp:265-347`, orders 0 and 1):

* a linear problem is solved in one step either way, and the potentials agree
  to `1e-10` relative (`:277-296`);
* a nonlinear problem reaches the same discrete solution, `1e-9` relative
  (`:298-313`);
* `GetNumLocalNLIterations()` is `> 0` for the old ordering and **exactly 0**
  for the new one (`:315-325`);
* the outer iteration is still quadratic, `r₁ < 100 r₀²` (`:327-347`).

**Choose `CondenseThenLinearise` when** you need `Mult` to be an exact function
of the trace across advancing linearisations (see §5.1), or when the local
problem is where the physics is and you want it solved to a tolerance you
control. The old third reason -- that the outer solver would not ask for a
gradient every iterate -- is gone; no solver has to.

**Choose `LineariseThenCondense` when** the per-element nonlinear solves
dominate, when they fail to converge and poison the outer iteration, or when
you want the method as NPC defines it. Any outer solver will do, LBFGS and LBB
included; the advice to avoid them belonged to the removed contract.

**The exact ordering is still more solvable, though by less than it was.**
The hard-coded correction count was measured and is gone, and all seven of the
caller's benchmark configurations now converge. On a wider sweep of the same
source -- 144 configurations, n = 8..24, k = 1..3, six widths from 0.02 to
0.001 -- the cases where `CondenseThenLinearise` converges and this ordering
does not went from six to three, with none added, and where both converge this
ordering took fewer iterations in 15 and more in 10. The three that remain are
widths where the frozen-Jacobian correction cannot converge at all; closing
them needs the local step globalised, or the local problem solved exactly,
which is the other ordering.

**From the caller's side**: `-lfirst` on `convdiff`
(`miniapps/hdg/convdiff.cpp:258-264`), `pconvdiff`
(`miniapps/hdg/pconvdiff.cpp:262-267`) and `navierstokes`
(`miniapps/hdg/navierstokes.cpp:450-452`); `-gm 0|1|2` selects the gradient
mode and the matching preconditioner (`miniapps/hdg/convdiff.cpp:274-280`,
`miniapps/hdg/darcyop.hpp:296-309`). A rate or accuracy study wants `-gm 0`
with a tight `-rtol`, because the default GS-preconditioned trace solve leaves
the solver error the same size as the discretisation error.

---

## 7. Suspected defects and inconsistencies

Recorded here because they were found while writing this and are the reason
some of the sections above are hedged. They were found by reading; the tree was
mid-build at the time and nothing was run.

**Status, since several have been acted on.** 1, 2 and 6 are **fixed**: the
plan document was rewritten, `SetNonlinearOrdering()`'s doxygen no longer
carries the `- r_lin` it had already stopped implementing, and it no longer
claims the whole non-purity gap is pinned when only its two smaller values are.
4 and 5 are **moot** rather than fixed -- they were about a solver contract
that no longer exists, and the warnings they asked for have been withdrawn from
`convdiff` and `pconvdiff` instead of added to `navierstokes`. 6 is now half
answered, as its own entry records. 3, 7, 8, 9 and 10 are **open**, and 3 and 8
are the two worth acting on: a wrong Schur complement read silently, and an
accessor that returns the wrong member.

1. **The plan document is stale in three ways.**
   `doc/HDG-LINEARISE-THEN-CONDENSE.md:12-18` says the matrix-free gradient is
   "still wrong for the *other* ordering", that `MFEM_DARCY_HYBRIDIZATION_GRAD_MAT`
   selects it, and that `LineariseThenCondense` "refuses that combination". The
   macro was deleted by `c849adffd5`; the gap was closed for both orderings in
   the same commit; and no refusal exists in the source. Only `LocalOpType::FluxNL`
   is refused, and by `GradientMode`, not by the ordering.

2. **`SetNonlinearOrdering()`'s doxygen describes the defect that was fixed.**
   It writes the substitution as
   `(q, u)(L) = (q, u)_lin + M⁻¹(−r_lin − [C; E](L − L_lin))` and lists
   "its local residual `r_lin`" among the things "refreshed by `GetGradient()`"
   (`fem/darcy/darcyhybridization.hpp:743-756`). The code deliberately does
   **not** retain `r_lin` (`hpp:383-386`, `cpp:3085-3089`), and applying a
   retained `r_lin` in the prediction is precisely what cost the gradient its
   exactness (`cpp:2983-2991`, `cpp:3054-3066`). The formula is only right if
   `r` is read as "the residual evaluated at the predicted fields", which the
   surrounding prose does not say.

3. **`LineariseThenCondense` has no guard for `LocalOpType::FluxNL`, and the
   diagnostic it does hit names the wrong feature.** In that mode `Df_data`
   holds the factored *linear potential mass* and the Schur complement is built
   into a temporary (`cpp:1329-1355`), while `MultInv()` reads the Schur
   complement out of `Df_data` (`cpp:3120`). Following the code: `NewtonSolver`
   calls `Mult` first, the cold-start block calls
   `ComputeH(GradientFactorOnly)` (`cpp:1872`, `:1891`), and
   `ComputeElementH()` aborts with *"GradientMode::MatrixFree is not
   supported…"* (`cpp:1342`) even though the caller selected the default
   `Assembled`. If a caller instead calls `GetGradient()` first, that abort is
   not reached and `MultInv()` would then use the factored linear potential
   mass as the Schur complement — silently wrong. `FluxNL` is reachable
   whenever only the flux mass is nonlinear *and* a potential mass is present
   (`cpp:2349-2356`); I did not confirm a miniapp flag combination that
   produces it. `GradientMode::MatrixFree` has an explicit refusal for this
   case; the ordering has none.

4. **`navierstokes` accepts `-lfirst` with no soundness warning.**
   `convdiff.cpp:856-871` and `pconvdiff.cpp:854-874` both warn when `-lfirst`
   is combined with LBFGS or LBB; `navierstokes.cpp:755-759` has no such check
   although it offers the same `-nls 1` (`navierstokes.cpp:426-428`).

5. **No miniapp warns for `-nls 4`, and `DarcyOperator` builds exactly the
   lazy-setup case the doxygen names.** `DarcyOperator` constructs
   `KINSolver(KIN_PICARD)` with `EnableAndersonAcc(10)`
   (`miniapps/hdg/darcyop.cpp:79-99`) and never calls `SetMaxSetupCalls()`.
   KINSOL calls `KINSolver::LinSysSetup` — the only route to
   `oper->GetGradient()` (`linalg/sundials.cpp:1945-1955`) — lazily. Whether
   `KIN_PICARD` with Anderson acceleration in fact advances every iterate I did
   not establish; the doxygen's warning
   (`fem/darcy/darcyhybridization.hpp:776-782`) says the default of ten setup
   calls is enough to produce a converged, wrong answer.

6. **The doxygen says the non-purity gap is "pinned by a unit test"; the test
   pins only its mildest value.** The three quoted numbers are 5.0e-10,
   4.8e-06 and 1.1e-02 (`hpp:764-766`); the test generates `c = 1` and
   `c = 10` with a bound of `1e-7`
   (`tests/unit/fem/test_darcy_linearise_first.cpp:624`, bound at `:653`), which the
   4.8e-06 figure would fail. Nothing in the tree pins the two larger values or
   the nonlinearity they correspond to.

   **Half of that is now answered.** The tree does carry the stiff regime, in
   `PedestalHDG` and the two cases that use it — one pinning the reduced
   gradient against a central difference where the fixed correction count put
   it 1e-05 out, one pinning convergence parity on the configuration that used
   to fail. What is still unpinned is the *non-purity gap itself*: those cases
   measure the gradient and the solve, not the residual's dependence on where
   the linearisation was last taken.

7. **The comparison in §6 exists only in a git commit message.** The residual
   histories, the 192/304-to-zero local iteration counts and the 4.5e-16 /
   7.4e-16 / 4.7e-16 agreement are in the message of `2e1752717f` and in no
   comment, doxygen block or test. The project's own rule is that a finding
   which lives only in markdown is thrown away; a finding that lives only in a
   commit message is barely better, since nothing next to the code points at it.

8. **`GetFluxMassNonlinearIntegrator()` returns the potential integrator.**
   `fem/darcy/darcyhybridization.hpp:936-937` — both accessors return
   `m_nlfi_p`; the flux one should return `m_nlfi_u`. The same pair appears at
   `fem/darcy/darcyreduction.hpp:144-145`. Pre-existing, and nothing in the
   tree calls either, so it is latent.

9. **Two unrelated meanings of "lin" in adjacent members.**
   `Af_lin_data` / `Df_lin_data` mean *the linear form's data*
   (`hpp:313`, `:346`); `lin_trace` / `lin_u` / `lin_p` / `lin_valid` mean *the
   linearisation point* (`hpp:388-391`). `ConstructGrad()` reads both senses
   within twenty lines (`cpp:3241-3256`).

10. **Pre-existing and not ordering-specific: the trace prolongation is applied
    inconsistently.** `Operator::Height()` is `c_fes.GetVSize()`
    (`fem/hybridization.cpp:33`), `ReduceRHS()` sizes the reduced RHS to the
    *conforming* width (`cpp:3469-3494`), and serial `Mult()`/`GetGradient()`
    index `x` by face VDofs with no prolongation (`cpp:1940-1946`), while
    `ParMultNL()` does prolong (`cpp:2216-2240`). For a `DG_Interface` trace
    space the conforming prolongation is null and the three agree, which is
    every case in the tree. For an `H1_Trace` (EDG) trace space with a
    nonlinear problem they would not.

11. **Low confidence, probably unreachable:** `ParOperator::GetGradient()`
    passes an empty dummy `Vector y` to `ParMultNL()`
    (`cpp:4238-4240`), which for `!ParallelC()` and a null trace restriction
    does `y.MakeRef(y_t, 0, c_fes.GetVSize())` on that empty vector
    (`cpp:2278-2282`). Nothing subsequently reads or writes `y` on a `Grad`
    pass, so the alias itself is the only hazard, and it needs a `ParOperator`
    over a serial trace space — which `ParDarcyForm` should never build. The
    serial `GetGradient()` avoids it because `MultNL` never touches `y` in
    `Grad` mode.
