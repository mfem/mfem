# Nonlinear `DarcyHybridization`: the reduced operator, and NPC

A technical reference for the two ways this class solves a nonlinear
hybridized system: the **reduced trace operator**, which condenses first and
linearises second and hands the outer solver an `Operator` on the **trace**,
and **NPC**
(`NPCResidual`/`NPCGradient`/`NPCReduce`/`NPCRecover`, wrapped as
`DarcyNPCOperator` + `DarcyNPCSolver`), which runs Newton on the **full**
`(q, u, λ)` system with the Jacobian solved by hybridized elimination.

**A third thing used to be here and is deleted.**
`NLOrdering::LineariseThenCondense` was an `Operator` on the trace alone
whose
local blocks were eliminated by a linear solve against a retained
linearisation, and its comments claimed it was the NPC method of Nguyen,
Peraire & Cockburn, JCP 228 (2009) 8841–8855, eqs (14)–(18). It was not: NPC's
fields are Newton state and a trace-only operator has nowhere to keep them.
Measured, it was *slower* than the plain condensation on stiff problems and
failed four configurations that one solves. Sections 3, 4.3, 5.1 and 5.2 of
this document were about it and are gone with it; what it taught is in the
doxygen of the things that replaced it.

**Citations are stale and are being kept only as landmarks.** The `file:line`
references below were read at commit `50c5d75def` and much has moved since —
the deletion alone removed some 290 lines from `darcyhybridization.cpp`. Treat
them as "look near here for this", not as coordinates. The names are accurate;
the numbers are not.

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

## 2. The reduced trace operator — condense first, linearise second

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

## 3. NPC — Newton on the full system

Not an ordering, and not reached through any mode setter: NPC's unknown is the
whole `(q, u, λ)` vector rather than the trace, so it is a different
`Operator`. One Newton step, per Nguyen, Peraire & Cockburn eqs (14)–(18):

```
assemble M and F_local at x_k          one factorisation per step
S   = H - C' M⁻¹ B_λ
rhs = -(F_λ - C' M⁻¹ F_local)          eq (18)
solve S Δλ = rhs
Δlocal = -M⁻¹ (F_local + B_λ Δλ)
x_{k+1} = x_k + Δx                     all three blocks advance
```

**One local factorisation and one local linear solve per outer step, no local
nonlinear iteration anywhere**, and the convergence test is on the full
residual — which is the half a trace-only operator cannot express.

### 3.1 The wrapper, which is what to use

```cpp
Array<int> offs(4);                    // {0, flux, potential, trace}
offs[0] = 0;
offs[1] = Vh.GetVSize();               // flux, L2 -> L-dofs are true dofs
offs[2] = Wh.GetVSize();               // potential, likewise
offs[3] = Mh.GetTrueVSize();           // trace: TRUE dofs, serial or parallel
offs.PartialSum();

BlockVector load(rhs, darcy.GetOffsets());   // (flux, potential) only
DarcyNPCOperator npc(*darcy.GetHybridization(), offs, load);

UMFPackSolver    trace;                // or CG/GMRES+AMG, or matrix-free
DarcyNPCSolver   lin(trace);

NewtonSolver newton;
newton.SetOperator(npc);
newton.SetSolver(lin);
newton.Mult(zero, x);                  // x is the full (q, u, λ) vector
```

`npc.Height()` is `offs.Last()`; `x` and the residual are that long.
`npc.LocalOffsets()` gives `{0, flux, potential}` for a caller splitting the
local half back out.

`DarcyNPCSolver` takes **any** `Solver` for the reduced trace system and
re-points it at the new `S` on every Newton step. Nothing else is needed:
`NewtonSolver` and `KINSolver` both drive this with no special support,
because the fields are in `x` where every outer solver already keeps its
state.

**`KINSolver` wants `SetMaxSetupCalls(1)`.** KINSOL calls `LinSysSetup` every
tenth step by default, which makes this a lagged-Jacobian Newton — legitimate,
and self-consistent because the reduction and recovery eliminate with whatever
factorisation is currently held, but it costs iterations: 12 against 4 on one
case, both converged to round-off.

**The line search is the caller's.** `NewtonSolver::ComputeScalingFactor` is
virtual, and backtracking on the full residual is what NPC wants — well
defined here precisely because the fields and the trace are one vector and
scale together. A dozen-line subclass converges three stiff configurations
that the deleted trace-only mode could not, in 13, 10 and 17 steps. Nothing
about it is Darcy-specific, so it is not in the library;
`miniapps/hdg/navierstokes.cpp` carries one behind `-ls` as a worked example.

### 3.2 The four raw calls

The wrapper is bookkeeping over these; use them directly to see the shape, or
to drive an iteration `NewtonSolver` cannot express.

```cpp
void      NPCResidual(const BlockVector &b, const BlockVector &x,
                      const Vector &x_tr, BlockVector &r, Vector &r_tr);
Operator &NPCGradient(const BlockVector &x, const Vector &x_tr);
void      NPCReduce  (const BlockVector &r, const Vector &r_tr, Vector &b_tr);
void      NPCRecover (const BlockVector &r, const Vector &dtr, BlockVector &dx);
```

`b`, `x`, `r` and `dx` are two-block `(flux, potential)`; `x_tr`, `r_tr`,
`b_tr` and `dtr` are trace vectors in **true dofs**. One step is

```cpp
dh.NPCResidual(b, x, x_tr, r, r_tr);        // F(q, u, λ)
Operator &S = dh.NPCGradient(x, x_tr);      // assemble + factor J
dh.NPCReduce(r, r_tr, b_tr);                // eq (18) right-hand side
solver.Mult(b_tr, dtr);                     // your trace solve
dh.NPCRecover(r, dtr, dx);
x += dx;  x_tr += dtr;
```

**Call `NPCGradient()` before `NPCReduce()` and `NPCRecover()`** — both need
the factored local blocks and the Schur complement it leaves behind, and both
apply the Jacobian's `(0,1)` block rather than the linear one.

`r`'s potential block carries the symmetrized sign convention when that is in
force. Its norm is unaffected and nothing but the calls above should read it.

### 3.3 The gradient, and not assembling it

`NPCGradient()` honours `SetGradientMode()`:

| | `S` is | trace solver |
|---|---|---|
| `Assembled` (default) | `SparseMatrix`, or `HypreParMatrix` in parallel | anything, including direct and AMG |
| `MatrixFree` | an `Operator` applying `S = H - C'M⁻¹[C;E]` element by element, **no global matrix** | a Krylov method needing only the action |

Both factor the local blocks and form the local Schur complement, so the
reduction and the recovery are identical either way; only the *global* trace
matrix is declined. The two agree at every iterate above round-off.

**The returned handle is solve-only and its `Mult()` aborts.** After
`ComputeH()` the local arrays hold the *factored* blocks, so `J` cannot be
applied out of them without unfactored copies of every block.
`DarcyNPCSolver::SetOperator` dynamic-casts for it, so a plain Krylov method
over `GetGradient()` fails loudly rather than silently — and **JFNK is
therefore unavailable**, since it needs exactly that action.

**A gradient-free outer solve is a different case and the answer is
measured, not inferred.** `LBFGSSolver` and `LBBSolver` never call
`GetGradient()` — they read `oper->Mult` only — so nothing stops a caller
handing them `DarcyNPCOperator`; `DarcyNPCSolver` is then simply unused and
the hybridized elimination never runs. Tried on a case `NewtonSolver` solves
in four steps, LBFGS **diverges to NaN**. Whether that is L-BFGS wanting a
gradient field the full residual does not provide, or a scaling between the
local and trace rows, has not been established. Either way it is not a route
to recommend.

### 3.4 Parallel

Supported, and the same calls. The flux and potential are L2, hence rank-local
with L-dofs equal to true dofs and **no communication at all**; the trace lives
on the skeleton and is the only thing shared, so it is prolonged on the way in
and assembled on the way out. Size the trace block with `GetTrueVSize()` as
above and everything else follows.

Pinned by `[NPC][Parallel]` in `tests/unit/fem/test_darcy_npc.cpp`, the first
`[Parallel]` Darcy case in this tree: on a problem whose full system is linear
one NPC step is exact on two ranks, to below 1e-13. That is the sharp check —
get any one of the four prolongation or assembly steps wrong and the second
residual is O(1), not 1e-10.

### 3.5 What it refuses, and what is missing

One hard refusal, a `MFEM_VERIFY` in `NPCCheck()`: **an H(div) flux space.**
The local rows would be a conforming scatter with sign conventions this has not
been checked against, and the RT paths are deliberately left alone.

`LocalOpType::FluxNL` was refused here too and no longer is. The Schur
complement had nowhere to live in that mode — `Df_data` holds the factored
*linear* potential mass, which the local solve needs — so `ComputeElementH()`
built it in a temporary and dropped it. It now goes to `Sf_data`; see §7.3 for
the withdrawal of what that entry claimed.

Missing rather than refused:

* **No NPC regression reference exists.** `navierstokes` is driven by NPC
  unconditionally and has no reference at all; `convdiff` and `pconvdiff` take
  `-npc`, through `DarcyOperator::SetNPC()`, but the flag defaults off and
  every one of the 129 + 98 references predates it. So the method is covered
  by unit tests and by nothing else;
* **the trace right-hand side has no slot.** `load` is `(flux, potential)`;
  a Neumann datum assembled on the trace has to ride in `b` of
  `NewtonSolver::Mult(b, x)`;
* **`ComputeSolution()` has not been exercised** against an NPC solution.

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

### 4.3 What NPC adds

Nothing per element. NPC reuses the same blocks — it assembles the Jacobian
into `Af_data`/`Bf_data`/`Bnl_data`/`Df_data` exactly as a gradient pass does,
and `ComputeH()` factors them exactly as it does for the reduced operator. The
retained-linearisation storage that the deleted mode carried (`lin_trace`,
`lin_u`, `lin_p` and their `_next` scratch, five vectors and a validity flag)
went with it.

What NPC adds is on the **caller's** side: the unknown is `n_flux + n_pot +
n_trace` rather than `n_trace`, and the caller holds the residual and the
increment at that size. That is the whole cost of making the fields Newton
state.

## 5. What changes from standard MFEM assumptions

### 5.1 and 5.2 are gone with the mode they were about

They asked whether `Mult` is a function of its argument, and what contract a
solver had to honour. Both were properties of the deleted trace-only mode,
which retained a linearisation between calls. The reduced operator solves
its local problem afresh every evaluation and NPC holds no state between
calls at all, so for both of them `Mult` is a function of its argument and
there is no contract to state.

The finding worth keeping out of them is not about either mode:

**A better Jacobian can converge to a different solution.** Where a coarse
discretisation carries more than one, an iteration driven by an inaccurate
gradient wanders and can settle on the branch a Picard iteration finds; with
the gradient right it converges faster and stays on its own. A caller had a
test pinning Newton against Anderson-Picard on one mesh at 1e-6, and after a
gradient fix it read 9.1e-05 — bit identical when the tolerance was tightened
four orders, so both were fully converged and the fixed points genuinely
differed, at 1e-13 on two other meshes and 3e-06 on a third with no trend.
That is a gate that was green for the wrong reason, not a regression. It is in
`DarcyHybridization`'s doxygen too.

### 5.3 Consequences for solvers

Both routes work with every outer solver MFEM offers, and neither places a
requirement on one. The table that stood here was mostly a list of what the
deleted mode broke.

| solver | reduced trace operator | NPC (full system) |
|---|---|---|
| `NewtonSolver` | works | works, and needs to know nothing |
| Newton + line search | works | works, and the search scales the fields with the trace |
| lagged Jacobian | works | works |
| `LBFGSSolver`, `LBBSolver` | works | accepted but **diverges to NaN** where Newton takes four steps; they never ask for a gradient, so the elimination goes unused. See §3.3 |
| `KINSolver`, matrix-based | works | works; **use `SetMaxSetupCalls(1)`** or KINSOL's lazy `LinSysSetup` gives a lagged-Jacobian Newton (12 iterations against 4, both converged) |
| JFNK / matrix-free outer solve | works | **unavailable** — it needs the Jacobian's action and the handle is solve-only |

`SetLocalNLSolver()` configures the local nonlinear solve that
the reduced operator runs per element. **NPC has no local nonlinear solve**,
so it is inert there, and `GetNumLocalNLIterations()` staying at zero is the
acceptance signal that NPC is doing what it claims: one local *linear* solve
per outer step.

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
| `NPCReduce` and `NPCRecover` | `true` — they eliminate with the *Jacobian*, so its (0,1) block is the one that belongs |
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
`ComputeH()`'s first half. **So the gap is closed** — the tests sweep `GradientMode` over the reduced
operator and over NPC alike (`tests/unit/fem/test_darcy_npc.cpp`).

What `MatrixFree` costs, in exchange: `GetGradient()` returns an `Operator`
with no matrix, so `GSSmoother`, `UMFPackSolver` and the algebraic
preconditioners abort (`fem/darcy/darcyhybridization.hpp:827-831`). It used to
be **refused for `LocalOpType::FluxNL`**, for the storage reason above; since
`Sf_data` exists it works in every mode. And two things the assembled
matrix gets for free have to be applied by hand: the unit row on essential
trace dofs, and `SetDiagIdentity()`'s regularisation of rows nothing
contributed to (`fem/darcy/darcyhybridization.cpp:4180-4205`, and
`MarkEmptyTraceRows()` at `:2787-2851`). Without the latter the two modes
differed by **0.497 on 64 of 160 trace rows** on a problem whose constraint has
no boundary face term (commit `c849adffd5`); the case is
`tests/unit/fem/test_darcy_linearise_first.cpp:761-798`.

### 5.5 Other places the default expectation is wrong

These apply to the nonlinear hybridized operator generally.

1. **The reduced right-hand side is zero, and the load lives inside the
   operator.** For a nonlinear problem `ReduceRHS()` stores the mixed RHS in
   `darcy_rhs` and returns a zeroed `b_r`
   (`fem/darcy/darcyhybridization.cpp:3445-3495`); `Mult()` returns the full
   residual, load included. `NewtonSolver::Mult(B, X)` then subtracts a zero
   `B`. `EliminateTraceTrueDofsInRHS()` for the nonlinear case only zeroes the
   essential rows (`fem/darcy/darcyhybridization.cpp:2587-2599`). A caller
   reasoning "`A x = b` with the `b` I was handed" will be surprised.
2. **`Mult()` is `const` and mutates a great deal** — the local block
   storage above all. It is `const` because `Operator` requires it, not
   because it is free of side effects.
3. **`GetGradient()` overwrites the local block storage.** After it,
   `Af_data`/`Df_data` hold the last Jacobian and its Schur complement, not the
   forms. For the reduced operator this is invisible, because the residual
   reads `Af_lin_data`/`Df_lin_data`. Under NPC it is why the Jacobian handle
   is solve-only: there is nothing left to apply `J` out of.
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
6. **`ComputeSolution()` has not been exercised against an NPC solution.**
   It reconstructs the fields from the trace, which is what condensation
   wants; under NPC the fields are already Newton state and the
   back-substitution is redundant at best. Unchecked either way.

7. **Moot.** It described `SetNonlinearOrdering()`'s cache invalidation, and
   both the method and the cache are deleted.

---

## 6. When to use which

They are different methods reaching the same discrete solution, so the choice
is about cost and robustness rather than correctness. Tests assert that both
get there.

**Which to reach for depends on one thing: what the element-local nonlinear
solve costs on your problem.** That is the whole of it, and an earlier version
of this section gave "reach for the reduced trace operator unless you have a
reason not to" as a general default. That was wrong as a default and `meq`
measured it so — see `HDG-NPC-GLOBALISATION-FROM-MEQ.md`.

**The reduced trace operator.** The outer unknown is the trace alone, much the
smaller vector, and the local problem is solved to a tolerance you control. It
is the only route that accepts an **H(div) flux**. Reach for it when the local
problems are mild — then the per-element Newton is cheap and solving it
properly is worth more than avoiding it.

**NPC** when the per-element nonlinear solves are the cost or the thing that
fails. Every local operation is one linear solve against one factorisation, so
a stiff local problem cannot stall the outer iteration the way it can under
condensation; the price is that the unknown is the whole system and the
convergence test with it.

**And it is often faster, which this section used to deny.** It said NPC's
advantage was uniformity of the local work "not fewer floating-point
operations". On a semilinear Grad-Shafranov discretisation `meq` measured 4.3x
and 3.7x of wall clock at the same or fewer iterations, and a whole test suite
at 872 s -> 499 s, because the element-local Newton the condensation runs per
element per residual evaluation *is* most of the cost and NPC deletes it. When
NPC fails it also fails 33x to 41x cheaper, having run no local nonlinear
iterations at all — which makes it the better thing to try first even on
problems it loses. The uniformity claim stands; the "not fewer operations"
half is withdrawn.

### The line search: read this before taking the recommendation

Measured here on the Darcy pedestal, undamped against backtracking on the full
residual, at the four configurations §7 used to cite:

| | undamped | with backtracking |
|---|---|---|
| n=8 k=2 | fails, 2.4e+00 | **ok, 13 steps** |
| n=12 k=3 | ok, 12 steps | ok, 10 steps |
| n=32 k=1 | fails, 4.9e-01 | **ok, 17 steps** |
| n=24 k=1 | fails, 1.4e+00 | fails, 2.9e-03 |

So on *this* problem the line search earns its place, and the baseline is
plain undamped `NewtonSolver` on NPC — not the deleted mode, which is what
`meq` asked. **On `meq`'s problem the same line search made every case worse**,
including five that converge undamped.

Both are true and the block structure is not what separates them, because it
is the same. Measured here: the flux row falls to 4e-17 and the trace row to
3e-14 on a full step, with the entire remainder in the potential row — exactly
the shape `meq` reports. **What differs is severity.** Here a full step
*reduces* the potential residual (4.7e-01 -> 1.7e-01); on `meq`'s problem it
multiplies it by 77 (8.1e-02 -> 6.25e+00), so `alpha = 1` is rejected and the
search retreats.

**Why is there a hand-rolled backtracking Newton here at all?** There should
not be. `KINSolver(KIN_LINESEARCH)` is KINSOL's Dennis & Schnabel line search
with both the sufficient-decrease and curvature conditions, a minimum-step test
that reports non-convergence instead of creeping, and a maximum-step
constraint. `NSBacktrackingNewton` in `miniapps/hdg/navierstokes.cpp`
reimplements a worse version of it, and exists only because **neither HDG tree
here is configured with SUNDIALS**, so `KINSolver` is not compiled. That is a
build reason, not a numerical one, and it is a bad reason to ship the worse one
as the reference other people copy. `meq`'s build has SUNDIALS on and should
use `KIN_LINESEARCH` rather than the copy.

It is a genuinely worse implementation: it accepts on `Norm(rt) < n0`, a
monotone test with **no sufficient-decrease constant**. For Newton on an l2
merit the direction is always a descent direction, so merit(a) ~ merit(0)(1-a)
and any small enough `a` passes — swept here, `a = 1.2e-4` still "succeeds",
improving the merit by 1e-4 relative. Where `a = 1` is rejected it therefore
does not fail, it **creeps**, which is the 1%-a-step crawl `meq` saw.

**But that defect is not what makes `meq`'s problem fail, and an earlier
version of this section implied it was.** `meq` also ran `KIN_LINESEARCH` —
the correct implementation, sufficient decrease and all — and reports it
failing on exactly the same cases. So a proper Armijo test does not rescue it.
Fixing ours would make the failure honest, not make it converge.

**Nor does a block-aware merit, which is what both we and `meq` reached for
next.** Take `meq`'s own numbers: the residual at `x` is (flux 7.93e-02,
potential 8.13e-02, trace 0) and after a full step (1.14e-16, 6.25e+00,
1.82e-13). For any diagonal merit with weights `(w_f, w_p, w_t)`, accepting
`a = 1` needs

    (w_p * 6.25)^2  <  (w_f * 0.0793)^2 + (w_p * 0.0813)^2   =>   w_f/w_p > 78.8

— you would have to weight the **linear flux row 79x above the nonlinear
potential row**, the opposite of damping the nonlinear block. And a merit that
all but ignores the potential row is one a Newton step annihilates exactly, so
the search accepts `a = 1` unconditionally and degenerates into no line search.
KINSOL already exposes exactly this knob (`KINSolver::Mult(x, x_scale,
fx_scale)`), so the idea is cheap to test and the arithmetic says not to
bother.

**What the evidence actually points at is non-monotonicity.** Undamped Newton
converges five cases that every monotone search kills, which is the signature
of an iteration that must be allowed uphill to reach the basin. The remedies
are a non-monotone acceptance test (accept below the max of the last M
merits), a trust region, or no line search — and KINSOL's own answer,
Anderson-accelerated `KIN_PICARD`/`KIN_FP` (`KINSolver::EnableAndersonAcc()`),
is precisely the production ladder `meq` already ships. None of that needs a
new solver written here.

## 7. Suspected defects and inconsistencies

Recorded here because they were found while writing this and are the reason
some of the sections above are hedged. They were found by reading; the tree was
mid-build at the time and nothing was run.

**Status.** 1, 2, 4, 5 and 6 are **gone or moot**: they were about the
deleted mode, its plan document (also deleted) or the solver contract it
imposed. **3 is fixed, and the entry that raised it was wrong about the
danger** — see its withdrawal below. 7, 8, 9 and 10 are **open**, and 8 is the
one worth acting on: an accessor that returns the wrong member.

1. and 2. **Gone with the mode.** One was about
   `doc/HDG-LINEARISE-THEN-CONDENSE.md`, which is deleted; the other about
   `SetNonlinearOrdering()`'s doxygen, which is deleted with the method.

3. **Fixed — and this entry overstated it, which is worth recording.** It
   said `LocalOpType::FluxNL` would let the *assembled* path eliminate with the
   wrong operator and return a silent wrong answer. It could not: the only
   readers of a Schur complement out of `Df_data` are the matrix-free gradient
   and NPC, and **both were themselves refused in that mode**, so the hazard
   was real in structure and unreachable in practice — held shut by the very
   refusals the entry complained about. Measured: the assembled answer is
   unchanged by the fix, to every digit, and its regression reference still
   passes. The real defect was narrower: the Schur complement had nowhere to
   live, so two capabilities were refused rather than one answer corrupted.
   `Sf_data` now holds it and both refusals are lifted; the reason is on
   `Sf_data`, `SetGradientMode()` and `NPCCheck()`.

4. and 5. **Moot.** Both asked for warnings about the deleted mode's solver
   contract, on a `-lfirst` flag that no longer exists in any miniapp.

6. **Moot.** It was about how much of the deleted mode's non-purity gap a
   test pinned.

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
