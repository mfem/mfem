# A Navier-Stokes miniapp on the HDG interface

`miniapps/hdg/navierstokes.cpp`, with `miniapps/hdg/nsflux.hpp` beside it. It
solves the **steady incompressible** Navier-Stokes equations by the HDG method
of Peraire, Nguyen and Cockburn (*A hybridizable discontinuous Galerkin method
for the compressible Euler and Navier-Stokes equations*, AIAA 2010-363), on
`DarcyForm` / `DarcyHybridization`, and is written so the compressible
equations drop in later.

Its purpose is §5 of `HDG-ROADMAP.md`: whether one `τ` can serve convection in
one coordinate direction and diffusion in another at the same time. Plane
Poiseuille flow is the smallest problem with that character, diffusive across
the channel and convective along it.

## The formulation, and why it fits a two-block form

The reference's first-order system, its Eq. (13), is

```
q − ν ∇u = 0,      ∇·(F(u) + G(u,q)) = s,
```

and the state is written in Chorin's artificial-compressibility variables

```
u = (p, v_1 … v_d),    neq = dim + 1,
F_{0,d}   = β v_d,             G_{0,d}   = 0,
F_{1+i,d} = v_i v_d + p δ_id,  G_{1+i,d} = −q_{1+i,d}.
```

Three things this buys:

* **It is the same shape as the compressible system.** Flux `q` of vdim
  `neq·dim`, potential `u` of vdim `neq`, trace of vdim `neq` — the triple
  `DarcyForm` already has. The continuity variable is first, exactly as
  `EulerFlux`'s `(ρ, ρv, ρE)`, so nothing that indexes the state changes when
  the flux function is swapped.
* **β does not perturb the answer.** The continuity row is `β ∇·v = s_0`, so at
  a root of the steady residual `∇·v = s_0/β` exactly, whatever β is. The
  incompressibility is imposed, not penalised. What β sets is the pressure's
  characteristic speed, and with it the stabilization floor.
* **The stabilization becomes the §5 object.** The normal flux Jacobian is
  `A_n = [0, β nᵀ; n, (v·n) I + v nᵀ]`, with eigenvalues `v·n` (multiplicity
  `d−1`) and `(v·n) ± √((v·n)² + β|n|²)`, so

  ```
  λ_max(u,n) = |v·n| + √((v·n)² + β|n|²).
  ```

  Along the flow that is `|v| + √(v² + β)`; across it, `√β`. **A single
  constant τ cannot be both**, which is exactly §5's question made concrete.
  `HDGLaxFriedrichsFlux` supplies `S = λ_max(û,n) I`, the reference's Eq. (6);
  `-tau <c>` swaps in the library's constant-`Ctau` `HDGFlux` so the two can be
  measured against each other.

The diffusive half of the stabilization stays separate, on
`HDGDiffusionIntegrator`, which is NPC-1 §3.6's `s = s_diff + s_conv` rather
than one lumped constant.

**The pressure has no gradient variable**, so equation 0's flux mass coupling,
divergence block and trace constraint are all zeroed and `q_0 ≡ 0`. It costs
the dofs it occupies. Writing it that way rather than shrinking the flux space
keeps every block square and lets `VectorBlockDiagonalIntegrator` do the
replication — and a *null* entry there shrinks the element matrix instead of
zeroing it, which the hybridization's size assertions then reject, so the zeros
are zero-valued integrators (`VectorDivergenceIntegrator(0)`,
`NormalTraceJumpIntegrator(0.)`).

## What is verified

Four exact solutions, all with **zero source term**, ordered by which terms of
the operator they can see. That ordering is the point: a single problem that
fails says nothing about which term failed.

| `-p` | problem | `q` | `∇·q` | first sees |
|---|---|---|---|---|
| 3 | uniform flow | 0 | 0 | inviscid flux, trace treatment |
| 4 | Couette | const | 0 | flux recovery `q = −ν∇u` and its constraint |
| 1 | plane Poiseuille | linear in y | ≠ 0 | everything |
| 2 | Kovasznay | — | ≠ 0 | genuine convection, `v·∇v ≠ 0` |

**Plane Poiseuille is a polynomial** — pressure linear in `x`, velocity
quadratic in `y` — so at order ≥ 2 it lies in the discrete space and the HDG
solution must reproduce it to round-off. Measured, `-nx 4 -ny 4`, relative L2:

| order | ‖q‖ | ‖p‖ | ‖v‖ |
|---|---|---|---|
| 1 | 1.58e-1 | 2.26e-2 | 2.89e-2 |
| 2 | 2.57e-15 | 2.48e-15 | 3.85e-16 |
| 3 | 4.74e-15 | 3.83e-15 | 6.51e-16 |
| 4 | 6.44e-15 | 8.05e-15 | 8.12e-16 |

Order 1 is inexact because the solution is not in `P^1`; that it is inexact is
as much of the check as that orders 2-4 are exact. Newton reaches these
quadratically — `3.5e-1, 2.8e-1, 4.6e-2, 1.2e-3, 1.2e-6, 7.4e-13, 3.9e-16`.

**Kovasznay** at `Re = 40` on `(-0.5,1) x (-0.5,0.5)`, relative L2, run with
`-cont`. `1/h` is the cell count across the channel:

| k | 1/h | ‖q‖ | rate | ‖p‖ | rate | ‖v‖ | rate | Newton |
|---|---|---|---|---|---|---|---|---|
| 1 | 4  | 2.853e-1 | –    | 8.861e-2 | –    | 8.915e-2 | –    | 4 |
| 1 | 8  | 1.116e-1 | 1.35 | 1.245e-2 | 2.83 | 1.709e-2 | 2.38 | 4 |
| 1 | 16 | 3.939e-2 | 1.50 | 2.343e-3 | 2.41 | 3.691e-3 | 2.21 | 4 |
| 1 | 32 | 1.330e-2 | 1.57 | 4.531e-4 | 2.37 | 8.528e-4 | 2.11 | 4 |
| 2 | 4  | 4.742e-2 | –    | 6.760e-3 | –    | 8.307e-3 | –    | 5 |
| 2 | 8  | 8.519e-3 | 2.48 | 1.093e-3 | 2.63 | 9.653e-4 | 3.11 | 4 |
| 2 | 16 | 1.418e-3 | 2.59 | 1.555e-4 | 2.81 | 1.133e-4 | 3.09 | 4 |

The potential converges at the optimal `k+1` — 2.11 at `k = 1`, 3.09 at
`k = 2`. **The flux rates lag and are still climbing** (1.35, 1.50, 1.57 at
`k = 1`; 2.48, 2.59 at `k = 2`), which is the roadmap's standing warning that
**rates must be taken asymptotically** showing up again on a new problem; on the
coarsest pair alone the `k = 1` flux would read 1.35 against an expected 2.
Newton takes 4-5 iterations independently of `h`.

A cold Newton from rest **diverges** on the coarser Kovasznay meshes — the
local element solve runs away before the trace system has any information in
it — so those runs use `-cont`, which solves the Stokes problem first and
continues onto Navier-Stokes from its answer, at the cost of one extra linear
solve. It is off by default, and deliberately: on plane Poiseuille the exact
profile has `v·∇v = 0`, so the Stokes and Navier-Stokes solutions *coincide*,
the continuation lands on the answer, and the second solve then starts at the
linear solver's noise floor where no relative reduction is achievable. Newton
spins to `max_iters` there — `r0 = 3.7e-10`, then 4.7e-10, 8.5e-10, 8.9e-10.
`-atol` is the floor that stops it, at the price of five orders of accuracy
(5.0e-10 against 2.6e-15), which is why the exactness check runs without
either.

Implementing the continuation was three lines, because nothing caches the flux:
`ArtificialCompressibilityFlux::SetStokes(bool)` flipped between two
`ode.Step()` calls is the whole of it.

The unit test `tests/unit/fem/test_hdg_nsflux.cpp` finite-differences every
analytic derivative in `nsflux.hpp`, and was checked against eleven deliberate
mutations of the header, all of which it caught.

## Two defects, both found by the residual-at-the-exact-solution measurement

The method throughout was the one this branch keeps arriving at: **apply the
assembled operator to the exact solution and see whether it is a root.** Both
defects were invisible to everything else.

### 1. The constraint was never installed on boundary faces

Symptom: at the exact trace the local solve returned the potential exact to
1e-16 and the **flux** wrong — `‖q_h − q_ex‖` of 0.3 on uniform flow, where
`q_ex = 0`.

Cause: row 1 must satisfy `−(u, ∇·v)_K + ⟨λ, v·n⟩_∂K = 0` identically for a
constant state, by the divergence theorem. The hybridized path never
*evaluates* `B`'s face integrators — only `AssembleDivLDGFaces()`, which the
reduction branch calls, does — but a boundary face integrator on `B` supplies a
**marker**, and `DarcyForm::Assemble()` reads `B->GetBFBFI_Marker()` and
installs the constraint integrator on exactly those attributes. Without one the
constraint is assembled on interior faces only, and the identity fails on every
element touching the boundary.

What made it findable rather than merely visible: sweeping the *scaling* of the
interior constraint over 0.25, 0.5, 0.75, 1, 2 gave 0.40, 0.34, 0.297, 0.30,
0.60 — a minimum, but never zero. **No scaling of a term that is present can
repair a term that is absent**, and that is what said to look for something
missing rather than something mis-signed.

### 2. `HyperbolicFormIntegrator::AssembleHDGFaceGrad` used the wrong dof
   ordering for a system

This one is a library defect, in `fem/hyperbolic.cpp`, and it is fixed here.

`AssembleHDGFaceVector` lays its output out **group-outermost** — all
`num_equations` fields of the element dofs, then all `num_equations` fields of
the trace dofs; its `trvect_mat` is based at `dof_dual_el * num_equations`.
`AssembleHDGFaceGrad` indexed as `elmat(di*dof_dual + ioff + i, …)` with
`dof_dual = dof_dual_el + dof_dual_tr`, which is **equation-outermost**: it
interleaves the two groups inside each equation.

The two agree when `num_equations == 1`, and when only one group is requested.
They differ otherwise — and `DarcyHybridization::ConstructGrad` asks for all
four blocks (`ELEM|TRACE|CONSTR|FACE`) in one call. So a *system* took a
gradient that did not match its own residual.

Nothing in the tree had exercised it. `convdiff` only ever uses scalar flux
functions (`AdvectionFlux`, `BurgersFlux`); `miniapps/plasma/braginskii_hdg.cpp`
drives Euler through an explicit IMEX split rather than the HDG face machinery;
`EulerFlux` has no unit-test coverage at all.

**Why it was hard to see, and what saw it.** A hybridized Jacobian is never
assembled globally, so a wrong one gives no wrong answer — only a Newton that
misbehaves. And the residual test could not find it either: the whole
stabilization vanishes when `u = û`, so at the exact solution the residual was
already 3e-16 with the gradient still wrong. What separated them was running
the *linear* Stokes problem, where a correct Jacobian must converge in exactly
one Newton step, and then **LBFGS against Newton**: LBFGS never asks for a
gradient, converged in 47 iterations onto the right answer, and Newton
diverged by a factor of 130 per step on the same operator. That is a clean
split of "the residual is wrong" from "the gradient is wrong", and it cost one
flag.

After the fix, all three linear problems converge in **one** Newton iteration.

## What is not done

* **`-bcphys` is wrong, and says so.** A boundary trace component that is not
  essential keeps the constraint row `⟨(F̂+q̂)·n, µ⟩ = 0`, which on a boundary
  face has only one side and so imposes *zero numerical flux* — not the
  intended condition. Measured on Poiseuille with the physical set the solve
  converges to 3e-13 and the answer is wrong by more than 100% at every order,
  while `-bcfull` on the same problem is exact to 2.5e-15. The default is
  therefore the full state on the whole boundary, which is the standard
  verification condition. Making `-bcphys` right needs the prescribed numerical
  flux on those faces: either the Neumann datum as a linear form on the trace,
  or the reference's characteristic `B̂ = A⁺_n(u−û) − A⁻_n(u_∞−û)`, which for
  this system needs the eigen-decomposition of `A_n`. **This is the next piece
  of work.**
* **`BdrHyperbolicDirichletIntegrator` cannot be used under hybridization**, and
  fails silently rather than aborting. It reads its prescribed state only when
  bit 0 of `type` is set, and `DarcyHybridization` never sets that bit on a
  boundary face — every `type |= 1` site is inside an interior-face branch.
  Registered on the hybridized form it degrades to an ordinary
  `HyperbolicFormIntegrator`: the interior state is used and the boundary datum
  is dropped, with no warning. Worth either fixing or making it abort.
* **Postprocessing is unavailable.** `DarcyForm::ReconstructTotalFlux` and the
  superconvergent reconstruction both `MFEM_VERIFY(fes_p->GetVDim() == 1)`.
  That is §4 of the roadmap.
* **Hagen-Poiseuille.** There is no axisymmetric support anywhere in the tree.
  The weak divergence in `(r,z)` is the Cartesian one under the measure
  `r dr dz`, so it needs the weight threaded through every integrator and a
  condition on the axis, but no new integrators.
* **Parallel.** No `pnavierstokes.cpp` yet.

## The §5 instrument

`-tau <c>` selects the library's constant-`Ctau` `HDGFlux` in place of
`S = λ_max(û,n) I`. Since `λ_max` is `√β` across the flow and
`|v| + √(v²+β)` along it, the ratio the constant has to straddle is set by the
mesh aspect ratio and by `-re`, and both are command-line knobs. The
measurement §5 wants is the convergence rate under each, swept in those two
parameters — with the caveat the roadmap already records, that **rates must be
taken asymptotically**, since the same configurations read 1.6 rather than 2.5
on the coarsest pair of meshes.

**One data point exists already, and it is not the expected one.** Kovasznay,
`Re = 40`, 24x16, `k = 2`, relative L2:

| stabilization | ‖q‖ | ‖p‖ | ‖v‖ |
|---|---|---|---|
| `S = λ_max(û,n) I` | 1.418e-3 | 1.555e-4 | 1.133e-4 |
| `S = 1` (constant `Ctau`) | 1.295e-3 | 1.491e-4 | 1.123e-4 |

The direction-blind constant is *marginally better* on every field. That is one
mesh at one Reynolds number on a problem with no strong directional
separation, so it settles nothing — but it is exactly the sort of result the
§5 sweep has to be built to survive, and it says the advantage of a
direction-aware `τ` is not going to be visible without a configuration whose
along-flow and across-flow Péclet numbers genuinely differ by orders of
magnitude. The channel with a large aspect ratio and a large `-re` is that
configuration; the sweep has not been run.

That configuration does work, and needs `-cont`:
`-p 1 -nx 16 -ny 4 -sx 4 -o 2 -re 200 -cont` is exact to 5e-15, one Newton
iteration for each of the two solves, where the same run cold diverges and
then stalls around `||r|| ~ 3e3`. So the §5 sweep is reachable; it is a
question of running it, not of making it run.

One thing found on the way that constrains how far this can go inside the
existing library: `MixedConductionNLFIntegrator`'s HDG face stabilization for
more than one equation is `face_w * TauVar(e)`, a single constant per equation
set once through `SetVariableStabilization()`. It cannot express a
stabilization that depends on the state or on the face normal — which is the
whole subject of §5. The route used here sidesteps it by carrying the
convective stabilization on the `NumericalFlux` instead, but a *viscous*
stabilization that varies with direction would run straight into it.

## Extending to the compressible equations

Three edits, and the file says so at its foot:

1. Swap `ArtificialCompressibilityFlux` for `EulerFlux(dim, γ)`; `neq` goes from
   `dim+1` to `dim+2`. `IsothermalFlux(dim, c_s)` is the intermediate step and
   the closer analogue — also `neq = dim+1`, with `p = c_s²ρ` playing the part
   `β v` plays here.
2. Replace the per-equation constant viscosity with a real `G(u,q)`. The
   compressible viscous flux is not diagonal in the equations, so the constant
   `VectorBlockDiagonalIntegrator` pair becomes a `MixedFluxFunction` driven by
   `MixedConductionNLFIntegrator` on `GetBlockNonlinearForm()`. That
   integrator's element terms are already generic in the number of equations;
   its LDG face terms are not, but the hybridized path never calls those.
3. Give the pressure row back its gradient — the three zeros above become the
   same integrators the other equations get.

Unchanged: the trace space, the hybridization, the stabilization splitting, and
the finding that the boundary datum has to reach the trace rather than the
numerical flux.
