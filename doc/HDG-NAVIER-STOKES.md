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

## The §5 measurement

`-tau <c>` selects the library's constant-`Ctau` `HDGFlux` in place of
`S = λ_max(û,n) I`. Since `λ_max` is `√β` across the flow and `|v| + √(v²+β)`
along it, that is a direction-aware stabilization against a direction-blind
one on the same discretisation, and it is what §5 asks about.

**937 runs, six waves, 7.2 CPU-hours**: Kovasznay and plane Poiseuille, orders
1 to 3, `Re` 10 to 1000, meshes from 4x4 to 128x8, cell aspect ratios from 1/4
to 4, `τ` from 0.125 to 10, and `β` over a factor of 16. All of it with
`-gm 0 -rtol 1e-8`, for the reason in the first control below.

### Two controls, run before the sweep

**The constant-`τ` path is consistent, so the comparison is fair.** If `-tau`
changed consistency rather than only the weight on the jump, it would be a
comparison between a correct method and a wrong one. Residual at the exact
plane-Poiseuille solution, `-xinit -atol 1e-8`, 8x8:

| | order 2 | order 3 |
|---|---|---|
| `λ_max(û,n)` | 6.25638e-16 | 2.78e-15 |
| `-tau 0.1` | 6.25638e-16 | 2.70e-15 |
| `-tau 1` | 6.25638e-16 | 2.76e-15 |
| `-tau 10` | 6.25638e-16 | 3.37e-15 |

Identical to every digit at order 2, as they must be: the stabilization
multiplies `(u − û)`, which vanishes exactly when the exact solution lies in
the space and the trace carries it.

**The default solver pollutes the errors above the level a rate study needs.**
At 48x32, order 2, `Re = 40`, the GS-preconditioned trace solve gives
`‖p‖ = 2.569e-5` and the direct one `2.274e-5` — 13% apart, with Newton having
satisfied its relative test both times. Errors are unchanged between
`-rtol 1e-6` and `-rtol 1e-8`, so `-gm 0 -rtol 1e-8` puts the solver an order
below the discretisation and the sweep uses it throughout.

### The answer, in one sentence

**The direction-aware `τ` is worse for accuracy and better for nonlinear
solvability, and the two requirements point in opposite directions.**

### Accuracy: `λ_max` is the constant `√β` in disguise

Sweeping `τ` finely — Kovasznay, order 2, `Re = 40`, 96x64, relative L2:

| | `λ_max` | 0.125 | 0.25 | 0.375 | 0.5 | 0.75 | 1 |
|---|---|---|---|---|---|---|---|
| ‖q‖ | 3.834e-5 | **1.944e-5** | 2.182e-5 | 2.419e-5 | 2.652e-5 | 3.110e-5 | 3.556e-5 |
| ‖v‖ | 1.675e-6 | 1.681e-6 | 1.676e-6 | 1.673e-6 | 1.671e-6 | **1.671e-6** | 1.673e-6 |

**The split between the fields is the finding, not the ratio.** In the flux the
error is monotone in `τ` and `λ_max` runs **2.0 to 3.6 times** the best value,
at every order and Reynolds number measured. In the velocity it is flat:
`λ_max` is 1.00 to 1.17 times the best. Over-stabilization also costs the flux
its rate — at order 2, `Re = 400`, the last measured rate in ‖q‖ is 2.60, 2.49,
2.48, 2.34, 2.18 for `τ` = 0.5, 1, `λ_max`, 2, 5.

The pressure behaves like the flux, and that has a cause specific to this
formulation: **the continuity row carries no diffusive stabilization at all**
(`stab[0]` is `HDGDiffusionIntegrator(zero_coeff)`), so `S` acts there alone.
It is the one component where the convective stabilization is the whole story
and it is the one that moves with `τ`.

**Why `λ_max` lands where it does, established by measurement rather than by
argument.** `β` is a free parameter: the continuity row is `β ∇·v = s_0`, so
`β` cannot change the steady answer, but it *does* set `λ_max = √β` on any face
where `v·n = 0`. Sweeping it is therefore a controlled experiment on the
stabilization alone. Kovasznay, `Re = 40`, order 2, 48x32:

| `β` | `√β` | `λ_max` ‖q‖ | `τ = √β` ‖q‖ | ratio ‖q‖ | ratio ‖p‖ | ratio ‖v‖ |
|---|---|---|---|---|---|---|
| 0.25 | 0.5 | 1.670e-4 | 1.437e-4 | 1.162 | 1.057 | 1.000 |
| 1 | 1 | 2.292e-4 | 2.096e-4 | 1.093 | 1.031 | 1.003 |
| 4 | 2 | 3.387e-4 | 3.233e-4 | 1.048 | 1.013 | 1.005 |

`λ_max`'s error **moves with `β`**, tracking `τ = √β` to within 5-16%, and the
gap closes as `β` grows because `√β` then dominates `|v·n|` on every face. The
control that makes this airtight: at *fixed* `τ = 1`, changing `β` by 16x moves
‖q‖ from 2.09594e-4 to 2.09644e-4 — **0.02%**. So `β` genuinely moves nothing
but the stabilization, and `λ_max` is a constant `√β` plus a penalty.

That is the substantive point. **`λ_max`'s stabilization level is set by an
arbitrary numerical parameter of the formulation, not by the physics**, and the
extra weight it puts on the along-flow faces is a cost rather than a benefit.

### Robustness: the other half, and it points the other way

`λ_max` converged on **every one of the ~300 Kovasznay cases** — every order,
Reynolds number and mesh. The constants fail, and they fail on *coarse* meshes,
recovering under refinement:

| order 2, `Re = 400`, with `-cont` | `λ_max` | 0.25 | 0.5 | 1 | 2 | 5 |
|---|---|---|---|---|---|---|
| 6x4 | ok | diverge | ok | ok | ok | ok |
| 12x8 | ok | diverge | ok | ok | ok | ok |
| 24x16 | ok | timeout | ok | ok | ok | ok |
| 48x32 | ok | timeout | ok | ok | ok | ok |

From a **cold** start, plane Poiseuille 16x8 at `Re = 100`:

| `λ_max` | 0.25 | 0.5 | 1 | 2 | 5 | 10 |
|---|---|---|---|---|---|---|
| **9 it** | fail | fail | fail | fail | 8 it | 8 it |

`λ_max` never exceeds 2.4 anywhere on this problem, yet `τ = 2` fails and only
`τ ≥ 5` recovers: it is not that `λ_max` supplies *more* stabilization but that
it supplies it where the convection is. The limit of that advantage is also
measured — at `Re = 400` a cold start fails for **every** stabilization
including `λ_max`, so continuation is a property of the problem there, not
something a better `τ` buys.

**The two halves meet.** At `Re = 400` the best converging `τ` is 0.375 at
48x32 and 0.5 at 24x16 — the accuracy optimum sits exactly at the robustness
boundary, and raising `Re` pushes the optimum out of reach.

### The mesh aspect ratio changes none of it

Kovasznay on a window stretched to 4x1 at `Re = 40`. Ratio of each
stabilization's error to the best at that cell shape, `ny = 16`:

| | aspect | `λ_max` | 0.25 | 0.5 | 1 | 2 |
|---|---|---|---|---|---|---|
| ‖q‖ | 4 | 2.36 | **1.00** | 1.40 | 2.06 | 2.99 |
| ‖q‖ | 1 | 2.34 | **1.00** | 1.41 | 2.15 | 3.28 |
| ‖v‖ | 4 | 1.21 | **1.00** | 1.01 | 1.09 | 1.22 |
| ‖v‖ | 1 | 1.03 | 1.02 | **1.00** | 1.02 | 1.12 |

Direction-awareness never helps, and in ‖v‖ its penalty is *largest* on the
cells stretched 4:1 along the flow (1.21 against 1.03 on square cells) — the
opposite of what it would do if knowing the face normal were worth anything
here.

**Why the window had to be stretched, which is a finding about the problem
rather than about `τ`.** Kovasznay's decay rate is

    λ = Re/2 − √(Re²/4 + 4π²) → −4π²/Re,

so **the parameter that makes it convective is the same one that flattens its
structure along the flow**: `e^{λx}` varies by 94x across the standard window
at `Re = 10`, 4.2x at `Re = 40` and **1.16x at `Re = 400`**. Measured
consequence — on the standard window at `Re = 400` the errors are identical to
four digits across a 16x range in `nx`. Plane Poiseuille is worse still: its
pressure is linear in `x`, so there is nothing to resolve along the flow at any
order. **Neither of this miniapp's exact solutions can pose the aspect-ratio
question on its own natural domain**, and a longer window at a moderate `Re` is
the cheapest repair — `-sx 4 -re 40` gives 76x of variation at a convective
Reynolds number.

### What this does and does not settle

It settles that on *these* problems the direction-aware `λ_max` of PNC eq (6)
buys nothing in accuracy and something real in robustness, and it identifies
the mechanism by a controlled sweep rather than by inspection. It does not
settle the general question, for one reason worth stating: both exact
solutions here have their sharp structure across the flow and little or none
along it, so the along-flow faces — the ones where `λ_max` differs from `√β` —
are exactly the faces where the solution is easiest to represent. A problem
with comparable structure in both directions would weight them differently.
`anisodiff -p 11` on `gf-hdg-subdomains-dev` is the linear-diffusion shape of
that, and a genuinely two-directional Navier-Stokes solution is not among the
four this miniapp has.

### Traps found while running it

* **A `converged in` line is not a success test when `-cont` is on.** It can
  come from the Stokes solve while the Navier-Stokes solve then diverges to
  `inf` and aborts on MFEM's `IsFinite` check with `rc = 134`. That misread 19
  divergences as successes on the first pass; they are recognisable by carrying
  no error norms, and every one of them is a constant `τ` rather than `λ_max`.
  Classify on the exit code.
* **`-cont` on plane Poiseuille at order ≥ 2 cannot converge, and no `atol`
  fixes it.** The exact profile has `v·∇v = 0`, so the Stokes and Navier-Stokes
  solutions coincide *and are in the space*; the second solve starts at the
  linear solver's noise floor with nothing to reduce. `-atol 1e-11` still spins
  past 160 s at 16x8. The block measures the convergence test, not the
  stabilization.
* **Threads belong across cases, not inside one.** MKL's default 8 threads made
  a solo run *slower* (13.7 s against 13.0 s at one thread, identical to every
  digit), and six workers at 8 threads each oversubscribed 16 cores badly
  enough that a 55 s case took 158 s. `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` per
  case, parallelism from `xargs -P`.
* **A free-parameter sweep is the cheapest mechanism test available.** `β` was
  in the miniapp as a formulation knob documented not to change the answer.
  That documented no-op is exactly what makes it a controlled experiment on the
  stabilization, and it converted "`λ_max` behaves like `τ = 1`, and `√β = 1`
  here" from a coincidence into a measurement.

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
