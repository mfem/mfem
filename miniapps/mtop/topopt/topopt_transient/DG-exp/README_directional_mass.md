# Accumulated-density (X-ray thickness) constraint for topology optimization

This documents the constraint that limits the **density accumulated along a
direction** — the line integral of material that an X-ray sees — for
**CT-inspectable** designs. There are two formulations of the *same* underlying
advection model, and this file is the single place to compare them:

* **Steady-state baseline — implemented & verified** (§3–§5, §7). A direct solve
  of the steady advection $\,\mathbf v\!\cdot\!\nabla\rho_a=\rho_p$ with the
  **discrete** (transpose) adjoint. Correct, machine-precision-verified, and a
  reusable building block.
* **Transient formulation — proposed/next** (§6). The time-dependent advection
  $\dot\rho_a+\mathbf v\!\cdot\!\nabla\rho_a=\rho_p$ marched to steady state, with
  the **continuous** adjoint and **per-ray** inspectability bounds. This is the
  target formulation (advisor's note); the baseline is the thing it will be
  measured against.

The narrative: the steady baseline is *not* wrong — it is the $t\to\infty$ limit
of the transient model and reuses directly (the DG-upwind transport operator, the
source coupling, the MMA wiring). It is expected to **fail or become impractical
in specific experiments** (many ray directions / a full sinogram, point sources,
non-grid-aligned directions on unstructured meshes, and discrete-adjoint
consistency), which is exactly what motivates the transient formulation that the
coworker's single-ray approach builds toward.

| File | what it is |
|------|------------|
| [directional_mass.hpp](directional_mass.hpp) | `DirectionalMass` (serial) / `ParDirectionalMass` (MPI): DG-upwind transport $T$, source coupling $S$, adjoint weights — the **steady baseline** |
| [elasticity_topopt_par.cpp](elasticity_topopt_par.cpp) | driver: `-dm` flag, the 2-constraint `MMAUpdaterParDM`, `-p check` test (3) |
| [verify_dirmass.cpp](verify_dirmass.cpp) | native serial harness (transport unit test + adjoint + FD gradient) |
| _transient solver_ | _planned (§6): explicit time-marched advection + continuous adjoint + per-ray bounds_ |

---

## 1. Motivation: X-ray attenuation and CT inspectability

We consider a single isotropic material. In X-ray tomography the **Beer–Lambert
law** governs intensity attenuation along a ray,

$$I=I_0\exp\!\Big(-\!\int \mu(\mathbf x)\,\mathrm d\mathbf x\Big),$$

where $I_0,I$ are the intensities before/after the object and $\mu$ is the linear
attenuation coefficient, taken **proportional to the physical density**
$\rho_p$. For the scanner to register a ray, $I$ must stay within the sensor band
$[I_{\min},I_{\max}]$ — equivalently the **accumulated attenuation**
$\int\mu\,\mathrm d\mathbf x\propto\int\rho_p\,\mathrm d\mathbf x$ along the ray
must lie between bounds. So the inspectability requirement is a bound (upper, and
sometimes lower) on the **per-ray line integral**

$$\int_0^l \rho_p\Big(\mathbf a+t\,\tfrac{\mathbf b-\mathbf a}{l}\Big)\,\mathrm dt,$$

for *every* X-ray $\mathbf a\!\to\!\mathbf b$ crossing the domain ($l=$ chord
length), over enough rays/directions to form a reliable **sinogram**.

**Two ways to evaluate the line integral.** (i) *Quadrature along the ray*:
sample $\rho_p$ at points $\mathbf p_j$ with weights $\omega_j$ and sum. In a FE
setting each $\mathbf p_j$ must be located in the mesh and the field interpolated
there — a point-location/evaluation that **ignores the discrete structure** of
the density field and must be repeated for very many rays. (ii) *Advection* —
solve a transport equation whose solution **is** the running line integral. This
document takes route (ii); the accumulated field is denoted $\rho_a$ (the
baseline's $m$).

## 2. Two formulations at a glance

Both formulations transport the physical density $\rho_p$ with a **unit velocity
$\mathbf v$ aligned with the ray direction**, so that $\rho_a$ equals the line
integral of $\rho_p$ from the inflow up to each point ($\rho_p$ is taken $0$
outside the design domain, i.e. the ray enters with zero accumulated density).

|  | **Steady baseline** (implemented) | **Transient** (proposed) |
|--|--|--|
| equation | $\mathbf v\!\cdot\!\nabla\rho_a=\rho_p$ | $\dot\rho_a+\mathbf v\!\cdot\!\nabla\rho_a=\rho_p$ |
| solve | one non-symmetric solve (GMRES+BlockILU) | explicit time march (SSP-RK), matrix-free; steady reached at $t=l$, fixed after |
| adjoint | **discrete** (exact transpose $T^{\mathsf T}$) | **continuous** backward advection (§6) |
| constrained quantity | global proxy $\int_\Omega\rho_a\,dx$ | **per-ray** outflow trace $\rho_a$, bounds over all rays (sinogram) |
| many directions / point sources | reassemble/resolve per direction; sweep can cycle on unstructured meshes | uniform matrix-free march per direction; point sources natural |
| status | done, verified (§5, §7) | planned (§6) |

**Why the steady baseline is kept.** It is the $t\to\infty$ limit of the
transient model; for a single grid-aligned direction it is exact (the upwind
matrix is block-triangular, GMRES converges in one sweep) and its gradient is
machine-precision-correct (§5). Everything in it is reusable: the DG-upwind
operator $T$, the source coupling $S$, and the MMA constraint wiring carry over
unchanged to the transient solver.

**Where it is expected to give way (the planned experiments).** The advisor's
note flags two issues explicitly — *"why the steady state is problematic"* and
*"why the discrete adjoint is problematic"* — and proposes the time domain
instead. The regimes that expose this:

* **Sinogram / many directions.** A real constraint needs many ray directions.
  The steady operator must be rebuilt and solved **per direction**; for general
  (non-grid-aligned) directions on unstructured meshes the characteristic sweep
  ordering can have **cycles**, so there is no triangular solve and each
  direction is a separate non-symmetric system. The explicit time march is
  matrix-free and identical for every direction.
* **Point sources.** Fan/cone-beam sources are naturally a spreading
  characteristic field — a time-dependent process — rather than a single
  parallel steady sweep.
* **Discrete-adjoint consistency.** The discrete adjoint (transpose of the
  upwind operator) gives the *exact discrete* gradient, but it is **not** in
  general a consistent discretization of the continuous adjoint advection,
  especially for an **outflow-trace** (per-ray) functional whose sensitivity is
  a boundary quantity; the continuous (optimize-then-discretize) adjoint of the
  transient problem is the cleaner, more general route.

The remaining sections document the baseline in detail (§3–§5), report its
results (§7), and lay out the transient formulation and its adjoints (§6).

---

## 3. Steady-state baseline — continuous formulation

### 3.1 Why not the naive functional $\int_\Omega \mathbf v\cdot\nabla\rho_p$

For a **constant** $\mathbf v$ the divergence theorem collapses it to a boundary
integral,

$$\int_\Omega \mathbf v\cdot\nabla\rho_p\,dx=\oint_{\partial\Omega}\rho_p\,(\mathbf v\cdot n)\,ds,$$

which is **blind to the interior** and $\approx0$ when the density is void near
$\partial\Omega$. Discretely the same thing happens through **discrete
conservation**: a consistent DG flux is single-valued per face, so testing the
transport operator against the constant telescopes every interior face away.
Nonzero element-interface values do not rescue it. The fix is to make the
density a **source** of an accumulation field, so the constrained quantity lives
in the interior.

### 3.2 The accumulation field

With $\mathbf v=\beta$ a constant unit vector and $\rho_p=\tilde\rho$ the
(filtered) density, the accumulation field $\rho_a\equiv m$ solves the steady
advection

$$\boxed{\;\beta\cdot\nabla m=\tilde\rho\ \text{ in }\Omega,\qquad m=0\ \text{ on }\Gamma_{\rm in}=\{x\in\partial\Omega:\beta\cdot n<0\}.\;}$$

Integrating along a characteristic $\dot x=\beta$ gives $\mathrm dm/\mathrm ds=\tilde\rho$, so
$m(x)=\int_{\Gamma_{\rm in}}^{x}\tilde\rho\,\mathrm ds$ is the material upstream of
$x$ along $\beta$ — i.e. the **running line integral** of §1.

**Baseline constrained quantity.** This implementation caps the global
proxy

$$g_{\rm dm}(\rho)=\int_\Omega m\,dx\ \le\ M^\star ,\qquad M^\star=\texttt{dmf}\cdot M_{\rm ref},$$

with $M_{\rm ref}$ the value at the initial uniform design. In 1-D
($\beta=e_x$, $[0,L]$), $\int_0^L m\,dx=\int_0^L\tilde\rho(s)(L-s)\,ds$ — a
**position-weighted mass** penalizing material far upstream along $\beta$. (This
is the quantity originally requested; the **per-ray** inspectability functional
of §1/§6 differs — it bounds the outflow trace $m$ of *each* ray, not this
domain integral. See §6.)

**Optimization problem.** With compliance $c(\rho)=\ell(u)$ unchanged,

$$\min_{\rho}\ c(\rho)\quad\text{s.t.}\quad a_\rho(u,v)=\ell(v)\,\forall v,\quad
\int_\Omega\tilde\rho\,dx\le\bar v|\Omega|,\quad
\int_\Omega m\,dx\le M^\star,\quad 0\le\rho\le1 .$$

## 4. Steady-state baseline — sensitivity (affine, discrete adjoint)

$m$ is **linear** in $\tilde\rho$, $\int_\Omega m\,dx$ is linear in $m$, and
$\tilde\rho$ is linear in $\rho$, so $g_{\rm dm}$ is **affine** and its gradient
is a **constant vector**, computed once. Discretely, with transport matrix $T$,
source coupling $S$, DG volume weights $c_m$ ($c_m^{\mathsf T}m=\int_\Omega m$):

$$T\,m=S\,\tilde\rho,\quad g_{\rm dm}=c_m^{\mathsf T}m
\;\Longrightarrow\;
\frac{\partial g_{\rm dm}}{\partial\tilde\rho}
=c_m^{\mathsf T}T^{-1}S=\big(\underbrace{S^{\mathsf T}T^{-\mathsf T}c_m}_{=\ \tilde w}\big)^{\!\mathsf T},$$

i.e. solve the **discrete adjoint transport** $T^{\mathsf T}\lambda=c_m$ once and
set $\tilde w=S^{\mathsf T}\lambda$; then $\partial g_{\rm dm}/\partial\rho=M\,F^{-1}\tilde w$
(the filter chain rule), constant. $T^{\mathsf T}$ is the discrete **downwind
sweep** (transport in $-\beta$) — the exact transpose, which is why the FD check
passes to machine precision (§5) but also why it is only the *discrete* adjoint
(see §2 and §6 for the continuous one).

## 5. Steady-state baseline — DG-upwind discretization & wiring

The transport lives in a **broken $L^2$ (DG) space** $V_h$. The steady upwind DG
form of $\beta\cdot\nabla m=\tilde\rho$ with $m=0$ on $\Gamma_{\rm in}$ is

$$a_h(m,w)=-\sum_K\int_K m\,\beta\cdot\nabla w
+\sum_{F\in\mathcal F_{\rm int}}\int_F (\beta\cdot n)\,\widehat m\,[\![w]\!]
+\sum_{F\subset\Gamma_{\rm out}}\int_F (\beta\cdot n)\,m\,w
=\int_\Omega\tilde\rho\,w,$$

with $\widehat m$ the **upwind** trace; the homogeneous inflow $m=0$ is the
natural BC (no inflow load assembled), and the outflow term is on boundary faces.
In MFEM building blocks (the `ex9`/`ex9p` integrators, opposite sign since `ex9`
advances $\dot u=-\beta\cdot\nabla u$):

```
T = ConvectionIntegrator(beta, +1)                          // volume  INT (beta.grad m) w
  + NonconservativeDGTraceIntegrator(beta, +1)  on interior + boundary faces   // upwind flux
S = MixedScalarMassIntegrator   (H1 density  ->  DG test)   // (S rho~)_i = INT rho~ phi_i^DG
c_m,i = INT phi_i^DG                                        // DG volume weights
```

Forward and adjoint transport use **GMRES + BlockILU** (block size = DG
dofs/element; the parallel solver uses the MPI communicator for global inner
products). For grid-aligned $\beta$ the upwind matrix is block-triangular, so the
solve is essentially a one-pass sweep.

**Optimizer wiring.** Both constraints are affine, $g_i(\rho)=\mathrm dg_i\cdot\rho-1$,
with constant gradients $\mathrm dg_{\rm vol}=v/V^\star$ and
$\mathrm dg_{\rm dm}=M F^{-1}\tilde w/M^\star$, handed to **MMA**
(`MMAUpdaterParDM`, two inequality constraints). No per-iteration transport solve
is required — only the one adjoint solve at setup.

## 6. Transient formulation (proposed)

This is the advisor's target formulation; it is **not yet implemented**. It keeps
the same advection model but solves it in time and differentiates it
continuously.

### 6.1 Single ray and the field equation

Along one ray (arc length $\xi$), $\dfrac{\mathrm d\rho_a}{\mathrm d\xi}=\rho_p$,
so $\rho_a(l)$ is the full line integral. Solving all parallel rays of direction
$\mathbf v$ at once gives the **time-dependent field equation**

$$\dot\rho_a(\mathbf x,t)+\mathbf v\cdot\nabla\rho_a(\mathbf x,t)=\rho_p(\mathbf x),\quad
\mathbf x\in\Omega,\ \ \rho_a(\cdot,0)=0,$$

marched to $T$ **larger than the longest projection of $\Omega$ onto $\mathbf v$**.
For unit $\mathbf v$ the steady state along a ray is reached at $t=l$ and stays
fixed afterward (for compactly supported $\rho_p$), so $\rho_a(\cdot,T)$ holds the
line integrals. Discretize in space with the **same DG-upwind operator** as §5
and in time with an **explicit SSP-RK** step (matrix-free; the DG mass matrix is
block-diagonal). Point/fan/cone sources extend this naturally.

### 6.2 Per-ray inspectability constraint

Faithful to §1, bound the **outflow trace** of $\rho_a$ for every ray — i.e.
$\rho_a$ on $\Gamma_{\rm out}$ (or $\rho_a(\cdot,T)$) — by an upper threshold (and
optionally a lower one for $[I_{\min},I_{\max}]$), over all directions in the
sinogram. The many per-ray bounds are aggregated for MMA with a smooth max
(KS / $p$-norm). This replaces the baseline's single $\int_\Omega m\,dx$.

### 6.3 Continuous adjoints

For a functional $j(\rho_a)$ with the design entering through the source
$\rho_p$:

* **Steady** $\mathbf v\cdot\nabla\rho_a=\rho_p$, $\rho_a=0$ on $\Gamma_{\rm in}$:
  the continuous adjoint is the **backward advection**

  $$-\mathbf v\cdot\nabla\lambda=\frac{\partial j}{\partial\rho_a}\ \text{ in }\Omega,\qquad
  \lambda=0\ \text{ on }\Gamma_{\rm out},\qquad
  \frac{\mathrm dj}{\mathrm d\rho_p}=\lambda .$$

  Its DG-upwind discretization (upwinding w.r.t. $-\mathbf v$) is *not* identical
  to the transpose $T^{\mathsf T}$ used by the baseline; the mismatch — most
  visible for the outflow-trace functional — is the "discrete adjoint is
  problematic" point.

* **Transient** $\dot\rho_a+\mathbf v\cdot\nabla\rho_a=\rho_p$ on
  $\Omega\times(0,T)$: the adjoint runs **backward in time and space**,

  $$-\dot\lambda-\mathbf v\cdot\nabla\lambda=\frac{\partial j}{\partial\rho_a},\qquad
  \lambda(\cdot,T)=0,\quad \lambda=0\ \text{ on }\Gamma_{\rm out},\qquad
  \frac{\mathrm dj}{\mathrm d\rho_p}(\mathbf x)=\int_0^T\lambda(\mathbf x,t)\,\mathrm dt .$$

  (Because the forward problem is linear, $\partial j/\partial\rho_a$ for a
  linear/aggregated functional does not depend on the forward trajectory, so the
  adjoint march needs no checkpointing in this linear case — the cost is the
  reverse-time advection itself.)

### 6.4 What changes vs. the baseline

Reused unchanged: the DG space and the upwind operator (§5), the source coupling
$S$, the filter chain rule, and the MMA constraint plumbing. New: an explicit
time integrator marched to $T$; the **continuous** adjoint (a second
time-marched advection); the **per-ray outflow** functional with KS/$p$-norm
aggregation; and a loop over **multiple directions** (sinogram) / point sources.

---

## 7. Results (steady baseline)

### 7.1 Correctness (`verify_dirmass.exe`, native serial; `-p check`, parallel)

| test | quantity | result |
|------|----------|--------|
| transport unit ($\beta=(1,0)$, $\tilde\rho=1\Rightarrow m=x$) | $\lVert m-x\rVert_{L^2}$ | $1.9\times10^{-14}$ |
| adjoint consistency | $\lvert c_m^{\mathsf T}(T^{-1}S\tilde\rho)-\tilde w^{\mathsf T}\tilde\rho\rvert/\lvert\cdot\rvert$ | $2.6\times10^{-15}$ (serial), $5.6\times10^{-16}$ (np=4) |
| gradient FD | best rel. error vs central difference | $6.3\times10^{-12}$ (serial), $8.9\times10^{-13}$ (np=4) |

The adjoint consistency check confirms the **discrete** adjoint is the exact
transpose, $S^{\mathsf T}T^{-\mathsf T}c_m = c_m^{\mathsf T}T^{-1}S$.

### 7.2 Cantilever study (`-p opt -opt mma`, 120×60, $\bar v=0.5$, $\beta=(1,0)$)

Same cantilever as the baseline (clamped $x=0$, traction patch at mid-right).
$\beta=(1,0)$ weights material near the **clamp** most; the unconstrained
two-chord optimum thickens the chords there, so its directional mass is **above**
the uniform reference ($M/M_{\rm ref}=1.091$) — the constraint bites where the
structure wants material.

| budget $M^\star$ | volume | dir. mass $M/M_{\rm ref}$ | compliance $c$ | $c/c_{\rm base}$ |
|:--|:--:|:--:|:--:|:--:|
| none (`dmf`$=5$, inactive) | 0.500 | 1.091 | $6.98\times10^{-1}$ | 1.00 |
| $0.9\,M_{\rm ref}$ | 0.463 | **0.900** | $8.25\times10^{-1}$ | 1.18 |
| $0.7\,M_{\rm ref}$ | 0.359 | **0.700** | $1.16\times10^{0}$  | 1.66 |

The directional mass hits its budget **exactly**; tightening it pushes material
off the high-weight clamp region and **raises compliance** (1.18× at 0.9, 1.66× at
0.7). Once the directional constraint binds it becomes the **governing**
constraint and the **volume goes slack** (0.50 → 0.46 → 0.36): the optimum uses
*less* total material, placed to respect the directional cap. (For fixed-volume
comparison, encode the volume budget as an equality — `MMAOptimizer::WithEqualities`
in [MMA_MFEM.hpp](MMA_MFEM.hpp).)

---

## 8. Build & run

**Native serial verifier (MSYS2 UCRT64):**

```bash
g++ -O3 -std=c++17 -D_USE_MATH_DEFINES -I./mfem verify_dirmass.cpp \
    -o verify_dirmass.exe -L./mfem -lmfem -lws2_32 -static
./verify_dirmass.exe -nx 80 -ny 40 -bx 1 -by 0
```

**Parallel driver (WSL toolchain, MMA enabled):**

```bash
bash build_dirmass.sh                                  # -> elasticity_topopt_par

mpirun -np 4 ./elasticity_topopt_par -p check -dm                 # FD + adjoint tests
mpirun -np 4 ./elasticity_topopt_par -p opt -opt mma             # baseline (no -dm)
mpirun -np 4 ./elasticity_topopt_par -p opt -dm -bx 1 -by 0 -dmf 0.9   # constrained
```

Options: `-dm`/`-no-dm` enable the constraint, `-bx -by` the direction $\beta$,
`-dmf` the budget as a fraction of the initial (uniform-design) directional mass.
The constraint requires MMA (multi-constraint); OC handles only the volume budget.
