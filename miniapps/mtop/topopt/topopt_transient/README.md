# Transient Topology Optimization

This directory contains a transient elastodynamics topology-optimization driver
for MFEM. The current code supports multiple forward problems and objective
functionals through a small problem interface, while keeping the RK4
forward/adjoint machinery in the solver layer.

The active executable is `TopOptTransient`.

The continuous adjoint, both nested forward/adjoint grid directions, full or
REVOLVE trajectory storage, and consistent mass are implemented. The
verification record and roadmap for the remaining physical 2D/3D studies are in
[CONTINUOUS_ADJOINT_IMPLEMENTATION_PLAN.md](CONTINUOUS_ADJOINT_IMPLEMENTATION_PLAN.md).

## What This Code Solves

The optimizer updates a material density field `rho` subject to a volume
constraint:

```text
minimize    J(rho)
subject to  M(rho) u_tt + C u_t + K(rho) u = f(t)
            int_Omega rho dx / target_volume - 1 <= 0
            1 - int_Omega rho dx / target_volume <= 0
            0 <= rho <= 1
```

The density workflow is:

```text
raw control rho (L2)
   -> Helmholtz filter
filtered rho_tilde (H1)
   -> SIMP mass/stiffness interpolation
transient elastodynamics
   -> time-integrated objective J
selected discrete RK4 adjoint or continuous RK4/Hermite adjoint
   -> dJ / d rho_tilde
filter transpose
   -> dJ / d rho
MMA
   -> next design
```

## Current Problems

Select the forward problem with `-problem`.

### `wave`

Wave-shielding reference problem.

- Mesh: `lamb-problem-damping-mesh-triangs.msh` by default.
- Load: Gaussian downward boundary traction on load-strip boundary attributes.
- Damping: boundary sponge plus absorbing boundaries.
- Objective: `DisplacementL2Objective`, minimizing time-integrated displacement
  energy in a protected circular region.

Example:

```bash
mpirun -np 8 ./TopOptTransient -problem wave -r 0 -o 1 -tf 0.3 -dt 1e-4 -vf 0.5 -fr 0.03 -mi 150 -mv 0.2 -pv
```

### `band-waveguide`

Generated 2D lift of a 1D transient waveguide/band-gap reference problem.

- Mesh: generated long, thin `8 x 0.5` rectangular band.
- Load: Gaussian-modulated carrier pulse on a narrow vertical strip at the
  center.
- Load direction: axial `[1, 0]`; the localized source launches waves both
  left and right.
- Carrier: frequency `5.0`; by default the modulated-Gaussian envelope spans
  the full simulated interval (`duration = t_final`). Use `-dur` to override it.
- Damping: sponge and absorbing-boundary impedance on the left/right ends only.
- Objective: `DisplacementL2Objective`, minimizing time-integrated displacement
  energy in symmetric rectangular receiver regions on both sides of the source.
- Material: same SIMP law, but with a raised material floor to discourage one
  near-disconnecting slit and favor multiple impedance interfaces.

Example:

```bash
mpirun -np 8 ./TopOptTransient -problem band-waveguide -lumped-mass -damp -r 0 -o 1 -tf 3.5 -dt 5e-4 -vf 0.5 -fr 0.05 -mi 100 -mv 0.2 -pv
```

### `band-mode-converter` and `band-mode-converter-reverse`

Generated `12 x 1` periodic-y spectral-separation pilots on a `384 x 64`
quadrilateral mesh. They preserve the legacy `band-waveguide` and use the
one-way layout

```text
left sponge | input collar | active design | output collar | right sponge
  0--0.75     0.75--4.25     4.25--8.00      8.00--11.25    11.25--12.00
                 source=1--4                    target=8.25--10.75
```

- The raw design in the collars and sponges is pinned to `rho=1`; only the
  middle section is subject to the active-region volume constraint. The global
  Helmholtz filter can create a short `rho_tilde` transition at each active
  interface, so the source and target begin five filter radii inside their
  passive collars at the default `r=0.05`.
- The left boundary is clamped behind its sponge, the right boundary has an
  absorbing impedance, and `y=0`/`y=1` are periodic.
- Source and target shapes use L2-normalized `sin^2` axial windows. The output
  objective also uses a smooth `sin^2` weight rather than a hard receiver edge.
- `band-mode-converter` launches the transverse fundamental and uses harmonic
  displacement tracking of `cos(n*pi*y/H)` in the output. This is the intended
  coarse-forward/fine-adjoint allocation.
- `band-mode-converter-reverse` launches mode `n` and maximizes harmonic
  correlation with the output fundamental. Its state-independent adjoint RHS
  remains low mode for the fine-forward/coarse-adjoint allocation.
- The default is `n=8`; select another positive even periodic mode with
  `-tm/--target-mode`, and set target/correlation amplitude with
  `-ta/--target-amplitude`. The default carrier is `f=5` and the default pulse
  duration is `1`; `-freq` and `-dur` override them. The objective carrier phase
  is aligned with the centered source carrier before propagation phase.

Initial spatial-spectrum audit:

```bash
./TopOptTransient -problem band-mode-converter -rhs-spectrum \
  -tm 8 -r 0 -o 1 -do 1 -tf 3.5 -dt 1e-3 -vf 0.5 \
  -lumped-mass -no-pv -out band_mode_spectrum_n8
```

At the uniform initial design, the resolved default produced 95%-energy spatial
frequencies `0.7015` (forward RHS) and `4.0054` (rest-state adjoint RHS), a
ratio of `5.710`. Reversing the roles produced `3.9941` and `0.7994`, i.e. a
forward/adjoint ratio of `4.997`. The original compact `8 x 0.5` sketch failed
this gate (ratios only `1.31--2.34`), which is why the passive collars were
lengthened before optimization work.

A solid-guide (`-vf 1`) forward-only check at `T=3.5`, after the leading
response has entered the target window but before the centroid-based envelope
peak estimate near `T=4`, gave terminal output
fundamental projections `1.8781526267e-4` at `dt=0.004` and
`1.8781530735e-4` at `dt=0.002`, a relative change of `2.38e-7`. The printed
tracking objective was `1.45833334` on both grids, although at the homogeneous
design it is dominated by the prescribed high-mode target baseline. This
supports the coarse forward observable; it does not yet establish adjoint-grid
convergence.

One-iteration continuous-adjoint wiring run:

```bash
mpirun -np 4 ./TopOptTransient -problem band-mode-converter \
  -tm 8 -r 0 -o 1 -do 1 -tf 0.008 -dt 1e-3 -vf 0.5 -fr 0.05 \
  -mi 1 -tol 0 -lumped-mass -adjoint-mode continuous \
  -adjoint-refinement 2 -trajectory-storage revolve -nchk 1 -no-pv \
  -out band_mode_wiring
```

These are deliberately manufactured Fourier-mode pilots. An axial-only cosine
is not by itself a power-normalized traveling elastic guided mode: a physical
mode also needs the correct polarization, axial phase, and sine/cosine
quadratures. Moreover, `-rhs-spectrum` is a rest-state spatial diagnostic
(using the selected production mass; the numbers above use lumped mass); it
does not measure temporal bandwidth or the state-dependent tracking residual
during optimization. A large RHS ratio is therefore an eligibility check, not
evidence for a timestep ratio. Independent
forward/adjoint temporal refinement, propagation/phase, mesh, sponge, Taylor,
and full-versus-REVOLVE checks are required before making the physical or
multirate-efficiency claim.

### Same-grid RK4 DO/OD experiment

`band-mode-converter-correlation` keeps the normal low-mode source and replaces
the tracking functional by signed harmonic correlation with target mode 8. The
opt-in `-objective-quadrature rk4-stage` functional is

```text
J_h = sum_n dt sum_i b_i ell(Y_i^n,t_n+c_i dt),
b = (1/6, 1/3, 1/3, 1/6).
```

It is available in both production gradient paths. With
`-adjoint-mode discrete`, objective seeds are reversed through all four RK4
stages and the result is the exact DO derivative. With
`-adjoint-mode continuous -adjoint-refinement 1`, the same reported objective
is paired with the deliberately naive continuous RK4/Hermite gradient.

`-rk4-adjoint-comparison` (currently restricted to `-do 1`, hence a raw Q0
control) evaluates the common objective once and writes
`rk4_adjoint_comparison.csv` for DO, a separately implemented transformed
partitioned adjoint (`OD_modified`), and `OD_naive_Hermite`. Optional
`-rk4-taylor-levels N` writes a volume-neutral design Taylor test. Raw-gradient
errors use the active Q0 design's L2-dual metric, including element-volume
Riesz weights.

For classical RK4, reverse RK4 supplied directly with `Y4,Y3,Y2,Y1` is already
the transformed/DO recurrence. Therefore the naive row intentionally samples
the accepted/Hermite forward trajectory; it must not be described as a
tableau-only comparison. The complete non-overwriting study harness and exact
spatial choices are documented in
`studies/rk4_do_od_band_converter_20260810/README.md`.

### `cantilever-compliance`

Generated 2D cantilever beam, modeled after the static topology-optimization
miniapp but driven by the transient solver.

- Mesh: generated Cartesian `3 x 1` beam.
- Boundary condition: left edge clamped.
- Load: constant concentrated downward body force near the free tip.
- Damping: optional uniform mass-proportional damping for dynamic relaxation.
- Objective: `ComplianceObjective`, minimizing time-integrated `int f . u`.

Recommended current command:

```bash
mpirun -np 8 ./TopOptTransient -problem cantilever-compliance -lumped-mass -damp -tf 10 -dt 1e-3 -vf 0.5 -fr 0.05 -mi 100 -mv 0.2 -tol 1e-4 -pv
```

`-lumped-mass` is usually the faster explicit path. `-iterative-mass` is also
supported with the clamped degrees of freedom eliminated from the
consistent-mass solve.

### `spherical-bandgap`

3D spherical wave-shielding problem on concentric spherical shells.

- Mesh: `spherical_bandgap.msh`, generated from `spherical_bandgap.geo`
  (see Meshes below). Element attributes: 1 source, 2 design, 3 receiver,
  4 gap, 5 damping; boundary attribute 100 is the outer r = 10 sphere.
- Load: unit-amplitude radial monopole body force in the central sphere
  (r < 0.5) with a modulated-Gaussian tone burst. Default carrier frequency
  `1.0` (lambda_p = 2.0 at c_p = 2, ~6-7 linear elements per wavelength on
  the lc = 0.3 mesh); by default its envelope spans the full simulated
  interval (`duration = t_final`).
  Higher carriers pack more Bragg bands into the design shell and give a
  richer band-gap result - override with `-freq`, but pair it with a finer
  mesh (`-freq 5` needs lc ~= 0.06, HPC scale). The default is the cheap
  local operating point.
- Damping: radial sponge in the outer shell (7.5 < r < 10) via
  `SphericalDampingField`, plus absorbing impedance on boundary 100. The
  sponge uses the analytic harmonic coordinate of a 3D spherical shell and is
  normalized against the passive SIMP mass so its requested amplitude target
  applies across the implemented ramp.
- Objective: `DisplacementL2Objective` over the receiver shell (6 < r < 7).
- Passive regions: source, receiver, gap, and damping shells are frozen at
  the volume-fraction density; only the design shell (0.5 < r < 6) is
  optimized.
- Timing: P-arrival at the receiver is t ~= 3. With the default
  `duration = t_final`, the envelope peaks at `t_final/2`; `-tf 9` lets that
  peak reach the receiver near t = 7.5. Forcing remains active through the
  final time, but energy emitted during the final ~3 time units necessarily
  cannot reach the receiver before the simulation ends. `Validate()` warns
  when `-tf` is shorter than even the first-arrival time.

Example (cluster-scale; use the coarse mesh for local wiring tests):

```bash
mpirun -np 8 ./TopOptTransient -problem spherical-bandgap -lumped-mass -damp -tf 9.0 -dt 1e-3 -vf 0.5 -fr 0.3 -mi 100 -mv 0.2 -pv
```

### `mode-converter-3d`

Fixed-density spectral pilot for comparing the forward and adjoint
right-hand sides.

- Mesh: generated `6 x 1 x 1` hexahedral waveguide.
- Forward source: smooth full-length fundamental axial body mode.
- Objective: harmonic tracking of a fourth-by-fourth transverse axial mode in
  the output collar.
- Boundary conditions: clamped left end, right sponge and absorbing boundary.
- Density: homogeneous `rho = rho_tilde = 1`.
- Diagnostic: `-rhs-spectrum` applies an M-inner-product Lanczos projection to
  `M_L^{-1} K` starting separately from `M_L^{-1} F` and
  `M_L^{-1} (dJ/du)`. It writes eigenvalue/frequency weights to
  `rhs_spectrum.csv`. With `-pv`, it also writes normalized spatial
  representatives `M_inv_F` and `M_inv_dJdu`.

Example:

```bash
./TopOptTransient -problem mode-converter-3d -rhs-spectrum \
  -r 0 -o 1 -do 1 -tf 1 -dt 1e-2 -lumped-mass -pv -out rhs_spectrum_run
```

This diagnostic compares source-resolved accuracy scales. The strict explicit
CFL limit remains the full operator's global spectral limit.

### `mode-converter-reverse-3d`

Fixed-density mirror diagnostic for the opposite allocation: a smooth,
full-length fourth-by-fourth forward source and a full-length fundamental-mode
correlation objective. The resulting adjoint source is independent of the
forward state and remains low-spectrum, so this is the appropriate pilot for
testing a fine forward grid with a coarser continuous-adjoint grid. It is not
yet a physical topology-optimization problem.

Unlike the tracking pilot, this objective has no target carrier tied to
`-freq`. The spectrum audit is purely spatial, so it does not need a carrier
override. Choose the forward carrier and spatial mesh together for a temporal
study; a high carrier must meet the usual spatial-resolution requirement.

```bash
./TopOptTransient -problem mode-converter-reverse-3d -rhs-spectrum \
  -r 0 -o 2 -do 1 -tf 1 -dt 1e-3 -lumped-mass \
  -out rhs_spectrum_reverse_run
```

The spectrum/CFL check establishes eligibility only. A separate temporal
convergence study must show that the fine forward step is accuracy-limited and
that the proposed coarser adjoint step remains below the same global RK4 CFL
limit.

For that forward-only check, add `-forward-modal-probe`. It reports the
terminal projection onto the driven `(4,4)` mode without changing the
fundamental-mode objective or its adjoint RHS. For example:

```bash
./TopOptTransient -problem mode-converter-reverse-3d -forward-only \
  -forward-modal-probe -r 0 -o 2 -do 1 -tf 0.2 -freq 1.5 \
  -dt 3.125e-3 -lumped-mass -out reverse_forward_dt3125e5
```

Both nested-grid directions use the same continuous state-adjoint RK4 kernel.
For this fixed-density fine-forward/coarse-adjoint diagnostic, an even ratio
`q = dt_a/dt_f` places every coarse-adjoint endpoint and midpoint stage on an
actual stored forward node. The production design path supports arbitrary
integer `q` and uses cubic-Hermite dense output where its adjoint or fine-grid
design quadrature requests off-node values.

```bash
./TopOptTransient -problem mode-converter-reverse-3d \
  -continuous-adjoint-check -adjoint-coarsening-max 16 \
  -r 0 -o 2 -do 1 -tf 0.2 -dt 7.8125e-4 -freq 1.5 \
  -lumped-mass -out reverse_continuous_convergence
```

The order-2 run above uses `q=2` as the finest interpolation-free reference:

| `q` | `dt_a` | relative error in `p(0)` | observed order |
|---:|---:|---:|---:|
| 2 | `0.0015625` | reference | -- |
| 4 | `0.003125` | `3.366851e-8` | -- |
| 8 | `0.00625` | `5.332438e-7` | `3.985` |
| 16 | `0.0125` | `6.967555e-6` | `3.708` |

All adjoint steps pass the same operator-level RK4 stability check; the
recommended ceiling for this configuration is `0.01465907`. Results are
written to `continuous_adjoint_coarsening.csv`. This command remains a
fixed-density, full-storage state-adjoint diagnostic. Continuous design
accumulation, full/REVOLVE storage, and adjoint coarsening are also available
through the production optimizer described below.

The reverse pilot intentionally has linear state dynamics and a linear
compliance objective, so its adjoint is independent of the numerical forward
state values. It proves stage alignment and that the low-spectrum adjoint can
be marched coarsely, but it is not an interpolation validation.

### Coarse-forward / fine-continuous-adjoint Hermite diagnostic

The complementary full-storage path stores coarse RK4 endpoint states and
evaluates the physical forward RHS at every endpoint. Piecewise cubic Hermite
polynomials then reconstruct the full first-order state at every fine-adjoint
endpoint and midpoint stage:

```bash
./TopOptTransient -problem mode-converter-3d \
  -continuous-adjoint-refinement-check -adjoint-refinement-max 16 \
  -r 0 -o 2 -do 1 -tf 0.2 -dt 1.25e-2 -freq 1.0 \
  -lumped-mass -out jobs/mode_converter_hermite_refinement_20260728
```

This run uses `N_f=16`, `dt_f=0.0125`, and
`m=dt_f/dt_a={1,2,4,8,16}`. The coarse forward step is below the common
order-2 lumped-RK4 safety recommendation `0.01465907`.

| `m` | `N_a` | `dt_a` | relative `p(0)` error vs. `m=16` | adjacent observed order |
|---:|---:|---:|---:|---:|
| 1 | 16 | `0.0125` | `2.193423e-4` | `3.921` |
| 2 | 32 | `0.00625` | `1.449658e-5` | `3.969` |
| 4 | 64 | `0.003125` | `9.220664e-7` | `3.988` |
| 8 | 128 | `0.0015625` | `5.468819e-8` | -- |
| 16 | 256 | `0.00078125` | reference | -- |

The order is computed from adjacent self-differences
`||p_m-p_(2m)||/||p_(2m)||`, rather than from the finest-reference column.
The same command compares the reconstructed trajectory with a freshly
integrated RK4 trajectory on a `32`-times-finer grid, including every
`m=16` adjoint midpoint:

| full-state relative RMS | displacement relative RMS | velocity relative RMS |
|---:|---:|---:|
| `4.397263e-6` | `3.224967e-6` | `4.402155e-6` |

Results are in
`jobs/mode_converter_hermite_refinement_20260728/continuous_adjoint_refinement.csv`
and `cubic_hermite_reconstruction_audit.csv`.

An independent forward-grid study repeated the audit with a four-times-finer
RK4 reference for each listed `dt_f`:

| `dt_f` | full-state RMS | displacement RMS | velocity RMS | state order |
|---:|---:|---:|---:|---:|
| `0.0125` | `4.379905e-6` | `3.162835e-6` | `4.385049e-6` | `3.993` |
| `0.00625` | `2.751112e-7` | `2.008255e-7` | `2.754232e-7` | `4.001` |
| `0.003125` | `1.717780e-8` | `1.263423e-8` | `1.719683e-8` | `4.001` |
| `0.0015625` | `1.073024e-9` | `7.919977e-10` | `1.074200e-9` | -- |

The displacement orders are `3.977`, `3.991`, and `3.996`; the velocity
orders are `3.993`, `4.001`, and `4.001`. These runs are under
`jobs/mode_converter_hermite_dt*_20260728` (the corrected `dt=0.003125`,
`f=1` run uses suffix `_f1_20260728`).

Cubic Hermite has polynomial degree three but fourth-order pointwise state
accuracy. The verification executable reproduces an exact cubic to
`5.551115e-16`; using actual RK4 harmonic-oscillator endpoints and recomputed
physical endpoint slopes, the finest off-node error is `5.186091e-10` and the
minimum observed order is `3.989725`. Thus the expected coupled state-adjoint
error is `O(dt_f^4 + dt_a^4)`.

The tracking derivative is `2 chi (u-u_target)`. Its prescribed target can
dominate the smaller state-dependent term, so the adjoint table primarily
validates the fine reverse RK4 march. The direct trajectory audit, oscillator
test, and independent forward-grid study validate reconstruction. This remains
a fixed-density state-adjoint convergence diagnostic. The coarse-forward /
fine-adjoint design path described below now performs continuous design
accumulation and checkpoint replay. At every design-gradient stage it
reevaluates the physical forward RHS at the reconstructed state; it never
obtains acceleration by differentiating the cubic.

### Continuous-adjoint optimization on either nested grid

The production optimizer has an opt-in continuous path. Coarse forward/fine
adjoint uses `m=N_a/N_f`:

```bash
./TopOptTransient -problem cantilever-compliance -iterative-mass \
  -adjoint-mode continuous -adjoint-refinement 2 \
  -trajectory-storage revolve -nchk 1 \
  -r 0 -o 1 -do 1 -tf 0.002 -dt 1e-3 \
  -mi 1 -no-pv -out continuous_refined_adjoint
```

Fine forward/coarse adjoint uses `q=N_f/N_a`:

```bash
./TopOptTransient -problem cantilever-compliance -iterative-mass \
  -adjoint-mode continuous -adjoint-coarsening 2 \
  -trajectory-storage revolve -nchk 1 \
  -r 0 -o 1 -do 1 -tf 0.002 -dt 1e-3 \
  -mi 1 -no-pv -out continuous_coarse_adjoint
```

Only one of `m` and `q` may exceed one. Use
`-trajectory-storage full` for a verification/debug trajectory; omit `-nchk`
in that mode. Both selected steps must pass the RK4 stability estimate for the
chosen lumped or consistent mass operator. The driver enforces the 80%-safe
componentwise wave/damping estimate because satisfying both raw scalar RK4
endpoints is not sufficient when damping and oscillation act together.

For an adjoint-finer run, REVOLVE schedules `N_f` one-forward-interval blocks;
each reverse callback consumes `m` adjoint RK4 steps. For a forward-finer run,
REVOLVE instead schedules `N_a` blocks; each callback replays `q` fine forward
intervals and consumes one coarse adjoint step. The coarse adjoint's accepted
endpoints/slopes define cubic-Hermite dense output, and the design contraction
is integrated with fourth-order Simpson quadrature on every fine forward
interval. Controller replay and local replay are reported as both block and
equivalent fine-interval counts.

The continuous verification records:

- adjoint-finer full storage versus REVOLVE agreement in objective, `p(0)`,
  filtered gradient, and raw filter-transposed gradient for `m={1,3}` and
  checkpoint counts one and two;
- an `m=3` fixed-`T` study with final-two orders `3.7303` for the objective,
  `3.9798` for `p(0)`, and `3.5691/3.5686` for the filtered/raw gradients;
- forward-finer odd `q=3` joint objective/state/design convergence and
  full/REVOLVE agreement for `q={2,3}` with poisoned block replay;
- constrained consistent-mass `q=3` full/REVOLVE relative differences of
  `2.20e-16` in the objective, `3.11e-16` in `p(0)`, and
  `1.02e-16`/`1.06e-16` in the filtered/raw gradients;
- optional consistent-mass `q=3`, `(N_f,N_a)=(96,32)` directional-FD errors
  `6.64e-8` filtered and `7.08e-8` raw.

Build and run the two dedicated temporal tests with:

```bash
make test_continuous_temporal_refinement test_continuous_forward_finer -j
mpirun -np 1 ./test_continuous_temporal_refinement
mpirun -np 1 ./test_continuous_forward_finer
mpirun -np 1 ./test_continuous_forward_finer -cfd  # optional FD audit
```

The exact discrete RK4/trapezoidal adjoint remains the default. A short
production validation has exercised both nested directions, full/REVOLVE
equality, consistent-mass restart, and persistent raw-design checkpoints. A
three-iteration Gaussian-initialized cantilever comparison uses consistent
mass, `q=2`, `(N_f,N_a)=(8,4)`, and one REVOLVE checkpoint. FULL and REVOLVE
histories and final raw checkpoint designs are byte-identical. From iterations
one to three, `J` changes from `2.27421376e-8` to `1.33037319e-8`, volume from
`0.422209` to `0.497632`, raw-gradient norm from `2.844942e-8` to
`1.650235e-8`, and filtered-gradient norm from `3.388296e-8` to
`1.960641e-8`. Including the retained checkpoint/trajectory and Hermite
reconstruction vectors, REVOLVE stores an estimated `0.2796440 MB/rank`
versus `0.5592651 MB/rank` for full storage. Its three-iteration
controller/local totals are `9/12` blocks and `18/24` equivalent fine
intervals.

Reproduce the three-iteration storage comparison and summarize it with:

```bash
./TopOptTransient -problem cantilever-compliance -r 0 -o 1 -do 1 \
  -tf 0.004 -dt 0.0005 -mi 3 -tol 0 -init gaussian -iterative-mass \
  -adjoint-mode continuous -adjoint-coarsening 2 \
  -trajectory-storage full -no-pv -out study_q2_full

./TopOptTransient -problem cantilever-compliance -r 0 -o 1 -do 1 \
  -tf 0.004 -dt 0.0005 -mi 3 -tol 0 -init gaussian -iterative-mass \
  -adjoint-mode continuous -adjoint-coarsening 2 \
  -trajectory-storage revolve -nchk 1 -no-pv -out study_q2_revolve

./analyze_optimization_histories.py \
  full=study_q2_full/optimization_history.txt \
  revolve=study_q2_revolve/optimization_history.txt
```

The same moving-design run also passes fresh-destination `-restart-from`. After
a one-iteration seed, the first resumed evaluation (iteration 2) exactly
matches the uninterrupted history at printed precision: `J=1.53354912e-8`,
volume `0.477546`, raw `1.930212e-8`, filtered `2.293974e-8`. Iteration 3
remains within about `6.1e-7` relative despite rebuilding the MMA state.

The remaining optimization work is the physical comparison: the tiny
band-waveguide wiring smoke ends before receiver arrival and has zero
objective/gradient, so the meaningful `T=3.5` band study and final 3D example
are still pending.

## Meshes

Meshes are generated artifacts (gitignored); regenerate them from the tracked
`.geo` sources:

```bash
# production spherical mesh (~250k tets, ~43k nodes)
gmsh -3 -format msh2 spherical_bandgap.geo -o spherical_bandgap.msh
# coarse variant for local smoke tests (pass it with -mesh)
gmsh -3 -format msh2 -clscale 2 spherical_bandgap.geo -o spherical_bandgap_coarse.msh
```

The spherical `.geo` builds the concentric shells with a single
`BooleanFragments` and classifies volumes/surfaces geometrically. Do not
replace this with chained `BooleanDifference` calls: OCC tags of disconnected
boolean results are unpredictable, which previously dropped the receiver
shell from the mesh entirely (objective identically zero) and put the
absorbing boundary on an interior surface.

## Build

From this directory in WSL:

```bash
make TopOptTransient test_adjoint_verification \
  test_continuous_temporal_refinement test_continuous_forward_finer -j8
```

From Windows PowerShell, using the repository path:

```powershell
wsl make -C /mnt/c/Users/cortescastil1/Desktop/mfem/miniapps/mtop/topopt/topopt_transient TopOptTransient test_adjoint_verification test_continuous_temporal_refinement test_continuous_forward_finer -j8
```

## Command-Line Options

Common options:

```text
-problem <name>              wave, band-waveguide, band-mode-converter,
                             band-mode-converter-correlation,
                             band-mode-converter-energy,
                             band-mode-converter-reverse,
                             cantilever-compliance, spherical-bandgap,
                             mode-converter-3d, or mode-converter-reverse-3d
-r,  --refine <int>          uniform refinement levels
-o,  --order <int>           forward/adjoint state H1 finite element order
-do, --design-order <int>    H1 order of rho_tilde; rho uses paired L2 order
                             max(0, design-order-1). Defaults to --order,
                             preserving the historical coupled discretization.
-tf, --t-final <real>        final simulation time
-dt, --time-step <real>      time step
-tm, --target-mode <int>     positive even transverse mode for the 2D converter
-ta, --target-amplitude <r>  2D converter target/correlation amplitude
-energy-low-penalty <r>      residual mode-0 penalty in the windowed
                             modal-energy band-converter objective
-energy-window-start <r>     start time of that output window
-energy-window-ramp <r>      sin-squared output-window ramp duration
-vf, --vol-frac <real>       target material volume fraction
-fr, --filter-radius <real>  Helmholtz filter radius
-mi, --max-it <int>          maximum MMA iterations
-mv, --move <real>           MMA move limit
-tol, --tol <real>           L1 design-change stopping tolerance
-init <mode>                 uniform, solid, void, gaussian, or modal-seed.
                             modal-seed is a volume-feasible target-mode
                             perturbation for the 2D band converter.
-mesh <file>                 mesh file for file-based problems
-pv / -no-pv                 enable or disable ParaView output
-damp / -no-damp             enable or disable problem damping
-forward-only                one filter + forward sweep; no MMA or adjoint.
-forward-modal-probe         with -forward-only, report the terminal projection
                             supplied by the selected problem.
-rhs-spectrum                compare spectral measures seen by F and dJ/du,
                             write rhs_spectrum.csv, then exit.
                             With -pv, write sampled RHS fields.
-continuous-adjoint-check    store the forward trajectory and run the
                             fine-forward/coarse-continuous-adjoint diagnostic.
-adjoint-coarsening-max <q>  largest power-of-two dt_a/dt_f ratio in that
                             diagnostic (minimum 2; q=2 is the reference).
-continuous-adjoint-refinement-check
                             store the coarse forward trajectory and run the
                             fine-continuous-adjoint diagnostic with cubic
                             Hermite reconstruction.
-adjoint-refinement-max <m>  test power-of-two dt_f/dt_a refinements through
                             this maximum (minimum 2); a 2m forward reference
                             audits every finest-adjoint RK4 stage time.
-adjoint-mode <mode>         optimization gradient: discrete (default) or
                             continuous. The continuous path uses
                             relation-aware REVOLVE blocks and cubic Hermite.
-objective-quadrature <q>    legacy (historical trapezoid/Simpson behavior) or
                             rk4-stage (common four-stage functional; same grid).
-rk4-adjoint-comparison      fixed-design DO / transformed / naive-Hermite
                             comparison, then exit.
-rk4-taylor-levels <n>       optional halved perturbation levels for that
                             comparison's volume-neutral Taylor test.
-rk4-taylor-linf-normalized  rescale that volume-neutral L2-Riesz direction
                             to unit active L-infinity norm, so epsilon is a
                             maximum raw-density perturbation.
-adjoint-refinement <m>      fine continuous-adjoint steps per coarse forward
                             interval; must be 1 in discrete mode.
-adjoint-coarsening <q>      fine forward steps per coarse continuous-adjoint
                             step; mutually exclusive with refinement > 1 and
                             must be 1 in discrete mode.
-trajectory-storage <mode>  revolve (default) or full; full is continuous-only
                             and retains every forward endpoint.
-iterative-mass              consistent mass solve with CG+AMG
-lumped-mass                 diagonal mass solve: row-sum on tensor-product
                             elements, positive scaled diagonal on high-order
                             triangles/tetrahedra
-freq <real>                 carrier frequency override (0 = problem default)
-dur <real>                  pulse duration override (0 = problem default)
-nchk <int>                  REVOLVE checkpoints per sweep (-1 = auto)
```

The driver prints a carrier-resolution report (elements per P-wavelength) and
warns when the mesh cannot resolve the requested frequency. Rule of thumb:
resolving a carrier at frequency `f` needs mesh size `h <~ c_p / (7 f)`.

The default mass path is `-iterative-mass`. For larger explicit runs,
`-lumped-mass` is usually much faster.

For high-order physics with a coarser design, specify both orders. For example,
`-o 8 --design-order 1` uses degree-8 displacement/adjoint fields while
retaining the historical degree-1 filter and degree-0 L2 control density.
High-order state runs support both mass paths. Tensor-product elements retain
row-sum lumping; high-order simplex elements use a positive scaled-diagonal
lump because their nodal row sums can be zero or negative. The startup CFL
estimate is computed from the selected lumped or consistent operator and checks
the larger of `dt_f` and `dt_a`.
Optimization checkpoints store only raw `rho`; they can restart across a state
order change when the design order, mesh refinement, and MPI rank count match.
The restarted MMA is still fresh, so treat a physics-order change as a new
optimization branch. Use `-restart-from <old-out>` with `-out <new-out>` to
preserve the old history while seeding the new branch.

## Output

The driver writes:

```text
optimization_history.txt
ParaView/TopOptTransient.pvd
ParaView/TopOptTransient_*.pvtu
```

Only ParaView output is written when `-pv` is enabled.

Continuous-run history headers record the nested grid and storage policy. Each
iteration records raw/filtered gradient norms, forward/adjoint seconds,
estimated trajectory memory, and controller/local replay counts in both
schedule blocks and equivalent fine forward intervals. Compare two or more
histories with the tracked analyzer:

The memory column estimates checkpoint/endpoint storage plus the principal
Hermite reconstruction vectors. It is not process peak RSS and excludes
assembled matrices, linear-solver data, and ordinary RK/adjoint work vectors.

```bash
./analyze_optimization_histories.py \
  full=run-full/optimization_history.txt \
  revolve=run-revolve/optimization_history.txt
```

It summarizes ranges and replay/timing totals, then reports pairwise relative
objective and gradient-norm differences on common iteration numbers. It reads
both the current 14-column history and older prefix schemas.

Useful iteration prints:

- `J`: current objective value.
- `vol`: current material volume fraction.
- `g`: MMA volume constraint value, `current_volume / target_volume - 1`.
  Positive means too much material; zero or negative is feasible.
- `dRho(L1)`: L1 design change used for the stopping test.

Useful solver prints:

- `Mass NNZ`, `Stiffness NNZ`, `Damping NNZ`, `ABC NNZ`: global sparse
  nonzero counts for the assembled Hypre matrices.
- `Inverse lumped mass range: [min, max]`: range of the diagonal inverse used
  by the row-lumped mass solve.

## Source Layout

The current structure intentionally keeps all problem-definition pieces in
`ProblemSpecification.hpp`.

```text
TopOptTransient.cpp
   Driver: parse CLI, select problem, build FE spaces, run MMA loop.

ProblemSpecification.hpp
   MaterialParams
   BoundaryLoadSpec
   load coefficients (directional, concentrated, rectangular, monopole)
   DampingParameters / DampingField / SphericalDampingField
   TransientTopOptConfig
   TransientTopOptProblem interface
   WaveShieldingProblem
   BandWaveguideProblem
   BandModeConverterProblem
   CantileverComplianceProblem
   SphericalBandGapProblem

ObjectiveFunctional.hpp
   TimeIntegratedObjective interface
   rectangular / circular / spherical-shell indicators
   DisplacementL2Objective (warns when the region has zero measure)
   HarmonicDisplacementTrackingObjective
   HarmonicModalCorrelationObjective
   ComplianceObjective

ElastodynamicsSolver.hpp
   SIMP coefficients
   ElastodynamicsOperator
   RK4 rollout helpers
   discrete adjoint helpers
   design-gradient integrators
   full-storage and interval-local continuous design-gradient kernels
   TransientDesignSolver (discrete or continuous, full or REVOLVE storage)

TrajectoryCheckpointing.hpp
   relation-aware REVOLVE schedule-block wrapper with validated metadata

OptimizationCheckpoint.hpp
   MMA design/state checkpointing for restartable optimization runs

test_adjoint_verification.cpp
   Jacobian transpose checks
   RK4 transpose checks
   objective Taylor checks
   raw-design Taylor checks for consistent/lumped mass
   clamped-BC and compliance-objective checks

test_continuous_temporal_refinement.cpp
   coupled coarse-forward/fine-adjoint objective/state/design convergence

test_continuous_forward_finer.cpp
   odd-q fine-forward/coarse-adjoint convergence and q-block REVOLVE audits
   constrained consistent-mass full/replay and optional directional-FD checks

analyze_optimization_histories.py
   summarizes and pairwise-compares optimization history/telemetry files

CONTINUOUS_ADJOINT_IMPLEMENTATION_PLAN.md
   staged band-waveguide implementation and verification gates
   nested forward/adjoint time-grid and checkpointing design
   progress ledger for the 3D mode-converter optimization
```

Reference and experimental files:

```text
ForwardElastodynamics.cpp       older forward-only reference miniapp
ElastTopOpt_static.cpp          static topopt comparison/reference
mtop-chkpt/                     checkpointing and adjoint reference code
DG-exp/                         directional-mass experiments
figures/                        existing documentation figures
```

## Architecture

The runtime flow is:

```text
TopOptTransient.cpp
   -> TransientTopOptProblem
      -> mesh, BC attributes, material, damping, load, objective
   -> TransientDesignSolver
      -> FilterFSolve
      -> PhysicsFSolve
      -> PhysicsASolve
      -> FilterASolve
   -> MMA update
```

`TransientDesignSolver` is the main optimization abstraction. The driver owns
the optimizer loop, while `TransientDesignSolver` bundles the invariant solver
setup and exposes the four canonical operations:

```cpp
design_solver.FilterFSolve(rho_tv);
const real_t J = design_solver.PhysicsFSolve(k);
design_solver.PhysicsASolve();
design_solver.FilterASolve(dJ_drho);
```

This keeps the driver independent of the details of RK4, adjoint stages, and
SIMP design sensitivities.

## Adding a Problem

Add a new subclass of `TransientTopOptProblem` in `ProblemSpecification.hpp`.

At minimum, provide:

```cpp
const TransientTopOptConfig &GetConfig() const override;
void GetEssentialBoundaryAttributes(Array<int> &attrs) const override;
void GetAbsorbingBoundaryAttributes(Array<int> &attrs) const override;
std::unique_ptr<VectorCoefficient> CreateBoundaryLoadCoefficient() const override;
std::unique_ptr<TimeIntegratedObjective>
CreateObjective(ParFiniteElementSpace *state_fes, MPI_Comm comm) const override;
```

Override `CreateMesh()` for generated geometry. Otherwise the base class reads
`cfg.mesh_file`.

Then register the problem in `TopOptTransient.cpp` where `-problem` is parsed.

## Adding an Objective

Implement the `TimeIntegratedObjective` interface in
`ObjectiveFunctional.hpp`:

```cpp
class MyObjective : public TimeIntegratedObjective
{
public:
   real_t AccumulateTimestep(const ParGridFunction &u,
                             real_t dt, int step, int total_steps) override;

   void ComputeObjectiveGradient(const ParGridFunction &u,
                                 real_t dt, int step, int total_steps,
                                 ParLinearForm &grad_form) override;
};
```

Then return it from the chosen problem's `CreateObjective()`.

## Damping and Boundary Conditions

`DampingField` owns the spatial damping coefficient supplied to the operator.
When damping is enabled, it can combine:

- sponge-layer damping from `DampingProfile` / `SpatialDampingCoefficient`
- uniform mass-proportional damping
- absorbing-boundary impedance

When `-no-damp` is used, `DampingField` supplies zero damping and zero
absorbing-boundary impedance. Absorbing boundary attributes may still be present,
but impedance zero makes them free boundaries.

For `spherical-bandgap`, the sponge uses the exact radial harmonic coordinate
between `r=7.5` and `r=10`, then applies the smooth exponential ramp. Its rate
is normalized by the integral of that ramp and by the passive SIMP mass, so the
configured target is the intended outgoing P-wave amplitude attenuation across
the layer. The scalar outer impedance is correspondingly the passive P-wave
impedance; a tensor P/S absorbing boundary remains a possible future upgrade.

Essential boundary attributes are projected in the operator. Row-lumped mass
zeros the constrained entries directly. Consistent mass solves the eliminated
free-DOF matrix, projects the right-hand side and result, and requires CG
convergence; the full assembled mass matrix is retained for variational inner
products and the generalized stability estimate.

## Verification

Build:

```bash
make test_adjoint_verification test_continuous_temporal_refinement \
  test_continuous_forward_finer -j8
```

Run a short check:

```bash
mpirun -np 4 ./test_adjoint_verification -r 0 -o 1 -ns 4 -nt 1
```

Together the verification targets check:

- Jacobian transpose action
- one-step RK4 transpose
- multi-step RK4 transpose
- nested forward/adjoint time-grid contract
- exact-cubic and fourth-order off-node Hermite reconstruction
- state-dependent continuous-adjoint directional derivative
- nonzero terminal-functional sign and physical RK4 stage times
- filtered and raw continuous-design finite differences
- full-storage versus interval-local REVOLVE objective/adjoint/gradient
- poisoned replay scratch, independently accepted endpoints, and physical
  endpoint slopes
- checkpoint metadata/reset-generation and production-scale late-time replay
- objective Taylor finite-difference behavior
- raw-design Taylor finite-difference behavior
- consistent and lumped mass design-gradient paths
- clamped-BC inverse residual/symmetry/essential-zero checks for consistent mass
- consistent-mass wave/damping CFL and full/REVOLVE continuous-gradient checks
- compliance-objective gradient path

The constrained consistent inverse currently records a free-space residual
`6.51e-13`, symmetry error `1.42e-14`, and zero essential-DOF leakage. The
optional forward-finer consistent-mass FD audit is enabled with
`./test_continuous_forward_finer -cfd`.

## Checkpoint / Restart

With `-ckpt` (default on), the driver saves a minimal checkpoint at the end of
every MMA iteration into `<out>/optimization_checkpoint/`:

```text
metadata.txt      iteration, J, volume fraction, ranks, refinement,
                  state order, design order
design.NNNNNN     per-rank binary control-density true-dof vector
```

The raw control layout must keep the SAME rank count, base mesh/partition,
mesh refinement, and design order. The state order may change. Existing
checkpoints predate the explicit design-order metadata, so they are read as
the historical coupled layout (`design order == saved state order`).

Continue in place:

```bash
srun -n <same N> ./TopOptTransient ... -out <same dir> -restart
```

Or start a separate higher-order physics branch from a saved design:

```bash
srun -n <same N> ./TopOptTransient ... -o 8 -do 1 -lumped-mass \
  -out <new dir> -restart -restart-from <old dir>
```

Only the density is restored - it becomes the initial guess of a fresh MMA
run (asymptotes rebuild within a couple of iterations); the iteration counter
continues for history/budget bookkeeping. Every checkpoint file is written to
`.tmp` and atomically renamed with the metadata committed last, so a job
killed at the wall-clock limit mid-save cannot corrupt the previous
checkpoint.

## TODO / Planned Work

- **Physical optimization comparison.** Run a nonzero-receiver 2D case through
  discrete, adjoint-finer, and forward-finer gradients; include temporal/mesh
  references, full/REVOLVE timing-memory-replay telemetry, restart provenance,
  gradient angles, volume residuals, and final-design comparisons. The tiny
  completed cases validate implementation wiring but not the scientific claim.
- **Two-material (convex-combination) interpolation.** Mass and stiffness
  currently share one SIMP law, so the local wave speed is design-independent
  and the optimizer can only exploit impedance contrast. The band-gap
  reference formulation interpolates between two materials,
  `K(a) = a K_1 + (1-a) K_2` and `M(a) = a M_1 + (1-a) M_2`, giving velocity
  contrast as well. Requires extending `StageMassDesignLFIntegrator` /
  `StageStiffnessDesignLFIntegrator` and re-running the Taylor verification.
- **Receiver-restricted objective assembly.** `AccumulateTimestep` and the
  adjoint objective-gradient linear form sweep ALL elements every step even
  though the indicator is nonzero only in the receiver (~15% of elements in
  the spherical mesh). Precomputing the supported-element list (or using mesh
  attributes - the spherical mesh carries region attributes 1-5) is an exact
  optimization with large 3D savings on both the forward and adjoint sweeps.

## Known Limitations

- REVOLVE is the default trajectory policy; continuous runs may select full
  endpoint storage for verification. First/last-iteration wave visualization
  streams only sampled frames. Tune the REVOLVE memory/recompute trade-off with
  `-nchk`.
- The correctness-first continuous replay regenerates each scheduled interval
  or q-interval block once locally in addition to controller-selected REVOLVE
  recomputation.
- The physical multi-iteration band comparison and final localized-source 3D
  example remain pending. Tiny short-time band runs are wiring checks and can
  finish before the signal reaches a receiver.
- Damping sponge geometry is rectangular-profile or radial-profile based, not
  a general signed-distance field from arbitrary mesh attributes.
- Spatial indicators (receivers, passive regions) are geometric (coordinates),
  not mesh-attribute based, even where the mesh carries region attributes.
- New problems are currently registered manually in `TopOptTransient.cpp`.

## Good Smoke Runs

Wave:

```bash
mpirun -np 4 ./TopOptTransient -problem wave -r 0 -o 1 -tf 0.0001 -dt 0.0001 -mi 1 -no-pv
```

Continuous coarse-forward / fine-adjoint:

```bash
mpirun -np 4 ./TopOptTransient -problem wave -r 0 -o 1 -do 1 \
  -tf 0.00015 -dt 5e-5 -mi 1 -no-pv -lumped-mass \
  -adjoint-mode continuous -adjoint-refinement 3 -nchk 1
```

Continuous fine-forward / coarse-adjoint with consistent mass:

```bash
mpirun -np 4 ./TopOptTransient -problem cantilever-compliance \
  -r 0 -o 1 -do 1 -tf 0.002 -dt 0.001 -mi 1 -no-pv \
  -iterative-mass -adjoint-mode continuous -adjoint-coarsening 2 \
  -trajectory-storage revolve -nchk 1
```

Cantilever:

```bash
mpirun -np 4 ./TopOptTransient -problem cantilever-compliance -lumped-mass -damp -tf 0.01 -dt 1e-3 -mi 1 -no-pv
```

Band waveguide:

```bash
mpirun -np 4 ./TopOptTransient -problem band-waveguide -lumped-mass -damp -tf 0.01 -dt 1e-4 -mi 1 -no-pv
```

Band mode-converter wiring (the warning about pulse-peak arrival is expected):

```bash
mpirun -np 4 ./TopOptTransient -problem band-mode-converter -tm 8 \
  -lumped-mass -damp -tf 0.008 -dt 1e-3 -mi 1 -tol 0 \
  -adjoint-mode continuous -adjoint-refinement 2 \
  -trajectory-storage revolve -nchk 1 -no-pv
```

Spherical band-gap (coarse mesh; expects a near-zero J warning since tf is
far below the receiver travel time):

```bash
mpirun -np 4 ./TopOptTransient -problem spherical-bandgap -mesh spherical_bandgap_coarse.msh -lumped-mass -damp -tf 0.01 -dt 1e-3 -mi 1 -no-pv
```

These are wiring checks, not production-quality optimization runs.
