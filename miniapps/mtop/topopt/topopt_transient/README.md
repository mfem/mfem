# Transient Topology Optimization

This directory contains a transient elastodynamics topology-optimization driver
for MFEM. The current code supports multiple forward problems and objective
functionals through a small problem interface, while keeping the RK4
forward/adjoint machinery in the solver layer.

The active executable is `TopOptTransient`.

## What This Code Solves

The optimizer updates a material density field `rho` subject to a volume
constraint:

```text
minimize    J(rho)
subject to  M(rho) u_tt + C u_t + K(rho) u = f(t)
            int_Omega rho dx / target_volume - 1 <= 0
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
discrete RK4 adjoint
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
- Carrier: frequency `5.0`, pulse duration `0.80`.
- Damping: sponge and absorbing-boundary impedance on the left/right ends only.
- Objective: `DisplacementL2Objective`, minimizing time-integrated displacement
  energy in symmetric rectangular receiver regions on both sides of the source.
- Material: same SIMP law, but with a raised material floor to discourage one
  near-disconnecting slit and favor multiple impedance interfaces.

Example:

```bash
mpirun -np 8 ./TopOptTransient -problem band-waveguide -lumped-mass -damp -r 0 -o 1 -tf 3.5 -dt 5e-4 -vf 0.5 -fr 0.05 -mi 100 -mv 0.2 -pv
```

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

For this problem, `-lumped-mass` is the preferred path right now because the
current essential-boundary projection is exact for the diagonal row-lumped mass.

## Build

From this directory in WSL:

```bash
make TopOptTransient test_adjoint_verification -j8
```

From Windows PowerShell, using the repository path:

```powershell
wsl make -C /mnt/c/Users/cortescastil1/Desktop/mfem/miniapps/mtop/topopt/topopt_transient TopOptTransient test_adjoint_verification -j8
```

## Command-Line Options

Common options:

```text
-problem <name>              wave, band-waveguide, or cantilever-compliance
-r,  --refine <int>          uniform refinement levels
-o,  --order <int>           H1 finite element order
-tf, --t-final <real>        final simulation time
-dt, --time-step <real>      time step
-vf, --vol-frac <real>       target material volume fraction
-fr, --filter-radius <real>  Helmholtz filter radius
-mi, --max-it <int>          maximum MMA iterations
-mv, --move <real>           MMA move limit
-tol, --tol <real>           L1 design-change stopping tolerance
-init <mode>                 uniform, solid, void, or gaussian
-mesh <file>                 mesh file for file-based problems
-pv / -no-pv                 enable or disable ParaView output
-damp / -no-damp             enable or disable problem damping
-iterative-mass              consistent mass solve with CG+AMG
-lumped-mass                 row-sum diagonal mass solve
```

The default mass path is `-iterative-mass`. For larger explicit runs,
`-lumped-mass` is usually much faster.

## Output

The driver writes:

```text
optimization_history.txt
ParaView/TopOptTransient.pvd
ParaView/TopOptTransient_*.pvtu
```

Only ParaView output is written when `-pv` is enabled.

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
   load coefficients
   DampingParameters / DampingField
   TransientTopOptConfig
   TransientTopOptProblem interface
   WaveShieldingProblem
   CantileverComplianceProblem

ObjectiveFunctional.hpp
   TimeIntegratedObjective interface
   RectangularIndicator / SubdomainIndicator
   DisplacementL2Objective
   ComplianceObjective

ElastodynamicsSolver.hpp
   SIMP coefficients
   ElastodynamicsOperator
   RK4 rollout helpers
   discrete adjoint helpers
   design-gradient integrators
   TransientDesignSolver

test_adjoint_verification.cpp
   Jacobian transpose checks
   RK4 transpose checks
   objective Taylor checks
   raw-design Taylor checks for consistent/lumped mass
   clamped-BC and compliance-objective checks
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

Essential boundary attributes are projected in the operator. This is exact for
the row-lumped mass path. The consistent-mass path is useful for verification
and unconstrained cases, but fully eliminated consistent-mass Dirichlet handling
is still a future cleanup.

## Verification

Build:

```bash
make test_adjoint_verification -j8
```

Run a short check:

```bash
mpirun -np 4 ./test_adjoint_verification -r 0 -o 1 -ns 4 -nt 1
```

The verification executable checks:

- Jacobian transpose action
- one-step RK4 transpose
- multi-step RK4 transpose
- objective Taylor finite-difference behavior
- raw-design Taylor finite-difference behavior
- consistent and lumped mass design-gradient paths
- clamped-BC projection consistency for lumped mass
- compliance-objective gradient path

## Known Limitations

- 2D workflows are the current focus.
- Full forward trajectory storage is used; checkpointing is not integrated yet.
- Damping sponge geometry is still rectangular/profile based, not a general
  signed-distance field from arbitrary mesh attributes.
- Consistent-mass Dirichlet enforcement is not fully eliminated.
- New problems are currently registered manually in `TopOptTransient.cpp`.

## Good Smoke Runs

Wave:

```bash
mpirun -np 4 ./TopOptTransient -problem wave -r 0 -o 1 -tf 0.0001 -dt 0.0001 -mi 1 -no-pv
```

Cantilever:

```bash
mpirun -np 4 ./TopOptTransient -problem cantilever-compliance -lumped-mass -damp -tf 0.01 -dt 1e-3 -mi 1 -no-pv
```

Band waveguide:

```bash
mpirun -np 4 ./TopOptTransient -problem band-waveguide -lumped-mass -damp -tf 0.01 -dt 1e-4 -mi 1 -no-pv
```

These are wiring checks, not production-quality optimization runs.
