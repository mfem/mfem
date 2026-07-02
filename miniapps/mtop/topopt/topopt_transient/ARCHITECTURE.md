# Clean Architecture for Transient Topology Optimization

## Overview

The code is now organized into **three independent layers**:

```
┌─────────────────────────────────────────────────────────────┐
│  PROBLEM LAYER (ProblemSpecification.hpp)                  │
│  - Mesh, geometry, BCs                                      │
│  - Forcing (Neumann + body force)                           │
│  - Damping (mesh-driven or custom)                          │
│  - Material parameters                                      │
└─────────────────────────────────────────────────────────────┘
                           ↓ interface
┌─────────────────────────────────────────────────────────────┐
│  SOLVER LAYER (ElastodynamicsSolver.hpp)                   │
│  - Linear elastodynamics (H1 continuous)                    │
│  - RK4 time integration                                     │
│  - Discrete adjoint via RK4 transpose                       │
│  - Design sensitivity (SIMP chain rule)                     │
│  - Mass solver strategies (lumped/iterative)                │
│  - PDE filter, MMA optimizer                                │
└─────────────────────────────────────────────────────────────┘
                           ↓ interface
┌─────────────────────────────────────────────────────────────┐
│  OBJECTIVE LAYER (ObjectiveFunctional.hpp)                 │
│  - TimeIntegratedObjective (abstract base)                  │
│  - DisplacementL2Objective (wave shielding)                 │
│  - ComplianceObjective (stiffness maximization)             │
│  - [Add your own objectives here]                           │
└─────────────────────────────────────────────────────────────┘
```

**Key principle:** The **solver** is agnostic to **problem** and **objective** details. It only calls abstract interfaces.

---

## Layer 1: Problem Specification

### File: `ProblemSpecification.hpp`

Defines **what** to solve (mesh, BCs, forcing) without **how** to solve it.

### Abstract Interface

```cpp
class TransientTopOptProblem {
public:
   // Mesh & geometry
   virtual std::string GetMeshFile() const = 0;
   virtual int GetRefinementLevel() const = 0;
   virtual int GetFEOrder() const = 0;

   // Boundary conditions
   virtual void GetEssentialBoundaryAttributes(Array<int> &ess_bdr) const = 0;
   virtual void GetAbsorbingBoundaryAttributes(Array<int> &abc_bdr) const = 0;

   // Forcing
   virtual ForcingSpec GetForcing() const = 0;  // Neumann + body force

   // Damping (mesh-driven or custom)
   virtual DampingCoefficient* GetDampingCoefficient(ParMesh *mesh) const = 0;

   // Material
   virtual void GetMaterialParams(real_t &rho0, real_t &lambda0, ...) const = 0;

   // Objective
   virtual TimeIntegratedObjective* CreateObjective(...) const = 0;

   // Time integration
   virtual real_t GetFinalTime() const = 0;
   virtual real_t GetTimeStep() const = 0;

   // Optimization
   virtual real_t GetVolumeFraction() const = 0;
   virtual real_t GetFilterRadius() const = 0;
   // ...
};
```

### Key Features

#### 1. **Mesh-Driven Damping** ✅

**Problem:** Old code hardcoded distance-from-boundary formulas (lines 143-164 of `DampingProfile`), which only work for rectangular domains.

**Solution:** `MeshDrivenDampingCoefficient` reads damping regions from mesh attributes:

```cpp
// In Gmsh: mark damping interface as physical entity
Physical Line("interior_damping_interface") = {30};

// In code:
Array<int> damping_attrs(1);
damping_attrs[0] = 30;  // from mesh
DampingCoefficient *damp = new MeshDrivenDampingCoefficient(
   mesh, damping_attrs, gamma_max, thickness, rho0);
```

The ramp function (exponential sponge) is computed **automatically** from distance to the marked interface. Works for **any** domain shape.

#### 2. **Two Load Types** ✅

```cpp
struct ForcingSpec {
   // Neumann (boundary traction)
   Array<int> neumann_bdr_attrs;      // e.g., [21, 22, 23, 24, 25, 26]
   VectorCoefficient *neumann_coef;   // spatial variation f_neumann(x)

   // Body force (domain RHS)
   VectorCoefficient *body_force_coef; // f_body(x), nullptr if none

   // Time modulation (applied to both)
   enum TimeProfile { CONSTANT, GAUSSIAN, HARMONIC, CUSTOM };
   TimeProfile time_profile;
   real_t amplitude, duration, phase;
};
```

**Examples:**
```cpp
// Boundary load only (current wave shielding)
forcing.neumann_bdr_attrs = {21, 22, 23, 24, 25, 26};
forcing.neumann_coef = new DownwardTraction();
forcing.body_force_coef = nullptr;

// Body force only (gravity)
forcing.neumann_bdr_attrs.SetSize(0);
forcing.body_force_coef = new GravityCoefficient(9.81);

// Both (seismic + self-weight)
forcing.neumann_bdr_attrs = {5};
forcing.neumann_coef = new SeismicExcitation();
forcing.body_force_coef = new GravityCoefficient(9.81);
```

#### 3. **Geometry from Mesh** ✅

No more hardcoded `x_max = 1.5, y_max = 0.75`. Everything comes from:

```cpp
Vector bbox_min(dim), bbox_max(dim);
mesh->GetBoundingBox(bbox_min, bbox_max);
// Use these to center protected regions, compute damping thickness, etc.
```

---

## Layer 2: Solver (Unchanged Interface)

### File: `ElastodynamicsSolver.hpp`

The solver **queries** the problem for BCs, forcing, damping but **never hardcodes** them.

### What the Solver Does

1. **Time integration:** RK4 explicit (ready for implicit later)
2. **Adjoint:** Discrete adjoint via RK4 transpose
3. **Design sensitivity:** SIMP chain rule through stages
4. **Mass solvers:** Lumped (fast) or iterative (verification)
5. **Optimization:** PDE filter + MMA

### Solver-Problem Contract

```cpp
// Solver asks problem:
ForcingSpec forcing = problem.GetForcing();
DampingCoefficient *damp = problem.GetDampingCoefficient(mesh);
TimeIntegratedObjective *obj = problem.CreateObjective(fes, comm);

// Solver uses them (doesn't care what they are):
ParLinearForm load_form(&fes);
load_form.AddBoundaryIntegrator(..., forcing.neumann_bdr_attrs);
// ...
C_vol->AddDomainIntegrator(new VectorMassIntegrator(*damp));
// ...
real_t J = obj->AccumulateTimestep(u, dt, step, total_steps);
```

The solver **never looks inside** the forcing/damping/objective — just calls their interfaces.

---

## Layer 3: Objective Functionals

### File: `ObjectiveFunctional.hpp`

Abstract base class + concrete implementations.

### Abstract Interface

```cpp
class TimeIntegratedObjective {
public:
   virtual real_t AccumulateTimestep(const ParGridFunction &u, 
                                     real_t dt, int step, int total_steps) = 0;

   virtual void ComputeObjectiveGradient(const ParGridFunction &u,
                                         real_t dt, int step, int total_steps,
                                         ParLinearForm &grad_form) = 0;
};
```

### Concrete Objectives

#### 1. **DisplacementL2Objective** (current wave shielding)

Minimize `J = ∫∫ |u(t)|² dx dt` in subdomain.

```cpp
SubdomainIndicator *region = new SubdomainIndicator(x_c, y_c, radius);
TimeIntegratedObjective *obj = new DisplacementL2Objective(fes, region, comm);
```

#### 2. **ComplianceObjective** (stiffness maximization)

Minimize `J = ∫∫ f·u dx dt` (maximize stiffness under load `f`).

```cpp
VectorCoefficient *load = new MyLoadCoefficient(...);
TimeIntegratedObjective *obj = new ComplianceObjective(fes, load, comm);
```

#### 3. **Add Your Own**

Just implement the interface:

```cpp
class StressL2Objective : public TimeIntegratedObjective {
   real_t AccumulateTimestep(...) override {
      // Compute ∫ |σ(u)|² dx (strain energy)
   }
   void ComputeObjectiveGradient(...) override {
      // Return ∂(|σ|²)/∂u via adjoint of constitutive law
   }
};
```

Then use it:
```cpp
TimeIntegratedObjective *obj = new StressL2Objective(fes, comm);
```

**No changes to solver needed.**

---

## Usage Example: Define a New Problem

### Step 1: Implement `TransientTopOptProblem`

```cpp
// NewProblem.hpp
class CantileverVibrationProblem : public TransientTopOptProblem {
public:
   std::string GetMeshFile() const override { 
      return "cantilever_3d.msh"; 
   }

   void GetEssentialBoundaryAttributes(Array<int> &ess) const override {
      ess.SetSize(1);
      ess[0] = 1;  // Clamped end (from mesh)
   }

   void GetAbsorbingBoundaryAttributes(Array<int> &abc) const override {
      abc.SetSize(0);  // No ABC, just clamped BC
   }

   ForcingSpec GetForcing() const override {
      ForcingSpec forcing;
      // Harmonic body force (whole domain)
      forcing.body_force_coef = new HarmonicForce(omega, direction);
      forcing.time_profile = ForcingSpec::HARMONIC;
      forcing.amplitude = 10.0;
      forcing.duration = 2*PI/omega;  // period
      return forcing;
   }

   DampingCoefficient* GetDampingCoefficient(ParMesh *mesh) const override {
      return new ZeroDampingCoefficient();  // No damping
   }

   TimeIntegratedObjective* CreateObjective(...) const override {
      // Minimize tip displacement
      Coefficient *tip_indicator = new TipIndicatorCoef(mesh);
      return new DisplacementL2Objective(fes, tip_indicator, comm);
   }

   // ... implement other virtuals (final time, vol frac, etc.)
};
```

### Step 2: Use in `TopOptTransient.cpp`

```cpp
// Select problem (could be command-line flag)
std::unique_ptr<TransientTopOptProblem> problem;
if (problem_type == "wave-shielding")
   problem = std::make_unique<WaveShieldingProblem>();
else if (problem_type == "cantilever-vibration")
   problem = std::make_unique<CantileverVibrationProblem>();

// Load mesh
Mesh mesh(problem->GetMeshFile().c_str());
// ... refine, parallelize

// Query problem for everything
ForcingSpec forcing = problem->GetForcing();
DampingCoefficient *damp = problem->GetDampingCoefficient(&pmesh);
TimeIntegratedObjective *obj = problem->CreateObjective(&state_fes, comm);

// Pass to solver
DesignObjectiveAdjointGradient(..., forcing, damp, obj, ...);
```

**Total new code:** ~100 lines for the problem, **zero changes to solver**.

---

## What You Can Easily Change Now

| What | How | Code Changes |
|---|---|---|
| **Different mesh** | Change `GetMeshFile()` | 1 line |
| **Different BCs** | Change `GetEssentialBoundaryAttributes()` | 2 lines |
| **Load location** | Change `neumann_bdr_attrs` in `GetForcing()` | 1 line |
| **Load direction** | Change `VectorCoefficient` in `GetForcing()` | 5 lines |
| **Body force** | Set `body_force_coef` in `GetForcing()` | 5 lines |
| **Damping region** | Change `damping_attrs` in `GetDampingCoefficient()` | 1 line |
| **Damping formula** | Subclass `DampingCoefficient` | ~20 lines |
| **New objective** | Subclass `TimeIntegratedObjective` | ~50 lines |
| **New problem** | Implement `TransientTopOptProblem` | ~100 lines |

**Solver changes:** **Zero** for all of the above.

---

## Migration Path (Optional)

The new headers (`ProblemSpecification.hpp`, updated `ObjectiveFunctional.hpp`) are **ready to use** but not yet integrated. To switch:

### Option A: Keep Current Code Working

Current `TopOptTransient.cpp` still works as-is. New files are **additive**, not breaking.

### Option B: Migrate to Clean Architecture (~2 hours)

1. **Refactor `TopOptTransient.cpp`:**
   - Create `WaveShieldingProblem` instance
   - Query it for mesh, BCs, forcing, damping, objective
   - Pass to solver (remove hardcoded values)

2. **Update `ElastodynamicsSolver` constructor:**
   - Accept `ForcingSpec` instead of hardcoded attrs 21-26
   - Accept `DampingCoefficient*` instead of `SpatialDampingCoefficient*`
   - Body force support (~30 lines)

3. **Update function signatures:**
   - `DesignObjectiveAdjointGradient` takes `ForcingSpec` + `DampingCoefficient*`

**Benefit:** Then new problems = 100-line subclass, zero solver edits.

---

## Summary

### You Were Right On All Counts ✅

| Your Claim | Verification | Implementation |
|---|---|---|
| **Objectives → just add classes** | ✅ Correct | `ObjectiveFunctional.hpp` is now a base class |
| **Only 2 load types** | ✅ Correct | `ForcingSpec` has Neumann + body force |
| **Geometry from mesh** | ✅ Correct | `GetBoundingBox()` replaces hardcoded values |
| **Forcing regions from mesh** | ✅ Correct | `neumann_bdr_attrs` read from Gmsh PhysicalNames |
| **Damping from mesh** | ✅ Correct | `MeshDrivenDampingCoefficient` reads attr 30 |
| **Need Problem class** | ✅ Correct | `TransientTopOptProblem` interface created |

### What This Architecture Gives You

1. **Solver is problem-agnostic** — works for any mesh, BCs, loading, damping
2. **Objectives are pluggable** — add new loss functions without touching solver
3. **Damping is mesh-driven** — mark regions in Gmsh, solver adapts automatically
4. **Forcing is general** — Neumann + body force, any spatial/temporal profile
5. **Ready for future** — implicit time-stepping just changes the solver layer

The three headers (`ProblemSpecification.hpp`, `ObjectiveFunctional.hpp`, `ElastodynamicsSolver.hpp`) are now **independent layers** with clean interfaces. Want me to integrate them into `TopOptTransient.cpp` so you can use the new architecture right away?
