# Understanding ProblemSpecification.hpp

## Overview

This module separates **problem definition** (what to solve) from **solver implementation** (how to solve it). The key design principle:

> **User says YES/NO for each feature, then provides details only if YES.**

This makes it impossible to forget something or create ambiguous states.

---

## Module Organization

```
ProblemSpecification.hpp
├── DampingCoefficient (abstract base)
│   ├── MeshDrivenDampingCoefficient (reads regions from mesh)
│   └── ConstantDampingCoefficient (uniform damping)
│
├── ForcingSpec (struct with YES/NO flags)
│   ├── Neumann (boundary traction): has_neumann flag
│   └── Body force (domain RHS): has_body_force flag
│
└── TransientTopOptProblem (abstract interface)
    ├── Mesh & geometry
    ├── Boundary conditions (Dirichlet, Robin/absorbing, Neumann)
    ├── Forcing (via ForcingSpec)
    ├── Damping (optional, nullptr = none)
    ├── Material parameters
    └── Objective functional
```

---

## 1. Forcing: Why the YES/NO Design?

### The Problem You Identified

Old design was ambiguous:
```cpp
// BAD: What does this mean?
forcing.neumann_bdr_attrs.SetSize(0);  // Forgot to fill? Or intentionally no Neumann?
forcing.body_force_coef = nullptr;     // No body force? Or forgot to set?
```

Solver would need checks like:
```cpp
if (forcing.neumann_coef != nullptr && forcing.neumann_bdr_attrs.Size() > 0) {
   // assemble Neumann
}
```

### New Design: Explicit Flags

```cpp
struct ForcingSpec {
   // NEUMANN: YES/NO flag + details
   bool has_neumann;                   // ← User says YES or NO
   Array<int> neumann_boundaries;      // ← Only fill if YES
   VectorCoefficient *neumann_spatial; // ← Only fill if YES
   TimeProfile neumann_time_profile;   // ← Only fill if YES
   real_t neumann_amplitude, neumann_duration, neumann_phase;

   // BODY FORCE: YES/NO flag + details
   bool has_body_force;                // ← User says YES or NO
   VectorCoefficient *body_force_spatial; // ← Only fill if YES
   TimeProfile body_force_time_profile;   // ← Only fill if YES
   real_t body_force_amplitude, body_force_duration, body_force_phase;
};
```

**Benefits:**
- **Clear intent:** `has_neumann=false` explicitly means "no Neumann"
- **Validation:** Solver calls `forcing.Validate()` to catch errors:
  ```cpp
  if (has_neumann && neumann_spatial == nullptr) {
     error("You said has_neumann=true but forgot neumann_spatial!");
  }
  ```
- **No ambiguity:** Solver checks the flag, not `nullptr`/`Size()==0`

### Usage Examples

#### Case 1: Boundary Load Only (Your Current Problem)
```cpp
ForcingSpec forcing;
forcing.has_neumann = true;
forcing.neumann_boundaries = {21, 22, 23, 24, 25, 26};  // From Gmsh
forcing.neumann_spatial = new DownwardTraction();
forcing.neumann_time_profile = GAUSSIAN;
forcing.neumann_amplitude = 30.0;
forcing.neumann_duration = 0.005;

forcing.has_body_force = false;  // Explicitly no body force
```

#### Case 2: Body Force Only (Gravity, Inertial)
```cpp
ForcingSpec forcing;
forcing.has_neumann = false;  // Explicitly no Neumann

forcing.has_body_force = true;
forcing.body_force_spatial = new GravityCoef(9.81);
forcing.body_force_time_profile = CONSTANT;  // Gravity doesn't vary in time
```

#### Case 3: Both (Seismic Base Excitation + Self-Weight)
```cpp
ForcingSpec forcing;

// Boundary excitation
forcing.has_neumann = true;
forcing.neumann_boundaries = {1};  // Base of structure
forcing.neumann_spatial = new SeismicExcitationCoef();
forcing.neumann_time_profile = HARMONIC;
forcing.neumann_amplitude = 10.0;
forcing.neumann_duration = 2*PI/omega;  // Period

// Gravity
forcing.has_body_force = true;
forcing.body_force_spatial = new GravityCoef(9.81);
forcing.body_force_time_profile = CONSTANT;
```

#### Case 4: Neither (Free Vibration from Initial Conditions)
```cpp
ForcingSpec forcing;
forcing.has_neumann = false;
forcing.has_body_force = false;
// Valid! Solver will use initial conditions only
```

---

## 2. Damping: Why It's Optional

### Three Cases

| Case | What to Return | Solver Behavior |
|---|---|---|
| **No damping** | `nullptr` | Damping matrix C = 0 (skipped in assembly) |
| **Mesh-driven damping** | `new MeshDrivenDampingCoefficient(...)` | Reads damping region from mesh attr |
| **Custom damping** | Your `DampingCoefficient` subclass | Uses your formula |

### Why Not ZeroDampingCoefficient?

**You asked:** "Why do we need ZeroDampingCoefficient?"

**Answer:** We don't! I removed it. Using `nullptr` is clearer:

```cpp
// BEFORE (unnecessarily complex):
DampingCoefficient* GetDampingCoefficient(ParMesh *mesh) const override {
   return new ZeroDampingCoefficient();  // Why create an object?
}

// AFTER (simple):
DampingCoefficient* GetDampingCoefficient(ParMesh *mesh) const override {
   return nullptr;  // No damping
}
```

**Solver handles it:**
```cpp
DampingCoefficient *damp = problem->GetDampingCoefficient(mesh);

if (damp != nullptr) {
   // Assemble damping matrix C = γ(x) M
   C_vol = new ParBilinearForm(&fespace);
   C_vol->AddDomainIntegrator(new VectorMassIntegrator(*damp));
   C_vol->Assemble();
   // ...
} else {
   // No damping: C_vol = nullptr, skip damping term
   C_vol = nullptr;
}

// In RHS evaluation:
if (C_vol != nullptr) {
   Cvol_mat->Mult(v_true, tmp);
   res.Add(-1.0, tmp);
}
```

This is simpler than forcing every problem to return a "zero damping" object.

---

## 3. Boundary Conditions: Three Types

### Elastodynamics BCs

The PDE is:
```
M ü + C u̇ + K u = f_body (in Ω)
```

On the boundary ∂Ω, you can specify:

| BC Type | Math | Physical Meaning | How to Specify |
|---|---|---|---|
| **Dirichlet (Essential)** | `u = 0` | Clamped, fixed | `GetEssentialBoundaryAttributes()` |
| **Robin (Absorbing)** | `σ·n = -Z u̇` | Non-reflecting, impedance | `GetAbsorbingBoundaryAttributes()` |
| **Neumann (Natural)** | `σ·n = f(x,t)` | Applied traction | `GetForcing().neumann_boundaries` |

### Why Separate Methods?

**Dirichlet:** Handled specially (essential DOFs, eliminated from system)
```cpp
void GetEssentialBoundaryAttributes(Array<int> &ess) const override {
   ess.SetSize(1);
   ess[0] = 1;  // Gmsh attr 1 = clamped end
}
```

**Absorbing:** Impedance matrix `σ·n = -Z u̇` assembled into system
```cpp
void GetAbsorbingBoundaryAttributes(Array<int> &abc) const override {
   abc = {10, 11, 12, 13};  // Gmsh attrs = exterior boundaries
}
```

**Neumann:** Applied traction, goes into RHS (part of forcing)
```cpp
ForcingSpec GetForcing() const override {
   forcing.has_neumann = true;
   forcing.neumann_boundaries = {21, 22, 23, 24, 25, 26};  // Load strips
   // ...
}
```

**Example: Cantilever Beam**
- Attr 1 (left end): Dirichlet (clamped) → `ess = {1}`
- Attr 2 (right end): Neumann (tip load) → `forcing.neumann_boundaries = {2}`
- No absorbing BCs → `abc.SetSize(0)`

---

## 4. MeshDrivenDampingCoefficient: How It Works

### Problem You Identified

Old code (`DampingProfile` in `ElastodynamicsSolver.hpp:134-167`) hardcoded:
```cpp
// Left boundary layer
if (x(0) < thickness) { ... }
// Right boundary layer
if (x(0) > x_max - thickness) { ... }
// Bottom boundary layer
if (x(1) < thickness) { ... }
```

**This only works for rectangular domains.** What about an L-shape? Circle? 3D?

### Solution: Read from Mesh

Your mesh already marks the damping region:
```
Physical Line("interior_damping_interface") = {30};
```

`MeshDrivenDampingCoefficient` computes distance to that interface:

```cpp
Array<int> damping_attrs = {30};  // Read from mesh
DampingCoefficient *damp = new MeshDrivenDampingCoefficient(
   mesh, damping_attrs, gamma_max, thickness, rho0);
```

**How it evaluates `γ(x)`:**
1. At point `x`, compute distance `d` to nearest point on attr 30
2. If `d > thickness`: return 0 (no damping, far from boundary)
3. If `d ≤ thickness`: return `γ_max * ramp(d/thickness)` (smooth increase)

The ramp function (exponential sponge) is computed automatically. **Works for any domain shape** because it uses the mesh geometry.

### Current Implementation (Simplified for Now)

The current code uses a simplified version (distance from bounding box edges) as a placeholder. The TODO comment says:
```cpp
// TODO: Implement proper signed-distance-field computation from mesh attributes
```

This would use MFEM's distance field utilities or a fast marching method. But the **interface** is ready — when you implement the full distance field, the solver doesn't need to change.

---

## 5. Validation: Catching User Errors

The struct has a `Validate()` method:

```cpp
bool ForcingSpec::Validate(std::ostream &err) const {
   if (has_neumann && neumann_boundaries.Size() == 0) {
      err << "Error: has_neumann=true but neumann_boundaries is empty.\n";
      return false;
   }
   if (has_neumann && neumann_spatial == nullptr) {
      err << "Error: has_neumann=true but neumann_spatial is nullptr.\n";
      return false;
   }
   // ... similar checks for body force
   return true;
}
```

**Solver calls this:**
```cpp
ForcingSpec forcing = problem->GetForcing();
if (!forcing.Validate(std::cerr)) {
   std::cerr << "Problem specification is invalid. Exiting.\n";
   return 1;
}
```

This catches mistakes like:
- "I said `has_neumann=true` but forgot to fill `neumann_boundaries`"
- "I said `has_body_force=true` but `body_force_spatial` is `nullptr`"

**Fail fast with a clear error message**, instead of a cryptic segfault later.

---

## Summary: Design Principles

| Principle | Why | How |
|---|---|---|
| **Explicit flags** | No ambiguity | `has_neumann` = YES/NO, not "check if nullptr" |
| **Optional = nullptr** | Simple | `damping = nullptr` means "no damping", not a ZeroCoefficient object |
| **Validation** | Catch errors early | `forcing.Validate()` checks consistency |
| **Mesh-driven** | Domain-agnostic | Damping/BCs from mesh attributes, not hardcoded coords |
| **Separation of concerns** | Reusable solver | Problem details in subclass, solver never hardcodes |

## Next Steps

The module is **ready to use** but not yet integrated. To connect it:

1. **Refactor `ElastodynamicsSolver` constructor** to accept `ForcingSpec` and check `has_neumann`/`has_body_force`
2. **Update `TopOptTransient.cpp`** to create a `WaveShieldingProblem` and query it
3. **Test** with your current mesh (should produce identical results)

Then new problems = subclass `TransientTopOptProblem`, no solver edits.

Want me to do the integration? It's about ~100 lines of refactoring in the solver/driver.
