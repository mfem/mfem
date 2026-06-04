# Tied Poisson Miniapp

## Description

The **Tied Poisson** miniapp solves a Poisson equation on two copies of a mesh with one face tied together using penalty constraints. This demonstrates how to couple two domains through interface constraints.

The problem solved is:
```
-∇²u = 1  in Ω₁ ∪ Ω₂
u = 1     on ∂Ω \ Γ
```

where Γ is the tied interface between the two domains. The tie constraint is enforced weakly by adding `α P Pᵀ` to the stiffness matrix, where:
- `α` is a penalty parameter
- `P` is a constraint matrix with rows of the form `[... +1 ... -1 ...]` for each tied node pair

## Usage

```bash
# Serial execution (1 MPI rank)
mpirun -np 1 ./tied-poisson [options]

# Parallel execution (multiple MPI ranks)
mpirun -np 4 ./tied-poisson [options]
```

### Options

- `-m <mesh>` : Mesh file to use (default: `../../data/beam-tri.mesh`)
- `-o <order>` : Finite element order (default: 1)
- `-a <alpha>` : Penalty parameter for tied interface (default: 1000)
- `-t <attr>` : Boundary attribute to tie between mesh copies (default: 1)
- `-sep <dist>` : Separation distance for visualization (default: auto)
- `-vis` / `-no-vis` : Enable/disable GLVis visualization (default: enabled)

### Example

```bash
# Run with default settings
mpirun -np 2 ./tied-poisson

# Use a different mesh with higher penalty
mpirun -np 2 ./tied-poisson -m ../../data/square-disc.mesh -a 1e4

# 3D mesh - specify which boundary to tie
mpirun -np 2 ./tied-poisson -m ../../data/beam-hex.mesh -t 2 -a 1e5

# Higher order elements with refinement
mpirun -np 4 ./tied-poisson -o 2 -r 1

# Disable visualization
mpirun -np 2 ./tied-poisson -no-vis
```

## Output

The miniapp produces:
- `tied-poisson-mesh1.mesh`, `tied-poisson-mesh2.mesh` : The two mesh copies
- `tied-poisson-sol1.gf`, `tied-poisson-sol2.gf` : The solution fields
- GLVis visualization windows (if enabled) showing both solutions side-by-side

## Implementation Details

### Tied Interface Constraint

The tie constraint enforces that the solution values match across the interface:
```
u₁(x) = u₂(x)  for x ∈ Γ
```

This is implemented as:
1. Identify corresponding DOFs on the tied boundaries of both meshes
2. Build constraint matrix P where each row represents: `u₁ᵢ - u₂ᵢ = 0`
3. Add penalty term `α PᵀP` to the global stiffness matrix
4. Solve the modified system

### Mesh Separation

For visualization purposes, the second mesh is translated along the x-axis by a distance equal to twice the bounding box width. This separation is only applied to the mesh coordinates, not during the computation.

## Parallel Implementation

The miniapp uses a penalty-based approach to enforce tied interface constraints in parallel:

1. **Global DOF correspondence**: Uses MPI communication (`MPI_Allgatherv`) to establish which DOFs on mesh1 correspond to which DOFs on mesh2 across all ranks
2. **Distributed matrix assembly**: Each MPI rank builds only the matrix rows it owns, properly handling both diagonal (on-processor) and off-diagonal (off-processor) coupling terms
3. **Block matrix structure**: Assembles a 2×2 block system with penalty terms coupling the two meshes

The implementation correctly handles the case where some MPI ranks own zero tied DOFs.

## Notes

- The tied boundary attribute must exist in the input mesh
- Both mesh copies must have the same boundary structure
- Increasing α strengthens the tie constraint but may affect conditioning
- The optimal α value depends on the mesh size and stiffness matrix entries
- **Reference implementation**: `tied-poisson-serial.cpp` contains a reference implementation with known parallel bugs; use `tied-poisson.cpp` instead
