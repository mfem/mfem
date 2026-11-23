# Contact miniapps

Parallel frictionless solid contact using MFEM elasticity, Tribol mortar
constraints, and an interior-point solver. The miniapps use first-order solid
and contact elements.

| Executable | Problems |
|---|---|
| `contact` | 0 two-block, 1 ironing, 2 beam/sphere |
| `contact_examples` | 3 onion shells, 4 folded strip, 5 ring, 6 C-clip/pin, 7 twisting cylinder, 8 twisting elastic strips |

Problem geometry, materials, and loading for cases 3–8 are described in
[EXAMPLES.md](EXAMPLES.md).

## Build and run

An MFEM build with MPI and Tribol is required. Run from `miniapps/contact`:

```bash
make -j12 contact contact_examples

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 mpirun -np 1 ./contact \
  -prob 0 -hex -sr 0 -pr 0 -nsteps 4 -msteps 0 \
  -amgf -amgf-fsolver auto -d cpu -no-vis -paraview -pl 2
```

Use `./contact -h` or `./contact_examples -h` for all options.
See [BUILD.md](BUILD.md) for dependencies, CUDA/HIP setup, and launch
instructions.

## Meshes and loading

Problems 0–2 generate their meshes. `-hex` and `-tet` select the cell type;
problems 0/1 default to hexes and problem 2 to tets.
Generated beam/sphere resolution is controlled by `-beam-nc`, `-beam-ny`,
`-sphere-n`, and `-sphere-nr`.

`-nsteps` controls loading. `-msteps` adds lateral sliding for problems 0/1,
continues the pressure ramp for problem 2, and controls unloading for problems
4–8. Problems 0/1 use linear elasticity; problem 2 supports `-nonlin`.
The additional examples use the material models listed in EXAMPLES.md.

## Common options

| Option | Meaning |
|---|---|
| `-sr N`, `-pr N` | Serial and parallel uniform refinements |
| `-amgf`, `-no-amgf` | CG with contact-subspace filtering or ordinary AMG (default) |
| `-amgf-fsolver auto` | Select an available direct solver for the filtered subspace |
| `-linear-solver cg` | Reduced-system solver; a direct backend such as `mumps` can also be selected |
| `-opt-maxit N` | IP iteration limit per solve |
| `-contact-max-updates N` | Geometry/material update limit, default 10 |
| `-d cpu`, `-d cuda`, `-d hip` | MFEM device backend |
| `-vis`, `-no-vis` | Enable or disable GLVis |
| `-paraview`, `-no-paraview` | Enable or disable saved visualization |
| `-pl 0/1/2/3` | Silent / summary / solver tables / debug output |

AMGF falls back to ordinary AMG when the global contact subspace is empty.
A converged IP solution at the geometry-update limit is accepted and reported
as `inexact`. Use `-require-converged-contact-updates` to require geometry
convergence. Failed IP solves return exit code 2.

## ParaView

Output directories use `<geometry>_<hex|tet>_<UTC timestamp>`, for example
`strips_hex_20261123_153045Z`, under `ParaView/`. Same-second runs receive a
numeric suffix. Each directory contains `run-info.txt` with the arguments and
resolved settings. Open the `.pvd` file to animate the saved configurations;
the mesh is already deformed.

To display exterior faces, process each body separately:
**Threshold by cell attribute → Clean to Grid → Extract Surface**. Use an
absolute cleaning tolerance of `1e-9` and display only the extracted surface
at opacity 1. Threshold before cleaning to keep separate contact bodies from
being welded together.
