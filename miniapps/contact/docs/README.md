# Contact miniapps

Parallel frictionless solid contact using MFEM elasticity, Tribol mortar
constraints, and an interior-point solver. The miniapps use first-order solid
and contact elements.

| Executable | Problems |
|---|---|
| `contact` | 0 two-block, 1 ironing, 2 beam/sphere |
| `contact_examples` | 3 onion shells, 4 folded strip, 5 ring, 6 C-clip/pin, 7 twisting elastic strips, 8 rope/cylinder tightening |

Problem geometry, materials, and loading for cases 3–8 are described in
[EXAMPLES.md](EXAMPLES.md).

## Build and run

An MFEM build with MPI and Tribol is required. Run from `miniapps/contact`:

```bash
make -j12 contact contact_examples

./run-contact.sh --driver contact -- \
  -prob 0 -load-steps 4 -extra-load-steps 0
```

The launcher defaults to laptop CPU execution with one MPI rank. Use
`--ranks N` to choose the number of MPI processes. Select `--site trike` or
`--site lc` for those environments. Use `./run-contact.sh --help` for
launcher options and sample runs, and `./contact -h` or `./contact_examples -h`
for solver options.
See [BUILD.md](BUILD.md) for dependencies, CUDA/HIP setup, and launch
instructions.

## Meshes and loading

Problems 0–2 generate their meshes. `-hex` and `-tet` select the cell type;
problems 0/1 default to hexes and problem 2 to tets.
Generated beam/sphere resolution is controlled by `-beam-nc`, `-beam-ny`,
`-sphere-n`, and `-sphere-nr`. Beam/sphere contact uses two independent
interfaces: the beam against the outer shell, and the insert against the
shell's inner surface. Only the beam's lower outer arc is a contact boundary.

`-load-steps` controls the loading ramp. The `contact` driver also accepts
`-extra-load-steps`: lateral sliding for problems 0/1 or continued pressure
loading beyond the initial ramp for problem 2. Problems 3–8 are loading-only.
Problems 0/1 use linear elasticity; problem 2 supports `-nonlin`.
The additional examples use the material models listed in EXAMPLES.md.

## Common options

Defaults are zero refinements, CG with AMGF, automatic filtered-backend selection,
CPU execution, GLVis and ParaView output, and print level 2. The automatic backend
uses MUMPS when available on CPU. Use `-no-vis` for runs without GLVis.

| Option | Meaning |
|---|---|
| `-load-steps N` | Load increments in the initial ramp |
| `-extra-load-steps N` | Additional sliding (0/1) or pressure-loading (2) increments; `contact` only, default 0 |
| `-sr N`, `-pr N` | Serial and parallel uniform refinements; both default to 0 |
| `-amgf`, `-no-amgf` | CG with contact-subspace filtering (default) or ordinary AMG |
| `-amgf-fsolver auto` | Select an available direct solver for the filtered subspace |
| `-linear-solver cg` | Reduced-system solver; a direct backend such as `mumps` can also be selected |
| `-opt-maxit N` | IP iteration limit per solve |
| `-contact-max-updates N` | Geometry/material update limit for nonlinear problems, default 1 |
| `-d cpu`, `-d cuda`, `-d hip` | MFEM device backend |
| `-vis`, `-no-vis` | Enable or disable GLVis; enabled by default |
| `-paraview`, `-no-paraview` | Enable or disable saved visualization; enabled by default |
| `-pl 0/1/2/3` | Silent / summary / solver tables / debug output |

Every nonlinear problem enables geometry/material updates, including problem 2
with `-nonlin`. The default limit is 1; launcher samples use 5 for problems 4–8
and 1 for nonlinear problem 2. Linear problems use one contact solve per load step.

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
