# Additional contact examples

Build `contact_examples` and run it from `miniapps/contact`. These examples use
the same solver and visualization options as the original `contact` driver;
see [README.md](README.md). The IDs remain 3–8, and the default is problem 8.

| ID | Example | Element order | Material/load behavior |
|---|---|---|---|
| 3 | Nine nested spherical shells | 1 | Linear elasticity; pressure on inner and outer surfaces |
| 4 | Folded U-strip | 1 | Neo-Hookean; pressure loading/unloading and self-contact |
| 5 | Compressed ring | 1 | Neo-Hookean; pressure loading/unloading and self-contact |
| 6 | C-clip over an elastic pin | 1 | Insertion/removal; closing pressure starts halfway through insertion |
| 7 | Short twisting cylinder | 1 | Opposed end rotations and optional axial shortening |
| 8 | Twisting elastic strips, or strip around fixed core | 1 | Opposed end rotations, frictionless sliding |

All examples use first-order displacement and geometry.

All examples use simulation units. Problems 4–8 always use nonlinear elasticity.
The model has no tearing/fracture law. Large rotation can still cause mesh
inversion or solver failure; the absence of a damage law does not prevent that.

## Solver choice and a first strip run

Use CG+AMGF for the normal solver path. The commands below assume MUMPS is
available for the filtered subspace; `auto` can select another enabled backend.
For a small CPU reference, `-linear-solver mumps` directly solves the full reduced
Newton system and can be much faster than CG for thin sheets.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 mpirun -np 1 ./contact_examples \
  -prob 8 -two-strips -tet -sr 0 -pr 0 -strip-nx 4 \
  -twist-angle 360 -nsteps 36 -msteps 0 \
  -amgf -amgf-fsolver mumps -d cpu -no-vis -paraview -pl 2
```

For the full direct reference, replace `-amgf -amgf-fsolver mumps` with
`-linear-solver mumps`. To twist twice, use `-twist-angle 720 -nsteps 72`.
Use `-strip-nx 24` for a finer first-order mesh; omitted axial resolution defaults
to 48 cells. Strips default to zero serial refinement.

## Twisting elastic strips (8)

Two initially touching elastic strips of length 3, width 0.6, and thickness
0.02 start flat, stacked face-to-face at z=0. The lower occupies `-0.02 <= z <= 0`, the upper
`0 <= z <= 0.02`. Their coincident faces have independent DOFs. At full load,
the left grip rotates by `-twist-angle/2`, and the right by `+twist-angle/2`.
Interior nodes deform freely. Contact remains frictionless.

| Option | Default and meaning |
|---|---|
| `-two-strips` | Two separate strips (default mode) |
| `-strip-width`, `-strip-thickness` | 0.6 and 0.02 |
| `-strip-young-modulus`, `-strip-poisson-ratio` | E=6000, nu=0.3 |
| `-strip-nx` | Coarse axial cells; defaults to 48, in either mode |
| `-strip-separation` | Centerline distance; defaults to supplied thickness (touching) |
| `-wrap-angle` | Initial stress-free twist; defaults to 0 for two strips |
| `-twist-angle` | Added relative end twist; defaults to 360 degrees |
| `-twist-shortening` | Total symmetric inward grip motion; defaults to 0 |
| `-twist-length` | Reference length, default 3 |
| `-msteps` | Additional steps untwisting to the reference shape; default 0 |

Separation below thickness is rejected. A larger separation adds clearance.
There are two thickness cells and four width cells per strip. `-hex` switches
from tetrahedra to hexahedra. A nonzero initial wrap is mapped at the selected
first-order geometry nodes before refinement.

The material is an isotropic elastic sheet approximation. With E=6000,
nu=0.3, and thickness h=0.02, the membrane scale E*h is 120 and the continuum
bending scale E*h^3/[12*(1-nu^2)] is approximately 0.0044. First-order solid
cells can lock in thin-sheet bending, especially tetrahedra.

Only the facing broad surfaces form the contact pair. Opposite backs and thin
edges remain free surfaces. This example does not implement arbitrary all-face
self-contact or contact between every possible newly folded edge/back.

| Boundary attribute | Meaning |
|---|---|
| 1, 2 | Left/right end grips |
| 3 | Lower strip's facing broad surface, mortar |
| 4 | Upper strip's facing broad surface, non-mortar |
| 6 | Free backs and thin edges |

A small integrated mortar gap alone does not certify pointwise nonpenetration.

## Elastic strip around a stationary core (8)

`-wrap-core` selects a ribbon wrapped around a fixed radius-0.5 cylinder.
The reference wrap defaults to 360 degrees, radial clearance to 0.02, and
added twist to 180 degrees. All core DOFs are fixed. The strip uses the same
thin material profile, with six width cells; it uses first-order cells.
`-twist-radius`, `-core-gap`, `-wrap-angle`, and `-strip-nx` control the geometry.

```bash
mpirun -np 1 ./contact_examples -prob 8 -wrap-core -tet \
  -sr 0 -pr 0 -strip-nx 24 -wrap-angle 360 -core-gap 0.02 \
  -twist-angle 360 -twist-shortening 0.2 -nsteps 72 -msteps 0 \
  -amgf -amgf-fsolver mumps -d cpu -no-vis -paraview -pl 2
```

In core mode boundary 3 is the strip lateral contact surface, 4 the core
surface, and 5 the fixed core ends. The grip attributes remain 1 and 2.

## Folded U-strip (4)

A connected hex solid has two length-6 arms joined by a semicircular bend of
radius 0.72, width 0.9, and thickness 0.3. The initial arm gap is 1.14.
A short outer bend patch is clamped; pressure on the upper arm brings two
separate facing arm patches into self-contact. The bend is excluded from the
contact pair. E=1000, nu=0.3.

```bash
mpirun -np 1 ./contact_examples -prob 4 -sr 0 -pr 0 \
  -u-pressure 0.06 -nsteps 20 -msteps 20 \
  -amgf -amgf-fsolver mumps -d cpu -no-vis -paraview -pl 2
```

`-u-pressure` controls maximum pressure. Boundary 1 supports the bend, 2 carries
pressure, 3/4 are the disjoint contact patches, and 5 is free. The coarse
mesh retains 24 cells per arm and bend, four across width and thickness.

## Compressed ring (5)

A connected periodic hex ring has centerline radius 1, thickness 0.12, and
width 0.6. A short lower outer arc is clamped; pressure on the upper outer arc
brings opposite inner arcs into contact. E=1000, nu=0.3. The coarse mesh has
64 angular cells and two width/radial cells; first-order nodes follow the
analytic circular geometry, including at the periodic seam.

```bash
mpirun -np 1 ./contact_examples -prob 5 -sr 0 -pr 0 \
  -ring-pressure 2.5 -nsteps 20 -msteps 20 \
  -amgf -amgf-fsolver mumps -d cpu -no-vis -paraview -pl 2
```

Boundary 1 is support, 3/4 are contact arcs, 6 carries pressure, and 2 is free.
`-ring-pressure` controls peak pressure. Positive `-msteps` unloads the ring.

## C-clip and elastic pin (6)

An open annular C-clip moves over a separate supported cylindrical pin.
The inner clip/pin radius is 1, clip thickness 0.2, and extrusion width 0.6.
The clip has E=1000; the pin defaults to E=500; both use nu=0.3.
Motion is plane strain across the extrusion. Small central pin end patches
anchor its axis while the outer cylinder can compress.

At half the default insertion travel, pressure begins closing the clip
from top to bottom. It ramps to `-clip-pressure` at full default insertion;
removal reverses the ramp. The default total travel is 1.9 and pressure 40.

```bash
mpirun -np 1 ./contact_examples -prob 6 -tet -sr 0 -pr 0 \
  -clip-travel 1.9 -clip-pressure 40 -pin-young-modulus 500 \
  -nsteps 32 -msteps 32 -amgf -amgf-fsolver mumps \
  -d cpu -no-vis -paraview -pl 2
```

`-hex` selects hex elements; `-tet` is the default. Analytic geometry maps act
on the chosen first-order nodes. The clip/pin have volume attributes 1/2.
Boundary 1 is the grip, 2 clip contact, 5 pin contact, 8/9 closing loads,
and 10 central pin support. Other extrusion/free patches enforce plane strain.

## Twisting cylinder (7)

A filled hex cylinder defaults to length 3 and radius 0.5, with a small
stress-free bow to select a buckling direction. E=1000, nu=0.3. Left and right
grips rotate in opposite directions. `-twist-shortening` moves them inward by
half the supplied total distance each (default total 0.2).

```bash
mpirun -np 1 ./contact_examples -prob 7 -sr 0 -pr 0 \
  -twist-length 3 -twist-radius 0.5 -twist-angle 720 \
  -twist-shortening 0.2 -nsteps 144 -msteps 0 \
  -amgf -amgf-fsolver mumps -d cpu -no-vis -paraview -pl 2
```

The coarse mesh has 36 axial and 6-by-6 cross-section cells. Contact pairs are
selected between separated longitudinal bands, with neighbor bands and patch
edges excluded. This is a controlled self-contact demonstration, not an
all-face self-contact detector. Rotation above 360 degrees does not by itself
guarantee contact; geometry and shortening also matter.

## Onion shells (3)

Nine disconnected spherical octant shells have radii 5.5 through 10 in
0.5 increments. Consecutive shells have independent surface nodes and
nonmatching angular grids. E=1000, nu=0.33. Eight mortar interfaces pair
consecutive shells. Symmetry planes x/y/z=0 fix their normal displacement.
Equal pressure loads the innermost and outermost surfaces.

```bash
mpirun -np 1 ./contact_examples -prob 3 -sr 0 -pr 0 \
  -nsteps 4 -msteps 0 -amgf -amgf-fsolver mumps \
  -d cpu -no-vis -paraview -pl 2
```

The shell generator sets first-order nodes on the intended spherical/radial
map. Volume attributes 1–9 identify shells; radial boundary attributes 1–18
identify inner/outer surfaces; 21–23 identify symmetry planes.

## Output and convergence

Open the `.pvd` in a timestamped directory such as
`ParaView/strips_hex_20261123_153045Z/`. Run settings are saved in `run-info.txt`. See the [exterior visualization
recipe](README.md#paraview); threshold bodies before cleaning duplicated points.
Do not apply another displacement warp.

The geometry-update limit defaults to **10**. A converged IP result at that
limit is accepted and reported as **inexact**. Use
`-require-converged-contact-updates` to fail at that limit, or change
`-contact-max-updates` and `-contact-update-tol`. An IP failure still stops the
run. Integrated mortar feasibility can coexist with local surface overlap;
inspect the saved geometry and mesh quality.
