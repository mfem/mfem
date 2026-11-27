# Additional contact examples

Build `contact_examples` and run it from `miniapps/contact`. These examples use
the same solver and visualization options as the original `contact` driver;
see [README.md](README.md). The launcher defaults to laptop CPU execution;
select `--site trike` or `--site lc` for those environments. The IDs are
3–8, and the default is problem 7.

| ID | Example | Element order | Material/load behavior |
|---|---|---|---|
| 3 | Nine nested spherical shells | 1 | Linear elasticity; pressure on inner and outer surfaces |
| 4 | Folded U-strip | 1 | Neo-Hookean; pressure loading and self-contact |
| 5 | Compressed ring | 1 | Neo-Hookean; pressure loading and self-contact |
| 6 | C-clip over an elastic pin | 1 | Insertion with elastic opening around the pin |
| 7 | Two twisting elastic strips | 1 | Opposed end rotations, frictionless sliding |
| 8 | Rope tightening around an elastic cylinder | 1 | Opposed winding and outward pulling; supported cylinder ends |

The examples apply increasing loads.
All examples use first-order displacement and geometry, with no uniform
refinement by default. Common solver/output options are listed in [README.md](README.md).

All examples use simulation units. Problems 4–8 always use nonlinear elasticity.
The model has no tearing/fracture law. Large rotation can still cause mesh
inversion or solver failure; the absence of a damage law does not prevent that.

## Solver choice and a first strip run

Use CG+AMGF for the normal solver path. The commands below assume MUMPS is
available for the filtered subspace; `auto` can select another enabled backend.
For a small CPU reference, `-linear-solver mumps` directly solves the full reduced
Newton system and can be much faster than CG for thin sheets.

```bash
./run-contact.sh --driver contact_examples -- \
  -prob 7 -tet -strip-nx 4 \
  -twist-angle 360 -load-steps 36 -contact-max-updates 5
```

For the full direct reference, add
`-linear-solver mumps`. To twist twice, use `-twist-angle 720 -load-steps 72`.
Use `-strip-nx 24` for a finer first-order mesh; omitted axial resolution defaults
to 48 cells. Strips default to zero serial refinement.

## Twisting elastic strips (7)

Two initially touching elastic strips of length 3, width 0.6, and thickness
0.02 start flat, stacked face-to-face at z=0. The lower occupies `-0.02 <= z <= 0`, the upper
`0 <= z <= 0.02`. Their coincident faces have independent DOFs. At full load,
the left grip rotates by `-twist-angle/2`, and the right by `+twist-angle/2`.
Interior nodes deform freely. Contact remains frictionless.

| Option | Default and meaning |
|---|---|
| `-strip-width`, `-strip-thickness` | 0.6 and 0.02 |
| `-strip-young-modulus`, `-strip-poisson-ratio` | E=6000, nu=0.3 |
| `-strip-nx` | Coarse axial cells; defaults to 48 |
| `-strip-separation` | Centerline distance; defaults to supplied thickness (touching) |
| `-wrap-angle` | Initial stress-free twist; defaults to 0 for two strips |
| `-twist-angle` | Added relative end twist; defaults to 360 degrees |
| `-twist-shortening` | Total symmetric inward grip motion; defaults to 0 |
| `-twist-length` | Reference length, default 3 |

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

## Rope tightening around an elastic cylinder (8)

A round rope starts in a stress-free two-turn helix around a solid cylinder.
The cylinder has radius 0.5 and length 3; the rope has radius 0.08 and spans
its middle 80%, with an initial radial clearance of 0.01. Only the cylinder
end caps are clamped. The rest of the cylinder can indent and compress.

The rope grips rotate in opposite directions to add 90 degrees of winding
and move axially outward by 0.1 each. Both motions ramp together over 48 load
steps. Winding and pulling stretch the rope, creating tension and inward
contact forces on the cylinder. Cylinder deformation comes from the contact
solve; no prescribed cylinder contraction or external squeezing pressure is
applied. Contact is frictionless.

Both bodies use Neo-Hookean elasticity. The rope defaults to E=3000 and the
cylinder to E=1000, both with nu=0.3. This is a solid elastic approximation of
a rope, without a fiber or braided constitutive model. These defaults aim for
a modest visible indentation; its magnitude depends on the mesh and solve.

```bash
./run-contact.sh --driver contact_examples -- -prob 8 -tet \
  -load-steps 48 -contact-max-updates 5
```

| Option | Default and meaning |
|---|---|
| `-rope-radius` | Round cross-section radius, 0.08 |
| `-rope-wrap-angle` | Initial stress-free winding, 720 degrees |
| `-rope-tighten-angle` | Additional total winding, 90 degrees; each grip rotates by half |
| `-rope-pull` | Total outward axial grip separation, 0.2 |
| `-rope-gap` | Initial radial clearance, 0.01 |
| `-rope-nx` | Cells along the rope centerline, 48 |
| `-rope-young-modulus`, `-rope-poisson-ratio` | E=3000, nu=0.3 |
| `-cylinder-radius`, `-cylinder-length` | 0.5 and 3 |
| `-cylinder-nx`, `-cylinder-nc` | 12 axial cells and 6 cells per cross-section direction |
| `-cylinder-young-modulus`, `-cylinder-poisson-ratio` | E=1000, nu=0.3 |

Decrease `-cylinder-young-modulus` to 500 for a softer cylinder. Increase
`-rope-tighten-angle` or `-rope-pull` for stronger tightening. The driver
reports the maximum inward displacement of cylinder surface nodes and its
percentage of the reference radius after each accepted step. This measures
local indentation; it is not a volume-average compression measure.

The rope has 48 cells along its centerline and a 4-by-4 filled cross-section.
The cylinder has 12 axial and 6-by-6 cross-section cells. Together these give
1,200 hexes or 7,200 tetrahedra. `-rope-nx`, `-cylinder-nx`, and `-cylinder-nc`
control the resolutions independently. `-hex` selects hexahedra; tetrahedra are the
default and share the same first-order geometry. Only the inward-facing half
of the rope participates in rope/cylinder contact. There is no rope-to-rope
self-contact; choose the radius and wrap so neighboring turns stay apart.

| Boundary attribute | Meaning |
|---|---|
| 1, 2 | Left/right rope grips |
| 3 | Inward-facing rope surface, mortar |
| 4 | Cylinder lateral surface, non-mortar |
| 5 | Supported cylinder end caps |
| 6 | Free outward-facing rope surface |

Volume attributes 1 and 2 identify the rope and cylinder. ParaView output is
named `rope_hex_<timestamp>` or `rope_tet_<timestamp>`. Color by volume
attribute to distinguish the two bodies, and use deformation scale 1 when
assessing contact and squeezing.

## Folded U-strip (4)

A connected hex solid has two length-6 arms joined by a semicircular bend of
radius 0.72, width 0.9, and thickness 0.3. The initial arm gap is 1.14.
A short outer bend patch is clamped; pressure on the upper arm brings two
separate facing arm patches into self-contact. The bend is excluded from the
contact pair. E=1000, nu=0.3.

```bash
./run-contact.sh --driver contact_examples -- -prob 4 \
  -u-pressure 0.06 -load-steps 20 -contact-max-updates 5
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
./run-contact.sh --driver contact_examples -- -prob 5 \
  -ring-pressure 2.5 -load-steps 20 -contact-max-updates 5
```

Boundary 1 is support, 3/4 are contact arcs, 6 carries pressure, and 2 is free.
`-ring-pressure` controls peak pressure.

## C-clip and elastic pin (6)

An open annular C-clip moves over a separate supported cylindrical pin.
The inner clip/pin radius is 1, clip thickness 0.2, and extrusion width 0.6.
The clip has E=1000; the pin defaults to E=500; both use nu=0.3.
Motion is plane strain across the extrusion. Small central pin end patches
anchor its axis while the outer cylinder can compress.

The grip translates the clip over the pin. Contact opens the lips during
entry, and the clip recovers toward its seated shape as insertion finishes.
The default total travel is 1.9. No closing pressure is applied.

```bash
./run-contact.sh --driver contact_examples -- -prob 6 -tet \
  -clip-travel 1.9 -pin-young-modulus 500 \
  -load-steps 32 -contact-max-updates 5
```

`-hex` selects hex elements; `-tet` is the default. Analytic geometry maps act
on the chosen first-order nodes. The clip/pin have volume attributes 1/2.
Boundary 1 is the grip, 2 clip contact, 5 pin contact, and 10 central pin
support. Other extrusion/free patches enforce plane strain.

## Onion shells (3)

Nine disconnected spherical octant shells have radii 5.5 through 10 in
0.5 increments. Consecutive shells have independent surface nodes and
nonmatching angular grids. E=1000, nu=0.33. Eight mortar interfaces pair
consecutive shells. Symmetry planes x/y/z=0 fix their normal displacement.
Equal pressure loads the innermost and outermost surfaces.

```bash
./run-contact.sh --driver contact_examples -- -prob 3 \
  -load-steps 4
```

The shell generator sets first-order nodes on the intended spherical/radial
map. Volume attributes 1–9 identify shells; radial boundary attributes 1–18
identify inner/outer surfaces; 21–23 identify symmetry planes.

## Output and convergence

Open the `.pvd` in a timestamped directory such as
`ParaView/strips_hex_20261123_153045Z/`. Run settings are saved in `run-info.txt`. See the [exterior visualization
recipe](README.md#paraview); threshold bodies before cleaning duplicated points.
Do not apply another displacement warp.

Nonlinear problems enable geometry/material updates automatically. The driver
limit defaults to **1**; the sample commands use **5**. A converged IP result
at the update limit is accepted and reported as **inexact**. Use
`-require-converged-contact-updates` to fail at that limit, or change
`-contact-max-updates` and `-contact-update-tol`. An IP failure still stops the
run. Integrated mortar feasibility can coexist with local surface overlap;
inspect the saved geometry and mesh quality.
