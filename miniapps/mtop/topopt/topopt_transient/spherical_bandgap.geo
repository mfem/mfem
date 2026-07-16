// spherical_bandgap.geo
// 3D spherical geometry for band-gap topology optimization
//
// Concentric spherical shells:
//   - Source sphere: r < 0.5 (wave generation, passive)
//   - Design region: 0.5 < r < 6.0 (optimize material distribution)
//   - Receiver shell: 6.0 < r < 7.0 (measure wave energy, passive)
//   - Gap shell: 7.0 < r < 7.5 (fixed material, passive)
//   - Damping shell: 7.5 < r < 10.0 (sponge layer, passive)
//
// PHYSICAL VOLUMES:
//   1: source
//   2: design
//   3: receiver
//   4: gap
//   5: damping
//
// PHYSICAL SURFACE:
//   100: outer_boundary (for absorbing BC)
//
// MESH COMMAND:
//   gmsh -3 -format msh2 spherical_bandgap.geo -o spherical_bandgap.msh

SetFactory("OpenCASCADE");  // Required for sphere operations

// Radii definitions (matching user requirements)
r_source = 0.5;
r_design = 6.0;
r_receiver = 7.0;
r_gap = 7.5;
r_outer = 10.0;

// Mesh size control (coarser in outer regions, finer in design)
lc_source = 0.3;
lc_design = 0.3;      // ~20 elements per diameter in design region
lc_receiver = 0.4;
lc_gap = 0.5;
lc_damping = 0.6;

// Create concentric spheres
Sphere(1) = {0, 0, 0, r_source};
Sphere(2) = {0, 0, 0, r_design};
Sphere(3) = {0, 0, 0, r_receiver};
Sphere(4) = {0, 0, 0, r_gap};
Sphere(5) = {0, 0, 0, r_outer};

// Get volume IDs (OpenCASCADE creates volumes automatically)
v_source = 1;
v_to_design = 2;
v_to_receiver = 3;
v_to_gap = 4;
v_to_outer = 5;

// Create shells via boolean difference
// Note: After each operation, volume IDs may change
// We'll use the resulting volume tags explicitly

// Design shell = sphere(r_design) - sphere(r_source)
BooleanDifference{Volume{v_to_design}; Delete;}{Volume{v_source};}

// After this operation:
// - Volume 1 = source (unchanged)
// - Volume 2 = design shell (result of difference)

// Receiver shell = sphere(r_receiver) - sphere(r_design)
// Need to reference the newly created volumes
BooleanDifference{Volume{v_to_receiver}; Delete;}{Volume{v_to_design};}

// After this operation:
// - Volume 1 = source
// - Volume 2 = design shell
// - Volume 3 = receiver shell

// Gap shell = sphere(r_gap) - sphere(r_receiver)
BooleanDifference{Volume{v_to_gap}; Delete;}{Volume{v_to_receiver};}

// After this operation:
// - Volume 1 = source
// - Volume 2 = design shell
// - Volume 3 = receiver shell
// - Volume 4 = gap shell

// Damping shell = sphere(r_outer) - sphere(r_gap)
BooleanDifference{Volume{v_to_outer}; Delete;}{Volume{v_to_gap};}

// Final volumes:
// - Volume 1 = source
// - Volume 2 = design shell
// - Volume 3 = receiver shell
// - Volume 4 = gap shell
// - Volume 5 = damping shell

// Physical Volumes (MFEM element attributes)
Physical Volume("source", 1) = {1};
Physical Volume("design", 2) = {2};
Physical Volume("receiver", 3) = {3};
Physical Volume("gap", 4) = {4};
Physical Volume("damping", 5) = {5};

// Physical Surface (outer boundary for absorbing BC)
// The outer surface of volume 5 (damping shell)
// OpenCASCADE creates surfaces automatically, we need to find the outermost
Physical Surface("outer_boundary", 100) = {6};  // Surface 6 is typically the outer surface

// Mesh size fields for different regions
Field[1] = MathEval;
Field[1].F = Sprintf("(x^2 + y^2 + z^2 < %g^2) ? %g : "
                     "(x^2 + y^2 + z^2 < %g^2) ? %g : "
                     "(x^2 + y^2 + z^2 < %g^2) ? %g : "
                     "(x^2 + y^2 + z^2 < %g^2) ? %g : %g",
                     r_source, lc_source,
                     r_design, lc_design,
                     r_receiver, lc_receiver,
                     r_gap, lc_gap,
                     lc_damping);
Background Field = 1;

// Mesh algorithm settings
Mesh.Algorithm3D = 4;           // Frontal algorithm (good quality)
Mesh.ElementOrder = 1;          // Linear elements
Mesh.OptimizeThreshold = 0.3;   // Optimize mesh quality
Mesh.MshFileVersion = 2.2;      // MFEM-compatible format

// Additional mesh quality controls
Mesh.Optimize = 1;              // Enable optimization
Mesh.OptimizeNetgen = 1;        // Use Netgen optimizer
