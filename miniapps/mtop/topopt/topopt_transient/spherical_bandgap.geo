// spherical_bandgap.geo
// 3D concentric-sphere geometry for the spherical band-gap topology
// optimization problem (SphericalBandGapProblem in ProblemSpecification.hpp).
//
// Regions (MFEM element attributes = physical volume tags):
//   1: source    r in [0.0, 0.5]   wave generation (passive)
//   2: design    r in [0.5, 6.0]   designable material (active)
//   3: receiver  r in [6.0, 7.0]   objective measurement shell (passive)
//   4: gap       r in [7.0, 7.5]   fixed buffer (passive)
//   5: damping   r in [7.5, 10.0]  sponge layer (passive)
//
// Boundary (MFEM boundary attribute = physical surface tag):
//   100: outer_boundary, the r = 10 sphere (absorbing BC)
//
// MESH COMMANDS:
//   production:  gmsh -3 -format msh2 spherical_bandgap.geo -o spherical_bandgap.msh
//   coarse test: gmsh -3 -format msh2 -clscale 2 spherical_bandgap.geo -o spherical_bandgap_coarse.msh
//
// CONSTRUCTION NOTES (why fragments, not chained differences):
// Chained BooleanDifference of nested balls creates disconnected results whose
// OCC tags are NOT predictable (this silently dropped the receiver shell in an
// earlier version of this file -> objective was identically zero).
// BooleanFragments of all balls at once yields non-overlapping, conformally
// glued regions; volumes and the outer surface are then identified
// geometrically (bounding box), never by guessed tags.

SetFactory("OpenCASCADE");

// Radii
r_source   = 0.5;
r_design   = 6.0;
r_receiver = 7.0;
r_gap      = 7.5;
r_outer    = 10.0;

// Target element sizes per region
lc_inner   = 0.3;   // source + design
lc_receiver = 0.4;
lc_gap     = 0.5;
lc_damping = 0.6;

// Nested balls; fragments split them into disjoint conformal shells.
Sphere(1) = {0, 0, 0, r_source};
Sphere(2) = {0, 0, 0, r_design};
Sphere(3) = {0, 0, 0, r_receiver};
Sphere(4) = {0, 0, 0, r_gap};
Sphere(5) = {0, 0, 0, r_outer};

vols() = BooleanFragments{ Volume{1, 2, 3, 4, 5}; Delete; }{};

// Classify the resulting volumes by bounding-box extent (each region has a
// unique outer radius, so xmax identifies it regardless of tag numbering).
eps = 1e-3;
For i In {0 : #vols()-1}
   bb() = BoundingBox Volume{vols(i)};
   xmax = bb(3);
   If (Fabs(xmax - r_source) < eps)
      src_vol = vols(i);
   ElseIf (Fabs(xmax - r_design) < eps)
      design_vol = vols(i);
   ElseIf (Fabs(xmax - r_receiver) < eps)
      receiver_vol = vols(i);
   ElseIf (Fabs(xmax - r_gap) < eps)
      gap_vol = vols(i);
   ElseIf (Fabs(xmax - r_outer) < eps)
      damping_vol = vols(i);
   EndIf
EndFor

Physical Volume("source",   1) = {src_vol};
Physical Volume("design",   2) = {design_vol};
Physical Volume("receiver", 3) = {receiver_vol};
Physical Volume("gap",      4) = {gap_vol};
Physical Volume("damping",  5) = {damping_vol};

// Outer boundary = the only surface that does not fit inside r < 9.
s_all()   = Surface In BoundingBox{-r_outer-1, -r_outer-1, -r_outer-1,
                                    r_outer+1,  r_outer+1,  r_outer+1};
s_inner() = Surface In BoundingBox{-9, -9, -9, 9, 9, 9};
s_all() -= s_inner();
Physical Surface("outer_boundary", 100) = {s_all()};

// Radially graded mesh size: min over per-region Ball fields.
Field[1] = Ball;
Field[1].XCenter = 0; Field[1].YCenter = 0; Field[1].ZCenter = 0;
Field[1].Radius = r_design;   Field[1].VIn = lc_inner;    Field[1].VOut = lc_damping;
Field[2] = Ball;
Field[2].XCenter = 0; Field[2].YCenter = 0; Field[2].ZCenter = 0;
Field[2].Radius = r_receiver; Field[2].VIn = lc_receiver; Field[2].VOut = lc_damping;
Field[3] = Ball;
Field[3].XCenter = 0; Field[3].YCenter = 0; Field[3].ZCenter = 0;
Field[3].Radius = r_gap;      Field[3].VIn = lc_gap;      Field[3].VOut = lc_damping;
Field[4] = Min;
Field[4].FieldsList = {1, 2, 3};
Background Field = 4;

// The background field is the single source of mesh size.
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 0;

// Delaunay 3D (no external dependencies) + optimization.
Mesh.Algorithm3D = 1;
Mesh.Optimize = 1;
Mesh.ElementOrder = 1;
Mesh.MshFileVersion = 2.2;
