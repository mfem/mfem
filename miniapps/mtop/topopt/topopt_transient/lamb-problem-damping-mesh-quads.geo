// wave_semi_infinite_with_damping_viz.geo
// Visualization mesh with damping boundary markers
//
// Shows the horseshoe damping region as embedded lines in gmsh.
// Loading strips are ONLY applied in the interior (undamped) region.
//
// DOMAIN: 1.5 m × 0.75 m
//   - Interior (undamped): x ∈ [0.25, 1.25], y ∈ [0.25, 0.75] m
//   - Damping: Horseshoe layer 0.25 m thick (left, right, bottom)
//   - Loading strips: 6 strips in interior only, x ∈ [0.35, 1.15] m
//
// The interior_damping_interface (curve 30) marks where damping starts.
// User can solve diffusion in region bounded by curves to determine γ(x).
//
// PHYSICAL SURFACE:
//   1: domain (single material, damping via coefficient)
//
// PHYSICAL CURVES:
//   10-13: exterior boundaries
//   21-26: loading strips
//   30: interior_damping_interface (VISUALIZATION MARKER)
//
// MESH COMMAND:
//   gmsh -2 -format msh2 wave_semi_infinite_viz_v2.geo -o wave_semi_infinite_viz.msh

SetFactory("Built-in");

// Domain
Lx = 1.5;
Ly = 0.75;

// Damping starts here (for visualization)
damp_thick = 0.25;
x_damp_left = damp_thick;
x_damp_right = Lx - damp_thick;
y_damp_bottom = damp_thick;

// Interior (undamped) region width
Lx_interior = x_damp_right - x_damp_left;  // 1.0 m

// Loading strips (ONLY in interior region, NOT in damping!)
Nstrip = 6;
strip_w = 0.05;
gap = (Lx_interior - Nstrip*strip_w)/(Nstrip + 1);  // Gap within interior region

// Mesh size
lc = 0.025;

// Output
Mesh.MshFileVersion = 2.2;
Mesh.Binary = 0;
Mesh.ElementOrder = 1;
Mesh.RecombineAll = 1;

// ============ POINTS ============
// Domain corners
p1 = newp; Point(p1) = {0, 0, 0, lc};
p2 = newp; Point(p2) = {Lx, 0, 0, lc};
p3 = newp; Point(p3) = {Lx, Ly, 0, lc};
p4 = newp; Point(p4) = {0, Ly, 0, lc};

// Damping interface markers (for visualization only)
p_damp_bl = newp; Point(p_damp_bl) = {x_damp_left, y_damp_bottom, 0, lc};
p_damp_br = newp; Point(p_damp_br) = {x_damp_right, y_damp_bottom, 0, lc};
p_damp_tl = newp; Point(p_damp_tl) = {x_damp_left, Ly, 0, lc};
p_damp_tr = newp; Point(p_damp_tr) = {x_damp_right, Ly, 0, lc};

// Top edge points: left damping, interior with loading strips, right damping
top_pts[] = {p4};  // Start at left corner

// Left damping region top (from left corner to interior left)
top_pts[] += {p_damp_tl};

// Interior region with loading strips (ONLY in undamped region!)
For i In {0:Nstrip-1}
  x_left = x_damp_left + gap + i*(strip_w + gap);
  x_right = x_left + strip_w;

  pl = newp; Point(pl) = {x_left, Ly, 0, lc};
  pr = newp; Point(pr) = {x_right, Ly, 0, lc};

  top_pts[] += {pl, pr};
EndFor

// Right damping region top (from interior right to right corner)
top_pts[] += {p_damp_tr};
top_pts[] += {p3};  // End at right corner

// ============ CURVES ============
// Exterior boundary
c_bot = newc; Line(c_bot) = {p1, p2};
c_right = newc; Line(c_right) = {p2, p3};
c_left = newc; Line(c_left) = {p4, p1};

// Top with loading strips
top_curves[] = {};
For i In {0:#top_pts[]-2}
  c = newc; Line(c) = {top_pts[i], top_pts[i+1]};
  top_curves[] += {c};
EndFor

// Damping interface markers (embedded lines for visualization)
c_damp_bot = newc; Line(c_damp_bot) = {p_damp_bl, p_damp_br};
c_damp_left = newc; Line(c_damp_left) = {p_damp_bl, p_damp_tl};
c_damp_right = newc; Line(c_damp_right) = {p_damp_br, p_damp_tr};

// ============ SURFACE ============
// Single domain
neg_top[] = {};
For i In {0:#top_curves[]-1}
  neg_top[] += {-top_curves[#top_curves[]-1-i]};
EndFor

cl = newc;
Curve Loop(cl) = {c_bot, c_right, neg_top[], c_left};
s = news;
Plane Surface(s) = {cl};

// Embed damping markers in surface (will show as internal lines)
Line{c_damp_bot, c_damp_left, c_damp_right} In Surface{s};

// ============ PHYSICAL GROUPS ============
// Single domain
Physical Surface("domain", 1) = {s};

// Exterior boundaries
Physical Curve("exterior_left", 10) = {c_left};
Physical Curve("exterior_bottom", 11) = {c_bot};
Physical Curve("exterior_right", 12) = {c_right};

// Top - non-loading (gaps between strips + left/right damping top)
top_exterior[] = {};
For i In {0:#top_curves[]-1}
  // Skip the 6 strip curves: indices 2, 4, 6, 8, 10, 12
  is_strip = (i == 2 || i == 4 || i == 6 || i == 8 || i == 10 || i == 12);
  If (!is_strip)
    top_exterior[] += {top_curves[i]};
  EndIf
EndFor
Physical Curve("exterior_top", 13) = {top_exterior[]};

// Loading strips (the 6 actual strip curves at even indices 2, 4, 6, 8, 10, 12)
Physical Curve("load_strip_1", 21) = {top_curves[2]};   // [0.35, 0.40]
Physical Curve("load_strip_2", 22) = {top_curves[4]};   // [0.50, 0.55]
Physical Curve("load_strip_3", 23) = {top_curves[6]};   // [0.65, 0.70]
Physical Curve("load_strip_4", 24) = {top_curves[8]};   // [0.80, 0.85]
Physical Curve("load_strip_5", 25) = {top_curves[10]};  // [0.95, 1.00]
Physical Curve("load_strip_6", 26) = {top_curves[12]};  // [1.10, 1.15]

// Damping interface markers (for visualization in gmsh)
Physical Curve("interior_damping_interface", 30) = {c_damp_bot, c_damp_left, c_damp_right};

// ============ MESH CONTROL ============
//
// Uniform boundary spacing approximately lc.
// Transfinite Curve counts NODES, including both endpoints.

// Exterior boundary
n_bottom   = Ceil(Lx / lc) + 1;       // 61 nodes, 60 elements
n_vertical = Ceil(Ly / lc) + 1;       // 31 nodes, 30 elements

Transfinite Curve {c_bot} = n_bottom Using Progression 1;
Transfinite Curve {c_left, c_right} = n_vertical Using Progression 1;

// Embedded damping interface
n_interface_horizontal =
    Ceil(Lx_interior / lc) + 1;        // 41 nodes, 40 elements

n_interface_vertical =
    Ceil((Ly - y_damp_bottom) / lc) + 1; // 21 nodes, 20 elements

Transfinite Curve {c_damp_bot}
    = n_interface_horizontal Using Progression 1;

Transfinite Curve {c_damp_left, c_damp_right}
    = n_interface_vertical Using Progression 1;

// Top boundary: outer damping portions
n_outer_top = Ceil(damp_thick / lc) + 1; // 11 nodes

Transfinite Curve {
    top_curves[0],
    top_curves[14]
} = n_outer_top Using Progression 1;

// Top boundary: seven gaps
n_gap = Ceil(gap / lc) + 1; // 5 nodes for each 0.10 m gap

For i In {0:Nstrip}
    Transfinite Curve {top_curves[2*i + 1]}
        = n_gap Using Progression 1;
EndFor

// Top boundary: six loading strips
n_strip = Ceil(strip_w / lc) + 1; // 3 nodes per 0.05 m strip

For i In {0:Nstrip-1}
    Transfinite Curve {top_curves[2*i + 2]}
        = n_strip Using Progression 1;
EndFor

// Global mesh-size controls
Mesh.MeshSizeMin = 0.8*lc;
Mesh.MeshSizeMax = lc;
Mesh.MeshSizeFromPoints = 1;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 1;

// Generate right-angle-oriented triangles suitable for recombination
Mesh.Algorithm = 8; // Frontal-Delaunay for quads

// Recombine only when acceptable
Recombine Surface {s};

Mesh.RecombinationAlgorithm = 1; // Blossom
Mesh.RecombineOptimizeTopology = 10;
Mesh.RecombineNodeRepositioning = 1;

// Reject low-quality quadrilateral candidates.
// Regions that cannot form acceptable quads remain triangular.
Mesh.RecombineMinimumQuality = 0.20;

Mesh.Smoothing = 10;
