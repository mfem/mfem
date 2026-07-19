// ---------------------------------------------------------------------------
// Bent elbow pipe (hollow) resembling MFEM's pipe-nurbs.mesh.
//
//   * Vertical run standing on the floor z = 0 (bottom annular face).
//   * A 90 degree sweep (bend) turning from +z to +x.
//   * A short horizontal stub ending in an opening facing +x.
//   * Annular cross-section (outer radius Ro, inner radius Ri).
//   * Approx. bounding box (-3,-3,0) .. (10,3,23), like pipe-nurbs.mesh.
//
// Physical tags -> MFEM attributes:
//   Physical Volume  "pipe"    -> domain attribute   1
//   Physical Surface "outer"   -> boundary attribute 1  (whole outer wall: contact candidate)
//   Physical Surface "inner"   -> boundary attribute 2  (bore)
//   Physical Surface "bottom"  -> boundary attribute 3  (foot on the floor, z = 0)
//   Physical Surface "opening" -> boundary attribute 4  (horizontal end opening, +x)
//
// Mesh with:
//     gmsh pipe_elbow.geo -3 -format msh22 -o pipe_elbow.msh
// ---------------------------------------------------------------------------

// ---- Parameters ----
Ro = 4.0;    // outer radius of the tube
Ri = 2.0;    // inner radius (bore)
H1 = 4.0;    // height of the vertical run (start of the bend centerline)
Rb = 7.0;    // bend (centerline) radius  -> outer of bend reaches x = Rb+Ro = 10
Lh = 4.0;    // length of the horizontal stub
lc = 0.6;    // cross-section mesh size
nz = 6;      // layers along the vertical run
nb = 12;     // layers around the bend
nh = 6;      // layers along the horizontal stub

// ---- Annular cross-section at z = 0 (normal +z), centered at origin ----
Point(1) = {0,   0,  0, lc};   // center
Point(2) = {Ro,  0,  0, lc};
Point(3) = {0,   Ro, 0, lc};
Point(4) = {-Ro, 0,  0, lc};
Point(5) = {0,  -Ro, 0, lc};
Point(6) = {Ri,  0,  0, lc};
Point(7) = {0,   Ri, 0, lc};
Point(8) = {-Ri, 0,  0, lc};
Point(9) = {0,  -Ri, 0, lc};

Circle(1) = {2, 1, 3};   Circle(2) = {3, 1, 4};
Circle(3) = {4, 1, 5};   Circle(4) = {5, 1, 2};   // outer
Circle(5) = {6, 1, 7};   Circle(6) = {7, 1, 8};
Circle(7) = {8, 1, 9};   Circle(8) = {9, 1, 6};   // inner

Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {5, 6, 7, 8};
Plane Surface(1) = {1, 2};        // annulus

// ---- Segment 1: vertical run, translate-extrude +z ----
v1[] = Extrude {0, 0, H1} { Surface{1}; Layers{nz}; Recombine; };
//   v1[0] = top face,  v1[1] = volume,
//   v1[2..5] = outer walls (arcs 1..4),  v1[6..9] = inner walls (arcs 5..8)

// ---- Segment 2: 90 deg bend, rotate-extrude about +y through (Rb,0,H1) ----
b[] = Extrude { {0,1,0}, {Rb,0,H1}, Pi/2 } { Surface{v1[0]}; Layers{nb}; Recombine; };

// ---- Segment 3: horizontal stub, translate-extrude +x ----
h[] = Extrude {Lh, 0, 0} { Surface{b[0]}; Layers{nh}; Recombine; };

// ---- Physical groups -> MFEM attributes ----
Physical Volume("pipe", 1) = { v1[1], b[1], h[1] };

Physical Surface("outer", 1) = { v1[2], v1[3], v1[4], v1[5],
                                 b[2],  b[3],  b[4],  b[5],
                                 h[2],  h[3],  h[4],  h[5] };
Physical Surface("inner", 2) = { v1[6], v1[7], v1[8], v1[9],
                                 b[6],  b[7],  b[8],  b[9],
                                 h[6],  h[7],  h[8],  h[9] };
Physical Surface("bottom",  3) = { 1 };       // foot at z = 0
Physical Surface("opening", 4) = { h[0] };    // horizontal end (+x)

Mesh.MshFileVersion = 2.2;
