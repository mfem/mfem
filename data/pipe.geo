SetFactory("Built-in");

// Set the geometry order (1, 2, ..., 9)
order = 2;

// Outer radius of the tube
Ro = 3.0;
// Inner radius (bore)
Ri = 2.0;
// Height of the vertical run, up to the start of the bend centerline
H1 = 4.0;
// Bend centerline radius (the outer of the bend reaches x = Rb + Ro = 10)
Rb = 7.0;
// Length of the horizontal stub
Lh = 4.0;

// Number of elements per quarter arc (4*nc elements around the ring)
nc = 1;
// Number of elements across the wall thickness (radial, Ri to Ro)
nr = 1;
// Layers along the vertical run, around the bend, and along the horizontal stub
nz = 1;
nb = 1;
nh = 1;

// Annular cross-section at z = 0 (normal +z), centred at the origin;
// point 1 is only an arc centre
Point(1) = {0,   0,  0};
Point(2) = {Ro,  0,  0};
Point(3) = {0,   Ro, 0};
Point(4) = {-Ro, 0,  0};
Point(5) = {0,  -Ro, 0};
Point(6) = {Ri,  0,  0};
Point(7) = {0,   Ri, 0};
Point(8) = {-Ri, 0,  0};
Point(9) = {0,  -Ri, 0};

// Outer and inner circle arcs (quarter circles)
Circle(1) = {2, 1, 3};   Circle(2) = {3, 1, 4};
Circle(3) = {4, 1, 5};   Circle(4) = {5, 1, 2};
Circle(5) = {6, 1, 7};   Circle(6) = {7, 1, 8};
Circle(7) = {8, 1, 9};   Circle(8) = {9, 1, 6};

// Radial spokes joining the inner and outer circles; they split the annulus
// into four quad blocks so it can be a structured O-grid
Line(11) = {6, 2};   Line(12) = {7, 3};
Line(13) = {8, 4};   Line(14) = {9, 5};

// Four transfinite quad blocks tiling the annulus
// (each: radial, outer arc, radial, inner arc)
Curve Loop(21) = {11, 1, -12, -5};   Plane Surface(31) = {21};
Curve Loop(22) = {12, 2, -13, -6};   Plane Surface(32) = {22};
Curve Loop(23) = {13, 3, -14, -7};   Plane Surface(33) = {23};
Curve Loop(24) = {14, 4, -11, -8};   Plane Surface(34) = {24};

// Circumferential and radial node counts, then make the blocks structured quads
Transfinite Curve{1:8}   = nc + 1;
Transfinite Curve{11:14} = nr + 1;
Transfinite Surface{31:34};
Recombine Surface{31:34};

// Segment 1: the vertical run, made by translating the section +z.  Extruding
// the four blocks returns six entries per block -- top face, volume, then the
// four side walls in the order radial, outer, radial, inner -- so per segment
// the outer walls land at indices 3, 9, 15, 21 and the inner walls at 5, 11,
// 17, 23.  This ordering survives every extrude, including the rotation below.
v[] = Extrude {0, 0, H1} { Surface{31:34}; Layers{nz}; Recombine; };

// Segment 2: the 90 degree bend, made by rotating the section a quarter turn
// about the +y axis through (Rb, 0, H1)
b[] = Extrude { {0, 1, 0}, {Rb, 0, H1}, Pi/2 }
      { Surface{v[{0:18:6}]}; Layers{nb}; Recombine; };

// Segment 3: the horizontal stub, made by translating the bend's end face +x
h[] = Extrude {Lh, 0, 0}
      { Surface{b[{0:18:6}]}; Layers{nh}; Recombine; };

// Tag volumes and surfaces: domain attribute 1 is the pipe; boundary
// attributes are 1 - outer wall, 2 - bore, 3 - foot on the floor (z = 0),
// 4 - horizontal end opening (+x)
vol[] = {1:19:6};   // extrude index 1: the volume
out[] = {3:21:6};   // index 3: outer lateral surface
inn[] = {5:23:6};   // index 5: inner lateral surface
top[] = {0:18:6};   // index 0: top face (opening)

Physical Volume ("pipe",    1) = { v[{vol[]}], b[{vol[]}], h[{vol[]}] };
Physical Surface("outer",   1) = { v[{out[]}], b[{out[]}], h[{out[]}] };
Physical Surface("inner",   2) = { v[{inn[]}], b[{inn[]}], h[{inn[]}] };
Physical Surface("bottom",  3) = { 31:34 };
Physical Surface("opening", 4) = { h[{top[]}] };

// Optimize the high-order mesh
// See https://gmsh.info/doc/texinfo/gmsh.html#index-Mesh_002eHighOrderOptimize
Mesh.HighOrderOptimize = 1;

// Generate 3D mesh, then raise it to the chosen order.  With order >= 2 gmsh
// places the new edge and face nodes on the true geometry -- the circle arcs of
// the section and the surface of revolution of the bend -- so the curved
// elements reproduce the pipe exactly instead of faceting it.
Mesh 3;
SetOrder order;

// The circle-centre points (e.g. point 1) are only arc centres, but gmsh still
// meshes them as stray nodes sitting inside the flat end faces.  Weld
// coincident nodes so those floating duplicates are removed.
Coherence Mesh;

Mesh.MshFileVersion = 2.2;

Save Sprintf("pipe-structured-o%01g.msh", order);
