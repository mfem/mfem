// Define the cube sizes
L_outer = 1.0;
L_inner = 0.5;

// Set mesh size and algorithm
mesh_size = 0.4;
Mesh.Algorithm3D = 1; // Delaunay algorithm for 3D mesh
Mesh.CharacteristicLengthFactor = 1.0;
Mesh.MshFileVersion = 2.2;

// Define center point for concentric cubes
cx = 0.5;
cy = 0.5;
cz = 0.5;

// Define the points (vertices of the outer cube)
Point(1) = {cx-L_outer/2, cy-L_outer/2, cz-L_outer/2, mesh_size};
Point(2) = {cx+L_outer/2, cy-L_outer/2, cz-L_outer/2, mesh_size};
Point(3) = {cx+L_outer/2, cy+L_outer/2, cz-L_outer/2, mesh_size};
Point(4) = {cx-L_outer/2, cy+L_outer/2, cz-L_outer/2, mesh_size};
Point(5) = {cx-L_outer/2, cy-L_outer/2, cz+L_outer/2, mesh_size};
Point(6) = {cx+L_outer/2, cy-L_outer/2, cz+L_outer/2, mesh_size};
Point(7) = {cx+L_outer/2, cy+L_outer/2, cz+L_outer/2, mesh_size};
Point(8) = {cx-L_outer/2, cy+L_outer/2, cz+L_outer/2, mesh_size};

// Define the points (vertices of the inner cube)
Point(9) = {cx-L_inner/2, cy-L_inner/2, cz-L_inner/2, mesh_size};
Point(10) = {cx+L_inner/2, cy-L_inner/2, cz-L_inner/2, mesh_size};
Point(11) = {cx+L_inner/2, cy+L_inner/2, cz-L_inner/2, mesh_size};
Point(12) = {cx-L_inner/2, cy+L_inner/2, cz-L_inner/2, mesh_size};
Point(13) = {cx-L_inner/2, cy-L_inner/2, cz+L_inner/2, mesh_size};
Point(14) = {cx+L_inner/2, cy-L_inner/2, cz+L_inner/2, mesh_size};
Point(15) = {cx+L_inner/2, cy+L_inner/2, cz+L_inner/2, mesh_size};
Point(16) = {cx-L_inner/2, cy+L_inner/2, cz+L_inner/2, mesh_size};

// Define the lines (edges of the outer cube)
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};
Line(9) = {1, 5};
Line(10) = {2, 6};
Line(11) = {3, 7};
Line(12) = {4, 8};

// Define the lines (edges of the inner cube)
Line(13) = {9, 10};
Line(14) = {10, 11};
Line(15) = {11, 12};
Line(16) = {12, 9};
Line(17) = {13, 14};
Line(18) = {14, 15};
Line(19) = {15, 16};
Line(20) = {16, 13};
Line(21) = {9, 13};
Line(22) = {10, 14};
Line(23) = {11, 15};
Line(24) = {12, 16};

// Define the surfaces (faces of the outer cube)
Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Line Loop(2) = {5, 6, 7, 8};
Plane Surface(2) = {2};

Line Loop(3) = {9, 5, -10, -1};
Plane Surface(3) = {3};

Line Loop(4) = {10, 6, -11, -2};
Plane Surface(4) = {4};

Line Loop(5) = {11, 7, -12, -3};
Plane Surface(5) = {5};

Line Loop(6) = {12, 8, -9, -4};
Plane Surface(6) = {6};

// Define the surfaces (faces of the inner cube)
Line Loop(7) = {13, 14, 15, 16};
Plane Surface(7) = {7};

Line Loop(8) = {17, 18, 19, 20};
Plane Surface(8) = {8};

Line Loop(9) = {21, 17, -22, -13};
Plane Surface(9) = {9};

Line Loop(10) = {22, 18, -23, -14};
Plane Surface(10) = {10};

Line Loop(11) = {23, 19, -24, -15};
Plane Surface(11) = {11};

Line Loop(12) = {24, 20, -21, -16};
Plane Surface(12) = {12};

// Define the volumes
Surface Loop(1) = {1, 2, 3, 4, 5, 6};
Surface Loop(2) = {7, 8, 9, 10, 11, 12};
Volume(1) = {1, 2}; // Outer volume with inner hole
Volume(2) = {2};    // Inner volume

// Assign physical groups
Physical Volume(1) = {1}; // Outer volume
Physical Volume(2) = {2}; // Inner volume

// Outer cube surfaces
Physical Surface(1) = {1}; // Outer bottom
Physical Surface(2) = {2}; // Outer top
Physical Surface(3) = {3}; // Outer front
Physical Surface(4) = {4}; // Outer right
Physical Surface(5) = {5}; // Outer back
Physical Surface(6) = {6}; // Outer left

// Inner cube surfaces
Physical Surface(7) = {7};  // Inner bottom (-xy)
Physical Surface(8) = {8};  // Inner top (+xy)
Physical Surface(9) = {9};  // Inner front (-xz)
Physical Surface(10) = {10}; // Inner right (+yz)
Physical Surface(11) = {11}; // Inner back (+xz)
Physical Surface(12) = {12}; // Inner left (-yz)

// Mesh control
Mesh.OptimizeNetgen = 1;
Mesh.Optimize = 1;
Mesh.ElementOrder = 1;
