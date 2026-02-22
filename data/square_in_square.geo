// Square-in-square 2D geometry for MFEM
// Creates concentric squares with different material attributes

// Define the square sizes
L_outer = 2.0;
L_inner = 0.5;

// Set mesh size and algorithm
mesh_size = 1.0;
Mesh.Algorithm = 6; // Frontal-Delaunay for 2D triangular mesh
Mesh.CharacteristicLengthFactor = 1.0;
Mesh.MshFileVersion = 2.2;

// Define center point for concentric squares
cx = 0.0;
cy = 0.0;

// Define the points (vertices of the outer square)
Point(1) = {cx-L_outer/2, cy-L_outer/2, 0, mesh_size}; // bottom-left outer
Point(2) = {cx+L_outer/2, cy-L_outer/2, 0, mesh_size}; // bottom-right outer
Point(3) = {cx+L_outer/2, cy+L_outer/2, 0, mesh_size}; // top-right outer
Point(4) = {cx-L_outer/2, cy+L_outer/2, 0, mesh_size}; // top-left outer

// Define the points (vertices of the inner square)
Point(5) = {cx-L_inner/2, cy-L_inner/2, 0, mesh_size}; // bottom-left inner
Point(6) = {cx+L_inner/2, cy-L_inner/2, 0, mesh_size}; // bottom-right inner
Point(7) = {cx+L_inner/2, cy+L_inner/2, 0, mesh_size}; // top-right inner
Point(8) = {cx-L_inner/2, cy+L_inner/2, 0, mesh_size}; // top-left inner

// Define the lines (edges of the outer square)
Line(1) = {1, 2}; // bottom edge
Line(2) = {2, 3}; // right edge
Line(3) = {3, 4}; // top edge
Line(4) = {4, 1}; // left edge

// Define the lines (edges of the inner square)
Line(5) = {5, 6}; // bottom edge
Line(6) = {6, 7}; // right edge
Line(7) = {7, 8}; // top edge
Line(8) = {8, 5}; // left edge

// Define the surfaces
// Outer square boundary
Line Loop(1) = {1, 2, 3, 4};

// Inner square boundary (hole in the outer region)
Line Loop(2) = {5, 6, 7, 8};

// Define the surface areas
// Outer region (annular region between squares)
Plane Surface(1) = {1, 2}; // Outer loop minus inner loop (creates hole)

// Inner region (solid inner square)
Plane Surface(2) = {2}; // Inner loop only

// Assign physical groups for materials
Physical Surface(1) = {1}; // Outer material (annular region)
Physical Surface(2) = {2}; // Inner material (solid square)

// Physical lines for boundary conditions
// Outer square boundary edges
Physical Line(1) = {1}; // outer bottom
Physical Line(2) = {2}; // outer right  
Physical Line(3) = {3}; // outer top
Physical Line(4) = {4}; // outer left

// Inner square boundary edges
Physical Line(5) = {5}; // inner bottom
Physical Line(6) = {6}; // inner right
Physical Line(7) = {7}; // inner top
Physical Line(8) = {8}; // inner left

// Mesh control for quality
Mesh.OptimizeNetgen = 1;
Mesh.Optimize = 1;
Mesh.ElementOrder = 1;
Mesh.RecombineAll = 0; // Keep triangular elements (don't recombine to quads)
