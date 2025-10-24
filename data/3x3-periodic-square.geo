// Parameters
L = 1.0;       // Length of the square side
nx = 3;        // Number of divisions along x
ny = 3;        // Number of divisions along y

// Create corner points
Point(1) = {-L, -L, 0};      // Bottom-left
Point(2) = {L, -L, 0};      // Bottom-right
Point(3) = {L, L, 0};      // Top-right
Point(4) = {-L, L, 0};      // Top-left

// Create lines
Line(1) = {1, 2}; // bottom
Line(2) = {2, 3}; // right
Line(3) = {3, 4}; // top
Line(4) = {4, 1}; // left

// Line loop and surface
Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Transfinite and recombine
Transfinite Line{1, 3} = nx + 1; // x-direction
Transfinite Line{2, 4} = ny + 1; // y-direction
Transfinite Surface{1} = {1, 2, 3, 4};
Recombine Surface{1};

// Mesh settings
Mesh.RecombineAll = 1;
Mesh.Algorithm = 1;

// Periodicity
// x-direction (left ↔ right)
Periodic Line{2} = {4} Translate {2*L, 0, 0};
// y-direction (bottom ↔ top)
Periodic Line{3} = {1} Translate {0, 2*L, 0};
