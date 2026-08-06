SetFactory("OpenCASCADE");

// Meshing algorithm (6 - Frontal-Delaunay) and quadratic (second-order) elements
Mesh.Algorithm = 6;
Mesh.ElementOrder = 2;

// Uniform target element size across the mesh
Mesh.CharacteristicLengthMin = 0.2;
Mesh.CharacteristicLengthMax = 0.2;

// Outer and inner cylinders sharing the same axis and end faces
Cylinder(1) = {0, 0.5, 0.5, 0.2, 0, 0, 0.5, 2*Pi};
Cylinder(2) = {0, 0.5, 0.5, 0.2, 0, 0, 0.2, 2*Pi};

// Subtract the inner cylinder from the outer one to leave a hollow wheel
BooleanDifference(50) = { Volume{1}; Delete; }{ Volume{2}; Delete; };

// Tag surfaces and volumes with positive integers
// 1 - outer surface, 2 - front face, 3 - back face, 4 - inner surface
Physical Volume(1) = {50};
Physical Surface(1) = {1};
Physical Surface(2) = {2};
Physical Surface(3) = {3};
Physical Surface(4) = {4};

Mesh.MshFileVersion = 2.2;

// Generate 3D mesh
Mesh 3;

Save Sprintf("wheel.msh");
