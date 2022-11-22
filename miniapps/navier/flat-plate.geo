// Gmsh project created on Tue Nov 08 10:43:03 2022
SetFactory("OpenCASCADE");

f = 1e-4;
scale = 20;

//+ Define Points
Point(1) = {0, 0, 0, f};
Point(2) = {0.11, 0, 0, f};
Point(3) = {0.11, 0.05, 0, f*scale};
Point(4) = {0, 0.05, 0, f*scale};

//+ Define Boundary Lines
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

//+ Define Boundary
Curve Loop(1) = {1, 2, 3, 4};

//+ Define Surface
Plane Surface(1) = {1};

//+ Define Curves for BC's
Physical Curve(1) = {1};
Physical Curve(2) = {2};
Physical Curve(3) = {3};
Physical Curve(4) = {4};

//+ Define Surface So Elements Will Be Generated
Physical Surface(1) = {1,2,3,4};

// https://mfem.org/mesh-formats/
//+
Mesh.MeshSizeMin = f;
Mesh.MeshSizeMax = f*scale;
Recombine Surface{1};
Mesh.MshFileVersion = 2.2;
