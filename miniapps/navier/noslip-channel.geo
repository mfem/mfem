// Gmsh project created on Tue Nov 08 10:43:03 2022
SetFactory("OpenCASCADE");

f = 0.5e-2;
h = 0.25;
w = 1.1;
start_plate = 0.1;
coarsening = 1.0; 

//+ Define Points
Point(1) = {0, 0, 0, f};
Point(2) = {w, 0, 0, f};
Point(3) = {w, h, 0, f*coarsening};
Point(4) = {0, h, 0, f*coarsening};
//Point(2) = {start_plate, 0, 0, f};
//Point(3) = {w, 0, 0, f};
//Point(4) = {w, h, 0, f*coarsening};
//Point(5) = {0, h, 0, f*coarsening};

//+ Define Boundary Lines
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
//Line(4) = {4, 5};
//Line(5) = {5, 1};

//+ Define Boundary
Curve Loop(1) = {1, 2, 3, 4};
//Curve Loop(1) = {1, 2, 3, 4, 5};

//+ Define Surface
Plane Surface(1) = {1};

//+ Define Curves for BC's
Physical Curve(1) = {4};
Physical Curve(2) = {2};
Physical Curve(3) = {3};
Physical Curve(4) = {1};
//Physical Curve(1) = {5};
//Physical Curve(2) = {3};
//Physical Curve(3) = {4};
//Physical Curve(4) = {1};
//Physical Curve(5) = {2};

Physical Surface("interior") = {1};

// https://mfem.org/mesh-formats/
//+
Mesh.MeshSizeMin = f;
Mesh.MeshSizeMax = f*coarsening;
Recombine Surface(1);
Mesh.MshFileVersion = 2.2;
