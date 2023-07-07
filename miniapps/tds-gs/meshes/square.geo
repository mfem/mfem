// Gmsh project created on Wed Sep 21 10:20:59 2022
SetFactory("OpenCASCADE");
//+
Rectangle(1) = {0, 0, 0, 1, 1, 0};
//+
Physical Curve("boundary", 5) = {4, 3, 2, 1};
//+
Physical Surface("interior", 6) = {1};
//+
Physical Curve("top", 7) = {3};
//+
Circle(5) = {.5, .5, 0, .3, 0, 2*Pi};
//+
Physical Curve("box", 8) = {4, 3, 2, 1};
//+
Curve Loop(2) = {4, 1, 2, 3};
//+
Curve Loop(3) = {5};
//+
Plane Surface(2) = {2, 3};
