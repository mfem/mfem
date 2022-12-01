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
