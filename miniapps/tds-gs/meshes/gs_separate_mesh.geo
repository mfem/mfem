// Gmsh project created on Thu Jun 30 12:06:47 2022
SetFactory("OpenCASCADE");
//+
Rectangle(1) = {-1, -1, 0, .5, 2, 0};
//+
Rectangle(2) = {.5, -1, 0, .5, 2, 0};
//+
Physical Curve("boundary", 9) = {4, 3, 2, 1, 8, 7, 6, 5};
//+
Physical Surface("left", 10) = {1};
//+
Physical Surface("right", 11) = {2};
