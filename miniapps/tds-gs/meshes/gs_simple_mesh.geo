// Gmsh project created on Thu Jun 30 09:20:00 2022
SetFactory("OpenCASCADE");
//+
Rectangle(1) = {1.75, -.25, 0, .5, .5, 0};
//+
Point(5) = {1, -1, 0, 1.0};
//+
Point(6) = {1, 1, 0, 1.0};
//+
Point(7) = {3, 1, 0, 1.0};
//+
Point(8) = {3, -1, 0, 1.0};
//+
Line(5) = {5, 6};
//+
Line(6) = {6, 7};
//+
Line(7) = {7, 8};
//+
Line(8) = {8, 5};
//+
Curve Loop(2) = {5, 6, 7, 8};
//+
Curve Loop(3) = {4, 1, 2, 3};
//+
Plane Surface(2) = {2, 3};
//+
Physical Curve("boundary", 9) = {5, 6, 7, 8};
//+
Physical Surface("interior", 10) = {1};
//+
Physical Surface("exterior", 2000) = {2};
