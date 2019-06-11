SetFactory("OpenCASCADE");

lc = DefineNumber[ 0.1, Name "lc" ];

Point(1) = {0, 0, 0, lc};
Point(2) = {2.2, 0, 0, lc};
Point(3) = {2.2, 0.41, 0, lc};
Point(4) = {0, 0.41, 0, lc};

Circle(1) = {0.2, 0.2, 0, 0.05, 0, 2*Pi};

Line(2) = {1, 2};
Line(3) = {2, 3};
Line(4) = {3, 4};
Line(5) = {4, 1};

Line Loop(1) = {1};
Line Loop(2) = {4, 5, 2, 3};
Plane Surface(1) = {1, 2};

Physical Curve("inlet") = {5};
Physical Curve("cyl") = {1};
Physical Curve("wall") = {2, 4};
Physical Curve("outlet") = {3};

Physical Surface(5) = {1};

