SetFactory("OpenCASCADE");
Mesh.ElementOrder = 2;
Mesh.SecondOrderLinear = 0;

lc = 1.0;

// channel height
H = 1.0;
// step height
h = H / 2.0;
// slab thickness
b = 4.0 * h;
// channel length
L = 30.0 * H;

Point(1) = {0, 0, 0, lc};
Point(2) = {L, 0, 0, lc};
Point(3) = {L, b, 0, lc};
Point(4) = {L, b+h, 0, lc};
Point(5) = {L, b+H, 0, lc};
Point(6) = {0, b+H, 0, lc};
Point(7) = {0, b+h, 0, lc};
Point(8) = {0, b, 0, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 1};
Line(9) = {8, 3};
Line(10) = {7, 4};

Curve Loop(1) = {10, 4, 5, 6};
Plane Surface(1) = {1};
Curve Loop(2) = {9, 3, -10, 7};
Plane Surface(2) = {2};
Curve Loop(3) = {9, -2, -1, -8};
Plane Surface(3) = {3};

Physical Surface("fluid", 1) = {1, 2};
Physical Surface("solid", 2) = {3};
Physical Curve("inlet", 1) = {6};
Physical Curve("outlet", 2) = {3, 4};
Physical Curve("interface", 3) = {9};
Physical Curve("wall fluid", 4) = {7, 5};
Physical Curve("wall side", 5) = {8, 2};
Physical Curve("wall bottom", 6) = {1};

Transfinite Surface {1:3};
Recombine Surface {1:3};

Transfinite Curve {1, -5, 10, 9} = 8 Using Progression 1;
Transfinite Curve {6, 4, 7, 3} = 1 Using Progression 1;
Transfinite Curve {8, 2} = 2 Using Progression 1;
