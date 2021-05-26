Mesh.Algorithm = 6;

lc = 0.1;
Point(1) = {0.0,0.0,0.0,lc};
Point(2) = {1,0.0,0.0,lc};
Point(3) = {0,1,0.0,lc};
Circle(1) = {2,1,3};
Point(4) = {-1,0,0.0,lc};
Point(5) = {0,-1,0.0,lc};
Circle(2) = {3,1,4};
Circle(3) = {4,1,5};
Circle(4) = {5,1,2};
Point(6) = {0,0,-1,lc};
Point(7) = {0,0,1,lc};
Circle(5) = {3,1,6};
Circle(6) = {6,1,5};
Circle(7) = {5,1,7};
Circle(8) = {7,1,3};
Circle(9) = {2,1,7};
Circle(10) = {7,1,4};
Circle(11) = {4,1,6};
Circle(12) = {6,1,2};
Curve Loop(13) = {2,8,-10};
Surface(14) = {13};
Curve Loop(15) = {10,3,7};
Surface(16) = {15};
Curve Loop(17) = {-8,-9,1};
Surface(18) = {17};
Curve Loop(19) = {-11,-2,5};
Surface(20) = {19};
Curve Loop(21) = {-5,-12,-1};
Surface(22) = {21};
Curve Loop(23) = {-3,11,6};
Surface(24) = {23};
Curve Loop(25) = {-7,4,9};
Surface(26) = {25};
Curve Loop(27) = {-4,12,-6};
Surface(28) = {27};
Surface Loop(29) = {28,26,16,14,20,24,22,18};
Volume(30) = {29};

Physical Surface(1) = {28,26,16,14,20,24,22,18};
Physical Volume(2) = 30;

// Generate 2D mesh
Mesh 2;
Mesh.MshFileVersion = 2.2;