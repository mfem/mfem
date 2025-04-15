SetFactory("OpenCASCADE");
  
// mesh size
lc=0.1;
L=1;

Point(1) = {0.0, 0.0, 0.0, lc};
Point(2) = {6*L, 0.0, 0.0, lc};
Point(3) = {6*L, 1*L, 0.0, lc};
Point(4) = {0*L, 1*L, 0.0, lc};

Line(10) = {1,2};
Line(11) = {2,3};
Line(12) = {3,4};
Line(13) = {4,1};

Curve Loop(1) = {10,11,12,13};
Plane Surface(1) = {1};

Circle(20) = {2*L, 0.5*L, 0, 0.1*L, 0, 2*Pi};

For i In {20:20}
   Curve Loop(i) = {i};
   Plane Surface(i) = {i};
EndFor

BooleanDifference{ Surface{1}; Delete; }{Surface{20};  Delete;}

Recombine Surface {1};

Physical Surface(1) = {1};
Physical Curve(1) = {1,4};
Physical Curve(2) = {2};
Physical Curve(3) = {3};
Physical Curve(4) = {5};

// Physical Curve(3) = {5};
// Physical Curve(4) = {6};
// Physical Curve(5) = {7};
// Physical Curve(6) = {9};
// Physical Curve(7) = {10};
// Physical Curve(8) = {11};
// Physical Curve(9) = {12};



// Generate 2D mesh
Mesh 2;
SetOrder 2;
Mesh.MshFileVersion = 2.2;
Mesh.ElementOrder = 2;
Mesh.HighOrderOptimize = 2;


