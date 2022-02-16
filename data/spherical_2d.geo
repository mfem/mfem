// SetFactory("OpenCASCADE");

//////////////////////////////////////////
//// Set the physical size of the mesh

square_distort = 0.5; // Sets the amount the square at the center bows out
square = 50;
r1 = 100; // Inner ring around the square - should be ~ 1.5-2x larger than the square
r2 = 800;

//////////////////////////////////
//// Set the mesh geometry

order = 4;
nr1 = 3; // Number of radial zones in the inner ring
nr2 = 10; // Number of radial zones in the outer ring
ntheta = 10; // Number of angular zones. The square will have ntheta**2 zones


Point(1) = {0, 0, 0, 1.0};

Point(2) = {0, square*square_distort, 0, 1.0};
Point(3) = {square*square_distort, 0, 0, 1.0};
Point(4) = {0, -square*square_distort, 0, 1.0};
Point(5) = {-square*square_distort, 0, 0, 1.0};

Point(6) = {square, square, 0, 1.0};
Point(7) = {square, -square, 0, 1.0};
Point(8) = {-square, -square, 0, 1.0};
Point(9) = {-square, square, 0, 1.0};

Circle(1) = {6, 5, 7};
Circle(2) = {7, 2, 8};
Circle(3) = {8, 3, 9};
Circle(4) = {9, 4, 6};


Curve Loop(1) = {1,2,3,4};
Plane Surface(1) = {1};
Transfinite Surface {1} = {6, 7, 8, 9};
Transfinite Curve {1,2,3,4} = ntheta Using Progression 1;

Point(10) = {r1, r1, 0, 1.0};
Point(11) = {r1, -r1, 0, 1.0};
Point(12) = {-r1, -r1, 0, 1.0};
Point(13) = {-r1, r1, 0, 1.0};

Line(5) = { 6, 10 };
Line(6) = { 7, 11 };
Line(7) = { 8, 12 };
Line(8) = { 9, 13 };
Circle(9) = { 10, 1, 11 };
Circle(10) = { 11, 1, 12 };
Circle(11) = { 12, 1, 13 };
Circle(12) = { 13, 1, 10 };

Curve Loop(2) = { 1 , 6, -9, -5};
Plane Surface(2) = { 2};
Transfinite Surface{ 2 } = { 6, 10, 11, 7};
Curve Loop(3) = { 2 , 7, -10, -6};
Plane Surface(3) = { 3};
Transfinite Surface{ 3 } = { 7, 11, 12, 8};
Curve Loop(4) = { 3 , 8, -11, -7};
Plane Surface(4) = { 4};
Transfinite Surface{ 4 } = { 8, 12, 13, 9};
Curve Loop(5) = { 4 , 5, -12, -8};
Plane Surface(5) = { 5};
Transfinite Surface{ 5 } = { 9, 13, 10, 6};



Point(14) = {r2, r2, 0, 1.0};
Point(15) = {r2, -r2, 0, 1.0};
Point(16) = {-r2, -r2, 0, 1.0};
Point(17) = {-r2, r2, 0, 1.0};

Line(13) = { 10, 14 };
Line(14) = { 11, 15 };
Line(15) = { 12, 16 };
Line(16) = { 13, 17 };
Circle(17) = { 14, 1, 15 };
Circle(18) = { 15, 1, 16 };
Circle(19) = { 16, 1, 17 };
Circle(20) = { 17, 1, 14 };

Curve Loop(6) = { 9 , 14, -17, -13};
Plane Surface(6) = { 6};
Transfinite Surface{ 6 } = { 10, 14, 15, 11};
Curve Loop(7) = { 10 , 15, -18, -14};
Plane Surface(7) = { 7};
Transfinite Surface{ 7 } = { 11, 15, 16, 12};
Curve Loop(8) = { 11 , 16, -19, -15};
Plane Surface(8) = { 8};
Transfinite Surface{ 8 } = { 12, 16, 17, 13};
Curve Loop(9) = { 12 , 13, -20, -16};
Plane Surface(9) = { 9};
Transfinite Surface{ 9 } = { 13, 17, 14, 10};

Transfinite Curve {  5 , 6 , 7 , 8   } = nr1 Using Progression 1;
Transfinite Curve { 9 , 10 , 11 , 12  } = ntheta Using Progression 1;
Transfinite Curve {  13 , 14 , 15 , 16   } = nr2 Using Progression 1;
Transfinite Curve { 17 , 18 , 19 , 20  } = ntheta Using Progression 1;

Physical Curve(1) = {17,18,19,20}; 
Physical Surface (1) = {1, 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9};

Mesh.RecombineAll = 1;
Mesh.ElementOrder = order;
Mesh.HighOrderOptimize = 1;
Mesh 2;
SetOrder order;
Mesh.MshFileVersion = 2.2;

Save Sprintf("spherical_2d.msh");
