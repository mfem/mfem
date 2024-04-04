SetFactory("OpenCASCADE");

order = 1;

R = 1;
r = 0.2;

Point(1) = {0,0,0};

Point(2) = {r/Sqrt(2),r/Sqrt(2),0};
Point(3) = {-r/Sqrt(2),r/Sqrt(2),0};
Point(4) = {-r/Sqrt(2),-r/Sqrt(2),0};
Point(5) = {r/Sqrt(2),-r/Sqrt(2),0};

Point(6) = {R,0,0};
Point(7) = {R/Sqrt(2),R/Sqrt(2),0};
Point(8) = {0,R,0};
Point(9) = {-R/Sqrt(2),R/Sqrt(2),0};
Point(10) = {-R,0,0};
Point(11) = {-R/Sqrt(2),-R/Sqrt(2),0};
Point(12) = {0,-R,0};
Point(13) = {R/Sqrt(2),-R/Sqrt(2),0};

Line(1) = {1,2};
Line(2) = {1,3};
Line(3) = {1,4};
Line(4) = {1,5};

Line(5) = {1,6};
Line(6) = {1,8};
Line(7) = {1,10};
Line(8) = {1,12};

Line(9) = {2,6};
Line(10) = {2,8};
Line(11) = {3,8};
Line(12) = {3,10};
Line(13) = {4,10};
Line(14) = {4,12};
Line(15) = {5,12};
Line(16) = {5,6};

Line(17) = {6,7};
Line(18) = {7,8};
Line(19) = {8,9};
Line(20) = {9,10};
Line(21) = {10,11};
Line(22) = {11,12};
Line(23) = {12,13};
Line(24) = {13,6};

Transfinite Curve{1:24} = 2;

Physical Curve("ENE") = {17};
Physical Curve("NNE") = {18};
Physical Curve("NNW") = {19};
Physical Curve("WNW") = {20};
Physical Curve("WSW") = {21};
Physical Curve("SSW") = {22};
Physical Curve("SSE") = {23};
Physical Curve("ESE") = {24};

Curve Loop(1) = {9,17,18,-10};
Curve Loop(2) = {11,19,20,-12};
Curve Loop(3) = {13,21,22,-14};
Curve Loop(4) = {15,23,24,-16};

Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
Plane Surface(4) = {4};

Transfinite Surface{1} = {2,6,7,8};
Transfinite Surface{2} = {3,8,9,10};
Transfinite Surface{3} = {4,10,11,12};
Transfinite Surface{4} = {5,12,13,6};
Recombine Surface{1:4};

Physical Surface("Base") = {1,2,3,4};

Curve Loop(5) = {1,10,-6};
Plane Surface(5) = {5};
Physical Surface("N Even") = {5};

Curve Loop(6) = {6,-11,-2};
Plane Surface(6) = {6};
Physical Surface("N Odd") = {6};

Curve Loop(7) = {2,12,-7};
Plane Surface(7) = {7};
Physical Surface("W Even") = {7};

Curve Loop(8) = {7,-13,-3};
Plane Surface(8) = {8};
Physical Surface("W Odd") = {8};

Curve Loop(9) = {3,14,-8};
Plane Surface(9) = {9};
Physical Surface("S Even") = {9};

Curve Loop(10) = {8,-15,-4};
Plane Surface(10) = {10};
Physical Surface("S Odd") = {10};

Curve Loop(11) = {4,16,-5};
Plane Surface(11) = {11};
Physical Surface("E Even") = {11};

Curve Loop(12) = {5,-9,-1};
Plane Surface(12) = {12};
Physical Surface("E Odd") = {12};

// Generate 2D mesh
Mesh 2;
SetOrder order;
Mesh.MshFileVersion = 2.2;

Save "compass.msh";
