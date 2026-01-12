SetFactory("OpenCASCADE");

DefineConstant[
  Nx = {2, Min 2, Max 10, Step 2,
    Name "Parameters/Element in x"}
  Ny = {2, Min 2, Max 10, Step 2,
    Name "Parameters/Element in y"}
  extrude_length = {0.001, Min .001, Max 1, Step .002, Name "Parameters/extrusion length"}
  extrude_layers = {2, Min 2, Max 10, Step 1, Name "Parameters/extrusion layers"}
];

Point(1) = {0,0,0};
Point(2) = {0.0005,0,0};
Point(3) = {0.001,0,0};
Point(4) = {0.001,0.0005,0};
Point(5) = {0.001,0.001,0};
Point(6) = {0.0005,0.001,0};
Point(7) = {0,0.001,0};
Point(8) = {0,0.0005,0};
Point(9) = {0.0005,0.0005,0};
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,6};
Line(6) = {6,7};
Line(7) = {7,8};
Line(8) = {8,1};
Line(9) = {2,9};
Line(10) = {4,9};
Line(11) = {6,9};
Line(12) = {8,9};

Curve Loop(21) = {1,9,-12,8};
Curve Loop(22) = {3,10,-9,2};
Curve Loop(23) = {5,11,-10,4};
Curve Loop(24) = {7,12,-11,6};

Plane Surface(31) = {21};
Plane Surface(32) = {22};
Plane Surface(33) = {23};
Plane Surface(34) = {24};
Recombine Surface {31:34};

// All the straight edges should have Nx, Ny elements
Transfinite Curve {1} = Nx+1;
Transfinite Curve {2} = Nx+1;
Transfinite Curve {10} = Nx+1;
Transfinite Curve {12} = Nx+1;
Transfinite Curve {5} = Nx+1;
Transfinite Curve {6} = Nx+1;
Transfinite Curve {3} = Ny+1;
Transfinite Curve {4} = Ny+1;
Transfinite Curve {9} = Ny+1;
Transfinite Curve {11} = Ny+1;
Transfinite Curve {7} = Ny+1;
Transfinite Curve {8} = Ny+1;
Transfinite Surface {31:34};

Mesh.Algorithm = 8;
Mesh.RecombinationAlgorithm = 3;
Mesh 3;
Mesh.MshFileVersion = 2.2;

Extrude {0, 0, extrude_length} { Surface{31:34}; Layers{extrude_layers}; Recombine; }

// start is x-y plane, z=0
Physical Surface("start", 1) = {31:34};
// left is y-z plane, x=0
Physical Surface("left", 2) = {38,48};
// front is x-z plane, y=0
Physical Surface("front", 3) = {35,42};
// top is x-y plane, z=extrude_length
Physical Surface("top-punch", 4) = {39};
Physical Surface("top-free", 5) = {43,47,50};

Physical Volume("block", 1) = {1:4};

