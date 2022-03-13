l = 3;
n = 100;
lc = l/n;

Point(1) = {0.0,0.0,0.0,lc};
Point(2) = {1.0,0.0,0.0,lc};
Point(3) = {1.0,1.0,0.0,lc};
Point(4) = {0.0,1.0,0.0,lc};
Line(1) = {4,3};
Line(2) = {3,2};
Line(3) = {2,1};
Line(4) = {1,4};

Transfinite Curve{1,2,3,4} = n;
Line Loop(5) = {2,3,4,1};
Plane Surface(6) = {5};
Transfinite Surface{6};
Recombine Surface{6};
tmp[] = Extrude {0,0.0,l} {
  Surface{6};  Layers{n}; Recombine; 
};

//Periodic Surface {28} = {6} Translate{0, 0, 3};
//Periodic Surface {23} = {15} Translate{-1, 0, 0};
//Periodic Surface {19} = {27} Translate{0,  -1, 0};


Physical Volume(1) = tmp[1];
Physical Surface(1) = {15,19,23,27};
Physical Surface(2) = {6};
Physical Surface(3) = {28};

// Generate 2D mesh
Mesh 3;
SetOrder 1;
Mesh.MshFileVersion = 2.2;
// Mesh.RecombineAll = 1;
// Mesh.RecombinationAlgorithm = 1;
// RecombineMesh;

Periodic Surface {28} = {6} Translate{0, 0, 1};
Periodic Surface {23} = {15} Translate{-1, 0, 0};
Periodic Surface {19} = {27} Translate{0,  -1, 0};
//Save "period.msh";




