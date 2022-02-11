lc = 0.3;
l = 3;

Point(1) = {0.0,0.0,0.0,lc};
Point(2) = {1.0,0.0,0.0,lc};
Point(3) = {1.0,1.0,0.0,lc};
Point(4) = {0.0,1.0,0.0,lc};
Line(1) = {4,3};
Line(2) = {3,2};
Line(3) = {2,1};
Line(4) = {1,4};
Line Loop(5) = {2,3,4,1};
Plane Surface(6) = {5};
tmp[] = Extrude {0,0.0,l} {
  Surface{6};
};
Physical Volume(1) = tmp[1];
Physical Surface(1) = {15,19,23,27};
Physical Surface(2) = {6};
Physical Surface(3) = {28};

// Generate 2D mesh
Mesh 3;
SetOrder 1;
Mesh.MshFileVersion = 2.2;


