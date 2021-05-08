L=1.0;
lc=L/20;

Point(1) = {0, 0, 0, lc};
Point(2) = {L*0.45,0,0,lc};
Point(3) = {L*0.55,0,0,lc};
Point(4) = {L,0,0,lc};
Point(5) = {L,L,0,lc};
Point(6) = {0,L,0,lc};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,6};
Line(6) = {6,1};

Line Loop(1) = {1,2,3,4,5,6};
Plane Surface(1) = {1};

Physical Curve(1) = {1,2,3,4,5,6};
Physical Surface(1) = {1};

// Generate 2D mesh
Mesh 2;
SetOrder 1;
Mesh.MshFileVersion = 2.2;



