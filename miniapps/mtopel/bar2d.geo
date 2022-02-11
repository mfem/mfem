
// mesh size
lc=0.05;
r=0.25;
l=3;
d=1.5;

Point(1) = {0, 0, 0, lc};
Point(2) = {l, 0, 0, lc};
Point(3) = {l, 1, 0, lc};
Point(4) = {0, 1, 0, lc};


Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

Line Loop(1) = {1,2,3,4};

Plane Surface(1) = {1};

Physical Curve(1) = {1,3};
Physical Curve(2) = {4}; //inlet
Physical Curve(3) = {2}; //outlet
Physical Surface(1) = {1};


// Generate 2D mesh
Mesh 2;
SetOrder 1;
Mesh.MshFileVersion = 2.2;
