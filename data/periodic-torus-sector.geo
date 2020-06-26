SetFactory("OpenCASCADE");

// Select periodic mesh by setting this to either 0 - standard, 1 - periodic
periodic = 1;

// Set the geometry order (1, 2, or 3)
order = 2;

R = 1.5;
r = 0.5;

Phi = Pi/3.0;

Torus(1) = {0,0,0, R, r, Phi};

pts() = PointsOf{ Volume{1}; };

Characteristic Length{ pts() } = 0.25;

// Set a rotation periodicity constraint:
If (periodic)
   Periodic Surface{3} = {2} Rotate{{0,0,1}, {0,0,0}, Phi};
EndIf

// Tag surfaces and volumes with positive integers
Physical Surface(1) = {1};
Physical Surface(2) = {2};
Physical Surface(3) = {3};
Physical Volume(1) = {1};

// Generate 3D mesh
Mesh 3;
SetOrder order;
Mesh.MshFileVersion = 2.2;

If (periodic)
   Save Sprintf("periodic-torus-sector-o%01g.msh", order);
Else
   Save Sprintf("torus-sector-o%01g.msh", order);
EndIf
