SetFactory("OpenCASCADE");

R = 1.5;
r = 0.5;

Torus(1) = {0,0,0, R, r, Pi/3};

pts() = PointsOf{ Volume{1}; };

Characteristic Length{ pts() } = 0.25;

// Set a rotation periodicity constraint:
Periodic Surface{3} = {2} Rotate{{0,0,1}, {0,0,0}, Pi/3};

// Tag surfaces and volumes with positive integers
Physical Surface(1) = {1};
Physical Surface(2) = {2};
Physical Surface(3) = {3};
Physical Volume(1) = {1};

// Generate 3D mesh
Mesh 3;

Mesh.MshFileVersion = 2.2;
Save "periodic-torus-sector.msh";
