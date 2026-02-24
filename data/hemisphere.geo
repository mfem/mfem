SetFactory("OpenCASCADE");

// Parameters
r = 0.4;
cx = 0.0; cy = 0.0; cz = 0.4;
res = 0.09;
lc_min = res;
lc_max = 2 * res;

Mesh.MshFileVersion = 2.2;
Mesh.ElementOrder = 2;
Mesh.Algorithm3D = 4;

// Equivalent to: gmsh.model.occ.addPoint(center[0], center[1], center[2] - r)
Point(1) = {cx, cy, cz - r, lc_min};

// Equivalent to: gmsh.model.occ.addSphere(..., angle1=-pi/2, angle2=0)
Sphere(1) = {cx, cy, cz, r, -Pi/2, 0, 2*Pi};

// Surface 1, 3: dome  -> sphere surface
// Surface 2: z = 0.4  -> flat face
Physical Surface("sphere_surface", 1) = {1, 3};   // sphere_surface tag = 1
Physical Surface("flat_surface", 2)   = {2};      // flat_surface tag = 2
Physical Volume("volume", 1)          = {1};

// Equivalent to the Distance + Threshold fields
Field[1] = Distance;
Field[1].NodesList = {1};

Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = lc_min;
Field[2].LcMax = lc_max;
Field[2].DistMin = 0.5 * r;
Field[2].DistMax = r;
Background Field = 2;
