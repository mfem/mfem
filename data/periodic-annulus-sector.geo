SetFactory("OpenCASCADE");

// Select periodic mesh by setting this to either 0 - standard, 1 - periodic
periodic = 1;

// Set the geometry order (1, 2, ..., 9)
order = 3;

// Set the element type (3 - triangles, 4 - quadrilaterals)
type = 3;

// Number of radial elements
nrad = 2;

// Number of azimuthal elements on inner arc
nazm1 = 3;

// Number of azimuthal elements on outer arc
nazm2 = 5;

// Note: Using type = 4 with nazm1 != nazm2 can lead to mixed meshes
//       containing both triangles and quadrilaterals.

// Inner and outer radii
R1 = 1.0;
R2 = 2.0;

// Angular size of the sector
Phi = Pi/3.0;

Point(1) = {0.0, 0, 0, 1.0};
Point(2) = {R1, 0, 0, 1.0};
Point(3) = {R2, 0, 0, 1.0};
Point(4) = {R1*Cos(Phi), R1*Sin(Phi), 0, 1.0};
Point(5) = {R2*Cos(Phi), R2*Sin(Phi), 0, 1.0};
Line(1) = {2, 3};
Line(2) = {4, 5};
Circle(3) = {2, 1, 4};
Circle(4) = {3, 1, 5};
Curve Loop(5) = {1, 4, -2, -3};
Plane Surface(1) = {5};

Transfinite Curve{1} = nrad+1;
Transfinite Curve{2} = nrad+1;
Transfinite Curve{3} = nazm1+1;
Transfinite Curve{4} = nazm2+1;

If (nazm1 == nazm2)
   Transfinite Surface{1};
EndIf

If (type == 4)
   Recombine Surface {1};
EndIf

// Set a rotation periodicity constraint:
If (periodic)
   Periodic Line{1} = {2} Rotate{{0,0,1}, {0,0,0}, -Phi};
EndIf

// Tag surfaces and volumes with positive integers
Physical Curve(1) = {3};
Physical Curve(2) = {4};
Physical Curve(3) = {1};
Physical Curve(4) = {2};
Physical Surface(1) = {1};

// Optimize the high-order mesh
// See https://gmsh.info/doc/texinfo/gmsh.html#index-Mesh_002eHighOrderOptimize
// Mesh.ElementOrder = order;
// Mesh.HighOrderOptimize = 1;

// Generate 2D mesh
Mesh 2;
SetOrder order;
Mesh.MshFileVersion = 2.2;

// Check the element quality (the Plugin may be called AnalyseCurvedMesh)
// Plugin(AnalyseMeshQuality).JacobianDeterminant = 1;
// Plugin(AnalyseMeshQuality).Run;

If (periodic)
   Save Sprintf("periodic-annulus-sector-t%01g-o%01g.msh", type, order);
Else
   Save Sprintf("annulus-sector-t%01g-o%01g.msh", type, order);
EndIf
