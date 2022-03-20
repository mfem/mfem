// Select periodic mesh by setting this to either 0 - standard, 1 - periodic
periodic = 1;

// Set the geometry order (1, 2, ..., 10 for tetrahedra or 9 for other types)
order = 3;

// Set the element type (4 - tetrahedra, 6 - wedges, 8 - hexahedra)
type = 8;

// Minor and major radii
R1 = 1.0;
R2 = 2.0;

// Side length of interior square
A1 = 0.8;

// Angular size of the sector
Phi = Pi/3.0;

// Number of azimuthal elements
nazm = 3;

// Number of elements around a quarter of the circle
narc = 2;

// Number of elements between surface and interior square
nshl = 1;

lc = 0.5;
a1 = A1 / Sqrt(2.0);

Point(1) = {R2+R1, 0, 0, lc};
Point(2) = {R2, 0, R1, lc};
Point(3) = {R2-R1, 0, 0, lc};
Point(4) = {R2, 0, -R1, lc};
Point(5) = {R2, 0, 0, lc};
Point(6) = {R2+a1, 0, 0, lc};
Point(7) = {R2, 0, a1, lc};
Point(8) = {R2-a1, 0, 0, lc};
Point(9) = {R2, 0, -a1, lc};

Circle(1) = {1,5,2};
Circle(2) = {2,5,3};
Circle(3) = {3,5,4};
Circle(4) = {4,5,1};

Line(5) = {6,1};
Line(6) = {7,2};
Line(7) = {8,3};
Line(8) = {9,4};

Line(9) = {6, 7};
Line(10) = {7, 8};
Line(11) = {8, 9};
Line(12) = {9, 6};

Line Loop(101) = {1, -6, -9, 5};
Line Loop(102) = {2, -7, -10, 6};
Line Loop(103) = {3, -8, -11, 7};
Line Loop(104) = {4, -5, -12, 8};
Line Loop(105) = {9, 10, 11, 12};

Plane Surface(201) = {101};
Plane Surface(202) = {102};
Plane Surface(203) = {103};
Plane Surface(204) = {104};
Plane Surface(205) = {105};

Transfinite Curve{1} = narc+1;
Transfinite Curve{2} = narc+1;
Transfinite Curve{3} = narc+1;
Transfinite Curve{4} = narc+1;

Transfinite Curve{5} = nshl+1;
Transfinite Curve{6} = nshl+1;
Transfinite Curve{7} = nshl+1;
Transfinite Curve{8} = nshl+1;

Transfinite Curve{9} = narc+1;
Transfinite Curve{10} = narc+1;
Transfinite Curve{11} = narc+1;
Transfinite Curve{12} = narc+1;

If (type == 8)
   Recombine Surface {201};
   Recombine Surface {202};
   Recombine Surface {203};
   Recombine Surface {204};
   Recombine Surface {205};

   Transfinite Surface {201} = {1,2,7,6};
   Transfinite Surface {202} = {2,3,8,7};
   Transfinite Surface {203} = {3,4,9,8};
   Transfinite Surface {204} = {4,1,6,9};
   Transfinite Surface {205} = {6,7,8,9};
EndIf

If (type == 4)
   Extrude { {0,0,1} , {0,0,0} , Phi} {
      Surface{201,202,203,204,205}; Layers{nazm};
}
Else
   Extrude { {0,0,1} , {0,0,0} , Phi} {
      Surface{201,202,203,204,205}; Layers{nazm}; Recombine;
}
EndIf

// Set a rotation periodicity constraint:
If (periodic)
   Periodic Surface{227} = {201} Rotate{{0,0,1}, {0,0,0}, Phi};
   Periodic Surface{249} = {202} Rotate{{0,0,1}, {0,0,0}, Phi};
   Periodic Surface{271} = {203} Rotate{{0,0,1}, {0,0,0}, Phi};
   Periodic Surface{293} = {204} Rotate{{0,0,1}, {0,0,0}, Phi};
   Periodic Surface{315} = {205} Rotate{{0,0,1}, {0,0,0}, Phi};
EndIf

// Tag surfaces and volumes with positive integers
Physical Surface(1) = {201,202,203,204,205};
Physical Surface(2) = {227,249,271,293,315};
Physical Surface(3) = {214,236,258,280};
Physical Volume(1) = {1,2,3,4,5};

// Optimize the high-order mesh
// See https://gmsh.info/doc/texinfo/gmsh.html#index-Mesh_002eHighOrderOptimize
// Mesh.ElementOrder = order;
// Mesh.HighOrderOptimize = 1;

// Generate 3D mesh
Mesh 3;
SetOrder order;
Mesh.MshFileVersion = 2.2;

// Check the element quality (the Plugin may be called AnalyseCurvedMesh)
// Plugin(AnalyseMeshQuality).JacobianDeterminant = 1;
// Plugin(AnalyseMeshQuality).Run;

If (periodic)
   Save Sprintf("periodic-torus-sector-t%01g-o%01g.msh", type, order);
Else
   Save Sprintf("torus-sector-t%01g-o%01g.msh", type, order);
EndIf
