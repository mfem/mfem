
// Select periodic mesh by setting this to either 0 - standard, 1 - periodic
periodic = 1;

// Set the geometry order (1, 2, ..., 10 for tetrahedra or 9 for other types)
order = 3;

// Set the element type (4 - tetrahedra, 6 - wedges, 8 - hexahedra)
type = 8;

// Minor and major radii
R1 = 1.0;
R2 = 2.0;

// Angular size of the sector
Phi = Pi/3.0;

// Number of azimuthal elements
nazm = 3;

// Number of elements around a quarter of the circle
narc = 2;

lc = 0.5;

Point(1) = {R2+R1, 0, 0, lc};
Point(2) = {R2, 0, R1, lc};
Point(3) = {R2-R1, 0, 0, lc};
Point(4) = {R2, 0, -R1, lc};
Point(5) = {R2, 0, 0, lc};

Circle(1) = {1,5,2};
Circle(2) = {2,5,3};
Circle(3) = {3,5,4};
Circle(4) = {4,5,1};

Line Loop(100) = {1,2,3,4};  
Plane Surface(200) = {100}; 

Transfinite Curve{1} = narc+1;
Transfinite Curve{2} = narc+1;
Transfinite Curve{3} = narc+1;
Transfinite Curve{4} = narc+1;

If (type == 8)
   Recombine Surface {200} ;
EndIf

If (type == 4)
   Extrude { {0,0,1} , {0,0,0} , Phi} {
      Surface{200}; Layers{nazm};
}
Else
   Extrude { {0,0,1} , {0,0,0} , Phi} {
      Surface{200}; Layers{nazm}; Recombine;
}
EndIf

// Set a rotation periodicity constraint:
If (periodic)
   Periodic Surface{222} = {200} Rotate { {0,0,1} , {0,0,0} , Phi};
EndIf

// Tag surfaces and volumes with positive integers
Physical Surface(1) = {200};
Physical Surface(2) = {222};
Physical Surface(3) = {221,217,213,209};
Physical Volume(1) = {1};

// Generate 3D mesh
Mesh 3;
SetOrder order;
Mesh.MshFileVersion = 2.2;

If (periodic)
   Save Sprintf("periodic-torus-sector-t%01g-o%01g.msh", type, order);
Else
   Save Sprintf("torus-sector-t%01g-o%01g.msh", type, order);
EndIf
