
// Select periodic mesh by setting this to either 0 - standard, 1 - periodic
periodic = 1;

// Set the geometry order (1, 2, or 3)
order = 3;

lc = 0.5;

Point(1) = {1, 0, 0, lc};
Point(2) = {0, 1, 0, lc};
Point(3) = {-1, 0, 0, lc};
Point(4) = {0, -1, 0, lc};
Point(5) = {0, 0, 0, lc};

Circle(1) = {1,5,2};
Circle(2) = {2,5,3};
Circle(3) = {3,5,4};
Circle(4) = {4,5,1};

Line Loop(100) = {1,2,3,4};  
Plane Surface(200) = {100}; 

Recombine Surface {200} ;


Extrude { {0,1,0} , {-1.5,0,1.0} , -Pi/4} {
  Surface{200}; Layers{7}; Recombine;
}


// Set a rotation periodicity constraint:
If (periodic)
	Periodic Surface{222} = {200} Rotate { {0,1,0} , {-1.5,0,1.0} , -Pi/4};
EndIf

// Tag surfaces and volumes with positive integers
Physical Surface(1) = {200};
Physical Surface(2) = {222};
Physical Surface(3) = {221,217,213,209};
Physical Volume(1) = {1};

// Generate 3D mesh
SetOrder order;
Mesh.MshFileVersion = 2.2;
Mesh 3;

If (periodic)
   Save Sprintf("periodic-sector-o%01g.msh", order);
Else
   Save Sprintf("sector-o%01g.msh", order);
EndIf



