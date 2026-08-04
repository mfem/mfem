SetFactory("OpenCASCADE");

// Set the geometry order (1, 2, ..., 9)
order = 2;

// Target element size
h = 0.80;

// Hemisphere radius
R = 1.0;

// Lower half of a sphere centred at (0,0,R): dome bottom at z=0, flat face at z=R.
Sphere(1) = {0, 0, R, R};
Box(2) = {-R, -R, 0, 2*R, 2*R, R};
BooleanIntersection(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };

// Uniform sizing everywhere
Mesh.MeshSizeMin = h;
Mesh.MeshSizeMax = h;

// Tag surfaces and volumes with positive integers
// 1 - spherical dome (bottom at z=0), 2 - flat face (z = R, on top)
eps = 1e-6 * R;
flat() = Surface In BoundingBox {-2*R,-2*R, R-eps, 2*R,2*R, R+eps};
all()  = Surface{:};
dome() = {};
For i In {0 : #all()-1}
   onflat = 0;
   For j In {0 : #flat()-1}
      If (all(i) == flat(j))
         onflat = 1;
      EndIf
   EndFor
   If (onflat == 0)
      dome() += all(i);
   EndIf
EndFor
Physical Surface(1) = {dome()};
Physical Surface(2) = {flat()};
Physical Volume(1)  = {3};

// Generate 3D mesh (tetrahedra)
Mesh 3;
SetOrder order;
Mesh.MshFileVersion = 2.2;

Save Sprintf("hemisphere-unstructured-o%01g.msh", order);
