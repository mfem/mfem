SetFactory("OpenCASCADE");

Mesh.Algorithm = 6;
Mesh.ElementOrder = 2;
Mesh.CharacteristicLengthMin = 0.1;
Mesh.CharacteristicLengthMax = 0.1;

Cylinder(1) = {0, 0.5, 0.5, 0.2, 0, 0, 0.5, 2*Pi};
Cylinder(2) = {0, 0.5, 0.5, 0.2, 0, 0, 0.2, 2*Pi};

BooleanDifference(50) = { Volume{1}; Delete; }{ Volume{2}; Delete; };

Physical Volume(1) = {50};
Physical Surface(1) = {1}; // out
Physical Surface(2) = {2}; // front 
Physical Surface(3) = {3}; // back 
Physical Surface(4) = {4}; // in

Mesh.MshFileVersion = 2.2;

Mesh 3;
