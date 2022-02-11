//Inputs
boxdim = 1;
gridsize = boxdim/3;

//Create 2D square mesh.
Point(1) = {0,0,0,gridsize};
Point(2) = {boxdim,0,0,gridsize};
Point(3) = {boxdim,boxdim,0,gridsize};
Point(4) = {0,boxdim,0,gridsize};

Line(5) = {1,2};
Line(6) = {2,3};
Line(7) = {3,4};
Line(8) = {4,1};

Line Loop(9) = {5,6,7,8};
Plane Surface(10) = 9;

Transfinite Line{5,6,7,8} = boxdim/gridsize+1;
Transfinite Surface{10};
Recombine Surface{10};

//Now make 3D by extrusion.
newEntities[] = 
Extrude { 0,0,1 }
{
	Surface{10};
	Layers{boxdim/gridsize};
	Recombine;
};

// Physical Surface("back") = {10};
// Physical Surface("front") = {newEntities[0]};
// Physical Surface("bottom") = {newEntities[2]};
// Physical Surface("right") = {newEntities[3]};
// Physical Surface("top") = {newEntities[4]};
// Physical Surface("left") = {newEntities[5]};
Physical Volume(1) = {newEntities[1]};


// Generate 3D mesh
Mesh 3;
SetOrder 1;
Mesh.MshFileVersion = 2.2;


