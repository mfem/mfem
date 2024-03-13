//+
SetFactory("OpenCASCADE");

// dimensions and gridsize as from example 9
boxdimx = 180.;
boxdimy = 64;
boxdimz = 64;

numx = 16; 
numy = 8;
numz = 8;

gridsizex = boxdimx/numx;
gridsizey = boxdimy/numy;
gridsizez = boxdimz/numz;

//Create 2D rectangular mesh
Point(1) = {0,0,0, gridsizex};
Point(2) = {boxdimx,0,0, gridsizex};
Point(3) = {boxdimx,boxdimy,0, gridsizey};
Point(4) = {0,boxdimy,0, gridsizey};

Line(5) = {1,2};
Line(6) = {2,3};
Line(7) = {3,4};
Line(8) = {4,1};

Line Loop(9) = {5,6,7,8};
Plane Surface(10) = 9;

Transfinite Curve {5, 7} = boxdimx/gridsizex;
Transfinite Curve {6, 8} = boxdimy/gridsizey;
//+
Transfinite Surface {10};
//+
Recombine Surface {10};

//Extrude 2D rectangular mesh into 3D
newEntities[] = 
Extrude {0,0,boxdimz}
{
    Surface{10};
    Layers{numz};
    Recombine;
};

Physical Surface("Wall_back") = {10};
Physical Surface("Wall_front") = {newEntities[0]};
Physical Surface("Wall_bottom") = {newEntities[2]};
Physical Surface("Outflow") = {newEntities[3]};
Physical Surface("Wall_top") = {newEntities[4]};
Physical Surface("Inflow") = {newEntities[5]};
Physical Volume(100) = {newEntities[1]};

Mesh.MshFileVersion = 2.2;//+