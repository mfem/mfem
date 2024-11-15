SetFactory("OpenCASCADE");


N = 19;
lc = 0.01;
R = 1.0;
xc= 0.0;
yc= 0.0;
a = 2*Pi/(N+1);


cl = newcl;
Circle(cl) = {0,0,0,R*1.1,0,2*Pi};

nc = newc;
Curve Loop(nc) = {cl};

s1 = news;
Plane Surface(s1) = {nc};


c = newc;
For i In {0:N}
	x =  xc+R*Sin(i*a);  
	y =  yc+R*Cos(i*a);
        Circle(c+i) = { x, y, 0, 0.05, 0, 2*Pi};
	Curve{c+i} In Surface{s1};	

EndFor

ck = newc;
Circle(ck) = { 0, 0, 0, 0.05, 0, 2*Pi};
Curve{ck} In Surface{s1};

// Recombine Surface {s1};


Physical Surface(1) = {s1};
Physical Curve(1) = {ck};
Physical Curve(2) = {cl};

For i In {0:N}
	Physical Curve(3+i) = {c+i};
EndFor

// Generate 2D mesh
Mesh 2;
SetOrder 2;
Mesh.MshFileVersion = 2.2;
Mesh.ElementOrder = 2;
Mesh.HighOrderOptimize = 2;
Mesh.CharacteristicLengthMin = 0.01;
Mesh.CharacteristicLengthMax = 0.02;
