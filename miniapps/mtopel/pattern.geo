SetFactory("OpenCASCADE");


N = 19;
lc = 0.01;
R = 1.0;
xc= 0.0;
yc= 0.0;
a = 2*Pi/(N+1);


clo = newcl;
Circle(clo) = {0,0,0,R*1.1,0,2*Pi};

nc = newc;
Curve Loop(nc) = {clo};

s1 = news;
Plane Surface(s1) = {nc};


c = newc;
nso = news;
For i In {0:N}
	x =  xc+R*Sin(i*a);  
	y =  yc+R*Cos(i*a);
        Circle(c+i) = { x, y, 0, 0.05, 0, 2*Pi};
	nc = newc;
	Curve Loop(nc) = {c+i};
	Plane Surface(nso+i) = {nc};
EndFor


For i In {0:N}
	se = news;
        BooleanDifference(se) = {Surface{s1}; Delete;}{Surface{nso+i}; Delete;};
        s1=se;
EndFor

ckc = newc;
Circle(ckc) = { 0, 0, 0, 0.05, 0, 2*Pi};
nc = newc;
Curve Loop(nc) = {ckc};
ns = news;
Plane Surface(ns) = {nc};
se = news;
BooleanDifference(se) = {Surface{s1}; Delete;}{Surface{ns}; Delete;};
s1=se;


// For i In {0:N}
//       Curve{c+i} In Surface{s1};
// EndFor
// Curve{ckc} In Surface{s1};


For i In {0:N}
	x =  xc+R*Sin(i*a);
        y =  yc+R*Cos(i*a);
	
	p() = Curve In BoundingBox{x-0.055,y-0.055, -0.1,
				   x+0.055,y+0.055,  0.1};
	
	Physical Curve(3+i) = {p()};
EndFor

p() = Curve In BoundingBox{0-0.055,0-0.055, -0.1,
                          0+0.055,0+0.055,  0.1};
Physical Curve(1) = {p()};



// Recombine Surface {s1};


Physical Surface(1) = {s1};
Physical Curve(2) = {clo};

// For i In {0:N}
//	Physical Curve(3+i) = {c+i};
// EndFor

// Generate 2D mesh
Mesh 2;
SetOrder 2;
Mesh.MshFileVersion = 2.2;
Mesh.ElementOrder = 2;
Mesh.HighOrderOptimize = 2;
Mesh.CharacteristicLengthMin = 0.01;
Mesh.CharacteristicLengthMax = 0.02;
