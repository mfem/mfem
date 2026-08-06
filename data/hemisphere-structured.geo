SetFactory("Built-in");

// Set the geometry order (1, 2, ..., 9)
order = 2;

// Set the element type (4 - hexahedra, 3 - tetrahedra)
type = 4;

// Number of elements across each of the six blocks
nR = 1;

// Hemisphere radius
R = 1.0;

// Inner cube half-size, and the equator (s2) and cap (s3) corner coordinates
a  = R/(2*Sqrt(3));
s2 = R/Sqrt(2);
s3 = R/Sqrt(3);

Point(1) = {0, 0, R, 1.0};
Point(2) = {-a, -a, R, 1.0};
Point(3) = {a, -a, R, 1.0};
Point(4) = {a, a, R, 1.0};
Point(5) = {-a, a, R, 1.0};
Point(6) = {-a, -a, R - a, 1.0};
Point(7) = {a, -a, R - a, 1.0};
Point(8) = {a, a, R - a, 1.0};
Point(9) = {-a, a, R - a, 1.0};
Point(10) = {-s2, -s2, R, 1.0};
Point(11) = {s2, -s2, R, 1.0};
Point(12) = {s2, s2, R, 1.0};
Point(13) = {-s2, s2, R, 1.0};
Point(14) = {-s3, -s3, R - s3, 1.0};
Point(15) = {s3, -s3, R - s3, 1.0};
Point(16) = {s3, s3, R - s3, 1.0};
Point(17) = {-s3, s3, R - s3, 1.0};

Line(100) = {2, 3};
Line(101) = {3, 4};
Line(102) = {4, 5};
Line(103) = {5, 2};
Line(104) = {6, 7};
Line(105) = {7, 8};
Line(106) = {8, 9};
Line(107) = {9, 6};
Line(108) = {3, 7};
Line(109) = {6, 2};
Line(110) = {4, 8};
Line(111) = {5, 9};
Circle(112) = {14, 1, 15};
Circle(113) = {15, 1, 16};
Circle(114) = {16, 1, 17};
Circle(115) = {17, 1, 14};
Line(116) = {7, 15};
Line(117) = {14, 6};
Line(118) = {8, 16};
Line(119) = {9, 17};
Line(120) = {3, 11};
Circle(121) = {11, 1, 10};
Line(122) = {10, 2};
Circle(123) = {11, 1, 15};
Circle(124) = {10, 1, 14};
Line(125) = {4, 12};
Circle(126) = {12, 1, 11};
Circle(127) = {12, 1, 16};
Line(128) = {5, 13};
Circle(129) = {13, 1, 12};
Circle(130) = {13, 1, 17};
Circle(131) = {10, 1, 13};

Curve Loop(200) = {100, 101, 102, 103};
Surface(200) = {200};
Curve Loop(201) = {104, 105, 106, 107};
Surface(201) = {201};
Curve Loop(202) = {100, 108, -104, 109};
Surface(202) = {202};
Curve Loop(203) = {101, 110, -105, -108};
Surface(203) = {203};
Curve Loop(204) = {102, 111, -106, -110};
Surface(204) = {204};
Curve Loop(205) = {103, -109, -107, -111};
Surface(205) = {205};
Curve Loop(206) = {112, 113, 114, 115};
Surface(206) = {206};
Curve Loop(207) = {104, 116, -112, 117};
Surface(207) = {207};
Curve Loop(208) = {105, 118, -113, -116};
Surface(208) = {208};
Curve Loop(209) = {106, 119, -114, -118};
Surface(209) = {209};
Curve Loop(210) = {107, -117, -115, -119};
Surface(210) = {210};
Curve Loop(211) = {100, 120, 121, 122};
Surface(211) = {211};
Curve Loop(212) = {120, 123, -116, -108};
Surface(212) = {212};
Curve Loop(213) = {121, 124, 112, -123};
Surface(213) = {213};
Curve Loop(214) = {122, -109, -117, -124};
Surface(214) = {214};
Curve Loop(215) = {101, 125, 126, -120};
Surface(215) = {215};
Curve Loop(216) = {125, 127, -118, -110};
Surface(216) = {216};
Curve Loop(217) = {126, 123, 113, -127};
Surface(217) = {217};
Curve Loop(218) = {102, 128, 129, -125};
Surface(218) = {218};
Curve Loop(219) = {128, 130, -119, -111};
Surface(219) = {219};
Curve Loop(220) = {129, 127, 114, -130};
Surface(220) = {220};
Curve Loop(221) = {103, -122, 131, -128};
Surface(221) = {221};
Curve Loop(222) = {131, 130, 115, -124};
Surface(222) = {222};

Surface Loop(900) = {200, 201, 202, 203, 204, 205};
Volume(1000) = {900};
Surface Loop(901) = {201, 206, 207, 208, 209, 210};
Volume(1001) = {901};
Surface Loop(902) = {211, 207, 202, 212, 213, 214};
Volume(1002) = {902};
Surface Loop(903) = {215, 208, 203, 216, 217, 212};
Volume(1003) = {903};
Surface Loop(904) = {218, 209, 204, 219, 220, 216};
Volume(1004) = {904};
Surface Loop(905) = {221, 210, 205, 214, 222, 219};
Volume(1005) = {905};

Transfinite Curve{100:131} = nR+1;
Transfinite Surface{200:222};

// Structured hexes need a transfinite volume; tets must NOT have one (its fixed
// interior pattern doesn't match the surface mesh, which makes MFEM reject the
// mesh).  So apply it only for type == 4; for type == 3 the volume is filled with
// tets that conform to the (still transfinite) surface mesh.
If (type == 4)
   Recombine Surface{200:222};
   Transfinite Volume{1000:1005};
EndIf

// Tag surfaces and volumes with positive integers
// 1 - spherical dome (points down, bottom at z = 0), 2 - flat face (z = R, on top)
Physical Surface(1) = {206, 213, 217, 220, 222};
Physical Surface(2) = {200, 211, 215, 218, 221};
Physical Volume(1) = {1000:1005};

// Optimize the high-order mesh
// See https://gmsh.info/doc/texinfo/gmsh.html#index-Mesh_002eHighOrderOptimize
Mesh.HighOrderOptimize = 1;

// Generate 3D mesh
Mesh 3;
SetOrder order;

// The sphere-center point (0,0,R) is only an arc centre, but gmsh still meshes
// it as a stray node sitting inside the flat top face.  Weld coincident nodes
// so that floating duplicate is removed.
Coherence Mesh;

Mesh.MshFileVersion = 2.2;

Save Sprintf("hemisphere-structured-t%01g-o%01g.msh", type, order);
