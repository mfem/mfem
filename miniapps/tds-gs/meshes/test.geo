// Gmsh project created on Fri Sep 02 12:29:06 2022
SetFactory("OpenCASCADE");
//+ define coils
Rectangle(803) = {.4, -1, 0, .1, 2, 0};
//+ 
Rectangle(804) = {.75, -1, 0, .1, .1, 0};
//+
Rectangle(805) = {.75, .9, 0, .1, .1, 0};
//+
Rectangle(806) = {1.2, -.75, 0, .1, .1, 0};
//+
Rectangle(807) = {1.2, .65, 0, .1, .1, 0};
//+
Rectangle(808) = {1.5, -.4, 0, .1, .1, 0};
//+
Rectangle(809) = {1.5, .3, 0, .1, .1, 0};
//+
Rectangle(810) = {0.625, -0.375, 0, .75, .75, 0};
//+ define outer boundary
Point(429) = {0, 2.5, 0, 1.0};
//+
Point(430) = {-0, -2.5, 0, 1.0};
//+
Point(431) = {0, 0, 0, 1.0};
//+
Circle(829) = {429, 431, 430};
//+
Line(830) = {429, 430};
//+
Physical Curve("boundary", 831) = {829};
//+
Physical Surface("coil1", 832) = {803};
//+
Physical Surface("coil2", 833) = {805};
//+
Physical Surface("coil3", 834) = {807};
//+
Physical Surface("coil4", 835) = {809};
//+
Physical Surface("coil5", 836) = {808};
//+
Physical Surface("coil6", 837) = {806};
//+
Physical Surface("coil7", 838) = {804};
//+
Physical Surface("limiter", 1000) = {810};
//+
Curve Loop(9) = {830, -829};
//+
Curve Loop(10) = {4, 1, 2, 3};
//+
Curve Loop(11) = {12, 9, 10, 11};
//+
Curve Loop(12) = {20, 17, 18, 19};
//+
Curve Loop(13) = {28, 25, 26, 27};
//+
Curve Loop(14) = {24, 21, 22, 23};
//+
Curve Loop(15) = {16, 13, 14, 15};
//+
Curve Loop(16) = {6, 7, 8, 5};
//+
Curve Loop(17) = {29, 30, 31, 32};
//+
Plane Surface(811) = {9, 10, 11, 12, 13, 14, 15, 16, 17};
//+
Physical Surface("exterior", 2000) = {811};
