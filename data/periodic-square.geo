// 0 for triangles, 1 for quads
tri_or_quad = 1;

Point(1) = {0, 0, 0, 1.0};
Point(2) = {1, 0, 0, 1.0};
Point(3) = {1, 1, 0, 1.0};
Point(4) = {0, 1, 0, 1.0};

Characteristic Length {:} = 0.25;

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Periodic Line {3} = {-1};
Periodic Line {2} = {-4};

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Transfinite Surface {1};

If (tri_or_quad == 1)
   Recombine Surface {1};
EndIf

Physical Surface(1) = {1};
Physical Curve(1) = {1, 2, 3, 4};

Mesh.MshFileVersion = 2.2;
Mesh 2;
