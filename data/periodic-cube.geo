// 0 for tetrahedra, 1 for hexahedra
tet_or_hex = 1;

Point(1) = {0, 0, 0, 1.0};
Point(2) = {1, 0, 0, 1.0};
Point(3) = {1, 1, 0, 1.0};
Point(4) = {0, 1, 0, 1.0};

Characteristic Length {:} = 0.25;

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Periodic Curve {1} = {-3};
Periodic Curve {2} = {-4};

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Transfinite Surface {1};

If (tet_or_hex == 1)
   Recombine Surface {1};
   out[] = Extrude {0, 0, 1} { Surface{1}; Layers{4}; Recombine; };
Else
   out[] = Extrude {0, 0, 1} { Surface{1}; Layers{4}; }
EndIf

Physical Volume(1) = {out[1]};
Physical Surface(1) = {1,out[0],out[2],out[3],out[4],out[5]};

Mesh 3;
Mesh.MshFileVersion = 2.2;

Periodic Surface {out[0]} = {1} Translate {0, 0, 1};
Periodic Surface {out[4]} = {out[2]} Translate {0, 1, 0};
Periodic Surface {out[3]} = {out[5]} Translate {1, 0, 0};
