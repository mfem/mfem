// Gmsh project created on Thu Aug 31 12:00:21 2023
SetFactory("OpenCASCADE");
//+
Rectangle(1) = {3, -6, 0, 7, 12, 0};
//+
Transfinite Surface {1};
//+
Transfinite Curve {3} = 257 Using Progression 1;
//+
Transfinite Curve {1} = 257 Using Progression 1;
//+
Transfinite Curve {4} = 513 Using Progression 1;
//+
Transfinite Curve {2} = 513 Using Progression 1;
//+
Recombine Surface {1};
