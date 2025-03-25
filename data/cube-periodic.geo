//https://bthierry.pages.math.cnrs.fr/tutorial/gmsh/occ_basics/
//http://onelab.info/pipermail/gmsh/2017/011277.html
//https://fossies.org/linux/gmsh/demos/boolean/periodic.geo
//https://scicomp.stackexchange.com/questions/30651/gmsh-for-3d-volume-with-inclusions

floor=0;

ep = 1.;
mp = 0.5*ep;
cl=mp/1.1;

Point(1) = { 0,   0,  0, cl} ; 
Point(2) = { 0,   0, ep, cl} ;
Point(3) = { 0,  ep, ep, cl} ;
Point(4) = { 0,  ep,  0, cl} ;

Point(5) = { mp,  0,  0, cl} ; 
Point(6) = { mp,  0, ep, cl} ;
Point(7) = { mp, ep, ep, cl} ;
Point(8) = { mp, ep,  0, cl} ;

Point(9)  = { ep,  0,  0, cl} ; 
Point(10) = { ep,  0, ep, cl} ;
Point(11) = { ep, ep, ep, cl} ;
Point(12) = { ep, ep,  0, cl} ;

Line(1) = {1,2}; 
Line(2) = {2,3};
Line(3) = {3,4}; 
Line(4) = {4,1};
Line Loop(1) = {  1, 2, 3, 4 } ;
Plane Surface(1) = {1};

Line(5) = {5,6}; 
Line(6) = {6,7};
Line(7) = {7,8}; 
Line(8) = {8,5};
Line Loop(2) = {  5, 6, 7, 8 } ;
Plane Surface(2) = {2};

Line(9)  = {9 ,10}; 
Line(10) = {10,11};
Line(11) = {11,12}; 
Line(12) = {12, 9};
Line Loop(3) = {  9, 10, 11, 12 } ;
Plane Surface(3) = {3};

Periodic Surface {3} = {1} Translate {ep,0,0};

Line(13) = {2, 6};
Line(14) = {1, 5};
Line(15) = {3, 7};
Line(16) = {4, 8};
Line(17) = {6, 10};
Line(18) = {5, 9};
Line(19) = {7, 11};
Line(20) = {8, 12};

Line Loop(4) = { 1, 13, -5, -14 };
Plane Surface(4) = {4};

Line Loop(5) = { 3, 16, -7, -15 };
Plane Surface(5) = {5};

Periodic Surface {5} = {4} Translate {0,ep,0};

Line Loop(6) = { 5, 17, -9, -18 };
Plane Surface(6) = {6};
Line Loop(7) = { 7, 20, -11, -19 };
Plane Surface(7) = {7};

Periodic Surface {7} = {6} Translate {0,ep,0};

Line Loop(8) = { 8, 18, -12, -20 };
Plane Surface(8) = {8};

Line Loop(9) = { 6, 19, -10, -17 };
Plane Surface(9) = {9};

Periodic Surface {9} = {8} Translate {0,0,ep};

Line Loop(10) = { 8, -14, -4, 16 };
Plane Surface(10) = {10};

Line Loop(11) = { 6, -15, -2, 13 };
Plane Surface(11) = {11};

Periodic Surface {11} = {10} Translate {0,0,ep};


Surface Loop (1) = {1, 2, 4, 5, 10, 11};
Volume (1) = {1};

Surface Loop (2) = {2, 3, 6, 7, 8, 9};
Volume (2) = {2};

Physical Volume(1)  = {1, 2};

//#For vo In {1:2}
//#    Physical Volume(vo)  = {vo}; 
//#EndFor
//For su In {1:11}
//    Physical Surface(su)  = {su}; 
//EndFor
//For li In {1:100}
//    Physical Line(li)  = {li}; 
//EndFor
//For pt In {1:100}
//    Physical Point(pt)  = {pt}; 
//EndFor

