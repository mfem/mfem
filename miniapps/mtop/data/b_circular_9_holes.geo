// Subfigure b: circular domain with a 3 by 3 array of equal-radius circular holes
// Parametric Gmsh input file for MFEM-compatible 2-D meshes.
//
// Generate a mesh after editing the USER OPTIONS block below:
//   gmsh b_circular_9_holes.geo -2 -o b_circular_9_holes.msh
//
// MFEM notes:
// - Output is forced to Gmsh MSH 2.2 ASCII.
// - All physical group IDs are strictly positive.
// - The physical surface tag is the MFEM element attribute.
// - The physical curve tags are the MFEM boundary attributes.

SetFactory("Built-in");

// -----------------------------------------------------------------------------
// USER OPTIONS - edit values in this block with any text editor.
// -----------------------------------------------------------------------------
// 0 = triangular mesh, 1 = quadrilateral mesh.
// In quad mode Gmsh uses Mesh.SubdivisionAlgorithm = 1, which converts the
// generated 2-D mesh to all quadrangles.
useQuads = 1;

// Linear elements are recommended for the most portable MFEM input.
elementOrder = 2;

// Target mesh size.  Decrease this for a finer mesh.
meshSize = 0.050;

// Circular-domain dimensions.
outerRadius = 1.000;
holeRadius = 0.050; // same radius for all circular holes in this file
gridSpacing = 0.650;
// -----------------------------------------------------------------------------
// END USER OPTIONS
// -----------------------------------------------------------------------------

// MFEM-oriented mesh output settings.
Mesh.MshFileVersion = 2.2;
Mesh.Binary = 0;
Mesh.SaveAll = 0;
Mesh.ElementOrder = elementOrder;
Mesh.CharacteristicLengthMin = meshSize;
Mesh.CharacteristicLengthMax = meshSize;

If (useQuads)
  Mesh.Algorithm = 6;
  Mesh.RecombineAll = 1;
  Mesh.SubdivisionAlgorithm = 1;
Else
  Mesh.Algorithm = 6;
  Mesh.RecombineAll = 0;
  Mesh.SubdivisionAlgorithm = 0;
EndIf

// -----------------------------------------------------------------------------
// Geometry
// -----------------------------------------------------------------------------
Point(1) = {0, 0, 0, meshSize};
Point(2) = {(0) + (outerRadius), 0, 0, meshSize};
Point(3) = {0, (0) + (outerRadius), 0, meshSize};
Point(4) = {(0) - (outerRadius), 0, 0, meshSize};
Point(5) = {0, (0) - (outerRadius), 0, meshSize};
Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 4};
Circle(3) = {4, 1, 5};
Circle(4) = {5, 1, 2};
Curve Loop(1) = {1, 2, 3, 4};
Physical Curve("outer_boundary", 1) = {1, 2, 3, 4};
Point(6) = {-gridSpacing, gridSpacing, 0, meshSize};
Point(7) = {(-gridSpacing) + (holeRadius), gridSpacing, 0, meshSize};
Point(8) = {-gridSpacing, (gridSpacing) + (holeRadius), 0, meshSize};
Point(9) = {(-gridSpacing) - (holeRadius), gridSpacing, 0, meshSize};
Point(10) = {-gridSpacing, (gridSpacing) - (holeRadius), 0, meshSize};
Circle(5) = {7, 6, 8};
Circle(6) = {8, 6, 9};
Circle(7) = {9, 6, 10};
Circle(8) = {10, 6, 7};
Curve Loop(2) = {5, 6, 7, 8};
Physical Curve("hole_r1_c1", 2) = {5, 6, 7, 8};
Point(11) = {0, gridSpacing, 0, meshSize};
Point(12) = {(0) + (holeRadius), gridSpacing, 0, meshSize};
Point(13) = {0, (gridSpacing) + (holeRadius), 0, meshSize};
Point(14) = {(0) - (holeRadius), gridSpacing, 0, meshSize};
Point(15) = {0, (gridSpacing) - (holeRadius), 0, meshSize};
Circle(9) = {12, 11, 13};
Circle(10) = {13, 11, 14};
Circle(11) = {14, 11, 15};
Circle(12) = {15, 11, 12};
Curve Loop(3) = {9, 10, 11, 12};
Physical Curve("hole_r1_c2", 3) = {9, 10, 11, 12};
Point(16) = {gridSpacing, gridSpacing, 0, meshSize};
Point(17) = {(gridSpacing) + (holeRadius), gridSpacing, 0, meshSize};
Point(18) = {gridSpacing, (gridSpacing) + (holeRadius), 0, meshSize};
Point(19) = {(gridSpacing) - (holeRadius), gridSpacing, 0, meshSize};
Point(20) = {gridSpacing, (gridSpacing) - (holeRadius), 0, meshSize};
Circle(13) = {17, 16, 18};
Circle(14) = {18, 16, 19};
Circle(15) = {19, 16, 20};
Circle(16) = {20, 16, 17};
Curve Loop(4) = {13, 14, 15, 16};
Physical Curve("hole_r1_c3", 4) = {13, 14, 15, 16};
Point(21) = {-gridSpacing, 0, 0, meshSize};
Point(22) = {(-gridSpacing) + (holeRadius), 0, 0, meshSize};
Point(23) = {-gridSpacing, (0) + (holeRadius), 0, meshSize};
Point(24) = {(-gridSpacing) - (holeRadius), 0, 0, meshSize};
Point(25) = {-gridSpacing, (0) - (holeRadius), 0, meshSize};
Circle(17) = {22, 21, 23};
Circle(18) = {23, 21, 24};
Circle(19) = {24, 21, 25};
Circle(20) = {25, 21, 22};
Curve Loop(5) = {17, 18, 19, 20};
Physical Curve("hole_r2_c1", 5) = {17, 18, 19, 20};
Point(26) = {0, 0, 0, meshSize};
Point(27) = {(0) + (holeRadius), 0, 0, meshSize};
Point(28) = {0, (0) + (holeRadius), 0, meshSize};
Point(29) = {(0) - (holeRadius), 0, 0, meshSize};
Point(30) = {0, (0) - (holeRadius), 0, meshSize};
Circle(21) = {27, 26, 28};
Circle(22) = {28, 26, 29};
Circle(23) = {29, 26, 30};
Circle(24) = {30, 26, 27};
Curve Loop(6) = {21, 22, 23, 24};
Physical Curve("hole_r2_c2", 6) = {21, 22, 23, 24};
Point(31) = {gridSpacing, 0, 0, meshSize};
Point(32) = {(gridSpacing) + (holeRadius), 0, 0, meshSize};
Point(33) = {gridSpacing, (0) + (holeRadius), 0, meshSize};
Point(34) = {(gridSpacing) - (holeRadius), 0, 0, meshSize};
Point(35) = {gridSpacing, (0) - (holeRadius), 0, meshSize};
Circle(25) = {32, 31, 33};
Circle(26) = {33, 31, 34};
Circle(27) = {34, 31, 35};
Circle(28) = {35, 31, 32};
Curve Loop(7) = {25, 26, 27, 28};
Physical Curve("hole_r2_c3", 7) = {25, 26, 27, 28};
Point(36) = {-gridSpacing, -gridSpacing, 0, meshSize};
Point(37) = {(-gridSpacing) + (holeRadius), -gridSpacing, 0, meshSize};
Point(38) = {-gridSpacing, (-gridSpacing) + (holeRadius), 0, meshSize};
Point(39) = {(-gridSpacing) - (holeRadius), -gridSpacing, 0, meshSize};
Point(40) = {-gridSpacing, (-gridSpacing) - (holeRadius), 0, meshSize};
Circle(29) = {37, 36, 38};
Circle(30) = {38, 36, 39};
Circle(31) = {39, 36, 40};
Circle(32) = {40, 36, 37};
Curve Loop(8) = {29, 30, 31, 32};
Physical Curve("hole_r3_c1", 8) = {29, 30, 31, 32};
Point(41) = {0, -gridSpacing, 0, meshSize};
Point(42) = {(0) + (holeRadius), -gridSpacing, 0, meshSize};
Point(43) = {0, (-gridSpacing) + (holeRadius), 0, meshSize};
Point(44) = {(0) - (holeRadius), -gridSpacing, 0, meshSize};
Point(45) = {0, (-gridSpacing) - (holeRadius), 0, meshSize};
Circle(33) = {42, 41, 43};
Circle(34) = {43, 41, 44};
Circle(35) = {44, 41, 45};
Circle(36) = {45, 41, 42};
Curve Loop(9) = {33, 34, 35, 36};
Physical Curve("hole_r3_c2", 9) = {33, 34, 35, 36};
Point(46) = {gridSpacing, -gridSpacing, 0, meshSize};
Point(47) = {(gridSpacing) + (holeRadius), -gridSpacing, 0, meshSize};
Point(48) = {gridSpacing, (-gridSpacing) + (holeRadius), 0, meshSize};
Point(49) = {(gridSpacing) - (holeRadius), -gridSpacing, 0, meshSize};
Point(50) = {gridSpacing, (-gridSpacing) - (holeRadius), 0, meshSize};
Circle(37) = {47, 46, 48};
Circle(38) = {48, 46, 49};
Circle(39) = {49, 46, 50};
Circle(40) = {50, 46, 47};
Curve Loop(10) = {37, 38, 39, 40};
Physical Curve("hole_r3_c3", 10) = {37, 38, 39, 40};
Plane Surface(1) = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
Plane Surface(4) = {4};
Plane Surface(5) = {5};
Plane Surface(6) = {6};
Plane Surface(7) = {7};
Plane Surface(8) = {8};
Plane Surface(9) = {9};
Plane Surface(10) = {10};


// -----------------------------------------------------------------------------
// Physical groups / MFEM attributes
// -----------------------------------------------------------------------------
// 2-D element attribute 1: domain
// boundary attribute 101: outer_boundary
// boundary attribute 102: hole_r1_c1
// boundary attribute 103: hole_r1_c2
// boundary attribute 104: hole_r1_c3
// boundary attribute 105: hole_r2_c1
// boundary attribute 106: hole_r2_c2
// boundary attribute 107: hole_r2_c3
// boundary attribute 108: hole_r3_c1
// boundary attribute 109: hole_r3_c2
// boundary attribute 110: hole_r3_c3
Physical Surface("domain", 1) = {1};
Physical Surface("hole_r1_c1", 2) = {2};
Physical Surface("hole_r1_c2", 3) = {3};
Physical Surface("hole_r1_c3", 4) = {4};
Physical Surface("hole_r2_c1", 5) = {5};
Physical Surface("hole_r2_c2", 6) = {6};
Physical Surface("hole_r2_c3", 7) = {7};
Physical Surface("hole_r3_c1", 8) = {8};
Physical Surface("hole_r3_c2", 9) = {9};
Physical Surface("hole_r3_c3", 10) = {10};
