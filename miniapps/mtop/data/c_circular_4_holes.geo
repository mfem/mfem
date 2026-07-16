// Subfigure c: circular domain with four equal-radius circular holes
// Parametric Gmsh input file for MFEM-compatible 2-D meshes.
//
// Generate a mesh after editing the USER OPTIONS block below:
//   gmsh c_circular_4_holes.geo -2 -o c_circular_4_holes.msh
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
elementOrder = 1;

// Target mesh size.  Decrease this for a finer mesh.
meshSize = 0.050;

// Circular-domain dimensions.
outerRadius = 1.000;
holeRadius = 0.050;       // same radius for all circular holes in this file
radialHoleOffset = 0.900;
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
Point(6) = {0, radialHoleOffset, 0, meshSize};
Point(7) = {(0) + (holeRadius), radialHoleOffset, 0, meshSize};
Point(8) = {0, (radialHoleOffset) + (holeRadius), 0, meshSize};
Point(9) = {(0) - (holeRadius), radialHoleOffset, 0, meshSize};
Point(10) = {0, (radialHoleOffset) - (holeRadius), 0, meshSize};
Circle(5) = {7, 6, 8};
Circle(6) = {8, 6, 9};
Circle(7) = {9, 6, 10};
Circle(8) = {10, 6, 7};
Curve Loop(2) = {5, 6, 7, 8};
Physical Curve("hole_north", 2) = {5, 6, 7, 8};
Point(11) = {-radialHoleOffset, 0, 0, meshSize};
Point(12) = {(-radialHoleOffset) + (holeRadius), 0, 0, meshSize};
Point(13) = {-radialHoleOffset, (0) + (holeRadius), 0, meshSize};
Point(14) = {(-radialHoleOffset) - (holeRadius), 0, 0, meshSize};
Point(15) = {-radialHoleOffset, (0) - (holeRadius), 0, meshSize};
Circle(9) = {12, 11, 13};
Circle(10) = {13, 11, 14};
Circle(11) = {14, 11, 15};
Circle(12) = {15, 11, 12};
Curve Loop(3) = {9, 10, 11, 12};
Physical Curve("hole_west", 3) = {9, 10, 11, 12};
Point(16) = {radialHoleOffset, 0, 0, meshSize};
Point(17) = {(radialHoleOffset) + (holeRadius), 0, 0, meshSize};
Point(18) = {radialHoleOffset, (0) + (holeRadius), 0, meshSize};
Point(19) = {(radialHoleOffset) - (holeRadius), 0, 0, meshSize};
Point(20) = {radialHoleOffset, (0) - (holeRadius), 0, meshSize};
Circle(13) = {17, 16, 18};
Circle(14) = {18, 16, 19};
Circle(15) = {19, 16, 20};
Circle(16) = {20, 16, 17};
Curve Loop(4) = {13, 14, 15, 16};
Physical Curve("hole_east", 4) = {13, 14, 15, 16};
Point(21) = {0, -radialHoleOffset, 0, meshSize};
Point(22) = {(0) + (holeRadius), -radialHoleOffset, 0, meshSize};
Point(23) = {0, (-radialHoleOffset) + (holeRadius), 0, meshSize};
Point(24) = {(0) - (holeRadius), -radialHoleOffset, 0, meshSize};
Point(25) = {0, (-radialHoleOffset) - (holeRadius), 0, meshSize};
Circle(17) = {22, 21, 23};
Circle(18) = {23, 21, 24};
Circle(19) = {24, 21, 25};
Circle(20) = {25, 21, 22};
Curve Loop(5) = {17, 18, 19, 20};
Physical Curve("hole_south", 5) = {17, 18, 19, 20};
Plane Surface(1) = {1, 2, 3, 4, 5};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
Plane Surface(4) = {4};
Plane Surface(5) = {5};

// -----------------------------------------------------------------------------
// Physical groups / MFEM attributes
// -----------------------------------------------------------------------------
// 2-D element attribute 1: domain
// boundary attribute 101: outer_boundary
// boundary attribute 102: hole_north
// boundary attribute 103: hole_west
// boundary attribute 104: hole_east
// boundary attribute 105: hole_south
Physical Surface("domain", 1) = {1};
Physical Surface("hole_north", 2) = {2};
Physical Surface("hole_west", 3) = {3};
Physical Surface("hole_east", 4) = {4};
Physical Surface("hole_south", 5) = {5};
