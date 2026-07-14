// Subfigure d: square domain with four equal-radius circular holes
// Parametric Gmsh input file for MFEM-compatible 2-D meshes.
//
// Generate a mesh after editing the USER OPTIONS block below:
//   gmsh d_square_4_holes.geo -2 -o d_square_4_holes.msh
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

// Square-domain dimensions.
squareSide = 1.000;
holeRadius = 0.050; // same radius for all circular holes in this file
cornerHoleOffset = 0.40;
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
Point(1) = {-(squareSide)/2, -(squareSide)/2, 0, meshSize};
Point(2) = { (squareSide)/2, -(squareSide)/2, 0, meshSize};
Point(3) = { (squareSide)/2,  (squareSide)/2, 0, meshSize};
Point(4) = {-(squareSide)/2,  (squareSide)/2, 0, meshSize};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Curve Loop(1) = {1, 2, 3, 4};
Physical Curve("outer_bottom", 1) = {1};
Physical Curve("outer_right", 2) = {2};
Physical Curve("outer_top", 3) = {3};
Physical Curve("outer_left", 4) = {4};
Point(5) = {-cornerHoleOffset, cornerHoleOffset, 0, meshSize};
Point(6) = {(-cornerHoleOffset) + (holeRadius), cornerHoleOffset, 0, meshSize};
Point(7) = {-cornerHoleOffset, (cornerHoleOffset) + (holeRadius), 0, meshSize};
Point(8) = {(-cornerHoleOffset) - (holeRadius), cornerHoleOffset, 0, meshSize};
Point(9) = {-cornerHoleOffset, (cornerHoleOffset) - (holeRadius), 0, meshSize};
Circle(5) = {6, 5, 7};
Circle(6) = {7, 5, 8};
Circle(7) = {8, 5, 9};
Circle(8) = {9, 5, 6};
Curve Loop(2) = {5, 6, 7, 8};
Physical Curve("hole_top_left", 5) = {5, 6, 7, 8};
Point(10) = {cornerHoleOffset, cornerHoleOffset, 0, meshSize};
Point(11) = {(cornerHoleOffset) + (holeRadius), cornerHoleOffset, 0, meshSize};
Point(12) = {cornerHoleOffset, (cornerHoleOffset) + (holeRadius), 0, meshSize};
Point(13) = {(cornerHoleOffset) - (holeRadius), cornerHoleOffset, 0, meshSize};
Point(14) = {cornerHoleOffset, (cornerHoleOffset) - (holeRadius), 0, meshSize};
Circle(9) = {11, 10, 12};
Circle(10) = {12, 10, 13};
Circle(11) = {13, 10, 14};
Circle(12) = {14, 10, 11};
Curve Loop(3) = {9, 10, 11, 12};
Physical Curve("hole_top_right", 6) = {9, 10, 11, 12};
Point(15) = {-cornerHoleOffset, -cornerHoleOffset, 0, meshSize};
Point(16) = {(-cornerHoleOffset) + (holeRadius), -cornerHoleOffset, 0, meshSize};
Point(17) = {-cornerHoleOffset, (-cornerHoleOffset) + (holeRadius), 0, meshSize};
Point(18) = {(-cornerHoleOffset) - (holeRadius), -cornerHoleOffset, 0, meshSize};
Point(19) = {-cornerHoleOffset, (-cornerHoleOffset) - (holeRadius), 0, meshSize};
Circle(13) = {16, 15, 17};
Circle(14) = {17, 15, 18};
Circle(15) = {18, 15, 19};
Circle(16) = {19, 15, 16};
Curve Loop(4) = {13, 14, 15, 16};
Physical Curve("hole_bottom_left", 7) = {13, 14, 15, 16};
Point(20) = {cornerHoleOffset, -cornerHoleOffset, 0, meshSize};
Point(21) = {(cornerHoleOffset) + (holeRadius), -cornerHoleOffset, 0, meshSize};
Point(22) = {cornerHoleOffset, (-cornerHoleOffset) + (holeRadius), 0, meshSize};
Point(23) = {(cornerHoleOffset) - (holeRadius), -cornerHoleOffset, 0, meshSize};
Point(24) = {cornerHoleOffset, (-cornerHoleOffset) - (holeRadius), 0, meshSize};
Circle(17) = {21, 20, 22};
Circle(18) = {22, 20, 23};
Circle(19) = {23, 20, 24};
Circle(20) = {24, 20, 21};
Curve Loop(5) = {17, 18, 19, 20};
Physical Curve("hole_bottom_right", 8) = {17, 18, 19, 20};
Plane Surface(1) = {1, 2, 3, 4, 5};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
Plane Surface(4) = {4};
Plane Surface(5) = {5};

// -----------------------------------------------------------------------------
// Physical groups / MFEM attributes
// -----------------------------------------------------------------------------
// 2-D element attribute 1: domain
// boundary attribute 101: outer_bottom
// boundary attribute 102: outer_right
// boundary attribute 103: outer_top
// boundary attribute 104: outer_left
// boundary attribute 105: hole_top_left
// boundary attribute 106: hole_top_right
// boundary attribute 107: hole_bottom_left
// boundary attribute 108: hole_bottom_right

Physical Surface("matrix", 1) = {1};
Physical Surface("hole_top_left", 2) = {2};
Physical Surface("hole_top_right", 3) = {3};
Physical Surface("hole_bottom_left", 4) = {4};
Physical Surface("hole_bottom_right", 5) = {5};