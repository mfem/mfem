// Subfigure a: circular domain with five equal-radius circular holes
// arranged in a regular pentagon pattern (equal angular spacing, equal radius
// from the center).
// Parametric Gmsh input file for MFEM-compatible 2-D meshes.
//
// Generate a mesh after editing the USER OPTIONS block below:
//   gmsh circular_5_holes_pentagon.geo -2 -o circular_5_holes_pentagon.msh
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
holeRadius = 0.050;       // same radius for all five circular holes
radialHoleOffset = 0.900; // distance from center to each hole center
// Angle (degrees) of the first hole, measured counter-clockwise from the
// positive x-axis. The remaining four holes are placed at 72-degree
// increments from this one, giving a regular pentagon arrangement.
startAngleDeg = 90;
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

// -----------------------------------------------------------------------------
// Five pentagon-arranged holes.
// Hole k (k = 0..4) is centered at:
//   cx_k = radialHoleOffset * Cos(angle_k)
//   cy_k = radialHoleOffset * Sin(angle_k)
// with angle_k = startAngleDeg + k*72 degrees, converted to radians.
// -----------------------------------------------------------------------------

// --- Hole 1 (top) ---
angle1 = (startAngleDeg + 0*72) * Pi / 180;
cx1 = radialHoleOffset * Cos(angle1);
cy1 = radialHoleOffset * Sin(angle1);
Point(6)  = {cx1, cy1, 0, meshSize};
Point(7)  = {cx1 + holeRadius, cy1, 0, meshSize};
Point(8)  = {cx1, cy1 + holeRadius, 0, meshSize};
Point(9)  = {cx1 - holeRadius, cy1, 0, meshSize};
Point(10) = {cx1, cy1 - holeRadius, 0, meshSize};
Circle(5) = {7, 6, 8};
Circle(6) = {8, 6, 9};
Circle(7) = {9, 6, 10};
Circle(8) = {10, 6, 7};
Curve Loop(2) = {5, 6, 7, 8};
Physical Curve("hole_1", 2) = {5, 6, 7, 8};

// --- Hole 2 ---
angle2 = (startAngleDeg + 1*72) * Pi / 180;
cx2 = radialHoleOffset * Cos(angle2);
cy2 = radialHoleOffset * Sin(angle2);
Point(11) = {cx2, cy2, 0, meshSize};
Point(12) = {cx2 + holeRadius, cy2, 0, meshSize};
Point(13) = {cx2, cy2 + holeRadius, 0, meshSize};
Point(14) = {cx2 - holeRadius, cy2, 0, meshSize};
Point(15) = {cx2, cy2 - holeRadius, 0, meshSize};
Circle(9)  = {12, 11, 13};
Circle(10) = {13, 11, 14};
Circle(11) = {14, 11, 15};
Circle(12) = {15, 11, 12};
Curve Loop(3) = {9, 10, 11, 12};
Physical Curve("hole_2", 3) = {9, 10, 11, 12};

// --- Hole 3 ---
angle3 = (startAngleDeg + 2*72) * Pi / 180;
cx3 = radialHoleOffset * Cos(angle3);
cy3 = radialHoleOffset * Sin(angle3);
Point(16) = {cx3, cy3, 0, meshSize};
Point(17) = {cx3 + holeRadius, cy3, 0, meshSize};
Point(18) = {cx3, cy3 + holeRadius, 0, meshSize};
Point(19) = {cx3 - holeRadius, cy3, 0, meshSize};
Point(20) = {cx3, cy3 - holeRadius, 0, meshSize};
Circle(13) = {17, 16, 18};
Circle(14) = {18, 16, 19};
Circle(15) = {19, 16, 20};
Circle(16) = {20, 16, 17};
Curve Loop(4) = {13, 14, 15, 16};
Physical Curve("hole_3", 4) = {13, 14, 15, 16};

// --- Hole 4 ---
angle4 = (startAngleDeg + 3*72) * Pi / 180;
cx4 = radialHoleOffset * Cos(angle4);
cy4 = radialHoleOffset * Sin(angle4);
Point(21) = {cx4, cy4, 0, meshSize};
Point(22) = {cx4 + holeRadius, cy4, 0, meshSize};
Point(23) = {cx4, cy4 + holeRadius, 0, meshSize};
Point(24) = {cx4 - holeRadius, cy4, 0, meshSize};
Point(25) = {cx4, cy4 - holeRadius, 0, meshSize};
Circle(17) = {22, 21, 23};
Circle(18) = {23, 21, 24};
Circle(19) = {24, 21, 25};
Circle(20) = {25, 21, 22};
Curve Loop(5) = {17, 18, 19, 20};
Physical Curve("hole_4", 5) = {17, 18, 19, 20};

// --- Hole 5 ---
angle5 = (startAngleDeg + 4*72) * Pi / 180;
cx5 = radialHoleOffset * Cos(angle5);
cy5 = radialHoleOffset * Sin(angle5);
Point(26) = {cx5, cy5, 0, meshSize};
Point(27) = {cx5 + holeRadius, cy5, 0, meshSize};
Point(28) = {cx5, cy5 + holeRadius, 0, meshSize};
Point(29) = {cx5 - holeRadius, cy5, 0, meshSize};
Point(30) = {cx5, cy5 - holeRadius, 0, meshSize};
Circle(21) = {27, 26, 28};
Circle(22) = {28, 26, 29};
Circle(23) = {29, 26, 30};
Circle(24) = {30, 26, 27};
Curve Loop(6) = {21, 22, 23, 24};
Physical Curve("hole_5", 6) = {21, 22, 23, 24};

// -----------------------------------------------------------------------------
// Surfaces
// -----------------------------------------------------------------------------
Plane Surface(1) = {1, 2, 3, 4, 5, 6};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
Plane Surface(4) = {4};
Plane Surface(5) = {5};
Plane Surface(6) = {6};

// -----------------------------------------------------------------------------
// Physical groups / MFEM attributes
// -----------------------------------------------------------------------------
// 2-D element attribute 1: domain
// boundary attribute 101 (curve id 1..4): outer_boundary
// boundary attribute 102 (curve id 5..8): hole_1  (top,        angle  90 deg)
// boundary attribute 103 (curve id 9..12): hole_2 (upper-left, angle 162 deg)
// boundary attribute 104 (curve id 13..16): hole_3 (lower-left, angle 234 deg)
// boundary attribute 105 (curve id 17..20): hole_4 (lower-right,angle 306 deg)
// boundary attribute 106 (curve id 21..24): hole_5 (upper-right,angle  18 deg)

Physical Surface("domain", 1) = {1};
Physical Surface("hole_1", 2) = {2};
Physical Surface("hole_2", 3) = {3};
Physical Surface("hole_3", 4) = {4};
Physical Surface("hole_4", 5) = {5};
Physical Surface("hole_5", 6) = {6};
