// mask_mfem_quad.geo
// Gmsh geometry from the supplied sketch: outer rectangle L x H with a
// rectangular mask attached to the top boundary. The mask has side offsets a
// and depth d. Its width is split into Nmask spans of length b.
//
// MFEM attributes:
//   Physical Surface 1: matrix/background subdomain
//   Physical Surface 2: mask subdomain
//
// Boundary attributes:
//   10 exterior_left
//   11 exterior_bottom
//   12 exterior_right
//   13 exterior_top_matrix       top exterior outside the mask
//   1401..1400+Nmask             split physical tags for mask exterior/top
//   15 mask_internal_interface   internal mask/matrix interface, U-shaped
//
// Export detail for the old boundary attribute 14:
//   The mask top is split into separate elementary curves and separate
//   Physical Curve tags. For Nmask = 7 these physical tags are
//   1401, 1402, 1403, 1404, 1405, 1406, and 1407. In MSH 2.2 these appear
//   as line elements whose first tag is the per-segment physical attribute.
//
// Mesh command:
//   gmsh -2 -format msh2 mask_mfem_quad.geo -o mask_mfem_quad.msh
SetFactory("Built-in");
// ---------------- Parameters ----------------
// Number of b-spans across the mask. The sketch shows seven b-spans.
Nmask = 7;
// Geometry. With this convention L = 2*a + Nmask*b.
a = 1.0;
b = 1.0;
L = 2*a + Nmask*b;
H = 4.0;
d = 1.5;
// Mesh subdivisions. These are numbers of quad elements, not numbers of nodes.
// Increase these for a finer structured quad mesh.
Na      = 2; // elements across each side offset a
Nb      = 2; // elements in each b-span
NyBase  = 5; // elements in the lower region, height H-d
NyMask  = 3; // elements through the mask depth d
lc = b;
Nx = Nmask + 2; // total number of x-intervals: a + Nmask*b + a
// Stable Gmsh elementary entity tags. The mask top uses the same numeric tags
// for both elementary curves and physical attributes: 1401, 1402, ... .
BottomTagBase        = 100;
MidTagBase           = 200;
TopSideTagBase       = 300;
VBaseTagBase         = 400;
VMaskTagBase         = 500;
MaskTopTagBase       = 1400;
SurfaceBottomTagBase = 1000;
SurfaceTopTagBase    = 2000;
LoopBottomTagBase    = 3000;
LoopTopTagBase       = 4000;
// MFEM-friendly output: ASCII MSH 2.2, first-order elements.
Mesh.MshFileVersion = 2.2;
Mesh.Binary = 0;
Mesh.ElementOrder = 1;
Mesh.SaveAll = 0;
Mesh.RecombineAll = 1;
Mesh.RecombinationAlgorithm = 1;
// ---------------- Points ----------------
// Three horizontal rows: y = 0, y = H-d, y = H.
For i In {0:Nx}
  If (i == 0)
    xx = 0;
  ElseIf (i == 1)
    xx = a;
  ElseIf (i == Nx)
    xx = L;
  Else
    xx = a + (i - 1)*b;
  EndIf
  p0[i] = newp; Point(p0[i]) = {xx, 0,   0, lc};
  p1[i] = newp; Point(p1[i]) = {xx, H-d, 0, lc};
  p2[i] = newp; Point(p2[i]) = {xx, H,   0, lc};
EndFor
// ---------------- Curves ----------------
For i In {0:Nx-1}
  bottom[i] = BottomTagBase + i;
  Line(bottom[i]) = {p0[i], p0[i+1]};
  mid[i] = MidTagBase + i;
  Line(mid[i]) = {p1[i], p1[i+1]};
  If (i == 0)
    top[i] = TopSideTagBase + i;
  ElseIf (i == Nx-1)
    top[i] = TopSideTagBase + i;
  Else
    // Split elementary line segments for the mask top.
    top[i] = MaskTopTagBase + i;
  EndIf
  Line(top[i]) = {p2[i], p2[i+1]};
EndFor
For i In {0:Nx}
  vbase[i] = VBaseTagBase + i;
  Line(vbase[i]) = {p0[i], p1[i]};
  vmask[i] = VMaskTagBase + i;
  Line(vmask[i]) = {p1[i], p2[i]};
EndFor
// ---------------- Structured quad surfaces ----------------
// Bottom row: all matrix. Top row: side columns are matrix, middle columns are mask.
matrix_surfaces[] = {};
mask_surfaces[] = {};
all_surfaces[] = {};
For i In {0:Nx-1}
  // Bottom surface in column i.
  clb = LoopBottomTagBase + i;
  Curve Loop(clb) = {bottom[i], vbase[i+1], -mid[i], -vbase[i]};
  sb = SurfaceBottomTagBase + i;
  Plane Surface(sb) = {clb};
  Transfinite Surface {sb} = {p0[i], p0[i+1], p1[i+1], p1[i]};
  matrix_surfaces[] += {sb};
  all_surfaces[] += {sb};
  // Top surface in column i.
  clt = LoopTopTagBase + i;
  Curve Loop(clt) = {mid[i], vmask[i+1], -top[i], -vmask[i]};
  st = SurfaceTopTagBase + i;
  Plane Surface(st) = {clt};
  Transfinite Surface {st} = {p1[i], p1[i+1], p2[i+1], p2[i]};
  all_surfaces[] += {st};
  If (i == 0)
    matrix_surfaces[] += {st};
  ElseIf (i == Nx-1)
    matrix_surfaces[] += {st};
  Else
    mask_surfaces[] += {st};
  EndIf
EndFor
// ---------------- Transfinite line counts ----------------
For i In {0:Nx-1}
  If (i == 0)
    nxline = Na + 1;
  ElseIf (i == Nx-1)
    nxline = Na + 1;
  Else
    nxline = Nb + 1;
  EndIf
  Transfinite Curve {bottom[i], mid[i], top[i]} = nxline;
EndFor
For i In {0:Nx}
  Transfinite Curve {vbase[i]} = NyBase + 1;
  Transfinite Curve {vmask[i]} = NyMask + 1;
EndFor
Transfinite Surface {all_surfaces[]};
Recombine Surface {all_surfaces[]};
// ---------------- Physical groups for MFEM ----------------
Physical Surface("matrix", 1) = {matrix_surfaces[]};
Physical Surface("mask", 2) = {mask_surfaces[]};
// Exterior boundary, split into non-overlapping attributes.
Physical Curve("exterior_left", 10) = {vbase[0], vmask[0]};
Physical Curve("exterior_bottom", 11) = {bottom[]};
Physical Curve("exterior_right", 12) = {vbase[Nx], vmask[Nx]};
Physical Curve("exterior_top_matrix", 13) = {top[0], top[Nx-1]};
// Mask part of the exterior top boundary with one physical tag per b-span.
// For Nmask = 7 this creates physical tags 1401 through 1407.
For i In {1:Nx-2}
  Physical Curve(Sprintf("mask_external_top_%g", i), MaskTopTagBase + i) = {top[i]};
EndFor
// Internal mask/matrix boundary: left side, bottom of mask, right side.
mask_internal_interface[] = {vmask[1]};
For i In {1:Nx-2}
  mask_internal_interface[] += {mid[i]};
EndFor
mask_internal_interface[] += {vmask[Nx-1]};
Physical Curve("mask_internal_interface", 15) = {mask_internal_interface[]};
