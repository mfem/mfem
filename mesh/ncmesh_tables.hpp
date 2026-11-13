// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_NCMESH_TABLES
#define MFEM_NCMESH_TABLES

namespace mfem
{

// derefinement tables
// The first n numbers in each line are the refined elements that contain
// the vertices of the parent element.  The next m numbers in each line
// are the refined elements that contain the faces attributes of the parent
// element.

static constexpr int quad_deref_table[3][4 + 4] =
{
   { 0, 1, 1, 0, /**/ 1, 1, 0, 0 }, // 1 - X
   { 0, 0, 1, 1, /**/ 0, 0, 1, 1 }, // 2 - Y
   { 0, 1, 2, 3, /**/ 1, 1, 3, 3 }  // 3 - iso
};

// For mixed-child patterns, each parent vertex/face also needs a local index
// in the child selected by the derefinement table.
static constexpr int quad_diag_02_deref[4 + 4] =
{ 0, 0, 0, 1, /**/ 0, 0, 1, 1 };
static constexpr int quad_diag_02_vertex_local[4] = { 0, 1, 2, 2 };
static constexpr int quad_diag_02_face_local[4] = { 0, 1, 1, 2 };

static constexpr int quad_diag_13_deref[4 + 4] =
{ 0, 0, 1, 1, /**/ 0, 1, 1, 0 };
static constexpr int quad_diag_13_vertex_local[4] = { 0, 1, 1, 2 };
static constexpr int quad_diag_13_face_local[4] = { 0, 0, 1, 2 };

static constexpr int quad_diag_both_deref[4 + 4] =
{ 0, 1, 2, 3, /**/ 0, 1, 2, 3 };
static constexpr int quad_diag_both_local[4] = { 0, 0, 0, 0 };
static constexpr Geometry::Type quad_two_triangle_children[2] =
{ Geometry::TRIANGLE, Geometry::TRIANGLE };
static constexpr Geometry::Type quad_four_triangle_children[4] =
{
   Geometry::TRIANGLE, Geometry::TRIANGLE,
   Geometry::TRIANGLE, Geometry::TRIANGLE
};
static constexpr Geometry::Type quad_three_triangle_children[3] =
{ Geometry::TRIANGLE, Geometry::TRIANGLE, Geometry::TRIANGLE };

// Edge i is bisected. Child 0 contains vertex i, child 1 contains vertex
// i+1, and child 2 contains the opposite parent edge (indices modulo four).
static constexpr int quad_edge_deref[4][4 + 4] =
{
   {0, 1, 1, 0, /**/ 0, 1, 2, 0},
   {0, 0, 1, 1, /**/ 0, 0, 1, 2},
   {1, 0, 0, 1, /**/ 2, 0, 0, 1},
   {1, 1, 0, 0, /**/ 1, 2, 0, 0}
};
static constexpr int quad_edge_vertex_local[4][4] =
{
   {0, 1, 2, 2}, {2, 0, 1, 2},
   {2, 2, 0, 1}, {1, 2, 2, 0}
};
static constexpr int quad_edge_face_local[4][4] =
{
   {0, 1, 1, 2}, {2, 0, 1, 1},
   {1, 2, 0, 1}, {1, 1, 2, 0}
};

static constexpr int tri_deref_table[4][3 + 3] =
{
   { 0, 1, 0, /**/ 0, 1, 0 }, // 1 - X
   { 0, 0, 1, /**/ 0, 1, 1 }, // 2 - Y
   { 0, 1, 2, /**/ 0, 1, 2 }, // 3 - iso
   { 0, 0, 1, /**/ 0, 0, 1 }  // 4 - Z
};

static constexpr int tri_corner_0_quad_deref[3 + 3] =
{ 0, 1, 1, /**/ 0, 1, 0 };
static constexpr int tri_corner_0_quad_vertex_local[3] = { 0, 1, 2 };
static constexpr int tri_corner_0_quad_face_local[3] = { 0, 1, 2 };

static constexpr int tri_corner_1_quad_deref[3 + 3] =
{ 1, 0, 1, /**/ 0, 0, 1 };
static constexpr int tri_corner_1_quad_vertex_local[3] = { 0, 1, 3 };
static constexpr int tri_corner_1_quad_face_local[3] = { 0, 1, 3 };

static constexpr int tri_corner_2_quad_deref[3 + 3] =
{ 1, 1, 0, /**/ 1, 0, 0 };
static constexpr int tri_corner_2_quad_vertex_local[3] = { 0, 1, 1 };
static constexpr int tri_corner_2_quad_face_local[3] = { 0, 0, 1 };

static constexpr int tri_center_deref[3 + 3] =
{ 0, 1, 2, /**/ 0, 1, 2 };
static constexpr int tri_center_tris_local[3] = { 0, 0, 0 };
static constexpr int tri_center_quads_vertex_local[3] = { 0, 1, 2 };
static constexpr int tri_center_quads_face_local[3] = { 0, 1, 2 };
static constexpr Geometry::Type tri_corner_quad_children[2] =
{ Geometry::TRIANGLE, Geometry::SQUARE };
static constexpr Geometry::Type tri_three_quad_children[3] =
{ Geometry::SQUARE, Geometry::SQUARE, Geometry::SQUARE };

static constexpr int tet_deref_table[7][4 + 4] =
{
   { 0, 1, 0, 0, /**/ 1, 0, 0, 0 }, // 1 - X, edge (0,1)
   { 0, 0, 1, 0, /**/ 1, 0, 0, 0 }, // 2 - Y, edge (0,2)
   {-1,-1,-1,-1, /**/ -1,-1,-1,-1 }, // 3 - unsupported
   { 0, 0, 0, 1, /**/ 1, 0, 0, 0 }, // 4 - Z, edge (0,3)
   {-1,-1,-1,-1, /**/ -1,-1,-1,-1 }, // 5 - unsupported
   {-1,-1,-1,-1, /**/ -1,-1,-1,-1 }, // 6 - unsupported
   { 0, 1, 2, 3, /**/ 1, 2, 3, 0 }  // 7 - isotropic
};

static constexpr int tet_edge12_deref[4 + 4] =
{ 0, 0, 1, 0, /**/ 0, 1, 0, 0 };

static constexpr int tet_edge13_deref[4 + 4] =
{ 0, 0, 0, 1, /**/ 0, 1, 0, 0 };

static constexpr int tet_edge23_deref[4 + 4] =
{ 0, 0, 0, 1, /**/ 0, 0, 1, 0 };

// Each row contains the corner vertex followed by the ordered vertices of the
// opposite face. The ordering preserves the tetrahedron orientation.
static constexpr int tet_corner_vertex_order[4][4] =
{
   {0, 1, 2, 3}, {1, 0, 3, 2}, {2, 0, 1, 3}, {3, 0, 2, 1}
};

static constexpr int tet_corner_deref[4][4 + 4] =
{
   {0, 1, 1, 1, /**/ 1, 1, 1, 1},
   {1, 0, 1, 1, /**/ 1, 1, 1, 1},
   {1, 1, 0, 1, /**/ 1, 1, 1, 1},
   {1, 1, 1, 0, /**/ 1, 1, 1, 1}
};

static constexpr int tet_corner_vertex_local[4][4] =
{
   {0, 3, 4, 5}, {3, 0, 5, 4}, {3, 4, 0, 5}, {3, 5, 4, 0}
};

static constexpr int tet_corner_face_local[4][4] =
{
   {1, 3, 4, 2}, {3, 1, 2, 4}, {3, 4, 1, 2}, {3, 2, 4, 1}
};

static constexpr Geometry::Type tet_corner_child_geometries[2] =
{ Geometry::TETRAHEDRON, Geometry::PRISM };

static constexpr int tet_all_faces_quads_deref[4 + 4] =
{ 0, 1, 2, 3, /**/ 1, 0, 0, 0 };
static constexpr int tet_all_faces_quads_vertex_local[4] = { 0, 0, 0, 0 };
static constexpr int tet_all_faces_quads_face_local[4] = { 1, 4, 1, 0 };
static constexpr Geometry::Type tet_all_faces_quads_child_geometries[4] =
{ Geometry::CUBE, Geometry::CUBE, Geometry::CUBE, Geometry::CUBE };

static constexpr int hex_deref_table[7][8 + 6] =
{
   { 0, 1, 1, 0, 0, 1, 1, 0, /**/ 1, 1, 1, 0, 0, 0 }, // 1 - X
   { 0, 0, 1, 1, 0, 0, 1, 1, /**/ 0, 0, 0, 1, 1, 1 }, // 2 - Y
   { 0, 1, 2, 3, 0, 1, 2, 3, /**/ 1, 1, 1, 3, 3, 3 }, // 3 - XY
   { 0, 0, 0, 0, 1, 1, 1, 1, /**/ 0, 0, 0, 1, 1, 1 }, // 4 - Z
   { 0, 1, 1, 0, 3, 2, 2, 3, /**/ 1, 1, 1, 3, 3, 3 }, // 5 - XZ
   { 0, 0, 1, 1, 2, 2, 3, 3, /**/ 0, 0, 0, 3, 3, 3 }, // 6 - YZ
   { 0, 1, 2, 3, 4, 5, 6, 7, /**/ 1, 1, 1, 7, 7, 7 }  // 7 - iso
};

static constexpr int prism_deref_table[7][6 + 5] =
{
   {-1,-1,-1,-1,-1,-1, /**/ -1,-1,-1,-1,-1 }, // 1
   {-1,-1,-1,-1,-1,-1, /**/ -1,-1,-1,-1,-1 }, // 2
   { 0, 1, 2, 0, 1, 2, /**/  0, 0, 0, 1, 0 }, // 3 - XY
   { 0, 0, 0, 1, 1, 1, /**/  0, 1, 0, 0, 0 }, // 4 - Z
   {-1,-1,-1,-1,-1,-1, /**/ -1,-1,-1,-1,-1 }, // 5
   {-1,-1,-1,-1,-1,-1, /**/ -1,-1,-1,-1,-1 }, // 6
   { 0, 1, 2, 4, 5, 6, /**/  0, 5, 0, 5, 0 }  // 7 - iso
};

static constexpr int pyramid_deref_table[7][5 + 5] =
{
   {-1,-1,-1,-1,-1, /**/ -1,-1,-1,-1,-1 }, // 1
   {-1,-1,-1,-1,-1, /**/ -1,-1,-1,-1,-1 }, // 2
   {-1,-1,-1,-1,-1, /**/ -1,-1,-1,-1,-1 }, // 3
   {-1,-1,-1,-1,-1, /**/ -1,-1,-1,-1,-1 }, // 4
   {-1,-1,-1,-1,-1, /**/ -1,-1,-1,-1,-1 }, // 5
   {-1,-1,-1,-1,-1, /**/ -1,-1,-1,-1,-1 }, // 6
   { 0, 1, 2, 3, 5, /**/  0, 5, 5, 5, 5 }  // 7 - iso
};

// A pattern ID is interpreted together with the geometry of its parent.
// IDs 1-7 retain their existing meanings, including the X/Y/Z bit masks for
// quadrilaterals and hexahedra. New geometry-specific patterns can use IDs 8+.
// The derefinement map contains one child index per parent vertex, followed by
// one child index per parent face. Segment derefinement is handled directly.
struct NCRefinementPattern
{
   int num_children;
   const int *deref;
   const Geometry::Type *child_geometries = nullptr;
   const int *vertex_local = nullptr;
   const int *face_local = nullptr;

   Geometry::Type ChildGeometry(Geometry::Type parent, int child) const
   {
      MFEM_ASSERT(child >= 0 && child < num_children, "Invalid child index");
      return child_geometries ? child_geometries[child] : parent;
   }

   int VertexLocal(int vertex) const
   { return vertex_local ? vertex_local[vertex] : vertex; }

   int FaceLocal(int face) const
   { return face_local ? face_local[face] : face; }
};

static constexpr Geometry::Type pyramid_child_geometries[10] =
{
   Geometry::PYRAMID, Geometry::PYRAMID, Geometry::PYRAMID,
   Geometry::PYRAMID, Geometry::PYRAMID, Geometry::PYRAMID,
   Geometry::TETRAHEDRON, Geometry::TETRAHEDRON,
   Geometry::TETRAHEDRON, Geometry::TETRAHEDRON
};

static constexpr NCRefinementPattern segment_patterns[8] =
{
   {}, {2, nullptr}, {}, {}, {}, {}, {}, {}
};
static constexpr NCRefinementPattern triangle_patterns[13] =
{
   {}, {2, tri_deref_table[0]}, {2, tri_deref_table[1]},
   {4, tri_deref_table[2]}, {2, tri_deref_table[3]}, {}, {}, {},
   {
      2, tri_corner_0_quad_deref, tri_corner_quad_children,
      tri_corner_0_quad_vertex_local, tri_corner_0_quad_face_local
   },
   {
      2, tri_corner_1_quad_deref, tri_corner_quad_children,
      tri_corner_1_quad_vertex_local, tri_corner_1_quad_face_local
   },
   {
      2, tri_corner_2_quad_deref, tri_corner_quad_children,
      tri_corner_2_quad_vertex_local, tri_corner_2_quad_face_local
   },
   {
      3, tri_center_deref, nullptr,
      tri_center_tris_local, tri_center_tris_local
   },
   {
      3, tri_center_deref, tri_three_quad_children,
      tri_center_quads_vertex_local, tri_center_quads_face_local
   }
};
static constexpr NCRefinementPattern quadrilateral_patterns[15] =
{
   {}, {2, quad_deref_table[0]}, {2, quad_deref_table[1]},
   {4, quad_deref_table[2]}, {}, {}, {}, {},
   {
      2, quad_diag_02_deref, quad_two_triangle_children,
      quad_diag_02_vertex_local, quad_diag_02_face_local
   },
   {
      2, quad_diag_13_deref, quad_two_triangle_children,
      quad_diag_13_vertex_local, quad_diag_13_face_local
   },
   {
      4, quad_diag_both_deref, quad_four_triangle_children,
      quad_diag_both_local, quad_diag_both_local
   },
   {
      3, quad_edge_deref[0], quad_three_triangle_children,
      quad_edge_vertex_local[0], quad_edge_face_local[0]
   },
   {
      3, quad_edge_deref[1], quad_three_triangle_children,
      quad_edge_vertex_local[1], quad_edge_face_local[1]
   },
   {
      3, quad_edge_deref[2], quad_three_triangle_children,
      quad_edge_vertex_local[2], quad_edge_face_local[2]
   },
   {
      3, quad_edge_deref[3], quad_three_triangle_children,
      quad_edge_vertex_local[3], quad_edge_face_local[3]
   }
};
static constexpr NCRefinementPattern tetrahedron_patterns[16] =
{
   {}, {2, tet_deref_table[0]}, {2, tet_deref_table[1]}, {},
   {2, tet_deref_table[3]}, {}, {}, {8, tet_deref_table[6]},
   {2, tet_edge12_deref}, {2, tet_edge13_deref}, {2, tet_edge23_deref},
   {
      2, tet_corner_deref[0], tet_corner_child_geometries,
      tet_corner_vertex_local[0], tet_corner_face_local[0]
   },
   {
      2, tet_corner_deref[1], tet_corner_child_geometries,
      tet_corner_vertex_local[1], tet_corner_face_local[1]
   },
   {
      2, tet_corner_deref[2], tet_corner_child_geometries,
      tet_corner_vertex_local[2], tet_corner_face_local[2]
   },
   {
      2, tet_corner_deref[3], tet_corner_child_geometries,
      tet_corner_vertex_local[3], tet_corner_face_local[3]
   },
   {
      4, tet_all_faces_quads_deref, tet_all_faces_quads_child_geometries,
      tet_all_faces_quads_vertex_local, tet_all_faces_quads_face_local
   }
};
static constexpr NCRefinementPattern hexahedron_patterns[8] =
{
   {}, {2, hex_deref_table[0]}, {2, hex_deref_table[1]},
   {4, hex_deref_table[2]}, {2, hex_deref_table[3]},
   {4, hex_deref_table[4]}, {4, hex_deref_table[5]},
   {8, hex_deref_table[6]}
};
static constexpr NCRefinementPattern prism_patterns[8] =
{
   {}, {}, {}, {4, prism_deref_table[2]},
   {2, prism_deref_table[3]}, {}, {}, {8, prism_deref_table[6]}
};
static constexpr NCRefinementPattern pyramid_patterns[8] =
{
   {}, {}, {}, {}, {}, {}, {},
   {10, pyramid_deref_table[6], pyramid_child_geometries}
};

static const NCRefinementPattern *const
nc_refinement_patterns[Geometry::NumGeom] =
{
   nullptr, segment_patterns, triangle_patterns, quadrilateral_patterns,
   tetrahedron_patterns, hexahedron_patterns, prism_patterns, pyramid_patterns
};

static constexpr int nc_refinement_pattern_counts[Geometry::NumGeom] =
{
   0, sizeof(segment_patterns) / sizeof(segment_patterns[0]),
   sizeof(triangle_patterns) / sizeof(triangle_patterns[0]),
   sizeof(quadrilateral_patterns) / sizeof(quadrilateral_patterns[0]),
   sizeof(tetrahedron_patterns) / sizeof(tetrahedron_patterns[0]),
   sizeof(hexahedron_patterns) / sizeof(hexahedron_patterns[0]),
   sizeof(prism_patterns) / sizeof(prism_patterns[0]),
   sizeof(pyramid_patterns) / sizeof(pyramid_patterns[0])
};

static const NCRefinementPattern &GetNCRefinementPattern(Geometry::Type geom,
                                                         int id)
{
   MFEM_VERIFY(geom >= 0 && geom < Geometry::NumGeom &&
               nc_refinement_patterns[geom] != nullptr && id > 0 &&
               id < nc_refinement_pattern_counts[geom] &&
               nc_refinement_patterns[geom][id].num_children > 0,
               "Invalid NC refinement pattern " << id << " for geometry " << geom);
   return nc_refinement_patterns[geom][id];
}

// child ordering tables

static constexpr char quad_hilbert_child_order[8][4] =
{
   {0,1,2,3}, {0,3,2,1}, {1,2,3,0}, {1,0,3,2},
   {2,3,0,1}, {2,1,0,3}, {3,0,1,2}, {3,2,1,0}
};

static constexpr char quad_hilbert_child_state[8][4] =
{
   {1,0,0,5}, {0,1,1,4}, {3,2,2,7}, {2,3,3,6},
   {5,4,4,1}, {4,5,5,0}, {7,6,6,3}, {6,7,7,2}
};

static constexpr char hex_hilbert_child_order[24][8] =
{
   {0,1,2,3,7,6,5,4}, {0,3,7,4,5,6,2,1}, {0,4,5,1,2,6,7,3},
   {1,0,3,2,6,7,4,5}, {1,2,6,5,4,7,3,0}, {1,5,4,0,3,7,6,2},
   {2,1,5,6,7,4,0,3}, {2,3,0,1,5,4,7,6}, {2,6,7,3,0,4,5,1},
   {3,0,4,7,6,5,1,2}, {3,2,1,0,4,5,6,7}, {3,7,6,2,1,5,4,0},
   {4,0,1,5,6,2,3,7}, {4,5,6,7,3,2,1,0}, {4,7,3,0,1,2,6,5},
   {5,1,0,4,7,3,2,6}, {5,4,7,6,2,3,0,1}, {5,6,2,1,0,3,7,4},
   {6,2,3,7,4,0,1,5}, {6,5,1,2,3,0,4,7}, {6,7,4,5,1,0,3,2},
   {7,3,2,6,5,1,0,4}, {7,4,0,3,2,1,5,6}, {7,6,5,4,0,1,2,3}
};

static constexpr char hex_hilbert_child_state[24][8] =
{
   {1,2,2,7,7,21,21,17},     {2,0,0,22,22,16,16,8},    {0,1,1,15,15,6,6,23},
   {4,5,5,10,10,18,18,14},   {5,3,3,19,19,13,13,11},   {3,4,4,12,12,9,9,20},
   {8,7,7,17,17,23,23,2},    {6,8,8,0,0,15,15,22},     {7,6,6,21,21,1,1,16},
   {11,10,10,14,14,20,20,5}, {9,11,11,3,3,12,12,19},   {10,9,9,18,18,4,4,13},
   {13,14,14,5,5,19,19,10},  {14,12,12,20,20,11,11,4}, {12,13,13,9,9,3,3,18},
   {16,17,17,2,2,22,22,7},   {17,15,15,23,23,8,8,1},   {15,16,16,6,6,0,0,21},
   {20,19,19,11,11,14,14,3}, {18,20,20,4,4,10,10,12},  {19,18,18,13,13,5,5,9},
   {23,22,22,8,8,17,17,0},   {21,23,23,1,1,7,7,15},    {22,21,21,16,16,2,2,6}
};


// child/parent reference domain transforms
using RefCoord = NCMesh::RefCoord;

// reference domain coordinates as fixed point numbers
static constexpr RefCoord T_HALF = (1ll << 59);
static constexpr RefCoord T_ONE = (1ll << 60);
static constexpr RefCoord T_TWO = (1ll << 61);

// (scaling factors have a different fixed point multiplier)
static constexpr RefCoord S_HALF = 1;
static constexpr RefCoord S_ONE = 2;
static constexpr RefCoord S_TWO = 4;

static constexpr RefCoord tri_corners[3][3] =
{
   {    0,     0, 0},
   {T_ONE,     0, 0},
   {    0, T_ONE, 0}
};

static constexpr RefCoord quad_corners[4][3] =
{
   {    0,     0, 0},
   {T_ONE,     0, 0},
   {T_ONE, T_ONE, 0},
   {    0, T_ONE, 0}
};

static constexpr RefCoord hex_corners[8][3] =
{
   {    0,     0,     0},
   {T_ONE,     0,     0},
   {T_ONE, T_ONE,     0},
   {    0, T_ONE,     0},
   {    0,     0, T_ONE},
   {T_ONE,     0, T_ONE},
   {T_ONE, T_ONE, T_ONE},
   {    0, T_ONE, T_ONE}
};

static constexpr RefCoord prism_corners[6][3] =
{
   {    0,     0,     0},
   {T_ONE,     0,     0},
   {    0, T_ONE,     0},
   {    0,     0, T_ONE},
   {T_ONE,     0, T_ONE},
   {    0, T_ONE, T_ONE}
};

static constexpr RefCoord pyramid_corners[5][3] =
{
   {    0,     0,     0},
   {T_ONE,     0,     0},
   {T_ONE, T_ONE,     0},
   {    0, T_ONE,     0},
   {    0,     0, T_ONE}
};

typedef RefCoord RefPoint[3];
static const RefPoint* geom_corners[8] =
{
   NULL, // point
   NULL, // segment
   tri_corners,
   quad_corners,
   NULL, // tetrahedron
   hex_corners,
   prism_corners,
   pyramid_corners
};

// reference domain transform: 3 scales, 3 translations, and optional
// off-diagonal terms for triangle bisections
struct RefTrf
{
   RefCoord s[3], t[3];
   RefCoord xy = 0, yx = 0;

   void Apply(const RefCoord src[3], RefCoord dst[3]) const
   {
      const RefCoord x = src[0], y = src[1], z = src[2];
      dst[0] = ((x*s[0] + y*xy) >> 1) + t[0];
      dst[1] = ((y*s[1] + x*yx) >> 1) + t[1];
      dst[2] = (z*s[2] >> 1) + t[2];
   }
};

static constexpr RefTrf quad_parent_rt1[2] =
{
   { {S_HALF, S_ONE, 0}, {     0, 0, 0} },
   { {S_HALF, S_ONE, 0}, {T_HALF, 0, 0} }
};

static constexpr RefTrf quad_child_rt1[2] =
{
   { {S_TWO, S_ONE, 0}, {     0, 0, 0} },
   { {S_TWO, S_ONE, 0}, {-T_ONE, 0, 0} }
};

static constexpr RefTrf quad_parent_rt2[2] =
{
   { {S_ONE, S_HALF, 0}, {0,      0, 0} },
   { {S_ONE, S_HALF, 0}, {0, T_HALF, 0} }
};

static constexpr RefTrf quad_child_rt2[2] =
{
   { {S_ONE, S_TWO, 0}, {0,      0, 0} },
   { {S_ONE, S_TWO, 0}, {0, -T_ONE, 0} }
};

static constexpr RefTrf quad_parent_rt3[4] =
{
   { {S_HALF, S_HALF, 0}, {     0,      0, 0} },
   { {S_HALF, S_HALF, 0}, {T_HALF,      0, 0} },
   { {S_HALF, S_HALF, 0}, {T_HALF, T_HALF, 0} },
   { {S_HALF, S_HALF, 0}, {     0, T_HALF, 0} }
};

static constexpr RefTrf quad_child_rt3[4] =
{
   { {S_TWO, S_TWO, 0}, {     0,      0, 0} },
   { {S_TWO, S_TWO, 0}, {-T_ONE,      0, 0} },
   { {S_TWO, S_TWO, 0}, {-T_ONE, -T_ONE, 0} },
   { {S_TWO, S_TWO, 0}, {     0, -T_ONE, 0} }
};

// Diagonal (0,2): (0,1,2) and (0,2,3).
static constexpr RefTrf quad_parent_rt8[2] =
{
   { {S_ONE, S_ONE, 0}, {0, 0, 0}, S_ONE, 0 },
   { {S_ONE, S_ONE, 0}, {0, 0, 0}, 0, S_ONE }
};
static constexpr RefTrf quad_child_rt8[2] =
{
   { {S_ONE, S_ONE, 0}, {0, 0, 0}, -S_ONE, 0 },
   { {S_ONE, S_ONE, 0}, {0, 0, 0}, 0, -S_ONE }
};

// Diagonal (1,3): (0,1,3) and (1,2,3).
static constexpr RefTrf quad_parent_rt9[2] =
{
   { {S_ONE, S_ONE, 0}, {0, 0, 0} },
   { {0, S_ONE, 0}, {T_ONE, 0, 0}, -S_ONE, S_ONE }
};
static constexpr RefTrf quad_child_rt9[2] =
{
   { {S_ONE, S_ONE, 0}, {0, 0, 0} },
   { {S_ONE, 0, 0}, {-T_ONE, T_ONE, 0}, S_ONE, -S_ONE }
};

// Both diagonals, with a new vertex at (1/2,1/2).
static constexpr RefTrf quad_parent_rt10[4] =
{
   { { S_ONE,  S_HALF, 0}, {0, 0, 0},  S_HALF, 0 },
   { {     0,  S_HALF, 0}, {T_ONE, 0, 0}, -S_HALF,  S_ONE },
   { {-S_ONE, -S_HALF, 0}, {T_ONE, T_ONE, 0}, -S_HALF, 0 },
   { {     0, -S_HALF, 0}, {0, T_ONE, 0}, S_HALF, -S_ONE }
};
static constexpr RefTrf quad_child_rt10[4] =
{
   { { S_ONE,  S_TWO, 0}, {0, 0, 0}, -S_ONE, 0 },
   { { S_ONE,      0, 0}, {-T_ONE, T_TWO, 0}, S_ONE, -S_TWO },
   { {-S_ONE,      0, 0}, {0, T_TWO, 0}, S_ONE, -S_TWO },
   { {-S_ONE,      0, 0}, {T_ONE, 0, 0}, -S_ONE, S_TWO }
};

static const RefTrf* quad_parent[11] =
{
   NULL,
   quad_parent_rt1,
   quad_parent_rt2,
   quad_parent_rt3,
   NULL, NULL, NULL, NULL,
   quad_parent_rt8,
   quad_parent_rt9,
   quad_parent_rt10
};

static const RefTrf* quad_child[11] =
{
   NULL,
   quad_child_rt1,
   quad_child_rt2,
   quad_child_rt3,
   NULL, NULL, NULL, NULL,
   quad_child_rt8,
   quad_child_rt9,
   quad_child_rt10
};

static constexpr RefTrf hex_parent_rt1[2] =
{
   { {S_HALF, S_ONE, S_ONE}, {     0, 0, 0} },
   { {S_HALF, S_ONE, S_ONE}, {T_HALF, 0, 0} }
};

static constexpr RefTrf hex_child_rt1[2] =
{
   { {S_TWO, S_ONE, S_ONE}, {     0, 0, 0} },
   { {S_TWO, S_ONE, S_ONE}, {-T_ONE, 0, 0} }
};

static constexpr RefTrf hex_parent_rt2[2] =
{
   { {S_ONE, S_HALF, S_ONE}, {0,      0, 0} },
   { {S_ONE, S_HALF, S_ONE}, {0, T_HALF, 0} }
};

static constexpr RefTrf hex_child_rt2[2] =
{
   { {S_ONE, S_TWO, S_ONE}, {0,      0, 0} },
   { {S_ONE, S_TWO, S_ONE}, {0, -T_ONE, 0} }
};

static constexpr RefTrf hex_parent_rt3[4] =
{
   { {S_HALF, S_HALF, S_ONE}, {     0,      0, 0} },
   { {S_HALF, S_HALF, S_ONE}, {T_HALF,      0, 0} },
   { {S_HALF, S_HALF, S_ONE}, {T_HALF, T_HALF, 0} },
   { {S_HALF, S_HALF, S_ONE}, {     0, T_HALF, 0} }
};

static constexpr RefTrf hex_child_rt3[4] =
{
   { {S_TWO, S_TWO, S_ONE}, {     0,      0, 0} },
   { {S_TWO, S_TWO, S_ONE}, {-T_ONE,      0, 0} },
   { {S_TWO, S_TWO, S_ONE}, {-T_ONE, -T_ONE, 0} },
   { {S_TWO, S_TWO, S_ONE}, {     0, -T_ONE, 0} }
};

static constexpr RefTrf hex_parent_rt4[2] =
{
   { {S_ONE, S_ONE, S_HALF}, {0, 0,      0} },
   { {S_ONE, S_ONE, S_HALF}, {0, 0, T_HALF} }
};

static constexpr RefTrf hex_child_rt4[2] =
{
   { {S_ONE, S_ONE, S_TWO}, {0, 0,      0} },
   { {S_ONE, S_ONE, S_TWO}, {0, 0, -T_ONE} }
};

static constexpr RefTrf hex_parent_rt5[4] =
{
   { {S_HALF, S_ONE, S_HALF}, {     0, 0,      0} },
   { {S_HALF, S_ONE, S_HALF}, {T_HALF, 0,      0} },
   { {S_HALF, S_ONE, S_HALF}, {T_HALF, 0, T_HALF} },
   { {S_HALF, S_ONE, S_HALF}, {     0, 0, T_HALF} }
};

static constexpr RefTrf hex_child_rt5[4] =
{
   { {S_TWO, S_ONE, S_TWO}, {     0, 0,      0} },
   { {S_TWO, S_ONE, S_TWO}, {-T_ONE, 0,      0} },
   { {S_TWO, S_ONE, S_TWO}, {-T_ONE, 0, -T_ONE} },
   { {S_TWO, S_ONE, S_TWO}, {     0, 0, -T_ONE} }
};

static constexpr RefTrf hex_parent_rt6[4] =
{
   { {S_ONE, S_HALF, S_HALF}, {0,      0,      0} },
   { {S_ONE, S_HALF, S_HALF}, {0, T_HALF,      0} },
   { {S_ONE, S_HALF, S_HALF}, {0,      0, T_HALF} },
   { {S_ONE, S_HALF, S_HALF}, {0, T_HALF, T_HALF} }
};

static constexpr RefTrf hex_child_rt6[4] =
{
   { {S_ONE, S_TWO, S_TWO}, {0,      0,      0} },
   { {S_ONE, S_TWO, S_TWO}, {0, -T_ONE,      0} },
   { {S_ONE, S_TWO, S_TWO}, {0,      0, -T_ONE} },
   { {S_ONE, S_TWO, S_TWO}, {0, -T_ONE, -T_ONE} }
};

static constexpr RefTrf hex_parent_rt7[8] =
{
   { {S_HALF, S_HALF, S_HALF}, {     0,      0,      0} },
   { {S_HALF, S_HALF, S_HALF}, {T_HALF,      0,      0} },
   { {S_HALF, S_HALF, S_HALF}, {T_HALF, T_HALF,      0} },
   { {S_HALF, S_HALF, S_HALF}, {     0, T_HALF,      0} },
   { {S_HALF, S_HALF, S_HALF}, {     0,      0, T_HALF} },
   { {S_HALF, S_HALF, S_HALF}, {T_HALF,      0, T_HALF} },
   { {S_HALF, S_HALF, S_HALF}, {T_HALF, T_HALF, T_HALF} },
   { {S_HALF, S_HALF, S_HALF}, {     0, T_HALF, T_HALF} }
};

static constexpr RefTrf hex_child_rt7[8] =
{
   { {S_TWO, S_TWO, S_TWO}, {     0,      0,      0} },
   { {S_TWO, S_TWO, S_TWO}, {-T_ONE,      0,      0} },
   { {S_TWO, S_TWO, S_TWO}, {-T_ONE, -T_ONE,      0} },
   { {S_TWO, S_TWO, S_TWO}, {     0, -T_ONE,      0} },
   { {S_TWO, S_TWO, S_TWO}, {     0,      0, -T_ONE} },
   { {S_TWO, S_TWO, S_TWO}, {-T_ONE,      0, -T_ONE} },
   { {S_TWO, S_TWO, S_TWO}, {-T_ONE, -T_ONE, -T_ONE} },
   { {S_TWO, S_TWO, S_TWO}, {     0, -T_ONE, -T_ONE} }
};

static const RefTrf* hex_parent[8] =
{
   NULL,
   hex_parent_rt1,
   hex_parent_rt2,
   hex_parent_rt3,
   hex_parent_rt4,
   hex_parent_rt5,
   hex_parent_rt6,
   hex_parent_rt7
};

static const RefTrf* hex_child[8] =
{
   NULL,
   hex_child_rt1,
   hex_child_rt2,
   hex_child_rt3,
   hex_child_rt4,
   hex_child_rt5,
   hex_child_rt6,
   hex_child_rt7
};

// X, Y, and Z bisect edges (0,1), (0,2), and (1,2), respectively.
// Shears map the bisection children onto the reference triangle.
static constexpr RefTrf tri_parent_rt1[2] =
{
   { {S_HALF, S_ONE, 0}, {     0, 0, 0} },
   { {S_HALF, S_ONE, 0}, {T_HALF, 0, 0}, -S_HALF, 0 }
};

static constexpr RefTrf tri_child_rt1[2] =
{
   { {S_TWO, S_ONE, 0}, {     0, 0, 0} },
   { {S_TWO, S_ONE, 0}, {-T_ONE, 0, 0}, S_ONE, 0 }
};

static constexpr RefTrf tri_parent_rt2[2] =
{
   { {S_ONE, S_HALF, 0}, {0,      0, 0} },
   { {S_ONE, S_HALF, 0}, {0, T_HALF, 0}, 0, -S_HALF }
};

static constexpr RefTrf tri_child_rt2[2] =
{
   { {S_ONE, S_TWO, 0}, {0,      0, 0} },
   { {S_ONE, S_TWO, 0}, {0, -T_ONE, 0}, 0, S_ONE }
};

static constexpr RefTrf tri_parent_rt3[4] =
{
   { { S_HALF,  S_HALF, 0}, {     0,      0, 0} },
   { { S_HALF,  S_HALF, 0}, {T_HALF,      0, 0} },
   { { S_HALF,  S_HALF, 0}, {     0, T_HALF, 0} },
   { {-S_HALF, -S_HALF, 0}, {T_HALF, T_HALF, 0} }
};

static constexpr RefTrf tri_child_rt3[4] =
{
   { { S_TWO,  S_TWO, 0}, {     0,      0, 0} },
   { { S_TWO,  S_TWO, 0}, {-T_ONE,      0, 0} },
   { { S_TWO,  S_TWO, 0}, {     0, -T_ONE, 0} },
   { {-S_TWO, -S_TWO, 0}, { T_ONE,  T_ONE, 0} }
};

static constexpr RefTrf tri_parent_rt4[2] =
{
   { {S_ONE, S_HALF, 0}, {0, 0, 0}, S_HALF, 0 },
   { {S_HALF, S_ONE, 0}, {0, 0, 0}, 0, S_HALF }
};

static constexpr RefTrf tri_child_rt4[2] =
{
   { {S_ONE, S_TWO, 0}, {0, 0, 0}, -S_ONE, 0 },
   { {S_TWO, S_ONE, 0}, {0, 0, 0}, 0, -S_ONE }
};

static const RefTrf* tri_parent[8] =
{
   NULL, tri_parent_rt1, tri_parent_rt2,
   tri_parent_rt3, tri_parent_rt4,
   NULL, NULL, NULL
};

static const RefTrf* tri_child[8] =
{
   NULL, tri_child_rt1, tri_child_rt2,
   tri_child_rt3, tri_child_rt4,
   NULL, NULL, NULL
};

static constexpr RefTrf prism_parent_rt3[4] =
{
   { { S_HALF,  S_HALF, S_ONE}, {     0,      0, 0} },
   { { S_HALF,  S_HALF, S_ONE}, {T_HALF,      0, 0} },
   { { S_HALF,  S_HALF, S_ONE}, {     0, T_HALF, 0} },
   { {-S_HALF, -S_HALF, S_ONE}, {T_HALF, T_HALF, 0} }
};

static constexpr RefTrf prism_child_rt3[4] =
{
   { { S_TWO,  S_TWO, S_ONE}, {     0,      0, 0} },
   { { S_TWO,  S_TWO, S_ONE}, {-T_ONE,      0, 0} },
   { { S_TWO,  S_TWO, S_ONE}, {     0, -T_ONE, 0} },
   { {-S_TWO, -S_TWO, S_ONE}, { T_ONE,  T_ONE, 0} }
};

static constexpr RefTrf prism_parent_rt4[2] =
{
   { {S_ONE, S_ONE, S_HALF}, {0, 0,      0} },
   { {S_ONE, S_ONE, S_HALF}, {0, 0, T_HALF} }
};

static constexpr RefTrf prism_child_rt4[2] =
{
   { {S_ONE, S_ONE, S_TWO}, {0, 0,      0} },
   { {S_ONE, S_ONE, S_TWO}, {0, 0, -T_ONE} }
};

static constexpr RefTrf prism_parent_rt7[8] =
{
   { { S_HALF,  S_HALF, S_HALF}, {     0,      0,      0} },
   { { S_HALF,  S_HALF, S_HALF}, {T_HALF,      0,      0} },
   { { S_HALF,  S_HALF, S_HALF}, {     0, T_HALF,      0} },
   { {-S_HALF, -S_HALF, S_HALF}, {T_HALF, T_HALF,      0} },
   { { S_HALF,  S_HALF, S_HALF}, {     0,      0, T_HALF} },
   { { S_HALF,  S_HALF, S_HALF}, {T_HALF,      0, T_HALF} },
   { { S_HALF,  S_HALF, S_HALF}, {     0, T_HALF, T_HALF} },
   { {-S_HALF, -S_HALF, S_HALF}, {T_HALF, T_HALF, T_HALF} }
};

static constexpr RefTrf prism_child_rt7[8] =
{
   { { S_TWO,  S_TWO, S_TWO}, {     0,      0,      0} },
   { { S_TWO,  S_TWO, S_TWO}, {-T_ONE,      0,      0} },
   { { S_TWO,  S_TWO, S_TWO}, {     0, -T_ONE,      0} },
   { {-S_TWO, -S_TWO, S_TWO}, { T_ONE,  T_ONE,      0} },
   { { S_TWO,  S_TWO, S_TWO}, {     0,      0, -T_ONE} },
   { { S_TWO,  S_TWO, S_TWO}, {-T_ONE,      0, -T_ONE} },
   { { S_TWO,  S_TWO, S_TWO}, {     0, -T_ONE, -T_ONE} },
   { {-S_TWO, -S_TWO, S_TWO}, { T_ONE,  T_ONE, -T_ONE} }
};

static const RefTrf* prism_parent[8] =
{
   NULL, NULL, NULL,
   prism_parent_rt3,
   prism_parent_rt4,
   NULL, NULL,
   prism_parent_rt7
};

static const RefTrf* prism_child[8] =
{
   NULL, NULL, NULL,
   prism_child_rt3,
   prism_child_rt4,
   NULL, NULL,
   prism_child_rt7
};

static const RefTrf** geom_parent[7] =
{
   NULL,
   NULL,
   tri_parent,
   quad_parent,
   NULL,
   hex_parent,
   prism_parent
};

static const RefTrf** geom_child[7] =
{
   NULL,
   NULL,
   tri_child,
   quad_child,
   NULL,
   hex_child,
   prism_child
};

} // namespace mfem

#endif // MFEM_NCMESH_TABLES
