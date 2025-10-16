// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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

static constexpr int ref_type_num_children[8] = { 0, 2, 2, 4, 2, 4, 4, 8 };

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

// reference domain transform: 3 scales, 3 translations
struct RefTrf
{
   RefCoord s[3], t[3];

   void Apply(const RefCoord src[3], RefCoord dst[3]) const
   {
      for (int i = 0; i < 3; i++)
      {
         dst[i] = (src[i]*s[i] >> 1) + t[i];
      }
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

static const RefTrf* quad_parent[4] =
{
   NULL,
   quad_parent_rt1,
   quad_parent_rt2,
   quad_parent_rt3
};

static const RefTrf* quad_child[4] =
{
   NULL,
   quad_child_rt1,
   quad_child_rt2,
   quad_child_rt3
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

static const RefTrf* tri_parent[4] =
{
   NULL, NULL, NULL,
   tri_parent_rt3
};

static const RefTrf* tri_child[4] =
{
   NULL, NULL, NULL,
   tri_child_rt3
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