// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_GMSH
#define MFEM_GMSH

namespace mfem
{

// Helpers for reading high order elements in Gmsh format

/** @name Gmsh High-Order Vertex Mappings

    These functions generate the mappings needed to translate the order of
    Gmsh's high-order vertices into MFEM's L2 degree of freedom ordering. The
    mapping is defined so that MFEM_DoF[i] = Gmsh_Vert[map[i]]. The @a map
    array must already be allocated with the proper number of entries for the
    element type at the given element @a order.
*/
///@{

/// @brief Generate Gmsh vertex mapping for a Segment
void GmshHOSegmentMapping(int order, int *map);

/// @brief Generate Gmsh vertex mapping for a Triangle
void GmshHOTriangleMapping(int order, int *map);

/// @brief Generate Gmsh vertex mapping for a Quadrilateral
void GmshHOQuadrilateralMapping(int order, int *map);

/// @brief Generate Gmsh vertex mapping for a Tetrahedron
void GmshHOTetrahedronMapping(int order, int *map);

/// @brief Generate Gmsh vertex mapping for a Hexahedron
void GmshHOHexahedronMapping(int order, int *map);

/// @brief Generate Gmsh vertex mapping for a Wedge
void GmshHOWedgeMapping(int order, int *map);

/// @brief Generate Gmsh vertex mapping for a Pyramid
void GmshHOPyramidMapping(int order, int *map);

///@}

} // namespace mfem

#endif
