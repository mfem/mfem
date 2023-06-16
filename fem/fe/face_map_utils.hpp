// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_FACE_MAP_UTILS_HPP
#define MFEM_FACE_MAP_UTILS_HPP

#include "../../general/array.hpp"
#include "../../general/backends.hpp"
#include <utility> // std::pair
#include <vector>

namespace mfem
{

namespace internal
{

/// Each face of a hexahedron is given by a level set x_i = l, where x_i is one
/// of x, y, or z (corresponding to i=0, i=1, i=2), and l is either 0 or 1.
/// Returns i and level.
std::pair<int,int> GetFaceNormal3D(const int face_id);

/// @brief Fills in the entries of the lexicographic face_map.
///
/// For use in FiniteElement::GetFaceMap.
///
/// n_face_dofs_per_component is the number of DOFs for each vector component
/// on the face (there is only one vector component in all cases except for 3D
/// Nedelec elements, where the face DOFs have two components to span the
/// tangent space).
///
/// The DOFs for the i-th vector component begin at offsets[i] (i.e. the number
/// of vector components is given by offsets.size()).
///
/// The DOFs for each vector component are arranged in a Cartesian grid defined
/// by strides and n_dofs_per_dim.
void FillFaceMap(const int n_face_dofs_per_component,
                 const std::vector<int> &offsets,
                 const std::vector<int> &strides,
                 const std::vector<int> &n_dofs_per_dim,
                 Array<int> &face_map);

/// Return the face map for nodal tensor elements (H1, L2, and Bernstein basis).
void GetTensorFaceMap(const int dim, const int order, const int face_id,
                      Array<int> &face_map);

// maps face index (in counter-clockwise order) to Lexocographic index
MFEM_HOST_DEVICE
inline int ToLexOrdering2D(const int face_id, const int size1d, const int i)
{
   if (face_id==2 || face_id==3)
   {
      return size1d-1-i;
   }
   else
   {
      return i;
   }
}

// permutes face index from native ordering to lexocographic
MFEM_HOST_DEVICE
inline int PermuteFace2D(const int face_id1, const int face_id2,
                         const int orientation,
                         const int size1d, const int index)
{
   int new_index;
   // Convert from lex ordering
   if (face_id1==2 || face_id1==3)
   {
      new_index = size1d-1-index;
   }
   else
   {
      new_index = index;
   }
   // Permute based on face orientations
   if (orientation==1)
   {
      new_index = size1d-1-new_index;
   }
   return ToLexOrdering2D(face_id2, size1d, new_index);
}

// maps quadrature index on face to (row, col) index pair
MFEM_HOST_DEVICE
inline std::pair<int, int> EdgeQuad2Lex2D(const int qi, const int nq,
                                        const int face_id0, const int face_id1, const int side)
{
   const int face_id = (side == 0) ? face_id0 : face_id1;
   const int edge_idx = (side == 0) ? qi : PermuteFace2D(face_id0, face_id1, side,
                                                         nq, qi);
   int i, j;
   if (face_id == 0 || face_id == 2)
   {
      i = edge_idx;
      j = (face_id == 0) ? 0 : (nq-1);
   }
   else
   {
      j = edge_idx;
      i = (face_id == 3) ? 0 : (nq-1);
   }

   return std::make_pair(i, j);
}

} // namespace internal

} // namespace mfem

#endif
