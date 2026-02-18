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

/// @brief Given a face DOF index in native (counter-clockwise) ordering, return
/// the corresponding DOF index in lexicographic ordering (for a quadrilateral
/// element).
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

/// @brief Given a face DOF index on a shared face, ordered lexicographically
/// relative to the element (where the local face is face_id), return the
/// corresponding face DOF index ordered lexicographically relative to the face
/// itself.
MFEM_HOST_DEVICE
inline int PermuteFace2D(const int face_id, const int orientation,
                         const int size1d, const int index)
{
   int new_index;
   // Convert from element 1 lex ordering to native ordering
   if (face_id == 2 || face_id == 3)
   {
      new_index = size1d-1-index;
   }
   else
   {
      new_index = index;
   }
   // Permute based on face orientations
   if (orientation == 1)
   {
      new_index = size1d-1-new_index;
   }
   return new_index;
}

/// @brief Given a face DOF index on a shared face, ordered lexicographically
/// relative to element 1, return the corresponding face DOF index ordered
/// lexicographically relative to element 2.
MFEM_HOST_DEVICE
inline int PermuteFace2D(const int face_id1, const int face_id2,
                         const int orientation, const int size1d,
                         const int index)
{
   const int new_index = PermuteFace2D(face_id1, orientation, size1d, index);
   return ToLexOrdering2D(face_id2, size1d, new_index);
}

/// @brief Given a face DOF index in native (counter-clockwise) ordering, return
/// the corresponding DOF index in lexicographic ordering (for a hexahedral
/// element).
MFEM_HOST_DEVICE
inline int ToLexOrdering3D(const int face_id, const int size1d, const int i,
                           const int j)
{
   if (face_id==2 || face_id==1 || face_id==5)
   {
      return i + j*size1d;
   }
   else if (face_id==3 || face_id==4)
   {
      return (size1d-1-i) + j*size1d;
   }
   else // face_id==0
   {
      return i + (size1d-1-j)*size1d;
   }
}

/// @brief Given the index of a face DOF in lexicographic ordering relative the
/// element (where the local face id is @a face_id), permute the index so that
/// it is lexicographically ordered relative to the face itself.
MFEM_HOST_DEVICE
inline int PermuteFace3D(const int face_id, const int orientation,
                         const int size1d, const int index)
{
   int i=0, j=0, new_i=0, new_j=0;
   i = index%size1d;
   j = index/size1d;
   // Convert from lex ordering
   if (face_id==3 || face_id==4)
   {
      i = size1d-1-i;
   }
   else if (face_id==0)
   {
      j = size1d-1-j;
   }
   // Permute based on face orientations
   switch (orientation)
   {
      case 0:
         new_i = i;
         new_j = j;
         break;
      case 1:
         new_i = j;
         new_j = i;
         break;
      case 2:
         new_i = j;
         new_j = (size1d-1-i);
         break;
      case 3:
         new_i = (size1d-1-i);
         new_j = j;
         break;
      case 4:
         new_i = (size1d-1-i);
         new_j = (size1d-1-j);
         break;
      case 5:
         new_i = (size1d-1-j);
         new_j = (size1d-1-i);
         break;
      case 6:
         new_i = (size1d-1-j);
         new_j = i;
         break;
      case 7:
         new_i = i;
         new_j = (size1d-1-j);
         break;
   }
   return new_i + new_j*size1d;
}

/// @brief Given the index of a face DOF in lexicographic ordering relative
/// element 1, permute the index so that it is lexicographically ordered
/// relative to element 2.
///
/// The given face corresponds to local face index @a face_id1 relative to
/// element 1, and @a face_id2 (with @a orientation) relative to element 2.
MFEM_HOST_DEVICE
inline int PermuteFace3D(const int face_id1, const int face_id2,
                         const int orientation,
                         const int size1d, const int index)
{
   const int new_index = PermuteFace3D(face_id1, orientation, size1d, index);
   const int new_i = new_index%size1d;
   const int new_j = new_index/size1d;
   return ToLexOrdering3D(face_id2, size1d, new_i, new_j);
}

/// @brief Given a face DOF (or quadrature) index ordered lexicographically
/// relative to element 1, return the associated (i, j) coordinates.
///
/// The returned coordinates will be relative to element 1 or element 2
/// according to the value of side (side == 0 corresponds element 1).
MFEM_HOST_DEVICE
inline void FaceIdxToVolIdx2D(const int qi, const int nq, const int face_id0,
                              const int face_id1, const int side, int &i, int &j)
{
   // Note: in 2D, a consistently ordered mesh will always have the element 2
   // face reversed relative to element 1, so orientation is determined entirely
   // by side. (In 3D, separate orientation information is needed).
   const int orientation = side;

   const int face_id = (side == 0) ? face_id0 : face_id1;
   const int edge_idx = (side == 0) ? qi : PermuteFace2D(face_id0, face_id1,
                                                         orientation, nq, qi);

   const int level = (face_id == 0 || face_id == 3) ? 0 : (nq-1);
   const bool x_axis = (face_id == 0 || face_id == 2);

   i = x_axis ? edge_idx : level;
   j = x_axis ? level : edge_idx;
}

/// @brief Given a face DOF (or quadrature) index ordered lexicographically
/// relative to element 1, return the associated (i, j, k) coordinates.
///
/// The returned coordinates will be relative to element 1 or element 2
/// according to the value of side (side == 0 corresponds element 1).
MFEM_HOST_DEVICE
inline void FaceIdxToVolIdx3D(const int index, const int size1d,
                              const int face_id0, const int face_id1,
                              const int side, const int orientation,
                              int& i, int& j, int& k)
{
   MFEM_VERIFY_KERNEL(face_id1 >= 0 || side == 0,
                      "Accessing second side but face_id1 is not valid.");

   const int face_id = (side == 0) ? face_id0 : face_id1;
   const int fidx = (side == 0) ? index
                    : PermuteFace3D(face_id0, face_id1, orientation, size1d, index);

   const bool xy_plane = (face_id == 0 || face_id == 5);
   const bool yz_plane = (face_id == 2 || face_id == 4);

   const int level = (face_id == 0 || face_id == 1 || face_id == 4)
                     ? 0 : (size1d-1);

   const int _i = fidx % size1d;
   const int _j = fidx / size1d;

   k = xy_plane ? level : _j;
   j = yz_plane ? _i : xy_plane ? _j : level;
   i = yz_plane ? level : _i;
}

MFEM_HOST_DEVICE
inline int FaceIdxToVolIdx(int dim, int i, int size1d, int face0, int face1,
                           int side, int orientation)
{
   if (dim == 2)
   {
      int ix, iy;
      internal::FaceIdxToVolIdx2D(i, size1d, face0, face1, side, ix, iy);
      return ix + iy*size1d;
   }
   else if (dim == 3)
   {
      int ix, iy, iz;
      internal::FaceIdxToVolIdx3D(i, size1d, face0, face1, side, orientation,
                                  ix, iy, iz);
      return ix + size1d*iy + size1d*size1d*iz;
   }
   else
   {
      MFEM_ABORT_KERNEL("Invalid dimension");
      return -1;
   }
};

} // namespace internal

} // namespace mfem

#endif
