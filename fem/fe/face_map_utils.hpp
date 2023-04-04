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

} // namespace internal

} // namespace mfem

#endif
