// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_L2P_MESH_UTILS_HPP
#define MFEM_L2P_MESH_UTILS_HPP

#include "../fem/fem.hpp"
#include "../mesh/hexahedron.hpp"
#include "../mesh/quadrilateral.hpp"

namespace mfem
{

// These methods are to be used exclusively inside the routines inside the
// transfer functionalties
namespace private_
{

Element *NewElem(const int type, const int *cells_data, const int attr);
void Finalize(Mesh &mesh, const bool generate_edges);

void MaxCol(const DenseMatrix &mat, double *vec, bool include_vec_elements);
void MinCol(const DenseMatrix &mat, double *vec, bool include_vec_elements);

int MaxVertsXFace(const int type);
double Sum(const DenseMatrix &mat);
} // namespace private_

} // namespace mfem

#endif // MFEM_L2P_MESH_UTILS_HPP
