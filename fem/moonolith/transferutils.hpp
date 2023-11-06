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

#ifndef MFEM_L2P_MESH_UTILS_HPP
#define MFEM_L2P_MESH_UTILS_HPP

#include "../../mesh/hexahedron.hpp"
#include "../../mesh/quadrilateral.hpp"
#include "../fem.hpp"

namespace mfem
{

// These methods are to be used exclusively inside the routines inside the
// transfer functionalities
namespace internal
{

/*!
 * @brief Creates a new element based on the type and cell data.
 * @param type The element type (Geometry::TRIANGLE,  Geometry::TETRAHEDRON,
 * Geometry::SQUARE, Geometry::CUBE).
 * @param cells_data The element connectivity.
 * @param attr The element attribute.
 */
Element *NewElem(const int type, const int *cells_data, const int attr);

/*!
 * @brief Finalizes the mesh based on the element type
 * @param mesh The mesh.
 * @param generate_edges  True if the generation of edges is requested, false if
 * not.
 */
void Finalize(Mesh &mesh, const bool generate_edges);

/*!
 * @brief Computes a column vector containing the maximum element for each row
 * @param mat The matrix
 * @param vec[out] The vector where we store the result
 * @param include_vec_elements True if we consider vec as an additional column
 * of the matrix, False otherwise.
 */
void MaxCol(const DenseMatrix &mat, double *vec, bool include_vec_elements);

/*!
 * @brief Computes a column vector containing the minimum element for each row
 * @param mat The matrix
 * @param vec[out] The vector where we store the result
 * @param include_vec_elements True if we consider vec as an additional column
 * of the matrix, False otherwise.
 */
void MinCol(const DenseMatrix &mat, double *vec, bool include_vec_elements);

/*!
 * @brief Returns the maximum number of vertices in a face.
 * @param type The element type of the face.
 * @return The number of vertices.
 */
int MaxVertsXFace(const int type);

/*!
 * @brief Computes the sum of the matrix entries.
 * @param mat The matrix.
 * @return The sum of the elements of the matrix.
 */
double Sum(const DenseMatrix &mat);
} // namespace internal

} // namespace mfem

#endif // MFEM_L2P_MESH_UTILS_HPP
