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

#ifndef MFEM_MESH_TEST_UTILS
#define MFEM_MESH_TEST_UTILS


#include "mfem.hpp"
#include "unit_tests.hpp"

namespace mfem
{

/**
 * @brief Helper function for performing an H1 Poisson solve on a serial mesh, with
 * homogeneous essential boundary conditions. Optionally can disable a boundary.
 *
 * @param mesh The SERIAL mesh to perform the Poisson solve on
 * @param order The polynomial order of the basis
 * @param disabled_boundary_attribute Optional boundary attribute to NOT apply
 * homogeneous Dirichlet boundary condition on. Default of -1 means no boundary
 * is disabled.
 * @return int The number of DOF that are fixed by the essential boundary condition.
 */
int CheckPoisson(Mesh &mesh, int order, int disabled_boundary_attribute = -1);

#ifdef MFEM_USE_MPI

/**
 * @brief Helper function for performing an H1 Poisson solve on a parallel mesh, with
 * homogeneous essential boundary conditions. Optionally can disable a boundary.
 *
 * @param mesh The PARALLEL mesh to perform the Poisson solve on
 * @param order The polynomial order of the basis
 * @param disabled_boundary_attribute Optional boundary attribute to NOT apply
 * homogeneous Dirichlet boundary condition on. Default of -1 means no boundary
 * is disabled.
 * @return int The number of DOF that are fixed by the essential boundary condition.
 */
void CheckPoisson(ParMesh &pmesh, int order,
                  int disabled_boundary_attribute = -1);

/**
 * @brief Check that a Parmesh generates the same number of boundary elements as
 * the serial mesh.
 *
 * @param smesh Serial mesh to be built from and compared against
 * @param partition Optional partition
 * @return std::unique_ptr<ParMesh> Pointer to the mesh in question.
 */
std::unique_ptr<ParMesh> CheckParMeshNBE(Mesh &smesh,
                                         const std::unique_ptr<int[]> &partition = nullptr);

/**
 * @brief Helper function to track if a face index is internal
 *
 * @param pmesh The mesh containing the face
 * @param f The face index
 * @param local_to_shared A map from local faces to shared faces
 * @return true the face is between domain attributes (and owned by this rank)
 * @return false the face is not between domain attributes or not owned by this rank
 */
bool CheckFaceInternal(ParMesh& pmesh, int f,
                       const std::map<int, int> &local_to_shared);

#endif

} // namespace mfem

#endif // MFEM_MESH_TEST_UTILS