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

#ifndef MFEM_FACE_NBR_GEOM
#define MFEM_FACE_NBR_GEOM

#include "../config/config.hpp"
#include "mesh.hpp"
#include "pmesh.hpp"

namespace mfem
{

/// @brief Class for accessing the geometric factors of face neighbor elements
/// (i.e. across boundaries of MPI mesh partitions).
///
/// @sa GeometricFactors
class FaceNeighborGeometricFactors
{
public:
   int num_neighbor_elems; ///< Number of face neighbor elements.

   /// @name Geometric factor data arrays
   /// These are stored with layout (NQ, VDIM, NE). See the documentation of
   /// GeometricFactors for more details.
   ///@{

   Vector X; ///< Physical coordinates of the mesh.
   Vector J; ///< Jacobian matrices
   Vector detJ; ///< Jacobian determinants

   ///@}

   /// Communicate (if needed) to gather the face neighbor geometric factors.
   FaceNeighborGeometricFactors(const GeometricFactors &geom_);

protected:
   const GeometricFactors &geom; ///< The GeometricFactors of the Mesh.

   /// @name Internal work arrays, used for MPI communication
   ///@{

   Vector send_data;
   Array<int> send_offsets, recv_offsets;

   ///@}

   /// @brief Given a Q-vector @a x_local with @a vdim components, fill the
   /// face-neighbor Q-vector @a x_shared by communicating with neighboring MPI
   /// partitions.
   void ExchangeFaceNbrQVectors(const Vector &x_local, Vector &x_shared,
                                const int vdim);
};

} // namespace mfem

#endif
