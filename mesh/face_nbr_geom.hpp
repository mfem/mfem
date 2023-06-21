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

#ifndef MFEM_FACE_NBR_GEOM
#define MFEM_FACE_NBR_GEOM

#include "../config/config.hpp"
#include "mesh.hpp"
#include "pmesh.hpp"

namespace mfem
{

class FaceNeighborGeometricFactors
{
public:
   const GeometricFactors &geom;
   int num_neighbor_elems = 0;
   Vector X;
   Vector J;
   Vector detJ;

   FaceNeighborGeometricFactors(const GeometricFactors &geom_);

protected:
   Vector send_data;
   Array<int> send_offsets, recv_offsets;

   void ExchangeFaceNbrData(const Vector &x_local, Vector &x_shared,
                            const int ndof_per_el);
};

} // namespace mfem

#endif
