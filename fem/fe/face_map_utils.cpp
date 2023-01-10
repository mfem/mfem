// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Finite Element Base classes

#include "face_map_utils.hpp"

namespace mfem
{

std::pair<int,int> GetFaceNormal3D(const int face_id)
{
   switch (face_id)
   {
      case 0: return std::make_pair(2, 0); // z = 0
      case 1: return std::make_pair(1, 0); // y = 0
      case 2: return std::make_pair(0, 1); // x = 1
      case 3: return std::make_pair(1, 1); // y = 1
      case 4: return std::make_pair(0, 0); // x = 0
      case 5: return std::make_pair(2, 1); // z = 1
      default: MFEM_ABORT("Invalid face ID.")
   }
   return std::make_pair(-1, -1); // invalid
}

void FillFaceMap(const int n_face_dofs_per_component,
                 const std::vector<int> offsets,
                 const std::vector<int> &strides,
                 const std::vector<int> &n_dofs_per_dim,
                 Array<int> &face_map)
{
   const int n_components = offsets.size();
   const int face_dim = strides.size() / n_components;
   for (int comp = 0; comp < n_components; ++comp)
   {
      const int offset = offsets[comp];
      for (int i = 0; i < n_face_dofs_per_component; ++i)
      {
         int idx = offset;
         int j = i;
         for (int d = 0; d < face_dim; ++d)
         {
            const int dof1d = n_dofs_per_dim[comp*(face_dim) + d];
            idx += strides[comp*(face_dim) + d]*(j % dof1d);
            j /= dof1d;
         }
         face_map[comp*n_face_dofs_per_component + i] = idx;
      }
   }
}

}
