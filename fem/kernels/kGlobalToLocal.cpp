// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../../general/okina.hpp"

namespace mfem
{

// *****************************************************************************
__kernel void kGlobalToLocal(const int NUM_VDIM,
                             const bool VDIM_ORDERING,
                             const int globalEntries,
                             const int localEntries,
                             const int* __restrict offsets,
                             const int* __restrict indices,
                             const double* __restrict globalX,
                             double* __restrict localX)
{
   MFEM_FORALL(i, globalEntries,
   {
      const int offset = offsets[i];
      const int nextOffset = offsets[i+1];
      for (int v = 0; v < NUM_VDIM; ++v)
      {
         const int g_offset = ijNMt(v,i,NUM_VDIM,globalEntries,VDIM_ORDERING);
         const double dofValue = globalX[g_offset];
         for (int j = offset; j < nextOffset; ++j)
         {
            const int l_offset = ijNMt(v,indices[j],NUM_VDIM,localEntries,VDIM_ORDERING);
            localX[l_offset] = dofValue;
         }
      }
   });
}

}
