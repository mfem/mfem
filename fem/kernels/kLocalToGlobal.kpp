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
void kLocalToGlobal(const int NUM_VDIM,
                    const bool VDIM_ORDERING,
                    const int globalEntries,
                    const int localEntries,
                    const int* __restrict offsets,
                    const int* __restrict indices,
                    const double* __restrict localX,
                    double* __restrict globalX)
{

   GET_CONST_ADRS_T(offsets,int);
   GET_CONST_ADRS_T(indices,int);
   GET_CONST_ADRS(localX);
   GET_ADRS(globalX);
   MFEM_FORALL(i, globalEntries,
   {
      const int offset = d_offsets[i];
      const int nextOffset = d_offsets[i + 1];
      for (int v = 0; v < NUM_VDIM; ++v)
      {
         double dofValue = 0;
         for (int j = offset; j < nextOffset; ++j)
         {
            const int l_offset = ijNMt(v,d_indices[j],NUM_VDIM,localEntries,VDIM_ORDERING);
            dofValue += d_localX[l_offset];
         }
         const int g_offset = ijNMt(v,i,NUM_VDIM,globalEntries,VDIM_ORDERING);
         d_globalX[g_offset] = dofValue;
      }
   });

}

}
