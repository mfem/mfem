// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
#include "../raja.hpp"

// *****************************************************************************
extern "C" kernel
void rLocalToGlobal0(const int globalEntries,
                     const int NUM_VDIM,
                     const bool VDIM_ORDERING,
                     const int localEntries,
                     const int* offsets,
                     const int* indices,
                     const double* localX,
                     double* __restrict globalX) {
#ifndef __LAMBDA__
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < globalEntries)
#else
    forall(i,globalEntries,
#endif
  {
    const int offset = offsets[i];
    const int nextOffset = offsets[i + 1];
    for (int v = 0; v < NUM_VDIM; ++v) {
      double dofValue = 0;
      for (int j = offset; j < nextOffset; ++j) {
        const int l_offset = ijNMt(v,indices[j],NUM_VDIM,localEntries,VDIM_ORDERING);
        dofValue += localX[l_offset];
      }
      const int g_offset = ijNMt(v,i,NUM_VDIM,globalEntries,VDIM_ORDERING);
      globalX[g_offset] = dofValue;
    }
  }
#ifdef __LAMBDA__
           );
#endif
}

// *****************************************************************************
void rLocalToGlobal(const int NUM_VDIM,
                    const bool VDIM_ORDERING,
                    const int globalEntries,
                    const int localEntries,
                    const int* offsets,
                    const int* indices,
                    const double* localX,
                    double* __restrict globalX) {
  push(Lime);
  cuKer(rLocalToGlobal,globalEntries,NUM_VDIM,VDIM_ORDERING,
        localEntries,offsets,indices,localX,globalX);
  pop();
}
