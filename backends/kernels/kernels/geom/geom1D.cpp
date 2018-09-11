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
#include "../kernels.hpp"

// *****************************************************************************
void rIniGeom1D(const int NUM_DOFS,
                const int NUM_QUAD,
                const int numElements,
                const double* __restrict__ dofToQuadD,
                const double* __restrict__ nodes,
                double* __restrict__ J,
                double* __restrict__ invJ,
                double* __restrict__ detJ){
#ifdef __NVCC__
   const int e = blockDim.x * blockIdx.x + threadIdx.x;
   if (e < numElements)
#else
   forall(e,numElements,
#endif
   {
      double s_nodes[NUM_DOFS];
      for (int q = 0; q < NUM_QUAD; ++q)
      {
         for (int d = q; d < NUM_DOFS; d += NUM_QUAD)
         {
            s_nodes[d] = nodes[ijkN(0,d,e,NUM_QUAD)];
         }
      }
      for (int q = 0; q < NUM_QUAD; ++q)
      {
         double J11 = 0;
         for (int d = 0; d < NUM_DOFS; ++d)
         {
            const double wx = dofToQuadD[ijN(q,d,NUM_DOFS)];
            J11 += wx * s_nodes[d];
         }
         J[ijN(q,e,NUM_QUAD)] = J11;
         invJ[ijN(q, e,NUM_QUAD)] = 1.0 / J11;
         detJ[ijN(q, e,NUM_QUAD)] = J11;
      }
   }
#ifndef __NVCC__
           );
#endif
}

