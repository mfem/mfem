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

#include "../../general/okina.hpp"

// *****************************************************************************
void kIntDiffusionAssemble(const int dim,
                           const int NUM_QUAD_1D,
                           const int numElements,
                           const double* quadWeights,
                           const double* J,
                           const double COEFF,
                           double* __restrict oper)
{
   GET_CONST_ADRS(quadWeights);
   GET_CONST_ADRS(J);
   GET_ADRS(oper);

   if (dim==1) { assert(false); }
   if (dim==3) { assert(false); }

   const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;

   forall(e,numElements,
   {
      for (int q = 0; q < NUM_QUAD_2D; ++q)
      {
         const double J11 = d_J[ijklNM(0,0,q,e,2,NUM_QUAD_2D)];
         const double J12 = d_J[ijklNM(1,0,q,e,2,NUM_QUAD_2D)];
         const double J21 = d_J[ijklNM(0,1,q,e,2,NUM_QUAD_2D)];
         const double J22 = d_J[ijklNM(1,1,q,e,2,NUM_QUAD_2D)];
         const double c_detJ = d_quadWeights[q] * COEFF / ((J11*J22)-(J21*J12));
         //printf("\n\t[rDiffusionAssemble2D0] %f, %f, %f, %f, %f",J11,J12,J21,J22,c_detJ);
         d_oper[ijkNM(0,q,e,3,NUM_QUAD_2D)] =  c_detJ * (J21*J21 + J22*J22);
         d_oper[ijkNM(1,q,e,3,NUM_QUAD_2D)] = -c_detJ * (J21*J11 + J22*J12);
         d_oper[ijkNM(2,q,e,3,NUM_QUAD_2D)] =  c_detJ * (J11*J11 + J12*J12);
      }
   });
}
