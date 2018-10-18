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

// **************************************************************************
static void kAssemble2D(const int NUM_QUAD_1D,
                        const int numElements,
                        const double* __restrict quadWeights,
                        const double* __restrict J,
                        const double COEFF,
                        double* __restrict oper) {
   const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
   forall(e,numElements,
   {
      for (int q = 0; q < NUM_QUAD_2D; ++q)
      {
         const double J11 = J[ijklNM(0,0,q,e,2,NUM_QUAD_2D)];
         const double J12 = J[ijklNM(1,0,q,e,2,NUM_QUAD_2D)];
         const double J21 = J[ijklNM(0,1,q,e,2,NUM_QUAD_2D)];
         const double J22 = J[ijklNM(1,1,q,e,2,NUM_QUAD_2D)];
         const double c_detJ = quadWeights[q] * COEFF / ((J11*J22)-(J21*J12));
         //printf("\n\t[rDiffusionAssemble2D0] %f, %f, %f, %f, %f",J11,J12,J21,J22,c_detJ);
         oper[ijkNM(0,q,e,3,NUM_QUAD_2D)] =  c_detJ * (J21*J21 + J22*J22);
         oper[ijkNM(1,q,e,3,NUM_QUAD_2D)] = -c_detJ * (J21*J11 + J22*J12);
         oper[ijkNM(2,q,e,3,NUM_QUAD_2D)] =  c_detJ * (J11*J11 + J12*J12);
      }
   });
}

// **************************************************************************
static void kAssemble3D(const int NUM_QUAD_1D,
                        const int numElements,
                        const double* __restrict quadWeights,
                        const double* __restrict J,
                        const double COEFF,
                        double* __restrict oper) {
   const int NUM_QUAD_3D = NUM_QUAD_1D*NUM_QUAD_1D*NUM_QUAD_1D;
   forall(e,numElements,
   {
      for (int q = 0; q < NUM_QUAD_3D; ++q)
      {
         const double J11 = J[ijklNM(0,0,q,e,3,NUM_QUAD_3D)];
         const double J12 = J[ijklNM(1,0,q,e,3,NUM_QUAD_3D)];
         const double J13 = J[ijklNM(2,0,q,e,3,NUM_QUAD_3D)];
         const double J21 = J[ijklNM(0,1,q,e,3,NUM_QUAD_3D)];
         const double J22 = J[ijklNM(1,1,q,e,3,NUM_QUAD_3D)];
         const double J23 = J[ijklNM(2,1,q,e,3,NUM_QUAD_3D)];
         const double J31 = J[ijklNM(0,2,q,e,3,NUM_QUAD_3D)];
         const double J32 = J[ijklNM(1,2,q,e,3,NUM_QUAD_3D)];
         const double J33 = J[ijklNM(2,2,q,e,3,NUM_QUAD_3D)];
         const double detJ = ((J11 * J22 * J33) + (J12 * J23 * J31) +
                              (J13 * J21 * J32) - (J13 * J22 * J31) -
                              (J12 * J21 * J33) - (J11 * J23 * J32));
         const double c_detJ = quadWeights[q] * COEFF / detJ;
         // adj(J)
         const double A11 = (J22 * J33) - (J23 * J32);
         const double A12 = (J23 * J31) - (J21 * J33);
         const double A13 = (J21 * J32) - (J22 * J31);
         const double A21 = (J13 * J32) - (J12 * J33);
         const double A22 = (J11 * J33) - (J13 * J31);
         const double A23 = (J12 * J31) - (J11 * J32);
         const double A31 = (J12 * J23) - (J13 * J22);
         const double A32 = (J13 * J21) - (J11 * J23);
         const double A33 = (J11 * J22) - (J12 * J21);
         // adj(J)^Tadj(J)
         oper[ijkNM(0,q,e,6,NUM_QUAD_3D)] = c_detJ * (A11*A11 + A21*A21 + A31*A31); // (1,1)
         oper[ijkNM(1,q,e,6,NUM_QUAD_3D)] = c_detJ * (A11*A12 + A21*A22 + A31*A32); // (1,2), (2,1)
         oper[ijkNM(2,q,e,6,NUM_QUAD_3D)] = c_detJ * (A11*A13 + A21*A23 + A31*A33); // (1,3), (3,1)
         oper[ijkNM(3,q,e,6,NUM_QUAD_3D)] = c_detJ * (A12*A12 + A22*A22 + A32*A32); // (2,2)
         oper[ijkNM(4,q,e,6,NUM_QUAD_3D)] = c_detJ * (A12*A13 + A22*A23 + A32*A33); // (2,3), (3,2)
         oper[ijkNM(5,q,e,6,NUM_QUAD_3D)] = c_detJ * (A13*A13 + A23*A23 + A33*A33); // (3,3)
      }
   });
}

// *****************************************************************************
void kIntDiffusionAssemble(const int dim,
                           const int NUM_QUAD_1D,
                           const int numElements,
                           const double* __restrict quadWeights,
                           const double* __restrict J,
                           const double COEFF,
                           double* __restrict oper)
{
   GET_CONST_ADRS(quadWeights);
   GET_CONST_ADRS(J);
   GET_ADRS(oper);
   if (dim==1) { assert(false); }
   if (dim==2) { kAssemble2D(NUM_QUAD_1D, numElements, d_quadWeights, d_J, COEFF, d_oper); }
   if (dim==3) { kAssemble3D(NUM_QUAD_1D, numElements, d_quadWeights, d_J, COEFF, d_oper); }
}
