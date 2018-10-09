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
#include "kernels.hpp"

/*
// *****************************************************************************
template<const int NUM_DOFS,
         const int NUM_QUAD> kernel
void rIniGeom2D(const int numElements,
                const double* __restrict dofToQuadD,
                const double* __restrict nodes,
                double* __restrict J,
                double* __restrict invJ,
                double* __restrict detJ){
#ifdef __NVCC__
   const int e = blockDim.x * blockIdx.x + threadIdx.x;
   if (e < numElements)
#else
#warning CPU forall
   forall(e,numElements,
#endif
   {
      double s_nodes[2 * NUM_DOFS];
      for (int q = 0; q < NUM_QUAD; ++q)
      {
         for (int d = q; d < NUM_DOFS; d +=NUM_QUAD)
         {
            const int s0 = ijN(0,d,2);
            const int s1 = ijN(1,d,2);
            const int x0 = ijkNM(0,d,e,2,NUM_DOFS);
            const int y0 = ijkNM(1,d,e,2,NUM_DOFS);
            printf("\n\t[rIniGeom2D] s0=%d, s1=%d, x0=%d, y0=%d", s0, s1, x0, y0);
            s_nodes[s0] = nodes[x0];
            s_nodes[s1] = nodes[y0];
            printf("\n\t[rIniGeom2D] s_nodes %f, %f",nodes[x0],nodes[y0]);
         }
      }
      for (int q = 0; q < NUM_QUAD; ++q)
      {
         double J11 = 0; double J12 = 0;
         double J21 = 0; double J22 = 0;
         for (int d = 0; d < NUM_DOFS; ++d)
         {
            const double wx = dofToQuadD[ijkNM(0,q,d,2,NUM_QUAD)];
            const double wy = dofToQuadD[ijkNM(1,q,d,2,NUM_QUAD)];
            printf("\n\t[rIniGeom2D] wx wy: %f, %f",wx, wy);
            const double x = s_nodes[ijN(0,d,2)];
            const double y = s_nodes[ijN(1,d,2)];
            J11 += (wx * x); J12 += (wx * y);
            J21 += (wy * x); J22 += (wy * y);
         }
         const double r_detJ = (J11 * J22)-(J12 * J21);
         J[ijklNM(0, 0, q, e,2,NUM_QUAD)] = J11;
         J[ijklNM(1, 0, q, e,2,NUM_QUAD)] = J12;
         J[ijklNM(0, 1, q, e,2,NUM_QUAD)] = J21;
         J[ijklNM(1, 1, q, e,2,NUM_QUAD)] = J22;
         const double r_idetJ = 1.0 / r_detJ;
         invJ[ijklNM(0, 0, q, e,2,NUM_QUAD)] =  J22 * r_idetJ;
         invJ[ijklNM(1, 0, q, e,2,NUM_QUAD)] = -J12 * r_idetJ;
         invJ[ijklNM(0, 1, q, e,2,NUM_QUAD)] = -J21 * r_idetJ;
         invJ[ijklNM(1, 1, q, e,2,NUM_QUAD)] =  J11 * r_idetJ;
         detJ[ijN(q, e,NUM_QUAD)] = r_detJ;
         assert(r_detJ!=0.0);
         printf("\n\t[rIniGeom2D] %f, %f, %f, %f, %f",J11,J12,J21,J22,r_detJ);
      }
   }
#ifndef __NVCC__
          );
#endif
}
*/

// *****************************************************************************
template<const int NUM_DOFS,
         const int NUM_QUAD> kernel
void rIniGeom2D(const int numElements,
                const double* __restrict dofToQuadD,
                const double* __restrict nodes,
                double* __restrict J,
                double* __restrict invJ,
                double* __restrict detJ){
   const int k = blockDim.x * blockIdx.x + threadIdx.x;
   if (k >= 1) return;
   for(int e=0; e<numElements; e+=1){
      for (int q = 0; q < NUM_QUAD; ++q) {
         double J11 = 0.0; double J12 = 0.0;
         double J21 = 0.0; double J22 = 0.0;
         for (int d = 0; d < NUM_DOFS; d +=1) {
            const int Nx = ijkNM(0,d,e,2,NUM_DOFS);
            const int Ny = ijkNM(1,d,e,2,NUM_DOFS);
            printf("\n\t[rIniGeom2D] Nx=%d, Ny=%d", Nx, Ny);
            const double wx = dofToQuadD[ijkNM(0,q,d,2,NUM_QUAD)];
            const double wy = dofToQuadD[ijkNM(1,q,d,2,NUM_QUAD)];
            printf("\n\t[rIniGeom2D] wx wy: %f, %f",wx, wy);
            const double x = nodes[Nx];
            const double y = nodes[Ny];
            J11 += (wx * x); J12 += (wx * y);
            J21 += (wy * x); J22 += (wy * y);
         }
         const double r_detJ = (J11 * J22)-(J12 * J21);
         printf("\n\t[rIniGeom2D] J11=%f, J12=%f, J21=%f, J22=%f, det=%f",J11, J12, J21, J22, r_detJ);
         assert(r_detJ!=0.0);
         J[ijklNM(0, 0, q, e,2,NUM_QUAD)] = J11;
         J[ijklNM(1, 0, q, e,2,NUM_QUAD)] = J12;
         J[ijklNM(0, 1, q, e,2,NUM_QUAD)] = J21;
         J[ijklNM(1, 1, q, e,2,NUM_QUAD)] = J22;
         const double r_idetJ = 1.0 / r_detJ;
         invJ[ijklNM(0, 0, q, e,2,NUM_QUAD)] =  J22 * r_idetJ;
         invJ[ijklNM(1, 0, q, e,2,NUM_QUAD)] = -J12 * r_idetJ;
         invJ[ijklNM(0, 1, q, e,2,NUM_QUAD)] = -J21 * r_idetJ;
         invJ[ijklNM(1, 1, q, e,2,NUM_QUAD)] =  J11 * r_idetJ;
         detJ[ijN(q, e,NUM_QUAD)] = r_detJ;
      }
   }
   }

template kernel void rIniGeom2D<4,1>(int, double const*, double const*, double*, double*, double*);
/*template kernel void rIniGeom2D<9,16>(int, double const*, double const*, double*, double*, double*);
template kernel void rIniGeom2D<16,36>(int, double const*, double const*, double*, double*, double*);
template kernel void rIniGeom2D<25,64>(int, double const*, double const*, double*, double*, double*);
template kernel void rIniGeom2D<36,100>(int, double const*, double const*, double*, double*, double*);
template kernel void rIniGeom2D<49,144>(int, double const*, double const*, double*, double*, double*);
template kernel void rIniGeom2D<64,196>(int, double const*, double const*, double*, double*, double*);
template kernel void rIniGeom2D<81,256>(int, double const*, double const*, double*, double*, double*);
template kernel void rIniGeom2D<100,324>(int, double const*, double const*, double*, double*, double*);
template kernel void rIniGeom2D<121,400>(int, double const*, double const*, double*, double*, double*);
template kernel void rIniGeom2D<144,484>(int, double const*, double const*, double*, double*, double*);
template kernel void rIniGeom2D<169,576>(int, double const*, double const*, double*, double*, double*);
template kernel void rIniGeom2D<196,676>(int, double const*, double const*, double*, double*, double*);
template kernel void rIniGeom2D<225,784>(int, double const*, double const*, double*, double*, double*);
template kernel void rIniGeom2D<256,900>(int, double const*, double const*, double*, double*, double*);
template kernel void rIniGeom2D<289,1024>(int, double const*, double const*, double*, double*, double*);
*/
