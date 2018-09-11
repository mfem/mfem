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

#define QUAD_2D_ID(X, Y) (X + ((Y) * NUM_QUAD_1D))

// *****************************************************************************
template<const int NUM_DOFS_1D,
         const int NUM_QUAD_1D> kernel
void rDiffusionMultAdd2D(const int numElements,
                         const double* __restrict dofToQuad,
                         const double* __restrict dofToQuadD,
                         const double* __restrict quadToDof,
                         const double* __restrict quadToDofD,
                         const double* __restrict oper,
                         const double* __restrict solIn,
                         double* __restrict solOut)
{
#ifdef __NVCC__
   const int e = blockDim.x * blockIdx.x + threadIdx.x;
   if (e < numElements)
#else
   forall(e,numElements,
#endif
   {
      double grad[NUM_QUAD_1D][NUM_QUAD_1D][2];
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
      {
         for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
         {
            grad[qy][qx][0] = 0;
            grad[qy][qx][1] = 0;
         }
      }

      for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
      {
         double gradX[NUM_QUAD_1D][2];
         for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
         {
            gradX[qx][0] = 0;
            gradX[qx][1] = 0;
         }

         for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
         {
            const double s = solIn[ijkN(dx,dy,e,NUM_DOFS_1D)];
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
            {
               gradX[qx][0] += s * dofToQuad[ijN(qx,dx,NUM_QUAD_1D)];
               gradX[qx][1] += s * dofToQuadD[ijN(qx,dx,NUM_QUAD_1D)];
            }
         }

         for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
         {
            const double wy  = dofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
            const double wDy = dofToQuadD[ijN(qy,dy,NUM_QUAD_1D)];
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
            {
               grad[qy][qx][0] += gradX[qx][1] * wy;
               grad[qy][qx][1] += gradX[qx][0] * wDy;
            }
         }
      }

      // Calculate Dxy, xDy in plane
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
      {
         for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
         {
            const int q = QUAD_2D_ID(qx, qy);
            const double O11 = oper[ijkNM(0,q,e,3,NUM_QUAD_1D)];//(0, q, e);
            const double O12 = oper[ijkNM(1,q,e,3,NUM_QUAD_1D)];//(1, q, e);
            const double O22 = oper[ijkNM(2,q,e,3,NUM_QUAD_1D)];//(2, q, e);

            const double gradX = grad[qy][qx][0];
            const double gradY = grad[qy][qx][1];

            grad[qy][qx][0] = (O11 * gradX) + (O12 * gradY);
            grad[qy][qx][1] = (O12 * gradX) + (O22 * gradY);
         }
      }

      for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
      {
         double gradX[NUM_DOFS_1D][2];
         for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
         {
            gradX[dx][0] = 0;
            gradX[dx][1] = 0;
         }

         for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
         {
            const double gX = grad[qy][qx][0];
            const double gY = grad[qy][qx][1];
            for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
            {
               const double wx  = quadToDof[ijN(dx,qx,NUM_DOFS_1D)];
               const double wDx = quadToDofD[ijN(dx,qx,NUM_DOFS_1D)];
               gradX[dx][0] += gX * wDx;
               gradX[dx][1] += gY * wx;
            }
         }

         for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
         {
            const double wy  = quadToDof[ijN(dy,qy,NUM_DOFS_1D)];
            const double wDy = quadToDofD[ijN(dy,qy,NUM_DOFS_1D)];
            for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
            {
               solOut[ijkN(dx,dy,e,NUM_DOFS_1D)] += ((gradX[dx][0] * wy) +
                                                     (gradX[dx][1] * wDy));
            }
         }
      }
   }
#ifndef __NVCC__
          );
#endif
}

// *****************************************************************************
typedef void (*fDiffusionMultAdd)(const int numElements,
                                  const double* __restrict dofToQuad,
                                  const double* __restrict dofToQuadD,
                                  const double* __restrict quadToDof,
                                  const double* __restrict quadToDofD,
                                  const double* __restrict oper,
                                  const double* __restrict solIn,
                                  double* __restrict solOut);

// *****************************************************************************
void rDiffusionMultAdd(const int DIM,
                       const int NUM_DOFS_1D,
                       const int NUM_QUAD_1D,
                       const int numElements,
                       const double* dofToQuad,
                       const double* dofToQuadD,
                       const double* quadToDof,
                       const double* quadToDofD,
                       const double* op,
                       const double* x,
                       double* __restrict y)
{
   push();
#ifdef __NVCC__
   const int blck = 256;
   const int grid = (numElements+blck-1)/blck;
#endif
   const unsigned int id = (DIM<<16)|((NUM_DOFS_1D-1)<<8)|(NUM_QUAD_1D>>1);
   static std::unordered_map<unsigned int, fDiffusionMultAdd> call = {
    {0x20001,&rDiffusionMultAdd2D<1,2>},    {0x20101,&rDiffusionMultAdd2D<2,2>},
    {0x20102,&rDiffusionMultAdd2D<2,4>},    {0x20202,&rDiffusionMultAdd2D<3,4>},
    {0x20203,&rDiffusionMultAdd2D<3,6>},    {0x20303,&rDiffusionMultAdd2D<4,6>},
    //{0x30001,&rDiffusionMultAdd3D<1,2>},    {0x30101,&rDiffusionMultAdd3D<2,2>},
    //{0x30102,&rDiffusionMultAdd3D<2,4>},    {0x30202,&rDiffusionMultAdd3D<3,4>},
   };
   if(!call[id]){
      printf("\n[rDiffusionMultAdd] id \033[33m0x%X\033[m ",id);
      fflush(stdout);
   }
   assert(call[id]);
   call0(dummy,id,grid,blck,
         numElements,dofToQuad,dofToQuadD,quadToDof,quadToDofD,op,x,y);
  
   pop();
}
