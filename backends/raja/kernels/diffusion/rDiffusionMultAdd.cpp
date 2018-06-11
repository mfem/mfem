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

#define QUAD_2D_ID(X, Y) (X + ((Y) * NUM_QUAD_1D))

// *****************************************************************************
#ifdef __TEMPLATES__
template<const int NUM_DOFS_1D,
         const int NUM_QUAD_1D> kernel
#endif
void rDiffusionMultAdd2D(
#ifndef __TEMPLATES__
   const int NUM_DOFS_1D,
   const int NUM_QUAD_1D,
#endif
   const int numElements,
   const double* restrict dofToQuad,
   const double* restrict dofToQuadD,
   const double* restrict quadToDof,
   const double* restrict quadToDofD,
   const double* restrict oper,
   const double* restrict solIn,
   double* restrict solOut)
{
#ifndef __LAMBDA__
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
#ifdef __LAMBDA__
           );
#endif
}

// *****************************************************************************
typedef void (*fDiffusionMultAdd)(const int numElements,
                                  const double* dofToQuad,
                                  const double* dofToQuadD,
                                  const double* quadToDof,
                                  const double* quadToDofD,
                                  const double* oper,
                                  const double* solIn,
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
#ifndef __LAMBDA__
   const int blck = 256;
   const int grid = (numElements+blck-1)/blck;
#endif
#ifdef __TEMPLATES__
   assert(LOG2(DIM)<=4);
   assert((NUM_QUAD_1D&1)==0);
   assert(LOG2(NUM_DOFS_1D-1)<=8);
   assert(LOG2(NUM_QUAD_1D>>1)<=8);
   const unsigned int id = (DIM<<16)|((NUM_DOFS_1D-1)<<8)|(NUM_QUAD_1D>>1);
   static std::unordered_map<unsigned int, fDiffusionMultAdd> call =
   {
      // 2D
      {0x20001,&rDiffusionMultAdd2D<1,2>},    {0x20101,&rDiffusionMultAdd2D<2,2>},
      {0x20102,&rDiffusionMultAdd2D<2,4>},    {0x20202,&rDiffusionMultAdd2D<3,4>},
      {0x20203,&rDiffusionMultAdd2D<3,6>},    {0x20303,&rDiffusionMultAdd2D<4,6>},
      {0x20304,&rDiffusionMultAdd2D<4,8>},    {0x20404,&rDiffusionMultAdd2D<5,8>},
      {0x20405,&rDiffusionMultAdd2D<5,10>},   {0x20505,&rDiffusionMultAdd2D<6,10>},
      {0x20506,&rDiffusionMultAdd2D<6,12>},   {0x20606,&rDiffusionMultAdd2D<7,12>},
      {0x20607,&rDiffusionMultAdd2D<7,14>},   {0x20707,&rDiffusionMultAdd2D<8,14>},
      {0x20708,&rDiffusionMultAdd2D<8,16>},   {0x20808,&rDiffusionMultAdd2D<9,16>},
      {0x20809,&rDiffusionMultAdd2D<9,18>},   {0x20909,&rDiffusionMultAdd2D<10,18>},
      {0x2090A,&rDiffusionMultAdd2D<10,20>},  {0x20A0A,&rDiffusionMultAdd2D<11,20>},
      {0x20A0B,&rDiffusionMultAdd2D<11,22>},  {0x20B0B,&rDiffusionMultAdd2D<12,22>},
      {0x20B0C,&rDiffusionMultAdd2D<12,24>},  {0x20C0C,&rDiffusionMultAdd2D<13,24>},
      {0x20C0D,&rDiffusionMultAdd2D<13,26>},  {0x20D0D,&rDiffusionMultAdd2D<14,26>},
      {0x20D0E,&rDiffusionMultAdd2D<14,28>},  {0x20E0E,&rDiffusionMultAdd2D<15,28>},
      {0x20E0F,&rDiffusionMultAdd2D<15,30>},  {0x20F0F,&rDiffusionMultAdd2D<16,30>},
      {0x20F10,&rDiffusionMultAdd2D<16,32>},  {0x21010,&rDiffusionMultAdd2D<17,32>},
   };
   if (!call[id])
   {
      printf("\n[rDiffusionMultAdd] id \033[33m0x%X\033[m ",id);
      fflush(stdout);
   }
   assert(call[id]);
   call0(rDiffusionMultAdd,id,grid,blck,
         numElements,dofToQuad,dofToQuadD,quadToDof,quadToDofD,op,x,y);
#else
   if (DIM==1) { assert(false); }
   if (DIM==2)
      call0(rDiffusionMultAdd2D,id,grid,blck,
            NUM_DOFS_1D,NUM_QUAD_1D,
            numElements,dofToQuad,dofToQuadD,quadToDof,quadToDofD,op,x,y);
   if (DIM==3) { assert(false); }
#endif
   pop();
}
