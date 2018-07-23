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
void rForceMult2D(const int NUM_DIM,
                  const int NUM_DOFS_1D,
                  const int NUM_QUAD_1D,
                  const int L2_DOFS_1D,
                  const int H1_DOFS_1D,
                  const int numElements,
                  const double* __restrict__ L2DofToQuad,
                  const double* __restrict__ H1QuadToDof,
                  const double* __restrict__ H1QuadToDofD,
                  const double* __restrict__ stressJinvT,
                  const double* __restrict__ e,
                  double* __restrict__ v){
   const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
#ifdef __LAMBDA__
   forall(el,numElements,
#else
   const int el = blockDim.x * blockIdx.x + threadIdx.x;
   if (el < numElements)
#endif
          {
             double e_xy[NUM_QUAD_2D];
             for (int i = 0; i < NUM_QUAD_2D; ++i)
{
   e_xy[i] = 0;
   }
   for (int dy = 0; dy < L2_DOFS_1D; ++dy)
{
   double e_x[NUM_QUAD_1D];
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
      {
         e_x[qy] = 0;
      }
      for (int dx = 0; dx < L2_DOFS_1D; ++dx)
      {
         const double r_e = e[ijkN(dx,dy,el,L2_DOFS_1D)];
         for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
         {
            e_x[qx] += L2DofToQuad[ijN(qx,dx,NUM_QUAD_1D)] * r_e;
         }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
      {
         const double wy = L2DofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
         for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
         {
            e_xy[ijN(qx,qy,NUM_QUAD_1D)] += wy * e_x[qx];
         }
      }
   }
   for (int c = 0; c < 2; ++c)
{
   for (int dy = 0; dy < H1_DOFS_1D; ++dy)
      {
         for (int dx = 0; dx < H1_DOFS_1D; ++dx)
         {
            v[_ijklNM(c,dx,dy,el,NUM_DOFS_1D,numElements)] = 0.0;
         }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
      {
         double Dxy[H1_DOFS_1D];
         double xy[H1_DOFS_1D];
         for (int dx = 0; dx < H1_DOFS_1D; ++dx)
         {
            Dxy[dx] = 0.0;
            xy[dx]  = 0.0;
         }
         for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
         {
            const double esx = e_xy[ijN(qx,qy,NUM_QUAD_1D)] *
                               stressJinvT[__ijklmNM(0,c,qx,qy,el,NUM_DIM,NUM_QUAD_1D)];
            const double esy = e_xy[ijN(qx,qy,NUM_QUAD_1D)] *
                               stressJinvT[__ijklmNM(1,c,qx,qy,el,NUM_DIM,NUM_QUAD_1D)];
            for (int dx = 0; dx < H1_DOFS_1D; ++dx)
            {
               Dxy[dx] += esx * H1QuadToDofD[ijN(dx,qx,H1_DOFS_1D)];
               xy[dx]  += esy * H1QuadToDof[ijN(dx,qx,H1_DOFS_1D)];
            }
         }
         for (int dy = 0; dy < H1_DOFS_1D; ++dy)
         {
            const double wy  = H1QuadToDof[ijN(dy,qy,H1_DOFS_1D)];
            const double wDy = H1QuadToDofD[ijN(dy,qy,H1_DOFS_1D)];
            for (int dx = 0; dx < H1_DOFS_1D; ++dx)
            {
               v[_ijklNM(c,dx,dy,el,NUM_DOFS_1D,numElements)] += wy* Dxy[dx] + wDy*xy[dx];
            }
         }
      }
   }
          }
#ifdef __LAMBDA__
         );
#endif
}

// *****************************************************************************
void rForceMultTranspose2D(const int NUM_DIM,
                           const int NUM_DOFS_1D,
                           const int NUM_QUAD_1D,
                           const int L2_DOFS_1D,
                           const int H1_DOFS_1D,
                           const int numElements,
                           const double* __restrict__ L2QuadToDof,
                           const double* __restrict__ H1DofToQuad,
                           const double* __restrict__ H1DofToQuadD,
                           const double* __restrict__ stressJinvT,
                           const double* __restrict__ v,
                           double* __restrict__ e)
{
   const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
#ifdef __LAMBDA__
   forall(el,numElements,
#else
   const int el = blockDim.x * blockIdx.x + threadIdx.x;
   if (el < numElements)
#endif
          {
             double vStress[NUM_QUAD_2D];
             for (int i = 0; i < NUM_QUAD_2D; ++i)
{
   vStress[i] = 0;
   }
   for (int c = 0; c < NUM_DIM; ++c)
{
   double v_Dxy[NUM_QUAD_2D];
      double v_xDy[NUM_QUAD_2D];
      for (int i = 0; i < NUM_QUAD_2D; ++i)
      {
         v_Dxy[i] = v_xDy[i] = 0;
      }
      for (int dy = 0; dy < H1_DOFS_1D; ++dy)
      {
         double v_x[NUM_QUAD_1D];
         double v_Dx[NUM_QUAD_1D];
         for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
         {
            v_x[qx] = v_Dx[qx] = 0;
         }

         for (int dx = 0; dx < H1_DOFS_1D; ++dx)
         {
            const double r_v = v[_ijklNM(c,dx,dy,el,NUM_DOFS_1D,numElements)];
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
            {
               v_x[qx]  += r_v * H1DofToQuad[ijN(qx,dx,NUM_QUAD_1D)];
               v_Dx[qx] += r_v * H1DofToQuadD[ijN(qx,dx,NUM_QUAD_1D)];
            }
         }
         for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
         {
            const double wy  = H1DofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
            const double wDy = H1DofToQuadD[ijN(qy,dy,NUM_QUAD_1D)];
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
            {
               v_Dxy[ijN(qx,qy,NUM_QUAD_1D)] += v_Dx[qx] * wy;
               v_xDy[ijN(qx,qy,NUM_QUAD_1D)] += v_x[qx]  * wDy;
            }
         }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
      {
         for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
         {
            vStress[ijN(qx,qy,NUM_QUAD_1D)] +=
               ((v_Dxy[ijN(qx,qy,NUM_QUAD_1D)] *
                 stressJinvT[__ijklmNM(0,c,qx,qy,el,NUM_DIM,NUM_QUAD_1D)]) +
                (v_xDy[ijN(qx,qy,NUM_QUAD_1D)] *
                 stressJinvT[__ijklmNM(1,c,qx,qy,el,NUM_DIM,NUM_QUAD_1D)]));
         }
      }
   }
   for (int dy = 0; dy < L2_DOFS_1D; ++dy)
{
   for (int dx = 0; dx < L2_DOFS_1D; ++dx)
      {
         e[ijkN(dx,dy,el,L2_DOFS_1D)] = 0;
      }
   }
   for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
{
   double e_x[L2_DOFS_1D];
      for (int dx = 0; dx < L2_DOFS_1D; ++dx)
      {
         e_x[dx] = 0;
      }
      for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
      {
         const double r_v = vStress[ijN(qx,qy,NUM_QUAD_1D)];
         for (int dx = 0; dx < L2_DOFS_1D; ++dx)
         {
            e_x[dx] += r_v * L2QuadToDof[ijN(dx,qx,L2_DOFS_1D)];
         }
      }
      for (int dy = 0; dy < L2_DOFS_1D; ++dy)
      {
         const double w = L2QuadToDof[ijN(dy,qy,L2_DOFS_1D)];
         for (int dx = 0; dx < L2_DOFS_1D; ++dx)
         {
            e[ijkN(dx,dy,el,L2_DOFS_1D)] += e_x[dx] * w;
         }
      }
   }
          }
#ifdef __LAMBDA__
         );
#endif
}
