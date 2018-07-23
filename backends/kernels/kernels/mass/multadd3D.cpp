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
void rMassMultAdd3D(const int NUM_DOFS_1D,
                    const int NUM_QUAD_1D,
                    const int numElements,
                    const double* dofToQuad,
                    const double* dofToQuadD,
                    const double* quadToDof,
                    const double* quadToDofD,
                    const double* oper,
                    const double* solIn,
                    double* __restrict__ solOut)
{
#ifndef __LAMBDA__
   const int e = blockDim.x * blockIdx.x + threadIdx.x;
   if (e < numElements)
#else
   forall(e,numElements,
#endif
   {
      double sol_xyz[NUM_QUAD_1D][NUM_QUAD_1D][NUM_QUAD_1D];
      for (int qz = 0; qz < NUM_QUAD_1D; ++qz)
      {
         for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
         {
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
            {
               sol_xyz[qz][qy][qx] = 0;
            }
         }
      }
      for (int dz = 0; dz < NUM_DOFS_1D; ++dz)
      {
         double sol_xy[NUM_QUAD_1D][NUM_QUAD_1D];
         for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
         {
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
            {
               sol_xy[qy][qx] = 0;
            }
         }
         for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
         {
            double sol_x[NUM_QUAD_1D];
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
            {
               sol_x[qx] = 0;
            }
            for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
            {
               const double s = solIn[ijklN(dx,dy,dz,e,NUM_DOFS_1D)];
               for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
               {
                  sol_x[qx] += dofToQuad[ijN(qx,dx,NUM_QUAD_1D)] * s;
               }
            }
            for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
            {
               const double wy = dofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
               for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
               {
                  sol_xy[qy][qx] += wy * sol_x[qx];
               }
            }
         }
         for (int qz = 0; qz < NUM_QUAD_1D; ++qz)
         {
            const double wz = dofToQuad[ijN(qz,dz,NUM_QUAD_1D)];
            for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
            {
               for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
               {
                  sol_xyz[qz][qy][qx] += wz * sol_xy[qy][qx];
               }
            }
         }
      }
      for (int qz = 0; qz < NUM_QUAD_1D; ++qz)
      {
         for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
         {
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
            {
               sol_xyz[qz][qy][qx] *= oper[ijklN(qx,qy,qz,e,NUM_QUAD_1D)];
            }
         }
      }
      for (int qz = 0; qz < NUM_QUAD_1D; ++qz)
      {
         double sol_xy[NUM_DOFS_1D][NUM_DOFS_1D];
         for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
         {
            for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
            {
               sol_xy[dy][dx] = 0;
            }
         }
         for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
         {
            double sol_x[NUM_DOFS_1D];
            for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
            {
               sol_x[dx] = 0;
            }
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
            {
               const double s = sol_xyz[qz][qy][qx];
               for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
               {
                  sol_x[dx] += quadToDof[ijN(dx,qx,NUM_DOFS_1D)] * s;
               }
            }
            for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
            {
               const double wy = quadToDof[ijN(dy,qy,NUM_DOFS_1D)];
               for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
               {
                  sol_xy[dy][dx] += wy * sol_x[dx];
               }
            }
         }
         for (int dz = 0; dz < NUM_DOFS_1D; ++dz)
         {
            const double wz = quadToDof[ijN(dz,qz,NUM_DOFS_1D)];
            for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
            {
               for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
               {
                  solOut[ijklN(dx,dy,dz,e,NUM_DOFS_1D)] += wz * sol_xy[dy][dx];
               }
            }
         }
      }
   }
#ifdef __LAMBDA__
           );
#endif
}
