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

// *****************************************************************************
namespace mfem
{
namespace kernels
{
namespace fem
{

// *****************************************************************************
__template __kernel 
void MassMultAdd2D(const int __range(1-4) NUM_DOFS_1D,
                   const int __range(1-5) NUM_QUAD_1D,
                   const int numElements,
                   const double* __restrict dofToQuad,
                   const double* __restrict dofToQuadD,
                   const double* __restrict quadToDof,
                   const double* __restrict quadToDofD,
                   const double* __restrict oper,
                   const double* __restrict solIn,
                   double* __restrict solOut)
{
   MFEM_FORALL(e,numElements,
   {
      double sol_xy[NUM_QUAD_1D][NUM_QUAD_1D];
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
      {
         for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
         {
            sol_xy[qy][qx] = 0.0;
         }
      }
      for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
      {
         double sol_x[NUM_QUAD_1D];
         for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
         {
            sol_x[qy] = 0.0;
         }
         for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
         {
            const double s = solIn[ijkN(dx,dy,e,NUM_DOFS_1D)];
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
            {
               sol_x[qx] += dofToQuad[ijN(qx,dx,NUM_QUAD_1D)]* s;
            }
         }
         for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
         {
            const double d2q = dofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
            {
               sol_xy[qy][qx] += d2q * sol_x[qx];
            }
         }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
      {
         for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
         {
            sol_xy[qy][qx] *= oper[ijkN(qx,qy,e,NUM_QUAD_1D)];
         }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
      {
         double sol_x[NUM_DOFS_1D];
         for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
         {
            sol_x[dx] = 0.0;
         }
         for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
         {
            const double s = sol_xy[qy][qx];
            for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
            {
               sol_x[dx] += quadToDof[ijN(dx,qx,NUM_DOFS_1D)] * s;
            }
         }
         for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
         {
            const double q2d = quadToDof[ijN(dy,qy,NUM_DOFS_1D)];
            for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
            {
               solOut[ijkN(dx,dy,e,NUM_DOFS_1D)] += q2d * sol_x[dx];
            }
         }
      }
   });
}

// *****************************************************************************
__template __kernel
void MassMultAdd3D(const int __range(1-5) NUM_DOFS_1D,
                   const int __range(1-5) NUM_QUAD_1D,
                   const int numElements,
                   const double* __restrict dofToQuad,
                   const double* __restrict dofToQuadD,
                   const double* __restrict quadToDof,
                   const double* __restrict quadToDofD,
                   const double* __restrict oper,
                   const double* __restrict solIn,
                   double* __restrict solOut)
{
   MFEM_FORALL(e,numElements,
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
   });
}

// *****************************************************************************
void PAMassMultAdd(const int dim,
                   const int NUM_DOFS_1D,
                   const int NUM_QUAD_1D,
                   const int numElements,
                   const double* __restrict dofToQuad,
                   const double* __restrict dofToQuadD,
                   const double* __restrict quadToDof,
                   const double* __restrict quadToDofD,
                   const double* __restrict op,
                   const double* __restrict x,
                   double* __restrict y)
{
   if (dim==1) mfem_error("Not implemented");
   if (dim==2)
      MassMultAdd2D(NUM_DOFS_1D,NUM_QUAD_1D,numElements,
                    dofToQuad, dofToQuadD, quadToDof, quadToDofD,
                    op, x, y);
   if (dim==3)
      MassMultAdd3D(NUM_DOFS_1D,NUM_QUAD_1D,numElements,
                    dofToQuad, dofToQuadD, quadToDof, quadToDofD,
                    op, x, y);
}

} // namespace fem
} // namespace kernels
} // namespace mfem
