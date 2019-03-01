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

namespace mfem
{
namespace kernels
{
namespace fem
{

// *****************************************************************************
template<const int NUM_DOFS_1D,
         const int NUM_QUAD_1D> static
void MassMultAdd2D(const int numElements,
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
template<const int NUM_DOFS_1D,
         const int NUM_QUAD_1D> static
void MassMultAdd3D(const int numElements,
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
typedef void (*fMassMultAdd)(const int numElements,
                             const double* __restrict dofToQuad,
                             const double* __restrict dofToQuadD,
                             const double* __restrict quadToDof,
                             const double* __restrict quadToDofD,
                             const double* __restrict oper,
                             const double* __restrict solIn,
                             double* __restrict solOut);

// *****************************************************************************
void MassMultAssembled(const int DIM,
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
   assert(LOG2(DIM)<=4);
   assert((NUM_QUAD_1D&1)==0);
   assert(LOG2(NUM_DOFS_1D)<=8);
   assert(LOG2(NUM_QUAD_1D)<=8);
   const unsigned int id = (DIM<<16)|((NUM_DOFS_1D)<<8)|(NUM_QUAD_1D);
   static std::unordered_map<unsigned int, fMassMultAdd> call =
   {
      // 2D
      {0x20101,&MassMultAdd2D<1,1>},
      {0x20201,&MassMultAdd2D<2,1>},
      {0x20202,&MassMultAdd2D<2,2>},
      {0x20204,&MassMultAdd2D<2,4>},
      {0x20302,&MassMultAdd2D<3,2>},
      {0x20304,&MassMultAdd2D<3,4>},
      {0x20303,&MassMultAdd2D<3,3>},
      {0x20403,&MassMultAdd2D<4,3>},
      {0x20408,&MassMultAdd2D<4,8>},
      {0x20508,&MassMultAdd2D<5,8>},
      /*
        {0x20304,&MassMultAdd2D<4,8>},    {0x20404,&MassMultAdd2D<5,8>},
        {0x20405,&MassMultAdd2D<5,10>},   {0x20505,&MassMultAdd2D<6,10>},
        {0x20506,&MassMultAdd2D<6,12>},   {0x20606,&MassMultAdd2D<7,12>},
        {0x20607,&MassMultAdd2D<7,14>},   {0x20707,&MassMultAdd2D<8,14>},
        {0x20708,&MassMultAdd2D<8,16>},   {0x20808,&MassMultAdd2D<9,16>},
        {0x20809,&MassMultAdd2D<9,18>},   {0x20909,&MassMultAdd2D<10,18>},
        {0x2090A,&MassMultAdd2D<10,20>},  {0x20A0A,&MassMultAdd2D<11,20>},
        {0x20A0B,&MassMultAdd2D<11,22>},  {0x20B0B,&MassMultAdd2D<12,22>},
        {0x20B0C,&MassMultAdd2D<12,24>},  {0x20C0C,&MassMultAdd2D<13,24>},
        {0x20C0D,&MassMultAdd2D<13,26>},  {0x20D0D,&MassMultAdd2D<14,26>},
        {0x20D0E,&MassMultAdd2D<14,28>},  {0x20E0E,&MassMultAdd2D<15,28>},
        {0x20E0F,&MassMultAdd2D<15,30>},  {0x20F0F,&MassMultAdd2D<16,30>},
        {0x20F10,&MassMultAdd2D<16,32>},  {0x21010,&MassMultAdd2D<17,32>},
      */
      // 3D
      {0x30102,&MassMultAdd3D<1,2>},
      {0x30202,&MassMultAdd3D<2,2>},
      {0x30204,&MassMultAdd3D<2,4>},
      {0x30304,&MassMultAdd3D<3,4>},
      /*
        {0x30203,&MassMultAdd3D<3,6>},    {0x30303,&MassMultAdd3D<4,6>},
        {0x30304,&MassMultAdd3D<4,8>},    {0x30404,&MassMultAdd3D<5,8>},
        {0x30405,&MassMultAdd3D<5,10>},   {0x30505,&MassMultAdd3D<6,10>},
        {0x30506,&MassMultAdd3D<6,12>},   {0x30606,&MassMultAdd3D<7,12>},
        {0x30607,&MassMultAdd3D<7,14>},   {0x30707,&MassMultAdd3D<8,14>},
        {0x30708,&MassMultAdd3D<8,16>},   {0x30808,&MassMultAdd3D<9,16>},
        {0x30809,&MassMultAdd3D<9,18>},   {0x30909,&MassMultAdd3D<10,18>},
        {0x3090A,&MassMultAdd3D<10,20>},  {0x30A0A,&MassMultAdd3D<11,20>},
        {0x30A0B,&MassMultAdd3D<11,22>},  {0x30B0B,&MassMultAdd3D<12,22>},
        {0x30B0C,&MassMultAdd3D<12,24>},  {0x30C0C,&MassMultAdd3D<13,24>},
        {0x30C0D,&MassMultAdd3D<13,26>},  {0x30D0D,&MassMultAdd3D<14,26>},
        {0x30D0E,&MassMultAdd3D<14,28>},  {0x30E0E,&MassMultAdd3D<15,28>},
        {0x30E0F,&MassMultAdd3D<15,30>},  {0x30F0F,&MassMultAdd3D<16,30>},
        {0x30F10,&MassMultAdd3D<16,32>},  {0x31010,&MassMultAdd3D<17,32>},
      */
   };
   if (!call[id])
   {
      printf("\n[MassMultAdd] id \033[33m0x%X\033[m ",id);
      fflush(0);
   }
   assert(call[id]);

   GET_CONST_PTR(dofToQuad);
   GET_CONST_PTR(dofToQuadD);
   GET_CONST_PTR(quadToDof);
   GET_CONST_PTR(quadToDofD);
   GET_CONST_PTR(op);
   GET_CONST_PTR(x);
   GET_PTR(y);
   call[id](numElements,
            d_dofToQuad, d_dofToQuadD, d_quadToDof, d_quadToDofD,
            d_op, d_x, d_y);
}

} // namespace fem
} // namespace kernels
} // namespace mfem
