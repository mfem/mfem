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
#include "../../linalg/device.hpp"

namespace mfem
{
namespace kernels
{
namespace fem
{

// *****************************************************************************
template<const int ND1d,
         const int NQ1d> static
void MassMultAdd2D(const int NE,
                   const double* __restrict _B,
                   const double* __restrict _Bt,
                   const double* __restrict _op,
                   const double* __restrict _x,
                   double* __restrict _y)
{
   const DeviceMatrix B(_B, NQ1d,ND1d);
   const DeviceMatrix Bt(_Bt, ND1d,NQ1d);
   const DeviceTensor<3> op(_op, NQ1d,NQ1d,NE);
   const DeviceTensor<3> x(_x, ND1d,ND1d,NE);
   DeviceTensor<3> y(_y, ND1d,ND1d,NE);
   MFEM_FORALL(e,NE,
   {
      double sol_xy[NQ1d][NQ1d];
      for (int qy = 0; qy < NQ1d; ++qy)
      {
         for (int qx = 0; qx < NQ1d; ++qx)
         {
            sol_xy[qy][qx] = 0.0;
         }
      }
      for (int dy = 0; dy < ND1d; ++dy)
      {
         double sol_x[NQ1d];
         for (int qy = 0; qy < NQ1d; ++qy)
         {
            sol_x[qy] = 0.0;
         }
         for (int dx = 0; dx < ND1d; ++dx)
         {
            const double s = x(dx,dy,e);
            for (int qx = 0; qx < NQ1d; ++qx)
            {
               sol_x[qx] += B(qx,dx)* s;
            }
         }
         for (int qy = 0; qy < NQ1d; ++qy)
         {
            const double d2q = B(qy,dy);
            for (int qx = 0; qx < NQ1d; ++qx)
            {
               sol_xy[qy][qx] += d2q * sol_x[qx];
            }
         }
      }
      for (int qy = 0; qy < NQ1d; ++qy)
      {
         for (int qx = 0; qx < NQ1d; ++qx)
         {
            sol_xy[qy][qx] *= op(qx,qy,e);
         }
      }
      for (int qy = 0; qy < NQ1d; ++qy)
      {
         double sol_x[ND1d];
         for (int dx = 0; dx < ND1d; ++dx)
         {
            sol_x[dx] = 0.0;
         }
         for (int qx = 0; qx < NQ1d; ++qx)
         {
            const double s = sol_xy[qy][qx];
            for (int dx = 0; dx < ND1d; ++dx)
            {
               sol_x[dx] += Bt(dx,qx) * s;
            }
         }
         for (int dy = 0; dy < ND1d; ++dy)
         {
            const double q2d = Bt(dy,qy);
            for (int dx = 0; dx < ND1d; ++dx)
            {
               y(dx,dy,e) += q2d * sol_x[dx];
            }
         }
      }
   });
}

// *****************************************************************************
template<const int ND1d,
         const int NQ1d> static
void MassMultAdd3D(const int NE,
                   const double* __restrict _B,
                   const double* __restrict _Bt,
                   const double* __restrict _op,
                   const double* __restrict _x,
                   double* __restrict _y)
{
   const DeviceMatrix B(_B, NQ1d,ND1d);
   const DeviceMatrix Bt(_Bt, ND1d,NQ1d);
   const DeviceTensor<4> op(_op, NQ1d,NQ1d,NQ1d,NE);
   const DeviceTensor<4> x(_x, ND1d,ND1d,ND1d,NE);
   DeviceTensor<4> y(_y, ND1d,ND1d,ND1d,NE);
   
   MFEM_FORALL(e,NE,
   {
      double sol_xyz[NQ1d][NQ1d][NQ1d];
      for (int qz = 0; qz < NQ1d; ++qz)
      {
         for (int qy = 0; qy < NQ1d; ++qy)
         {
            for (int qx = 0; qx < NQ1d; ++qx)
            {
               sol_xyz[qz][qy][qx] = 0.0;
            }
         }
      }
      for (int dz = 0; dz < ND1d; ++dz)
      {
         double sol_xy[NQ1d][NQ1d];
         for (int qy = 0; qy < NQ1d; ++qy)
         {
            for (int qx = 0; qx < NQ1d; ++qx)
            {
               sol_xy[qy][qx] = 0.0;
            }
         }
         for (int dy = 0; dy < ND1d; ++dy)
         {
            double sol_x[NQ1d];
            for (int qx = 0; qx < NQ1d; ++qx)
            {
               sol_x[qx] = 0;
            }
            for (int dx = 0; dx < ND1d; ++dx)
            {
               const double s = x(dx,dy,dz,e);
               for (int qx = 0; qx < NQ1d; ++qx)
               {
                  sol_x[qx] += B(qx,dx) * s;
               }
            }
            for (int qy = 0; qy < NQ1d; ++qy)
            {
               const double wy = B(qy,dy);
               for (int qx = 0; qx < NQ1d; ++qx)
               {
                  sol_xy[qy][qx] += wy * sol_x[qx];
               }
            }
         }
         for (int qz = 0; qz < NQ1d; ++qz)
         {
            const double wz = B(qz,dz);
            for (int qy = 0; qy < NQ1d; ++qy)
            {
               for (int qx = 0; qx < NQ1d; ++qx)
               {
                  sol_xyz[qz][qy][qx] += wz * sol_xy[qy][qx];
               }
            }
         }
      }
      for (int qz = 0; qz < NQ1d; ++qz)
      {
         for (int qy = 0; qy < NQ1d; ++qy)
         {
            for (int qx = 0; qx < NQ1d; ++qx)
            {
               sol_xyz[qz][qy][qx] *= op(qx,qy,qz,e);
            }
         }
      }
      for (int qz = 0; qz < NQ1d; ++qz)
      {
         double sol_xy[ND1d][ND1d];
         for (int dy = 0; dy < ND1d; ++dy)
         {
            for (int dx = 0; dx < ND1d; ++dx)
            {
               sol_xy[dy][dx] = 0;
            }
         }
         for (int qy = 0; qy < NQ1d; ++qy)
         {
            double sol_x[ND1d];
            for (int dx = 0; dx < ND1d; ++dx)
            {
               sol_x[dx] = 0;
            }
            for (int qx = 0; qx < NQ1d; ++qx)
            {
               const double s = sol_xyz[qz][qy][qx];
               for (int dx = 0; dx < ND1d; ++dx)
               {
                  sol_x[dx] += Bt(dx,qx) * s;
               }
            }
            for (int dy = 0; dy < ND1d; ++dy)
            {
               const double wy = Bt(dy,qy);
               for (int dx = 0; dx < ND1d; ++dx)
               {
                  sol_xy[dy][dx] += wy * sol_x[dx];
               }
            }
         }
         for (int dz = 0; dz < ND1d; ++dz)
         {
            const double wz = Bt(dz,qz);
            for (int dy = 0; dy < ND1d; ++dy)
            {
               for (int dx = 0; dx < ND1d; ++dx)
               {
                  y(dx,dy,dz,e) += wz * sol_xy[dy][dx];
               }
            }
         }
      }
   });
}

// *****************************************************************************
typedef void (*fMassMultAdd)(const int NE,
                             const double* __restrict B,
                             const double* __restrict Bt,
                             const double* __restrict oper,
                             const double* __restrict solIn,
                             double* __restrict solOut);

// *****************************************************************************
void MassMultAssembled(const int dim,
                       const int ND1d,
                       const int NQ1d,
                       const int NE,
                       const double* __restrict B,
                       const double* __restrict Bt,
                       const double* __restrict op,
                       const double* __restrict x,
                       double* __restrict y)
{
   assert(LOG2(dim)<=4);
   assert(LOG2(ND1d)<=8);
   assert(LOG2(NQ1d)<=8);
   const unsigned int id = (dim<<16)|((ND1d)<<8)|(NQ1d);
   static std::unordered_map<unsigned int, fMassMultAdd> call =
   {
      // 2D
      //{0x20101,&MassMultAdd2D<1,1>},
      //{0x20201,&MassMultAdd2D<2,1>},
      {0x20202,&MassMultAdd2D<2,2>},
      //{0x20204,&MassMultAdd2D<2,4>},
      //{0x20302,&MassMultAdd2D<3,2>},
      //{0x20304,&MassMultAdd2D<3,4>},
      //{0x20303,&MassMultAdd2D<3,3>},
      //{0x20403,&MassMultAdd2D<4,3>},
      //{0x20408,&MassMultAdd2D<4,8>},
      //{0x20508,&MassMultAdd2D<5,8>},
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
      //{0x30102,&MassMultAdd3D<1,2>},
      //{0x30202,&MassMultAdd3D<2,2>},
      //{0x30204,&MassMultAdd3D<2,4>},
      {0x30203,&MassMultAdd3D<2,3>},
      //{0x30304,&MassMultAdd3D<3,4>},
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
      printf("dim=%d, ND1d=%d and NQ1d=%d",dim, ND1d, NQ1d);
      mfem_error("MassMultAssembled kernel not instanciated");
   }
   call[id](NE, B, Bt, op, x, y);
}

} // namespace fem
} // namespace kernels
} // namespace mfem
