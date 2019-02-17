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
static void MassMultAdd2D(const int ND1d,
                          const int NQ1d,
                          const int NE,
                          const double* __restrict _B,
                          const double* __restrict _Bt,
                          const double* __restrict _op,
                          const double* __restrict _x,
                          double* __restrict _y)
{
   const int NQ = NQ1d*NQ1d;
   const int Nspt = NQ + std::max(NQ1d,ND1d);
   const Vector B(NQ1d,ND1d,_B);
   const Vector Bt(ND1d,NQ1d,_Bt);
   const Vector op(NQ1d,NQ1d,NE,_op);
   const Vector x(ND1d,ND1d,NE,_x);
   Vector y(ND1d,ND1d,NE,_y);

   MFEM_FORALL_SHARED(e, NE, Nspt,
   {
      Vector2 sol_xy(NQ1d,NQ1d,__shared);
      for (int qy = 0; qy < NQ1d; ++qy)
      {
         for (int qx = 0; qx < NQ1d; ++qx)
         {
            sol_xy(qy,qx) = 0.0;
         }
      }
      for (int dy = 0; dy < ND1d; ++dy)
      {
         Vector1 sol_x(NQ1d,__shared + NQ);
         for (int qy = 0; qy < NQ1d; ++qy)
         {
            sol_x(qy) = 0.0;
         }
         for (int dx = 0; dx < ND1d; ++dx)
         {
            const double s = x(dx,dy,e);
            for (int qx = 0; qx < NQ1d; ++qx)
            {
               sol_x(qx) += B(qx,dx)* s;
            }
         }
         for (int qy = 0; qy < NQ1d; ++qy)
         {
            const double d2q = B(qy,dy);
            for (int qx = 0; qx < NQ1d; ++qx)
            {
               sol_xy(qy,qx) += d2q * sol_x(qx);
            }
         }
      }
      for (int qy = 0; qy < NQ1d; ++qy)
      {
         for (int qx = 0; qx < NQ1d; ++qx)
         {
            sol_xy(qy,qx) *= op(qx,qy,e);
         }
      }
      for (int qy = 0; qy < NQ1d; ++qy)
      {
         Vector1 sol_x(ND1d,__shared + NQ);
         for (int dx = 0; dx < ND1d; ++dx)
         {
            sol_x(dx) = 0.0;
         }
         for (int qx = 0; qx < NQ1d; ++qx)
         {
            const double s = sol_xy(qy,qx);
            for (int dx = 0; dx < ND1d; ++dx)
            {
               sol_x(dx) += Bt(dx,qx) * s;
            }
         }
         for (int dy = 0; dy < ND1d; ++dy)
         {
            const double q2d = Bt(dy,qy);
            for (int dx = 0; dx < ND1d; ++dx)
            {
               y(dx,dy,e) += q2d * sol_x(dx);
            }
         }
      }
   });
}

// *****************************************************************************
static void MassMultAdd3D(const int ND1d,
                          const int NQ1d,
                          const int NE,
                          const double* __restrict _B,
                          const double* __restrict _Bt,
                          const double* __restrict _oper,
                          const double* __restrict _solIn,
                          double* __restrict _solOut)
{
   const int NQ = NQ1d*NQ1d*NQ1d;
   const int maxDQ = std::max(NQ1d,ND1d);
   const int maxDQ2 = maxDQ * maxDQ;
   const int Nspt = NQ + maxDQ2 + maxDQ;

   const Vector B(NQ1d,ND1d,_B);
   const Vector Bt(ND1d,NQ1d,_Bt);
   const Vector oper(NQ1d,NQ1d,NQ1d,NE,_oper);
   const Vector x(ND1d,ND1d,ND1d,NE,_solIn);
   Vector y(ND1d,ND1d,ND1d,NE,_solOut);

   MFEM_FORALL_SHARED(e, NE, Nspt,
   {
      Vector3 sol_xyz(NQ1d,NQ1d,NQ1d,__shared);
      for (int qz = 0; qz < NQ1d; ++qz)
      {
         for (int qy = 0; qy < NQ1d; ++qy)
         {
            for (int qx = 0; qx < NQ1d; ++qx)
            {
               sol_xyz(qz,qy,qx) = 0.0;
            }
         }
      }
      for (int dz = 0; dz < ND1d; ++dz)
      {
         Vector2 sol_xy(NQ1d,NQ1d,__shared + NQ);
         for (int qy = 0; qy < NQ1d; ++qy)
         {
            for (int qx = 0; qx < NQ1d; ++qx)
            {
               sol_xy(qy,qx) = 0.0;
            }
         }
         for (int dy = 0; dy < ND1d; ++dy)
         {
            Vector1 sol_x(NQ1d, __shared + NQ + maxDQ2);
            for (int qx = 0; qx < NQ1d; ++qx)
            {
               sol_x(qx) = 0.0;
            }
            for (int dx = 0; dx < ND1d; ++dx)
            {
               const double s = x(dx,dy,dz,e);
               for (int qx = 0; qx < NQ1d; ++qx)
               {
                  sol_x(qx) += B(qx,dx) * s;
               }
            }
            for (int qy = 0; qy < NQ1d; ++qy)
            {
               const double wy = B(qy,dy);
               for (int qx = 0; qx < NQ1d; ++qx)
               {
                  sol_xy(qy,qx) += wy * sol_x(qx);
               }
            }
         }
         for (int qz = 0; qz < NQ1d; ++qz)
         {
            const double wz =  B(qz,dz);
            for (int qy = 0; qy < NQ1d; ++qy)
            {
               for (int qx = 0; qx < NQ1d; ++qx)
               {
                  sol_xyz(qz,qy,qx) += wz * sol_xy(qy,qx);
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
               sol_xyz(qz,qy,qx) *= oper(qx,qy,qz,e);
            }
         }
      }
      for (int qz = 0; qz < NQ1d; ++qz)
      {
         Vector2 sol_xy(ND1d,ND1d,__shared + NQ);
         for (int dy = 0; dy < ND1d; ++dy)
         {
            for (int dx = 0; dx < ND1d; ++dx)
            {
               sol_xy(dy,dx) = 0.0;
            }
         }
         for (int qy = 0; qy < NQ1d; ++qy)
         {
            Vector1 sol_x(ND1d, __shared + NQ + maxDQ2);
            for (int dx = 0; dx < ND1d; ++dx)
            {
               sol_x(dx) = 0.0;
            }
            for (int qx = 0; qx < NQ1d; ++qx)
            {
               const double s = sol_xyz(qz,qy,qx);
               for (int dx = 0; dx < ND1d; ++dx)
               {
                  sol_x(dx) += Bt(dx,qx) * s;
               }
            }
            for (int dy = 0; dy < ND1d; ++dy)
            {
               const double wy = Bt(dy,qy);
               for (int dx = 0; dx < ND1d; ++dx)
               {
                  sol_xy(dy,dx) += wy * sol_x(dx);
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
                  y(dx,dy,dz,e) += wz * sol_xy(dy,dx);
               }
            }
         }
      }
   });
}

// *****************************************************************************
void MassMultAssembled(const int DIM,
                       const int ND1d,
                       const int NQ1d,
                       const int NE,
                       const double* __restrict B,
                       const double* __restrict Bt,
                       const double* __restrict op,
                       const double* __restrict x,
                       double* __restrict y)
{
   if (DIM==2)
   {
      MassMultAdd2D(ND1d, NQ1d, NE, B, Bt, op, x, y);
      return;
   }
   if (DIM==3)
   {
      MassMultAdd3D(ND1d, NQ1d, NE, B, Bt, op, x, y);
      return;
   }
}

} // namespace fem
} // namespace kernels
} // namespace mfem
