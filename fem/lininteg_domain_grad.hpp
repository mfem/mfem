// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Internal header, included only by .cpp files

#include "fem.hpp"

#include "../general/forall.hpp"
#include "../fem/kernels.hpp"
#include "../linalg/kernels.hpp"

namespace mfem
{

namespace internal
{

namespace linearform_extension
{

////////////////////////////////////////////////////////////////////////////////
template<int D1D=0, int Q1D=0> static
void VectorDomainLFGradIntegratorAssemble2D(const int vdim,
                                            const bool byVDIM,
                                            const int ND,
                                            const int NE,
                                            const int d1d,
                                            const int q1d,
                                            const double *marks,
                                            const double *b,
                                            const double *g,
                                            const int *idx,
                                            const double *jacobians,
                                            const double *weights,
                                            const Vector &coeff,
                                            double *y)
{
   constexpr int DIM = 2;
   constexpr bool USE_SMEM = D1D > 0 && Q1D > 0;

   const int cdim = vdim == 1 ? DIM : vdim;
   const bool cst_coeff = coeff.Size() == cdim;

   const auto F = coeff.Read();
   const auto M = Reshape(marks, NE);
   const auto B = Reshape(b, q1d,d1d);
   const auto G = Reshape(g, q1d,d1d);
   const auto J = Reshape(jacobians, q1d,q1d,DIM,DIM,NE);
   const auto W = Reshape(weights, q1d,q1d);
   const auto I = Reshape(idx, d1d,d1d, NE);
   const auto C = cst_coeff ?
                  Reshape(F,DIM,cdim/DIM,1,1,1):
                  Reshape(F,DIM,cdim/DIM,q1d,q1d,NE);

   auto Y = Reshape(y, byVDIM ? vdim : ND, byVDIM ? ND : vdim);

   const int sm_size = 2*d1d*q1d + 4*q1d*q1d;

   const int GRID = USE_SMEM ? 0 : 128;
   double *gmem = nullptr;
   static Vector *d_buffer = nullptr;
   if (!USE_SMEM)
   {
      if (!d_buffer)
      {
         d_buffer = new Vector;
         d_buffer->UseDevice(true);
      }
      d_buffer->SetSize(sm_size*GRID);
      gmem = d_buffer->Write();
   }

   MFEM_FORALL_3D_GRID(e, NE, q1d,q1d,1, GRID,
   {
      if (M(e) < 1.0) { return; }

      const int bid = MFEM_BLOCK_ID(x);
      const int sm_SIZE = 2*D1D*Q1D + 4*Q1D*Q1D;
      MFEM_SHARED double SMEM[USE_SMEM ? sm_SIZE : 1];
      double *sm = USE_SMEM ? SMEM : (gmem + sm_size*bid);

      const DeviceMatrix Bt(DeviceMemAlloc(sm,q1d*d1d),d1d,q1d);
      const DeviceMatrix Gt(DeviceMemAlloc(sm,q1d*d1d),d1d,q1d);
      kernels::internal::LoadBGt(d1d,q1d,B,G,Bt,Gt);

      const DeviceMatrix QQ0(DeviceMemAlloc(sm,q1d*q1d),q1d,q1d);
      const DeviceMatrix QQ1(DeviceMemAlloc(sm,q1d*q1d),q1d,q1d);

      const DeviceMatrix DQ0(DeviceMemAlloc(sm,d1d*q1d),d1d,q1d);
      const DeviceMatrix DQ1(DeviceMemAlloc(sm,d1d*q1d),d1d,q1d);

      for (int c = 0; c < cdim/DIM; ++c)
      {
         const double cst_val0 = C(0,c,0,0,0);
         const double cst_val1 = C(1,c,0,0,0);

         MFEM_FOREACH_THREAD(qx,x,q1d)
         {
            MFEM_FOREACH_THREAD(qy,y,q1d)
            {
               double Jloc[4],Jinv[4];
               Jloc[0] = J(qx,qy,0,0,e);
               Jloc[1] = J(qx,qy,1,0,e);
               Jloc[2] = J(qx,qy,0,1,e);
               Jloc[3] = J(qx,qy,1,1,e);
               const double detJ = kernels::Det<2>(Jloc);
               kernels::CalcInverse<2>(Jloc, Jinv);
               const double weight = W(qx,qy);
               const double u = cst_coeff ? cst_val0 : C(0,c,qx,qy,e);
               const double v = cst_coeff ? cst_val1 : C(1,c,qx,qy,e);
               QQ0(qy,qx) = Jinv[0]*u + Jinv[2]*v;
               QQ1(qy,qx) = Jinv[1]*u + Jinv[3]*v;
               QQ0(qy,qx) *= weight * detJ;
               QQ1(qy,qx) *= weight * detJ;
            }
         }
         MFEM_SYNC_THREAD;
         kernels::internal::Atomic2DGradTranspose(d1d,q1d,Bt,Gt,
                                                  QQ0,QQ1,DQ0,DQ1,
                                                  I,Y,c,e,byVDIM);
      }
   });
}

template<int D1D=0, int Q1D=0> static
void VectorDomainLFGradIntegratorAssemble3D(const int vdim,
                                            const bool byVDIM,
                                            const int ND,
                                            const int NE,
                                            const int d1d,
                                            const int q1d,
                                            const double *marks,
                                            const double *b,
                                            const double *g,
                                            const int *idx,
                                            const double *jacobians,
                                            const double *weights,
                                            const Vector &coeff,
                                            double *y)
{
   constexpr int DIM = 3;
   constexpr bool USE_SMEM = D1D > 0 && Q1D > 0;

   const int cdim = vdim == 1 ? DIM : vdim;
   const bool cst_coeff = coeff.Size() == cdim;

   const auto F = coeff.Read();
   const auto M = Reshape(marks, NE);
   const auto B = Reshape(b, q1d,d1d);
   const auto G = Reshape(g, q1d,d1d);
   const auto J = Reshape(jacobians, q1d,q1d,q1d,DIM,DIM,NE);
   const auto W = Reshape(weights, q1d,q1d,q1d);
   const auto I = Reshape(idx, d1d,d1d,d1d, NE);
   const auto C = cst_coeff ?
                  Reshape(F,DIM,cdim/DIM,1,1,1,1):
                  Reshape(F,DIM,cdim/DIM,q1d,q1d,q1d,NE);

   auto Y = Reshape(y, byVDIM ? vdim : ND, byVDIM ? ND : vdim);

   const int sm_size = 2*q1d*d1d + 6*q1d*q1d*q1d;

   const int GRID = USE_SMEM ? 0 : 128;
   double *gmem = nullptr;
   static Vector *d_buffer = nullptr;
   if (!USE_SMEM)
   {
      if (!d_buffer)
      {
         d_buffer = new Vector;
         d_buffer->UseDevice(true);
      }
      d_buffer->SetSize(sm_size*GRID);
      gmem = d_buffer->Write();
   }

   MFEM_FORALL_3D_GRID(e, NE, q1d,q1d,1, GRID,
   {
      if (M(e) < 1.0) { return; }

      const int bid = MFEM_BLOCK_ID(x);
      const int sm_SIZE = 2*Q1D*D1D + 6*Q1D*Q1D*Q1D;
      MFEM_SHARED double SMEM[USE_SMEM ? sm_SIZE : 1];
      double *sm = USE_SMEM ? SMEM : (gmem + sm_size*bid);

      const DeviceMatrix Bt(DeviceMemAlloc(sm,q1d*d1d),d1d,q1d);
      const DeviceMatrix Gt(DeviceMemAlloc(sm,q1d*d1d),d1d,q1d);
      kernels::internal::LoadBGt(d1d,q1d,B,G,Bt,Gt);

      const DeviceCube QQ0(DeviceMemAlloc(sm,q1d*q1d*q1d),q1d,q1d,q1d);
      const DeviceCube QQ1(DeviceMemAlloc(sm,q1d*q1d*q1d),q1d,q1d,q1d);
      const DeviceCube QQ2(DeviceMemAlloc(sm,q1d*q1d*q1d),q1d,q1d,q1d);

      const DeviceCube QD0(DeviceMemAlloc(sm,q1d*q1d*q1d),q1d,q1d,d1d);
      const DeviceCube QD1(DeviceMemAlloc(sm,q1d*q1d*q1d),q1d,q1d,d1d);
      const DeviceCube QD2(DeviceMemAlloc(sm,q1d*q1d*q1d),q1d,q1d,d1d);

      const DeviceCube DD0(QQ0,q1d,d1d,d1d);
      const DeviceCube DD1(QQ1,q1d,d1d,d1d);
      const DeviceCube DD2(QQ2,q1d,d1d,d1d);

      for (int c = 0; c < cdim/DIM; ++c)
      {
         const double cst_val_0 = C(0,c,0,0,0,0);
         const double cst_val_1 = C(1,c,0,0,0,0);
         const double cst_val_2 = C(2,c,0,0,0,0);

         MFEM_FOREACH_THREAD(qx,x,q1d)
         {
            MFEM_FOREACH_THREAD(qy,y,q1d)
            {
               for (int qz = 0; qz < q1d; ++qz)
               {
                  double Jloc[9],Jinv[9];
                  for (int col = 0; col < 3; col++)
                  {
                     for (int row = 0; row < 3; row++)
                     {
                        Jloc[row+3*col] = J(qx,qy,qz,row,col,e);
                     }
                  }
                  const double detJ = kernels::Det<3>(Jloc);
                  kernels::CalcInverse<3>(Jloc, Jinv);
                  const double weight = W(qx,qy,qz);
                  const double u = cst_coeff ? cst_val_0 : C(0,c,qx,qy,qz,e);
                  const double v = cst_coeff ? cst_val_1 : C(1,c,qx,qy,qz,e);
                  const double w = cst_coeff ? cst_val_2 : C(2,c,qx,qy,qz,e);
                  QQ0(qz,qy,qx) = Jinv[0]*u + Jinv[3]*v + Jinv[6]*w;
                  QQ1(qz,qy,qx) = Jinv[1]*u + Jinv[4]*v + Jinv[7]*w;
                  QQ2(qz,qy,qx) = Jinv[2]*u + Jinv[5]*v + Jinv[8]*w;
                  QQ0(qz,qy,qx) *= weight * detJ;
                  QQ1(qz,qy,qx) *= weight * detJ;
                  QQ2(qz,qy,qx) *= weight * detJ;
               }
            }
         }
         MFEM_SYNC_THREAD;
         kernels::internal::Atomic3DGrad(d1d,q1d,Bt,Gt,
                                         QQ0,QQ1,QQ2,
                                         QD0,QD1,QD2,
                                         DD0,DD1,DD2,
                                         I,Y,c,e,byVDIM);
      }
   });
}

} // namespace linearform_extension

} // namespace internal

} // namespace mfem
