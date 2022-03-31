// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
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
template<int D=0, int Q=0> static
void VectorDomainLFGradIntegratorAssemble2D(const int vdim,
                                            const bool byVDIM,
                                            const int ND,
                                            const int NE,
                                            const int d,
                                            const int q,
                                            const int *markers,
                                            const double *b,
                                            const double *g,
                                            const int *idx,
                                            const double *jacobians,
                                            const double *detJ,
                                            const double *weights,
                                            const Vector &coeff,
                                            double *y)
{
   constexpr int DIM = 2;
   constexpr bool USE_SMEM = D > 0 && Q > 0;

   const bool cst_coeff = coeff.Size() == vdim*DIM;

   const auto F = coeff.Read();
   const auto M = Reshape(markers, NE);
   const auto B = Reshape(b, q,d);
   const auto G = Reshape(g, q,d);
   const auto J = Reshape(jacobians, q,q, DIM,DIM, NE);
   const auto DetJ = Reshape(detJ, q,q, NE);
   const auto W = Reshape(weights, q,q);
   const auto I = Reshape(idx, d,d, NE);
   const auto C = cst_coeff ?
                  Reshape(F,DIM,vdim,1,1,1):
                  Reshape(F,DIM,vdim,q,q,NE);

   auto Y = Reshape(y, byVDIM ? vdim : ND, byVDIM ? ND : vdim);

   const int sm_size = 2*d*q + 4*q*q;
   const int GRID = USE_SMEM ? 0 : 128;
   double *gmem = kernels::internal::pool::SetSize<GRID>(sm_size);

   MFEM_FORALL_3D_GRID(e, NE, q,q,1, GRID,
   {
      if (M(e) == 0) { /* ignore */ return; }

      const int bid = MFEM_BLOCK_ID(x);
      constexpr int SM_SIZE = 2*D*Q + 4*Q*Q;
      constexpr bool USE_SMEM = D > 0 && Q > 0;
      MFEM_SHARED double SMEM[USE_SMEM ? SM_SIZE : 1];
      double *sm = USE_SMEM ? SMEM : (gmem + sm_size*bid);

      const DeviceMatrix Bt(kernels::internal::pool::Alloc(sm,q*d), d,q);
      const DeviceMatrix Gt(kernels::internal::pool::Alloc(sm,q*d), d,q);
      kernels::internal::load::BGt(d,q,B,G,Bt,Gt);

      const DeviceMatrix QQ0(kernels::internal::pool::Alloc(sm,q*q), q,q);
      const DeviceMatrix QQ1(kernels::internal::pool::Alloc(sm,q*q), q,q);

      const DeviceMatrix DQ0(kernels::internal::pool::Alloc(sm,d*q), d,q);
      const DeviceMatrix DQ1(kernels::internal::pool::Alloc(sm,d*q), d,q);

      for (int c = 0; c < vdim; ++c)
      {
         const double cst_val0 = C(0,c,0,0,0);
         const double cst_val1 = C(1,c,0,0,0);

         MFEM_FOREACH_THREAD(x,x,q)
         {
            MFEM_FOREACH_THREAD(y,y,q)
            {
               double Jloc[4], Jinv[4];
               Jloc[0] = J(x,y,0,0,e);
               Jloc[1] = J(x,y,1,0,e);
               Jloc[2] = J(x,y,0,1,e);
               Jloc[3] = J(x,y,1,1,e);
               const double detJ = DetJ(x,y,e);
               kernels::CalcInverse<2>(Jloc, Jinv);
               const double weight = W(x,y);
               const double u = cst_coeff ? cst_val0 : C(0,c,x,y,e);
               const double v = cst_coeff ? cst_val1 : C(1,c,x,y,e);
               QQ0(y,x) = Jinv[0]*u + Jinv[2]*v;
               QQ1(y,x) = Jinv[1]*u + Jinv[3]*v;
               QQ0(y,x) *= weight * detJ;
               QQ1(y,x) *= weight * detJ;
            }
         }
         MFEM_SYNC_THREAD;
         kernels::internal::grad::fast::MultTranspose(d,q,Bt,Gt,
                                                      QQ0,QQ1,DQ0,DQ1,
                                                      I,Y,c,e,byVDIM);
      }
   });
}

template<int D=0, int Q=0> static
void VectorDomainLFGradIntegratorAssemble3D(const int vdim,
                                            const bool byVDIM,
                                            const int ND,
                                            const int NE,
                                            const int d,
                                            const int q,
                                            const int *markers,
                                            const double *b,
                                            const double *g,
                                            const int *idx,
                                            const double *jacobians,
                                            const double *detJ,
                                            const double *weights,
                                            const Vector &coeff,
                                            double *y)
{
   constexpr int DIM = 3;
   constexpr bool USE_SMEM = D > 0 && Q > 0;

   const bool cst_coeff = coeff.Size() == vdim*DIM;

   const auto F = coeff.Read();
   const auto M = Reshape(markers, NE);
   const auto B = Reshape(b, q,d);
   const auto G = Reshape(g, q,d);
   const auto J = Reshape(jacobians, q,q,q, DIM,DIM, NE);
   const auto DetJ = Reshape(detJ, q,q,q, NE);
   const auto W = Reshape(weights, q,q,q);
   const auto I = Reshape(idx, d,d,d, NE);
   const auto C = cst_coeff ?
                  Reshape(F,DIM,vdim,1,1,1,1):
                  Reshape(F,DIM,vdim,q,q,q,NE);

   auto Y = Reshape(y, byVDIM ? vdim : ND, byVDIM ? ND : vdim);

   const int sm_size = 2*q*d + 6*q*q*q;

   const int GRID = USE_SMEM ? 0 : 128;
   double *gmem = kernels::internal::pool::SetSize<GRID>(sm_size);

   MFEM_FORALL_3D_GRID(e, NE, q,q,1, GRID,
   {
      if (M(e) == 0) { /* ignore */ return; }

      const int bid = MFEM_BLOCK_ID(x);
      constexpr int SM_SIZE = 2*Q*D + 6*Q*Q*Q;
      constexpr bool USE_SMEM = D > 0 && Q > 0;
      MFEM_SHARED double SMEM[USE_SMEM ? SM_SIZE : 1];
      double *sm = USE_SMEM ? SMEM : (gmem + sm_size*bid);

      const DeviceMatrix Bt(kernels::internal::pool::Alloc(sm,q*d), d,q);
      const DeviceMatrix Gt(kernels::internal::pool::Alloc(sm,q*d), d,q);
      kernels::internal::load::BGt(d,q,B,G,Bt,Gt);

      const DeviceCube QQ0(kernels::internal::pool::Alloc(sm,q*q*q), q,q,q);
      const DeviceCube QQ1(kernels::internal::pool::Alloc(sm,q*q*q), q,q,q);
      const DeviceCube QQ2(kernels::internal::pool::Alloc(sm,q*q*q), q,q,q);

      const DeviceCube QD0(kernels::internal::pool::Alloc(sm,q*q*q), q,q,d);
      const DeviceCube QD1(kernels::internal::pool::Alloc(sm,q*q*q), q,q,d);
      const DeviceCube QD2(kernels::internal::pool::Alloc(sm,q*q*q), q,q,d);

      const DeviceCube DD0(QQ0,q,d,d);
      const DeviceCube DD1(QQ1,q,d,d);
      const DeviceCube DD2(QQ2,q,d,d);

      for (int c = 0; c < vdim; ++c)
      {
         const double cst_val_0 = C(0,c,0,0,0,0);
         const double cst_val_1 = C(1,c,0,0,0,0);
         const double cst_val_2 = C(2,c,0,0,0,0);

         MFEM_FOREACH_THREAD(x,x,q)
         {
            MFEM_FOREACH_THREAD(y,y,q)
            {
               for (int z = 0; z < q; ++z)
               {
                  double Jloc[9], Jinv[9];
                  for (int j = 0; j < 3; j++)
                  {
                     for (int i = 0; i < 3; i++)
                     {
                        Jloc[i+3*j] = J(x,y,z,i,j,e);
                     }
                  }
                  const double detJ = DetJ(x,y,z,e);
                  kernels::CalcInverse<3>(Jloc, Jinv);
                  const double weight = W(x,y,z);
                  const double u = cst_coeff ? cst_val_0 : C(0,c,x,y,z,e);
                  const double v = cst_coeff ? cst_val_1 : C(1,c,x,y,z,e);
                  const double w = cst_coeff ? cst_val_2 : C(2,c,x,y,z,e);
                  QQ0(z,y,x) = Jinv[0]*u + Jinv[3]*v + Jinv[6]*w;
                  QQ1(z,y,x) = Jinv[1]*u + Jinv[4]*v + Jinv[7]*w;
                  QQ2(z,y,x) = Jinv[2]*u + Jinv[5]*v + Jinv[8]*w;
                  QQ0(z,y,x) *= weight * detJ;
                  QQ1(z,y,x) *= weight * detJ;
                  QQ2(z,y,x) *= weight * detJ;
               }
            }
         }
         MFEM_SYNC_THREAD;
         kernels::internal::grad::fast::MultTranspose(d,q,Bt,Gt,
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
