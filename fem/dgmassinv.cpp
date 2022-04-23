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

#include "dgmassinv.hpp"
#include "bilinearform.hpp"
#include "../general/forall.hpp"

namespace mfem
{

DGMassInverse::DGMassInverse(FiniteElementSpace &fes_, Coefficient *coeff,
                             int btype)
   : Solver(fes_.GetTrueVSize()),
     fes_orig(fes_),
     fec(fes_.GetMaxElementOrder(), fes_.GetMesh()->Dimension(), btype),
     fes(fes_orig.GetMesh(), &fec)
{
   MFEM_VERIFY(fes.IsDGSpace(), "Space must be DG.");
   MFEM_VERIFY(!fes.IsVariableOrder(), "Variable orders not supported.");

   if (coeff) { m = new MassIntegrator(*coeff); }
   else { m = new MassIntegrator; }

   BilinearForm M(&fes);
   M.AddDomainIntegrator(m);
   M.UseExternalIntegrators();
   M.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   M.Assemble();

   diag_inv.SetSize(height);
   M.AssembleDiagonal(diag_inv);

   // TODO: need to move this FORALL into its own function, can't have a
   // FORALL loop in constructor.
   // auto dinv = diag_inv.ReadWrite();
   //MFEM_FORALL(i, height, dinv[i] = 1.0/dinv[i]; );
   auto dinv = diag_inv.HostReadWrite();
   for (int i = 0; i < height; ++i) { dinv[i] = 1.0/dinv[i]; }

   r.SetSize(height);
   d.SetSize(height);
   z.SetSize(height);
}

DGMassInverse::DGMassInverse(FiniteElementSpace &fes_, int btype)
   : DGMassInverse(fes_, nullptr, btype) { }

MFEM_HOST_DEVICE inline
void PAMassApply2D(const int e,
                   const int NE,
                   const double *b_,
                   const double *bt_,
                   const double *d_,
                   const double *x_,
                   double *y_,
                   const int d1d = 0,
                   const int q1d = 0)
{
   const int D1D = d1d;
   const int Q1D = q1d;
   MFEM_VERIFY_KERNEL(D1D <= MAX_D1D, "Size too large");
   MFEM_VERIFY_KERNEL(Q1D <= MAX_Q1D, "Size too large");
   auto B = ConstDeviceMatrix(b_, Q1D, D1D);
   auto Bt = ConstDeviceMatrix(bt_, D1D, Q1D);
   auto D = ConstDeviceCube(d_, Q1D, Q1D, NE);
   auto X = ConstDeviceCube(x_, D1D, D1D, NE);
   auto Y = DeviceCube(y_, D1D, D1D, NE);

   for (int dy = 0; dy < D1D; ++dy)
   {
      for (int dx = 0; dx < D1D; ++dx)
      {
         Y(dx, dy, e) = 0.0;
      }
   }

   constexpr int max_D1D = MAX_D1D;
   constexpr int max_Q1D = MAX_Q1D;
   double sol_xy[max_Q1D][max_Q1D];
   for (int qy = 0; qy < Q1D; ++qy)
   {
      for (int qx = 0; qx < Q1D; ++qx)
      {
         sol_xy[qy][qx] = 0.0;
      }
   }
   for (int dy = 0; dy < D1D; ++dy)
   {
      double sol_x[max_Q1D];
      for (int qy = 0; qy < Q1D; ++qy)
      {
         sol_x[qy] = 0.0;
      }
      for (int dx = 0; dx < D1D; ++dx)
      {
         const double s = X(dx,dy,e);
         for (int qx = 0; qx < Q1D; ++qx)
         {
            sol_x[qx] += B(qx,dx)* s;
         }
      }
      for (int qy = 0; qy < Q1D; ++qy)
      {
         const double d2q = B(qy,dy);
         for (int qx = 0; qx < Q1D; ++qx)
         {
            sol_xy[qy][qx] += d2q * sol_x[qx];
         }
      }
   }
   for (int qy = 0; qy < Q1D; ++qy)
   {
      for (int qx = 0; qx < Q1D; ++qx)
      {
         sol_xy[qy][qx] *= D(qx,qy,e);
      }
   }
   for (int qy = 0; qy < Q1D; ++qy)
   {
      double sol_x[max_D1D];
      for (int dx = 0; dx < D1D; ++dx)
      {
         sol_x[dx] = 0.0;
      }
      for (int qx = 0; qx < Q1D; ++qx)
      {
         const double s = sol_xy[qy][qx];
         for (int dx = 0; dx < D1D; ++dx)
         {
            sol_x[dx] += Bt(dx,qx) * s;
         }
      }
      for (int dy = 0; dy < D1D; ++dy)
      {
         const double q2d = Bt(dy,qy);
         for (int dx = 0; dx < D1D; ++dx)
         {
            Y(dx,dy,e) += q2d * sol_x[dx];
         }
      }
   }
}

template<int T_D1D = 0, int T_Q1D = 0>
MFEM_HOST_DEVICE inline
void SmemPAMassApply2D(const int e,
                       const int NE,
                       const double *b_,
                       const double *bt_,
                       const double *d_,
                       const double *x_,
                       double *y_)
{
   const int d1d = 0;
   const int q1d = 0;

   MFEM_CONTRACT_VAR(bt_);
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int NBZ = 1;
   constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
   constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
   constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
   MFEM_VERIFY_KERNEL(D1D <= MD1, "Size too large");
   MFEM_VERIFY_KERNEL(Q1D <= MQ1, "Size too large");

   auto b = ConstDeviceMatrix(b_, Q1D, D1D);
   auto D = ConstDeviceCube(d_, Q1D, Q1D, NE);
   auto x = ConstDeviceCube(x_, D1D, D1D, NE);
   auto Y = DeviceCube(y_, D1D, D1D, NE);

   const int tidz = MFEM_THREAD_ID(z);

   MFEM_SHARED double BBt[MQ1*MD1];
   double (*B)[MD1] = (double (*)[MD1]) BBt;
   double (*Bt)[MQ1] = (double (*)[MQ1]) BBt;
   MFEM_SHARED double sm0[NBZ][MDQ*MDQ];
   MFEM_SHARED double sm1[NBZ][MDQ*MDQ];
   double (*X)[MD1] = (double (*)[MD1]) (sm0 + tidz);
   double (*DQ)[MQ1] = (double (*)[MQ1]) (sm1 + tidz);
   double (*QQ)[MQ1] = (double (*)[MQ1]) (sm0 + tidz);
   double (*QD)[MD1] = (double (*)[MD1]) (sm1 + tidz);


   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         X[dy][dx] = x(dx,dy,e);
      }
   }
   if (tidz == 0)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(q,x,Q1D)
         {
            B[q][dy] = b(q,dy);
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double dq = 0.0;
         for (int dx = 0; dx < D1D; ++dx)
         {
            dq += X[dy][dx] * B[qx][dx];
         }
         DQ[dy][qx] = dq;
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double qq = 0.0;
         for (int dy = 0; dy < D1D; ++dy)
         {
            qq += DQ[dy][qx] * B[qy][dy];
         }
         QQ[qy][qx] = qq * D(qx, qy, e);
      }
   }
   MFEM_SYNC_THREAD;
   if (tidz == 0)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(q,x,Q1D)
         {
            Bt[dy][q] = b(q,dy);
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         double dq = 0.0;
         for (int qx = 0; qx < Q1D; ++qx)
         {
            dq += QQ[qy][qx] * Bt[dx][qx];
         }
         QD[qy][dx] = dq;
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         double dd = 0.0;
         for (int qy = 0; qy < Q1D; ++qy)
         {
            dd += (QD[qy][dx] * Bt[dy][qy]);
         }
         Y(dx, dy, e) = dd; // changed from += to =
      }
   }
}

MFEM_HOST_DEVICE inline
void PAMassApply3D(const int e,
                   const int NE,
                   const double *b_,
                   const double *bt_,
                   const double *d_,
                   const double *x_,
                   double *y_,
                   const int d1d,
                   const int q1d)
{
   const int D1D = d1d;
   const int Q1D = q1d;
   MFEM_VERIFY_KERNEL(D1D <= MAX_D1D, "Size too large");
   MFEM_VERIFY_KERNEL(Q1D <= MAX_Q1D, "Size too large");
   auto B = ConstDeviceMatrix(b_, Q1D, D1D);
   auto Bt = ConstDeviceMatrix(bt_, D1D, Q1D);
   auto D = DeviceTensor<4,const double>(d_, Q1D, Q1D, Q1D, NE);
   auto X = DeviceTensor<4,const double>(x_, D1D, D1D, D1D, NE);
   auto Y = DeviceTensor<4,double>(y_, D1D, D1D, D1D, NE);

   for (int dz = 0; dz < D1D; ++dz)
   {
      for (int dy = 0; dy < D1D; ++dy)
      {
         for (int dx = 0; dx < D1D; ++dx)
         {
            Y(dx, dy, dz, e) = 0.0;
         }
      }
   }

   constexpr int max_D1D = MAX_D1D;
   constexpr int max_Q1D = MAX_Q1D;
   double sol_xyz[max_Q1D][max_Q1D][max_Q1D];
   for (int qz = 0; qz < Q1D; ++qz)
   {
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            sol_xyz[qz][qy][qx] = 0.0;
         }
      }
   }
   for (int dz = 0; dz < D1D; ++dz)
   {
      double sol_xy[max_Q1D][max_Q1D];
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            sol_xy[qy][qx] = 0.0;
         }
      }
      for (int dy = 0; dy < D1D; ++dy)
      {
         double sol_x[max_Q1D];
         for (int qx = 0; qx < Q1D; ++qx)
         {
            sol_x[qx] = 0;
         }
         for (int dx = 0; dx < D1D; ++dx)
         {
            const double s = X(dx,dy,dz,e);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_x[qx] += B(qx,dx) * s;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            const double wy = B(qy,dy);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_xy[qy][qx] += wy * sol_x[qx];
            }
         }
      }
      for (int qz = 0; qz < Q1D; ++qz)
      {
         const double wz = B(qz,dz);
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_xyz[qz][qy][qx] += wz * sol_xy[qy][qx];
            }
         }
      }
   }
   for (int qz = 0; qz < Q1D; ++qz)
   {
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            sol_xyz[qz][qy][qx] *= D(qx,qy,qz,e);
         }
      }
   }
   for (int qz = 0; qz < Q1D; ++qz)
   {
      double sol_xy[max_D1D][max_D1D];
      for (int dy = 0; dy < D1D; ++dy)
      {
         for (int dx = 0; dx < D1D; ++dx)
         {
            sol_xy[dy][dx] = 0;
         }
      }
      for (int qy = 0; qy < Q1D; ++qy)
      {
         double sol_x[max_D1D];
         for (int dx = 0; dx < D1D; ++dx)
         {
            sol_x[dx] = 0;
         }
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const double s = sol_xyz[qz][qy][qx];
            for (int dx = 0; dx < D1D; ++dx)
            {
               sol_x[dx] += Bt(dx,qx) * s;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            const double wy = Bt(dy,qy);
            for (int dx = 0; dx < D1D; ++dx)
            {
               sol_xy[dy][dx] += wy * sol_x[dx];
            }
         }
      }
      for (int dz = 0; dz < D1D; ++dz)
      {
         const double wz = Bt(dz,qz);
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               Y(dx,dy,dz,e) += wz * sol_xy[dy][dx];
            }
         }
      }
   }
}

template<int T_D1D, int T_Q1D>
MFEM_HOST_DEVICE inline
void SmemPAMassApply3D(const int e,
                       const int NE,
                       const double *b_,
                       const double *bt_,
                       const double *d_,
                       const double *x_,
                       double *y_)
{
   MFEM_CONTRACT_VAR(bt_);
   constexpr int D1D = T_D1D ? T_D1D : 1;
   constexpr int Q1D = T_Q1D ? T_Q1D : 1;
   constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
   constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
   constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
   MFEM_VERIFY_KERNEL(D1D <= MD1, "Size too large.");
   MFEM_VERIFY_KERNEL(Q1D <= MQ1, "Size too large.");

   auto b = ConstDeviceMatrix(b_, Q1D, D1D);
   auto d = DeviceTensor<4,const double>(d_, Q1D, Q1D, Q1D, NE);
   auto x = DeviceTensor<4,const double>(x_, D1D, D1D, D1D, NE);
   auto y = DeviceTensor<4,double>(y_, D1D, D1D, D1D, NE);

   MFEM_SHARED double sDQ[MQ1*MD1];
   double (*B)[MD1] = (double (*)[MD1]) sDQ;
   double (*Bt)[MQ1] = (double (*)[MQ1]) sDQ;
   MFEM_SHARED double sm0[MDQ*MDQ*MDQ];
   MFEM_SHARED double sm1[MDQ*MDQ*MDQ];
   double (*X)[MD1][MD1]   = (double (*)[MD1][MD1]) sm0;
   double (*DDQ)[MD1][MQ1] = (double (*)[MD1][MQ1]) sm1;
   double (*DQQ)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) sm0;
   double (*QQQ)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) sm1;
   double (*QQD)[MQ1][MD1] = (double (*)[MQ1][MD1]) sm0;
   double (*QDD)[MD1][MD1] = (double (*)[MD1][MD1]) sm1;
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         MFEM_UNROLL(MD1)
         for (int dz = 0; dz < D1D; ++dz)
         {
            X[dz][dy][dx] = x(dx,dy,dz,e);
         }
      }
      MFEM_FOREACH_THREAD(dx,x,Q1D)
      {
         B[dx][dy] = b(dx,dy);
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double u[D1D];
         MFEM_UNROLL(MD1)
         for (int dz = 0; dz < D1D; dz++)
         {
            u[dz] = 0;
         }
         MFEM_UNROLL(MD1)
         for (int dx = 0; dx < D1D; ++dx)
         {
            MFEM_UNROLL(MD1)
            for (int dz = 0; dz < D1D; ++dz)
            {
               u[dz] += X[dz][dy][dx] * B[qx][dx];
            }
         }
         MFEM_UNROLL(MD1)
         for (int dz = 0; dz < D1D; ++dz)
         {
            DDQ[dz][dy][qx] = u[dz];
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double u[D1D];
         MFEM_UNROLL(MD1)
         for (int dz = 0; dz < D1D; dz++)
         {
            u[dz] = 0;
         }
         MFEM_UNROLL(MD1)
         for (int dy = 0; dy < D1D; ++dy)
         {
            MFEM_UNROLL(MD1)
            for (int dz = 0; dz < D1D; dz++)
            {
               u[dz] += DDQ[dz][dy][qx] * B[qy][dy];
            }
         }
         MFEM_UNROLL(MD1)
         for (int dz = 0; dz < D1D; dz++)
         {
            DQQ[dz][qy][qx] = u[dz];
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double u[Q1D];
         MFEM_UNROLL(MQ1)
         for (int qz = 0; qz < Q1D; qz++)
         {
            u[qz] = 0;
         }
         MFEM_UNROLL(MD1)
         for (int dz = 0; dz < D1D; ++dz)
         {
            MFEM_UNROLL(MQ1)
            for (int qz = 0; qz < Q1D; qz++)
            {
               u[qz] += DQQ[dz][qy][qx] * B[qz][dz];
            }
         }
         MFEM_UNROLL(MQ1)
         for (int qz = 0; qz < Q1D; qz++)
         {
            QQQ[qz][qy][qx] = u[qz] * d(qx,qy,qz,e);
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(di,y,D1D)
   {
      MFEM_FOREACH_THREAD(q,x,Q1D)
      {
         Bt[di][q] = b(q,di);
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         double u[Q1D];
         MFEM_UNROLL(MQ1)
         for (int qz = 0; qz < Q1D; ++qz)
         {
            u[qz] = 0;
         }
         MFEM_UNROLL(MQ1)
         for (int qx = 0; qx < Q1D; ++qx)
         {
            MFEM_UNROLL(MQ1)
            for (int qz = 0; qz < Q1D; ++qz)
            {
               u[qz] += QQQ[qz][qy][qx] * Bt[dx][qx];
            }
         }
         MFEM_UNROLL(MQ1)
         for (int qz = 0; qz < Q1D; ++qz)
         {
            QQD[qz][qy][dx] = u[qz];
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         double u[Q1D];
         MFEM_UNROLL(MQ1)
         for (int qz = 0; qz < Q1D; ++qz)
         {
            u[qz] = 0;
         }
         MFEM_UNROLL(MQ1)
         for (int qy = 0; qy < Q1D; ++qy)
         {
            MFEM_UNROLL(MQ1)
            for (int qz = 0; qz < Q1D; ++qz)
            {
               u[qz] += QQD[qz][qy][dx] * Bt[dy][qy];
            }
         }
         MFEM_UNROLL(MQ1)
         for (int qz = 0; qz < Q1D; ++qz)
         {
            QDD[qz][dy][dx] = u[qz];
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         double u[D1D];
         MFEM_UNROLL(MD1)
         for (int dz = 0; dz < D1D; ++dz)
         {
            u[dz] = 0;
         }
         MFEM_UNROLL(MQ1)
         for (int qz = 0; qz < Q1D; ++qz)
         {
            MFEM_UNROLL(MD1)
            for (int dz = 0; dz < D1D; ++dz)
            {
               u[dz] += QDD[qz][dy][dx] * Bt[dz][qz];
            }
         }
         MFEM_UNROLL(MD1)
         for (int dz = 0; dz < D1D; ++dz)
         {
            y(dx,dy,dz,e) = u[dz]; // changed from += to =
         }
      }
   }
}

template <int DIM, int D1D, int Q1D>
MFEM_HOST_DEVICE inline
void DGMassApply(const int e,
                 const int NE,
                 const double *B,
                 const double *Bt,
                 const double *pa_data,
                 const double *x,
                 double *y,
                 const int d1d = 0,
                 const int q1d = 0)
{
   constexpr bool use_smem = (D1D > 0 && Q1D > 0);
   if (use_smem)
   {
      if (DIM == 2)
      {
         SmemPAMassApply2D<D1D,Q1D>(e, NE, B, Bt, pa_data, x, y);
      }
      else if (DIM == 3)
      {
         SmemPAMassApply3D<D1D,Q1D>(e, NE, B, Bt, pa_data, x, y);
      }
      else
      {
         MFEM_ABORT_KERNEL("Unsupported dimension.");
      }
   }
   else
   {
      if (DIM == 2)
      {
         PAMassApply2D(e, NE, B, Bt, pa_data, x, y, d1d, q1d);
      }
      else if (DIM == 3)
      {
         PAMassApply3D(e, NE, B, Bt, pa_data, x, y, d1d, q1d);
      }
      else
      {
         MFEM_ABORT_KERNEL("Unsupported dimension.");
      }
   }
}

MFEM_HOST_DEVICE inline
void DGMassPreconditioner(const int e,
                          const int NE,
                          const int ND,
                          const double *dinv,
                          const double *x,
                          double *y)
{
   const auto X = ConstDeviceMatrix(x, ND, NE);
   const auto D = ConstDeviceMatrix(dinv, ND, NE);
   auto Y = DeviceMatrix(y, ND, NE);

   const int tid = MFEM_THREAD_ID(x) + MFEM_THREAD_SIZE(x)*MFEM_THREAD_ID(y);
   const int bxy = MFEM_THREAD_SIZE(x)*MFEM_THREAD_SIZE(y);

   for (int i = tid; i < ND; i += bxy)
   {
      Y(i, e) = D(i, e)*X(i, e);
   }
}

MFEM_HOST_DEVICE inline
void DGMassAxpy(const int e,
                const int NE,
                const int ND,
                const double a,
                const double *x,
                const double b,
                const double *y,
                double *z)
{
   const auto X = ConstDeviceMatrix(x, ND, NE);
   const auto Y = ConstDeviceMatrix(y, ND, NE);
   auto Z = DeviceMatrix(z, ND, NE);

   const int tid = MFEM_THREAD_ID(x) + MFEM_THREAD_SIZE(x)*MFEM_THREAD_ID(y);
   const int bxy = MFEM_THREAD_SIZE(x)*MFEM_THREAD_SIZE(y);

   for (int i = tid; i < ND; i += bxy)
   {
      Z(i, e) = a*X(i, e) + b*Y(i, e);
   }
}

template <int NB>
MFEM_HOST_DEVICE inline
double DGMassDot(const int e,
                 const int NE,
                 const int ND,
                 const double *x,
                 const double *y)
{
   const auto X = ConstDeviceMatrix(x, ND, NE);
   const auto Y = ConstDeviceMatrix(y, ND, NE);

   const int tid = MFEM_THREAD_ID(x) + MFEM_THREAD_SIZE(x)*MFEM_THREAD_ID(y);
   const int bxy = MFEM_THREAD_SIZE(x)*MFEM_THREAD_SIZE(y);

   MFEM_SHARED double s_dot[NB*NB];
   s_dot[tid] = 0.0;

   for (int i = tid; i < ND; i += bxy) { s_dot[tid] += X(i,e)*Y(i,e); }
   MFEM_SYNC_THREAD;

   if (bxy > 512 && tid + 512 < bxy) { s_dot[tid] += s_dot[tid + 512]; }
   MFEM_SYNC_THREAD;

   if (bxy > 256 && tid < 256 && tid + 256 < bxy) { s_dot[tid] += s_dot[tid + 256]; }
   MFEM_SYNC_THREAD;

   if (bxy > 128 && tid < 128 && tid + 128 < bxy) { s_dot[tid] += s_dot[tid + 128]; }
   MFEM_SYNC_THREAD;

   if (bxy > 64 && tid < 64 && tid + 64 < bxy) { s_dot[tid] += s_dot[tid + 64]; }
   MFEM_SYNC_THREAD;

   if (bxy > 32 && tid < 32 && tid + 32 < bxy) { s_dot[tid] += s_dot[tid + 32]; }
   MFEM_SYNC_THREAD;

   if (bxy > 16 && tid < 16 && tid + 16 < bxy) { s_dot[tid] += s_dot[tid + 16]; }
   MFEM_SYNC_THREAD;

   if (bxy > 8 && tid < 8 && tid + 8 < bxy) { s_dot[tid] += s_dot[tid + 8]; }
   MFEM_SYNC_THREAD;

   if (bxy > 4 && tid < 4 && tid + 4 < bxy) { s_dot[tid] += s_dot[tid + 4]; }
   MFEM_SYNC_THREAD;

   if (bxy > 2 && tid < 2 && tid + 2 < bxy) { s_dot[tid] += s_dot[tid + 2]; }
   MFEM_SYNC_THREAD;

   if (bxy > 1 && tid < 1 && tid + 1 < bxy) { s_dot[tid] += s_dot[tid + 1]; }
   MFEM_SYNC_THREAD;

   return s_dot[0];
}

template<int DIM, int D1D = 0, int Q1D = 0>
static void DGMassCGIteration(const int NE,
                              const Array<double> &B_,
                              const Array<double> &Bt_,
                              const Vector &pa_data_,
                              const Vector &dinv_,
                              const double rel_tol,
                              const double abs_tol,
                              const int max_iter,
                              const Vector &b_,
                              Vector &r_,
                              Vector &d_,
                              Vector &z_,
                              Vector &u_,
                              const int d1d = 0,
                              const int q1d = 0)
{
   const int ND = pow(d1d, DIM);

   const auto B = B_.Read();
   const auto Bt = Bt_.Read();
   const auto pa_data = pa_data_.Read();
   const auto dinv = dinv_.Read();
   const auto b = b_.Read();
   auto r = r_.Write();
   auto d = d_.Write();
   auto z = z_.Write();
   auto u = u_.ReadWrite();

   const int NB = Q1D ? Q1D : 1; // block size

   // printf("  El.     It.    (Br,r)\n");
   // printf("=============================\n");
   MFEM_FORALL_2D(e, NE, NB, NB, 1,
   {
      // const int tid = MFEM_THREAD_ID(x) + NB*MFEM_THREAD_ID(y);
      // int final_iter;
      // double final_norm;
      // bool converged;

      DGMassApply<DIM,D1D,Q1D>(e, NE, B, Bt, pa_data, u, r, d1d, q1d);
      DGMassAxpy(e, NE, ND, 1.0, b, -1.0, r, r); // r = b - r

      // TODO: get rid of extra memory usage for z
      DGMassPreconditioner(e, NE, ND, dinv, r, z);
      DGMassAxpy(e, NE, ND, 1.0, z, 0.0, z, d); // d = z

      double nom0 = DGMassDot<NB>(e, NE, ND, d, r);
      double nom = nom0;
      // MFEM_ASSERT(IsFinite(nom), "nom = " << nom);

      if (nom < 0.0)
      {
         return; // Not positive definite...
      }
      double r0 = fmax(nom*rel_tol*rel_tol, abs_tol*abs_tol);
      if (nom <= r0)
      {
         // converged = true;
         // final_iter = 0;
         // final_norm = sqrt(nom);
         return;
      }

      DGMassApply<DIM,D1D,Q1D>(e, NE, B, Bt, pa_data, d, z, d1d, q1d);
      double den = DGMassDot<NB>(e, NE, ND, z, d);
      if (den <= 0.0)
      {
         const double d2 = DGMassDot<NB>(e, NE, ND, d, d);
         if (d2 > 0.0) { printf("Not positive definite.\n"); }
         if (den == 0.0)
         {
            // converged = false;
            // final_iter = 0;
            // final_norm = sqrt(nom);
            return;
         }
      }

      // start iteration
      int i = 1;
      while (true)
      {
         const double alpha = nom/den;
         DGMassAxpy(e, NE, ND, 1.0, u, alpha, d, u); // u = u + alpha*d
         DGMassAxpy(e, NE, ND, 1.0, r, -alpha, z, r); // r = r - alpha*A*d

         DGMassPreconditioner(e, NE, ND, dinv, r, z);

         double betanom = DGMassDot<NB>(e, NE, ND, r, z);
         if (betanom < 0.0)
         {
            printf("Not positive definite.\n");
            // converged = false;
            // final_iter = i;
            return;
         }

         // if (tid == 0) { printf(" %4d    %4d    %10.6e\n", e, i, betanom); }

         if (betanom <= r0)
         {
            // converged = true;
            // final_iter = i;
            return;
         }

         if (++i > max_iter) { return; }

         const double beta = betanom/nom;
         DGMassAxpy(e, NE, ND, 1.0, z, beta, d, d); // d = z + beta*d
         DGMassApply<DIM,D1D,Q1D>(e, NE, B, Bt, pa_data, d, z, d1d, q1d); // z = A d
         den = DGMassDot<NB>(e, NE, ND, d, z);
         if (den <= 0.0)
         {
            const double d2 = DGMassDot<NB>(e, NE, ND, d, d);
            if (d2 > 0.0) { printf("Not positive definite.\n"); }
            if (den == 0.0)
            {
               // final_iter = i;
               return;
            }
         }
         nom = betanom;
      }
   });
}

void DGMassInverse::Mult(const Vector &Mu, Vector &u) const
{
   const int dim = fes.GetMesh()->Dimension();
   const int NE = fes.GetNE();
   const int d1d = m->dofs1D;
   const int q1d = m->quad1D;
   const auto &pa_data = m->pa_data;
   const auto &B = m->maps->B;
   const auto &Bt = m->maps->Bt;

   const int id = (d1d << 4) | q1d;

   if (dim == 2)
   {
      switch (id)
      {
         case 0x22: return DGMassCGIteration<2,2,2>(NE, B, Bt, pa_data, diag_inv,
                                                       rel_tol, abs_tol, max_iter, Mu, r, d, z, u, d1d, q1d);
         case 0x33: return DGMassCGIteration<2,3,3>(NE, B, Bt, pa_data, diag_inv,
                                                       rel_tol, abs_tol, max_iter, Mu, r, d, z, u, d1d, q1d);
         case 0x44: return DGMassCGIteration<2,4,4>(NE, B, Bt, pa_data, diag_inv,
                                                       rel_tol, abs_tol, max_iter, Mu, r, d, z, u, d1d, q1d);
         default:
            printf("id = %x\n", id);
            MFEM_ABORT("Fallback");
            return DGMassCGIteration<2>(NE, B, Bt, pa_data, diag_inv, rel_tol, abs_tol,
                                        max_iter, Mu, r, d, z, u, d1d, q1d);
      }
   }
   else if (dim == 3)
   {
      switch (id)
      {
         case 0x22: return DGMassCGIteration<3,2,2>(NE, B, Bt, pa_data, diag_inv,
                                                       rel_tol, abs_tol, max_iter, Mu, r, d, z, u, d1d, q1d);
         case 0x23: return DGMassCGIteration<3,2,3>(NE, B, Bt, pa_data, diag_inv,
                                                       rel_tol, abs_tol, max_iter, Mu, r, d, z, u, d1d, q1d);
         case 0x33: return DGMassCGIteration<3,3,3>(NE, B, Bt, pa_data, diag_inv,
                                                       rel_tol, abs_tol, max_iter, Mu, r, d, z, u, d1d, q1d);
         case 0x34: return DGMassCGIteration<3,3,4>(NE, B, Bt, pa_data, diag_inv,
                                                       rel_tol, abs_tol, max_iter, Mu, r, d, z, u, d1d, q1d);
         case 0x44: return DGMassCGIteration<3,4,4>(NE, B, Bt, pa_data, diag_inv,
                                                       rel_tol, abs_tol, max_iter, Mu, r, d, z, u, d1d, q1d);
         case 0x45: return DGMassCGIteration<3,4,5>(NE, B, Bt, pa_data, diag_inv,
                                                       rel_tol, abs_tol, max_iter, Mu, r, d, z, u, d1d, q1d);
         case 0x55: return DGMassCGIteration<3,5,5>(NE, B, Bt, pa_data, diag_inv,
                                                       rel_tol, abs_tol, max_iter, Mu, r, d, z, u, d1d, q1d);
         case 0x56: return DGMassCGIteration<3,5,6>(NE, B, Bt, pa_data, diag_inv,
                                                       rel_tol, abs_tol, max_iter, Mu, r, d, z, u, d1d, q1d);
         case 0x66: return DGMassCGIteration<3,6,6>(NE, B, Bt, pa_data, diag_inv,
                                                       rel_tol, abs_tol, max_iter, Mu, r, d, z, u, d1d, q1d);
         case 0x67: return DGMassCGIteration<3,6,7>(NE, B, Bt, pa_data, diag_inv,
                                                       rel_tol, abs_tol, max_iter, Mu, r, d, z, u, d1d, q1d);
         default:
            printf("id = 0x%x\n", id);
            MFEM_ABORT("Fallback");
            return DGMassCGIteration<3>(NE, B, Bt, pa_data, diag_inv, rel_tol, abs_tol,
                                        max_iter, Mu, r, d, z, u, d1d, q1d);
      }
   }
}

void DGMassInverse::SetOperator(const Operator &op)
{
   MFEM_ABORT("SetOperator not supported with DGMassInverse.")
}

void DGMassInverse::SetRelTol(const double rel_tol_) { rel_tol = rel_tol_; }

void DGMassInverse::SetAbsTol(const double abs_tol_) { abs_tol = abs_tol_; }

void DGMassInverse::SetMaxIter(const double max_iter_) { max_iter = max_iter_; }

DGMassInverse::~DGMassInverse()
{
   delete m;
}

} // namespace mfem
