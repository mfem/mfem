// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../tmop.hpp"
#include "tmop_pa.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/kernels.hpp"

namespace mfem
{

MFEM_REGISTER_TMOP_KERNELS(void, AssembleDiagonalPA_Kernel_3D,
                           const int NE,
                           const Array<fptype> &b,
                           const Array<fptype> &g,
                           const DenseTensor &j,
                           const Vector &h,
                           Vector &diagonal,
                           const int d1d,
                           const int q1d)
{
   constexpr int DIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto G = Reshape(g.Read(), Q1D, D1D);
   const auto J = Reshape(j.Read(), DIM, DIM, Q1D, Q1D, Q1D, NE);
   const auto H = Reshape(h.Read(), DIM, DIM, DIM, DIM, Q1D, Q1D, Q1D, NE);

   auto D = Reshape(diagonal.ReadWrite(), D1D, D1D, D1D, DIM, NE);

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int DIM = 3;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      MFEM_SHARED fptype qqd[MQ1*MQ1*MD1];
      MFEM_SHARED fptype qdd[MQ1*MD1*MD1];
      DeviceTensor<3,fptype> QQD(qqd, MQ1, MQ1, MD1);
      DeviceTensor<3,fptype> QDD(qdd, MQ1, MD1, MD1);

      for (int v = 0; v < DIM; ++v)
      {
         for (int i = 0; i < DIM; i++)
         {
            for (int j = 0; j < DIM; j++)
            {
               // first tensor contraction, along z direction
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  MFEM_FOREACH_THREAD(qy,y,Q1D)
                  {
                     MFEM_FOREACH_THREAD(dz,z,D1D)
                     {
                        QQD(qx,qy,dz) = 0.0;
                        MFEM_UNROLL(MQ1)
                        for (int qz = 0; qz < Q1D; ++qz)
                        {
                           const fptype *Jtr = &J(0,0,qx,qy,qz,e);
                           fptype jrt[9];
                           ConstDeviceMatrix Jrt(jrt,3,3);
                           kernels::CalcInverse<3>(Jtr, jrt);
                           const fptype Bz = B(qz,dz);
                           const fptype Gz = G(qz,dz);
                           const fptype L = i==2 ? Gz : Bz;
                           const fptype R = j==2 ? Gz : Bz;
                           const fptype Jij = Jrt(i,i) * Jrt(j,j);
                           const fptype h = H(v,i,v,j,qx,qy,qz,e);
                           QQD(qx,qy,dz) += L * Jij * h * R;
                        }
                     }
                  }
               }
               MFEM_SYNC_THREAD;
               // second tensor contraction, along y direction
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  MFEM_FOREACH_THREAD(dz,z,D1D)
                  {
                     MFEM_FOREACH_THREAD(dy,y,D1D)
                     {
                        QDD(qx,dy,dz) = 0.0;
                        MFEM_UNROLL(MQ1)
                        for (int qy = 0; qy < Q1D; ++qy)
                        {
                           const fptype By = B(qy,dy);
                           const fptype Gy = G(qy,dy);
                           const fptype L = i==1 ? Gy : By;
                           const fptype R = j==1 ? Gy : By;
                           QDD(qx,dy,dz) += L * QQD(qx,qy,dz) * R;
                        }
                     }
                  }
               }
               MFEM_SYNC_THREAD;
               // third tensor contraction, along x direction
               MFEM_FOREACH_THREAD(dz,z,D1D)
               {
                  MFEM_FOREACH_THREAD(dy,y,D1D)
                  {
                     MFEM_FOREACH_THREAD(dx,x,D1D)
                     {
                        fptype d = 0.0;
                        MFEM_UNROLL(MQ1)
                        for (int qx = 0; qx < Q1D; ++qx)
                        {
                           const fptype Bx = B(qx,dx);
                           const fptype Gx = G(qx,dx);
                           const fptype L = i==0 ? Gx : Bx;
                           const fptype R = j==0 ? Gx : Bx;
                           d += L * QDD(qx,dy,dz) * R;
                        }
                        D(dx,dy,dz,v,e) += d;
                     }
                  }
               }
               MFEM_SYNC_THREAD;
            }
         }
      }
   });
}

void TMOP_Integrator::AssembleDiagonalPA_3D(Vector &D) const
{
   const int N = PA.ne;
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const DenseTensor &J = PA.Jtr;
   const Array<fptype> &B = PA.maps->B;
   const Array<fptype> &G = PA.maps->G;
   const Vector &H = PA.H;

   MFEM_LAUNCH_TMOP_KERNEL(AssembleDiagonalPA_Kernel_3D,id,N,B,G,J,H,D);
}

} // namespace mfem
