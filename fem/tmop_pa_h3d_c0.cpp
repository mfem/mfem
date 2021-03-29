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

#include "tmop.hpp"
#include "tmop_pa.hpp"
#include "../general/forall.hpp"
#include "../linalg/kernels.hpp"

namespace mfem
{

MFEM_REGISTER_TMOP_KERNELS(void, AssembleDiagonalPA_Kernel_C0_3D,
                           const int NE,
                           const Array<double> &b,
                           const Vector &h0,
                           Vector &diagonal,
                           const int d1d,
                           const int q1d)
{
   constexpr int DIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto H0 = Reshape(h0.Read(), DIM, DIM, Q1D, Q1D, Q1D, NE);

   auto D = Reshape(diagonal.ReadWrite(), D1D, D1D, D1D, DIM, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      constexpr int DIM = 3;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;

      MFEM_SHARED double qqd[MQ1*MQ1*MD1];
      MFEM_SHARED double qdd[MQ1*MD1*MD1];
      DeviceTensor<3,double> QQD(qqd, MQ1, MQ1, MD1);
      DeviceTensor<3,double> QDD(qdd, MQ1, MD1, MD1);

      for (int v = 0; v < DIM; ++v)
      {
         // first tensor contraction, along z direction
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(dz,z,D1D)
               {
                  QQD(qx,qy,dz) = 0.0;
                  MFEM_UNROLL(MQ1);
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     const double Bz = B(qz,dz);
                     QQD(qx,qy,dz) += Bz * H0(v,v,qx,qy,qz,e) * Bz;
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
                  MFEM_UNROLL(MQ1);
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const double By = B(qy,dy);
                     QDD(qx,dy,dz) += By * QQD(qx,qy,dz) * By;
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
                  double d = 0.0;
                  MFEM_UNROLL(MQ1);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const double Bx = B(qx,dx);
                     d += Bx * QDD(qx,dy,dz) * Bx;
                  }
                  D(dx,dy,dz, v, e) += d;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

void TMOP_Integrator::AssembleDiagonalPA_C0_3D(Vector &D) const
{
   const int N = PA.ne;
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const Array<double> &B = PA.maps->B;
   const Vector &H0 = PA.H0;

   MFEM_LAUNCH_TMOP_KERNEL(AssembleDiagonalPA_Kernel_C0_3D,id,N,B,H0,D);
}

} // namespace mfem
