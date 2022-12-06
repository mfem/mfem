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

#include "../tmop.hpp"
#include "tmop_pa.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/kernels.hpp"

namespace mfem
{

MFEM_JIT
template<int T_D1D = 0, int T_Q1D = 0, int T_MAX = 4>
void TMOP_AssembleDiagonalPA_3D(const int NE,
                                const ConstDeviceMatrix &B,
                                const ConstDeviceMatrix &G,
                                const DeviceTensor<6, const double> &J,
                                const DeviceTensor<8, const double> &H,
                                DeviceTensor<5> &D,
                                const int d1d = 0,
                                const int q1d = 0,
                                const int max = 4)
{
   const int Q1D = T_Q1D ? T_Q1D : q1d;

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
                           const double *Jtr = &J(0,0,qx,qy,qz,e);
                           double jrt[9];
                           ConstDeviceMatrix Jrt(jrt,3,3);
                           kernels::CalcInverse<3>(Jtr, jrt);
                           const double Bz = B(qz,dz);
                           const double Gz = G(qz,dz);
                           const double L = i==2 ? Gz : Bz;
                           const double R = j==2 ? Gz : Bz;
                           const double Jij = Jrt(i,i) * Jrt(j,j);
                           const double h = H(v,i,v,j,qx,qy,qz,e);
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
                           const double By = B(qy,dy);
                           const double Gy = G(qy,dy);
                           const double L = i==1 ? Gy : By;
                           const double R = j==1 ? Gy : By;
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
                        double d = 0.0;
                        MFEM_UNROLL(MQ1)
                        for (int qx = 0; qx < Q1D; ++qx)
                        {
                           const double Bx = B(qx,dx);
                           const double Gx = G(qx,dx);
                           const double L = i==0 ? Gx : Bx;
                           const double R = j==0 ? Gx : Bx;
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

void TMOP_Integrator::AssembleDiagonalPA_3D(Vector &diagonal) const
{
   const int NE = PA.ne;
   constexpr int DIM = 3;
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;

   const auto B = Reshape(PA.maps->B.Read(), Q1D, D1D);
   const auto G = Reshape(PA.maps->G.Read(), Q1D, D1D);
   const auto J = Reshape(PA.Jtr.Read(), DIM, DIM, Q1D, Q1D, Q1D, NE);
   const auto H = Reshape(PA.H.Read(), DIM, DIM, DIM, DIM, Q1D, Q1D, Q1D, NE);
   auto D = Reshape(diagonal.ReadWrite(), D1D, D1D, D1D, DIM, NE);

   decltype(&TMOP_AssembleDiagonalPA_3D<>) ker = TMOP_AssembleDiagonalPA_3D;
#ifndef MFEM_USE_JIT
   const int d=D1D, q=Q1D;
   if (d==2 && q==2) { ker = TMOP_AssembleDiagonalPA_3D<2,2>; }
   if (d==2 && q==3) { ker = TMOP_AssembleDiagonalPA_3D<2,3>; }
   if (d==2 && q==4) { ker = TMOP_AssembleDiagonalPA_3D<2,4>; }
   if (d==2 && q==5) { ker = TMOP_AssembleDiagonalPA_3D<2,5>; }
   if (d==2 && q==6) { ker = TMOP_AssembleDiagonalPA_3D<2,6>; }

   if (d==3 && q==3) { ker = TMOP_AssembleDiagonalPA_3D<3,3>; }
   if (d==3 && q==4) { ker = TMOP_AssembleDiagonalPA_3D<3,4>; }
   if (d==3 && q==5) { ker = TMOP_AssembleDiagonalPA_3D<3,5>; }
   if (d==3 && q==6) { ker = TMOP_AssembleDiagonalPA_3D<3,6>; }

   if (d==4 && q==4) { ker = TMOP_AssembleDiagonalPA_3D<4,4>; }
   if (d==4 && q==5) { ker = TMOP_AssembleDiagonalPA_3D<4,5>; }
   if (d==4 && q==6) { ker = TMOP_AssembleDiagonalPA_3D<4,6>; }

   if (d==5 && q==5) { ker = TMOP_AssembleDiagonalPA_3D<5,5>; }
   if (d==5 && q==6) { ker = TMOP_AssembleDiagonalPA_3D<5,6>; }
#endif
   ker(NE,B,G,J,H,D,D1D,Q1D,4);
}

} // namespace mfem
