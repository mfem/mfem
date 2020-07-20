// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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

MFEM_REGISTER_TMOP_KERNELS(void, AssembleDiagonalPA_Kernel_3D,
                           const int NE,
                           const Array<double> &b,
                           const Array<double> &g,
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

   MFEM_FORALL(e, NE,
   {
      constexpr int DIM = 3;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;

      double qqd[DIM*DIM * MQ1*MQ1*MD1];
      double qdd[DIM*DIM * MQ1*MD1*MD1];
      DeviceTensor<5,double> QQD(qqd, DIM, DIM, MQ1, MQ1, MD1);
      DeviceTensor<5,double> QDD(qdd, DIM, DIM, MQ1, MD1, MD1);

      for (int v = 0; v < DIM; ++v)
      {
         // first tensor contraction, along z direction
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int dz = 0; dz < D1D; ++dz)
               {
                  for (int i = 0; i < DIM; i++)
                  {
                     for (int j = 0; j < DIM; j++)
                     {
                        QQD(i,j,qx,qy,dz) = 0.0;
                     }
                  }
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     const double *Jtr = &J(0,0,qx,qy,qz,e);
                     double j[9];
                     ConstDeviceMatrix Jrt(j,3,3);
                     kernels::CalcInverse<3>(Jtr, j);

                     const double Bz = B(qz,dz);
                     const double Gz = G(qz,dz);
                     for (int i = 0; i < DIM; i++)
                     {
                        for (int j = 0; j < DIM; j++)
                        {
                           const double L = i==2 ? Gz : Bz;
                           const double R = j==2 ? Gz : Bz;
                           const double Jij = Jrt(i,i) * Jrt(j,j);
                           const double h = Jij * H(v,i,v,j,qx,qy,qz,e);
                           QQD(i,j,qx,qy,dz) += L * h * R;
                        }
                     }
                  }
               }
            }
         }
         // second tensor contraction, along y direction
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int dz = 0; dz < D1D; ++dz)
            {
               for (int dy = 0; dy < D1D; ++dy)
               {
                  for (int i = 0; i < DIM; i++)
                  {
                     for (int j = 0; j < DIM; j++)
                     {
                        QDD(i,j,qx,dy,dz) = 0.0;
                     }
                  }
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const double By = B(qy,dy);
                     const double Gy = G(qy,dy);
                     for (int i = 0; i < DIM; i++)
                     {
                        for (int j = 0; j < DIM; j++)
                        {
                           const double L = i==1 ? Gy : By;
                           const double R = j==1 ? Gy : By;
                           QDD(i,j,qx,dy,dz) += L * QQD(i,j,qx,qy,dz) * R;
                        }
                     }
                  }
               }
            }
         }
         // third tensor contraction, along x direction
         for (int dz = 0; dz < D1D; ++dz)
         {
            for (int dy = 0; dy < D1D; ++dy)
            {
               for (int dx = 0; dx < D1D; ++dx)
               {
                  double d = 0.0;
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const double Bx = B(qx,dx);
                     const double Gx = G(qx,dx);
                     for (int i = 0; i < DIM; i++)
                     {
                        for (int j = 0; j < DIM; j++)
                        {
                           const double L = i==0 ? Gx : Bx;
                           const double R = j==0 ? Gx : Bx;
                           d += L * QDD(i,j,qx,dy,dz) * R;
                        }
                     }
                  }
                  D(dx,dy,dz, v, e) += d;
               }
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
   const Array<double> &B = PA.maps->B;
   const Array<double> &G = PA.maps->G;
   const Vector &H = PA.H;

   MFEM_LAUNCH_TMOP_KERNEL(AssembleDiagonalPA_Kernel_3D,id,N,B,G,J,H,D);
}

} // namespace mfem
