// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
#include "../linearform.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/kernels.hpp"

namespace mfem
{

MFEM_REGISTER_TMOP_KERNELS(void, AddMultGradPA_Kernel_Fit_3D,
                           const int NE,
                           const Array<int> &fe_,
                           const Vector &h0_,
                           const Vector &r_,
                           Vector &c_,
                           const int d1d,
                           const int q1d)
{
   constexpr int DIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const Array<int> FE = fe_;

   const auto H0 = Reshape(h0_.Read(), DIM, DIM, D1D, D1D, D1D, NE);
   const auto R = Reshape(r_.Read(), D1D, D1D, D1D, DIM, NE);

   auto Y = Reshape(c_.ReadWrite(), D1D, D1D, D1D, DIM, NE);

   mfem::forall_3D(NE, D1D, D1D, D1D, [=] MFEM_HOST_DEVICE (int e)
   {
      if (FE.Find(e) != -1)
      {
      const int D1D = T_D1D ? T_D1D : d1d;

      MFEM_FOREACH_THREAD(qz,z,D1D)
      {
         MFEM_FOREACH_THREAD(qy,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,D1D)
            {
               real_t Xh[3];
               real_t H_data[9];
               DeviceMatrix H(H_data,3,3);
               for (int i = 0; i < DIM; i++)
               {
                  Xh[i] = R(qx,qy,qz,i,e);
                  for (int j = 0; j < DIM; j++)
                  {
                     H(i,j) = H0(i,j,qx,qy,qz,e);
                  }
               }
               
               real_t p2[3];
               kernels::Mult(3,3,H_data,Xh,p2);

               for (int i = 0; i < DIM; i++)
               {
                  Y(qx,qy,qz,i,e) += p2[i];
               }
            }
         }
      }
      }
   });
}

void TMOP_Integrator::AddMultGradPA_Fit_3D(const Vector &R, Vector &C) const
{
   const int N = PA.ne;
   const int meshOrder = surf_fit_gf->FESpace()->GetMaxElementOrder();
   const int D1D = meshOrder + 1;
   const int Q1D = D1D;
   const int id = (D1D << 4 ) | Q1D;
   const Vector &H0 = PA.H0Fit;

   const Array<int> &FE = PA.FE;

   MFEM_LAUNCH_TMOP_KERNEL(AddMultGradPA_Kernel_Fit_3D,id,N,FE,H0,R,C);
}

} // namespace mfem
