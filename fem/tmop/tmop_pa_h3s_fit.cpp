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
#include "../../linalg/dinvariants.hpp"

namespace mfem
{

MFEM_REGISTER_TMOP_KERNELS(void, SetupGradPA_Fit_3D,
                           const int NE,
                           const real_t &pw_,
                           const real_t &n0_,
                           const Vector &s0_,
                           const Vector &dc_,
                           const Vector &m0_,
                           const Vector &d1_,
                           const Vector &d2_,
                           const Array<int> &fe_,
                           Vector &h0_,
                           const int d1d,
                           const int q1d)
{
   constexpr int DIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;

   const auto PW = pw_;
   const auto N0 = n0_;
   const auto S0 = Reshape(s0_.Read(), D1D, D1D, D1D, NE);
   const auto DC = Reshape(dc_.Read(), D1D, D1D, D1D, NE);
   const auto M0 = Reshape(m0_.Read(), D1D, D1D, D1D, NE);
   const auto D1 = Reshape(d1_.Read(), D1D, D1D, D1D, DIM, NE);
   const auto D2 = Reshape(d2_.Read(), D1D, D1D, D1D, DIM, DIM, NE);
   const Array<int> FE = fe_;

   auto H0 = Reshape(h0_.Write(), DIM, DIM, D1D, D1D, D1D, NE);

   mfem::forall_3D(FE.Size(), D1D, D1D, D1D, [=] MFEM_HOST_DEVICE (int i)
   {
      const int e = FE[i];
      const int D1D = T_D1D ? T_D1D : d1d;

      MFEM_FOREACH_THREAD(qz,z,D1D)
      {
        MFEM_FOREACH_THREAD(qy,y,D1D)
        {
            MFEM_FOREACH_THREAD(qx,x,D1D)
            {
                const real_t sigma = S0(qx,qy,qz,e);
                const real_t dof_count = DC(qx,qy,qz,e);
                const real_t marker = M0(qx,qy,qz,e); 
                const real_t coeff = PW;
                const real_t normal = N0;

                
                double w = marker * normal * coeff * 1.0/dof_count;
                for (int i = 0; i < DIM; i++)
                {
                    for (int j = 0; j <= i; j++)
                    {
                        const real_t dxi = D1(qx,qy,qz,i,e);
                        const real_t dxj = D1(qx,qy,qz,j,e);
                        const real_t d2x = D2(qx,qy,qz,i,j,e);

                        const real_t entry = 2 * w * (dxi*dxj + sigma * d2x);
                        H0(i,j,qx,qy,qz,e) = entry;
                        if (i != j) { H0(j,i,qx,qy,qz,e) = entry;}   
                    }
                }
                
            }
        }
      }
      MFEM_SYNC_THREAD;
   });

}
void TMOP_Integrator::AssembleGradPA_Fit_3D(const Vector &X) const
{
   const int N = PA.ne;
   const int meshOrder = surf_fit_gf->FESpace()->GetMaxElementOrder();
   const int D1D = meshOrder + 1;
   const int Q1D = D1D;
   const int id = (D1D << 4 ) | Q1D;

   const real_t &PW = PA.PW;
   const real_t &N0 = PA.N0;
   const Vector &S0 = PA.S0;
   const Vector &DC = PA.DC;
   const Vector &M0 = PA.M0;
   const Vector &D1 = PA.D1;
   const Vector &D2 = PA.D2;

   Vector &H0 = PA.H0Fit;
   const Array<int> &FE = PA.FE;

   MFEM_LAUNCH_TMOP_KERNEL(SetupGradPA_Fit_3D,id,N,PW,N0,S0,DC,M0,D1,D2,FE,H0);
}

} // namespace mfem