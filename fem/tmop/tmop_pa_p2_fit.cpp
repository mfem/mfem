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

MFEM_REGISTER_TMOP_KERNELS(void, AddMultPA_Kernel_Fit_2D,
                           const int NE,
                           const real_t &pw_,
                           const real_t &n0_,
                           const Vector &s0_,
                           const Vector &dc_,
                           const Vector &m0_,
                           const Vector &d1_,
                           Vector &y_,
                           const int d1d,
                           const int q1d)
{
   constexpr int DIM = 2;
   constexpr int NBZ = 1;
   const int D1D = T_D1D ? T_D1D : d1d;

   const auto PW = pw_;
   const auto N0 = n0_;
   const auto S0 = Reshape(s0_.Read(), D1D, D1D, NE);
   const auto DC = Reshape(dc_.Read(), D1D, D1D, NE);
   const auto M0 = Reshape(m0_.Read(), D1D, D1D, NE);
   const auto D1 = Reshape(d1_.Read(), D1D, D1D, DIM, NE);

   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, DIM, NE);

   mfem::forall_2D_batch(NE, D1D, D1D, NBZ, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;

      MFEM_FOREACH_THREAD(qy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,D1D)
         {
            const real_t sigma = S0(qx,qy,e);
            const real_t dof_count = DC(qx,qy,e);
            const real_t marker = M0(qx,qy,e);
            const real_t coeff = PW;
            const real_t normal = N0;

            const real_t dx = D1(qx,qy,0,e);
            const real_t dy = D1(qx,qy,1,e);

            double w = marker * normal * coeff * 1.0/dof_count;
            Y(qx,qy,0,e) += 2 * w * sigma * dx;
            Y(qx,qy,1,e) += 2 * w * sigma * dy;
         }
      }
      MFEM_SYNC_THREAD;
   });

}
void TMOP_Integrator::AddMultPA_Fit_2D(const Vector &X, Vector &Y) const
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

   MFEM_LAUNCH_TMOP_KERNEL(AddMultPA_Kernel_Fit_2D,id,N,PW,N0,S0,DC,M0,D1,Y);
}

} // namespace mfem