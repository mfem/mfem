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

namespace mfem
{

MFEM_REGISTER_TMOP_KERNELS(void, AddMultPA_Kernel_Fit_2D,
                           const int NE,
                           const real_t coeff,
                           const real_t normal,
                           const Vector &s0_,
                           const Vector &dc_,
                           const Vector &m0_,
                           const Vector &d1_,
                           const Array<int> &fe_,
                           Vector &y_,
                           const int d1d)
{
   constexpr int DIM = 2;
   const int D1D = T_D1D ? T_D1D : d1d;

   const auto S0 = Reshape(s0_.Read(), D1D, D1D, NE);
   const auto DC = Reshape(dc_.Read(), D1D, D1D, NE);
   const auto M0 = Reshape(m0_.Read(), D1D, D1D, NE);
   const auto D1 = Reshape(d1_.Read(), D1D, D1D, DIM, NE);
   const auto FE = fe_.Read();
   const auto nel_fit = fe_.Size();


   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, DIM, NE);

   mfem::forall_2D(nel_fit, D1D, D1D, [=] MFEM_HOST_DEVICE (int i)
   {
      const int e = FE[i];
      const int D1D = T_D1D ? T_D1D : d1d;

      MFEM_FOREACH_THREAD(qy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,D1D)
         {
            const real_t sigma = S0(qx,qy,e);
            const real_t dof_count = DC(qx,qy,e);
            const real_t marker = M0(qx,qy,e);

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

   const real_t &PW = PA.SFC;
   const real_t &N0 = surf_fit_normal;
   const Vector &S0 = PA.SFV;
   const Vector &DC = PA.SFDC;
   const Vector &M0 = PA.SFM;
   const Vector &D1 = PA.SFG;

   const Array<int> &FE = PA.SFEList;

   MFEM_LAUNCH_TMOP_NODAL_KERNEL(AddMultPA_Kernel_Fit_2D,D1D,N,PW,N0,S0,DC,M0,
                                 D1,FE,Y);
}

} // namespace mfem
