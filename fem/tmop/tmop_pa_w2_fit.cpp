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
#include "../qinterp/dispatch.hpp"


namespace mfem
{

MFEM_REGISTER_TMOP_KERNELS(real_t, EnergyPA_Fit_2D,
                           const int NE,
                           const real_t coeff,
                           const real_t normal,
                           const Vector &s0_,
                           const Vector &dc_,
                           const Vector &m0_,
                           const Vector &ones,
                           const Array<int> &fe_,
                           Vector &energy,
                           const int d1d,
                           const int q1d)
{
   const int D1D = T_D1D ? T_D1D : d1d;



   const auto S0 = Reshape(s0_.Read(), D1D, D1D, NE);
   const auto DC = Reshape(dc_.Read(), D1D, D1D, NE);
   const auto M0 = Reshape(m0_.Read(), D1D, D1D, NE);
   const auto FE = fe_.Read();
   const auto nel_fit = fe_.Size();

   auto E = Reshape(energy.Write(), D1D, D1D, NE);


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

            double w = marker * coeff * normal * 1.0/dof_count;
            E(qx,qy,i) = w * sigma * sigma;
         }
      }
   });
   return energy * ones;
}

real_t TMOP_Integrator::GetLocalStateEnergyPA_Fit_2D(const Vector &X) const
{
   const int N = PA.ne;
   const int meshOrder = surf_fit_gf->FESpace()->GetMaxElementOrder();
   const int D1D = meshOrder + 1;
   const int Q1D = D1D;
   const int id = (D1D << 4 ) | Q1D;
   const Vector &O = PA.SFO;
   Vector &E = PA.SFE;


   const real_t &PW = PA.SFC;
   const real_t &N0 = surf_fit_normal;
   const Vector &S0 = PA.SFV;
   const Vector &DC = PA.SFDC;
   const Vector &M0 = PA.SFM;

   const Array<int> &FE = PA.SFList;

   MFEM_LAUNCH_TMOP_KERNEL(EnergyPA_Fit_2D,id,N,PW,N0,S0,DC,M0,O,FE,E);
}

} // namespace mfem
