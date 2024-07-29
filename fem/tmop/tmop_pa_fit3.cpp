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

MFEM_REGISTER_TMOP_KERNELS(real_t, EnergyPA_Fit_3D,
                           const int NE,
                           const real_t &c1_,
                           const real_t &c2_,
                           const Vector &x1_,
                           const Vector &x2_,
                           const Vector &x3_,
                           const Vector &ones,
                           Vector &energy,
                           const int d1d,
                           const int q1d)
{
   constexpr int DIM = 3;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto C1 = c1_;
   const auto C2 = c2_;
   const auto X1 = Reshape(x1_.Read(), D1D, D1D, D1D, NE);
   const auto X2 = Reshape(x2_.Read(), D1D, D1D, D1D, NE);
   const auto X3 = Reshape(x3_.Read(), D1D, D1D, D1D, NE);

   auto E = Reshape(energy.Write(), D1D, D1D, D1D, NE);

   mfem::forall_3D(NE, D1D, D1D, D1D, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = 1;


      MFEM_FOREACH_THREAD(qz,z,D1D)
      {
        MFEM_FOREACH_THREAD(qy,y,D1D)
        {
            MFEM_FOREACH_THREAD(qx,x,D1D)
            {
            const real_t sigma = X1(qx,qy,qz,e);
            const real_t dof_count = X2(qx,qy,qz,e);
            const real_t marker = X3(qx,qy,qz,e); 
            const real_t coeff = C1;
            const real_t normal = C2;

            if (marker == 0) {continue;}
            double w = coeff * normal * 1.0/dof_count;
            E(qx,qy,qz,e) = w * sigma * sigma;   
            }
        }
      }  
   });
   return energy * ones; 
}

real_t TMOP_Integrator::GetLocalStateEnergyPA_Fit_3D(const Vector &X) const
{
   const int N = PA.ne;
   const int meshOrder = surf_fit_gf->FESpace()->GetMaxElementOrder();
   const int D1D = meshOrder + 1;
   const int Q1D = D1D;
   const int id = (D1D << 4 ) | Q1D;
   const Vector &O = PA.O;
   Vector &E = PA.E;
   
   
   const real_t &C1 = PA.C1;
   const real_t &C2 = PA.C2;
   const Vector &X1 = PA.X1;
   const Vector &X2 = PA.X2;
   const Vector &X3 = PA.X3;

   MFEM_LAUNCH_TMOP_KERNEL(EnergyPA_Fit_3D,id,N,C1,C2,X1,X2,X3,O,E);
}

} // namespace mfem