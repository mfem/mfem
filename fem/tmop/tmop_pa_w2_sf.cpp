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

#include "../tmop.hpp"
#include "tmop_pa.hpp"
#include "../linearform.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/kernels.hpp"

namespace mfem
{

MFEM_REGISTER_TMOP_KERNELS(double, EnergyPA_SF_2D,
                           const double surf_fit_normal,
                           const Vector &surf_fit_gf,
                           const Vector &surf_fit_mask,
                           const Vector &c0sf_,
                           const int NE,
                           const Vector &ones,
                           Vector &energy,
                           const int d1d,
                           const int q1d)
{
   const bool const_c0 = c0sf_.Size() == 1;

   constexpr int NBZ = 1;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto C0SF = const_c0 ?
                     Reshape(c0sf_.Read(), 1, 1, 1) :
                     Reshape(c0sf_.Read(), Q1D, Q1D, NE);
   const auto SFG = Reshape(surf_fit_gf.Read(), D1D, D1D, NE);
   const auto SFM = Reshape(surf_fit_mask.Read(), D1D, D1D, NE);
   auto E = Reshape(energy.Write(), Q1D, Q1D, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double sf, sfm;
            const double surf_fit_coeff = const_c0 ? C0SF(0,0,0) : C0SF(qx,qy,e);
            E(qx, qy, e) = surf_fit_normal * surf_fit_coeff *
            SFM(qx, qy, e) * SFG(qx, qy, e) * SFG(qx, qy, e);
         }
      }
   });
   return energy * ones;
}

double TMOP_Integrator::GetLocalStateEnergyPA_SF_2D(const Vector &X) const
{
   const int N = PA.ne;
   const int D1D = PA.maps_surf->ndof;
   const int Q1D = PA.maps_surf->nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const double sn = surf_fit_normal;
   const Vector &SFG = PA.SFG;
   const Vector &SFM = PA.SFM;
   const Array<double> &W   = PA.irsf->GetWeights();
   const Array<double> &B   = PA.maps_surf->B;
   const Array<double> &BLD = PA.maps_surf->B;
   MFEM_VERIFY(PA.maps_surf->ndof == D1D, "");
   MFEM_VERIFY(PA.maps_surf->nqpt == Q1D, "");
   const Vector &C0SF = PA.C0sf;
   const Vector &O = PA.Osf;
   Vector &E = PA.Esf;

   MFEM_LAUNCH_TMOP_KERNEL(EnergyPA_SF_2D,id,sn,SFG,SFM,C0SF,N,O,E);
}

} // namespace mfem
