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

MFEM_REGISTER_TMOP_KERNELS(real_t, EnergyPA_Fit_2D,
                           const real_t lim_normal,
                           const Vector &lim_dist,
                           const Vector &c0_,
                           const int NE,
                           const DenseTensor &j_,
                           const Array<real_t> &w_,
                           const Array<real_t> &b_,
                           const Array<real_t> &bld_,
                           const Vector &x0_,
                           const Vector &x1_,
                           const Vector &ones,
                           Vector &energy,
                           const int d1d,
                           const int q1d)
{
   return 0;
}

real_t TMOP_Integrator::GetLocalStateEnergyPA_Fit_2D(const Vector &X) const
{
   const int N = PA.ne;
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const real_t ln = lim_normal;
   const Vector &LD = PA.LD;
   const DenseTensor &J = PA.Jtr;
   const Array<real_t> &W   = PA.ir->GetWeights();
   const Array<real_t> &B   = PA.maps->B;
   const Array<real_t> &BLD = PA.maps_lim->B;
   MFEM_VERIFY(PA.maps_lim->ndof == D1D, "");
   MFEM_VERIFY(PA.maps_lim->nqpt == Q1D, "");
   const Vector &X0 = PA.X0;
   const Vector &C0 = PA.C0;
   const Vector &O = PA.O;
   Vector &E = PA.E;

   MFEM_LAUNCH_TMOP_KERNEL(EnergyPA_Fit_2D,id,ln,LD,C0,N,J,W,B,BLD,X0,X,O,E);
}

} // namespace mfem