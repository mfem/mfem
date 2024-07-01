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
                           const int NE,
                           const DenseTensor &j_,
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
   return 0; //TODO
}

real_t TMOP_Integrator::GetLocalStateEnergyPA_Fit_2D(const Vector &X) const
{
   const int N = PA.ne;
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;
   const int id = (D1D<< 4) | Q1D;
   const DenseTensor &J = PA.Jtr;
   MFEM_VERIFY(PA.maps_lim->ndof == D1D, "");
   MFEM_VERIFY(PA.maps_lim->nqpt == Q1D, "");
   const Vector &O = PA.O;
   Vector &E = PA.E;
   
   
   const real_t &C1 = PA.C1;
   const real_t &C2 = PA.C2;
   const Vector &X1 = PA.X1;
   const Vector &X2 = PA.X2;
   const Vector &X3 = PA.X3;

   MFEM_LAUNCH_TMOP_KERNEL(EnergyPA_Fit_2D,id,N,J,C1,C2,X1,X2,X3,O,E);
}

} // namespace mfem