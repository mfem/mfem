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

#include "../../kernel_dispatch.hpp"

#include "../pa.hpp"
#include "../mult/p2.hpp"
#include "../setup/h2s.hpp"

namespace mfem
{

struct TMOP_PA_Metric_001 : TMOP_PA_Metric_2D
{
   MFEM_HOST_DEVICE
   void EvalP(const real_t (&Jpt)[4], const real_t *w, real_t (&P)[4]) override
   {
      MFEM_CONTRACT_VAR(w);
      real_t dI1[4];
      kernels::InvariantsEvaluator2D ie(Args().J(Jpt).dI1(dI1));
      kernels::Set(2, 2, 1.0, ie.Get_dI1(), P);
   }

   MFEM_HOST_DEVICE
   void AssembleH(const int qx,
                  const int qy,
                  const int e,
                  const real_t weight,
                  const real_t (&Jpt)[4],
                  const real_t *w,
                  const DeviceTensor<7> &H) override
   {
      MFEM_CONTRACT_VAR(w);
      // weight * ddI1
      real_t ddI1[4];
      kernels::InvariantsEvaluator2D ie(Args().J(Jpt).ddI1(ddI1));
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi1(ie.Get_ddI1(i, j), DIM, DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  const real_t h = ddi1(r, c);
                  H(r, c, i, j, qx, qy, e) = weight * h;
               }
            }
         }
      }
   }
};

using kernel_t = void (*)(TMOPPASetupGrad2D &);
using metric_t = TMOP_PA_Metric_001;

MFEM_REGISTER_KERNELS(Setup001, kernel_t, (int, int));

template <int D, int Q>
kernel_t Setup001::Kernel()
{
   return TMOPPASetupGrad2D::Mult<metric_t, D, Q>;
}

kernel_t Setup001::Fallback(int, int)
{
   return TMOPPASetupGrad2D::Mult<metric_t>;
}

template <>
void TMOPAssembleGradPA<1>(TMOPPASetupGrad2D &ker)
{
   const static auto specialized_kernels = [] { return TMOPAdd<Setup001>(); }();
   Setup001::Run(ker.Ndof(), ker.Nqpt(), ker);
}

void TMOPAddMultPA_001(TMOPAddMultPA2D &ker)
{
   TMOPKernelLaunch<metric_t>(ker);
}

} // namespace mfem
