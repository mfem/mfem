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

struct TMOP_PA_Metric_056 : TMOP_PA_Metric_2D
{
   MFEM_HOST_DEVICE
   void EvalP(const real_t (&Jpt)[4], const real_t *w, real_t (&P)[4]) override
   {
      MFEM_CONTRACT_VAR(w);
      // 0.5*(1 - 1/I2b^2)*dI2b
      real_t dI2b[4];
      kernels::InvariantsEvaluator2D ie(Args().J(Jpt).dI2b(dI2b));
      const real_t I2b = ie.Get_I2b();
      kernels::Set(2, 2, 0.5 * (1.0 - 1.0 / (I2b * I2b)), ie.Get_dI2b(), P);
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
      // (0.5 - 0.5/I2b^2)*ddI2b + (1/I2b^3)*(dI2b x dI2b)
      real_t dI2b[4], ddI2b[4];
      kernels::InvariantsEvaluator2D ie(Args().J(Jpt).dI2b(dI2b).ddI2b(ddI2b));
      const real_t I2b = ie.Get_I2b();
      ConstDeviceMatrix di2b(ie.Get_dI2b(), DIM, DIM);
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi2b(ie.Get_ddI2b(i, j), DIM, DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  H(r, c, i, j, qx, qy, e) =
                     weight * (0.5 - 0.5 / (I2b * I2b)) * ddi2b(r, c) +
                     weight / (I2b * I2b * I2b) * di2b(r, c) * di2b(i, j);
               }
            }
         }
      }
   }
};

using kernel_t = void (*)(TMOPPASetupGrad2D &);
using metric_t = TMOP_PA_Metric_056;

MFEM_REGISTER_KERNELS(Setup056, kernel_t, (int, int));

template <int D, int Q>
kernel_t Setup056::Kernel()
{
   return TMOPPASetupGrad2D::Mult<metric_t, D, Q>;
}

kernel_t Setup056::Fallback(int, int)
{
   return TMOPPASetupGrad2D::Mult<metric_t>;
}

template <>
void TMOPAssembleGradPA<56>(TMOPPASetupGrad2D &ker)
{
   const static auto specialized_kernels = [] { return TMOPAdd<Setup056>(); }();
   Setup056::Run(ker.Ndof(), ker.Nqpt(), ker);
}

void TMOPAddMultPA_056(TMOPAddMultPA2D &ker)
{
   TMOPKernelLaunch<metric_t>(ker);
}

} // namespace mfem
