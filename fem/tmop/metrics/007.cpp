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

struct TMOP_PA_Metric_007 : TMOP_PA_Metric_2D
{
   MFEM_HOST_DEVICE
   void EvalP(const real_t (&Jpt)[4], const real_t *w, real_t (&P)[4]) override
   {
      MFEM_CONTRACT_VAR(w);
      real_t dI1[4], dI2[4], dI2b[4];
      kernels::InvariantsEvaluator2D ie(
         Args().J(Jpt).dI1(dI1).dI2(dI2).dI2b(dI2b));
      const real_t I2 = ie.Get_I2();
      kernels::Add(2, 2, 1.0 + 1.0 / I2, ie.Get_dI1(), -ie.Get_I1() / (I2 * I2),
                   ie.Get_dI2(), P);
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
      real_t ddI1[4], ddI2[4], dI1[4], dI2[4], dI2b[4];
      kernels::InvariantsEvaluator2D ie(
         Args().J(Jpt).ddI1(ddI1).ddI2(ddI2).dI1(dI1).dI2(dI2).dI2b(dI2b));
      const real_t c1 = 1. / ie.Get_I2();
      const real_t c2 = weight * c1 * c1;
      const real_t c3 = ie.Get_I1() * c2;
      ConstDeviceMatrix di1(ie.Get_dI1(), DIM, DIM);
      ConstDeviceMatrix di2(ie.Get_dI2(), DIM, DIM);

      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi1(ie.Get_ddI1(i, j), DIM, DIM);
            ConstDeviceMatrix ddi2(ie.Get_ddI2(i, j), DIM, DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  H(r, c, i, j, qx, qy, e) =
                     weight * (1.0 + c1) * ddi1(r, c) - c3 * ddi2(r, c) -
                     c2 * (di1(i, j) * di2(r, c) + di2(i, j) * di1(r, c)) +
                     2.0 * c1 * c3 * di2(r, c) * di2(i, j);
               }
            }
         }
      }
   }
};

using kernel_t = void (*)(TMOPPASetupGrad2D &);
using metric_t = TMOP_PA_Metric_007;

MFEM_REGISTER_KERNELS(Setup007, kernel_t, (int, int));

template <int D, int Q>
kernel_t Setup007::Kernel()
{
   return TMOPPASetupGrad2D::Mult<metric_t, D, Q>;
}

kernel_t Setup007::Fallback(int, int)
{
   return TMOPPASetupGrad2D::Mult<metric_t>;
}

template <>
void TMOPAssembleGradPA<7>(TMOPPASetupGrad2D &ker)
{
   const static auto specialized_kernels = [] { return TMOPAdd<Setup007>(); }();
   Setup007::Run(ker.Ndof(), ker.Nqpt(), ker);
}

void TMOPAddMultPA_007(TMOPAddMultPA2D &ker)
{
   TMOPKernelLaunch<metric_t>(ker);
}

} // namespace mfem
