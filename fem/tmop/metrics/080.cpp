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
#include "../mult/2d/mult.hpp"
#include "../assemble/2d/grad.hpp"

namespace mfem
{

struct TMOP_PA_Metric_080 : TMOP_PA_Metric_2D
{
   MFEM_HOST_DEVICE
   void EvalP(const real_t (&Jpt)[4], const real_t *w, real_t (&P)[4]) override
   {
      MFEM_CONTRACT_VAR(w);
      // w0 P_2 + w1 P_77
      real_t dI1b[4], dI2[4], dI2b[4];
      kernels::InvariantsEvaluator2D ie(
         Args().J(Jpt).dI1b(dI1b).dI2(dI2).dI2b(dI2b));
      kernels::Set(2, 2, w[0] * 0.5, ie.Get_dI1b(), P);
      const real_t I2 = ie.Get_I2();
      kernels::Add(2, 2, w[1] * 0.5 * (1.0 - 1.0 / (I2 * I2)), ie.Get_dI2(), P);
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
      // w0 H_2 + w1 H_77
      real_t ddI1[4], ddI1b[4], dI2[4], dI2b[4], ddI2[4];
      kernels::InvariantsEvaluator2D ie(
         Args().J(Jpt).dI2(dI2).ddI1(ddI1).ddI1b(ddI1b).dI2b(dI2b).ddI2(ddI2));

      const real_t I2 = ie.Get_I2(), I2inv_sq = 1.0 / (I2 * I2);
      ConstDeviceMatrix di2(ie.Get_dI2(), DIM, DIM);
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi1b(ie.Get_ddI1b(i, j), DIM, DIM);
            ConstDeviceMatrix ddi2(ie.Get_ddI2(i, j), DIM, DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  H(r, c, i, j, qx, qy, e) =
                     w[0] * 0.5 * weight * ddi1b(r, c) +
                     w[1] * (weight * 0.5 * (1.0 - I2inv_sq) * ddi2(r, c) +
                             weight * (I2inv_sq / I2) * di2(r, c) * di2(i, j));
               }
            }
         }
      }
   }
};

using metric_t = TMOP_PA_Metric_080;
using mult_t = TMOPAddMultPA2D;
using setup_t = TMOPSetup2D;

using setup = func_t<setup_t>;
using mult = func_t<mult_t>;

// TMOP PA Setup, metric: 080
MFEM_REGISTER_KERNELS(S080, setup, (int, int));

template <int D, int Q>
setup S080::Kernel()
{
   return setup_t::Mult<metric_t, D, Q>;
}

setup S080::Fallback(int, int) { return setup_t::Mult<metric_t>; }

template <>
void TMOPKernel<80>(setup_t &ker)
{
   const static auto setup_kernels = []
   { return KernelSpecializations<S080>(); }();
   S080::Run(ker.Ndof(), ker.Nqpt(), ker);
}

// TMOP PA Mult, metric: 080

MFEM_REGISTER_KERNELS(K080, mult, (int, int));

template <int D, int Q>
mult K080::Kernel()
{
   return mult_t::Mult<metric_t, D, Q>;
}

mult K080::Fallback(int, int) { return mult_t::Mult<metric_t>; }

template <>
void TMOPKernel<80>(mult_t &ker)
{
   const static auto mult_kernels = []
   { return KernelSpecializations<K080>(); }();
   K080::Run(ker.Ndof(), ker.Nqpt(), ker);
}

} // namespace mfem
