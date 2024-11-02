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
#include "../assemble/h2s.hpp"

namespace mfem
{

struct TMOP_PA_Metric_094 : TMOP_PA_Metric_2D
{
   MFEM_HOST_DEVICE
   void EvalP(const real_t (&Jpt)[4], const real_t *w, real_t (&P)[4]) override
   {
      // w0 P_2 + w1 P_56
      real_t dI1b[4], dI2b[4];
      kernels::InvariantsEvaluator2D ie(Args().J(Jpt).dI1b(dI1b).dI2b(dI2b));
      kernels::Set(2, 2, w[0] * 0.5, ie.Get_dI1b(), P);
      const real_t I2b = ie.Get_I2b();
      kernels::Add(2, 2, w[1] * 0.5 * (1.0 - 1.0 / (I2b * I2b)), ie.Get_dI2b(),
                   P);
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
      // w0 H_2 + w1 H_56
      real_t ddI1[4], ddI1b[4], dI2b[4], ddI2b[4];
      kernels::InvariantsEvaluator2D ie(
         Args().J(Jpt).ddI1(ddI1).ddI1b(ddI1b).dI2b(dI2b).ddI2b(ddI2b));
      const real_t I2b = ie.Get_I2b();
      ConstDeviceMatrix di2b(ie.Get_dI2b(), DIM, DIM);
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi1b(ie.Get_ddI1b(i, j), DIM, DIM);
            ConstDeviceMatrix ddi2b(ie.Get_ddI2b(i, j), DIM, DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  H(r, c, i, j, qx, qy, e) =
                     w[0] * 0.5 * weight * ddi1b(r, c) +
                     w[1] *
                        (weight * (0.5 - 0.5 / (I2b * I2b)) * ddi2b(r, c) +
                         weight / (I2b * I2b * I2b) * di2b(r, c) * di2b(i, j));
               }
            }
         }
      }
   }
};

using metric_t = TMOP_PA_Metric_094;
using mult_t = TMOPAddMultPA2D;
using setup_t = TMOPSetup2D;

using setup = func_t<setup_t>;
using mult = func_t<mult_t>;

// TMOP PA Setup, metric: 094
MFEM_REGISTER_KERNELS(S094, setup, (int, int));

template <int D, int Q>
setup S094::Kernel()
{
   return setup_t::Mult<metric_t, D, Q>;
}

setup S094::Fallback(int, int) { return setup_t::Mult<metric_t>; }

template <>
void TMOPKernel<94>(setup_t &ker)
{
   const static auto setup_kernels = []
   { return KernelSpecializations<S094>(); }();
   S094::Run(ker.Ndof(), ker.Nqpt(), ker);
}

// TMOP PA Mult, metric: 094

MFEM_REGISTER_KERNELS(K094, mult, (int, int));

template <int D, int Q>
mult K094::Kernel()
{
   return mult_t::Mult<metric_t, D, Q>;
}

mult K094::Fallback(int, int) { return mult_t::Mult<metric_t>; }

template <>
void TMOPKernel<94>(mult_t &ker)
{
   const static auto mult_kernels = []
   { return KernelSpecializations<K094>(); }();
   K094::Run(ker.Ndof(), ker.Nqpt(), ker);
}

} // namespace mfem
