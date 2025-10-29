// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../pa.hpp"
#include "../mult/mult2.hpp"
#include "../tools/energy2.hpp"
#include "../assemble/grad2.hpp"

namespace mfem
{

struct TMOP_PA_Metric_080 : TMOP_PA_Metric_2D
{
   MFEM_HOST_DEVICE real_t EvalW(const real_t (&Jpt)[4], const real_t *w) override
   {
      MFEM_CONTRACT_VAR(w);
      kernels::InvariantsEvaluator2D ie(Args().J(Jpt));
      const real_t eval_w_02 = 0.5 * ie.Get_I1b() - 1.0;
      const real_t I2b = ie.Get_I2b();
      const real_t eval_w_77 = 0.5 * (I2b * I2b + 1. / (I2b * I2b) - 2.);
      return w[0] * eval_w_02 + w[1] * eval_w_77;
   };

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

using metric = TMOP_PA_Metric_080;

using assemble = TMOPAssembleGradPA2D;
using energy = TMOPEnergyPA2D;
using mult = TMOPAddMultPA2D;

MFEM_TMOP_REGISTER_METRIC(metric, assemble, energy, mult, 80);

} // namespace mfem
