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
#include "../mult/mult3.hpp"
#include "../tools/energy3.hpp"
#include "../assemble/grad3.hpp"

namespace mfem
{

struct TMOP_PA_Metric_332 : TMOP_PA_Metric_3D
{
   MFEM_HOST_DEVICE real_t EvalW(const real_t (&Jpt)[DIM * DIM],
                                 const real_t *w) override
   {
      real_t B[9];
      MFEM_CONTRACT_VAR(w);
      kernels::InvariantsEvaluator3D ie(Args().J(Jpt).B(B));
      const real_t eval_w_302 = ie.Get_I1b() * ie.Get_I2b() / 9. - 1.;
      const real_t a = ie.Get_I3b() - 1.0;
      const real_t eval_w_315 = a * a;
      return w[0] * eval_w_302 + w[1] * eval_w_315;
   }

   MFEM_HOST_DEVICE
   void EvalP(const real_t (&Jpt)[9], const real_t *w, real_t (&P)[9]) override
   {
      // w0 P_302 + w1 P_315
      real_t B[9];
      real_t dI1b[9], dI2[9], dI2b[9], dI3b[9];
      kernels::InvariantsEvaluator3D ie(
         Args().J(Jpt).B(B).dI1b(dI1b).dI2(dI2).dI2b(dI2b).dI3b(dI3b));
      const real_t alpha = w[0] * ie.Get_I1b() / 9.;
      const real_t beta = w[0] * ie.Get_I2b() / 9.;
      kernels::Add(3, 3, alpha, ie.Get_dI2b(), beta, ie.Get_dI1b(), P);
      real_t sign_detJ;
      const real_t I3b = ie.Get_I3b(sign_detJ);
      kernels::Add(3, 3, w[1] * 2.0 * (I3b - 1.0), ie.Get_dI3b(sign_detJ), P);
   }

   MFEM_HOST_DEVICE
   void AssembleH(const int qx,
                  const int qy,
                  const int qz,
                  const int e,
                  const real_t weight,
                  real_t *Jrt,
                  real_t *Jpr,
                  const real_t (&Jpt)[9],
                  const real_t *w,
                  const DeviceTensor<8> &H) const override
   {
      real_t B[9];
      real_t dI1b[9], /*ddI1[9],*/ ddI1b[9];
      real_t dI2[9], dI2b[9], ddI2[9], ddI2b[9];
      real_t *dI3b = Jrt, *ddI3b = Jpr;
      // w0 H_302 + w1 H_315
      kernels::InvariantsEvaluator3D ie(Args()
                                        .J(Jpt)
                                        .B(B)
                                        .dI1b(dI1b)
                                        .ddI1b(ddI1b)
                                        .dI2(dI2)
                                        .dI2b(dI2b)
                                        .ddI2(ddI2)
                                        .ddI2b(ddI2b)
                                        .dI3b(dI3b)
                                        .ddI3b(ddI3b));
      real_t sign_detJ;
      const real_t c1 = weight / 9.0;
      const real_t I1b = ie.Get_I1b();
      const real_t I2b = ie.Get_I2b();
      const real_t I3b = ie.Get_I3b(sign_detJ);
      ConstDeviceMatrix di1b(ie.Get_dI1b(), DIM, DIM);
      ConstDeviceMatrix di2b(ie.Get_dI2b(), DIM, DIM);
      ConstDeviceMatrix di3b(ie.Get_dI3b(sign_detJ), DIM, DIM);
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi1b(ie.Get_ddI1b(i, j), DIM, DIM);
            ConstDeviceMatrix ddi2b(ie.Get_ddI2b(i, j), DIM, DIM);
            ConstDeviceMatrix ddi3b(ie.Get_ddI3b(i, j), DIM, DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  const real_t dp_302 =
                     (di2b(r, c) * di1b(i, j) + di1b(r, c) * di2b(i, j)) +
                     ddi2b(r, c) * I1b + ddi1b(r, c) * I2b;
                  const real_t dp_315 =
                     2.0 * weight * (I3b - 1.0) * ddi3b(r, c) +
                     2.0 * weight * di3b(r, c) * di3b(i, j);
                  H(r, c, i, j, qx, qy, qz, e) =
                     w[0] * c1 * dp_302 + w[1] * dp_315;
               }
            }
         }
      }
   }
};

using metric = TMOP_PA_Metric_332;

using assemble = TMOPAssembleGradPA3D;
using energy = TMOPEnergyPA3D;
using mult = TMOPAddMultPA3D;

MFEM_TMOP_REGISTER_METRIC(metric, assemble, energy, mult, 332);

} // namespace mfem
