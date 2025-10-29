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

struct TMOP_PA_Metric_303 : TMOP_PA_Metric_3D
{
   MFEM_HOST_DEVICE real_t EvalW(const real_t (&Jpt)[DIM * DIM],
                                 const real_t *w) override
   {
      real_t B[9];
      MFEM_CONTRACT_VAR(w);
      kernels::InvariantsEvaluator3D ie(Args().J(Jpt).B(B));
      // mu_303 = I1b/3 - 1
      return ie.Get_I1b() / 3. - 1.;
   }

   MFEM_HOST_DEVICE
   void EvalP(const real_t (&Jpt)[9], const real_t *w, real_t (&P)[9]) override
   {
      MFEM_CONTRACT_VAR(w);
      // dI1b/3
      real_t B[9];
      real_t dI1b[9], dI3b[9];
      kernels::InvariantsEvaluator3D ie(
         Args().J(Jpt).B(B).dI1b(dI1b).dI3b(dI3b));
      kernels::Set(3, 3, 1. / 3., ie.Get_dI1b(), P);
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
      MFEM_CONTRACT_VAR(w);
      real_t B[9];
      real_t dI1b[9], ddI1[9], ddI1b[9];
      real_t dI2[9], dI2b[9], ddI2[9], ddI2b[9];
      real_t *dI3b = Jrt, *ddI3b = Jpr;

      // ddI1b/3
      kernels::InvariantsEvaluator3D ie(Args()
                                        .J(Jpt)
                                        .B(B)
                                        .dI1b(dI1b)
                                        .ddI1(ddI1)
                                        .ddI1b(ddI1b)
                                        .dI2(dI2)
                                        .dI2b(dI2b)
                                        .ddI2(ddI2)
                                        .ddI2b(ddI2b)
                                        .dI3b(dI3b)
                                        .ddI3b(ddI3b));

      const real_t c1 = weight / 3.;
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi1b(ie.Get_ddI1b(i, j), DIM, DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  const real_t dp = ddi1b(r, c);
                  H(r, c, i, j, qx, qy, qz, e) = c1 * dp;
               }
            }
         }
      }
   }
};

using metric = TMOP_PA_Metric_303;

using assemble = TMOPAssembleGradPA3D;
using energy = TMOPEnergyPA3D;
using mult = TMOPAddMultPA3D;

MFEM_TMOP_REGISTER_METRIC(metric, assemble, energy, mult, 303);

} // namespace mfem
