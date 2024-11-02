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

#include "../pa.hpp"
#include "../mult/p3.hpp"
#include "../setup/h3s.hpp"

namespace mfem
{

struct TMOP_PA_Metric_302 : TMOP_PA_Metric_3D
{
   MFEM_HOST_DEVICE
   void EvalP(const real_t (&Jpt)[9], const real_t *w, real_t (&P)[9]) override
   {
      MFEM_CONTRACT_VAR(w);
      // (I1b/9)*dI2b + (I2b/9)*dI1b
      real_t B[9];
      real_t dI1b[9], dI2[9], dI2b[9], dI3b[9];
      kernels::InvariantsEvaluator3D ie(
         Args().J(Jpt).B(B).dI1b(dI1b).dI2(dI2).dI2b(dI2b).dI3b(dI3b));
      const real_t alpha = ie.Get_I1b() / 9.;
      const real_t beta = ie.Get_I2b() / 9.;
      kernels::Add(3, 3, alpha, ie.Get_dI2b(), beta, ie.Get_dI1b(), P);
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
      MFEM_CONTRACT_VAR(Jrt);
      MFEM_CONTRACT_VAR(Jpr);
      MFEM_CONTRACT_VAR(w);
      real_t B[9];
      real_t dI1b[9], ddI1b[9];
      real_t dI2[9], dI2b[9], ddI2[9], ddI2b[9];
      real_t dI3b[9]; // = Jrt;
      // (dI2b*dI1b + dI1b*dI2b)/9 + (I1b/9)*ddI2b + (I2b/9)*ddI1b
      kernels::InvariantsEvaluator3D ie(Args()
                                           .J(Jpt)
                                           .B(B)
                                           .dI1b(dI1b)
                                           .ddI1b(ddI1b)
                                           .dI2(dI2)
                                           .dI2b(dI2b)
                                           .ddI2(ddI2)
                                           .ddI2b(ddI2b)
                                           .dI3b(dI3b));

      const real_t c1 = weight / 9.;
      const real_t I1b = ie.Get_I1b();
      const real_t I2b = ie.Get_I2b();
      ConstDeviceMatrix di1b(ie.Get_dI1b(), DIM, DIM);
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
                  const real_t dp =
                     (di2b(r, c) * di1b(i, j) + di1b(r, c) * di2b(i, j)) +
                     ddi2b(r, c) * I1b + ddi1b(r, c) * I2b;
                  H(r, c, i, j, qx, qy, qz, e) = c1 * dp;
               }
            }
         }
      }
   }
};

void TMOPAssembleGradPA_302(TMOPSetupGradPA3D &ker)
{
   TMOPKernelLaunch<TMOP_PA_Metric_302>(ker);
}

void TMOPAddMultPA_302(TMOPAddMultPA3D &ker)
{
   TMOPKernelLaunch<TMOP_PA_Metric_302>(ker);
}

} // namespace mfem
