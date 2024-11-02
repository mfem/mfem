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

struct TMOP_PA_Metric_318 : TMOP_PA_Metric_3D
{
   // P_318 = (I3b - 1/I3b^3)*dI3b.
   // Uses the I3b form, as dI3 and ddI3 were not implemented at the time
   MFEM_HOST_DEVICE
   void EvalP(const real_t (&Jpt)[9], const real_t *w, real_t (&P)[9]) override
   {
      MFEM_CONTRACT_VAR(w);
      real_t dI3b[9];
      kernels::InvariantsEvaluator3D ie(Args().J(Jpt).dI3b(dI3b));

      real_t sign_detJ;
      const real_t I3b = ie.Get_I3b(sign_detJ);
      kernels::Set(3, 3, I3b - 1.0 / (I3b * I3b * I3b), ie.Get_dI3b(sign_detJ),
                   P);
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
      real_t *dI3b = Jrt, *ddI3b = Jpr;
      // dP_318 = (I3b - 1/I3b^3)*ddI3b + (1 + 3/I3b^4)*(dI3b x dI3b)
      // Uses the I3b form, as dI3 and ddI3 were not implemented at the time
      kernels::InvariantsEvaluator3D ie(Args().J(Jpt).dI3b(dI3b).ddI3b(ddI3b));
      real_t sign_detJ;
      const real_t I3b = ie.Get_I3b(sign_detJ);
      ConstDeviceMatrix di3b(ie.Get_dI3b(sign_detJ), DIM, DIM);
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi3b(ie.Get_ddI3b(i, j), DIM, DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  const real_t dp =
                     weight * (I3b - 1.0 / (I3b * I3b * I3b)) * ddi3b(r, c) +
                     weight * (1.0 + 3.0 / (I3b * I3b * I3b * I3b)) *
                        di3b(r, c) * di3b(i, j);
                  H(r, c, i, j, qx, qy, qz, e) = dp;
               }
            }
         }
      }
   }
};

void TMOPAssembleGradPA_318(TMOPSetupGradPA3D &ker)
{
   TMOPKernelLaunch<TMOP_PA_Metric_318>(ker);
}

void TMOPAddMultPA_318(TMOPAddMultPA3D &ker)
{
   TMOPKernelLaunch<TMOP_PA_Metric_318>(ker);
}

} // namespace mfem
