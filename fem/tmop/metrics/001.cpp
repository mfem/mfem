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

struct TMOP_PA_Metric_001 : TMOP_PA_Metric_2D
{
   MFEM_HOST_DEVICE real_t EvalW(const real_t (&Jpt)[4], const real_t *w) override
   {
      MFEM_CONTRACT_VAR(w);
      kernels::InvariantsEvaluator2D ie(Args().J(Jpt));
      return ie.Get_I1();
   };

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

using metric = TMOP_PA_Metric_001;

using assemble = TMOPAssembleGradPA2D;
using energy = TMOPEnergyPA2D;
using mult = TMOPAddMultPA2D;

MFEM_TMOP_REGISTER_METRIC(metric, assemble, energy, mult, 1);

} // namespace mfem
