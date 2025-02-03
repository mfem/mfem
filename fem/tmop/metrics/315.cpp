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
#include "../mult/mult3.hpp"
#include "../assemble/grad3.hpp"

namespace mfem
{

struct TMOP_PA_Metric_315 : TMOP_PA_Metric_3D
{
   MFEM_HOST_DEVICE
   void EvalP(const real_t (&Jpt)[9], const real_t *w, real_t (&P)[9]) override
   {
      MFEM_CONTRACT_VAR(w);
      // 2*(I3b - 1)*dI3b
      real_t dI3b[9];
      kernels::InvariantsEvaluator3D ie(Args().J(Jpt).dI3b(dI3b));
      real_t sign_detJ;
      const real_t I3b = ie.Get_I3b(sign_detJ);
      kernels::Set(3, 3, 2.0 * (I3b - 1.0), ie.Get_dI3b(sign_detJ), P);
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
      // 2*(dI3b x dI3b) + 2*(I3b - 1)*ddI3b
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
                  const real_t dp = 2.0 * weight * (I3b - 1.0) * ddi3b(r, c) +
                                    2.0 * weight * di3b(r, c) * di3b(i, j);
                  H(r, c, i, j, qx, qy, qz, e) = dp;
               }
            }
         }
      }
   }
};

using metric_t = TMOP_PA_Metric_315;
using mult_t = TMOPAddMultPA3D;
using setup_t = TMOPSetupGradPA3D;

using setup = tmop::func_t<setup_t>;
using mult = tmop::func_t<mult_t>;

// TMOP PA Setup, metric: 315
MFEM_REGISTER_KERNELS(S315, setup, (int, int));
MFEM_TMOP_ADD_SPECIALIZED_KERNELS(S315);

template <int D, int Q>
setup S315::Kernel()
{
   return setup_t::Mult<metric_t, D, Q>;
}

setup S315::Fallback(int, int) { return setup_t::Mult<metric_t>; }

template <>
void tmop::Kernel<315>(setup_t &ker)
{
   S315::Run(ker.Ndof(), ker.Nqpt(), ker);
}

// TMOP PA Mult, metric: 315
MFEM_REGISTER_KERNELS(K315, mult, (int, int));
MFEM_TMOP_ADD_SPECIALIZED_KERNELS(K315);

template <int D, int Q>
mult K315::Kernel()
{
   return mult_t::Mult<metric_t, D, Q>;
}

mult K315::Fallback(int, int) { return mult_t::Mult<metric_t>; }

template <>
void tmop::Kernel<315>(mult_t &ker)
{
   K315::Run(ker.Ndof(), ker.Nqpt(), ker);
}

} // namespace mfem
