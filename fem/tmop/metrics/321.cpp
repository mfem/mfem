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

struct TMOP_PA_Metric_321 : TMOP_PA_Metric_3D
{
   MFEM_HOST_DEVICE
   void EvalP(const real_t (&Jpt)[9], const real_t *w, real_t (&P)[9]) override
   {
      MFEM_CONTRACT_VAR(w);
      // dI1 + (1/I3)*dI2 - (2*I2/I3b^3)*dI3b
      real_t B[9];
      real_t dI1[9], dI2[9], dI3b[9];
      kernels::InvariantsEvaluator3D ie(
         Args().J(Jpt).B(B).dI1(dI1).dI2(dI2).dI3b(dI3b));
      real_t sign_detJ;
      const real_t I3 = ie.Get_I3();
      const real_t alpha = 1.0 / I3;
      const real_t beta = -2. * ie.Get_I2() / (I3 * ie.Get_I3b(sign_detJ));
      kernels::Add(3, 3, alpha, ie.Get_dI2(), beta, ie.Get_dI3b(sign_detJ), P);
      kernels::Add(3, 3, ie.Get_dI1(), P);
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

      // ddI1 + (-2/I3b^3)*(dI2 x dI3b + dI3b x dI2)
      //      + (1/I3)*ddI2
      //      + (6*I2/I3b^4)*(dI3b x dI3b)
      //      + (-2*I2/I3b^3)*ddI3b
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
      real_t sign_detJ;
      const real_t I2 = ie.Get_I2();
      const real_t I3b = ie.Get_I3b(sign_detJ);
      ConstDeviceMatrix di2(ie.Get_dI2(), DIM, DIM);
      ConstDeviceMatrix di3b(ie.Get_dI3b(sign_detJ), DIM, DIM);
      const real_t c0 = 1.0 / I3b;
      const real_t c1 = weight * c0 * c0;
      const real_t c2 = -2 * c0 * c1;
      const real_t c3 = c2 * I2;
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi1(ie.Get_ddI1(i, j), DIM, DIM);
            ConstDeviceMatrix ddi2(ie.Get_ddI2(i, j), DIM, DIM);
            ConstDeviceMatrix ddi3b(ie.Get_ddI3b(i, j), DIM, DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  const real_t dp =
                     weight * ddi1(r, c) + c1 * ddi2(r, c) + c3 * ddi3b(r, c) +
                     c2 * ((di2(r, c) * di3b(i, j) + di3b(r, c) * di2(i, j))) -
                     3 * c0 * c3 * di3b(r, c) * di3b(i, j);
                  H(r, c, i, j, qx, qy, qz, e) = dp;
               }
            }
         }
      }
   }
};

using metric_t = TMOP_PA_Metric_321;
using mult_t = TMOPAddMultPA3D;
using setup_t = TMOPSetupGradPA3D;

using setup = tmop::func_t<setup_t>;
using mult = tmop::func_t<mult_t>;

// TMOP PA Setup, metric: 321
MFEM_REGISTER_KERNELS(S321, setup, (int, int));
MFEM_TMOP_ADD_SPECIALIZED_KERNELS(S321);

template <int D, int Q>
setup S321::Kernel()
{
   return setup_t::Mult<metric_t, D, Q>;
}

setup S321::Fallback(int, int) { return setup_t::Mult<metric_t>; }

template <>
void tmop::Kernel<321>(setup_t &ker)
{
   S321::Run(ker.Ndof(), ker.Nqpt(), ker);
}

// TMOP PA Mult, metric: 321
MFEM_REGISTER_KERNELS(K321, mult, (int, int));
MFEM_TMOP_ADD_SPECIALIZED_KERNELS(K321);

template <int D, int Q>
mult K321::Kernel()
{
   return mult_t::Mult<metric_t, D, Q>;
}

mult K321::Fallback(int, int) { return mult_t::Mult<metric_t>; }

template <>
void tmop::Kernel<321>(mult_t &ker)
{
   K321::Run(ker.Ndof(), ker.Nqpt(), ker);
}

} // namespace mfem
