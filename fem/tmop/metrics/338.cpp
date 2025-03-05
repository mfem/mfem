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

struct TMOP_PA_Metric_338 : TMOP_PA_Metric_3D
{
   MFEM_HOST_DEVICE real_t EvalW(const real_t (&Jpt)[DIM * DIM],
                                 const real_t *w) override
   {
      real_t B[9];
      MFEM_CONTRACT_VAR(w);
      kernels::InvariantsEvaluator3D ie(Args().J(Jpt).B(B));
      const real_t eval_w_302 = ie.Get_I1b() * ie.Get_I2b() / 9. - 1.;
      const real_t I3 = ie.Get_I3();
      const real_t eval_w_318 = 0.5 * (I3 + 1.0 / I3) - 1.0;
      return w[0] * eval_w_302 + w[1] * eval_w_318;
   }

   MFEM_HOST_DEVICE
   void EvalP(const real_t (&Jpt)[9], const real_t *w, real_t (&P)[9]) override
   {
      // w0 P_302 + w1 P_318
      real_t B[9];
      real_t dI1b[9], dI2[9], dI2b[9], dI3b[9];
      kernels::InvariantsEvaluator3D ie(
         Args().J(Jpt).B(B).dI1b(dI1b).dI2(dI2).dI2b(dI2b).dI3b(dI3b));
      const real_t alpha = w[0] * ie.Get_I1b() / 9.;
      const real_t beta = w[0] * ie.Get_I2b() / 9.;
      kernels::Add(3, 3, alpha, ie.Get_dI2b(), beta, ie.Get_dI1b(), P);
      real_t sign_detJ;
      const real_t I3b = ie.Get_I3b(sign_detJ);
      kernels::Add(3, 3, w[1] * (I3b - 1.0 / (I3b * I3b * I3b)),
                   ie.Get_dI3b(sign_detJ), P);
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
      real_t dI1b[9], ddI1b[9];
      real_t dI2[9], dI2b[9], ddI2[9], ddI2b[9];
      real_t *dI3b = Jrt, *ddI3b = Jpr;
      // w0 H_302 + w1 H_318
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
      const real_t c1 = weight / 9.;
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
                  const real_t dp_318 =
                     weight * (I3b - 1.0 / (I3b * I3b * I3b)) * ddi3b(r, c) +
                     weight * (1.0 + 3.0 / (I3b * I3b * I3b * I3b)) *
                     di3b(r, c) * di3b(i, j);
                  H(r, c, i, j, qx, qy, qz, e) =
                     w[0] * c1 * dp_302 + w[1] * dp_318;
               }
            }
         }
      }
   }
};

using metric_t = TMOP_PA_Metric_338;

// TMOP PA Setup, metric: 338
using setup_t = TMOPSetupGradPA3D;
using setup = tmop::func_t<setup_t>;
MFEM_REGISTER_KERNELS(S338, setup, (int, int));
MFEM_TMOP_ADD_SPECIALIZED_KERNELS(S338);

template <int D, int Q>
setup S338::Kernel()
{
   return setup_t::Mult<metric_t, D, Q>;
}

setup S338::Fallback(int, int) { return setup_t::Mult<metric_t>; }

template <>
void tmop::Kernel<338>(setup_t &ker)
{
   S338::Run(ker.Ndof(), ker.Nqpt(), ker);
}

// TMOP PA Mult, metric: 338
using mult_t = TMOPAddMultPA3D;
using mult = tmop::func_t<mult_t>;
MFEM_REGISTER_KERNELS(K338, mult, (int, int));
MFEM_TMOP_ADD_SPECIALIZED_KERNELS(K338);

template <int D, int Q>
mult K338::Kernel()
{
   return mult_t::Mult<metric_t, D, Q>;
}

mult K338::Fallback(int, int) { return mult_t::Mult<metric_t>; }

template <>
void tmop::Kernel<338>(mult_t &ker)
{
   K338::Run(ker.Ndof(), ker.Nqpt(), ker);
}

// TMOP PA Energy, metric: 338
using energy_t = TMOPEnergyPA3D;
using energy = tmop::func_t<energy_t>;
MFEM_REGISTER_KERNELS(E338, energy, (int, int));
MFEM_TMOP_ADD_SPECIALIZED_KERNELS(E338);

template <int D, int Q>
energy E338::Kernel()
{
   return energy_t::Mult<metric_t, D, Q>;
}

energy E338::Fallback(int, int) { return energy_t::Mult<metric_t>; }

template <>
void tmop::Kernel<338>(energy_t &ker)
{
   E338::Run(ker.Ndof(), ker.Nqpt(), ker);
}

} // namespace mfem
