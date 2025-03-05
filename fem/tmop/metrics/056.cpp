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

struct TMOP_PA_Metric_056 : TMOP_PA_Metric_2D
{
   MFEM_HOST_DEVICE real_t EvalW(const real_t (&Jpt)[4], const real_t *w) override
   {
      MFEM_CONTRACT_VAR(w);
      kernels::InvariantsEvaluator2D ie(Args().J(Jpt));
      const real_t I2b = ie.Get_I2b();
      return 0.5 * (I2b + 1.0 / I2b) - 1.0;
   };

   MFEM_HOST_DEVICE
   void EvalP(const real_t (&Jpt)[4], const real_t *w, real_t (&P)[4]) override
   {
      MFEM_CONTRACT_VAR(w);
      // 0.5*(1 - 1/I2b^2)*dI2b
      real_t dI2b[4];
      kernels::InvariantsEvaluator2D ie(Args().J(Jpt).dI2b(dI2b));
      const real_t I2b = ie.Get_I2b();
      kernels::Set(2, 2, 0.5 * (1.0 - 1.0 / (I2b * I2b)), ie.Get_dI2b(), P);
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
      // (0.5 - 0.5/I2b^2)*ddI2b + (1/I2b^3)*(dI2b x dI2b)
      real_t dI2b[4], ddI2b[4];
      kernels::InvariantsEvaluator2D ie(Args().J(Jpt).dI2b(dI2b).ddI2b(ddI2b));
      const real_t I2b = ie.Get_I2b();
      ConstDeviceMatrix di2b(ie.Get_dI2b(), DIM, DIM);
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi2b(ie.Get_ddI2b(i, j), DIM, DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  H(r, c, i, j, qx, qy, e) =
                     weight * (0.5 - 0.5 / (I2b * I2b)) * ddi2b(r, c) +
                     weight / (I2b * I2b * I2b) * di2b(r, c) * di2b(i, j);
               }
            }
         }
      }
   }
};

using metric_t = TMOP_PA_Metric_056;
using mult_t = TMOPAddMultPA2D;
using setup_t = TMOPSetupGradPA2D;

using setup = tmop::func_t<setup_t>;
using mult = tmop::func_t<mult_t>;

// TMOP PA Setup, metric: 056
MFEM_REGISTER_KERNELS(S056, setup, (int, int));
MFEM_TMOP_ADD_SPECIALIZED_KERNELS(S056);

template <int D, int Q>
setup S056::Kernel()
{
   return setup_t::Mult<metric_t, D, Q>;
}

setup S056::Fallback(int, int) { return setup_t::Mult<metric_t>; }

template <>
void tmop::Kernel<56>(setup_t &ker)
{
   S056::Run(ker.Ndof(), ker.Nqpt(), ker);
}

// TMOP PA Mult, metric: 056
MFEM_REGISTER_KERNELS(K056, mult, (int, int));
MFEM_TMOP_ADD_SPECIALIZED_KERNELS(K056);

template <int D, int Q>
mult K056::Kernel()
{
   return mult_t::Mult<metric_t, D, Q>;
}

mult K056::Fallback(int, int) { return mult_t::Mult<metric_t>; }

template <>
void tmop::Kernel<56>(mult_t &ker)
{
   K056::Run(ker.Ndof(), ker.Nqpt(), ker);
}

// TMOP PA Energy, metric: 056
using energy_t = TMOPEnergyPA2D;
using energy = tmop::func_t<energy_t>;

MFEM_REGISTER_KERNELS(E056, energy, (int, int));
MFEM_TMOP_ADD_SPECIALIZED_KERNELS(E056);

template <int D, int Q>
energy E056::Kernel()
{
   return energy_t::Mult<metric_t, D, Q>;
}

energy E056::Fallback(int, int) { return energy_t::Mult<metric_t>; }

template <>
void tmop::Kernel<56>(energy_t &ker)
{
   E056::Run(ker.Ndof(), ker.Nqpt(), ker);
}

} // namespace mfem
