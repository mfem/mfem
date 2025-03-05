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

struct TMOP_PA_Metric_002 : TMOP_PA_Metric_2D
{
   MFEM_HOST_DEVICE real_t EvalW(const real_t (&Jpt)[4], const real_t *w) override
   {
      MFEM_CONTRACT_VAR(w);
      kernels::InvariantsEvaluator2D ie(Args().J(Jpt));
      return 0.5 * ie.Get_I1b() - 1.0;
   };

   MFEM_HOST_DEVICE
   void EvalP(const real_t (&Jpt)[4], const real_t *w, real_t (&P)[4]) override
   {
      MFEM_CONTRACT_VAR(w);
      real_t dI1b[4], dI2b[4];
      kernels::InvariantsEvaluator2D ie(Args().J(Jpt).dI1b(dI1b).dI2b(dI2b));
      kernels::Set(2, 2, 1. / 2., ie.Get_dI1b(), P);
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
      // 0.5 * weight * dI1b
      real_t ddI1[4], ddI1b[4], dI2b[4];
      kernels::InvariantsEvaluator2D ie(
         Args().J(Jpt).ddI1(ddI1).ddI1b(ddI1b).dI2b(dI2b));
      const real_t half_weight = 0.5 * weight;
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi1b(ie.Get_ddI1b(i, j), DIM, DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  const real_t h = ddi1b(r, c);
                  H(r, c, i, j, qx, qy, e) = half_weight * h;
               }
            }
         }
      }
   }
};

using metric_t = TMOP_PA_Metric_002;

// TMOP PA Setup, metric: 002
using setup_t = TMOPSetupGradPA2D;
using setup = tmop::func_t<setup_t>;

MFEM_REGISTER_KERNELS(S002, setup, (int, int));
MFEM_TMOP_ADD_SPECIALIZED_KERNELS(S002);

template <int D, int Q>
setup S002::Kernel()
{
   return setup_t::Mult<metric_t, D, Q>;
}

setup S002::Fallback(int, int) { return setup_t::Mult<metric_t>; }

template <>
void tmop::Kernel<2>(setup_t &ker)
{
   S002::Run(ker.Ndof(), ker.Nqpt(), ker);
}

// TMOP PA Mult, metric: 002
using mult_t = TMOPAddMultPA2D;
using mult = tmop::func_t<mult_t>;

MFEM_REGISTER_KERNELS(K002, mult, (int, int));
MFEM_TMOP_ADD_SPECIALIZED_KERNELS(K002);

template <int D, int Q>
mult K002::Kernel()
{
   return mult_t::Mult<metric_t, D, Q>;
}

mult K002::Fallback(int, int) { return mult_t::Mult<metric_t>; }

template <>
void tmop::Kernel<2>(mult_t &ker)
{
   K002::Run(ker.Ndof(), ker.Nqpt(), ker);
}

// TMOP PA Energy, metric: 002
using energy_t = TMOPEnergyPA2D;
using energy = tmop::func_t<energy_t>;

MFEM_REGISTER_KERNELS(E002, energy, (int, int));
MFEM_TMOP_ADD_SPECIALIZED_KERNELS(E002);

template <int D, int Q>
energy E002::Kernel()
{
   return energy_t::Mult<metric_t, D, Q>;
}

energy E002::Fallback(int, int) { return energy_t::Mult<metric_t>; }

template <>
void tmop::Kernel<2>(energy_t &ker)
{
   E002::Run(ker.Ndof(), ker.Nqpt(), ker);
}

} // namespace mfem
