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

using metric_t = TMOP_PA_Metric_001;

// TMOP PA Setup, metric: 001
using setup_t = TMOPSetupGradPA2D;
using setup = tmop::func_t<setup_t>;

MFEM_REGISTER_KERNELS(S001, setup, (int, int));
MFEM_TMOP_ADD_SPECIALIZED_KERNELS(S001);

template <int D, int Q>
setup S001::Kernel()
{
   return setup_t::Mult<metric_t, D, Q>;
}

setup S001::Fallback(int, int) { return setup_t::Mult<metric_t>; }

template <>
void tmop::Kernel<1>(setup_t &ker)
{
   S001::Run(ker.Ndof(), ker.Nqpt(), ker);
}

// TMOP PA Mult, metric: 001
using mult_t = TMOPAddMultPA2D;
using mult = tmop::func_t<mult_t>;

MFEM_REGISTER_KERNELS(K001, mult, (int, int));
MFEM_TMOP_ADD_SPECIALIZED_KERNELS(K001);

template <int D, int Q>
mult K001::Kernel()
{
   return mult_t::Mult<metric_t, D, Q>;
}

mult K001::Fallback(int, int) { return mult_t::Mult<metric_t>; }

template <>
void tmop::Kernel<1>(mult_t &ker)
{
   K001::Run(ker.Ndof(), ker.Nqpt(), ker);
}

// TMOP PA Energy, metric: 001
using energy_t = TMOPEnergyPA2D;
using energy = tmop::func_t<energy_t>;

MFEM_REGISTER_KERNELS(E001, energy, (int, int));
MFEM_TMOP_ADD_SPECIALIZED_KERNELS(E001);

template <int D, int Q>
energy E001::Kernel()
{
   return energy_t::Mult<metric_t, D, Q>;
}

energy E001::Fallback(int, int) { return energy_t::Mult<metric_t>; }

template <>
void tmop::Kernel<1>(energy_t &ker)
{
   E001::Run(ker.Ndof(), ker.Nqpt(), ker);
}

} // namespace mfem
