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

#include "../../kernel_dispatch.hpp"

#include "../pa.hpp"
#include "../mult/mult3.hpp"
#include "../assemble/grad3.hpp"

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

using metric_t = TMOP_PA_Metric_302;
using mult_t = TMOPAddMultPA3D;
using setup_t = TMOPSetupGradPA3D;

using setup = tmop::func_t<setup_t>;
using mult = tmop::func_t<mult_t>;

// TMOP PA Setup, metric: 302
MFEM_REGISTER_KERNELS(S302, setup, (int, int));

template <int D, int Q>
setup S302::Kernel()
{
   return setup_t::Mult<metric_t, D, Q>;
}

setup S302::Fallback(int, int) { return setup_t::Mult<metric_t>; }

template <>
void tmop::Kernel<302>(setup_t &ker)
{
   const static auto setup_kernels = []
   { return KernelSpecializations<S302>(); }();
   S302::Run(ker.Ndof(), ker.Nqpt(), ker);
}

// TMOP PA Mult, metric: 302

MFEM_REGISTER_KERNELS(K302, mult, (int, int));

template <int D, int Q>
mult K302::Kernel()
{
   return mult_t::Mult<metric_t, D, Q>;
}

mult K302::Fallback(int, int) { return mult_t::Mult<metric_t>; }

template <>
void tmop::Kernel<302>(mult_t &ker)
{
   const static auto mult_kernels = []
   { return KernelSpecializations<K302>(); }();
   K302::Run(ker.Ndof(), ker.Nqpt(), ker);
}

} // namespace mfem
