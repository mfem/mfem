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
#include "../../tmop.hpp"
#include "../../kernels.hpp"
#include "../../kernels_sm.hpp"
#include "../../../general/forall.hpp"
#include "../../../linalg/kernels.hpp"

namespace mfem
{

template <int T_D1D = 0, int T_Q1D = 0>
void TMOP_EnergyPA_C0_3D(const real_t lim_normal,
                         const DeviceTensor<4, const real_t> &LD,
                         const bool const_c0,
                         const DeviceTensor<4, const real_t> &C0,
                         const int NE,
                         const DeviceTensor<6, const real_t> &J,
                         const ConstDeviceCube &W,
                         const ConstDeviceMatrix &b,
                         const ConstDeviceMatrix &bld,
                         const DeviceTensor<5, const real_t> &X0,
                         const DeviceTensor<5, const real_t> &X1,
                         DeviceTensor<4> &E,
                         const bool exp_lim,
                         const int d1d,
                         const int q1d)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;

      MFEM_SHARED real_t sB[MQ1 * MD1];
      MFEM_SHARED real_t sBLD[MQ1 * MD1];
      kernels::internal::LoadB<MD1, MQ1>(D1D, Q1D, bld, sBLD);

      MFEM_SHARED real_t sm0[MDQ * MDQ * MDQ];
      MFEM_SHARED real_t sm1[MDQ * MDQ * MDQ];

      MFEM_SHARED real_t s00[3][MDQ * MDQ * MDQ];
      MFEM_SHARED real_t s01[3][MDQ * MDQ * MDQ];

      MFEM_SHARED real_t s10[3][MDQ * MDQ * MDQ];
      MFEM_SHARED real_t s11[3][MDQ * MDQ * MDQ];

      kernels::internal::sm::LoadX<MDQ>(e, D1D, LD, sm0);
      kernels::internal::sm::LoadX<MDQ>(e, D1D, X0, s00);
      kernels::internal::sm::LoadX<MDQ>(e, D1D, X1, s10);

      kernels::internal::LoadB<MD1, MQ1>(D1D, Q1D, b, sB);

      kernels::internal::sm::EvalX<MD1, MQ1>(D1D, Q1D, sBLD, sm0, sm1);
      kernels::internal::sm::EvalY<MD1, MQ1>(D1D, Q1D, sBLD, sm1, sm0);
      kernels::internal::sm::EvalZ<MD1, MQ1>(D1D, Q1D, sBLD, sm0, sm1);

      kernels::internal::sm::EvalX<MD1, MQ1>(D1D, Q1D, sB, s00, s01);
      kernels::internal::sm::EvalY<MD1, MQ1>(D1D, Q1D, sB, s01, s00);
      kernels::internal::sm::EvalZ<MD1, MQ1>(D1D, Q1D, sB, s00, s01);

      kernels::internal::sm::EvalX<MD1, MQ1>(D1D, Q1D, sB, s10, s11);
      kernels::internal::sm::EvalY<MD1, MQ1>(D1D, Q1D, sB, s11, s10);
      kernels::internal::sm::EvalZ<MD1, MQ1>(D1D, Q1D, sB, s10, s11);

      MFEM_FOREACH_THREAD(qz, z, Q1D)
      {
         MFEM_FOREACH_THREAD(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD(qx, x, Q1D)
            {
               real_t D, p0[3], p1[3];
               const real_t *Jtr = &J(0, 0, qx, qy, qz, e);
               const real_t detJtr = kernels::Det<3>(Jtr);
               const real_t weight = W(qx, qy, qz) * detJtr;
               const real_t coeff0 =
                  const_c0 ? C0(0, 0, 0, 0) : C0(qx, qy, qz, e);

               kernels::internal::sm::PullEval<MDQ>(Q1D, qx, qy, qz, sm1, D);
               kernels::internal::sm::PullEval<MDQ>(Q1D, qx, qy, qz, s01, p0);
               kernels::internal::sm::PullEval<MDQ>(Q1D, qx, qy, qz, s11, p1);

               const real_t dist = D; // GetValues, default comp set to 0
               real_t id2 = 0.0;
               real_t dsq = 0.0;
               if (!exp_lim)
               {
                  id2 = 0.5 / (dist * dist);
                  dsq = kernels::DistanceSquared<3>(p1, p0) * id2;
                  E(qx, qy, qz, e) = weight * lim_normal * dsq * coeff0;
               }
               else
               {
                  id2 = 1.0 / (dist * dist);
                  dsq = kernels::DistanceSquared<3>(p1, p0) * id2;
                  E(qx, qy, qz, e) =
                     weight * lim_normal * exp(10.0 * (dsq - 1.0)) * coeff0;
               }
            }
         }
      }
   });
}

MFEM_TMOP_REGISTER_KERNELS(TMOPEnergyPAC03D, TMOP_EnergyPA_C0_3D);
MFEM_TMOP_ADD_SPECIALIZED_KERNELS(TMOPEnergyPAC03D);

real_t TMOP_Integrator::GetLocalStateEnergyPA_C0_3D(const Vector &x) const
{
   constexpr int DIM = 3;
   const real_t ln = lim_normal;
   const bool const_c0 = PA.C0.Size() == 1;
   const int NE = PA.ne, d = PA.maps->ndof, q = PA.maps->nqpt;

   MFEM_VERIFY(PA.maps_lim->ndof == d, "");
   MFEM_VERIFY(PA.maps_lim->nqpt == q, "");

   const auto C0 = const_c0 ? Reshape(PA.C0.Read(), 1, 1, 1, 1)
                   : Reshape(PA.C0.Read(), q, q, q, NE);
   const auto LD = Reshape(PA.LD.Read(), d, d, d, NE);
   const auto J = Reshape(PA.Jtr.Read(), DIM, DIM, q, q, q, NE);
   const auto B = Reshape(PA.maps->B.Read(), q, d);
   const auto BLD = Reshape(PA.maps_lim->B.Read(), q, d);
   const auto W = Reshape(PA.ir->GetWeights().Read(), q, q, q);
   const auto X0 = Reshape(PA.X0.Read(), d, d, d, DIM, NE);
   const auto X1 = Reshape(x.Read(), d, d, d, DIM, NE);
   auto E = Reshape(PA.E.Write(), q, q, q, NE);

   auto el = dynamic_cast<TMOP_ExponentialLimiter *>(lim_func);
   const bool exp_lim = (el) ? true : false;

   TMOPEnergyPAC03D::Run(d, q,
                         ln, LD, const_c0, C0, NE, J, W,
                         B, BLD, X0, X1, E, exp_lim, d, q);

   return PA.E * PA.O;
}

} // namespace mfem
