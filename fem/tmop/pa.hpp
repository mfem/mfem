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

#ifndef MFEM_TMOP_PA_HPP
#define MFEM_TMOP_PA_HPP

#include "../../config/config.hpp"
#include "../../linalg/dinvariants.hpp"

namespace mfem
{

/// Abstract base class for the 2D metric TMOP PA kernels.
struct TMOP_PA_Metric_2D
{
   static constexpr int DIM = 2;
   using Args = kernels::InvariantsEvaluator2D::Buffers;

   virtual MFEM_HOST_DEVICE void
   EvalP(const real_t (&Jpt)[4], const real_t *w, real_t (&P)[4]) = 0;

   constexpr operator int() const { return 0; }

   operator std::string() const { return "TMOP_PA_Metric_2D"; }

   virtual MFEM_HOST_DEVICE void AssembleH(const int qx,
                                           const int qy,
                                           const int e,
                                           const real_t weight,
                                           const real_t (&Jpt)[4],
                                           const real_t *w,
                                           const DeviceTensor<7> &H) = 0;
};

/// Abstract base class for the 3D metric TMOP PA kernels.
struct TMOP_PA_Metric_3D
{
   static constexpr int DIM = 3;
   using Args = kernels::InvariantsEvaluator3D::Buffers;

   virtual MFEM_HOST_DEVICE void EvalP(const real_t (&Jpt)[DIM * DIM],
                                       const real_t *w,
                                       real_t (&P)[DIM * DIM]) = 0;

   virtual MFEM_HOST_DEVICE void
   AssembleH(const int qx,
             const int qy,
             const int qz,
             const int e,
             const real_t weight,
             real_t *Jrt,
             real_t *Jpr,
             const real_t (&Jpt)[DIM * DIM],
             const real_t *w,
             const DeviceTensor<5 + DIM> &H) const = 0;
};

template <typename M /* metric */, typename K /* kernel */>
static void TMOPKernelLaunch(K &ker)
{
   const int d = ker.Ndof(), q = ker.Nqpt();

   if (d == 2 && q == 2) { return ker.template operator()<M, 2, 2>(); }
   if (d == 2 && q == 3) { return ker.template operator()<M, 2, 3>(); }
   if (d == 2 && q == 4) { return ker.template operator()<M, 2, 4>(); }
   if (d == 2 && q == 5) { return ker.template operator()<M, 2, 5>(); }
   if (d == 2 && q == 6) { return ker.template operator()<M, 2, 6>(); }

   if (d == 3 && q == 3) { return ker.template operator()<M, 3, 3>(); }
   if (d == 3 && q == 4) { return ker.template operator()<M, 3, 4>(); }
   if (d == 3 && q == 5) { return ker.template operator()<M, 3, 5>(); }
   if (d == 3 && q == 6) { return ker.template operator()<M, 3, 6>(); }

   if (d == 4 && q == 4) { return ker.template operator()<M, 4, 4>(); }
   if (d == 4 && q == 5) { return ker.template operator()<M, 4, 5>(); }
   if (d == 4 && q == 6) { return ker.template operator()<M, 4, 6>(); }

   if (d == 5 && q == 5) { return ker.template operator()<M, 5, 5>(); }
   if (d == 5 && q == 6) { return ker.template operator()<M, 5, 6>(); }

   ker.template operator()<M, 0, 0>();
}

template <typename K>
int TMOPAdd()
{
   K::template Specialization<2, 2>::Add();
   K::template Specialization<2, 3>::Add();
   K::template Specialization<2, 4>::Add();
   K::template Specialization<2, 5>::Add();
   K::template Specialization<2, 6>::Add();

   K::template Specialization<3, 3>::Add();
   K::template Specialization<3, 4>::Add();
   K::template Specialization<3, 5>::Add();
   K::template Specialization<3, 6>::Add();

   K::template Specialization<4, 4>::Add();
   K::template Specialization<4, 5>::Add();
   K::template Specialization<4, 6>::Add();

   K::template Specialization<5, 5>::Add();
   K::template Specialization<5, 6>::Add();
   return 0;
}

} // namespace mfem

#endif // MFEM_TMOP_PA_HPP
