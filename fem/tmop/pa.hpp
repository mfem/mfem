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

#ifndef MFEM_TMOP_PA_HPP
#define MFEM_TMOP_PA_HPP

#include "../kernel_dispatch.hpp"
#include "../../config/config.hpp"
#include "../../linalg/dinvariants.hpp"

namespace mfem
{

/// Abstract base class for the 2D metric TMOP PA kernels.
struct TMOP_PA_Metric_2D
{
   static constexpr int DIM = 2;
   using Args = kernels::InvariantsEvaluator2D::Buffers;

   virtual MFEM_HOST_DEVICE real_t EvalW(const real_t (&Jpt)[DIM * DIM],
                                         const real_t *w) = 0;

   virtual MFEM_HOST_DEVICE void EvalP(const real_t (&Jpt)[DIM * DIM],
                                       const real_t *w,
                                       real_t (&P)[DIM * DIM]) = 0;

   virtual MFEM_HOST_DEVICE void AssembleH(const int qx,
                                           const int qy,
                                           const int e,
                                           const real_t weight,
                                           const real_t (&Jpt)[DIM * DIM],
                                           const real_t *w,
                                           const DeviceTensor<7> &H) = 0;

};

/// Abstract base class for the 3D metric TMOP PA kernels.
struct TMOP_PA_Metric_3D
{
   static constexpr int DIM = 3;
   using Args = kernels::InvariantsEvaluator3D::Buffers;

   virtual MFEM_HOST_DEVICE real_t EvalW(const real_t (&Jpt)[DIM * DIM],
                                         const real_t *w) = 0;

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

namespace tmop
{

template <typename T>
using func_t = void (*)(T &);

template <int Metric, typename Ker>
void Kernel(Ker &);

#define MFEM_TMOP_REGISTER_KERNELS(Name, Ker)                           \
   using Ker##_t = decltype(&Ker<>);                                    \
   MFEM_REGISTER_KERNELS(Name, Ker##_t, (int, int));                    \
   template <int D, int Q> Ker##_t Name::Kernel() { return Ker<D, Q>; } \
   Ker##_t Name::Fallback(int, int) { return Ker<>; }

#define MFEM_TMOP_REGISTER_KERNELS_1(Name, Ker)               \
   using Ker##_t = decltype(&Ker<>);                          \
   MFEM_REGISTER_KERNELS(Name, Ker##_t, (int));               \
   template <int Q> Ker##_t Name::Kernel() { return Ker<Q>; } \
   Ker##_t Name::Fallback(int) { return Ker<>; }

#define MFEM_TMOP_ADD_SPECIALIZED_KERNELS(Name) \
namespace { static bool k##Name { (tmop::KernelSpecializations<Name>(), true)}; }

#define MFEM_TMOP_ADD_SPECIALIZED_KERNELS_1(Name) \
namespace { static bool k##Name { (tmop::KernelSpecializations1<Name>(), true)}; }

template <typename Kernel>
int KernelSpecializations()
{
   Kernel::template Specialization<2, 2>::Add();
   Kernel::template Specialization<2, 3>::Add();
   Kernel::template Specialization<2, 4>::Add();
   Kernel::template Specialization<2, 5>::Add();
   Kernel::template Specialization<2, 6>::Add();

   Kernel::template Specialization<3, 3>::Add();
   Kernel::template Specialization<3, 4>::Add();
   Kernel::template Specialization<3, 5>::Add();
   Kernel::template Specialization<3, 6>::Add();

   Kernel::template Specialization<4, 4>::Add();
   Kernel::template Specialization<4, 5>::Add();
   Kernel::template Specialization<4, 6>::Add();

   Kernel::template Specialization<5, 5>::Add();
   Kernel::template Specialization<5, 6>::Add();
   return 0;
}

template <typename Kernel>
int KernelSpecializations1()
{
   Kernel::template Specialization<2>::Add();
   Kernel::template Specialization<3>::Add();
   Kernel::template Specialization<4>::Add();
   Kernel::template Specialization<5>::Add();
   Kernel::template Specialization<6>::Add();
   return 0;
}

} // namespace tmop

} // namespace mfem

#endif // MFEM_TMOP_PA_HPP
