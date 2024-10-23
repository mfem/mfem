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

#define DBG_COLOR ::debug::kGreen
#include "general/debug.hpp"

#include "../tmop_pa.hpp"
#include "../tmop_pa_h2s.hpp"

namespace std
{
template <>
struct hash<mfem::TMOP_PA_Metric_001>
{
   size_t operator()(const mfem::TMOP_PA_Metric_001 &) const noexcept
   {
      dbg("TMOP_PA_Metric_001 hash");
      return 1;
   }
};
} // namespace std

#include "../../kernel_dispatchT.hpp"

namespace mfem
{

using kernel_t = void (*)(const TMOP_Integrator *, const Vector &x);

using metric_t = decltype(typename mfem::TMOP_PA_Metric_001{});

MFEM_REGISTER_KERNELS_T(Kernels, kernel_t, (metric_t, int, int));

template <metric_t M, int D, int Q>
kernel_t Kernels::Kernel()
{
   dbg("TMOP_PA_Metric_001 decltype(kernel_t):");
   printTypes<kernel_t>();

   // dbg("TMOP_PA_Metric_001 decltype(Args):");
   // printTypes<decltype(Args)...>();

   // dbg("TMOP_PA_Metric_001 printValues(Args):");
   // printValues(Args...);

   // (std::cout << ... << Args) << std::endl;

   // dbg(
   //    "TMOP_PA_Metric_001 decltype(TMOPSetupGradPA2D_Kernel<metric_t, 2,
   //    3>):");
   // printTypes<decltype(TMOPSetupGradPA2D_Kernel<metric_t, 2, 3>)>();

   // dbg("TMOP_PA_Metric_001 decltype(TMOPSetupGradPA2D_Kernel<Args...>):");
   // printTypes<decltype(TMOPSetupGradPA2D_Kernel<Args...>)>();

   // return TMOPSetupGradPA2D_Kernel<args...>;
   // return TMOPSetupGradPA2D_Kernel<metric_t, 2, 3>;
   return TMOPSetupGradPA2D_Kernel<decltype(M), D, Q>;
}

kernel_t Kernels::Fallback(metric_t, int, int)
{
   dbg("TMOP_PA_Metric_001, Fallback");
   return TMOPSetupGradPA2D_Kernel<metric_t>;
}

// static auto add_kernels = [] { return (TMOPAdd<metric_t, Kernels>(), 0); }();
static auto add_kernels = []
{
   return (Kernels::template Specialization<metric_t{}, 2, 2>::Add(),
           Kernels::template Specialization<metric_t{}, 2, 3>::Add(), 0);
}();

void TMOPAssembleGradPA_001(TMOPSetupGradPA2D &ker,
                            const TMOP_Integrator *ti,
                            const Vector &x)
{
   // TMOPKernelLaunch<TMOP_PA_Metric_001>(ker);

   const int d = ker.Ndof(), q = ker.Nqpt();
   dbg("TMOP_PA_Metric_001, d:{}, q:{}", d, q);
   Kernels::Run(metric_t{}, d, q, ti, x);
}

} // namespace mfem
