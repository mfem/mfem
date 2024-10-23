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

// #define DBG_COLOR ::debug::kGreen
// #include "general/debug.hpp"

#include "../tmop_pa.hpp"
#include "../tmop_pa_h2s.hpp"

#include "../../kernel_dispatch.hpp"

namespace mfem
{

using kernel_t = void (*)(const TMOP_Integrator *, const Vector &x);
using metric_t = mfem::TMOP_PA_Metric_001;

MFEM_REGISTER_KERNELS(Kernels, kernel_t, (int, int));

template <int D, int Q>
kernel_t Kernels::Kernel()
{
   return TMOPSetupGradPA2D_Kernel<metric_t, D, Q>;
}

kernel_t Kernels::Fallback(int, int)
{
   return TMOPSetupGradPA2D_Kernel<metric_t>;
}

void TMOPAssembleGradPA_001(TMOPSetupGradPA2D &ker,
                            const TMOP_Integrator *ti,
                            const Vector &x)
{
   static const auto specialized_kernels = [] { return TMOPAdd<Kernels>(); }();
   const auto d = ker.Ndof(), q = ker.Nqpt();
   Kernels::Run(d, q, ti, x);
}

} // namespace mfem
