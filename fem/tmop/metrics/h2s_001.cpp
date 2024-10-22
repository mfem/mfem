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
#include "../../kernel_dispatch.hpp"

#include "../tmop_pa_h2s.hpp"

namespace mfem
{

struct TMOPAssembleGradPA_001_Kernels
{
   using TMOPAssembleGradPA_Kernels_t = void (*)(const TMOP_Integrator *,
                                                 const Vector &x);

   MFEM_REGISTER_KERNELS(TMOPAssembleGradPA_Kernels,
                         TMOPAssembleGradPA_Kernels_t,
                         (int /*D*/, int /*Q*/, int /*MAX*/));

   static struct Kernels
   {
      Kernels();
   } kernels;
};

template <int D, int Q, int MAX>
TMOPAssembleGradPA_001_Kernels::TMOPAssembleGradPA_Kernels_t
TMOPAssembleGradPA_001_Kernels::TMOPAssembleGradPA_Kernels::Kernel()
{
   dbg("\t\033[33mreturning D:{}, Q:{}, MAX:{}", D, Q, MAX);
   return TMOPSetupGradPA2D_Kernel<TMOP_PA_Metric_001, D, Q, MAX>;
}

TMOPAssembleGradPA_001_Kernels::Kernels::Kernels()
{
   dbg("Adding kernels for TMOPAssembleGradPA_001");
   using Ker = TMOPAssembleGradPA_001_Kernels::TMOPAssembleGradPA_Kernels;

   Ker::Specialization<2, 2, 4>::Add();
   Ker::Specialization<2, 3, 4>::Add();
   Ker::Specialization<2, 4, 4>::Add();
   Ker::Specialization<2, 5, 4>::Add();
   Ker::Specialization<2, 6, 4>::Add();
}

// populate the kernel map
TMOPAssembleGradPA_001_Kernels::Kernels TMOPAssembleGradPA_001_Kernels::kernels;

TMOPAssembleGradPA_001_Kernels::TMOPAssembleGradPA_Kernels_t
TMOPAssembleGradPA_001_Kernels::TMOPAssembleGradPA_Kernels::Fallback(int,
                                                                     int,
                                                                     int)
{
   MFEM_ABORT("Fallback not implemented.");
}

void TMOPAssembleGradPA_001(TMOPSetupGradPA2D &ker,
                            const TMOP_Integrator *ti,
                            const Vector &x)
{
   // TMOPKernelLaunch<TMOP_PA_Metric_001>(ker);

   const int d = ker.Ndof(), q = ker.Nqpt();
   dbg("TMOP_PA_Metric_001, d:{}, q:{}", d, q);

   TMOPAssembleGradPA_001_Kernels::TMOPAssembleGradPA_Kernels::Run(d, q, 4, //
                                                                   ti, x);
}

} // namespace mfem
