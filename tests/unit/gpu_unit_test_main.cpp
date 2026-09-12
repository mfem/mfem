// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#define CATCH_CONFIG_RUNNER
#include "mfem.hpp"
#include "run_unit_tests.hpp"

int main(int argc, char *argv[])
{
#ifdef MFEM_USE_SINGLE
   std::cout << "\nThe serial GPU unit tests are not supported in single"
             " precision.\n\n";
   return MFEM_SKIP_RETURN_VALUE;
#endif

   // **MFEM_TEST_DEVICE, so this binary's cases can be run on the DEBUG
   // backend as well as on the GPU.** Unset, nothing changes.
   //
   // Device("debug") has device memory semantics with host arithmetic and
   // mprotects the host page, so a raw host read of a device-valid buffer is
   // an MmuError with a backtrace rather than a wrong number. That matters
   // here because debug_device_tests is built from
   // miniapps/test_debug_device.cpp ALONE, so no [GPU]-tagged case can reach
   // it by any other route.
   //
   // **A green GPU run is not on its own evidence of device discipline.** An
   // unregistered or stale host pointer usually still reads on CUDA, and
   // often reads the right value, where this backend traps it. Sweep per
   // case on both.
   //
   // Not every [GPU] case can take it: upstream kernels that abort with "This
   // kernel should only be used on GPU" (SmemPACurlCurlApply3D, reached from
   // test_pa_coeff.cpp) refuse any non-GPU backend by design. So sweep the
   // cases you own rather than the whole binary.
   const char *dev_spec = getenv("MFEM_TEST_DEVICE");
   mfem::Device device(dev_spec ? dev_spec : "gpu");

   // Include only tests labeled with GPU. Exclude parallel tests.
   return RunCatchSession(argc, argv, {"[GPU]", "~[Parallel]"});
}
