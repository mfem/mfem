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

/// Tests which make sure adding user-defined kernel specializations work.
/// These tests are compile/link-only tests

#include "mfem.hpp"
#include "unit_tests.hpp"

#include "fem/lor/lor_batched_kernels.hpp"

TEST_CASE("Batched LOR Kernel Specializations", "[Specializations]")
{
   using namespace mfem;
   using namespace mfem::internal;

   BatchedLORAssemblyKernels<BatchedLOR_H1>::AssembleKernels::Specialization<6>::Add();
   BatchedLORAssemblyKernels<BatchedLOR_ND>::AssembleKernels::Specialization<6>::Add();
   BatchedLORAssemblyKernels<BatchedLOR_RT>::AssembleKernels::Specialization<6>::Add();
   BatchedLORAssemblyKernels<BatchedLOR_DG>::AssembleKernels::Specialization<6>::Add();
}
