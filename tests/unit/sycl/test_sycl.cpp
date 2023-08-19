// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mfem.hpp"
using namespace mfem;

#include "unit_tests.hpp"

#define MFEM_DEBUG_COLOR 51
#include "general/debug.hpp"

#ifdef MFEM_USE_SYCL

void sycl_lmem();
void sycl_kernels();
void sycl_diffusion();

static void sycl_tests()
{
   sycl_lmem();
   sycl_kernels();
   sycl_diffusion();
}

#ifndef MFEM_SYCL_DEVICE
TEST_CASE("Sycl", "[Sycl]")
{
   sycl_tests();
}
#else // MFEM_SYCL_DEVICE
TEST_CASE("Sycl", "[Sycl]")
{
   mfem::Device device;
   device.Configure(MFEM_SYCL_DEVICE);
   device.Print();
   sycl_tests();
}
#endif // MFEM_SYCL_DEVICE

#endif // MFEM_USE_SYCL
