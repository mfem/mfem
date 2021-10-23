// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_TENSOR_UTIL_CONSTANTS
#define MFEM_TENSOR_UTIL_CONSTANTS

namespace mfem
{

/// Compile time constant used to signify dynamic dimensions.
static constexpr int Dynamic = 0;

/// Compile time constant used to signify an error.
static constexpr int Error = -1;

/// The arbitrary maximum dynamic dimension size for stack allocated tensors.
static constexpr int DynamicMaxSize = 16;

/// Compile time constant indicating if the code being compiled is for device.
#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__))
static constexpr bool is_device = true;
#else
static constexpr bool is_device = false;
#endif

} // mfem namespace

#endif // MFEM_TENSOR_UTIL_CONSTANTS
