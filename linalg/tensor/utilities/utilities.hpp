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

#ifndef MFEM_TENSOR_UTILITIES
#define MFEM_TENSOR_UTILITIES

/**
 * Utilities regroup important tensor abstractions to implement algorithms on
 * tensors.
 * */

/// Helper constants
#include "helper_constants.hpp"
/// Factories for kernel configurations
#include "config.hpp"
/// Function to iterate on tensor dimensions
#include "foreach.hpp"
/// MFEM_FORALL_CONFIG selecting threading strategy according to the KernelConfig
#include "forall.hpp"
/// Utility functions, and TMP patterns
#include "util.hpp"

#endif // MFEM_TENSOR_UTILITIES
