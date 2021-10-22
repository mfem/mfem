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

#ifndef MFEM_TENSOR_GRAD
#define MFEM_TENSOR_GRAD

/// Non-tensor gradient algorithms for non-tensor elements
#include "grad_non-tensor.hpp"
/// Tensor gradient algorithms for tensor elements
#include "grad_tensor.hpp"
/// Tensor gradient algorithms for non-tensor elements with vdim
#include "grad_tensor_with_vdim.hpp"
/// Tensor gradient algorithms from: SmemPADiffusionApply3D
#include "grad_legacy.hpp"
/// Tensor gradient that compute matrix entries instead of using tensor
/// contractions
#include "grad_untensorized.hpp"

#endif // MFEM_TENSOR_GRAD
