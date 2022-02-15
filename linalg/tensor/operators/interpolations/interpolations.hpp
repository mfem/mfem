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

#ifndef MFEM_TENSOR_INTERP
#define MFEM_TENSOR_INTERP

/// Interpolation algorithms for non-tensor elements
#include "intepr_non-tensor.hpp"
/// Interpolation algorithms for tensor elements
#include "interp_tensor.hpp"
/// Interpolation algorithms for non-tensor elements with vdim
#include "interp_tensor_with_vdim.hpp"
/// Interpolation algorithms from: SmemPAMassApply3D
#include "interp_legacy.hpp"
/// Interpolation that compute matrix entries instead of tensor contractions
#include "interp_untensorized.hpp"
/// Interpolation algorithms for Nedelec elements
#include "interp_nedelec.hpp"

#endif // MFEM_TENSOR_INTERP
