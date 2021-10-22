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

#ifndef MFEM_TENSOR_OPERATORS
#define MFEM_TENSOR_OPERATORS

/// All the contractions for the tensors
#include "contractions/contractions.hpp"
/// Interpolation operators at quadrature points, ex: B * u
#include "interpolations/interpolations.hpp"
/// Gradients operators at quadratire points, ex: grad(B) * u
#include "gradients/gradients.hpp"
/// Divergence operators at quadrature points, ex: div(B) * u
#include "divergence/divergence.hpp"
/// Curl operators at quadrature points, ex: curl(B) * u
#include "curl/curl.hpp"
/// Point-wise multiplications at quadrature points
#include "point-wise_multiplications/point-wise_multiplications.hpp"
/// Determinant operators for matrices
#include "determinant.hpp"

#endif // MFEM_TENSOR_OPERATORS
