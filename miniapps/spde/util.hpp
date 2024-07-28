// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details

#ifndef SPDE_UTIL_HPP
#define SPDE_UTIL_HPP

#include <random>
#include <vector>
#include "mfem.hpp"

namespace mfem
{

/// Fills the vector x with random numbers between a and b.
void FillWithRandomNumbers(std::vector<real_t> &x, real_t a = 0.0,
                           real_t b = 1.0);

/// This function creates random rotation matrices (3 x 3) and stores them in
/// the vector. That means, x[0-8] is the first rotation matrix, x[9-17] is
/// the second and so forth. Size of the vector determines the number of
/// rotation that fit into the vector and should be a multiple of 9.
void FillWithRandomRotations(std::vector<real_t> &x);

}  // namespace mfem

#endif  // SPDE_UTIL_HPP
