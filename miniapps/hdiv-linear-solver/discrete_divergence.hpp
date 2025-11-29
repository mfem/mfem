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

#ifndef MFEM_DISCRETE_DIVERGENCE_HPP
#define MFEM_DISCRETE_DIVERGENCE_HPP

#include "mfem.hpp"

namespace mfem
{

/// @brief Eliminates columns in the given HypreParMatrix.
///
/// This is similar to HypreParMatrix::EliminateBC, except that only the columns
/// are eliminated.
void EliminateColumns(HypreParMatrix &D, const Array<int> &ess_dofs);

HypreParMatrix *FormDiscreteDivergenceMatrix(ParFiniteElementSpace &fes_rt,
                                             ParFiniteElementSpace &fes_l2,
                                             const Array<int> &ess_dofs);

} // namespace mfem

#endif
