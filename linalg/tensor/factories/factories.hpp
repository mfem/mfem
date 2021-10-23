// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_FACTORIES
#define MFEM_FACTORIES

/**
 * Factories are functions creating different types of objects according to
 * their inputs. The following factories represent higher level mathematical
 * concepts using tensors as their main data structure representation.
 * */

/// Factories to build objects representing basis functions
#include "basis/basis.hpp"
/// Factories to build objects representing degrees of freedom
#include "degrees_of_freedom/degrees_of_freedom.hpp"
/// Factories to build objects representing quadrature data
#include "qdata/qdata.hpp"
/// Factories to build objects representing symmetric quadrature data
#include "qdata/symm_qdata.hpp"
/// Factories to build diagonal tensors
#include "diagonal_tensor.hpp"
/// Factories to build symmetric diagonal tensors
#include "diagonal_symm_tensor.hpp"

#endif // MFEM_FACTORIES
