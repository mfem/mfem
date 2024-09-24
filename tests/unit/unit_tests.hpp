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

#ifndef MFEM_UNIT_TEST
#define MFEM_UNIT_TEST

#include "catch.hpp"

/// Command line '--all' option to launch all non-regression tests.
extern bool launch_all_non_regression_tests;

/// Command line '--data' argument for path to mfem/data repo.
/** If no --data path is provided, then mfem_data_dir will be the empty string,
    and tests tagged with [MFEMData] will be skipped. */
extern std::string mfem_data_dir;

/// Command line '--verbose' argument, false by default.
/** Enables extra log messages and the mfem::out stream. */
extern bool verbose_tests;

/** @brief MFEM_Approx can be used to compare floating point values within an
    absolute tolerance of @a abs_tol (default value 1e-12) and relative
    tolerance of @a rel_tol (default value 1e-12). */
inline Approx MFEM_Approx(double val,
                          double abs_tol = 1e-12,
                          double rel_tol = 1e-12)
{
   return Approx(val).margin(abs_tol).epsilon(rel_tol);
}

#endif
