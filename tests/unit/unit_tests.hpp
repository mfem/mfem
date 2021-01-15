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

#ifndef MFEM_UNIT_TEST
#define MFEM_UNIT_TEST

#include "catch.hpp"

/// MFEM_Approx can be used to compare floating point values within an absolute
/// tolerance of `margin` (default value 1e-12).
inline Approx MFEM_Approx(double val, double margin = 1e-12)
{
   return Approx(val).margin(margin);
}

#endif
