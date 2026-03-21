// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mfem.hpp"
#include "unit_tests.hpp"

#ifdef MFEM_USE_ENZYME

using namespace mfem::future;

tensor<double, 3, 3> shift(
    const tensor<double, 3, 3>& X, const tensor<double, 3, 3>& p)
{
  return X + p;
}

tensor<double, 3, 3> sym_and_shift(const tensor<double, 3, 3>& X, 
                                   const tensor<double, 3, 3>& p)
{
  return sym(X) - p;
}

TEST_CASE("Enzyme derivatives of tensors", "[tensor]")
{
    tensor<double, 3, 3> x{{{ 1.0, -1.0,  0.0},
                            {-1.0,  2.0, -1.0},
                            { 0.0, -1.0,  1.0}}};
    tensor p = 3*IdentityMatrix<3>();
    tensor<double, 3, 3> x_dot{{{1.0, 0.0, 0.0},
                                {0.0, 0.0, 0.0},
                                {0.0, 0.0, 0.0}}};

    SECTION("Tensor function with constant parameter")
    {
        auto y_dot = __enzyme_fwddiff<tensor<double, 3, 3>>((void*)shift, enzyme_dup, x, x_dot, enzyme_const, p);

        // correct answer is simply y_dot = x_dot
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                CHECK(y_dot[i][j] == MFEM_Approx(x_dot[i][j]));        
            }
        }
    }
}

#endif // MFEM_USE_ENZYME
