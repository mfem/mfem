// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "mfem.hpp"
#include "catch.hpp"

#ifdef MFEM_USE_EXCEPTIONS

using namespace mfem;

TEST_CASE("Operator", "[Operator]")
{
   // Define diagonal sparse matrix
   Vector diag(5);
   diag = 12.34;

   SparseMatrix A(diag);

   CGSolver cg;
   cg.SetOperator(A);

   SECTION("ProductNotIterative")
   {
      // When cg is in a product (on the right), we require cg->iterative_mode to be false.

      ProductOperator *product = NULL;
      RAPOperator *rap = NULL;
      TripleProductOperator *triple = NULL;

      // First, test that the failing version throws an exception.
      cg.iterative_mode = true;
      REQUIRE_THROWS(product = new ProductOperator(&A, &cg, false, false));
      REQUIRE_THROWS(triple = new TripleProductOperator(&A, &cg, &cg, false, false,
                                                        false));
      REQUIRE_THROWS(rap = new RAPOperator(A, cg, cg));

      // Second, test that the correct version does not throw.
      cg.iterative_mode = false;
      REQUIRE_NOTHROW(product = new ProductOperator(&A, &cg, false, false));
      REQUIRE_NOTHROW(triple = new TripleProductOperator(&A, &cg, &cg, false, false,
                                                         false));
      REQUIRE_NOTHROW(rap = new RAPOperator(A, cg, cg));
   }
}

#endif  // MFEM_USE_EXCEPTIONS
