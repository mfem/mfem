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

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

#ifdef MFEM_USE_EXCEPTIONS

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

      // First, test that the failing version throws an exception.
      cg.iterative_mode = true;
      REQUIRE_THROWS(ProductOperator(&A, &cg, false, false));
      REQUIRE_THROWS(TripleProductOperator(&A, &cg, &cg, false, false,
                                           false));
      REQUIRE_THROWS(RAPOperator(A, cg, cg));

      // Second, test that the correct version does not throw.
      cg.iterative_mode = false;
      REQUIRE_NOTHROW(ProductOperator(&A, &cg, false, false));
      REQUIRE_NOTHROW(TripleProductOperator(&A, &cg, &cg, false, false,
                                            false));
      REQUIRE_NOTHROW(RAPOperator(A, cg, cg));
   }
}

#endif  // MFEM_USE_EXCEPTIONS

double constrained_mult_application(Operator &op, Array<int> &list,
                                    const Vector &input, const Vector &truth, const bool transpose = false,
                                    const Operator::DiagonalPolicy diag_policy = Operator::DiagonalPolicy::DIAG_ONE)
{
   const ConstrainedOperator constrained_op(&op, list, false, diag_policy);
   // Make sure test is well formed.
   CHECK(op.Width() == input.Size());
   CHECK(op.Height() == truth.Size());
   Vector y(op.Height());
   if (transpose)
   {
      constrained_op.MultTranspose(input,y);
   }
   else
   {
      constrained_op.Mult(input,y);
   }
   auto error = truth;
   error -= y;
   auto error_norm = error.Norml2() / truth.Norml2();
   return error_norm;
}

TEST_CASE("ConstrainedOperator", "[ConstrainedOperator][Operator]")
{
   INFO("Constrained Operator");
   // Compare against manual calculation with random 5x5 matrix and input vector.
   // Should leave first and fourth entries the same for DIAG_ONE, and zero them
   // out for DIAG_ZERO.
   DenseMatrix A(
   {
      {27.531558467881045, 89.30012682807859, 10.363408976942745, 78.97400291889993, 18.703638414621903 },
      {79.33627624921924, 73.99743336818197, 85.27832370283267, 11.13213120570734, 27.59542336254316},
      {26.474414966916925, 17.38636366801234, 41.423691595967114, 94.06135498225382, 18.379018138899884},
      {45.83203742468528, 90.10126513894627, 3.8488872448446343, 41.03858238887901, 14.429143614063412},
      {26.2225932381016, 3.8232081630501513, 17.820832452264256, 3.919068726019015, 92.66801110040682}
   });
   Array<int> list(2);
   list[0] = 0;
   list[1] = 3;
   Vector      x({62.06906909143156, 63.31143800813616, 59.6546764326512, 48.10287136113324, 0.4275152133050852});
   // DIAG_ONE checks
   Vector y_true({62.06906909143156, 9783.932185967293, 3579.7299142176153, 48.10287136113324, 1344.7657848396123});
   Vector y_true_transpose({62.06906909143156, 5723.696294059853, 7877.828900340113, 48.10287136113324, 2883.1173002839714});
   REQUIRE(constrained_mult_application(A, list, x, y_true) == MFEM_Approx(0.0));
   REQUIRE(constrained_mult_application(A, list, x, y_true_transpose,
                                        true) == MFEM_Approx(0.0));
   // DIAG_ZERO checks
   Vector y_true_zero({0.0, 9783.932185967293, 3579.7299142176153, 0.0, 1344.7657848396123});
   Vector y_true_zero_transpose({0.0, 5723.696294059853, 7877.828900340113, 0.0, 2883.1173002839714});
   REQUIRE(constrained_mult_application(A, list, x, y_true_zero, false,
                                        Operator::DiagonalPolicy::DIAG_ZERO) == MFEM_Approx(0.0));
   REQUIRE(constrained_mult_application(A, list, x, y_true_zero_transpose, true,
                                        Operator::DiagonalPolicy::DIAG_ZERO) == MFEM_Approx(0.0));
}
