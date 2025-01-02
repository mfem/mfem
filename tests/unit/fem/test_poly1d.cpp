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
#include "catch.hpp"

using namespace mfem;
/*
TEST_CASE("1D Jacobi Polynomials (alpha=0, beta=0)","[Poly_1D]")
{
   // For alpha = beta = 0 the Jacobi polynomials should equal the Legendre
   // polynomials
   int p = 6;
   int order = 2 * p + 1;

   const IntegrationRule &ir = IntRules.Get(Geometry::SEGMENT, order);

   DenseMatrix J(p+1, ir.GetNPoints());
   DenseMatrix L(p+1, ir.GetNPoints());

   for (int i=0; i<ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      poly1d.CalcJacobi(p, 0.0, 0.0, ip.x, J.GetColumn(i));
      poly1d.CalcLegendre(p, ip.x, L.GetColumn(i));
   }

   J.Add(-1.0, L);

   double max_norm = J.MaxMaxNorm();
   REQUIRE(max_norm < 1e-11);
}

TEST_CASE("1D Jacobi Polynomials (alpha=beta=-0.5)","[Poly_1D]")
{
   // For alpha = beta = -1/2 the Jacobi polynomials should equal the Chebyshev
   // polynomials
   int p = 6;
   int order = 2 * p + 1;

   const IntegrationRule &ir = IntRules.Get(Geometry::SEGMENT, order);

   DenseMatrix J(p+1, ir.GetNPoints());
   DenseMatrix C(p+1, ir.GetNPoints());

   for (int i=0; i<ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      poly1d.CalcJacobi(p, -0.5, -0.5, ip.x, J.GetColumn(i));
      poly1d.CalcBasis(p, ip.x, C.GetColumn(i));
   }

   // Jacobi polynomials are normalized so that P^(a,b)_n(1) = binom(n+a,n)
   // whereas Chebyshev satisfy T_n(1) = 1.
   Vector w(p+1);
   w(0) = 1.0;
   w(1) = 2.0;
   w(2) = 8.0 / 3.0;
   w(3) = 16.0 / 5.0;
   w(4) = 128.0 / 35.0;
   w(5) = 256.0 / 63.0;
   w(6) = 1024.0 / 231.0;

   // Rescale Jacobi polynomials to match Chebyshev convention
   J.LeftScaling(w);

   J.Add(-1.0, C);

   // Measure error
   double max_norm = J.MaxMaxNorm();
   REQUIRE(max_norm < 1e-11);
}

TEST_CASE("1D Jacobi Polynomials (alpha=2, beta=0)","[Poly_1D]")
{
   // Check orthogonality of Jacobi polynomials up to order 6
   int p = 6;
   int order = 2 * p + 3;
   double alpha = 2.0;

   const IntegrationRule &ir = IntRules.Get(Geometry::SEGMENT, order);

   DenseMatrix A(p+1, ir.GetNPoints());
   Vector w(ir.GetNPoints());

   // Compute polynomial values and weight function
   for (int i=0; i<ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      poly1d.CalcJacobi(p, alpha, 0.0, ip.x, A.GetColumn(i));
      w(i) = pow(2.0 - 2.0 * ip.x, alpha) * ip.weight;
   }

   // Compute integrals
   DenseMatrix ADAt(p+1);
   MultADAt(A, w, ADAt);

   // Remove expected diagonal entries
   for (int i=0; i<=p; i++)
   {
      ADAt(i,i) -= 4.0 / (2.0 * i + 3.0);
   }

   // Measure error
   double max_norm = ADAt.MaxMaxNorm();
   REQUIRE(max_norm < 1e-11);
}
*/
TEST_CASE("1D Legendre Polynomials","[Poly_1D]")
{
   // Check orthogonality of Legendre polynomials up to order 6
   int p = 6;
   int order = 2 * p + 1;

   const IntegrationRule &ir = IntRules.Get(Geometry::SEGMENT, order);

   DenseMatrix A(p+1, ir.GetNPoints());
   Vector w(ir.GetNPoints());

   // Compute polynomial values and weight function
   for (int i=0; i<ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      poly1d.CalcLegendre(p, ip.x, A.GetColumn(i));
      w(i) = ip.weight;
   }

   // Compute integrals
   DenseMatrix ADAt(p+1);
   MultADAt(A, w, ADAt);

   // Remove expected diagonal entries
   for (int i=0; i<=p; i++)
   {
      ADAt(i,i) -= 1.0 / (2.0 * i + 1.0);
   }

   // Measure error
   double max_norm = ADAt.MaxMaxNorm();
   REQUIRE(max_norm < 1e-11);
}
