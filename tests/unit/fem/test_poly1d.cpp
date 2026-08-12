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
#include "catch.hpp"

using namespace mfem;

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
