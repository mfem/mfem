// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "unit_tests.hpp"
#include "mfem.hpp"

using namespace mfem;

TEST_CASE("Hyperbolic Face Gradient", "[HyperbolicFormIntegrator]")
{
   Mesh mesh = Mesh::MakeCartesian1D(2);
   L2_FECollection fec(1, 1);
   FiniteElementSpace fes(&mesh, &fec);

   VectorConstantCoefficient b(Vector({1.0}));
   AdvectionFlux flux(b);
   RusanovFlux rusanov(flux);

   NonlinearForm N(&fes);
   N.AddInteriorFaceIntegrator(new HyperbolicFormIntegrator(rusanov, 0));

   const int n = fes.GetVSize();
   Vector u(n);
   u.Randomize(1);
   auto &J = static_cast<SparseMatrix&>(N.GetGradient(u));

   // Compare the assembled gradient with centered finite difference
   const real_t h = 1e-6;
   real_t error = 0.0;
   for (int j = 0; j < n; j++)
   {
      Vector up(u), um(u), rp(n), rm(n), e(n), Jcol(n);
      up(j) += h;
      um(j) -= h;
      N.Mult(up, rp);
      N.Mult(um, rm);
      e = 0.0;
      e(j) = 1.0;
      J.Mult(e, Jcol);
      for (int i = 0; i < n; i++)
      {
         error = std::max(error, std::abs(Jcol(i) - (rp(i) - rm(i))/(2*h)));
      }
   }
   REQUIRE(error == MFEM_Approx(0.0, 1e-6));
}
