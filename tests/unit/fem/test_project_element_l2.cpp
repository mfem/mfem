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

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

namespace project_element_l2
{

TEST_CASE("ELEMENT_L2 ProjectCoefficient H1 Tetrahedra",
          "[GridFunction]")
{
   const int order = GENERATE(1, 2, 3, 4);
   CAPTURE(order);

   // Two tetrahedra sharing a face. The integrals of the cubic H1 edge basis
   // functions vanish on each element.
   Mesh mesh(3, 5, 2);
   mesh.AddVertex(0.0, 0.0, 0.0);
   mesh.AddVertex(1.0, 0.0, 0.0);
   mesh.AddVertex(0.0, 1.0, 0.0);
   mesh.AddVertex(0.0, 0.0, 1.0);
   mesh.AddVertex(0.0, 0.0, -0.5);
   mesh.AddTet(0, 1, 2, 3);
   mesh.AddTet(2, 1, 0, 4);
   mesh.FinalizeTetMesh(1, 0, true);

   H1_FECollection fec(order, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);

   // The element-local L2 projections reproduce functions in the space
   FunctionCoefficient coeff([](const Vector &x)
   {
      return 1.0 + 2.0*x(0) + 3.0*x(1) + 4.0*x(2);
   });

   GridFunction gf(&fes);
   gf.ProjectCoefficient(coeff, ProjectType::ELEMENT_L2);

   for (int i = 0; i < gf.Size(); i++)
   {
      REQUIRE(std::isfinite(gf(i)));
   }
   REQUIRE(gf.ComputeL2Error(coeff) == MFEM_Approx(0.0));
}

} // namespace project_element_l2
