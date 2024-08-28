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

TEST_CASE("Change of Basis", "[ChangeOfBasis][CUDA]")
{
   const auto mesh_fname = GENERATE(
                              "../../data/star.mesh",
                              "../../data/star-q3.mesh",
                              "../../data/fichera.mesh",
                              "../../data/fichera-q3.mesh"
                           );
   const int order = GENERATE(1, 2, 3);
   const int btype1 = GENERATE(BasisType::GaussLobatto,
                               BasisType::ClosedUniform,
                               BasisType::Positive);
   const int btype2 = GENERATE(BasisType::GaussLobatto,
                               BasisType::ClosedUniform,
                               BasisType::Positive);

   if (btype1 == btype2) { return; }

   CAPTURE(mesh_fname, order, btype1, btype2);

   Mesh mesh = Mesh::LoadFromFile(mesh_fname);
   H1_FECollection fec1(order, mesh.Dimension(), btype1);
   H1_FECollection fec2(order, mesh.Dimension(), btype2);
   FiniteElementSpace fes1(&mesh, &fec1);
   FiniteElementSpace fes2(&mesh, &fec2);

   DiscreteLinearOperator op1(&fes1, &fes2);
   op1.AddDomainInterpolator(new IdentityInterpolator);
   op1.Assemble();

   ChangeOfBasis op2(fes1, btype2);

   GridFunction x1(&fes1), x2(&fes1), y1(&fes2), y2(&fes2);
   x1.Randomize(1);

   op1.Mult(x1, y1);
   op2.Mult(x1, y2);
   op2.MultInverse(y2, x2);

   y2 -= y1;
   x2 -= x1;

   REQUIRE(y2.Normlinf() == MFEM_Approx(0.0));
   REQUIRE(x2.Normlinf() == MFEM_Approx(0.0));
}

TEST_CASE("Change of Basis Legendre", "[ChangeOfBasis][CUDA]")
{
   const auto mesh_fname = GENERATE(
                              "../../data/star.mesh",
                              "../../data/star-q3.mesh",
                              "../../data/fichera.mesh",
                              "../../data/fichera-q3.mesh"
                           );
   const int order = GENERATE(1, 2, 3);
   const int btype = GENERATE(BasisType::GaussLobatto,
                              BasisType::GaussLegendre,
                              BasisType::ClosedUniform,
                              BasisType::Positive);

   CAPTURE(mesh_fname, order, btype);

   Mesh mesh = Mesh::LoadFromFile(mesh_fname);
   L2_FECollection fec(order, mesh.Dimension(), btype);
   FiniteElementSpace fes(&mesh, &fec);

   ChangeOfBasis op(fes, ChangeOfBasis::LEGENDRE);

   GridFunction x1(&fes), x2(&fes), y(&fes);
   x1.Randomize(1);
   op.Mult(x1, y);
   op.MultInverse(y, x2);

   x2 -= x1;

   REQUIRE(x2.Normlinf() == MFEM_Approx(0.0));
}
