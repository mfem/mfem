// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information on source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

namespace mixed_pa_basis
{

real_t scalarInput(const Vector &x)
{
   return x(0);
}

real_t constantInput(const Vector &)
{
   return 1.0;
}

void vectorInput(const Vector &x, Vector &v)
{
   v = 0.0;
   v(0) = x(1);
}

void constantVectorInput(const Vector &, Vector &v)
{
   v = 0.0;
   v(0) = 1.0;
}

real_t maxAbsDifference(const Vector &x, const Vector &y)
{
   Vector diff(x);
   diff -= y;
   return diff.Normlinf();
}

struct GradientResult
{
   real_t forward_error;
   real_t transpose_error;
};

GradientResult compareMixedVectorGradient(int trial_basis, int test_basis,
                                          bool constant)
{
   Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL,
                                     true, 1.0, 1.0);
   H1_FECollection h1_fec(3, 2, trial_basis);
   ND_FECollection nd_fec(3, 2, test_basis, BasisType::GaussLegendre);
   FiniteElementSpace trial_fes(&mesh, &h1_fec);
   FiniteElementSpace test_fes(&mesh, &nd_fec);

   ConstantCoefficient coeff(1.0);
   const IntegrationRule &ir = IntRules.Get(Geometry::SQUARE, 10);

   MixedBilinearForm fa(&trial_fes, &test_fes);
   MixedBilinearForm pa(&trial_fes, &test_fes);
   pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);

   auto *fa_integ = new MixedVectorGradientIntegrator(coeff);
   auto *pa_integ = new MixedVectorGradientIntegrator(coeff);
   fa_integ->SetIntRule(&ir);
   pa_integ->SetIntRule(&ir);
   fa.AddDomainIntegrator(fa_integ);
   pa.AddDomainIntegrator(pa_integ);
   fa.Assemble();
   pa.Assemble();

   FunctionCoefficient input(constant ? constantInput : scalarInput);
   GridFunction x(&trial_fes);
   x.ProjectCoefficient(input);

   Vector y_fa(test_fes.GetVSize());
   Vector y_pa(test_fes.GetVSize());
   fa.Mult(x, y_fa);
   pa.Mult(x, y_pa);
   const real_t forward_error = maxAbsDifference(y_fa, y_pa);

   GridFunction z(&test_fes);
   z.Randomize(1234);
   Vector xt_fa(trial_fes.GetVSize());
   Vector xt_pa(trial_fes.GetVSize());
   fa.MultTranspose(z, xt_fa);
   pa.MultTranspose(z, xt_pa);
   const real_t transpose_error = maxAbsDifference(xt_fa, xt_pa);

   return {forward_error, transpose_error};
}

real_t compareMixedVectorWeakCurl(int trial_basis, int test_basis,
                                  bool constant)
{
   Mesh mesh = Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON,
                                     1.0, 1.0, 1.0);
   ND_FECollection trial_fec(3, 3, trial_basis, BasisType::GaussLegendre);
   ND_FECollection test_fec(3, 3, test_basis, BasisType::GaussLegendre);
   FiniteElementSpace trial_fes(&mesh, &trial_fec);
   FiniteElementSpace test_fes(&mesh, &test_fec);

   ConstantCoefficient coeff(1.0);
   const IntegrationRule &ir = IntRules.Get(Geometry::CUBE, 10);

   MixedBilinearForm fa(&trial_fes, &test_fes);
   MixedBilinearForm pa(&trial_fes, &test_fes);
   pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);

   auto *fa_integ = new MixedVectorWeakCurlIntegrator(coeff);
   auto *pa_integ = new MixedVectorWeakCurlIntegrator(coeff);
   fa_integ->SetIntRule(&ir);
   pa_integ->SetIntRule(&ir);
   fa.AddDomainIntegrator(fa_integ);
   pa.AddDomainIntegrator(pa_integ);
   fa.Assemble();
   pa.Assemble();

   VectorFunctionCoefficient input(
      3, constant ? constantVectorInput : vectorInput);
   GridFunction x(&trial_fes);
   x.ProjectCoefficient(input);

   Vector y_fa(test_fes.GetVSize());
   Vector y_pa(test_fes.GetVSize());
   fa.Mult(x, y_fa);
   pa.Mult(x, y_pa);
   return maxAbsDifference(y_fa, y_pa);
}

TEST_CASE("MixedVectorGradient PA preserves trial closed basis",
          "[MixedVectorGradientIntegrator][PartialAssembly]")
{
   const int trial_basis = GENERATE(BasisType::GaussLobatto,
                                    BasisType::ClosedUniform);
   const int test_basis = GENERATE(BasisType::GaussLobatto,
                                   BasisType::ClosedUniform);
   const bool constant = GENERATE(false, true);
   CAPTURE(trial_basis, test_basis, constant);

   const GradientResult error =
      compareMixedVectorGradient(trial_basis, test_basis, constant);
   CHECK(error.forward_error <= 1e-11);
   CHECK(error.transpose_error <= 1e-11);
}

TEST_CASE("MixedVectorWeakCurl PA preserves trial closed basis",
          "[MixedVectorWeakCurlIntegrator][PartialAssembly]")
{
   const int trial_basis = GENERATE(BasisType::GaussLobatto,
                                    BasisType::ClosedUniform);
   const int test_basis = GENERATE(BasisType::GaussLobatto,
                                   BasisType::ClosedUniform);
   const bool constant = GENERATE(false, true);
   CAPTURE(trial_basis, test_basis, constant);

   CHECK(compareMixedVectorWeakCurl(trial_basis, test_basis, constant)
         <= 1e-11);
}

} // namespace mixed_pa_basis
