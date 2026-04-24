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

#include "unit_tests.hpp"
#include "mfem.hpp"

#include <fem/qinterp/grad.hpp>

using namespace mfem;

namespace pa_kernels
{

template<int DIM>
void test_nl_convection_pa_grad(const char *filename, int p)
{
   CAPTURE(filename, DIM, p);

   Mesh mesh(filename);
   MFEM_VERIFY(mesh.Dimension() == DIM, "Mesh dimension mismatch");
   dbg("filename: {}, DIM: {}, p: {}", filename, DIM, p);

   H1_FECollection fec(p, DIM);
   FiniteElementSpace fes(&mesh, &fec, DIM);

   GridFunction x(&fes), dx(&fes), y_fa(&fes), y_pa(&fes);
   x.Randomize(0x100001b3);
   dx.Randomize(0x9e3779b9);

   ConstantCoefficient const_coeff(M_PI);
   FunctionCoefficient funct_coeff([](const Vector &x)
   { return M_1_PI + x[0] * x[0]; });

   NonlinearForm nlf_fa(&fes);
   nlf_fa.AddDomainIntegrator(new VectorConvectionNLFIntegrator);
   nlf_fa.AddDomainIntegrator(new VectorConvectionNLFIntegrator(const_coeff));
   nlf_fa.AddDomainIntegrator(new VectorConvectionNLFIntegrator(funct_coeff));

   NonlinearForm nlf_pa(&fes);
   nlf_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   nlf_pa.AddDomainIntegrator(new VectorConvectionNLFIntegrator);
   nlf_pa.AddDomainIntegrator(new VectorConvectionNLFIntegrator(const_coeff));
   nlf_pa.AddDomainIntegrator(new VectorConvectionNLFIntegrator(funct_coeff));
   nlf_pa.Setup();

   SECTION("Action")
   {
      nlf_fa.Mult(x, y_fa), nlf_pa.Mult(x, y_pa);
      y_fa -= y_pa;
      REQUIRE(y_fa.Norml2() == MFEM_Approx(0.0));
   }

   SECTION("Gradient")
   {
      Operator &nlf_fa_grad = nlf_fa.GetGradient(x);
      Operator &nlf_pa_grad = nlf_pa.GetGradient(x);
      nlf_pa_grad.Mult(dx, y_pa);
      nlf_fa_grad.Mult(dx, y_fa);
      y_fa -= y_pa;
      REQUIRE(y_fa.Norml2() == MFEM_Approx(0.0));
   }

   SECTION("Diagonal")
   {
      Vector diag_fa(fes.GetVSize()), diag_pa(fes.GetVSize());
      dynamic_cast<SparseMatrix &>(nlf_fa.GetGradient(x)).GetDiag(diag_fa);
      nlf_pa.GetGradient(x).AssembleDiagonal(diag_pa);
      diag_fa -= diag_pa;
      REQUIRE(diag_fa.Norml2() == MFEM_Approx(0.0));
   }
}

TEST_CASE("NL Convection PA Gradient",
          "[PartialAssembly][NonlinearPA][GPU][NLConv]")
{
   const bool all_tests = launch_all_non_regression_tests;
   const auto p = !all_tests ? GENERATE(1, 2) : GENERATE(2, 3, 4);

   if (static auto done = false; !std::exchange(done, true))
   {
      using Grad = QuadratureInterpolator::GradKernels;
      Grad::Specialization<2, QVectorLayout::byNODES, false, 2, 2, 7>::Add();
      Grad::Specialization<2, QVectorLayout::byNODES, false, 2, 3, 7>::Add();
      Grad::Specialization<2, QVectorLayout::byNODES, false, 2, 4, 8>::Add();
      Grad::Specialization<3, QVectorLayout::byNODES, false, 3, 2, 7>::Add();
      Grad::Specialization<3, QVectorLayout::byNODES, false, 3, 3, 7>::Add();
      Grad::Specialization<3, QVectorLayout::byNODES, false, 3, 3, 8>::Add();
      Grad::Specialization<3, QVectorLayout::byNODES, false, 3, 4, 9>::Add();
   }

   SECTION("2D")
   {
      const auto filename2d = all_tests
                                 ? GENERATE("../../data/star-q2.mesh",
                                            "../../data/star-q3.mesh",
                                            "../../data/rt-2d-q3.mesh",
                                            "../../data/inline-quad.mesh",
                                            "../../data/periodic-square.mesh")
                                 : GENERATE("../../data/inline-quad.mesh",
                                            "../../data/periodic-square.mesh");
      test_nl_convection_pa_grad<2>(filename2d, p);
   }

   SECTION("3D")
   {
      const auto filename3d = all_tests
                                 ? GENERATE("../../data/beam-hex.mesh",
                                            "../../data/fichera.mesh",
                                            "../../data/fichera-q2.mesh",
                                            "../../data/fichera-q3.mesh",
                                            "../../data/inline-hex.mesh",
                                            "../../data/periodic-cube.mesh",
                                            "../../data/toroid-hex.mesh")
                                 : GENERATE("../../data/inline-hex.mesh",
                                            "../../data/periodic-cube.mesh");
      test_nl_convection_pa_grad<3>(filename3d, p);
   }
}

} // namespace pa_kernels
