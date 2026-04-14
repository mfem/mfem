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

using namespace mfem;

namespace pa_kernels
{

template <int DIM>
void test_nl_convection_pa_grad(const char *filename, int p)
{
   CAPTURE(filename, DIM, p);
   // dbg("filename: {}, DIM: {}, p: {}", filename, DIM, p);

   Mesh mesh(filename);
   MFEM_VERIFY(mesh.Dimension() == DIM, "Mesh dimension mismatch");

   H1_FECollection fec(p, DIM);

   // ⚠️ PA only supports Ordering::byNODES
   constexpr int ordering = mfem::Ordering::byNODES;
   FiniteElementSpace vfes(&mesh, &fec, DIM, ordering);

   GridFunction x(&vfes), dx(&vfes), y_fa(&vfes), y_pa(&vfes);
   x.Randomize(0x100001b3);
   dx.Randomize(0x9e3779b9);

   // ⚠️ only ConstantCoefficient is supported
   // const auto rho = [](const Vector &xyz)
   // {
   //    const real_t x = xyz(0), y = xyz(1), z = DIM == 3 ? xyz(2) : 0.0;
   //    real_t r = M_PI * pow(x, 2);
   //    if (DIM >= 2) { r += pow(y, 3); }
   //    if (DIM >= 3) { r += pow(z, 4); }
   //    return r;
   // };
   // FunctionCoefficient rho_fc(rho);
   ConstantCoefficient rho_cc(1.0/*M_PI*/);

   NonlinearForm nlf_fa(&vfes);
   nlf_fa.AddDomainIntegrator(new VectorConvectionNLFIntegrator(rho_cc));

   NonlinearForm nlf_pa(&vfes);
   nlf_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   nlf_pa.AddDomainIntegrator(new VectorConvectionNLFIntegrator(rho_cc));
   nlf_pa.Setup();

   //SECTION("Action")
   {
      nlf_fa.Mult(x, y_fa), nlf_pa.Mult(x, y_pa);
      y_fa -= y_pa;
      REQUIRE(y_fa.Norml2() == MFEM_Approx(0.0));
   }

   // SECTION("Gradient")
   {
      Operator &nlf_fa_grad = nlf_fa.GetGradient(x);
      Operator &nlf_pa_grad = nlf_pa.GetGradient(x);
      nlf_pa_grad.Mult(dx, y_pa);
      nlf_fa_grad.Mult(dx, y_fa);
      y_fa -= y_pa;
      // dbg("y_fa.Norml2(): {}", y_fa.Norml2());
      REQUIRE(y_fa.Norml2() == MFEM_Approx(0.0));
   }
}

TEST_CASE("NL Convection PA Gradient",
          "[PartialAssembly][NonlinearPA][GPU][CHOP]")
{
   dbg("NL Convection PA Gradient");
   const bool all_tests = launch_all_non_regression_tests;
   const auto p = !all_tests ? 2 : GENERATE(1, 2, 3, 4);
   SECTION("2D")
   {
      const auto filename2d =
         all_tests ?
         GENERATE(
            // "../../data/star-q3.mesh",
            // "../../data/rt-2d-q3.mesh",
            "../../data/inline-quad.mesh",
            "../../data/periodic-square.mesh",
            "../../data/star-q2.mesh"
         )
         :
         GENERATE("../../data/inline-quad.mesh",
                  "../../data/periodic-square.mesh")
         ;
      test_nl_convection_pa_grad<2>(filename2d, p);
   }
   // SECTION("3D") { test_nl_convection_pa(3); }
}

} // namespace pa_kernels
