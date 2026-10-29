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

#ifdef _WIN32
#define _USE_MATH_DEFINES
#include <cmath>
#endif

#include "unit_tests.hpp"
#include "mfem.hpp"

#include "fem/integ/bilininteg_vecdiv_pa.hpp" // IWYU pragma: keep
#include "fem/qinterp/grad.hpp" // IWYU pragma: keep

#include <algorithm>
#include <utility>

using namespace mfem;

namespace pa_kernels
{

template <typename INTEGRATOR, bool TRANSPOSE>
void pa_mixed_test(FiniteElementSpace &fes1,
                   FiniteElementSpace &fes2,
                   const IntegrationRule &ir)
{
   MixedBilinearForm bform_pa(&fes1, &fes2);
   if constexpr (TRANSPOSE)
   {
      auto *integ = new TransposeIntegrator(new INTEGRATOR);
      integ->SetIntRule(&ir);
      bform_pa.AddDomainIntegrator(integ);
   }
   else
   {
      auto *integ = new INTEGRATOR;
      integ->SetIntRule(&ir);
      bform_pa.AddDomainIntegrator(integ);
   }
   bform_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   bform_pa.Assemble();

   MixedBilinearForm bform_fa(&fes1, &fes2);
   if constexpr (TRANSPOSE)
   {
      auto *integ = new TransposeIntegrator(new INTEGRATOR);
      integ->SetIntRule(&ir);
      bform_fa.AddDomainIntegrator(integ);
   }
   else
   {
      auto *integ = new INTEGRATOR;
      integ->SetIntRule(&ir);
      bform_fa.AddDomainIntegrator(integ);
   }
   bform_fa.Assemble();
   bform_fa.Finalize();

   GridFunction x(&fes1), y_pa(&fes2), y_fa(&fes2);
   x.Randomize(0x100001b3);

   bform_pa.Mult(x, y_pa);
   bform_fa.Mult(x, y_fa);

   y_pa -= y_fa;
   REQUIRE(y_pa.Normlinf() == MFEM_Approx(0.0));
}

template<int DIM>
void test_pa_divergence(const char *filename, int vp, int sp)
{
   CAPTURE(filename, DIM, vp, sp);

   Mesh mesh(filename);
   MFEM_VERIFY(mesh.Dimension() == DIM, "Mesh dimension mismatch");

   // Vector
   H1_FECollection vfec(vp, DIM);
   FiniteElementSpace vfes(&mesh, &vfec, DIM);

   // Scalar
   H1_FECollection sfec(sp, DIM);
   FiniteElementSpace sfes(&mesh, &sfec);

   // Shared-memory PA kernels require q1d >= max(trial_d1d, test_d1d)
   const auto &trial_fe = *vfes.GetTypicalFE();
   const auto &test_fe = *sfes.GetTypicalFE();
   const auto &Trans = *mesh.GetTypicalElementTransformation();
   int order = Trans.OrderGrad(&trial_fe) + test_fe.GetOrder() + Trans.OrderJ();
   const int min_q1d = std::max(vp, sp) + 1;
   order = std::max(order, 2 * min_q1d - 1);
   const IntegrationRule &ir = IntRules.Get(trial_fe.GetGeomType(), order);

   pa_mixed_test<VectorDivergenceIntegrator, false>(vfes, sfes, ir);
   pa_mixed_test<VectorDivergenceIntegrator, true>(sfes, vfes, ir);
}

TEST_CASE("VecDivPA", "[PartialAssembly][VecDivPA][GPU]")
{
   if (static auto done = false; !std::exchange(done, true))
   {
      using Grad = QuadratureInterpolator::GradKernels;
      Grad::Specialization<2, QVectorLayout::byNODES, false, 2, 3, 5>::Add();
      Grad::Specialization<3, QVectorLayout::byNODES, false, 3, 3, 7>::Add();
      Grad::Specialization<3, QVectorLayout::byNODES, false, 3, 4, 9>::Add();

      using VDiv = VectorDivergenceIntegrator::VectorDivergenceAddMultPA;
      using VDivT = VectorDivergenceIntegrator::VectorDivergenceAddMultTransposePA;
      VDiv::Specialization<2, 2, 3, 3>::Add();
      VDivT::Specialization<2, 2, 3, 3>::Add();
   }

   // Vector (vp) and scalar (sp) space orders
   const auto vp_base = {1, 2}, vp_extra = {3, 4};
   const auto sp_base = {1, 2}, sp_extra = {3, 4};
   const auto vp = MFEM_GENERATE_RANGES(vp_base, vp_extra);
   const auto sp = MFEM_GENERATE_RANGES(sp_base, sp_extra);

   SECTION("2D")
   {
      const auto meshs = { "../../data/inline-quad.mesh" };
      const auto extra = { "../../data/star-q2.mesh",
                           "../../data/star-q3.mesh",
                           "../../data/rt-2d-q3.mesh",
                           "../../data/periodic-square.mesh"
                         };
      test_pa_divergence<2>(MFEM_GENERATE_RANGES(meshs, extra), vp, sp);
   }

   SECTION("3D")
   {
      const auto meshs = { "../../data/inline-hex.mesh" };
      const auto extra = { "../../data/fichera.mesh",
                           "../../data/beam-hex.mesh",
                           "../../data/toroid-hex.mesh",
                           "../../data/fichera-q2.mesh",
                           "../../data/fichera-q3.mesh",
                           "../../data/periodic-cube.mesh"
                         };
      test_pa_divergence<3>(MFEM_GENERATE_RANGES(meshs, extra), vp, sp);
   }
}

} // namespace pa_kernels
