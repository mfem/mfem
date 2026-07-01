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

#include "fem/qinterp/grad.hpp" // IWYU pragma: keep

#include <utility>

using namespace mfem;

namespace pa_kernels
{

template <typename INTEGRATOR, bool TRANSPOSE>
void pa_mixed_test(FiniteElementSpace &fes1,
                   FiniteElementSpace &fes2)
{
   MixedBilinearForm bform_pa(&fes1, &fes2);
   if constexpr (TRANSPOSE)
   {
      bform_pa.AddDomainIntegrator(new TransposeIntegrator(new INTEGRATOR));
   }
   else
   {
      bform_pa.AddDomainIntegrator(new INTEGRATOR);
   }
   bform_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   bform_pa.Assemble();

   MixedBilinearForm bform_fa(&fes1, &fes2);
   if constexpr (TRANSPOSE)
   {
      bform_fa.AddDomainIntegrator(new TransposeIntegrator(new INTEGRATOR));
   }
   else
   {
      bform_fa.AddDomainIntegrator(new INTEGRATOR);
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
void test_pa_divergence(const char *filename, int p)
{
   CAPTURE(filename, DIM, p);

   Mesh mesh(filename);
   MFEM_VERIFY(mesh.Dimension() == DIM, "Mesh dimension mismatch");

   // Vector
   H1_FECollection vfec(p, DIM);
   FiniteElementSpace vfes(&mesh, &vfec, DIM);

   // Scalar
   H1_FECollection sfec(p, DIM);
   FiniteElementSpace sfes(&mesh, &sfec);

   pa_mixed_test<VectorDivergenceIntegrator, false>(vfes, sfes);
   pa_mixed_test<VectorDivergenceIntegrator, true>(sfes, vfes);
}

TEST_CASE("VecDivPA", "[PartialAssembly][VecDivPA][GPU]")
{
   if (static auto done = false; !std::exchange(done, true))
   {
      using Grad = QuadratureInterpolator::GradKernels;
      Grad::Specialization<2, QVectorLayout::byNODES, false, 2, 3, 5>::Add();
      Grad::Specialization<3, QVectorLayout::byNODES, false, 3, 3, 7>::Add();
      Grad::Specialization<3, QVectorLayout::byNODES, false, 3, 4, 9>::Add();
   }

   SECTION("2D")
   {
      const auto meshs = { "../../data/inline-quad.mesh" };
      const auto extra = { "../../data/star-q2.mesh",
                           "../../data/star-q3.mesh",
                           "../../data/rt-2d-q3.mesh",
                           "../../data/periodic-square.mesh"
                         };
      test_pa_divergence<2>(GenAll(meshs, extra), GenAll({1, 2}, {3, 4}));
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
      test_pa_divergence<3>(GenAll(meshs, extra), GenAll({1, 2}, {3, 4}));
   }
}

} // namespace pa_kernels
