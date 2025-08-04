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

#include "../unit_tests.hpp"
#include "mfem.hpp"

#ifdef MFEM_USE_MPI

using namespace mfem;
using namespace mfem::future;
using mfem::future::tensor;

namespace dfem_pa_kernels
{

template <int DIM>
void dfem_functional(const char *filename, int p, const int r)
{
   CAPTURE(filename, DIM, p, r);

   Mesh smesh(filename);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   pmesh.EnsureNodes();
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   p = std::max(p, pmesh.GetNodalFESpace()->GetMaxElementOrder());
   smesh.Clear();

   Array<int> all_domain_attr;
   if (pmesh.attributes.Size() > 0)
   {
      all_domain_attr.SetSize(pmesh.attributes.Max());
      all_domain_attr = 1;
   }

   H1_FECollection fec(p, DIM);
   ParFiniteElementSpace fes(&pmesh, &fec);
   const auto *ir = &IntRules.Get(pmesh.GetTypicalElementGeometry(), 2 * p + r);

   ParGridFunction x(&fes), y(&fes), z(&fes);
   Vector X(fes.GetTrueVSize()), Y(fes.GetTrueVSize()), Z(fes.GetTrueVSize());

   X = 1.0;
   x.SetFromTrueDofs(X);

   ConstantCoefficient one(1.0);
   ParBilinearForm blf(&fes);
   blf.AddDomainIntegrator(new MassIntegrator(one, ir));
   blf.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf.Assemble();
   blf.Mult(x, y);
   fes.GetProlongationMatrix()->MultTranspose(y, Y);

   real_t sum_g, sum_l = Y.Sum();
   MPI_Allreduce(&sum_l, &sum_g, 1, MPI_DOUBLE, MPI_SUM, pmesh.GetComm());

   static constexpr int U = 0, Coords = 1;
   const auto sol = std::vector{ FieldDescriptor{ U, &fes } };
   DifferentiableOperator dop(sol, {{Coords, nodes->ParFESpace()}}, pmesh);

   const auto functional_qf =
      [] MFEM_HOST_DEVICE(
#ifdef MFEM_USE_ENZYME
         const real_t &u,
#else
         const dual<real_t, real_t> &u,
#endif
         const tensor<real_t, DIM, DIM> &J,
         const real_t &w)
   {
      return tuple{u * w * det(J)};
   };

   auto derivatives = std::integer_sequence<size_t, U> {};
   dop.AddDomainIntegrator(functional_qf,
                           tuple{ Value<U>{}, Gradient<Coords>{}, Weight{} },
                           tuple{ Sum<U>{} },
                           *ir, all_domain_attr, derivatives);
   dop.SetParameters({ nodes });

   fes.GetRestrictionMatrix()->Mult(x, X);
   Vector sum(1);
   dop.Mult(X, sum);

   REQUIRE(MFEM_Approx(0.0) == std::abs(sum_g - sum(0)));

   auto dRdu = dop.GetDerivative(U, {&x}, {nodes});
   Vector d(1);
   dRdu->Mult(X, d);
   REQUIRE(MFEM_Approx(0.0) == std::abs(d(0) - sum(0)));

   MPI_Barrier(MPI_COMM_WORLD);
}

TEST_CASE("DFEM Functional", "[Parallel][DFEM][Functional]")
{
   const bool all_tests = launch_all_non_regression_tests;

   const auto p = !all_tests ? 2 : GENERATE(1, 2, 3);
   const auto r = !all_tests ? 1 : GENERATE(0, 1, 2, 3);

   SECTION("2D p=" + std::to_string(p) + " r=" + std::to_string(r))
   {
      const auto filename =
         GENERATE("../../data/star.mesh",
                  "../../data/star-q3.mesh",
                  "../../data/rt-2d-q3.mesh",
                  "../../data/inline-quad.mesh",
                  "../../data/periodic-square.mesh");
      dfem_functional<2>(filename, p, r);
   }

   SECTION("3D p=" + std::to_string(p) + " r=" + std::to_string(r))
   {
      const auto filename =
         GENERATE("../../data/fichera.mesh",
                  "../../data/fichera-q3.mesh",
                  "../../data/inline-hex.mesh",
                  "../../data/toroid-hex.mesh",
                  "../../data/periodic-cube.mesh");
      dfem_functional<3>(filename, p, r);
   }
}

} // namespace dfem_pa_kernels

#endif // MFEM_USE_MPI
