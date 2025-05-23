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

#include <utility>

#include "mfem.hpp"
#include "unit_tests.hpp"

#ifdef MFEM_USE_MPI

#include "fem/dfem/doperator.hpp"
#include "linalg/tensor.hpp"

#if defined(__has_include) && __has_include("general/nvtx.hpp") && !defined(_WIN32)
#undef NVTX_COLOR
#define NVTX_COLOR ::nvtx::kGold
#include "general/nvtx.hpp"
#else
#define dbg(...)
#endif

using namespace mfem;
using mfem::internal::tensor;
using results_t = std::array<real_t, 3>;

namespace dfem_derivative_assembly
{

template <int DIM>
void DFemDerivativeAssembly(const char *filename, int p,
                            const results_t& expected_fnorms)
{
   using vecd_t = tensor<real_t, DIM>;
   using matd_t = tensor<real_t, DIM, DIM>;

   CAPTURE(filename, DIM, p);

   Mesh smesh(filename);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   MFEM_VERIFY(pmesh.Dimension() == DIM, "Mesh dimension mismatch");

   pmesh.SetCurvature(p);
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   MFEM_VERIFY(nodes, "Mesh does not have nodes");
   p = std::max(p, pmesh.GetNodalFESpace()->GetMaxElementOrder());
   smesh.Clear();

   Array<int> all_domain_attr;
   if (pmesh.bdr_attributes.Size() > 0)
   {
      all_domain_attr.SetSize(pmesh.bdr_attributes.Max());
      all_domain_attr = 1;
   }

   H1_FECollection fec(p, DIM);
   ParFiniteElementSpace pfes(&pmesh, &fec);
   ParFiniteElementSpace *mfes = nodes->ParFESpace();
   MFEM_VERIFY(DIM * pfes.GetVSize() == nodes->Size(), "nodes size mismatch");
   Vector x(*nodes, 0, pfes.GetVSize());
   const auto *ir = &IntRules.Get(pmesh.GetTypicalElementGeometry(), 2 * p + 1);

   static constexpr int U = 0, Coords = 1;
   DifferentiableOperator dop_mf({{ U, &pfes }}, {{Coords, mfes}}, pmesh);
   auto mf_apply_qf = [](const vecd_t &dudxi, const matd_t &J, const real_t &w)
   {
      return mfem::tuple{ (dudxi * inv(J)) * transpose(inv(J)) * det(J) * w };
   };
   dop_mf.AddDomainIntegrator(mf_apply_qf,
                              mfem::tuple{ Gradient<U>{}, Gradient<Coords>{}, Weight{} },
                              mfem::tuple{ Gradient<U>{} },
                              *ir, all_domain_attr,
                              std::integer_sequence<size_t, Coords> {});
   HypreParMatrix A;
   auto dop_mf_derivative = dop_mf.GetDerivative(Coords, {&x}, {nodes});
   dop_mf_derivative->Assemble(A);
   const real_t fnorm = A.FNorm();

   const auto expected_fnorm = expected_fnorms[static_cast<size_t>(p)-1];
   dbg("A FNorm: {} ({})", fnorm, expected_fnorm);
   constexpr auto tol = 1e-15;
   REQUIRE(fabs(fnorm - expected_fnorm) == MFEM_Approx(0.0, tol, tol));
}

TEST_CASE("DFEM Derivative Assembly", "[Parallel][DFEM][Assembly]")
{
   dbg();
   const auto all_tests = launch_all_non_regression_tests;
   const auto p = !all_tests ? 2 : GENERATE(1, 2, 3);
   static_assert(std::tuple_size<results_t>::value == 3);

   SECTION("2D p=" + std::to_string(p))
   {
      auto [filename, expected_fnorms] =
         GENERATE(table<std::string, results_t>(
      {
         {
            "../../data/star.mesh",
            {24.987721237546996, 146.52903344877006, 539.651561004031}
         },
         {
            "../../data/star-q3.mesh",
            {24.95956386897864, 150.755043841781, 736.2018345106584}
         },
         {
            "../../data/inline-quad.mesh",
            {15.270705433752719, 110.0016890554948, 419.94593064913283}
         },
         {
            "../../data/periodic-square.mesh",
            {6.733003290855952, 70.85200869436, 6656.219250863574}
         }
      }));
      DFemDerivativeAssembly<2>(filename.c_str(), p, expected_fnorms);
   }
}

} // namespace dfem_derivative_assembly

#endif // MFEM_USE_MPI