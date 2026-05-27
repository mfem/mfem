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

#include "../unit_tests.hpp" // IWYU pragma: keep

#include "mfem.hpp"
using mfem::real_t;

using namespace mfem;
using namespace mfem::future;

#include "../../../fem/dfem/doperator.hpp"

#ifdef NVTX_DBG_FMT
#include NVTX_DBG_FMT // IWYU pragma: keep
#endif

#ifdef MFEM_USE_MPI

// ────────────────────────────────────────────────────────────────────────────
#ifdef MFEM_USE_ENZYME
using dreal_t = mfem::real_t;
#else
using dreal_t = mfem::future::dual<real_t, real_t>;
#endif

using vfds_t = std::vector<FieldDescriptor>;

template<int DIM>
using dvecd_t = mfem::future::tensor<dreal_t, DIM>;

template<int DIM>
using matd_t = mfem::future::tensor<real_t, DIM, DIM>;

// ────────────────────────────────────────────────────────────────────────────
static constexpr int U = 0, V = 1, 𝚵 = 2;
constexpr auto U_V_derivatives = std::make_index_sequence<2> {};

// ────────────────────────────────────────────────────────────────────────────
template <int DIM>
void TangentAdjointConsistencyTest(const char *filename, int p)
{
   CAPTURE(filename, DIM, p);
   dbg("{} {} {}", filename, DIM, p);

   Mesh smesh(filename);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   MFEM_VERIFY(pmesh.Dimension() == DIM, "Mesh dimension mismatch");

   pmesh.EnsureNodes();
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   smesh.Clear();

   p = std::max(p, pmesh.GetNodalFESpace()->GetMaxElementOrder());

   Array<int> all_domain_attr;
   if (pmesh.attributes.Size() > 0)
   {
      all_domain_attr.SetSize(pmesh.attributes.Max());
      all_domain_attr = 1;
   }

   H1_FECollection fec(p, DIM);
   ParFiniteElementSpace fes(&pmesh, &fec);
   const auto *nfes = nodes->ParFESpace();

   const auto geom = pmesh.GetTypicalElementGeometry();
   const auto ir = &IntRules.Get(geom, 2 * p + 1);

   // ───────────────────────────────────────────────────────────────
   // Setup differentiable operator
   const vfds_t in_fds = { {U, &fes}, {V, &fes}, {𝚵, nfes} };
   const vfds_t out_fds = { {U, &fes} };
   DifferentiableOperator F(in_fds, out_fds, pmesh);
   const auto q_fn = [] (const dreal_t &x,
                         const dreal_t &y,
                         const tensor<real_t, DIM, DIM> &J,
                         const real_t &w,
                         dreal_t &z)
   {
      z = (x * y + sin(x)*cos(y)) * w * det(J);
   };
   F.AddDomainIntegrator<LocalQFBackend>(
      q_fn,
      tuple{ Value<U>{}, Value<V>{}, Gradient<𝚵>{}, Weight{}},
      tuple{ Value<U>{} },
      *ir, all_domain_attr,
      U_V_derivatives);
   F.SetMultLevel(DifferentiableOperator::MultLevel::LVECTOR);

   // ───────────────────────────────────────────────────────────────
   // Forward pass
   ParGridFunction x(&fes), y(&fes), z(&fes);
   x.Randomize(0x9e3779b9);
   y.Randomize(0x9e3779b1);
   z = M_PI;

   MultiVector state{x, y, *nodes}, mz{z};
   // F.Mult(state, mz);

   // ───────────────────────────────────────────────────────────────
   // Forward tangent (sensitivities δz)
   ParGridFunction δx(&fes), δy(&fes), δz(&fes), δw(&fes);
   δx.Randomize(0x01000193);
   δy.Randomize(0x1b873593);
   δz = 0.0;

   auto ∂Fu = F.GetDerivative(U, state);
   auto ∂Fv = F.GetDerivative(V, state);

   MultiVector mδx{δx}, mδy{δy}, mδz{δz}, mδw{δw};

   // δz = ∂F/∂x * δx
   ∂Fu->Mult(δx, mδz);

   // δz += ∂F/∂y * δy
   ∂Fv->Mult(δy, mδw);
   δz += δw;

   // ───────────────────────────────────────────────────────────────
   // Adjoint pass (MultTranspose from seed)
   ParGridFunction bar_x(&fes), bar_y(&fes), bar_z(&fes);
   bar_z.Randomize(0x7ed55d16);

   MultiVector bar_mx{bar_x}, bar_my{bar_y}, bar_mz{bar_z};
   ∂Fu->MultTranspose(bar_mz, bar_mx); // \bar{x} = (∂F/∂x)^T * \bar{z}
   ∂Fv->MultTranspose(bar_mz, bar_my); // \bar{y} = (∂F/∂y)^T * \bar{z}

   // ───────────────────────────────────────────────────────────────
   // Tangent-Adjoint Consistency Test
   const real_t left = bar_z * δz;
   const real_t right = (bar_x * δx) + (bar_y * δy);
   REQUIRE(left == MFEM_Approx(right));
}

TEST_CASE("dFEM Tangent-Adjoint Consistency 2D",
          "[Parallel][dFEM][GPU][ADJOINT][2D]")
{
   const auto all_tests = launch_all_non_regression_tests;
   const auto p = !all_tests ? 1 : GENERATE(1, 2, 3);
   const auto mesh =
      GENERATE("../../data/star.mesh",
               "../../data/star-q3.mesh",
               "../../data/rt-2d-q3.mesh",
               "../../data/inline-quad.mesh",
               "../../data/periodic-square.mesh");
   TangentAdjointConsistencyTest<2>(mesh, p);
}

TEST_CASE("dFEM Tangent-Adjoint Consistency 3D",
          "[Parallel][dFEM][GPU][ADJOINT][3D]")
{
   const auto all_tests = launch_all_non_regression_tests;
   const auto p = !all_tests ? 1 : GENERATE(1, 2, 3);
   const auto mesh =
      GENERATE("../../data/fichera.mesh",
               "../../data/fichera-q3.mesh",
               "../../data/inline-hex.mesh",
               "../../data/toroid-hex.mesh",
               "../../data/periodic-cube.mesh");
   TangentAdjointConsistencyTest<3>(mesh, p);
}

#endif // MFEM_USE_MPI
