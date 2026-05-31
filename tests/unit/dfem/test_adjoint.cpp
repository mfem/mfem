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
#include "../../../linalg/tensor_arrays.hpp"

#ifdef MFEM_USE_MPI

// ────────────────────────────────────────────────────────────────────────────
#ifdef MFEM_USE_ENZYME
using dreal_t = real_t;
#else
using dreal_t = dual<real_t, real_t>;
#endif

// ────────────────────────────────────────────────────────────────────────────
static constexpr int U = 0, V = 1, 𝚵 = 2;

// ────────────────────────────────────────────────────────────────────────────
template <int DIM> struct global_qf
{
   void operator()(
      tensor_array<const dreal_t> &x,
      tensor_array<const dreal_t> &y,
      tensor_array<const real_t, DIM, DIM> &J,
      tensor_array<const real_t> &w,
      tensor_array<dreal_t> &z) const
   {
      for (size_t q = 0; q < x.size(); q++)
      {
         const dreal_t xq = x(q), yq = y(q);
         const real_t wq = w(q);
         z(q) = sin(xq) * cos(yq) * (xq + yq) * wq * det(J(q));
      }
   }
};

// ────────────────────────────────────────────────────────────────────────────
template <int DIM> struct local_qf
{
   inline MFEM_HOST_DEVICE
   void operator()(const dreal_t &x,
                   const dreal_t &y,
                   const tensor<real_t, DIM, DIM> &J,
                   const real_t &w,
                   dreal_t &z) const
   {
      z = sin(x) * cos(y) * (x + y) * w * det(J);
   }
};

// ────────────────────────────────────────────────────────────────────────────
// Shared forward-tangent / adjoint consistency check on an operator whose
// domain integrator has already been added. Verifies
//   <bar_z, J·δ> == <Jᵀ·bar_z, δ>
// for the derivatives with respect to U and V.
template <int DIM>
static void RunTangentAdjointConsistency(DifferentiableOperator &F,
                                         ParFiniteElementSpace &fes,
                                         ParGridFunction &nodes)
{
   F.SetMultLevel(DifferentiableOperator::MultLevel::LVECTOR);

   // Forward pass state
   ParGridFunction x(&fes), y(&fes), z(&fes);
   x.Randomize(0x9e3779b9);
   y.Randomize(0x9e3779b1);
   z = M_PI;
   MultiVector state{x, y, nodes}, mz{z};

   // Forward tangent (sensitivities δz)
   ParGridFunction δx(&fes), δy(&fes), δz(&fes), δu(&fes), δv(&fes);
   δx.Randomize(0x01000193);
   δy.Randomize(0x1b873593);
   δz = 0.0;

   auto dFu = F.GetDerivative(U, state);
   auto dFv = F.GetDerivative(V, state);

   MultiVector mδx{δx}, mδy{δy}, mδu{δu}, mδv{δv};

   dFu->Mult(δx, mδu); // δu = ∂F/∂x * δx
   dFv->Mult(δy, mδv); // δv = ∂F/∂y * δy
   add(δu, δv, δz);

   // Adjoint pass (MultTranspose from seed)
   ParGridFunction bar_x(&fes), bar_y(&fes), bar_z(&fes);
   bar_z.Randomize(0x7ed55d16);

   MultiVector bar_mx{bar_x}, bar_my{bar_y}, bar_mz{bar_z};
   dFu->MultTranspose(bar_mz, bar_mx); // \bar{x} = (∂F/∂x)^T * \bar{z}
   dFv->MultTranspose(bar_mz, bar_my); // \bar{y} = (∂F/∂y)^T * \bar{z}

   // Tangent-Adjoint consistency
   const real_t left = InnerProduct(bar_z, δz);
   const real_t right = InnerProduct(bar_x, δx) + InnerProduct(bar_y, δy);
   REQUIRE(left == MFEM_Approx(right));
}

// ────────────────────────────────────────────────────────────────────────────
template<int DIM, int Q1D, typename QT, typename IT, typename OT>
inline void AddLocalSpecializations()
{
   AddActionLO<DIM, Q1D, QT, IT, OT>();

   AddDerivativeActionLO<DIM, Q1D, U, QT, IT, OT>();
   AddDerivativeSetupLO<DIM, Q1D, U, QT, IT, OT>();
   AddDerivativeApplyLO<DIM, Q1D, U, QT, IT, OT>();
   AddDerivativeApplyTransposeLO<DIM, Q1D, U, QT, IT, OT>();

   AddDerivativeActionLO<DIM, Q1D, V, QT, IT, OT>();
   AddDerivativeSetupLO<DIM, Q1D, V, QT, IT, OT>();
   AddDerivativeApplyLO<DIM, Q1D, V, QT, IT, OT>();
   AddDerivativeApplyTransposeLO<DIM, Q1D, V, QT, IT, OT>();
}

// ────────────────────────────────────────────────────────────────────────────
// Tangent-adjoint consistency test
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

   using vfds_t = std::vector<FieldDescriptor>;
   const vfds_t in_fds = { {U, &fes}, {V, &fes}, {𝚵, nfes} };
   const vfds_t out_fds = { {U, &fes} };
   DifferentiableOperator F(in_fds, out_fds, pmesh);

   using ITS = Inputs<Value<U>, Value<V>, Gradient<𝚵>, Weight>;
   using OTS = Outputs<Value<U>>;
   using DID = Derivatives<U, V>;

   global_qf<DIM> q_gfn {};
   F.AddDomainIntegrator<GlobalQFBackend>(
      q_gfn, ITS {}, OTS {}, *ir, all_domain_attr, DID {});

   local_qf<DIM> q_lfn {};
   AddLocalSpecializations<DIM, 3, local_qf<DIM>, ITS, OTS>();
   F.AddDomainIntegrator<LocalQFBackend>(
      q_lfn, ITS {}, OTS {}, *ir, all_domain_attr, DID {});

   RunTangentAdjointConsistency<DIM>(F, fes, *nodes);
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
