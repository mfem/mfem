// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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
#ifndef _WIN32

#include "../../../fem/dfem/doperator.hpp"
#include "../../../linalg/tensor_arrays.hpp"

using namespace mfem;
using namespace mfem::future;

#ifdef MFEM_USE_ENZYME
using dreal_t = real_t;
#else
using dreal_t = dual<real_t, real_t>;
#endif

static constexpr int U = 0, V = 1, X = 2;

// ────────────────────────────────────────────────────────────────────────────
template<int DIM>
struct global_qf
{
   void operator()(tensor_array<const dreal_t> &x,
                   tensor_array<const dreal_t> &y,
                   tensor_array<const real_t, DIM, DIM> &J,
                   tensor_array<const real_t> &w,
                   tensor_array<dreal_t> &z) const
   {
      mfem::forall(x.size(),
                   [=] MFEM_HOST_DEVICE(int q)
      {
         const dreal_t xq = x(q), yq = y(q);
         z(q) = sin(xq) * cos(yq) * (xq + yq) * w(q) * det(J(q));
      });
   }
};

// ────────────────────────────────────────────────────────────────────────────
template<int DIM>
struct local_qf
{
   inline MFEM_HOST_DEVICE void operator()(const dreal_t &x,
                                           const dreal_t &y,
                                           const tensor<real_t, DIM, DIM> &J,
                                           const real_t &w,
                                           dreal_t &z) const
   { z = sin(x) * cos(y) * (x + y) * w * det(J); }
};

// ────────────────────────────────────────────────────────────────────────────
template<int DIM>
static void VerifyJvpVjp(DifferentiableOperator &F,
                         ParFiniteElementSpace &fes,
                         ParGridFunction &nodes)
{
   const auto nfes = nodes.ParFESpace();
   const auto tvsize = fes.GetTrueVSize();
   const auto ntvsize = nfes->GetTrueVSize();
   const auto comm = fes.GetParMesh()->GetComm();

   Vector X_bar(tvsize), Y_bar(tvsize), N_bar(ntvsize);

   X_bar.Randomize(0x9e3779b9);
   Y_bar.Randomize(0x9e3779b1);
   nodes.GetTrueDofs(N_bar);

   MultiVector state{ X_bar, Y_bar, N_bar };

   Vector dX(tvsize), dY(tvsize), dZ(tvsize);
   dX.Randomize(0x01000193);
   dY.Randomize(0x1b873593);

   Vector dU(tvsize), dV(tvsize);
   MultiVector mdU{ dU }, mdV{ dV };

   const auto dFu = F.GetDerivative(U, state, true);
   const auto dFv = F.GetDerivative(V, state, true);
   dFu->Mult(dX, mdU); // dU = (∂F/∂u) dX
   dFv->Mult(dY, mdV); // dV = (∂F/∂v) dY
   add(dU, dV, dZ);

   Vector dX_star(tvsize), dY_star(tvsize), dZ_star(tvsize);
   dZ_star.Randomize(0x7ed55d16);

   MultiVector mdZ_star{ dZ_star }, mdX_star{ dX_star }, mdY_star{ dY_star };
   dFu->MultTranspose(mdZ_star, mdX_star); // dX* = (∂F/∂u)^T dZ*
   dFv->MultTranspose(mdZ_star, mdY_star); // dY* = (∂F/∂v)^T dZ*

   // Tangent/cotangent consistency test:
   // <dZ*, dZ> = <dX*, dX> + <dY*, dY>
   REQUIRE(InnerProduct(comm, dZ_star, dZ) ==
           MFEM_Approx(InnerProduct(comm, dX_star, dX) +
                       InnerProduct(comm, dY_star, dY)));

   REQUIRE(InnerProduct(comm, dZ_star, dU) ==
           MFEM_Approx(InnerProduct(comm, dX_star, dX)));

   REQUIRE(InnerProduct(comm, dZ_star, dV) ==
           MFEM_Approx(InnerProduct(comm, dY_star, dY)));
}

// ────────────────────────────────────────────────────────────────────────────
template<int DIM>
void TestJvpVjp(const char *filename, int p)
{
   CAPTURE(filename, DIM, p);

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
   const vfds_t in_fds = { { U, &fes }, { V, &fes }, { X, nfes } };
   const vfds_t out_fds = { { U, &fes } };
   DifferentiableOperator F(in_fds, out_fds, pmesh);

   using IT = Inputs<Value<U>, Value<V>, Gradient<X>, Weight>;
   using OT = Outputs<Value<U>>;
   using DT = Derivatives<U, V>;

   constexpr auto kernels =
      DerivativeKernels::Apply |
      DerivativeKernels::ApplyTranspose;

#ifndef MFEM_USE_CUDA_OR_HIP
   {
      global_qf<DIM> q_gfn{};
      F.AddDomainIntegrator<GlobalQFBackend, kernels>(
         q_gfn, IT{}, OT{}, *ir, all_domain_attr, DT{});
   }
#endif

   using LQT = local_qf<DIM>;
   local_qf<DIM> q_lfn{};
   F.AddDomainIntegrator<LocalQFBackend, kernels>(
      q_lfn, IT{}, OT{}, *ir, all_domain_attr, DT{});
   AddLocalSpecializations<DIM, 3, LQT, IT, OT, Entity::Element, DT,
                           kernels>();

   VerifyJvpVjp<DIM>(F, fes, *nodes);
}

// ────────────────────────────────────────────────────────────────────────────
TEST_CASE("dFEM JVP-VJP 2D", "[Parallel][dFEM][GPU]")
{
   const auto p = GenAll({ 1 }, { 2, 3 });
   const auto meshs = { "../../data/inline-quad.mesh" };
   const auto extra = { "../../data/star.mesh",
                        "../../data/star-q3.mesh",
                        "../../data/rt-2d-q3.mesh",
                        "../../data/periodic-square.mesh"
                      };
   TestJvpVjp<2>(GenAll(meshs, extra), p);
}

// ────────────────────────────────────────────────────────────────────────────
TEST_CASE("dFEM JVP-VJP 3D", "[Parallel][dFEM][GPU]")
{
   const auto p = GenAll({ 1 }, { 2, 3 });
   const auto meshs = { "../../data/inline-hex.mesh" };
   const auto extra = { "../../data/fichera.mesh",
                        "../../data/fichera-q3.mesh",
                        "../../data/toroid-hex.mesh",
                        "../../data/periodic-cube.mesh"
                      };
   TestJvpVjp<3>(GenAll(meshs, extra), p);
}

// ────────────────────────────────────────────────────────────────────────────
template<int DIM>
struct cubic_qf
{
   inline MFEM_HOST_DEVICE void operator()(const dreal_t &u,
                                           const tensor<real_t, DIM, DIM> &J,
                                           const real_t &w,
                                           dreal_t &v) const
   { v = u * u * u * w * det(J); }
};

// Two derivatives with respect to the same field at different states must not
// affect each other, whatever the order in which they are used.
TEST_CASE("dFEM derivative cache per state", "[Parallel][dFEM]")
{
   Mesh smesh("../../data/inline-quad.mesh");
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   pmesh.EnsureNodes();
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   smesh.Clear();

   Array<int> attr(pmesh.attributes.Max());
   attr = 1;
   H1_FECollection fec(2, 2);
   ParFiniteElementSpace fes(&pmesh, &fec);
   const auto &ir = IntRules.Get(pmesh.GetTypicalElementGeometry(), 6);
   const int n = fes.GetTrueVSize();

   DifferentiableOperator F({ { U, &fes }, { X, nodes->ParFESpace() } },
                            { { U, &fes } }, pmesh);
   constexpr auto kernels = DerivativeKernels::Action |
                            DerivativeKernels::Apply |
                            DerivativeKernels::ApplyTranspose |
                            DerivativeKernels::AssembleMatrix;
   cubic_qf<2> qf;
   F.AddDomainIntegrator<LocalQFBackend, kernels>(
      qf, Inputs<Value<U>, Gradient<X>, Weight> {}, Outputs<Value<U>> {}, ir,
      attr, Derivatives<U> {});

   Vector N, X1(n), X2(n), dir(n), ref(n), y1(n), y2(n), y3(n);
   nodes->GetTrueDofs(N);
   X1.Randomize(1);
   X2.Randomize(2);
   X2 += 1.0;
   dir.Randomize(3);
   MultiVector S1{ X1, N }, S2{ X2, N }, mdir{ dir }, mref{ ref };
   MultiVector m1{ y1 }, m2{ y2 }, m3{ y3 };

   // Reference at S1 from the direct action, which uses no cache.
   F.GetDerivative(U, S1, false)->Mult(dir, mref);

   const auto comm = pmesh.GetComm();
   auto max_diff = [&](const Vector &a, const Vector &b)
   {
      Vector d(a);
      d -= b;
      return GlobalLpNorm(infinity(), d.Normlinf(), comm);
   };

   // Apply the derivative at S1, then the one at S2, then the one at S1 again.
   auto interleave = [&](auto apply)
   {
      auto d1 = F.GetDerivative(U, S1, true), d2 = F.GetDerivative(U, S2, true);
      apply(*d1, m1);
      apply(*d2, m2);
      apply(*d1, m3);
      REQUIRE(max_diff(y1, ref) == MFEM_Approx(0.0));
      REQUIRE(max_diff(y2, ref) > 1e-6); // the two states differ
      REQUIRE(max_diff(y3, ref) == MFEM_Approx(0.0));
   };

   SECTION("Mult")
   {
      interleave([&](DerivativeOperator &d, MultiVector &y)
      { d.Mult(dir, y); });
   }

   // The derivative is symmetric, so the reference applies here too.
   SECTION("MultTranspose")
   {
      interleave([&](DerivativeOperator &d, MultiVector &y)
      { d.MultTranspose(mdir, y); });
   }

   SECTION("Assemble")
   {
      auto d1 = F.GetDerivative(U, S1, false);
      auto d2 = F.GetDerivative(U, S2, false);
      SparseMatrix *A1 = nullptr, *A2 = nullptr, *A3 = nullptr;
      d1->Assemble(A1);
      d2->Assemble(A2);
      d1->Assemble(A3);
      std::unique_ptr<SparseMatrix> D12(Add(1.0, *A1, -1.0, *A2));
      std::unique_ptr<SparseMatrix> D13(Add(1.0, *A1, -1.0, *A3));
      REQUIRE(D12->MaxNorm() > 1e-6);
      REQUIRE(D13->MaxNorm() == MFEM_Approx(0.0));
      delete A1;
      delete A2;
      delete A3;
   }
}

#endif // _WIN32
#endif // MFEM_USE_MPI
