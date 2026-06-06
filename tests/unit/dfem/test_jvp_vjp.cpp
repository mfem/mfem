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

#include "../../../fem/dfem/doperator.hpp"
#include "../../../linalg/tensor_arrays.hpp"

using namespace mfem;
using namespace mfem::future;

#ifdef MFEM_USE_ENZYME
using dreal_t = real_t;
#else
using dreal_t = dual<real_t, real_t>;
#endif

static constexpr int U = 0, V = 1, 𝚵 = 2;

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

   const auto dFu = F.GetDerivative(U, state);
   const auto dFv = F.GetDerivative(V, state);
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
   const vfds_t in_fds = { { U, &fes }, { V, &fes }, { 𝚵, nfes } };
   const vfds_t out_fds = { { U, &fes } };
   DifferentiableOperator F(in_fds, out_fds, pmesh);

   using IT = Inputs<Value<U>, Value<V>, Gradient<𝚵>, Weight>;
   using OT = Outputs<Value<U>>;
   using DT = Derivatives<U, V>;

   if constexpr (!mfem_use_gpu)
   {
      global_qf<DIM> q_gfn{};
      F.AddDomainIntegrator<GlobalQFBackend>(
         q_gfn, IT{}, OT{}, *ir, all_domain_attr, DT{});
   }

   using LQT = local_qf<DIM>;
   local_qf<DIM> q_lfn{};
   F.AddDomainIntegrator<LocalQFBackend>(
      q_lfn, IT{}, OT{}, *ir, all_domain_attr, DT{});
   AddLocalSpecializations<DIM, 3, LQT, IT, OT, DT>();

   VerifyJvpVjp<DIM>(F, fes, *nodes);
}

// ────────────────────────────────────────────────────────────────────────────
TEST_CASE("dFEM JVP-VJP 2D", "[Parallel][dFEM][GPU][JVP][2D]")
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
TEST_CASE("dFEM JVP-VJP 3D", "[Parallel][dFEM][GPU][JVP][3D]")
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

#endif // MFEM_USE_MPI
