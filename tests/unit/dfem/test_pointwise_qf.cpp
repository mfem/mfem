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

#include "../../../fem/dfem/doperator.hpp"
#include "../linalg/test_same_matrices.hpp"

using namespace mfem;
using namespace mfem::future;

namespace pointwise_qf
{

using dscalar_t =
   std::conditional_t<mfem::UseEnzyme, real_t,
   mfem::future::dual<real_t, real_t>>;

constexpr int DIM = 2;
constexpr int U = 0, COORDINATES = 1;

// Nonlinear mass plus scaled diffusion, the same pointwise q-function runs on
// both backends.
struct MassDiffusionQF
{
   real_t kappa;

   MFEM_HOST_DEVICE inline
   void operator()(const dscalar_t &u,
                   const tensor<dscalar_t, DIM> &dudxi,
                   const tensor<real_t, DIM, DIM> &J,
                   const real_t &w,
                   dscalar_t &mass,
                   tensor<dscalar_t, DIM> &diffusion) const
   {
      const auto invJ = inv(J);
      const real_t detJw = det(J) * w;
      mass = u * u * detJw;
      diffusion = (dudxi * invJ) * transpose(invJ) * (kappa * detJw);
   }
};

static_assert(detail::is_local_qfunc_v<MassDiffusionQF>);

template <typename backend_t>
std::unique_ptr<DifferentiableOperator> MakeOperator(
   ParFiniteElementSpace &fes, ParMesh &pmesh, const IntegrationRule &ir,
   const Array<int> &attributes)
{
   const std::vector<FieldDescriptor> in_fds
   {
      {U, &fes},
      {COORDINATES, pmesh.GetNodes()->FESpace()}
   };
   const std::vector<FieldDescriptor> out_fds { {U, &fes} };
   auto dop = std::make_unique<DifferentiableOperator>(in_fds, out_fds, pmesh);
   MassDiffusionQF qfunc{2.0};
   dop->AddDomainIntegrator<backend_t>(
      qfunc,
      Inputs<Value<U>, Gradient<U>, Gradient<COORDINATES>, Weight> {},
      Outputs<Value<U>, Gradient<U>> {},
      ir, attributes, Derivatives<U> {});
   return dop;
}

real_t MaxDiff(const Vector &a, const Vector &b)
{
   const real_t *ah = a.HostRead(), *bh = b.HostRead();
   real_t local = 0.0, global = 0.0;
   for (int i = 0; i < a.Size(); i++)
   {
      local = std::max(local, std::abs(ah[i] - bh[i]));
   }
   MPI_Allreduce(&local, &global, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                 MPI_COMM_WORLD);
   return global;
}

} // namespace pointwise_qf

TEST_CASE("dFEM PointwiseQF", "[Parallel][dFEM][GPU][PointwiseQF]")
{
   using namespace pointwise_qf;

   const auto p = GenAll({ 2 }, { 1, 3 });
   Mesh smesh("../../data/inline-quad.mesh");
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   pmesh.EnsureNodes();
   smesh.Clear();

   H1_FECollection fec(p, DIM);
   ParFiniteElementSpace fes(&pmesh, &fec);
   const auto &ir = IntRules.Get(pmesh.GetTypicalElementGeometry(), 2 * p);
   Array<int> attributes(pmesh.attributes.Max());
   attributes = 1;

   auto local = MakeOperator<LocalQFBackend>(fes, pmesh, ir, attributes);
   auto global = MakeOperator<GlobalQFBackend>(fes, pmesh, ir, attributes);

   const int n = fes.GetTrueVSize();
   Vector u(n), du(n), coords;
   u.Randomize(1);
   du.Randomize(2);
   static_cast<ParGridFunction *>(pmesh.GetNodes())->GetTrueDofs(coords);
   MultiVector X{u, coords};

   Vector yl(n), yg(n);
   MultiVector Yl{yl}, Yg{yg};

   SECTION("Action")
   {
      local->Mult(X, Yl);
      global->Mult(X, Yg);
      REQUIRE(MaxDiff(yl, yg) == MFEM_Approx(0.0));
   }

   auto dl = local->GetDerivative(U, X);
   auto dg = global->GetDerivative(U, X);

   SECTION("Derivative action")
   {
      dl->Mult(du, Yl);
      dg->Mult(du, Yg);
      REQUIRE(MaxDiff(yl, yg) == MFEM_Approx(0.0));
   }

   SECTION("Derivative transpose")
   {
      MultiVector W{du};
      dl->MultTranspose(W, Yl);
      dg->MultTranspose(W, Yg);
      REQUIRE(MaxDiff(yl, yg) == MFEM_Approx(0.0));
   }

   SECTION("Cached derivative action")
   {
      auto dl_cached = local->GetDerivative(U, X, true);
      auto dg_cached = global->GetDerivative(U, X, true);
      dl_cached->Mult(du, Yl);
      dg_cached->Mult(du, Yg);
      REQUIRE(MaxDiff(yl, yg) == MFEM_Approx(0.0));
   }

   SECTION("Derivative assembly")
   {
      SparseMatrix *Al = nullptr, *Ag = nullptr;
      dl->Assemble(Al);
      dg->Assemble(Ag);
      TestSameMatrices(*Al, *Ag);
      delete Al;
      delete Ag;

      dl->AssembleDiagonal(yl);
      dg->AssembleDiagonal(yg);
      REQUIRE(MaxDiff(yl, yg) == MFEM_Approx(0.0));
   }

   MPI_Barrier(MPI_COMM_WORLD);
}

#endif // MFEM_USE_MPI
