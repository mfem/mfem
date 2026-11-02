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

#include "../linalg/test_same_matrices.hpp"

#include "../../../fem/dfem/doperator.hpp"
#include "../../../fem/dfem/backends/local_qf/prelude.hpp"

using namespace mfem;
using namespace mfem::future;

#ifdef MFEM_USE_ENZYME
using dscalar_t = real_t;
#else
using dscalar_t = dual<real_t, real_t>;
#endif

// ────────────────────────────────────────────────────────────────────────────
template <int DIM> struct l2_mass_qf
{
   MFEM_HOST_DEVICE inline
   void operator()(const dscalar_t &u,
                   const tensor<real_t, DIM, DIM> &J,
                   const real_t &w,
                   dscalar_t &v) const
   {
      v = u * w * det(J);
   }
};

template <int DIM> struct l2_vector_mass_qf
{
   MFEM_HOST_DEVICE inline
   void operator()(const tensor<dscalar_t, DIM> &u,
                   const tensor<real_t, DIM, DIM> &J,
                   const real_t &w,
                   tensor<dscalar_t, DIM> &v) const
   {
      v = u * w * det(J);
   }
};

// Nonlinear in u, so that the derivative depends on the state
template <int DIM> struct l2_square_qf
{
   MFEM_HOST_DEVICE inline
   void operator()(const dscalar_t &u,
                   const tensor<real_t, DIM, DIM> &J,
                   const real_t &w,
                   dscalar_t &v) const
   {
      v = u * u * w * det(J);
   }
};

static real_t GlobalMax(const Vector &v, MPI_Comm comm)
{
   real_t norm_global = 0.0, norm_local = v.Normlinf();
   MPI_Allreduce(&norm_local, &norm_global, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MAX, comm);
   return norm_global;
}

// ────────────────────────────────────────────────────────────────────────────
// Assembled derivative of a mass operator from trial_fes to test_fes, where at
// least one of the spaces is L2, compared against the classic assembly and
// against the matrix-free derivative action
template <int DIM, typename qf_t>
void l2_mass_assemble(ParMesh &pmesh, ParFiniteElementSpace &trial_fes,
                      ParFiniteElementSpace &test_fes, BilinearFormIntegrator *bfi,
                      const IntegrationRule &ir)
{
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   ParFiniteElementSpace *mfes = nodes->ParFESpace();

   Array<int> all_domain_attr(pmesh.attributes.Max());
   all_domain_attr = 1;

   ParMixedBilinearForm blf(&trial_fes, &test_fes);
   blf.AddDomainIntegrator(bfi);
   blf.Assemble();
   blf.Finalize();

   static constexpr int U = 0, V = 1, Coords = 2;
   DifferentiableOperator dop(
   { { U, &trial_fes }, { Coords, mfes } },
   { { V, &test_fes } }, pmesh);

   qf_t qfn;
   constexpr auto kernels =
      DerivativeKernels::Action |
      DerivativeKernels::AssembleMatrix;
   dop.AddDomainIntegrator<LocalQFBackend, kernels>(
      qfn,
      Inputs<Value<U>, Gradient<Coords>, Weight> {},
      Outputs<Value<V>> {},
      ir, all_domain_attr,
      Derivatives<U> {});

   Vector X(trial_fes.GetTrueVSize()), N;
   X.Randomize(1);
   nodes->GetTrueDofs(N);
   MultiVector MX{X, N};

   auto dRdU = dop.GetDerivative(U, MX);

   SECTION("SparseMatrix")
   {
      SparseMatrix *A = nullptr;
      dRdU->Assemble(A);
      REQUIRE(A->Height() == blf.SpMat().Height());
      REQUIRE(A->Width() == blf.SpMat().Width());
      TestSameMatrices(*A, blf.SpMat());
      TestSameMatrices(blf.SpMat(), *A);
      delete A;
   }

   SECTION("HypreParMatrix")
   {
      HypreParMatrix *Amfem = blf.ParallelAssemble();
      HypreParMatrix *Adfem = nullptr;
      dRdU->Assemble(Adfem);
      REQUIRE(Adfem->Height() == Amfem->Height());
      REQUIRE(Adfem->Width() == Amfem->Width());
      TestSameMatrices(*Adfem, *Amfem);
      TestSameMatrices(*Amfem, *Adfem);

      Vector D(trial_fes.GetTrueVSize()), Y(test_fes.GetTrueVSize()),
             Z(test_fes.GetTrueVSize());
      D.Randomize(7);
      MultiVector MZ{Z};
      dRdU->Mult(D, MZ);
      Adfem->Mult(D, Y);
      Y -= Z;
      REQUIRE(GlobalMax(Y, pmesh.GetComm()) == MFEM_Approx(0.0));
      delete Amfem;
      delete Adfem;
   }
}

template <int DIM>
void l2_assemble(const char *filename, int p)
{
   CAPTURE(filename, DIM, p);

   Mesh smesh(filename);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   MFEM_VERIFY(pmesh.Dimension() == DIM, "Mesh dimension mismatch");
   pmesh.EnsureNodes();
   smesh.Clear();

   p = std::max(p, pmesh.GetNodalFESpace()->GetMaxElementOrder());
   const auto &ir = IntRules.Get(pmesh.GetTypicalElementGeometry(), 2 * p + 2);

   H1_FECollection h1fec(p, DIM);
   L2_FECollection l2fec(p - 1, DIM);
   ParFiniteElementSpace h1fes(&pmesh, &h1fec);
   ParFiniteElementSpace l2fes(&pmesh, &l2fec);
   ParFiniteElementSpace vh1fes(&pmesh, &h1fec, DIM, Ordering::byVDIM);
   ParFiniteElementSpace vl2fes(&pmesh, &l2fec, DIM, Ordering::byNODES);
   ConstantCoefficient one(1.0);

   SECTION("L2 x L2")
   {
      l2_mass_assemble<DIM, l2_mass_qf<DIM>>(
                                             pmesh, l2fes, l2fes, new MassIntegrator(one, &ir), ir);
   }
   SECTION("H1 -> L2")
   {
      l2_mass_assemble<DIM, l2_mass_qf<DIM>>(
                                             pmesh, h1fes, l2fes, new MassIntegrator(one, &ir), ir);
   }
   SECTION("L2 -> H1")
   {
      l2_mass_assemble<DIM, l2_mass_qf<DIM>>(
                                             pmesh, l2fes, h1fes, new MassIntegrator(one, &ir), ir);
   }
   SECTION("Vector H1 byVDIM -> L2 byNODES")
   {
      l2_mass_assemble<DIM, l2_vector_mass_qf<DIM>>(
                                                    pmesh, vh1fes, vl2fes, new VectorMassIntegrator(one, &ir), ir);
   }
}

// ────────────────────────────────────────────────────────────────────────────
// The derivative is evaluated at the state given to GetDerivative, which must
// not be aliased by the operator, as with an identity prolongation
template <int DIM>
void derivative_state_aliasing(const char *filename, int p)
{
   CAPTURE(filename, DIM, p);

   Mesh smesh(filename);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   MFEM_VERIFY(pmesh.Dimension() == DIM, "Mesh dimension mismatch");
   pmesh.EnsureNodes();
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   smesh.Clear();

   p = std::max(p, pmesh.GetNodalFESpace()->GetMaxElementOrder());
   const auto &ir = IntRules.Get(pmesh.GetTypicalElementGeometry(), 3 * p + 2);

   Array<int> all_domain_attr(pmesh.attributes.Max());
   all_domain_attr = 1;

   L2_FECollection fec(p, DIM);
   ParFiniteElementSpace fes(&pmesh, &fec);
   const int tvsize = fes.GetTrueVSize();

   static constexpr int U = 0, Coords = 1;
   DifferentiableOperator dop(
   { { U, &fes }, { Coords, nodes->ParFESpace() } },
   { { U, &fes } }, pmesh);
   l2_square_qf<DIM> qfn;
   constexpr auto kernels =
      DerivativeKernels::Action |
      DerivativeKernels::Apply;
   dop.AddDomainIntegrator<LocalQFBackend, kernels>(
      qfn,
      Inputs<Value<U>, Gradient<Coords>, Weight> {},
      Outputs<Value<U>> {},
      ir, all_domain_attr,
      Derivatives<U> {});

   Vector X(tvsize), X0(tvsize), N, D(tvsize), Z(tvsize), Z0(tvsize);
   X.Randomize(1);
   X0 = X;
   D.Randomize(2);
   nodes->GetTrueDofs(N);
   MultiVector MX{X, N}, MX0{X0, N}, MZ{Z}, MZ0{Z0};

   for (const bool use_cache : { false, true })
   {
      CAPTURE(use_cache);
      X = X0;
      auto dRdU = dop.GetDerivative(U, MX, use_cache);
      auto dRdU0 = dop.GetDerivative(U, MX0, use_cache);

      // Modify the state after the derivative has been built
      X.Randomize(3);
      dRdU->Mult(D, MZ);
      dRdU0->Mult(D, MZ0);
      Z -= Z0;
      REQUIRE(GlobalMax(Z, pmesh.GetComm()) == MFEM_Approx(0.0));
   }
}

// ────────────────────────────────────────────────────────────────────────────
TEST_CASE("dFEM L2 Assemble 2D", "[Parallel][dFEM]")
{
   const auto p = GenAll({1}, {2, 3});
   const auto meshs = { "../../data/inline-quad.mesh" };
   const auto extra = { "../../data/star.mesh",
                        "../../data/star-q3.mesh",
                        "../../data/periodic-square.mesh"
                      };
   SECTION("Assemble")
   {
      l2_assemble<2>(GenAll(meshs, extra), p);
   }
   SECTION("State Aliasing")
   {
      derivative_state_aliasing<2>(GenAll(meshs, extra), p);
   }
}

TEST_CASE("dFEM L2 Assemble 3D", "[Parallel][dFEM]")
{
   const auto p = GenAll({1}, {2});
   const auto meshs = { "../../data/inline-hex.mesh" };
   const auto extra = { "../../data/fichera.mesh",
                        "../../data/fichera-q3.mesh",
                        "../../data/periodic-cube.mesh"
                      };
   SECTION("Assemble")
   {
      l2_assemble<3>(GenAll(meshs, extra), p);
   }
   SECTION("State Aliasing")
   {
      derivative_state_aliasing<3>(GenAll(meshs, extra), p);
   }
}

#endif // MFEM_USE_MPI
