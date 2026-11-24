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

#include "unit_tests.hpp"
#include "mfem.hpp"

#include <memory>

using namespace mfem;

#ifdef MFEM_USE_AMGX

namespace
{

constexpr real_t amgx_tol = 1e-8;

// Relative residual of the solution of A x = b given by the solver.
real_t RelativeResidual(Solver &solver, const Operator &A, const Vector &b)
{
   Vector x(b.Size()), r(b.Size());
   x = 0.0;
   solver.Mult(b, x);
   A.Mult(x, r);
   r -= b;
#ifdef MFEM_USE_MPI
   if (auto *pA = dynamic_cast<const HypreParMatrix *>(&A))
   {
      return std::sqrt(InnerProduct(pA->GetComm(), r, r) /
                       InnerProduct(pA->GetComm(), b, b));
   }
#endif
   return r.Norml2() / b.Norml2();
}

// Solves A x = b with two instances, destroys the first one and keeps using
// the second one, then destroys the last one and creates a new one.
template <typename MakeSolver>
void CheckSharedResource(MakeSolver make_solver, const Operator &A,
                         const Vector &b)
{
   std::unique_ptr<AmgXSolver> s1 = make_solver();
   std::unique_ptr<AmgXSolver> s2 = make_solver();
   s1->SetOperator(A);
   s2->SetOperator(A);
   REQUIRE(RelativeResidual(*s1, A, b) < amgx_tol);
   REQUIRE(RelativeResidual(*s2, A, b) < amgx_tol);

   s1.reset();
   REQUIRE(RelativeResidual(*s2, A, b) < amgx_tol);
   s2->SetOperator(A);
   REQUIRE(RelativeResidual(*s2, A, b) < amgx_tol);

   s2.reset();
   std::unique_ptr<AmgXSolver> s3 = make_solver();
   s3->SetOperator(A);
   REQUIRE(RelativeResidual(*s3, A, b) < amgx_tol);
}

} // namespace

// The AmgX library and resource are shared by the AmgXSolver instances. They
// shall be kept until the last instance is destroyed, and be set up again for
// a new instance. In an MPI build, the serial unit tests do not initialize MPI.
TEST_CASE("AmgXSolver shared resource", "[AmgX]")
{
   Mesh mesh = Mesh::MakeCartesian2D(8, 8, Element::QUADRILATERAL);
   H1_FECollection fec(1, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);
   Array<int> ess_tdofs;
   fes.GetBoundaryTrueDofs(ess_tdofs);

   BilinearForm a(&fes);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.Assemble();
   ConstantCoefficient one(1.0);
   LinearForm b(&fes);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();
   GridFunction x(&fes);
   x = 0.0;
   SparseMatrix A;
   Vector B, X;
   a.FormLinearSystem(ess_tdofs, x, b, A, X, B);

   CheckSharedResource([]()
   {
      return std::unique_ptr<AmgXSolver>(
                new AmgXSolver(AmgXSolver::SOLVER, false));
   }, A, B);
}

#ifdef MFEM_USE_MPI

// Each process uses the GPU, or the processes of a node share one GPU (MPI
// teams).
TEST_CASE("AmgXSolver shared resource in parallel", "[AmgX][Parallel]")
{
   const bool mpi_teams = GENERATE(false, true);
   CAPTURE(mpi_teams);
   Mesh smesh = Mesh::MakeCartesian2D(8, 8, Element::QUADRILATERAL);
   ParMesh mesh(MPI_COMM_WORLD, smesh);
   smesh.Clear();
   H1_FECollection fec(1, mesh.Dimension());
   ParFiniteElementSpace fes(&mesh, &fec);
   Array<int> ess_tdofs;
   fes.GetBoundaryTrueDofs(ess_tdofs);

   ParBilinearForm a(&fes);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.Assemble();
   ConstantCoefficient one(1.0);
   ParLinearForm b(&fes);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();
   ParGridFunction x(&fes);
   x = 0.0;
   HypreParMatrix A;
   Vector B, X;
   a.FormLinearSystem(ess_tdofs, x, b, A, X, B);

   CheckSharedResource([mpi_teams]()
   {
      return std::unique_ptr<AmgXSolver>(
                mpi_teams ?
                new AmgXSolver(MPI_COMM_WORLD, 1, AmgXSolver::SOLVER, false) :
                new AmgXSolver(MPI_COMM_WORLD, AmgXSolver::SOLVER, false));
   }, A, B);
}

#endif // MFEM_USE_MPI

#endif // MFEM_USE_AMGX
