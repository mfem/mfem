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
#include "../../config/config.hpp"

namespace mfem
{
#ifdef MFEM_USE_MPI

class DenseMatrixSolver : public Solver
{
private:
   DenseMatrix A;
   LUFactors LU;
   int * ipiv = nullptr;
public:
   DenseMatrixSolver() : Solver() {}

   void SetOperator(const Operator &op) override
   {
      auto Oph = const_cast<HypreParMatrix *>(dynamic_cast<const HypreParMatrix *>
                                              (&op));
      MFEM_VERIFY(Oph, "Not a compatible matrix type");
      SparseMatrix Sp;
      Oph->MergeDiagAndOffd(Sp);
      Sp.ToDenseMatrix(A);
      delete [] ipiv;
      ipiv = new int[A.Height()];
      LU.data = A.Data();
      LU.ipiv = ipiv;
      LU.Factor(A.Height());
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      y = x;
      LU.Solve(A.Height(), 1, y.GetData());
   }

   ~DenseMatrixSolver() { delete [] ipiv; }
};

HypreParMatrix * GetProlongationMatrix(const ParFiniteElementSpace* pfes,
                                       int element_attribute)
{
   Array<int> dofs;
   Array<int> tdofs;
   for (int i = 0; i < pfes->GetNE(); i++)
   {
      if (pfes->GetAttribute(i) == element_attribute)
      {
         pfes->GetElementVDofs(i, dofs);
         tdofs.Append(dofs);
      }
   }
   tdofs.Sort();
   tdofs.Unique();

   HYPRE_BigInt h = tdofs.Size();
   SparseMatrix St(h,pfes->GlobalTrueVSize());

   for (int i = 0; i<h; i++)
   {
      int col = tdofs[i];
      St.Set(i,col,1.0);
   }
   St.Finalize();
   HYPRE_BigInt rows[2];
   HYPRE_BigInt cols[2];
   int nrows = St.Height();

   rows[0] = 0; rows[1] = nrows;
   for (int i = 0; i < 2; i++)
   {
      cols[i] = pfes->GetTrueDofOffsets()[i];
   }
   HYPRE_BigInt glob_nrows = nrows;
   HYPRE_BigInt glob_ncols = pfes->GlobalTrueVSize();

   HYPRE_BigInt * J;
#if !(defined(HYPRE_BIGINT) || defined(HYPRE_MIXEDINT))
   J = St.GetJ();
#else
   J = new HYPRE_BigInt[St.NumNonZeroElems()];
   std::copy(St.GetJ(), St.GetJ() + St.NumNonZeroElems(), J);
#endif
   HypreParMatrix * Pt = new HypreParMatrix(pfes->GetComm(), nrows, glob_nrows,
                                            glob_ncols, St.GetI(), J,
                                            St.GetData(), rows,cols);

   HypreParMatrix * P = Pt->Transpose();
   delete Pt;
#if (defined(HYPRE_BIGINT) || defined(HYPRE_MIXEDINT))
   delete [] J;
#endif
   return P;
}


TEST_CASE("FilteredSolver and AMGFSolver", "[Parallel]")
{
   // Note: This test is restricted to a single processor for convenience,
   // allowing the use of a serial dense direct solver on the filtered subspace
   // and avoiding any dependency on external parallel sparse direct solvers.
   // In general, both AMGFSolver and FilteredSolver are designed to work in parallel.
   if (Mpi::Root())
   {
      MPI_Comm comm = MPI_COMM_SELF;

      auto ref_levels = 2;

      auto [order, eps, iteration_bound] =
         GENERATE(table<int, real_t, int>(
      {
         {2, 1e-3, 12},
         {2, 1e-4, 12},
         {3, 1e-3, 18},
         {3, 1e-4, 18}
      }));

      CAPTURE(order, eps);

      Mesh mesh = Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL, 1.0, 1.0);
      mesh.EnsureNodes();
      GridFunction *nodes = mesh.GetNodes();
      (*nodes)[2] = 0.5*(1-eps);   (*nodes)[10] = 0.5*(1-eps);
      (*nodes)[18] = 0.5*(1-eps);  (*nodes)[26] = 0.5*(1-eps);
      (*nodes)[4] = 0.5*(1+eps);   (*nodes)[12] = 0.5*(1+eps);
      (*nodes)[20] = 0.5*(1+eps);  (*nodes)[28] = 0.5*(1+eps);
      mesh.SetAttribute(3, 2);
      mesh.SetAttribute(6, 2);
      mesh.SetAttribute(7, 2);
      mesh.SetAttributes();

      int dim = mesh.Dimension();

      ParMesh pmesh(comm, mesh);
      mesh.Clear();
      for (int l = 0; l < ref_levels; l++)
      {
         pmesh.UniformRefinement();
      }

      H1_FECollection fec(order, dim);
      ParFiniteElementSpace fespace(&pmesh, &fec);

      Array<int> ess_bdr, ess_tdof_list;
      ess_bdr.SetSize(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      ParGridFunction x(&fespace);   x = 0.0;

      ParLinearForm b(&fespace);
      ConstantCoefficient one(1.0);
      b.AddDomainIntegrator(new DomainLFIntegrator(one));
      b.Assemble();

      ParBilinearForm a(&fespace);
      ConstantCoefficient eps_cf(eps);;
      Vector vec(pmesh.attributes.Max());
      vec(0) = 1.0;
      vec(1) = 1/eps;
      PWConstCoefficient eps_cg(vec);
      a.AddDomainIntegrator(new DiffusionIntegrator(eps_cg));
      a.Assemble();

      OperatorPtr A;
      Vector B, X;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      HypreParMatrix *P = GetProlongationMatrix(&fespace, 2);

      // 1st preconditioner: AMG
      HypreBoomerAMG amg;
      // 2nd preconditioner: AMGF
      AMGFSolver amgf;
      amg.SetPrintLevel(0);
      DenseMatrixSolver subspacesolver;
      amgf.GetAMG().SetPrintLevel(0);
      amgf.SetFilteredSubspaceSolver(subspacesolver);
      amgf.SetFilteredSubspaceTransferOperator(*P);
      // 3rd preconditioner: FilteredSolver
      FilteredSolver fs;
      fs.SetSolver(amg);
      fs.SetFilteredSubspaceTransferOperator(*P);
      fs.SetFilteredSubspaceSolver(subspacesolver);

      X = 0.0;
      Vector Xamgf(X);
      Vector Xfs(X);

      CGSolver cg(comm);
      cg.SetAbsTol(1e-16);
      cg.SetMaxIter(5000);
      cg.SetPrintLevel(3);
      cg.SetPreconditioner(amg);
      cg.SetOperator(*A);
      cg.Mult(B, X);

      cg.SetPreconditioner(amgf);
      cg.SetOperator(*A);
      cg.Mult(B, Xamgf);
      int amgf_iter = cg.GetNumIterations();

      cg.SetPreconditioner(fs);
      cg.SetOperator(*A);
      cg.Mult(B, Xfs);
      int fs_iter = cg.GetNumIterations();

      Xamgf -= X;
      REQUIRE(Xamgf.Norml2() == MFEM_Approx(0.0,1e-7));
      Xfs -= X;
      REQUIRE(Xfs.Norml2() == MFEM_Approx(0.0,1e-7));

      REQUIRE(amgf_iter == fs_iter);
      REQUIRE(amgf_iter <= iteration_bound);
      REQUIRE(fs_iter <= iteration_bound);

      // Reversed and additive combination modes.
      using Mode = FilteredSolver::Mode;
      for (Mode mode : {Mode::REVERSED, Mode::ADDITIVE})
      {
         amgf.SetMode(mode);
         Xamgf = 0.0;
         cg.SetPreconditioner(amgf);
         cg.SetOperator(*A);
         cg.Mult(B, Xamgf);
         REQUIRE(cg.GetNumIterations() <= 2*iteration_bound);
      }

      // Automatically generated filtered subspace.
      AMGFSolver amgf_auto;
      amgf_auto.GetAMG().SetPrintLevel(0);
      auto factory = []() -> std::unique_ptr<Solver>
      {
         return std::make_unique<DenseMatrixSolver>();
      };
      amgf_auto.EnableAutoFilteredSubspace(true, factory);
      Xamgf = 0.0;
      cg.SetPreconditioner(amgf_auto);
      cg.SetOperator(*A);
      cg.Mult(B, Xamgf);
      REQUIRE(cg.GetNumIterations() <= iteration_bound);

      delete P;
   }
}

// Subspace solver that, like most parallel direct solvers (e.g. SuperLU), only
// accepts operators with the local size it was first set up with. It applies
// the inverse of the diagonal of the operator.
class FixedLocalSizeSolver : public Solver
{
private:
   bool initialized = false;
   Vector inv_diag;
public:
   bool size_changed = false;

   void SetOperator(const Operator &op) override
   {
      auto Oph = dynamic_cast<const HypreParMatrix *>(&op);
      MFEM_VERIFY(Oph, "Not a compatible matrix type");
      if (initialized && Oph->Height() != height) { size_changed = true; }
      initialized = true;
      height = width = Oph->Height();
      Oph->GetDiag(inv_diag);
      for (int i = 0; i < inv_diag.Size(); i++)
      {
         inv_diag(i) = 1.0/inv_diag(i);
      }
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      for (int i = 0; i < x.Size(); i++) { y(i) = inv_diag(i)*x(i); }
   }
};

// Tridiagonal matrix with n_loc rows per rank, -1 off the diagonal and 4 on the
// diagonal, except for the global rows in large_rows, which get 1e6 instead.
HypreParMatrix * MakeTridiagonalMatrix(
   MPI_Comm comm, int n_loc, const std::vector<HYPRE_BigInt> &large_rows)
{
   int rank, nranks;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &nranks);
   HYPRE_BigInt n_glob = (HYPRE_BigInt)n_loc*nranks;
   HYPRE_BigInt row_begin = (HYPRE_BigInt)n_loc*rank;
   HYPRE_BigInt rows[2] = {row_begin, row_begin+n_loc};

   std::vector<int> I = {0};
   std::vector<HYPRE_BigInt> J;
   std::vector<real_t> data;
   for (HYPRE_BigInt row = rows[0]; row < rows[1]; row++)
   {
      bool large = std::find(large_rows.begin(), large_rows.end(), row) !=
                   large_rows.end();
      if (row > 0) { J.push_back(row-1); data.push_back(-1.0); }
      J.push_back(row); data.push_back(large ? 1e6 : 4.0);
      if (row < n_glob-1) { J.push_back(row+1); data.push_back(-1.0); }
      I.push_back((int)J.size());
   }
   return new HypreParMatrix(comm, n_loc, n_glob, n_glob, I.data(), J.data(),
                             data.data(), rows, rows);
}

TEST_CASE("AMGFSolver auto subspace moving between ranks", "[Parallel]")
{
   // The automatically generated filtered subspace can keep the same global
   // size while its rows move between ranks. The subspace solver must then
   // be rebuilt, since it only accepts operators of its original local size.
   MPI_Comm comm = MPI_COMM_WORLD;
   const int n_loc = 8;
   const HYPRE_BigInt n_glob = (HYPRE_BigInt)n_loc*Mpi::WorldSize();

   // Both operators have two rows of much larger norm: on rank 0 for the
   // first one, and on the first and last rank for the second one.
   std::unique_ptr<HypreParMatrix> A1(
      MakeTridiagonalMatrix(comm, n_loc, {1, 2}));
   std::unique_ptr<HypreParMatrix> A2(
      MakeTridiagonalMatrix(comm, n_loc, {1, n_glob-2}));

   FixedLocalSizeSolver *subspacesolver = nullptr;
   AMGFSolver amgf;
   amgf.GetAMG().SetPrintLevel(0);
   amgf.EnableAutoFilteredSubspace(true, [&subspacesolver]()
                                   -> std::unique_ptr<Solver>
   {
      auto s = std::make_unique<FixedLocalSizeSolver>();
      subspacesolver = s.get();
      return s;
   });

   Vector b(n_loc), x(n_loc);
   b = 1.0;
   for (HypreParMatrix *A : {A1.get(), A2.get()})
   {
      amgf.SetOperator(*A);
      REQUIRE(amgf.FilteringEnabled());
      // The subspace solver is set up on the first application.
      x = 0.0;
      amgf.Mult(b, x);
      REQUIRE(!subspacesolver->size_changed);
   }
}
#endif

} // namespace mfem
