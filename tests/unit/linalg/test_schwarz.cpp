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

namespace mfem
{
#ifdef MFEM_USE_MPI

// Dense serial reference for the additive Schwarz smoother:
// y = sum_s w_s R_s^T A_s^{-1} R_s x (or with A_s^{-T} if transpose is true).
static void ReferenceSchwarz(const DenseMatrix &A,
                             const std::vector<Array<HYPRE_BigInt>> &subdomains,
                             const Vector &x, Vector &y, bool transpose)
{
   const int n = A.Height();
   std::vector<Array<int>> doms(subdomains.size());
   Vector counts(n);
   counts = 0.0;
   for (size_t s = 0; s < subdomains.size(); s++)
   {
      for (HYPRE_BigInt i : subdomains[s]) { doms[s].Append((int) i); }
      doms[s].Sort();
      doms[s].Unique();
      for (int i : doms[s]) { counts(i) += 1.0; }
   }

   y.SetSize(n);
   y = 0.0;
   for (const Array<int> &dofs : doms)
   {
      const int m = dofs.Size();
      real_t sum = 0.0;
      DenseMatrix As(m);
      Vector xs(m), ys(m);
      for (int p = 0; p < m; p++)
      {
         sum += counts(dofs[p]);
         xs(p) = x(dofs[p]);
         for (int q = 0; q < m; q++) { As(p, q) = A(dofs[p], dofs[q]); }
      }
      if (transpose) { As.Transpose(); }
      DenseMatrixInverse inv(As);
      inv.Mult(xs, ys);
      for (int p = 0; p < m; p++) { y(dofs[p]) += (m / sum) * ys(p); }
   }
}

TEST_CASE("AdditiveSchwarz", "[Parallel], [AdditiveSchwarz]")
{
   const int rank = Mpi::WorldRank();
   const int nranks = Mpi::WorldSize();
   const int n = 60;

   // Nonsymmetric, diagonally dominant global matrix, known on every rank.
   // The (i+17) coupling connects dofs owned by different ranks.
   DenseMatrix Ad(n);
   Ad = 0.0;
   for (int i = 0; i < n; i++)
   {
      Ad(i, i) = 6.0 + 0.1 * (i % 5);
      if (i > 0) { Ad(i, i-1) = -1.3; }
      if (i < n-1) { Ad(i, i+1) = -0.7; }
      Ad(i, (i + 17) % n) += 0.4;
   }

   // Uneven row partitioning, where low ranks may own no rows at all.
   auto row_start = [&](int r)
   {
      return (HYPRE_BigInt) (n * r * r) / (nranks * nranks);
   };
   const HYPRE_BigInt lo = row_start(rank), hi = row_start(rank + 1);
   Array<HYPRE_BigInt> part;
   if (HYPRE_AssumedPartitionCheck()) { part = Array<HYPRE_BigInt>({lo, hi}); }
   else
   {
      for (int r = 0; r <= nranks; r++) { part.Append(row_start(r)); }
   }

   Array<int> I({0});
   Array<HYPRE_BigInt> J;
   Array<real_t> V;
   for (HYPRE_BigInt i = lo; i < hi; i++)
   {
      for (int j = 0; j < n; j++)
      {
         if (Ad((int) i, j) != 0.0)
         {
            J.Append(j);
            V.Append(Ad((int) i, j));
         }
      }
      I.Append(J.Size());
   }
   HypreParMatrix A(MPI_COMM_WORLD, (int) (hi - lo), n, n, I.GetData(),
                    J.GetData(), V.GetData(), part.GetData(), part.GetData());

   // Overlapping windows covering all dofs, a subdomain spanning all ranks
   // (given unsorted and with a repeated dof), and a single-dof subdomain.
   std::vector<Array<HYPRE_BigInt>> all_subdomains;
   for (int start = 0; start < n; start += 8)
   {
      Array<HYPRE_BigInt> window;
      for (int i = start; i < std::min(start + 12, n); i++)
      {
         window.Append(i);
      }
      all_subdomains.push_back(window);
   }
   all_subdomains.push_back(Array<HYPRE_BigInt>({59, 0, 31, 17, 0, 45}));
   all_subdomains.push_back(Array<HYPRE_BigInt>({23}));

   // Rank 0 owns no subdomains when running on more than one rank.
   std::vector<Array<HYPRE_BigInt>> my_subdomains;
   for (size_t s = 0; s < all_subdomains.size(); s++)
   {
      const int owner = (nranks > 1) ? 1 + (int) s % (nranks - 1) : 0;
      if (owner == rank) { my_subdomains.push_back(all_subdomains[s]); }
   }

   AdditiveSchwarz schwarz;
   schwarz.SetOperator(A);
   schwarz.SetSubdomains(my_subdomains);

   Vector x(n), y0(n);
   for (int i = 0; i < n; i++)
   {
      x(i) = std::sin(0.7 * i + 0.3);
      y0(i) = std::cos(1.3 * i);
   }
   Vector x_loc(x.GetData() + lo, (int) (hi - lo));
   Vector y0_loc(y0.GetData() + lo, (int) (hi - lo));

   auto max_error = [&](const Vector &y_loc, const Vector &y_ref)
   {
      real_t err = 0.0;
      for (int i = 0; i < y_loc.Size(); i++)
      {
         err = std::max(err, std::abs(y_loc(i) - y_ref((int) lo + i)));
      }
      MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_MAX, MPI_COMM_WORLD);
      return err;
   };

   const bool transpose = GENERATE(false, true);
   CAPTURE(transpose);
   Vector y_loc((int) (hi - lo)), y_ref;

   SECTION("Mult")
   {
      schwarz.iterative_mode = false;
      y_loc = 1e10; // must be overwritten
      if (transpose) { schwarz.MultTranspose(x_loc, y_loc); }
      else { schwarz.Mult(x_loc, y_loc); }
      ReferenceSchwarz(Ad, all_subdomains, x, y_ref, transpose);
      REQUIRE(max_error(y_loc, y_ref) < 1e-12);
   }

   SECTION("iterative_mode")
   {
      // y = y0 + M (x - A y0)
      schwarz.iterative_mode = true;
      y_loc = y0_loc;
      if (transpose) { schwarz.MultTranspose(x_loc, y_loc); }
      else { schwarz.Mult(x_loc, y_loc); }
      Vector r(n);
      if (transpose) { Ad.MultTranspose(y0, r); }
      else { Ad.Mult(y0, r); }
      subtract(x, r, r);
      ReferenceSchwarz(Ad, all_subdomains, r, y_ref, transpose);
      y_ref += y0;
      REQUIRE(max_error(y_loc, y_ref) < 1e-12);
   }
}

TEST_CASE("AdditiveSchwarz PCG", "[Parallel], [AdditiveSchwarz]")
{
   Mesh serial_mesh = Mesh::MakeCartesian2D(8, 8, Element::QUADRILATERAL);
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();
   H1_FECollection fec(2, mesh.Dimension());
   ParFiniteElementSpace fes(&mesh, &fec);

   Array<int> ess_tdof_list;
   fes.GetBoundaryTrueDofs(ess_tdof_list);
   ParBilinearForm a(&fes);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.Assemble();
   ParLinearForm b(&fes);
   ConstantCoefficient one(1.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();
   ParGridFunction x(&fes);
   x = 0.0;
   HypreParMatrix A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   // One subdomain per local vertex: the global true dofs of the local
   // elements around it. Subdomains near rank boundaries include dofs owned by
   // neighboring ranks.
   std::unique_ptr<Table> vert_elem(mesh.GetVertexToElementTable());
   std::vector<Array<HYPRE_BigInt>> subdomains(mesh.GetNV());
   Array<int> dofs;
   for (int v = 0; v < mesh.GetNV(); v++)
   {
      for (int k = 0; k < vert_elem->RowSize(v); k++)
      {
         fes.GetElementDofs(vert_elem->GetRow(v)[k], dofs);
         for (int d : dofs)
         {
            subdomains[v].Append(fes.GetGlobalTDofNumber(d));
         }
      }
   }
   AdditiveSchwarz schwarz(A, subdomains);

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-10);
   cg.SetMaxIter(200);
   cg.SetOperator(A);
   cg.SetPreconditioner(schwarz);
   cg.Mult(B, X);
   REQUIRE(cg.GetConverged());

   // The smoother is symmetric for a symmetric operator.
   Vector u(A.Height()), v(A.Height()), Mu(A.Height()), Mv(A.Height());
   u.Randomize(1);
   v.Randomize(2);
   schwarz.Mult(u, Mu);
   schwarz.Mult(v, Mv);
   const real_t vMu = InnerProduct(MPI_COMM_WORLD, v, Mu);
   const real_t uMv = InnerProduct(MPI_COMM_WORLD, u, Mv);
   REQUIRE(std::abs(vMu - uMv) < 1e-12 * std::abs(vMu));
}

#endif // MFEM_USE_MPI
} // namespace mfem
