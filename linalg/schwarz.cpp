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

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "schwarz.hpp"
#include "sparsemat.hpp"
#include <algorithm>

namespace mfem
{

AdditiveSchwarz::AdditiveSchwarz(
   const HypreParMatrix &A_,
   const std::vector<Array<HYPRE_BigInt>> &subdomains_)
   : Solver()
{
   SetSubdomains(subdomains_);
   SetOperator(A_);
}

void AdditiveSchwarz::SetSubdomains(
   const std::vector<Array<HYPRE_BigInt>> &subdomains_)
{
   subdomains = subdomains_;
   for (Array<HYPRE_BigInt> &dofs : subdomains)
   {
      dofs.Sort();
      dofs.Unique();
   }
   subdomains_set = true;
   if (A) { Setup(); }
}

void AdditiveSchwarz::SetOperator(const Operator &op)
{
   A = dynamic_cast<const HypreParMatrix *>(&op);
   MFEM_VERIFY(A, "AdditiveSchwarz requires a HypreParMatrix operator.");
   MFEM_VERIFY(A->Height() == A->Width() &&
               A->GetGlobalNumRows() == A->GetGlobalNumCols(),
               "AdditiveSchwarz requires a square operator.");
   height = A->Height();
   width = A->Width();
   if (subdomains_set) { Setup(); }
}

void AdditiveSchwarz::Setup()
{
   MPI_Comm comm = A->GetComm();
   const HYPRE_BigInt glob_n = A->GetGlobalNumRows();
   const int num_sub = static_cast<int>(subdomains.size());

   // 1. Lay out the local subdomain dofs contiguously, subdomain by subdomain.
   offsets.SetSize(num_sub + 1);
   offsets[0] = 0;
   for (int s = 0; s < num_sub; s++)
   {
      offsets[s+1] = offsets[s] + subdomains[s].Size();
   }
   const int n_loc = offsets[num_sub];

   // 2. Build the boolean restriction R, whose row offsets[s] + p selects the
   // global dof subdomains[s][p]. Its rows are owned by this rank, and its
   // columns follow the partitioning of A.
   Array<int> I(n_loc + 1);
   Array<HYPRE_BigInt> J(n_loc);
   Vector ones(n_loc);
   ones = 1.0;
   for (int s = 0; s < num_sub; s++)
   {
      for (int p = 0; p < subdomains[s].Size(); p++)
      {
         const HYPRE_BigInt dof = subdomains[s][p];
         MFEM_VERIFY(0 <= dof && dof < glob_n, "AdditiveSchwarz: dof " << dof
                     << " of subdomain " << s << " is out of range.");
         J[offsets[s] + p] = dof;
      }
   }
   for (int k = 0; k <= n_loc; k++) { I[k] = k; }

   HYPRE_BigInt n_loc_big = n_loc, glob_n_sub;
   MPI_Allreduce(&n_loc_big, &glob_n_sub, 1, HYPRE_MPI_BIG_INT, MPI_SUM, comm);
   Array<HYPRE_BigInt> row_starts;
   if (HYPRE_AssumedPartitionCheck())
   {
      row_starts.SetSize(2);
      MPI_Scan(&n_loc_big, &row_starts[1], 1, HYPRE_MPI_BIG_INT, MPI_SUM, comm);
      row_starts[0] = row_starts[1] - n_loc_big;
   }
   else
   {
      int num_ranks;
      MPI_Comm_size(comm, &num_ranks);
      row_starts.SetSize(num_ranks + 1);
      row_starts[0] = 0;
      MPI_Allgather(&n_loc_big, 1, HYPRE_MPI_BIG_INT, &row_starts[1], 1,
                    HYPRE_MPI_BIG_INT, comm);
      row_starts.PartialSum();
   }

   R.reset(new HypreParMatrix(comm, n_loc, glob_n_sub, glob_n, I.GetData(),
                              J.GetData(), ones.HostRead(),
                              row_starts.GetData(), A->ColPart()));

   // 3. Compute the weights: c = R^T 1 counts, for every dof, the subdomains
   // (on all ranks) that contain it, and R c gathers these counts onto the
   // local subdomain dofs.
   Vector counts(A->Width());
   R->MultTranspose(ones, counts);
   xs.SetSize(n_loc);
   R->Mult(counts, xs);
   const real_t *h_counts = xs.HostRead();
   weights.SetSize(num_sub);
   for (int s = 0; s < num_sub; s++)
   {
      real_t sum = 0.0;
      for (int k = offsets[s]; k < offsets[s+1]; k++) { sum += h_counts[k]; }
      weights(s) = (sum > 0.0) ? subdomains[s].Size() / sum : 0.0;
   }

   // 4. Fetch the rows of A needed by the local subdomains: row
   // offsets[s] + p of R A is the global row subdomains[s][p] of A.
   std::unique_ptr<HypreParMatrix> RA(ParMult(R.get(), A));
   RA->HostRead();
   SparseMatrix diag, offd;
   HYPRE_BigInt *cmap;
   RA->GetDiag(diag);
   RA->GetOffd(offd, cmap);
   const HYPRE_BigInt first_col =
      hypre_ParCSRMatrixFirstColDiag(static_cast<hypre_ParCSRMatrix *>(*RA));
   const int *diag_I = diag.HostReadI(), *diag_J = diag.HostReadJ();
   const int *offd_I = offd.HostReadI(), *offd_J = offd.HostReadJ();
   const real_t *diag_V = diag.HostReadData(), *offd_V = offd.HostReadData();

   // 5. Assemble each dense subdomain matrix A_s from the entries of those
   // rows whose columns are also in the subdomain, then LU factor it.
   lu.resize(num_sub);
   ipiv.SetSize(n_loc);
   for (int s = 0; s < num_sub; s++)
   {
      const Array<HYPRE_BigInt> &dofs = subdomains[s];
      const int n = dofs.Size();
      DenseMatrix &As = lu[s];
      As.SetSize(n);
      As = 0.0;
      for (int p = 0; p < n; p++)
      {
         const int row = offsets[s] + p;
         auto add_entry = [&](HYPRE_BigInt col, real_t val)
         {
            const HYPRE_BigInt *it =
               std::lower_bound(dofs.begin(), dofs.end(), col);
            if (it != dofs.end() && *it == col)
            {
               As(p, static_cast<int>(it - dofs.begin())) += val;
            }
         };
         for (int k = diag_I[row]; k < diag_I[row+1]; k++)
         {
            add_entry(first_col + diag_J[k], diag_V[k]);
         }
         for (int k = offd_I[row]; k < offd_I[row+1]; k++)
         {
            add_entry(cmap[offd_J[k]], offd_V[k]);
         }
      }
      LUFactors factors(As.Data(), ipiv.GetData() + offsets[s]);
      MFEM_VERIFY(factors.Factor(n), "AdditiveSchwarz: the matrix of subdomain "
                  << s << " is singular.");
   }
}

void AdditiveSchwarz::Mult(const Vector &x, Vector &y) const
{
   Apply(x, y, false);
}

void AdditiveSchwarz::MultTranspose(const Vector &x, Vector &y) const
{
   Apply(x, y, true);
}

void AdditiveSchwarz::Apply(const Vector &x, Vector &y, bool transpose) const
{
   MFEM_VERIFY(R, "AdditiveSchwarz: SetSubdomains() and SetOperator() must be "
               "called before use.");
   MFEM_VERIFY(x.Size() == Width(), "invalid input vector");
   MFEM_VERIFY(y.Size() == Height(), "invalid output vector");

   const Vector *res = &x;
   if (iterative_mode)
   {
      r.SetSize(height);
      if (transpose) { A->MultTranspose(y, r); }
      else { A->Mult(y, r); }
      subtract(x, r, r); // r = x - A y (or x - A^T y)
      res = &r;
   }

   // Restrict the residual onto the local subdomain dofs, solve with each
   // local subdomain matrix (or its transpose), and weight the corrections.
   R->Mult(*res, xs);
   real_t *h_xs = xs.HostReadWrite();
   // LUFactors only reads its data and pivots in Solve() and RightSolve(), but
   // stores them as non-const pointers.
   int *h_ipiv = const_cast<int *>(ipiv.GetData());
   for (int s = 0; s < static_cast<int>(lu.size()); s++)
   {
      const int n = lu[s].Height();
      real_t *xs_s = h_xs + offsets[s];
      LUFactors factors(lu[s].Data(), h_ipiv + offsets[s]);
      // RightSolve(n, 1, x) computes x^T A_s^{-1}, i.e. A_s^{-T} x.
      if (transpose) { factors.RightSolve(n, 1, xs_s); }
      else { factors.Solve(n, 1, xs_s); }
      for (int p = 0; p < n; p++) { xs_s[p] *= weights(s); }
   }

   // Sum the corrections of all subdomains (on all ranks) into y.
   R->MultTranspose(1.0, xs, iterative_mode ? 1.0 : 0.0, y);
}

} // namespace mfem

#endif // MFEM_USE_MPI
