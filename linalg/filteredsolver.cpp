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

#include "filteredsolver.hpp"
#include "sparsemat.hpp"
#ifdef MFEM_USE_PETSC
#include "petsc.hpp"
#endif
#ifdef MFEM_USE_MPI
#include "hypre_parcsr.hpp"
#include "../general/communication.hpp"
#include <cmath>
#include <limits>
#include <algorithm>
#endif

namespace mfem
{

std::unique_ptr<const Operator> FilteredSolver::GetPtAP(const Operator *Aop,
                                                        const Operator *Pop) const
{
#ifdef MFEM_USE_MPI
   const HypreParMatrix * Ah = dynamic_cast<const HypreParMatrix*>(Aop);
   const HypreParMatrix * Ph = dynamic_cast<const HypreParMatrix*>(Pop);
   if (Ah && Ph) { return std::unique_ptr<const Operator>(RAP(Ah, Ph)); }
#endif
#ifdef MFEM_USE_PETSC
   PetscParMatrix* Ap = const_cast<PetscParMatrix*>(
                           dynamic_cast<const PetscParMatrix*>(Aop));
   PetscParMatrix* Pp = const_cast<PetscParMatrix*>(
                           dynamic_cast<const PetscParMatrix*>(Pop));
   if (Ap && Pp) { return std::unique_ptr<const Operator>(RAP(Ap, Pp)); }
#endif
   const SparseMatrix * Asp = dynamic_cast<const SparseMatrix*>(Aop);
   const SparseMatrix * Psp = dynamic_cast<const SparseMatrix*>(Pop);
   if (Asp && Psp)   { return std::unique_ptr<const Operator>(RAP(*Asp, *Psp)); }

   return std::unique_ptr<const Operator>(new RAPOperator(*Pop, *Aop, *Pop));
}

void FilteredSolver::InitVectors() const
{
   MFEM_VERIFY(A, "Operator not set");
   MFEM_VERIFY(P, "Transfer operator not set");
   MFEM_VERIFY(B, "Solver is not set.");
   MFEM_VERIFY(S, "Filtered space solver is not set.");

   z.SetSize(height);
   z.UseDevice(true);
   r.SetSize(height);
   r.UseDevice(true);
   xf.SetSize(P->Width());
   xf.UseDevice(true);
   rf.SetSize(P->Width());
   rf.UseDevice(true);
}

void FilteredSolver::MakeSolver() const
{
   if (solver_set) { return; }

   InitVectors();

   // Original space solver
   B->SetOperator(*A);

   // Filtered space operator
   PtAP = GetPtAP(A, P);

   // Filtered space solver
   S->SetOperator(*PtAP);

   solver_set = true;
}

void FilteredSolver::SetOperator(const Operator &op)
{
   A = &op;
   height = op.Height();
   width = op.Width();
   solver_set = false;
}

void FilteredSolver::SetSolver(Solver &B_)
{
   B = &B_;
   solver_set = false;
}

void FilteredSolver::SetFilteredSubspaceTransferOperator(const Operator &P_)
{
   P = &P_;
   solver_set = false;
}

void FilteredSolver::SetFilteredSubspaceSolver(Solver &S_)
{
   S = &S_;
   solver_set = false;
}

void FilteredSolver::Mult(const Vector &b, Vector &x) const
{
   MFEM_VERIFY(b.Size() == x.Size(), "Inconsistent b and x size");
   MakeSolver();

   switch (mode)
   {
      case Mode::MULTIPLICATIVE:
         MultMultiplicative(b, x);
         break;
      case Mode::REVERSED:
         MultReversed(b, x);
         break;
      case Mode::ADDITIVE:
         MultAdditive(b, x);
         break;
      default:
         MFEM_ABORT("FilteredSolver::Mult: unknown mode");
   }
}

void FilteredSolver::MultMultiplicative(const Vector &b, Vector &x) const
{
   x = 0.0;
   r = b;

   // z = B x
   B->Mult(b, z);
   // x = x + z
   x+=z;

   // r = b - A x = r - A z
   A->AddMult(z, r, -1.0);

   // rf = Pᵀ r
   P->MultTranspose(r, rf);

   // xf = S rf
   S->Mult(rf, xf);

   // z = P xf
   P->Mult(xf, z);

   // x = x + z
   x+=z;

   // r = b - A x = r - A z
   A->AddMult(z, r, -1.0);

   // z = B r
   B->Mult(r, z);
   x+=z;
}

void FilteredSolver::MultReversed(const Vector &b, Vector &x) const
{
   x = 0.0;
   r = b;

   // rf = Pᵀ b
   P->MultTranspose(b, rf);

   // xf = S rf
   S->Mult(rf, xf);

   // z = P xf
   P->Mult(xf, z);

   // x = x + z
   x+=z;

   // r = b - A x = r - A z
   A->AddMult(z, r, -1.0);

   // z = B r
   B->Mult(r, z);
   x+=z;

   // r = b - A x = r - A z
   A->AddMult(z, r, -1.0);

   // rf = Pᵀ r
   P->MultTranspose(r, rf);

   // xf = S rf
   S->Mult(rf, xf);

   // z = P xf
   P->Mult(xf, z);
   x+=z;
}

void FilteredSolver::MultAdditive(const Vector &b, Vector &x) const
{
   // x = B b
   B->Mult(b, x);

   // rf = Pᵀ b
   P->MultTranspose(b, rf);

   // xf = S rf
   S->Mult(rf, xf);

   // z = P xf
   P->Mult(xf, z);

   // x = x + z
   x+=z;
}

#ifdef MFEM_USE_MPI

void AMGFSolver::SetOperator(const Operator &A_)
{
   auto Ah_ = dynamic_cast<const HypreParMatrix*>(&A_);
   MFEM_VERIFY(Ah_, "AMGFSolver::SetOperator: HypreParMatrix expected.");
   Ah = Ah_;
   // Assume filtering is enabled for the new operator, until/unless a call
   // to GenerateFilteredSubspaceTransferOperator() says otherwise.
   filtering_enabled = true;
   FilteredSolver::SetOperator(*Ah);
   if (auto_subspace)
   {
      GenerateFilteredSubspaceTransferOperator(auto_max_iter, auto_tol,
                                               auto_jump_threshold);
   }
}

void AMGFSolver::SetFilteredSubspaceTransferOperator(const HypreParMatrix &Pop)
{
   FilteredSolver::SetFilteredSubspaceTransferOperator(Pop);
}

void AMGFSolver::EnableAutoFilteredSubspace(
   bool enable, std::function<std::unique_ptr<Solver>()> solver_factory,
   int max_iter, real_t tol, real_t jump_threshold)
{
   MFEM_VERIFY(!enable || solver_factory,
               "AMGFSolver::EnableAutoFilteredSubspace: a solver_factory is "
               "required when enable is true.");
   auto_subspace = enable;
   auto_subspace_solver_factory = std::move(solver_factory);
   auto_subspace_solver.reset();
   auto_subspace_width = -1;
   auto_max_iter = max_iter;
   auto_tol = tol;
   auto_jump_threshold = jump_threshold;
}

namespace
{
/// RAII guard that frees an array allocated with hypre's allocator.
struct HypreArrayGuard
{
   real_t *ptr;
   ~HypreArrayGuard() { if (ptr) { mfem_hypre_TFree(ptr); } }
};

/// log of the (unnormalized) 1D Gaussian density, used so that the E-step
/// responsibilities can be computed via a numerically robust log-sum-exp.
inline real_t LogGaussian(real_t x, real_t mean, real_t var, real_t log_weight)
{
   return log_weight - 0.5*std::log(2.0*M_PI*var) -
          (x-mean)*(x-mean)/(2.0*var);
}
} // anonymous namespace

bool AMGFSolver::GenerateFilteredSubspaceTransferOperator(int max_iter,
                                                          real_t tol,
                                                          real_t jump_threshold)
{
   MFEM_VERIFY(Ah, "AMGFSolver::GenerateFilteredSubspaceTransferOperator: "
               "SetOperator must be called first.");
   MPI_Comm comm = Ah->GetComm();
   const int nrows_local = Ah->Height();

   // 1. Compute the L1 norm of every (local) row of Ah. Option 1 gives the
   // true L1 row norm: the sum of |a_ij| over the full row, including both
   // the diagonal and off-diagonal (other-rank) blocks of the distributed
   // matrix, so this already reflects contributions from all ranks that
   // share entries in a given row.
   real_t *l1_norms_raw = nullptr;
   hypre_ParCSRComputeL1Norms(*Ah, 1, NULL, &l1_norms_raw);
   MFEM_VERIFY(l1_norms_raw,
               "AMGFSolver::GenerateFilteredSubspaceTransferOperator: "
               "failed to compute L1 row norms (zero row in operator?).");
   HypreArrayGuard l1_guard{l1_norms_raw};
   const Vector x(l1_norms_raw, nrows_local);

   // Global number of rows, across all ranks.
   HYPRE_BigInt n_local = nrows_local, n_global;
   MPI_Allreduce(&n_local, &n_global, 1, MPITypeMap<HYPRE_BigInt>::mpi_type,
                 MPI_SUM, comm);

   // 2. Fit a two-component 1D GMM to the row norms with EM, computed in
   // parallel: the E-step responsibilities are purely local (one row of the
   // distributed operator lives on exactly one rank), and only the six
   // scalar M-step sums need to be reduced across ranks each iteration.
   real_t local_min = (nrows_local > 0) ? x.Min() :
                      std::numeric_limits<real_t>::max();
   real_t local_max = (nrows_local > 0) ? x.Max() :
                      std::numeric_limits<real_t>::lowest();
   real_t global_min, global_max;
   MPI_Allreduce(&local_min, &global_min, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MIN, comm);
   MPI_Allreduce(&local_max, &global_max, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MAX, comm);

   const real_t var_floor = std::numeric_limits<real_t>::epsilon() *
                            std::max(global_max*global_max, real_t(1.0));

   if (global_max - global_min <= var_floor)
   {
      // All row norms are (numerically) identical: there is no separation
      // to find, so fall back to plain AMG.
      filtering_enabled = false;
      return false;
   }

   real_t mu[2] = {global_min, global_max};
   real_t var[2];
   var[0] = var[1] = std::max((global_max-global_min)*(global_max-global_min)/4.0,
                              var_floor);
   real_t log_pi[2] = {std::log(0.5), std::log(0.5)};

   Vector resp0(nrows_local), resp1(nrows_local);

   for (int iter = 0; iter < max_iter; iter++)
   {
      // E-step (local to this rank).
      for (int i = 0; i < nrows_local; i++)
      {
         real_t xi = x(i);
         real_t log_p0 = LogGaussian(xi, mu[0], var[0], log_pi[0]);
         real_t log_p1 = LogGaussian(xi, mu[1], var[1], log_pi[1]);
         real_t m = std::max(log_p0, log_p1);
         real_t p0 = std::exp(log_p0-m);
         real_t p1 = std::exp(log_p1-m);
         real_t s = p0+p1;
         resp0(i) = p0/s;
         resp1(i) = p1/s;
      }

      // Local partial sums for the M-step: N_k, S_k = sum r_ik x_i,
      // Q_k = sum r_ik x_i^2, for k = 0, 1.
      real_t local_sums[6] = {0,0,0,0,0,0};
      for (int i = 0; i < nrows_local; i++)
      {
         real_t xi = x(i);
         local_sums[0] += resp0(i);
         local_sums[1] += resp1(i);
         local_sums[2] += resp0(i)*xi;
         local_sums[3] += resp1(i)*xi;
         local_sums[4] += resp0(i)*xi*xi;
         local_sums[5] += resp1(i)*xi*xi;
      }
      real_t global_sums[6];
      MPI_Allreduce(local_sums, global_sums, 6, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, comm);

      real_t mu_old[2] = {mu[0], mu[1]};
      for (int k = 0; k < 2; k++)
      {
         real_t Nk = global_sums[k];
         if (Nk <= 0.0) { continue; }
         real_t Sk = global_sums[2+k];
         real_t Qk = global_sums[4+k];
         mu[k] = Sk/Nk;
         var[k] = std::max(Qk/Nk - mu[k]*mu[k], var_floor);
         log_pi[k] = std::log(Nk / (real_t)n_global);
      }

      real_t dmu0 = std::abs(mu[0]-mu_old[0]) /
                    std::max(std::abs(mu_old[0]), var_floor);
      real_t dmu1 = std::abs(mu[1]-mu_old[1]) /
                    std::max(std::abs(mu_old[1]), var_floor);
      if (std::max(dmu0, dmu1) < tol) { break; }
   }

   // 3. Decide whether the larger-norm cluster is separated enough from the
   // smaller-norm one to be worth filtering.
   const int small = (mu[0] <= mu[1]) ? 0 : 1;
   const int large = 1-small;

   if (mu[large] <= jump_threshold * mu[small])
   {
      filtering_enabled = false;
      return false;
   }

   // 4. Build the filtered-subspace transfer operator: select the local
   // rows whose final responsibility favors the larger-norm cluster (a
   // purely local decision, needing no further communication), then form a
   // selection matrix mapping those rows into the full space and transpose
   // it to get the subspace-to-full-space transfer operator P.
   Array<int> selected_rows;
   for (int i = 0; i < nrows_local; i++)
   {
      real_t xi = x(i);
      real_t log_p_small = LogGaussian(xi, mu[small], var[small],
                                       log_pi[small]);
      real_t log_p_large = LogGaussian(xi, mu[large], var[large],
                                       log_pi[large]);
      if (log_p_large > log_p_small) { selected_rows.Append(i); }
   }

   const int nrows_f = selected_rows.Size();
   SparseMatrix Pft(nrows_f, Ah->GetGlobalNumCols());
   for (int i = 0; i < nrows_f; i++)
   {
      int col = Ah->ColPart()[0] + selected_rows[i];
      Pft.Set(i, col, 1.0);
   }
   Pft.Finalize();

   HYPRE_BigInt nrows_f_bigint = nrows_f;
   HYPRE_BigInt row_offset_f;
   MPI_Scan(&nrows_f_bigint, &row_offset_f, 1,
            MPITypeMap<HYPRE_BigInt>::mpi_type, MPI_SUM, comm);
   row_offset_f -= nrows_f_bigint;
   HYPRE_BigInt rows_f[2] = {row_offset_f, row_offset_f+nrows_f_bigint};

   HYPRE_BigInt glob_nrows_f;
   MPI_Allreduce(&nrows_f_bigint, &glob_nrows_f, 1,
                 MPITypeMap<HYPRE_BigInt>::mpi_type, MPI_SUM, comm);

   HYPRE_BigInt *J;
#ifndef HYPRE_BIGINT
   J = Pft.GetJ();
#else
   J = new HYPRE_BigInt[Pft.NumNonZeroElems()];
   for (int i = 0; i < Pft.NumNonZeroElems(); i++)
   {
      J[i] = Pft.GetJ()[i];
   }
#endif

   std::unique_ptr<HypreParMatrix> P_ft(
      new HypreParMatrix(comm, nrows_f, glob_nrows_f, Ah->GetGlobalNumCols(),
                         Pft.GetI(), J, Pft.GetData(), rows_f, Ah->ColPart()));
#ifdef HYPRE_BIGINT
   delete [] J;
#endif

   generated_transfer.reset(P_ft->Transpose());
   SetFilteredSubspaceTransferOperator(*generated_transfer);

   // The subspace size just changed (or this is the first time), so a fixed
   // subspace solver generally cannot be reused: most direct solvers assume
   // a fixed operator size for their lifetime. Build a fresh one via the
   // factory whenever the (global) subspace size differs from the one the
   // current solver was built for.
   if (auto_subspace_solver_factory &&
       (!auto_subspace_solver || auto_subspace_width != glob_nrows_f))
   {
      auto_subspace_solver = auto_subspace_solver_factory();
      MFEM_VERIFY(auto_subspace_solver,
                  "AMGFSolver::GenerateFilteredSubspaceTransferOperator: "
                  "auto_subspace_solver_factory returned a null solver.");
      SetFilteredSubspaceSolver(*auto_subspace_solver);
      auto_subspace_width = glob_nrows_f;
   }

   filtering_enabled = true;
   return true;
}

void AMGFSolver::Mult(const Vector &b, Vector &x) const
{
   if (!filtering_enabled)
   {
      // No filtered subspace: fall back to a single application of AMG.
      if (!solver_set)
      {
         B->SetOperator(*A);
         solver_set = true;
      }
      B->Mult(b, x);
      return;
   }
   FilteredSolver::Mult(b, x);
}

#endif

} // namespace mfem
