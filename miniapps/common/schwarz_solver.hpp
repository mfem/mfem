// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_SCHWARZ_SOLVER_HPP
#define MFEM_SCHWARZ_SOLVER_HPP

#include "mfem.hpp"
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <unordered_map>

using namespace mfem;

// HYPRE and LAPACK declarations
extern "C"
{
#include "HYPRE.h"
#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.h"

void dpotrf_(const char *uplo, const int *n, double *a, const int *lda, int *info);
void dgetrf_(const int *m, const int *n, double *a, const int *lda, int *ipiv, int *info);
void dpotrs_(char *uplo, int *n, int *nrhs, double *a, int *lda,
             double *b, int *ldb, int *info);
void dgetrs_(const char *trans, int *n, int *nrhs, double *a, int *lda,
             int *ipiv, double *b, int *ldb, int *info);

typedef struct
{
   HYPRE_Int      variant;
   HYPRE_Int      domain_type;
   HYPRE_Int      overlap;
   HYPRE_Int      num_functions;
   HYPRE_Int      use_nonsymm;
   HYPRE_Real   relax_weight;

   hypre_CSRMatrix *domain_structure;
   hypre_CSRMatrix *A_boundary;
   hypre_ParVector *Vtemp;
   HYPRE_Real  *scale;
   HYPRE_Int     *dof_func;
   HYPRE_Int     *pivots;

} hypre_SchwarzData;
}

// Dummy setup function for HYPRE Schwarz solver
static HYPRE_Int DummyParSolverFcn(HYPRE_Solver solver,
                                   HYPRE_ParCSRMatrix A,
                                   HYPRE_ParVector b,
                                   HYPRE_ParVector x)
{
   (void)solver;
   (void)A;
   (void)b;
   (void)x;
   return 0;
}

/// Schwarz domain decomposition solver wrapping HYPRE's Schwarz implementation
class HypreSchwarz : public HypreSolver
{
public:
   static constexpr HYPRE_Int RequiredVariant = 2;
   HYPRE_Solver schwarz_solver;
   int print_level = 0;
   MPI_Comm custom_comm = MPI_COMM_NULL;
   HYPRE_BigInt custom_local_row_start = 0;
   HYPRE_Int custom_local_num_rows = 0;
   bool custom_use_nonsymm = false;
   bool use_custom_solve = false;
   std::vector<std::vector<HYPRE_BigInt>> custom_subdomains;
   std::vector<HYPRE_Int> custom_dense_offsets;
   std::vector<HYPRE_Int> custom_pivot_offsets;
   std::vector<HYPRE_Real> custom_dense_data;
   std::vector<HYPRE_Int> custom_pivots;
   std::unordered_map<HYPRE_BigInt, HYPRE_Real> custom_scale;
   std::vector<int> rhs_request_send_counts;
   std::vector<int> rhs_request_send_displs;
   std::vector<int> rhs_request_recv_counts;
   std::vector<int> rhs_request_recv_displs;
   std::vector<HYPRE_BigInt> rhs_request_send_ids;
   std::vector<HYPRE_BigInt> rhs_request_recv_ids;
   std::unordered_map<HYPRE_BigInt, HYPRE_Int> rhs_value_index;
   std::vector<int> correction_send_counts;
   std::vector<int> correction_send_displs;
   std::vector<int> correction_recv_counts;
   std::vector<int> correction_recv_displs;
   std::vector<HYPRE_BigInt> correction_send_ids;
   std::vector<HYPRE_BigInt> correction_recv_ids;

   static std::unordered_map<HYPRE_Solver, int> & PrintLevels()
   {
      static std::unordered_map<HYPRE_Solver, int> levels;
      return levels;
   }

   static std::unordered_map<HYPRE_Solver, HypreSchwarz *> & Instances()
   {
      static std::unordered_map<HYPRE_Solver, HypreSchwarz *> instances;
      return instances;
   }

   void SetPrintLevel(int level)
   {
      print_level = level;
      if (schwarz_solver)
      {
         PrintLevels()[schwarz_solver] = level;
         Instances()[schwarz_solver] = this;
      }
   }

   void SetOperator(const Operator &op) override
   {
      const HypreParMatrix *new_A = dynamic_cast<const HypreParMatrix *>(&op);
      MFEM_VERIFY(new_A, "new Operator must be a HypreParMatrix!");

      //HYPRE_SchwarzDestroy(schwarz_solver); TODO: fix leak :)
      //HYPRE_SchwarzCreate(&schwarz_solver);

      // update base classes: Operator, Solver, HypreSolver
      height = new_A->Height();
      width  = new_A->Width();
      A = const_cast<HypreParMatrix *>(new_A);
      setup_called = 0;
      delete X;
      delete B;
      B = X = NULL;
      auxB.Delete(); auxB.Reset();
      auxX.Delete(); auxX.Reset();

      if (schwarz_solver)
      {
         PrintLevels()[schwarz_solver] = print_level;
         Instances()[schwarz_solver] = this;
      }
   }

   HYPRE_CSRMatrix BuildDomainStructure(
      const std::vector<std::vector<HYPRE_Int>>& subdomains)
   {
      const HYPRE_Int num_domains = static_cast<HYPRE_Int>(subdomains.size());

      HYPRE_Int max_domain_size = 0;
      HYPRE_Int total_memberships = 0;
      HYPRE_Int total_dense_data = 0;

      for (const auto& domain : subdomains)
      {
         const HYPRE_Int sz = static_cast<HYPRE_Int>(domain.size());
         max_domain_size = std::max(max_domain_size, sz);
         total_memberships += sz;
         total_dense_data += sz * sz;
      }

      hypre_CSRMatrix* csr =
         hypre_CSRMatrixCreate(num_domains, max_domain_size, total_memberships);

      hypre_CSRMatrixInitialize(csr);

      HYPRE_Int* I = hypre_CSRMatrixI(csr);
      HYPRE_Int* J = hypre_CSRMatrixJ(csr);

      HYPRE_Int nnz_counter = 0;
      I[0] = 0;

      for (HYPRE_Int i = 0; i < num_domains; i++)
      {
         const auto& domain = subdomains[i];
         for (HYPRE_Int k = 0; k < static_cast<HYPRE_Int>(domain.size()); k++)
         {
            J[nnz_counter++] = domain[k];
         }
         I[i + 1] = nnz_counter;
      }

      HYPRE_Real* data =
         hypre_CTAlloc(HYPRE_Real, total_dense_data, HYPRE_MEMORY_HOST);

      for (HYPRE_Int i = 0; i < total_dense_data; i++)
      {
         data[i] = 0.0;
      }

      hypre_CSRMatrixData(csr) = data;

      return (HYPRE_CSRMatrix) csr;
   }

   HYPRE_CSRMatrix domain_structure = nullptr;
   HYPRE_Int *pivots = nullptr;
   HYPRE_Real *scale = nullptr;

   HYPRE_Real LookupLocalEntry(HYPRE_BigInt gi, HYPRE_BigInt gj,
                               HYPRE_BigInt local_row_start,
                               HYPRE_BigInt local_row_end,
                               HYPRE_Int *Ai, HYPRE_Int *Aj, HYPRE_Real *Aa,
                               HYPRE_Int *Ai_offd, HYPRE_Int *Aj_offd,
                               HYPRE_Real *Aa_offd,
                               HYPRE_BigInt *col_map_offd) const
   {
      if (gi < local_row_start || gi >= local_row_end)
      {
         return 0.0;
      }

      const HYPRE_Int local_i = static_cast<HYPRE_Int>(gi - local_row_start);
      if (gj >= local_row_start && gj < local_row_end)
      {
         const HYPRE_Int local_j = static_cast<HYPRE_Int>(gj - local_row_start);
         for (HYPRE_Int jj = Ai[local_i]; jj < Ai[local_i + 1]; ++jj)
         {
            if (Aj[jj] == local_j)
            {
               return Aa[jj];
            }
         }
         return 0.0;
      }

      for (HYPRE_Int jj = Ai_offd[local_i]; jj < Ai_offd[local_i + 1]; ++jj)
      {
         if (col_map_offd[Aj_offd[jj]] == gj)
         {
            return Aa_offd[jj];
         }
      }
      return 0.0;
   }

   void BuildCustomScale(const std::vector<std::vector<HYPRE_BigInt>> &subdomains,
                         MPI_Comm comm,
                         HYPRE_Real relax_weight,
                         bool unweighted,
                         HYPRE_Real uniform_weight)
   {
      custom_scale.clear();

      if (uniform_weight >= 0.0)
      {
         for (const auto &subdomain : subdomains)
         {
            for (HYPRE_BigInt gdof : subdomain)
            {
               custom_scale[gdof] = uniform_weight;
            }
         }
         return;
      }

      if (unweighted)
      {
         for (const auto &subdomain : subdomains)
         {
            for (HYPRE_BigInt gdof : subdomain)
            {
               custom_scale[gdof] = 1.0;
            }
         }
         return;
      }

      int nranks = 1;
      MPI_Comm_size(comm, &nranks);
      int local_memberships = 0;
      for (const auto &subdomain : subdomains)
      {
         local_memberships += static_cast<int>(subdomain.size());
      }

      std::vector<int> membership_counts(nranks, 0);
      MPI_Allgather(&local_memberships, 1, MPI_INT,
                    membership_counts.data(), 1, MPI_INT, comm);

      std::vector<int> membership_displs(nranks + 1, 0);
      for (int r = 0; r < nranks; ++r)
      {
         membership_displs[r + 1] = membership_displs[r] + membership_counts[r];
      }

      std::vector<HYPRE_BigInt> local_flat_subdomains;
      local_flat_subdomains.reserve(local_memberships);
      for (const auto &subdomain : subdomains)
      {
         local_flat_subdomains.insert(local_flat_subdomains.end(),
                                      subdomain.begin(), subdomain.end());
      }

      std::vector<HYPRE_BigInt> all_flat_subdomains(membership_displs[nranks]);
      MPI_Allgatherv(local_flat_subdomains.empty() ? nullptr : local_flat_subdomains.data(),
                     local_memberships, HYPRE_MPI_BIG_INT,
                     all_flat_subdomains.empty() ? nullptr : all_flat_subdomains.data(),
                     membership_counts.data(), membership_displs.data(),
                     HYPRE_MPI_BIG_INT, comm);

      std::unordered_map<HYPRE_BigInt, int> multiplicity;
      multiplicity.reserve(all_flat_subdomains.size());
      for (HYPRE_BigInt gdof : all_flat_subdomains)
      {
         multiplicity[gdof]++;
      }

      custom_scale.reserve(multiplicity.size());
      for (const auto &entry : multiplicity)
      {
         custom_scale[entry.first] = relax_weight / entry.second;
      }
   }

   HYPRE_Int ApplyCustomSolve(HYPRE_ParCSRMatrix A,
                              HYPRE_ParVector b,
                              HYPRE_ParVector x) const
   {
      (void)A;

      hypre_Vector *b_local = hypre_ParVectorLocalVector((hypre_ParVector *)b);
      hypre_Vector *x_local = hypre_ParVectorLocalVector((hypre_ParVector *)x);
      HYPRE_Real *b_data = hypre_VectorData(b_local);
      HYPRE_Real *x_data = hypre_VectorData(x_local);

      std::vector<HYPRE_Real> rhs_response_values(rhs_request_recv_ids.size(), 0.0);
      for (size_t i = 0; i < rhs_request_recv_ids.size(); ++i)
      {
         const HYPRE_BigInt gdof = rhs_request_recv_ids[i];
         MFEM_VERIFY(gdof >= custom_local_row_start &&
                     gdof < custom_local_row_start + custom_local_num_rows,
                     "Requested RHS DOF is not locally owned.");
         rhs_response_values[i] = b_data[gdof - custom_local_row_start];
      }

      std::vector<HYPRE_Real> remote_rhs_values(rhs_request_send_ids.size(), 0.0);
      MPI_Alltoallv(rhs_response_values.empty() ? nullptr : rhs_response_values.data(),
                    rhs_request_recv_counts.data(), rhs_request_recv_displs.data(),
                    HYPRE_MPI_REAL,
                    remote_rhs_values.empty() ? nullptr : remote_rhs_values.data(),
                    rhs_request_send_counts.data(), rhs_request_send_displs.data(),
                    HYPRE_MPI_REAL, custom_comm);

      std::vector<HYPRE_Real> local_correction(custom_local_num_rows, 0.0);
      std::unordered_map<HYPRE_BigInt, HYPRE_Real> remote_correction;
      remote_correction.reserve(correction_send_ids.size());
      char uplo = 'L';
      const char trans = 'N';
      int nrhs = 1;

      for (size_t d = 0; d < custom_subdomains.size(); ++d)
      {
         const int n = static_cast<int>(custom_subdomains[d].size());
         if (n == 0) { continue; }

         std::vector<HYPRE_Real> rhs(n, 0.0);
         for (int i = 0; i < n; ++i)
         {
            const HYPRE_BigInt gdof = custom_subdomains[d][i];
            if (gdof >= custom_local_row_start &&
                gdof < custom_local_row_start + custom_local_num_rows)
            {
               rhs[i] = b_data[gdof - custom_local_row_start];
            }
            else
            {
               auto it = rhs_value_index.find(gdof);
               MFEM_VERIFY(it != rhs_value_index.end(),
                           "Missing remote RHS value for Schwarz subdomain DOF.");
               rhs[i] = remote_rhs_values[it->second];
            }
         }

         int info = 0;
         int nn = n;
         int ldb = n;
         HYPRE_Real *factor = const_cast<HYPRE_Real *>(&custom_dense_data[custom_dense_offsets[d]]);
         if (custom_use_nonsymm)
         {
            HYPRE_Int *pivot = const_cast<HYPRE_Int *>(&custom_pivots[custom_pivot_offsets[d]]);
            dgetrs_(&trans, &nn, &nrhs, factor, &nn,
                    reinterpret_cast<int *>(pivot), rhs.data(), &ldb, &info);
            MFEM_VERIFY(info == 0, "LU backsolve failed for Schwarz subdomain.");
         }
         else
         {
            dpotrs_(&uplo, &nn, &nrhs, factor, &nn,
                    rhs.data(), &ldb, &info);
            MFEM_VERIFY(info == 0, "Cholesky backsolve failed for Schwarz subdomain.");
         }

         for (int i = 0; i < n; ++i)
         {
            const HYPRE_BigInt gdof = custom_subdomains[d][i];
            const auto it = custom_scale.find(gdof);
            const HYPRE_Real dof_scale = (it != custom_scale.end()) ? it->second : 1.0;
            const HYPRE_Real contribution = dof_scale * rhs[i];
            if (gdof >= custom_local_row_start &&
                gdof < custom_local_row_start + custom_local_num_rows)
            {
               local_correction[gdof - custom_local_row_start] += contribution;
            }
            else
            {
               remote_correction[gdof] += contribution;
            }
         }
      }

      std::vector<HYPRE_Real> correction_send_values(correction_send_ids.size(), 0.0);
      for (size_t i = 0; i < correction_send_ids.size(); ++i)
      {
         auto it = remote_correction.find(correction_send_ids[i]);
         if (it != remote_correction.end())
         {
            correction_send_values[i] = it->second;
         }
      }

      std::vector<HYPRE_Real> correction_recv_values(correction_recv_ids.size(), 0.0);
      MPI_Alltoallv(correction_send_values.empty() ? nullptr : correction_send_values.data(),
                    correction_send_counts.data(), correction_send_displs.data(),
                    HYPRE_MPI_REAL,
                    correction_recv_values.empty() ? nullptr : correction_recv_values.data(),
                    correction_recv_counts.data(), correction_recv_displs.data(),
                    HYPRE_MPI_REAL, custom_comm);

      for (HYPRE_Int i = 0; i < custom_local_num_rows; ++i)
      {
         x_data[i] = local_correction[i];
      }

      for (size_t i = 0; i < correction_recv_ids.size(); ++i)
      {
         const HYPRE_BigInt gdof = correction_recv_ids[i];
         MFEM_VERIFY(gdof >= custom_local_row_start &&
                     gdof < custom_local_row_start + custom_local_num_rows,
                     "Received Schwarz correction for non-local DOF.");
         x_data[gdof - custom_local_row_start] += correction_recv_values[i];
      }

      return 0;
   }

   void SetCustomSubdomains(const std::vector<std::vector<HYPRE_BigInt>> &subdomains,
                            HypreParMatrix &A,
                            HYPRE_Real relax_weight = 1.0,
                            HYPRE_Int use_nonsymm = 0,
                            bool unweighted = false,
                            HYPRE_Real uniform_weight = -1.0)
   {
      hypre_SchwarzData *sd = (hypre_SchwarzData *)schwarz_solver;
      MFEM_VERIFY(sd != nullptr, "Invalid Schwarz solver.");
      if (sd->variant != RequiredVariant)
      {
         int myrank = 0;
         MPI_Comm_rank(hypre_ParCSRMatrixComm((hypre_ParCSRMatrix *)A), &myrank);
         if (myrank == 0)
         {
            mfem::out << "HypreSchwarz: overriding Schwarz variant " << sd->variant
                      << " with required variant " << RequiredVariant << std::endl;
         }
         HYPRE_SchwarzSetVariant(schwarz_solver, RequiredVariant);
      }
      sd->relax_weight = relax_weight;
      sd->use_nonsymm = use_nonsymm;

      hypre_ParCSRMatrix *parA = (hypre_ParCSRMatrix *)A;
      hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(parA);
      MFEM_VERIFY(A_diag != nullptr, "A_diag is null.");

      HYPRE_Int num_dofs = hypre_CSRMatrixNumRows(A_diag);

      // STEP 4a: Add MPI context and gather row partition
      MPI_Comm comm = hypre_ParCSRMatrixComm(parA);
      int myrank, nranks;
      MPI_Comm_rank(comm, &myrank);
      MPI_Comm_size(comm, &nranks);

      // Get row partition for all ranks
      // With assumed partition: row_starts is size 2 [local_start, local_end]
      // Without: row_starts is size nranks+1 with all boundaries
      const HYPRE_BigInt *row_starts = hypre_ParCSRMatrixRowStarts(parA);
      const HYPRE_BigInt local_row_start = row_starts[0];
      const HYPRE_BigInt local_row_end = row_starts[1];
      const HYPRE_Int local_num_rows = local_row_end - local_row_start;

      MFEM_VERIFY(num_dofs == local_num_rows,
                  "A_diag size does not match local row count");

      // Gather all row partitions
      std::vector<HYPRE_BigInt> all_row_starts(nranks);
      std::vector<HYPRE_BigInt> all_row_ends(nranks);
      MPI_Allgather(&local_row_start, 1, HYPRE_MPI_BIG_INT,
                    all_row_starts.data(), 1, HYPRE_MPI_BIG_INT, comm);
      MPI_Allgather(&local_row_end, 1, HYPRE_MPI_BIG_INT,
                    all_row_ends.data(), 1, HYPRE_MPI_BIG_INT, comm);

      // Access diagonal and off-diagonal blocks
      hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(parA);
      HYPRE_BigInt *col_map_offd = hypre_ParCSRMatrixColMapOffd(parA);
      HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(A_offd);

      HYPRE_Int *Ai_offd = hypre_CSRMatrixI(A_offd);
      HYPRE_Int *Aj_offd = hypre_CSRMatrixJ(A_offd);
      HYPRE_Real *Aa_offd = hypre_CSRMatrixData(A_offd);

      // Build mapping from global PTAP index to extended local index
      // Extended indexing: [0, num_dofs) = local, [num_dofs, num_dofs+num_cols_offd) = ghost
      std::map<HYPRE_BigInt, HYPRE_Int> global_to_extended_local;

      // Map local DOFs
      for (HYPRE_Int i = 0; i < num_dofs; ++i) {
         global_to_extended_local[local_row_start + i] = i;
      }

      // Map ghost DOFs (from A_offd's column map)
      for (HYPRE_Int j = 0; j < num_cols_offd; ++j) {
         global_to_extended_local[col_map_offd[j]] = num_dofs + j;
      }

      // STEP 4a VERIFICATION
      if (myrank == 0 && print_level > 1)
      {
         mfem::out << "SetCustomSubdomains: Processing " << subdomains.size()
                   << " local subdomains, " << num_dofs << " local DOFs, "
                   << num_cols_offd << " ghost DOFs" << std::endl;
      }

      if (domain_structure)
      {
         hypre_CSRMatrixDestroy((hypre_CSRMatrix *)domain_structure);
         domain_structure = nullptr;
      }
      if (pivots)
      {
         hypre_TFree(pivots, HYPRE_MEMORY_HOST);
         pivots = nullptr;
      }
      if (scale)
      {
         hypre_TFree(scale, HYPRE_MEMORY_HOST);
         scale = nullptr;
      }

      if (sd->Vtemp)
      {
         hypre_ParVectorDestroy(sd->Vtemp);
         sd->Vtemp = nullptr;
      }

      if (sd->A_boundary)
      {
         hypre_CSRMatrixDestroy(sd->A_boundary);
         sd->A_boundary = nullptr;
      }

      {
         hypre_ParVector *Vtemp =
            hypre_ParVectorCreate(hypre_ParCSRMatrixComm(parA),
                                  hypre_ParCSRMatrixGlobalNumRows(parA),
                                  hypre_ParCSRMatrixRowStarts(parA));
         hypre_ParVectorInitialize(Vtemp);
         sd->Vtemp = Vtemp;
      }

      const HYPRE_Int num_domains = (HYPRE_Int)subdomains.size();
      HYPRE_Int total_pivots = 0;
      size_t total_dense_data_sz = 0;
      for (const auto &dom : subdomains)
      {
         const HYPRE_Int sz = (HYPRE_Int)dom.size();
         MFEM_VERIFY(sz >= 0, "Invalid subdomain size.");
         total_pivots += sz;
         total_dense_data_sz += (size_t)sz * (size_t)sz;
      }

      MFEM_VERIFY(total_dense_data_sz <= (size_t)std::numeric_limits<HYPRE_Int>::max(),
                  "Dense subdomain storage exceeds HYPRE_Int capacity.");

      custom_comm = comm;
      custom_local_row_start = local_row_start;
      custom_local_num_rows = local_num_rows;
      custom_use_nonsymm = (use_nonsymm != 0);
      use_custom_solve = true;
      custom_subdomains = subdomains;
      custom_dense_offsets.assign(num_domains + 1, 0);
      custom_pivot_offsets.assign(num_domains + 1, 0);
      for (HYPRE_Int d = 0; d < num_domains; ++d)
      {
         const HYPRE_Int sz = (HYPRE_Int)custom_subdomains[d].size();
         custom_dense_offsets[d + 1] = custom_dense_offsets[d] + sz * sz;
         custom_pivot_offsets[d + 1] = custom_pivot_offsets[d] + sz;
      }
      custom_dense_data.assign(custom_dense_offsets[num_domains], 0.0);
      custom_pivots.assign(custom_use_nonsymm ? custom_pivot_offsets[num_domains] : 0, 0);

      auto owner_rank = [&](HYPRE_BigInt gdof) -> int
      {
         for (int r = 0; r < nranks; ++r)
         {
            if (gdof >= all_row_starts[r] && gdof < all_row_ends[r])
            {
               return r;
            }
         }
         return -1;
      };

      std::vector<std::set<HYPRE_BigInt>> rhs_request_sets(nranks);
      std::vector<std::set<HYPRE_BigInt>> correction_send_sets(nranks);
      for (const auto &domain : custom_subdomains)
      {
         for (HYPRE_BigInt gdof : domain)
         {
            const int owner = owner_rank(gdof);
            MFEM_VERIFY(owner >= 0, "Failed to determine owner of Schwarz DOF.");
            if (owner != myrank)
            {
               rhs_request_sets[owner].insert(gdof);
               correction_send_sets[owner].insert(gdof);
            }
         }
      }

      rhs_request_send_counts.assign(nranks, 0);
      rhs_request_send_displs.assign(nranks + 1, 0);
      for (int r = 0; r < nranks; ++r)
      {
         rhs_request_send_counts[r] = static_cast<int>(rhs_request_sets[r].size());
         rhs_request_send_displs[r + 1] = rhs_request_send_displs[r] + rhs_request_send_counts[r];
      }
      rhs_request_send_ids.resize(rhs_request_send_displs[nranks]);
      int rhs_idx = 0;
      for (int r = 0; r < nranks; ++r)
      {
         for (HYPRE_BigInt gdof : rhs_request_sets[r])
         {
            rhs_request_send_ids[rhs_idx++] = gdof;
         }
      }
      rhs_request_recv_counts.assign(nranks, 0);
      rhs_request_recv_displs.assign(nranks + 1, 0);
      MPI_Alltoall(rhs_request_send_counts.data(), 1, MPI_INT,
                   rhs_request_recv_counts.data(), 1, MPI_INT, comm);
      for (int r = 0; r < nranks; ++r)
      {
         rhs_request_recv_displs[r + 1] = rhs_request_recv_displs[r] + rhs_request_recv_counts[r];
      }
      rhs_request_recv_ids.resize(rhs_request_recv_displs[nranks]);
      MPI_Alltoallv(rhs_request_send_ids.empty() ? nullptr : rhs_request_send_ids.data(),
                    rhs_request_send_counts.data(), rhs_request_send_displs.data(),
                    HYPRE_MPI_BIG_INT,
                    rhs_request_recv_ids.empty() ? nullptr : rhs_request_recv_ids.data(),
                    rhs_request_recv_counts.data(), rhs_request_recv_displs.data(),
                    HYPRE_MPI_BIG_INT, comm);
      rhs_value_index.clear();
      rhs_value_index.reserve(rhs_request_send_ids.size());
      for (size_t i = 0; i < rhs_request_send_ids.size(); ++i)
      {
         rhs_value_index[rhs_request_send_ids[i]] = static_cast<HYPRE_Int>(i);
      }

      correction_send_counts.assign(nranks, 0);
      correction_send_displs.assign(nranks + 1, 0);
      for (int r = 0; r < nranks; ++r)
      {
         correction_send_counts[r] = static_cast<int>(correction_send_sets[r].size());
         correction_send_displs[r + 1] =
            correction_send_displs[r] + correction_send_counts[r];
      }
      correction_send_ids.resize(correction_send_displs[nranks]);
      int corr_idx = 0;
      for (int r = 0; r < nranks; ++r)
      {
         for (HYPRE_BigInt gdof : correction_send_sets[r])
         {
            correction_send_ids[corr_idx++] = gdof;
         }
      }
      correction_recv_counts.assign(nranks, 0);
      correction_recv_displs.assign(nranks + 1, 0);
      MPI_Alltoall(correction_send_counts.data(), 1, MPI_INT,
                   correction_recv_counts.data(), 1, MPI_INT, comm);
      for (int r = 0; r < nranks; ++r)
      {
         correction_recv_displs[r + 1] =
            correction_recv_displs[r] + correction_recv_counts[r];
      }
      correction_recv_ids.resize(correction_recv_displs[nranks]);
      MPI_Alltoallv(correction_send_ids.empty() ? nullptr : correction_send_ids.data(),
                    correction_send_counts.data(), correction_send_displs.data(),
                    HYPRE_MPI_BIG_INT,
                    correction_recv_ids.empty() ? nullptr : correction_recv_ids.data(),
                    correction_recv_counts.data(), correction_recv_displs.data(),
                    HYPRE_MPI_BIG_INT, comm);

      BuildCustomScale(subdomains, comm, relax_weight, unweighted, uniform_weight);
      Instances()[schwarz_solver] = this;

      HYPRE_Int *Ai = hypre_CSRMatrixI(A_diag);
      HYPRE_Int *Aj = hypre_CSRMatrixJ(A_diag);
      HYPRE_Real *Aa = hypre_CSRMatrixData(A_diag);

      struct MatrixEntryRequest
      {
         HYPRE_BigInt row;
         HYPRE_BigInt col;
      };

      std::vector<std::vector<MatrixEntryRequest>> requests_to_send(nranks);
      for (const auto &domain : custom_subdomains)
      {
         for (HYPRE_BigInt gi : domain)
         {
            if (gi >= local_row_start && gi < local_row_end)
            {
               continue;
            }

            int owner_rank = -1;
            for (int r = 0; r < nranks; ++r)
            {
               if (gi >= all_row_starts[r] && gi < all_row_ends[r])
               {
                  owner_rank = r;
                  break;
               }
            }

            MFEM_VERIFY(owner_rank >= 0, "Failed to determine owner of remote subdomain row.");
            for (HYPRE_BigInt gj : domain)
            {
               requests_to_send[owner_rank].push_back({gi, gj});
            }
         }
      }

      std::vector<int> send_counts(nranks, 0), recv_counts(nranks, 0);
      for (int r = 0; r < nranks; ++r)
      {
         send_counts[r] = 2 * static_cast<int>(requests_to_send[r].size());
      }
      MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);

      std::vector<int> send_displs(nranks + 1, 0), recv_displs(nranks + 1, 0);
      for (int r = 0; r < nranks; ++r)
      {
         send_displs[r + 1] = send_displs[r] + send_counts[r];
         recv_displs[r + 1] = recv_displs[r] + recv_counts[r];
      }

      std::vector<HYPRE_BigInt> send_buffer(send_displs[nranks]);
      int idx = 0;
      for (int r = 0; r < nranks; ++r)
      {
         for (const auto &req : requests_to_send[r])
         {
            send_buffer[idx++] = req.row;
            send_buffer[idx++] = req.col;
         }
      }

      std::vector<HYPRE_BigInt> recv_buffer(recv_displs[nranks]);
      MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(),
                    HYPRE_MPI_BIG_INT, recv_buffer.data(), recv_counts.data(),
                    recv_displs.data(), HYPRE_MPI_BIG_INT, comm);

      std::vector<HYPRE_Real> response_values(recv_buffer.size() / 2, 0.0);
      for (size_t i = 0; i < response_values.size(); ++i)
      {
         response_values[i] = LookupLocalEntry(recv_buffer[2 * i], recv_buffer[2 * i + 1],
                                               local_row_start, local_row_end,
                                               Ai, Aj, Aa,
                                               Ai_offd, Aj_offd, Aa_offd,
                                               col_map_offd);
      }

      std::vector<int> send_value_counts(nranks, 0), recv_value_counts(nranks, 0);
      std::vector<int> send_value_displs(nranks + 1, 0), recv_value_displs(nranks + 1, 0);
      for (int r = 0; r < nranks; ++r)
      {
         send_value_counts[r] = recv_counts[r] / 2;
         recv_value_counts[r] = send_counts[r] / 2;
         send_value_displs[r + 1] = send_value_displs[r] + send_value_counts[r];
         recv_value_displs[r + 1] = recv_value_displs[r] + recv_value_counts[r];
      }

      std::vector<HYPRE_Real> recv_values(send_buffer.size() / 2, 0.0);
      MPI_Alltoallv(response_values.data(), send_value_counts.data(),
                    send_value_displs.data(), HYPRE_MPI_REAL,
                    recv_values.data(), recv_value_counts.data(),
                    recv_value_displs.data(), HYPRE_MPI_REAL, comm);

      std::map<std::pair<HYPRE_BigInt, HYPRE_BigInt>, HYPRE_Real> remote_entries;
      idx = 0;
      for (int r = 0; r < nranks; ++r)
      {
         for (const auto &req : requests_to_send[r])
         {
            remote_entries[{req.row, req.col}] = recv_values[idx++];
         }
      }

      char uplo = 'L';
      for (HYPRE_Int d = 0; d < num_domains; ++d)
      {
         const HYPRE_Int n = (HYPRE_Int)custom_subdomains[d].size();
         HYPRE_Real *AE = &custom_dense_data[custom_dense_offsets[d]];
         std::fill(AE, AE + (size_t)n * (size_t)n, 0.0);

         for (HYPRE_Int i = 0; i < n; ++i)
         {
            const HYPRE_BigInt gi = custom_subdomains[d][i];
            for (HYPRE_Int j = 0; j < n; ++j)
            {
               const HYPRE_BigInt gj = custom_subdomains[d][j];
               if (gi >= local_row_start && gi < local_row_end)
               {
                  AE[i + j * n] = LookupLocalEntry(gi, gj, local_row_start, local_row_end,
                                                   Ai, Aj, Aa, Ai_offd, Aj_offd,
                                                   Aa_offd, col_map_offd);
               }
               else
               {
                  auto it = remote_entries.find({gi, gj});
                  if (it != remote_entries.end())
                  {
                     AE[i + j * n] = it->second;
                  }
               }
            }
         }

         if (n > 0)
         {
            int info = 0;
            int nn = (int)n;
            if (custom_use_nonsymm)
            {
               dgetrf_(&nn, &nn, AE, &nn,
                       reinterpret_cast<int *>(&custom_pivots[custom_pivot_offsets[d]]), &info);
               MFEM_VERIFY(info == 0, "LU factorization failed for Schwarz subdomain.");
            }
            else
            {
               dpotrf_(&uplo, &nn, AE, &nn, &info);
               MFEM_VERIFY(info == 0, "Cholesky factorization failed for Schwarz subdomain.");
            }
         }
      }

      sd->domain_structure = nullptr;
      sd->pivots = nullptr;
      sd->scale = nullptr;
      sd->A_boundary = nullptr;
   }

   operator HYPRE_Solver() const override { return schwarz_solver; }
   HYPRE_PtrToParSolverFcn SetupFcn() const override
   { return (HYPRE_PtrToParSolverFcn) DummyParSolverFcn; }
   HYPRE_PtrToParSolverFcn SolveFcn() const override
   {
      static auto wrapped_solve = [](HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                                      HYPRE_ParVector b, HYPRE_ParVector x) -> HYPRE_Int
      {
         HypreSchwarz *instance =
            Instances().count(solver) ? Instances()[solver] : nullptr;
         const int print_level =
            PrintLevels().count(solver) ? PrintLevels()[solver] : 0;
         const double solve_start = MPI_Wtime();
         HYPRE_Int result =
            (instance && instance->use_custom_solve) ?
            instance->ApplyCustomSolve(A, b, x) :
            HYPRE_SchwarzSolve(solver, A, b, x);
         const double solve_elapsed_local = MPI_Wtime() - solve_start;
         double solve_elapsed = 0.0;
         MPI_Comm comm = hypre_ParCSRMatrixComm((hypre_ParCSRMatrix*)A);
         MPI_Allreduce(&solve_elapsed_local, &solve_elapsed, 1, MPI_DOUBLE, MPI_MAX, comm);
         int myrank = 0;
         MPI_Comm_rank(comm, &myrank);
         if (myrank == 0 && print_level > 0)
         {
            mfem::out << "HypreSchwarz solve time [s]: " << solve_elapsed << std::endl;
         }

         // Compute residual: r = b - A*x
         hypre_ParVector *residual = hypre_ParVectorCreate(
            hypre_ParCSRMatrixComm((hypre_ParCSRMatrix*)A),
            hypre_ParCSRMatrixGlobalNumRows((hypre_ParCSRMatrix*)A),
            hypre_ParCSRMatrixRowStarts((hypre_ParCSRMatrix*)A));
         hypre_ParVectorInitialize(residual);
         hypre_ParVectorCopy((hypre_ParVector*)b, residual);
         hypre_ParCSRMatrixMatvec(-1.0, (hypre_ParCSRMatrix*)A,
                                  (hypre_ParVector*)x, 1.0, residual);

         HYPRE_Real residual_norm = hypre_ParVectorInnerProd(residual, residual);
         residual_norm = std::sqrt(residual_norm);

         HYPRE_Real b_norm = hypre_ParVectorInnerProd((hypre_ParVector*)b, (hypre_ParVector*)b);
         b_norm = std::sqrt(b_norm);

         if (myrank == 0 && print_level > 1)
         {
            mfem::out << "HypreSchwarz residual norm: " << residual_norm
                      << ", RHS norm: " << b_norm << std::endl;
         }

         hypre_ParVectorDestroy(residual);
         return result;
      };
      return (HYPRE_PtrToParSolverFcn) +wrapped_solve;
   }
   using HypreSolver::Mult;
};

#endif // MFEM_SCHWARZ_SOLVER_HPP
