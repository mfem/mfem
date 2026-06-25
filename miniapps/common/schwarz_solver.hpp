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

using namespace mfem;

// HYPRE and LAPACK declarations
extern "C"
{
#include "HYPRE.h"
#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.h"

void dpotrf_(const char *uplo, const int *n, double *a, const int *lda, int *info);
void dgetrf_(const int *m, const int *n, double *a, const int *lda, int *ipiv, int *info);

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
   HYPRE_Solver schwarz_solver;

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

   void SetCustomSubdomains(const std::vector<std::vector<HYPRE_Int>> &subdomains,
                            HypreParMatrix &A,
                            HYPRE_Real relax_weight = 1.0,
                            HYPRE_Int use_nonsymm = 0,
                            bool unweighted = false,
                            HYPRE_Real uniform_weight = -1.0)
   {
      hypre_SchwarzData *sd = (hypre_SchwarzData *)schwarz_solver;
      sd->relax_weight = relax_weight;
      sd->use_nonsymm = use_nonsymm;
      MFEM_VERIFY(sd != nullptr, "Invalid Schwarz solver.");

      hypre_ParCSRMatrix *parA = (hypre_ParCSRMatrix *)A;
      hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(parA);
      MFEM_VERIFY(A_diag != nullptr, "A_diag is null.");

      HYPRE_Int num_dofs = hypre_CSRMatrixNumRows(A_diag);

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
      HYPRE_Int max_domain_size = 0;
      HYPRE_Int total_memberships = 0;
      size_t total_dense_data_sz = 0;
      HYPRE_Int total_pivots = 0;

      for (const auto &dom : subdomains)
      {
         HYPRE_Int sz = (HYPRE_Int)dom.size();
         MFEM_VERIFY(sz >= 0, "Invalid subdomain size.");
         max_domain_size = std::max(max_domain_size, sz);
         total_memberships += sz;
         total_pivots += sz;

         size_t ssz = (size_t)sz;
         total_dense_data_sz += ssz * ssz;
      }

      MFEM_VERIFY(total_dense_data_sz <= (size_t)std::numeric_limits<HYPRE_Int>::max(),
                  "Dense subdomain storage exceeds HYPRE_Int capacity.");

      HYPRE_Int total_dense_data = (HYPRE_Int)total_dense_data_sz;

      hypre_CSRMatrix *csr =
         hypre_CSRMatrixCreate(num_domains, num_dofs, total_memberships);
      hypre_CSRMatrixInitialize(csr);

      HYPRE_Int *I = hypre_CSRMatrixI(csr);
      HYPRE_Int *J = hypre_CSRMatrixJ(csr);

      I[0] = 0;
      HYPRE_Int p = 0;
      
      for (HYPRE_Int d = 0; d < num_domains; d++)
      {
         const HYPRE_Int sz = (HYPRE_Int)subdomains[d].size();
         for (HYPRE_Int k = 0; k < sz; k++)
         {
            HYPRE_Int gdof = subdomains[d][k];
               MFEM_VERIFY(gdof >= 0 && gdof < num_dofs,
                        "Subdomain DOF out of local A_diag range.");
            J[p++] = gdof;
         }
         I[d + 1] = p;
      }

      HYPRE_Real *data = hypre_CTAlloc(HYPRE_Real, total_dense_data, HYPRE_MEMORY_HOST);
      hypre_CSRMatrixData(csr) = data;

      HYPRE_Int *local_to_global =
         hypre_CTAlloc(HYPRE_Int, std::max<HYPRE_Int>(max_domain_size, 1), HYPRE_MEMORY_HOST);
      HYPRE_Int *global_to_local =
         hypre_CTAlloc(HYPRE_Int, std::max<HYPRE_Int>(num_dofs, 1), HYPRE_MEMORY_HOST);

      for (HYPRE_Int i = 0; i < num_dofs; i++)
      {
         global_to_local[i] = -1;
      }

      HYPRE_Int *Ai = hypre_CSRMatrixI(A_diag);
      HYPRE_Int *Aj = hypre_CSRMatrixJ(A_diag);
      HYPRE_Real *Aa = hypre_CSRMatrixData(A_diag);

      if (use_nonsymm)
      {
         pivots = hypre_CTAlloc(HYPRE_Int, std::max<HYPRE_Int>(total_pivots, 1), HYPRE_MEMORY_HOST);
      }

      HYPRE_Int data_offset = 0;
      HYPRE_Int piv_offset = 0;
      char uplo = 'L';

      for (HYPRE_Int d = 0; d < num_domains; d++)
      {
         HYPRE_Int n = I[d + 1] - I[d];
         MFEM_VERIFY(n >= 0, "Negative subdomain size.");
         MFEM_VERIFY((size_t)data_offset + (size_t)n * (size_t)n <= total_dense_data_sz,
                     "Dense block exceeds allocated storage.");

         for (HYPRE_Int i = 0; i < n; i++)
         {
            HYPRE_Int gdof = J[I[d] + i];
            local_to_global[i] = gdof;
            global_to_local[gdof] = i;
         }

         HYPRE_Real *AE = &data[data_offset];
         std::fill(AE, AE + (size_t)n * (size_t)n, 0.0);

         for (HYPRE_Int i = 0; i < n; i++)
         {
            HYPRE_Int gdof = local_to_global[i];

            for (HYPRE_Int jj = Ai[gdof]; jj < Ai[gdof + 1]; jj++)
            {
               HYPRE_Int gj = Aj[jj];
               HYPRE_Int jloc = (gj >= 0 && gj < num_dofs) ? global_to_local[gj] : -1;

               if (jloc >= 0)
               {
                  AE[i + jloc * n] = Aa[jj];
               }
            }
         }

         if (n > 0)
         {
            if (use_nonsymm)
            {
               int nn = (int)n;
               int info = 0;
               dgetrf_(&nn, &nn, AE, &nn,
                       reinterpret_cast<int *>(&pivots[piv_offset]), &info);
               MFEM_VERIFY(info == 0, "LU factorization failed for Schwarz subdomain.");
               piv_offset += n;
            }
            else
            {
               int nn = (int)n;
               int info = 0;
               dpotrf_(&uplo, &nn, AE, &nn, &info);
               MFEM_VERIFY(info == 0, "Cholesky factorization failed for Schwarz subdomain.");
            }
         }

         data_offset += n * n;

         for (HYPRE_Int i = 0; i < n; i++)
         {
            global_to_local[local_to_global[i]] = -1;
         }
      }

      hypre_TFree(local_to_global, HYPRE_MEMORY_HOST);
      hypre_TFree(global_to_local, HYPRE_MEMORY_HOST);

      domain_structure = (HYPRE_CSRMatrix)csr;

      hypre_GenerateScale(csr, num_dofs, relax_weight, &scale);

      // If uniform_weight is specified (>= 0), override per-DOF scaling with the specified value
      if (uniform_weight >= 0.0)
      {
         for (int i = 0; i < num_dofs; ++i)
         {
            scale[i] = uniform_weight;
         }
      }
      // Otherwise, if unweighted flag is set, override per-DOF scaling with uniform scaling of 1.0
      else if (unweighted)
      {
         for (int i = 0; i < num_dofs; ++i)
         {
            scale[i] = 1.0;
         }
      }

      sd->domain_structure = csr;
      sd->pivots = pivots;
      sd->scale = scale;
      sd->A_boundary = nullptr;

      HYPRE_SchwarzSetDomainStructure(schwarz_solver, domain_structure);
   }

   operator HYPRE_Solver() const override { return schwarz_solver; }
   HYPRE_PtrToParSolverFcn SetupFcn() const override
   { return (HYPRE_PtrToParSolverFcn) DummyParSolverFcn; }
   HYPRE_PtrToParSolverFcn SolveFcn() const override
   {
      static auto wrapped_solve = [](HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                                      HYPRE_ParVector b, HYPRE_ParVector x) -> HYPRE_Int
      {
         HYPRE_Int result = HYPRE_SchwarzSolve(solver, A, b, x);

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

         std::cout << "Schwarz solve residual norm: " << residual_norm << ", RHS norm: " << b_norm << std::endl;

         hypre_ParVectorDestroy(residual);
         return result;
      };
      return (HYPRE_PtrToParSolverFcn) +wrapped_solve;
   }
   using HypreSolver::Mult;
};

#endif // MFEM_SCHWARZ_SOLVER_HPP
