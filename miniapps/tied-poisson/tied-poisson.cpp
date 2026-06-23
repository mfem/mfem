// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. All Rights reserved.
// See files LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Tied Poisson Problem
// ====================
// This miniapp demonstrates solving a Poisson equation on two copies of a mesh
// with one face tied together via penalty constraints. The tied interface is
// enforced by adding alpha * penalty terms coupling matched tied DOFs.

#include "mfem.hpp"
#include "solver_utils.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <vector>

extern "C"
{
#include "HYPRE.h"
#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.h"

void dpotrf_(const char *uplo, const int *n, double *a, const int *lda, int *info);
void dgetrf_(const int *m, const int *n, double *a, const int *lda, int *ipiv, int *info);
void dsyev_(char *jobz, char *uplo, int *n, double *a, int *lda, double *w, double *work, int *lwork, int *info);

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

using namespace std;
using namespace mfem;


HYPRE_Int DummyParSolverFcn(HYPRE_Solver solver,
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
                           HYPRE_Int use_nonsymm = 0)
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

      mfem::out << "Num domains: " << num_domains << endl;
      for (HYPRE_Int d = 0; d < num_domains; d++)
      {
         HYPRE_Int n = I[d + 1] - I[d];
         MFEM_VERIFY(n >= 0, "Negative subdomain size.");
         MFEM_VERIFY((size_t)data_offset + (size_t)n * (size_t)n <= total_dense_data_sz,
                     "Dense block exceeds allocated storage.");

         mfem::out << "Subdomain: ";
         for (HYPRE_Int i = 0; i < n; i++)
         {
            HYPRE_Int gdof = J[I[d] + i];
            local_to_global[i] = gdof;
            global_to_local[gdof] = i;
            mfem::out << gdof << " ";
         }
         mfem::out << endl;

         HYPRE_Real *AE = &data[data_offset];
         std::fill(AE, AE + (size_t)n * (size_t)n, 0.0);

         mfem::out << "Subdomain d = " << d << ", size n = " << n << std::endl;

         for (HYPRE_Int i = 0; i < n; i++)
         {
            HYPRE_Int gdof = local_to_global[i];
            mfem::out << "  local row i = " << i
                     << ", global dof = " << gdof
                     << ", Ai[" << gdof << "] = " << Ai[gdof]
                     << ", Ai[" << gdof + 1 << "] = " << Ai[gdof + 1]
                     << std::endl;

            for (HYPRE_Int jj = Ai[gdof]; jj < Ai[gdof + 1]; jj++)
            {
               HYPRE_Int gj = Aj[jj];
               HYPRE_Int jloc = (gj >= 0 && gj < num_dofs) ? global_to_local[gj] : -1;

               mfem::out << "    jj = " << jj
                        << ", gj = " << gj
                        << ", jloc = " << jloc
                        << ", Aa[jj] = " << Aa[jj]
                        << std::endl;

               if (jloc >= 0)
               {
                  AE[i + jloc * n] = Aa[jj];
                  mfem::out << "      inserted AE(" << i << "," << jloc
                           << ") = " << Aa[jj]
                           << std::endl;
               }
            }
         }

         mfem::out << "  Dense AE block before factorization:" << std::endl;
         for (HYPRE_Int row = 0; row < n; row++)
         {
            mfem::out << "    ";
            for (HYPRE_Int col = 0; col < n; col++)
            {
               mfem::out << AE[row + col * n] << " ";
            }
            mfem::out << std::endl;
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
   { return (HYPRE_PtrToParSolverFcn) HYPRE_SchwarzSolve; }
   using HypreSolver::Mult;
};

   bool SymmetricEigenDecomposition(mfem::DenseMatrix &A,
                                    std::vector<double> &eigvals)
   {
      const int n = A.Height();
      if (A.Width() != n) { return false; }

      eigvals.resize(n);

      char jobz = 'V'; // compute eigenvalues and eigenvectors
      char uplo = 'U'; // use upper triangle of A
      int lda = n;
      int info = 0;

      // Workspace query
      int lwork = -1;
      double work_size = 0.0;

      dsyev_(&jobz, &uplo, &lda, A.Data(), &lda,
            eigvals.data(), &work_size, &lwork, &info);

      if (info != 0) { return false; }

      lwork = static_cast<int>(work_size);
      std::vector<double> work(lwork);

      dsyev_(&jobz, &uplo, &lda, A.Data(), &lda,
            eigvals.data(), work.data(), &lwork, &info);

      return info == 0;
   }

// -----------------------------------------------------------------------------
// Spectrum analysis helper
// -----------------------------------------------------------------------------
void analyze_spectrum(HypreParMatrix &A)
{
   if (Mpi::Root())
   {
      mfem::out << "\nComputing eigenvalues..." << endl;
   }

   const int total_size = A.N();

   Vector x_rand(total_size);
   x_rand.Randomize(12345);
   Vector y(total_size);

   double lambda_max = 0.0;
   for (int i = 0; i < 20; i++)
   {
      A.Mult(x_rand, y);
      lambda_max = InnerProduct(x_rand, y);
      double norm = sqrt(InnerProduct(y, y));
      x_rand = y;
      x_rand *= 1.0 / norm;
   }

   HypreBoomerAMG amg_inv(A);
   amg_inv.SetPrintLevel(0);

   GMRESSolver inv_solver(MPI_COMM_WORLD);
   inv_solver.SetRelTol(1e-6);
   inv_solver.SetMaxIter(100);
   inv_solver.SetPrintLevel(0);
   inv_solver.SetOperator(A);
   inv_solver.SetPreconditioner(amg_inv);

   x_rand.Randomize(54321);
   double lambda_min = 0.0;
   for (int i = 0; i < 20; i++)
   {
      inv_solver.Mult(x_rand, y);
      lambda_min = InnerProduct(x_rand, y);
      double norm = sqrt(InnerProduct(y, y));
      x_rand = y;
      x_rand *= 1.0 / norm;
   }
   lambda_min = 1.0 / lambda_min;

   if (Mpi::Root())
   {
      mfem::out << "\nEstimated eigenvalue range:" << endl;
      mfem::out << "  lambda_min = " << lambda_min << endl;
      mfem::out << "  lambda_max = " << lambda_max << endl;
      mfem::out << "  Condition number estimate = "
                << lambda_max / lambda_min << endl;
      mfem::out << endl;
   }
}

// -----------------------------------------------------------------------------
// Small helper type for tying by boundary vertex position
// -----------------------------------------------------------------------------
struct GdofPos
{
   HYPRE_BigInt gdof;
   double x[3];
};

// -----------------------------------------------------------------------------
// Utility helpers
// -----------------------------------------------------------------------------
int FindOwner(HYPRE_BigInt row, const std::vector<HYPRE_BigInt> &row_starts)
{
   auto it = std::upper_bound(row_starts.begin(), row_starts.end(), row);
   return static_cast<int>(it - row_starts.begin()) - 1;
}

bool SamePointTol(const GdofPos &a, const GdofPos &b, double tol)
{
   return std::abs(a.x[0] - b.x[0]) < tol &&
          std::abs(a.x[1] - b.x[1]) < tol &&
          std::abs(a.x[2] - b.x[2]) < tol;
}

// -----------------------------------------------------------------------------
// Build a combined serial mesh containing the original mesh and a translated
// copy. The copied tied boundary is given a distinct attribute so the two
// tied sides can be distinguished in the combined mesh.
// -----------------------------------------------------------------------------
std::unique_ptr<Mesh> BuildCombinedMesh(Mesh &serial_mesh,
                                        int tied_bdr_attr,
                                        int copy_tied_attr,
                                        double &separation)
{
   Mesh serial_mesh_copy(serial_mesh, true);

   for (int i = 0; i < serial_mesh.GetNBE(); i++)
   {
      Element *bel = serial_mesh_copy.GetBdrElement(i);
      if (bel->GetAttribute() == tied_bdr_attr)
      {
         bel->SetAttribute(copy_tied_attr);
      }
   }
   serial_mesh_copy.SetAttributes();

   if (separation == 0.0)
   {
      Vector bb_min, bb_max;
      serial_mesh.GetBoundingBox(bb_min, bb_max);
      separation = 2.0 * (bb_max(0) - bb_min(0));
   }

   for (int i = 0; i < serial_mesh_copy.GetNV(); i++)
   {
      double *coord = serial_mesh_copy.GetVertex(i);
      coord[0] += separation;
   }

   Mesh *mesh_array[] = { &serial_mesh, &serial_mesh_copy };
   return std::make_unique<Mesh>(mesh_array, 2);
}

// -----------------------------------------------------------------------------
// Collect local boundary vertex true-DOF / position pairs for the original tied
// side and copied tied side.
//
// Assumptions:
// - H1 scalar space
// - vertex-based tied DOFs only
// -----------------------------------------------------------------------------
void CollectLocalBoundaryGdofPositions(ParMesh &pmesh,
                                       ParFiniteElementSpace &fespace,
                                       int tied_bdr_attr,
                                       int copy_tied_attr,
                                       std::vector<GdofPos> &tied_side_data,
                                       std::vector<GdofPos> &copy_side_data)
{
   tied_side_data.clear();
   copy_side_data.clear();

   const int dim = pmesh.SpaceDimension();
   const HYPRE_BigInt tdof_offset = fespace.GetMyTDofOffset();

   std::set<int> seen_tied_vertices;
   std::set<int> seen_copy_vertices;

   for (int be = 0; be < pmesh.GetNBE(); ++be)
   {
      Element *bel = pmesh.GetBdrElement(be);
      const int attr = bel->GetAttribute();

      if (attr != tied_bdr_attr && attr != copy_tied_attr)
      {
         continue;
      }

      Array<int> verts;
      bel->GetVertices(verts);

      Array<int> vdofs;
      fespace.GetBdrElementVDofs(be, vdofs);

      MFEM_VERIFY(verts.Size() == vdofs.Size(),
                  "Expected one H1 vertex dof per boundary vertex");

      for (int j = 0; j < verts.Size(); ++j)
      {
         const int v = verts[j];
         const int vdof = vdofs[j];

         std::set<int> *seen =
            (attr == tied_bdr_attr) ? &seen_tied_vertices : &seen_copy_vertices;

         if (!seen->insert(v).second)
         {
            continue;
         }

         const int ltdof = fespace.GetLocalTDofNumber(vdof);
         if (ltdof < 0)
         {
            continue;
         }

         GdofPos entry;
         entry.gdof = tdof_offset + ltdof;

         const double *X = pmesh.GetVertex(v);
         entry.x[0] = X[0];
         entry.x[1] = (dim > 1) ? X[1] : 0.0;
         entry.x[2] = (dim > 2) ? X[2] : 0.0;

         if (attr == tied_bdr_attr)
         {
            tied_side_data.push_back(entry);
         }
         else
         {
            copy_side_data.push_back(entry);
         }
      }
   }
}

// -----------------------------------------------------------------------------
// Gather local (gdof, position) data from all ranks and sort lexicographically
// by position, with gdof as a tie-breaker.
// -----------------------------------------------------------------------------
std::vector<GdofPos> GatherAndSortGdofPositions(const std::vector<GdofPos> &local_data,
                                                MPI_Comm comm)
{
   const int nranks = Mpi::WorldSize();
   const int local_n = static_cast<int>(local_data.size());

   std::vector<int> counts(nranks, 0), displs(nranks, 0);
   MPI_Allgather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);

   int total_n = 0;
   for (int p = 0; p < nranks; ++p)
   {
      displs[p] = total_n;
      total_n += counts[p];
   }

   std::vector<HYPRE_BigInt> send_gdofs(local_n);
   std::vector<double> send_x(local_n), send_y(local_n), send_z(local_n);

   for (int i = 0; i < local_n; ++i)
   {
      send_gdofs[i] = local_data[i].gdof;
      send_x[i] = local_data[i].x[0];
      send_y[i] = local_data[i].x[1];
      send_z[i] = local_data[i].x[2];
   }

   std::vector<HYPRE_BigInt> recv_gdofs(total_n);
   std::vector<double> recv_x(total_n), recv_y(total_n), recv_z(total_n);

   MPI_Allgatherv(local_n ? send_gdofs.data() : nullptr, local_n, HYPRE_MPI_BIG_INT,
                  total_n ? recv_gdofs.data() : nullptr, counts.data(), displs.data(),
                  HYPRE_MPI_BIG_INT, comm);

   MPI_Allgatherv(local_n ? send_x.data() : nullptr, local_n, MPI_DOUBLE,
                  total_n ? recv_x.data() : nullptr, counts.data(), displs.data(),
                  MPI_DOUBLE, comm);

   MPI_Allgatherv(local_n ? send_y.data() : nullptr, local_n, MPI_DOUBLE,
                  total_n ? recv_y.data() : nullptr, counts.data(), displs.data(),
                  MPI_DOUBLE, comm);

   MPI_Allgatherv(local_n ? send_z.data() : nullptr, local_n, MPI_DOUBLE,
                  total_n ? recv_z.data() : nullptr, counts.data(), displs.data(),
                  MPI_DOUBLE, comm);

   std::vector<GdofPos> global_data(total_n);
   for (int i = 0; i < total_n; ++i)
   {
      global_data[i].gdof = recv_gdofs[i];
      global_data[i].x[0] = recv_x[i];
      global_data[i].x[1] = recv_y[i];
      global_data[i].x[2] = recv_z[i];
   }

   std::sort(global_data.begin(), global_data.end(),
             [](const GdofPos &a, const GdofPos &b)
             {
                if (a.x[0] != b.x[0]) { return a.x[0] < b.x[0]; }
                if (a.x[1] != b.x[1]) { return a.x[1] < b.x[1]; }
                if (a.x[2] != b.x[2]) { return a.x[2] < b.x[2]; }
                return a.gdof < b.gdof;
             });

   return global_data;
}

// -----------------------------------------------------------------------------
// Build tied_pairs and all_tied_gdofs_array from the gathered and sorted data.
//
// The copied mesh is shifted by +separation in x, so the verification step
// subtracts that translation before comparing positions.
// -----------------------------------------------------------------------------
void BuildTiedPairsFromSortedData(const std::vector<GdofPos> &global_tied_side_data,
                                  const std::vector<GdofPos> &global_copy_side_data,
                                  double separation,
                                  double tol,
                                  bool symmetric_tie,
                                  std::map<HYPRE_BigInt, HYPRE_BigInt> &tied_pairs,
                                  Array<HYPRE_BigInt> &all_tied_gdofs_array)
{
   auto dedup_by_position = [&](const std::vector<GdofPos> &in,
                                std::vector<GdofPos> &out)
   {
      out.clear();
      for (const auto &e : in)
      {
         if (out.empty() || !SamePointTol(out.back(), e, tol))
         {
            out.push_back(e);
         }
      }
   };

   std::vector<GdofPos> tied_unique;
   std::vector<GdofPos> copy_unique;

   dedup_by_position(global_tied_side_data, tied_unique);
   dedup_by_position(global_copy_side_data, copy_unique);

   MFEM_VERIFY(tied_unique.size() == copy_unique.size(),
               "Tied sides have different numbers of unique positions");

   tied_pairs.clear();
   all_tied_gdofs_array.SetSize(2 * static_cast<int>(tied_unique.size()));

   for (int i = 0; i < static_cast<int>(tied_unique.size()); ++i)
   {
      const GdofPos &a = tied_unique[i];
      const GdofPos &b = copy_unique[i];

      const double bx = b.x[0] - separation;
      const double by = b.x[1];
      const double bz = b.x[2];

      MFEM_VERIFY(std::abs(a.x[0] - bx) < tol &&
                  std::abs(a.x[1] - by) < tol &&
                  std::abs(a.x[2] - bz) < tol,
                  "Sorted tied points do not match between original and copied mesh");

      tied_pairs[a.gdof] = b.gdof;
      if (symmetric_tie)
      {
         tied_pairs[b.gdof] = a.gdof;
      }

      all_tied_gdofs_array[2 * i]     = a.gdof;
      all_tied_gdofs_array[2 * i + 1] = b.gdof;
   }
}

// -----------------------------------------------------------------------------
// Reconstruct the full global row partition from the local row interval stored
// on each rank.
// -----------------------------------------------------------------------------
std::vector<HYPRE_BigInt> GetGlobalRowPartition(const HypreParMatrix &A,
                                                MPI_Comm comm)
{
   const int nranks = Mpi::WorldSize();

   const HYPRE_BigInt *local_row_starts = A.GetRowStarts();
   const HYPRE_BigInt local_row_start = local_row_starts[0];
   const HYPRE_BigInt local_row_end   = local_row_starts[1];

   std::vector<HYPRE_BigInt> all_row_starts(nranks);
   std::vector<HYPRE_BigInt> all_row_ends(nranks);

   MPI_Allgather(&local_row_start, 1, HYPRE_MPI_BIG_INT,
                 all_row_starts.data(), 1, HYPRE_MPI_BIG_INT, comm);

   MPI_Allgather(&local_row_end, 1, HYPRE_MPI_BIG_INT,
                 all_row_ends.data(), 1, HYPRE_MPI_BIG_INT, comm);

   std::vector<HYPRE_BigInt> row_starts(nranks + 1);
   for (int r = 0; r < nranks; ++r)
   {
      row_starts[r] = all_row_starts[r];
   }
   row_starts[nranks] = all_row_ends[nranks - 1];

   return row_starts;
}

// -----------------------------------------------------------------------------
// Assemble the local sparse matrix with added penalty terms.
//
// Each rank:
// 1. copies its owned rows from the merged matrix
// 2. emits penalty contributions as global (row, col, value) entries
// 3. ships each contribution to the rank that owns its destination row
// 4. inserts only received contributions for locally owned rows
// -----------------------------------------------------------------------------
SparseMatrix BuildPenaltyModifiedMatrix(const HypreParMatrix &A,
                                        const SparseMatrix &merged_A,
                                        const std::map<HYPRE_BigInt, HYPRE_BigInt> &tied_pairs,
                                        const Array<HYPRE_BigInt> &all_tied_gdofs_array,
                                        double alpha,
                                        MPI_Comm comm,
                                        bool even_weighting = true)
{
   const HYPRE_BigInt row_start = A.GetRowStarts()[0];
   const HYPRE_BigInt row_end   = A.GetRowStarts()[1];
   const int local_num_rows     = A.GetNumRows();
   const HYPRE_BigInt global_num_cols = A.N();

   const int nranks = Mpi::WorldSize();
   const std::vector<HYPRE_BigInt> row_starts = GetGlobalRowPartition(A, comm);

   SparseMatrix modified_A(local_num_rows, global_num_cols);

   std::vector<HYPRE_BigInt> local_rows;
   std::vector<HYPRE_BigInt> local_cols;
   std::vector<real_t> local_vals;

   auto emit = [&](HYPRE_BigInt r, HYPRE_BigInt c, real_t v)
   {
      local_rows.push_back(r);
      local_cols.push_back(c);
      local_vals.push_back(v);
   };

   HYPRE_BigInt paired_gdof;
   for (int local_row = 0; local_row < local_num_rows; ++local_row)
   {
      const HYPRE_BigInt global_row = row_start + local_row;

      const int *cols = merged_A.GetRowColumns(local_row);
      const real_t *vals = merged_A.GetRowEntries(local_row);
      const int row_size = merged_A.RowSize(local_row);

      for (int j = 0; j < row_size; ++j)
      {
         modified_A.Add(local_row, cols[j], vals[j]);
      }

      auto it = tied_pairs.find(global_row);
      if (it == tied_pairs.end())
      {
         continue;
      }

      paired_gdof = it->second;
      const real_t value = alpha;

      std::vector<HYPRE_BigInt> group;
      group.push_back(paired_gdof);

      for (int j = 0; j < row_size; ++j)
      {
         const HYPRE_BigInt c = cols[j];

         if (c == global_row || c == paired_gdof)
         {
            continue;
         }

         if (all_tied_gdofs_array.Find(c) != -1)
         {
            group.push_back(c);
         }
      }

      std::sort(group.begin(), group.end());
      group.erase(std::unique(group.begin(), group.end()), group.end());

      const int m = static_cast<int>(group.size());
      if (m == 0)
      {
         continue;
      }

      const real_t w = 1.0 / m;

      // Penalty:
      // value * (u_i - sum_k w v_k)^2

      emit(global_row, global_row, value);

      for (int k = 0; k < m; ++k)
      {
         const HYPRE_BigInt vk = group[k];
         const real_t w = even_weighting ? 1.0 / m : vk == paired_gdof ? 0.5 : 0.5 / (static_cast<real_t>(m) - 1.0);
         emit(global_row, vk, -value * w);
         emit(vk, global_row, -value * w);
      }

      for (int k = 0; k < m; ++k)
      {
         const HYPRE_BigInt vk = group[k];
         const real_t wk = even_weighting ? 1.0 / m : vk == paired_gdof ? 0.5 : 0.5 / (static_cast<real_t>(m) - 1.0);
         for (int l = 0; l < m; ++l)
         {
            const HYPRE_BigInt vl = group[l];
            const real_t wl = even_weighting ? 1.0 / m : vl == paired_gdof ? 0.5 : 0.5 / (static_cast<real_t>(m) - 1.0);
            emit(vk, vl, value * wk * wl);
         }
      }
   }

   std::vector<std::vector<HYPRE_BigInt>> send_rows_by_rank(nranks);
   std::vector<std::vector<HYPRE_BigInt>> send_cols_by_rank(nranks);
   std::vector<std::vector<real_t>> send_vals_by_rank(nranks);

   for (int i = 0; i < static_cast<int>(local_rows.size()); ++i)
   {
      const HYPRE_BigInt r = local_rows[i];
      const HYPRE_BigInt c = local_cols[i];
      const real_t v = local_vals[i];

      const int owner = FindOwner(r, row_starts);
      MFEM_VERIFY(owner >= 0 && owner < nranks, "Invalid row owner");

      send_rows_by_rank[owner].push_back(r);
      send_cols_by_rank[owner].push_back(c);
      send_vals_by_rank[owner].push_back(v);
   }

   for (int r = 0; r < nranks; ++r)
   {
      MFEM_VERIFY(send_rows_by_rank[r].size() == send_cols_by_rank[r].size(),
                  "Per-rank row/col size mismatch");
      MFEM_VERIFY(send_rows_by_rank[r].size() == send_vals_by_rank[r].size(),
                  "Per-rank row/val size mismatch");
   }

   std::vector<int> send_counts(nranks, 0), recv_counts(nranks, 0);
   for (int r = 0; r < nranks; ++r)
   {
      send_counts[r] = static_cast<int>(send_rows_by_rank[r].size());
   }

   MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                recv_counts.data(), 1, MPI_INT, comm);

   std::vector<int> send_displs(nranks, 0), recv_displs(nranks, 0);
   for (int r = 1; r < nranks; ++r)
   {
      send_displs[r] = send_displs[r - 1] + send_counts[r - 1];
      recv_displs[r] = recv_displs[r - 1] + recv_counts[r - 1];
   }

   const int total_send = send_displs[nranks - 1] + send_counts[nranks - 1];
   const int total_recv = recv_displs[nranks - 1] + recv_counts[nranks - 1];

   std::vector<HYPRE_BigInt> send_rows(total_send);
   std::vector<HYPRE_BigInt> send_cols(total_send);
   std::vector<real_t> send_vals(total_send);

   for (int r = 0; r < nranks; ++r)
   {
      int p = send_displs[r];
      const int end = p + send_counts[r];

      for (int i = 0; i < send_counts[r]; ++i)
      {
         MFEM_VERIFY(p < end, "Packing overflow");

         send_rows[p] = send_rows_by_rank[r][i];
         send_cols[p] = send_cols_by_rank[r][i];
         send_vals[p] = send_vals_by_rank[r][i];
         ++p;
      }

      MFEM_VERIFY(p == end, "Packing underflow/overflow");
   }

   std::vector<HYPRE_BigInt> recv_rows(total_recv);
   std::vector<HYPRE_BigInt> recv_cols(total_recv);
   std::vector<real_t> recv_vals(total_recv);

   MPI_Alltoallv(total_send ? send_rows.data() : nullptr,
                 send_counts.data(), send_displs.data(), HYPRE_MPI_BIG_INT,
                 total_recv ? recv_rows.data() : nullptr,
                 recv_counts.data(), recv_displs.data(), HYPRE_MPI_BIG_INT,
                 comm);

   MPI_Alltoallv(total_send ? send_cols.data() : nullptr,
                 send_counts.data(), send_displs.data(), HYPRE_MPI_BIG_INT,
                 total_recv ? recv_cols.data() : nullptr,
                 recv_counts.data(), recv_displs.data(), HYPRE_MPI_BIG_INT,
                 comm);

   MPI_Alltoallv(total_send ? send_vals.data() : nullptr,
                 send_counts.data(), send_displs.data(),
                 sizeof(real_t) == sizeof(double) ? MPI_DOUBLE : MPI_FLOAT,
                 total_recv ? recv_vals.data() : nullptr,
                 recv_counts.data(), recv_displs.data(),
                 sizeof(real_t) == sizeof(double) ? MPI_DOUBLE : MPI_FLOAT,
                 comm);

   for (int i = 0; i < total_recv; ++i)
   {
      const HYPRE_BigInt r = recv_rows[i];
      const HYPRE_BigInt c = recv_cols[i];
      const real_t v = recv_vals[i];

      MFEM_VERIFY(r >= row_start && r < row_end, "Received nonlocal row");

      const int local_row = static_cast<int>(r - row_start);
      modified_A.Add(local_row, c, v);
   }

   modified_A.Finalize();
   return modified_A;
}

// -----------------------------------------------------------------------------
// Convert the merged local SparseMatrix back into a HypreParMatrix by splitting
// entries into diagonal and off-diagonal blocks.
// -----------------------------------------------------------------------------
HypreParMatrix *BuildHypreParMatrixFromMerged(const SparseMatrix &modified_A,
                                              const HypreParMatrix &A)
{
   const int local_num_rows = A.GetNumRows();
   const HYPRE_BigInt col_start = A.GetColStarts()[0];
   const HYPRE_BigInt col_end   = A.GetColStarts()[1];

   const int *I = modified_A.GetI();
   const int *J = modified_A.GetJ();
   const real_t *data = modified_A.GetData();

   std::vector<int> diag_I_vec(1, 0), diag_J_vec;
   std::vector<real_t> diag_data_vec;
   std::vector<int> offd_I_vec(1, 0), offd_J_vec;
   std::vector<real_t> offd_data_vec;
   std::set<HYPRE_BigInt> offd_cols_set;

   for (int i = 0; i < local_num_rows; ++i)
   {
      for (int idx = I[i]; idx < I[i + 1]; ++idx)
      {
         const HYPRE_BigInt global_col = J[idx];
         const real_t value = data[idx];

         if (global_col >= col_start && global_col < col_end)
         {
            diag_J_vec.push_back(static_cast<int>(global_col - col_start));
            diag_data_vec.push_back(value);
         }
         else
         {
            offd_cols_set.insert(global_col);
            offd_J_vec.push_back(static_cast<int>(global_col));
            offd_data_vec.push_back(value);
         }
      }

      diag_I_vec.push_back(static_cast<int>(diag_J_vec.size()));
      offd_I_vec.push_back(static_cast<int>(offd_J_vec.size()));
   }

   const int offd_num_cols = static_cast<int>(offd_cols_set.size());

   HYPRE_BigInt *new_cmap =
      mfem::Memory<HYPRE_BigInt>(offd_num_cols, Device::GetHostMemoryType());

   std::map<HYPRE_BigInt, int> global_to_local_offd;
   int idx = 0;
   for (HYPRE_BigInt gcol : offd_cols_set)
   {
      new_cmap[idx] = gcol;
      global_to_local_offd[gcol] = idx;
      ++idx;
   }

   for (int &col : offd_J_vec)
   {
      col = global_to_local_offd[col];
   }

   HYPRE_Int *new_diag_I =
      mfem::Memory<HYPRE_Int>(local_num_rows + 1, Device::GetHostMemoryType());
   HYPRE_Int *new_diag_J =
      mfem::Memory<HYPRE_Int>(diag_J_vec.size(), Device::GetHostMemoryType());
   real_t *new_diag_data =
      mfem::Memory<real_t>(diag_data_vec.size(), Device::GetHostMemoryType());

   HYPRE_Int *new_offd_I =
      mfem::Memory<HYPRE_Int>(local_num_rows + 1, Device::GetHostMemoryType());
   HYPRE_Int *new_offd_J =
      mfem::Memory<HYPRE_Int>(offd_J_vec.size(), Device::GetHostMemoryType());
   real_t *new_offd_data =
      mfem::Memory<real_t>(offd_data_vec.size(), Device::GetHostMemoryType());

   std::copy(diag_I_vec.begin(), diag_I_vec.end(), new_diag_I);
   std::copy(diag_J_vec.begin(), diag_J_vec.end(), new_diag_J);
   std::copy(diag_data_vec.begin(), diag_data_vec.end(), new_diag_data);

   std::copy(offd_I_vec.begin(), offd_I_vec.end(), new_offd_I);
   std::copy(offd_J_vec.begin(), offd_J_vec.end(), new_offd_J);
   std::copy(offd_data_vec.begin(), offd_data_vec.end(), new_offd_data);

   return new HypreParMatrix(
      MPI_COMM_WORLD,
      A.M(), A.N(),
      A.GetRowStarts(),
      A.GetColStarts(),
      new_diag_I, new_diag_J, new_diag_data,
      new_offd_I, new_offd_J, new_offd_data,
      offd_num_cols,
      new_cmap,
      false);
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   MPI_Comm comm = MPI_COMM_WORLD;

   string mesh_file = "../../data/beam-tet.mesh";
   int ref_levels = 4;
   double alpha = 1e3;
   int tied_bdr_attr = 1;
   bool visualization = true;
   bool compute_eigenvalues = false;
   double separation = 0.0;
   int pcg_max_iters = 10000;
   double diffusion_ratio = 1.0;
   bool uniform_alpha = true;
   bool do_amgf = false;
   bool schwarz_subspace_filter = false;
   bool amg_subspace_filter = false;
   bool precondition_subspace_cg = true;
   bool select_schwarz_domains_from_eigendecomp = false;
   bool visualize_spectrum = false;
   int subdomain_cg_iters = 0;
   bool one_level_amg = false;
   bool symmetric_tie = false;
   bool even_weighting = false;
   bool use_schur_complement = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&alpha, "-a", "--alpha",
                  "Penalty parameter for tied interface.");
   args.AddOption(&tied_bdr_attr, "-t", "--tied-attr",
                  "Boundary attribute to tie between mesh copies.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&compute_eigenvalues, "-eig", "--eigenvalues", "-no-eig",
                  "--no-eigenvalues",
                  "Compute and display min/max eigenvalues.");
   args.AddOption(&separation, "-sep", "--separation",
                  "Separation distance for visualization (0 = auto).");
   args.AddOption(&pcg_max_iters, "-p", "--pcg-max-iters",
                  "Max PCG Iterations");
   args.AddOption(&diffusion_ratio, "-d", "--diffusion-ratio",
                  "Ratio of the diffusion coeffecients on each mesh copy.");
   args.AddOption(&do_amgf, "-amgf", "--amgf", "-no-amgf", "--no-amgf",
                  "Enable or disable AMG with Filtering solver.");
   args.AddOption(&schwarz_subspace_filter, "-schwarz-filter", "--schwarz-filter",
                  "-no-schwarz-filter", "--no-schwarz-filter",
                  "Enable or disable use of Schwarz solver on subspace in AMGF.");
   args.AddOption(&amg_subspace_filter, "-amg-filter", "--amg-filter",
                  "-no-amg-filter", "--no-amg-filter",
                  "Enable or disable use of AMG solver on subspace in AMGF.");
   args.AddOption(&precondition_subspace_cg, "-precondition-subspace-cg", "--precondition-subspace-cg",
                  "-no-precondition-subspace-cg", "--no-precondition-subspace-cg",
                  "Enable or disable use of PCG for iterative solver on subspace");
   args.AddOption(&select_schwarz_domains_from_eigendecomp, "-select-subdomains-from-spectrum", "--select-subdomains-from-spectrum",
                  "-no-select-subdomains-from-spectrum", "--no-select-subdomains-from-spectrum",
                  "Enable or disable use of eigendecomposition to select subsubdomains for Schwarz on the filtered space.");
   args.AddOption(&uniform_alpha, "-uniform-alpha", "--uniform-alpha",
                  "-nonuniform-alpha", "--nonuniform-alpha",
                  "Weighting parameter controlling the degree of nonuniformity.");
   args.AddOption(&visualize_spectrum, "-vis-spectrum", "--vis-spectrum",
                  "-no-vis-spectrum", "--no-vis-spectrum",
                  "Enable or disable visualization of spectrum on the subspace.");
   args.AddOption(&one_level_amg, "-one-level-amg", "--one-level-amg",
                  "-no-one-level-amg", "--no-one-level-amg",
                  "Enable or disable using a non multigrid preconditioner.");
   args.AddOption(&symmetric_tie, "-symmetric-tie", "-no-symmetric-tie",
                  "--symmetric-tie", "--no-symmetric-tie",
                  "Tie both faces to each other.");
   args.AddOption(&even_weighting, "-even-weighting", "-no-even-weighting",
                  "--even-weighting", "--no-even-weighting",
                  "Use an evenly weighted kernel to tied DoFs.");
   args.AddOption(&subdomain_cg_iters, "-s", "--subdomain-pcg-max-iters",
                  "Max PCG Iterations for subdomain (0 applies Schwarz/AMG only).");
   args.AddOption(&use_schur_complement, "-schur", "--use-schur-complement",
                  "-no-schur", "--no-use-schur-complement",
                  "Use Schur complement reduction with pure AMG-preconditioned CG.");
   args.ParseCheck();

   if (schwarz_subspace_filter && amg_subspace_filter)
   {
      mfem::out << "Both Schwarz and AMG subspace filter cannot be used at the same time." << endl;
      return 1;
   }

   // --------------------------------------------------------------------------
   // Mesh setup
   // --------------------------------------------------------------------------
   Mesh serial_mesh(mesh_file);
   const int dim = serial_mesh.Dimension();

   for (int i = 0; i < serial_mesh.GetNE(); i++)
   {
      serial_mesh.GetElement(i)->SetAttribute(1);
   }
   serial_mesh.SetAttributes();

   const int copy_tied_attr = serial_mesh.bdr_attributes.Max() + 1;

   for (int l = 0; l < ref_levels; l++)
   {
      serial_mesh.UniformRefinement();
   }

   std::unique_ptr<Mesh> combined_mesh =
      BuildCombinedMesh(serial_mesh, tied_bdr_attr, copy_tied_attr, separation);

   ParMesh pmesh(comm, *combined_mesh);

   // --------------------------------------------------------------------------
   // Finite element space
   // --------------------------------------------------------------------------
   H1_FECollection fec(1, dim);
   ParFiniteElementSpace fespace(&pmesh, &fec);

   // --------------------------------------------------------------------------
   // Build tied DOF pairs from boundary vertex positions
   // --------------------------------------------------------------------------
   std::vector<GdofPos> tied_side_data;
   std::vector<GdofPos> copy_side_data;

   CollectLocalBoundaryGdofPositions(pmesh, fespace,
                                     tied_bdr_attr, copy_tied_attr,
                                     tied_side_data, copy_side_data);

   std::vector<GdofPos> global_tied_side_data =
      GatherAndSortGdofPositions(tied_side_data, comm);

   std::vector<GdofPos> global_copy_side_data =
      GatherAndSortGdofPositions(copy_side_data, comm);

   const double tol = 1e-10;

   std::map<HYPRE_BigInt, HYPRE_BigInt> tied_pairs;
   Array<HYPRE_BigInt> all_tied_gdofs_array;

   BuildTiedPairsFromSortedData(global_tied_side_data,
                                global_copy_side_data,
                                separation,
                                tol,
                                symmetric_tie,
                                tied_pairs,
                                all_tied_gdofs_array);

   const HYPRE_BigInt total_size = fespace.GetVSize();

   if (Mpi::Root())
   {
      const int num_pairs = symmetric_tie
                          ? static_cast<int>(tied_pairs.size()) / 2
                          : static_cast<int>(tied_pairs.size());

      mfem::out << "Total DOFs: " << total_size << endl;
      mfem::out << "Total Tied Pairs: " << num_pairs << endl;
   }

   // --------------------------------------------------------------------------
   // Essential boundary conditions
   // --------------------------------------------------------------------------
   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 1;
   ess_bdr[tied_bdr_attr - 1] = 0;
   ess_bdr[copy_tied_attr - 1] = 0;

   Array<int> ess_tdof;
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof);

   // --------------------------------------------------------------------------
   // RHS and stiffness operator
   // --------------------------------------------------------------------------
   ConstantCoefficient one(1.0);

   ParLinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   PWConstCoefficient diffusion_coeff(pmesh.GetElementAttributes().Max());
   diffusion_coeff(1) = 1.0;
   diffusion_coeff(2) = 1.0 * diffusion_ratio;

   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator(diffusion_coeff));
   a.Assemble();

   ParGridFunction x(&fespace);
   x = 0.0;

   for (int i = 0; i < ess_tdof.Size(); i++)
   {
      x(ess_tdof[i]) = 0.0;
   }

   HypreParMatrix A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof, x, b, A, X, B);

   // --------------------------------------------------------------------------
   // Build penalty-modified operator
   // --------------------------------------------------------------------------
   SparseMatrix merged_A;
   A.MergeDiagAndOffd(merged_A);

   SparseMatrix modified_A =
      BuildPenaltyModifiedMatrix(A, merged_A,
                                 tied_pairs,
                                 all_tied_gdofs_array,
                                 alpha,
                                 comm,
                                 even_weighting);

   HypreParMatrix *tiedA = BuildHypreParMatrixFromMerged(modified_A, A);

   if (compute_eigenvalues)
   {
      analyze_spectrum(A);
   }

   // --------------------------------------------------------------------------
   // Preconditioner setup
   // --------------------------------------------------------------------------
   Solver *prec = nullptr;
   Solver *subspacesolver = nullptr;
   HypreParMatrix *P_tied_T = nullptr;
   HypreParMatrix *P_tied = nullptr;
   HypreParMatrix *PTAP = nullptr;

   if (!use_schur_complement)
   {
      Array<int> owned_tied_gdofs;
      for (int i = Mpi::WorldRank();
            i < all_tied_gdofs_array.Size() / 2;
            i += Mpi::WorldSize())
      {
         owned_tied_gdofs.Append(all_tied_gdofs_array[2 * i]);
         owned_tied_gdofs.Append(all_tied_gdofs_array[2 * i + 1]);
      }

      const int nrows_tied = owned_tied_gdofs.Size();
      SparseMatrix Pct(nrows_tied, fespace.GlobalTrueVSize());

      for (int i = 0; i < nrows_tied; ++i)
      {
         Pct.Set(i, owned_tied_gdofs[i], 1.0);
      }
      Pct.Finalize();

      HYPRE_BigInt rows_c[2];

      HYPRE_BigInt row_offset_tied;
      HYPRE_BigInt nrows_tied_bigint = nrows_tied;
      MPI_Scan(&nrows_tied_bigint, &row_offset_tied, 1, MPI_INT,
               MPI_SUM, comm);

      row_offset_tied -= nrows_tied_bigint;
      rows_c[0] = row_offset_tied;
      rows_c[1] = row_offset_tied + nrows_tied;

      HYPRE_BigInt glob_nrows_tied;
      HYPRE_BigInt glob_ncols_tied = fespace.GlobalTrueVSize();
      MPI_Allreduce(&nrows_tied_bigint, &glob_nrows_tied, 1,
                     MPI_INT, MPI_SUM, comm);

      HYPRE_BigInt *J;
#ifndef HYPRE_BIGINT
      J = Pct.GetJ();
#else
      J = new HYPRE_BigInt[Pct.NumNonZeroElems()];
      for (int i = 0; i < Pct.NumNonZeroElems(); i++)
      {
         J[i] = Pct.GetJ()[i];
      }
#endif

      P_tied_T = new HypreParMatrix(comm,
                                    nrows_tied, glob_nrows_tied,
                                    glob_ncols_tied,
                                    Pct.GetI(), J, Pct.GetData(),
                                    rows_c, fespace.GetTrueDofOffsets());

      P_tied = P_tied_T->Transpose();
      PTAP = RAP(tiedA, P_tied);

      if (do_amgf)
      {
         prec = new AMGFSolver();
      auto *amgfprec = dynamic_cast<AMGFSolver *>(prec);
      amgfprec->GetAMG().SetPrintLevel(0);
      amgfprec->GetAMG().SetStrengthThresh(0.5);
      amgfprec->SetFilteredSubspaceTransferOperator(*P_tied);

      if (one_level_amg)
      {
         amgfprec->GetAMG().SetMaxLevels(1);
      }

      if (schwarz_subspace_filter)
      {
         subspacesolver = new HypreSchwarz();
         auto scwharzprec = dynamic_cast<HypreSchwarz*>(subspacesolver);
         HYPRE_SchwarzCreate(&scwharzprec->schwarz_solver);
         HYPRE_SchwarzSetVariant(scwharzprec->schwarz_solver, 0);
         std::vector<std::vector<HYPRE_Int>> subdomains;
         if (select_schwarz_domains_from_eigendecomp)
         {
            TODO:
         }
         else
         {
            subdomains.resize(all_tied_gdofs_array.Size() / 2);
            for (int i = 0; i < all_tied_gdofs_array.Size() / 2; ++i)
            {
               subdomains[i].push_back(owned_tied_gdofs.Find(all_tied_gdofs_array[2 * i]));
               subdomains[i].push_back(owned_tied_gdofs.Find(all_tied_gdofs_array[2 * i + 1]));

               const int *cols = merged_A.GetRowColumns(all_tied_gdofs_array[2 * i]);
               const int row_size = merged_A.RowSize(all_tied_gdofs_array[2 * i]);

               for (int j = 0; j < row_size; ++j)
               {
                  const HYPRE_BigInt c = cols[j];

                  if (c == all_tied_gdofs_array[2 * i] || c == all_tied_gdofs_array[2 * i + 1])
                  {
                     continue;
                  }

                  if (all_tied_gdofs_array.Find(c) != -1)
                  {
                     subdomains[i].push_back(owned_tied_gdofs.Find(c));
                  }
               }
            }
         }
         scwharzprec->SetCustomSubdomains(subdomains, *PTAP);
      }
      else if (amg_subspace_filter)
      {
         subspacesolver = new HypreBoomerAMG(*PTAP);
         HypreBoomerAMG * boomeramgsolver = dynamic_cast<HypreBoomerAMG*>(subspacesolver);
         boomeramgsolver->SetPrintLevel(1);
         boomeramgsolver->SetRelaxType(8);
      }
      else
      {
         subspacesolver = new ParallelDirectSolver(comm, "superlu");
         ParallelDirectSolver *subspacedirectsolver =
            dynamic_cast<ParallelDirectSolver *>(subspacesolver);
         subspacedirectsolver->SetPrintLevel(0);
      }

      if ((amg_subspace_filter || schwarz_subspace_filter) && subdomain_cg_iters > 0)
      {
         Solver * subprec = subspacesolver;
         subspacesolver = new CGSolver();
         CGSolver * cgsubspacesolver = dynamic_cast<CGSolver*>(subspacesolver);
         cgsubspacesolver->SetMaxIter(subdomain_cg_iters);
         cgsubspacesolver->SetRelTol(0);
         cgsubspacesolver->SetAbsTol(0);
         cgsubspacesolver->SetPrintLevel(1);
         if (precondition_subspace_cg)
         {
            cgsubspacesolver->SetPreconditioner(*subprec);
         }
      }
      amgfprec->SetFilteredSubspaceSolver(*subspacesolver);

#ifdef HYPRE_BIGINT
         delete [] J;
#endif
      }
      else
      {
         prec = new HypreBoomerAMG();
         auto *amgprec = dynamic_cast<HypreBoomerAMG *>(prec);
         amgprec->SetPrintLevel(0);
         amgprec->SetStrengthThresh(0.125);

         if (one_level_amg)
         {
            amgprec->SetMaxLevels(1);
         }
      }
   }

   // --------------------------------------------------------------------------
   // Solve
   // --------------------------------------------------------------------------
   if (use_schur_complement)
   {
      // Schur complement approach: eliminate tied DOFs
      // System partitioning:
      // [A_ff  A_fc] [x_f]   [b_f]
      // [A_cf  A_cc] [x_c] = [b_c]
      //
      // Reduced system: (A_ff - A_fc * A_cc^{-1} * A_cf) x_f = b_f - A_fc * A_cc^{-1} * b_c

      const HYPRE_BigInt global_size = fespace.GlobalTrueVSize();
      std::set<HYPRE_BigInt> tied_dofs_set(all_tied_gdofs_array.GetData(),
                                            all_tied_gdofs_array.GetData() + all_tied_gdofs_array.Size());

      const HYPRE_BigInt local_row_start = tiedA->GetRowStarts()[0];
      const HYPRE_BigInt local_row_end = tiedA->GetRowStarts()[1];
      const int local_num_rows = tiedA->GetNumRows();

      // Build projectors for free and constrained DOFs
      int nrows_free = 0;
      int nrows_constrained = 0;

      for (int i = 0; i < local_num_rows; ++i)
      {
         if (tied_dofs_set.find(local_row_start + i) == tied_dofs_set.end())
         {
            nrows_free++;
         }
         else
         {
            nrows_constrained++;
         }
      }

      Array<int> free_dof_indices, constrained_dof_indices;
      for (int i = 0; i < local_num_rows; ++i)
      {
         if (tied_dofs_set.find(local_row_start + i) == tied_dofs_set.end())
         {
            free_dof_indices.Append(i);
         }
         else
         {
            constrained_dof_indices.Append(i);
         }
      }

      // Build free DOF projector
      SparseMatrix Pf_local(nrows_free, global_size);
      for (int i = 0; i < nrows_free; ++i)
      {
         Pf_local.Set(i, local_row_start + free_dof_indices[i], 1.0);
      }
      Pf_local.Finalize();

      HYPRE_BigInt rows_f[2];
      HYPRE_BigInt nrows_free_bigint = nrows_free;
      HYPRE_BigInt row_offset_free;
      MPI_Scan(&nrows_free_bigint, &row_offset_free, 1, HYPRE_MPI_BIG_INT, MPI_SUM, comm);
      row_offset_free -= nrows_free_bigint;
      rows_f[0] = row_offset_free;
      rows_f[1] = row_offset_free + nrows_free;

      HYPRE_BigInt glob_nrows_free;
      MPI_Allreduce(&nrows_free_bigint, &glob_nrows_free, 1, HYPRE_MPI_BIG_INT, MPI_SUM, comm);

      HYPRE_BigInt *J_f;
#ifndef HYPRE_BIGINT
      J_f = Pf_local.GetJ();
#else
      J_f = new HYPRE_BigInt[Pf_local.NumNonZeroElems()];
      for (int i = 0; i < Pf_local.NumNonZeroElems(); i++)
      {
         J_f[i] = Pf_local.GetJ()[i];
      }
#endif

      HypreParMatrix *P_free_T = new HypreParMatrix(comm,
                                                     nrows_free, glob_nrows_free,
                                                     global_size,
                                                     Pf_local.GetI(), J_f, Pf_local.GetData(),
                                                     rows_f, fespace.GetTrueDofOffsets());

      HypreParMatrix *P_free = P_free_T->Transpose();

      // Build constrained DOF projector
      SparseMatrix Pc_local(nrows_constrained, global_size);
      for (int i = 0; i < nrows_constrained; ++i)
      {
         Pc_local.Set(i, local_row_start + constrained_dof_indices[i], 1.0);
      }
      Pc_local.Finalize();

      HYPRE_BigInt rows_c[2];
      HYPRE_BigInt nrows_constrained_bigint = nrows_constrained;
      HYPRE_BigInt row_offset_constrained;
      MPI_Scan(&nrows_constrained_bigint, &row_offset_constrained, 1, HYPRE_MPI_BIG_INT, MPI_SUM, comm);
      row_offset_constrained -= nrows_constrained_bigint;
      rows_c[0] = row_offset_constrained;
      rows_c[1] = row_offset_constrained + nrows_constrained;

      HYPRE_BigInt glob_nrows_constrained;
      MPI_Allreduce(&nrows_constrained_bigint, &glob_nrows_constrained, 1, HYPRE_MPI_BIG_INT, MPI_SUM, comm);

      HYPRE_BigInt *J_c;
#ifndef HYPRE_BIGINT
      J_c = Pc_local.GetJ();
#else
      J_c = new HYPRE_BigInt[Pc_local.NumNonZeroElems()];
      for (int i = 0; i < Pc_local.NumNonZeroElems(); i++)
      {
         J_c[i] = Pc_local.GetJ()[i];
      }
#endif

      HypreParMatrix *P_constrained_T = new HypreParMatrix(comm,
                                                           nrows_constrained, glob_nrows_constrained,
                                                           global_size,
                                                           Pc_local.GetI(), J_c, Pc_local.GetData(),
                                                           rows_c, fespace.GetTrueDofOffsets());

      HypreParMatrix *P_constrained = P_constrained_T->Transpose();

      // Extract blocks: A_ff, A_fc, A_cf, A_cc
      HypreParMatrix *A_Pf = ParMult(tiedA, P_free);
      HypreParMatrix *A_ff = ParMult(P_free_T, A_Pf);

      HypreParMatrix *A_Pc = ParMult(tiedA, P_constrained);
      HypreParMatrix *A_fc = ParMult(P_free_T, A_Pc);
      HypreParMatrix *A_cf = ParMult(P_constrained_T, A_Pf);
      HypreParMatrix *A_cc = ParMult(P_constrained_T, A_Pc);

      // Extract RHS blocks
      HypreParVector *b_f = new HypreParVector(comm, glob_nrows_free, rows_f);
      HypreParVector *b_c = new HypreParVector(comm, glob_nrows_constrained, rows_c);
      P_free_T->Mult(B, *b_f);
      P_constrained_T->Mult(B, *b_c);

      // Compute A_cc^{-1} * b_c and A_cc^{-1} * A_cf using direct solve
      // For single MPI rank, we can use a direct solver or iterative solver
      HypreParVector *A_cc_inv_b_c = new HypreParVector(comm, glob_nrows_constrained, rows_c);
      *A_cc_inv_b_c = 0.0;

      // Setup solver for A_cc
      HypreBoomerAMG *A_cc_inv_prec = new HypreBoomerAMG(*A_cc);
      A_cc_inv_prec->SetPrintLevel(0);

      CGSolver A_cc_inv_solver(comm);
      A_cc_inv_solver.SetRelTol(1e-12);
      A_cc_inv_solver.SetMaxIter(1000);
      A_cc_inv_solver.SetPrintLevel(0);
      A_cc_inv_solver.SetPreconditioner(*A_cc_inv_prec);
      A_cc_inv_solver.SetOperator(*A_cc);

      // Solve A_cc * y = b_c
      A_cc_inv_solver.Mult(*b_c, *A_cc_inv_b_c);

      // Compute A_fc * A_cc^{-1} * b_c
      HypreParVector *A_fc_A_cc_inv_b_c = new HypreParVector(comm, glob_nrows_free, rows_f);
      A_fc->Mult(*A_cc_inv_b_c, *A_fc_A_cc_inv_b_c);

      // Compute modified RHS: b_f - A_fc * A_cc^{-1} * b_c
      HypreParVector *b_schur = new HypreParVector(comm, glob_nrows_free, rows_f);
      add(*b_f, -1.0, *A_fc_A_cc_inv_b_c, *b_schur);

      // Compute A_cc^{-1} * A_cf by solving A_cc * Y = A_cf for each column
      // This is expensive, so we'll form the Schur complement S = A_ff - A_fc * A_cc^{-1} * A_cf
      // by computing A_fc * (A_cc^{-1} * A_cf) column by column

      // For simplicity with single rank, convert to dense if small enough
      if (Mpi::Root())
      {
         mfem::out << "Building Schur complement matrix..." << endl;
         mfem::out << "  Free DOFs: " << glob_nrows_free << endl;
         mfem::out << "  Constrained DOFs: " << glob_nrows_constrained << endl;
      }

      // Build Schur complement: S = A_ff - A_fc * A_cc^{-1} * A_cf
      // We compute this by: for each column j of A_cf, solve A_cc * y_j = A_cf[:,j]
      // then compute A_fc * y_j to get column j of A_fc * A_cc^{-1} * A_cf

      HypreParMatrix *S = new HypreParMatrix(*A_ff);  // Start with A_ff

      // Iterate over columns of A_cf (equivalently, rows of A_fc)
      // Extract A_cf and A_fc as local sparse matrices
      SparseMatrix A_cf_diag, A_cf_offd;
      A_cf->MergeDiagAndOffd(A_cf_diag);

      SparseMatrix A_fc_diag, A_fc_offd;
      A_fc->MergeDiagAndOffd(A_fc_diag);

      // For each free DOF i, we need to compute the Schur correction
      // S[i,j] -= sum_k A_fc[i,k] * (A_cc^{-1})[k,l] * A_cf[l,j]

      // Compute A_cc^{-1} * A_cf by solving multiple RHS
      DenseMatrix A_cc_inv_A_cf_dense(A_cc->Height(), A_cf->Width());
      A_cc_inv_A_cf_dense = 0.0;

      for (int j = 0; j < A_cf->Width(); ++j)
      {
         HypreParVector rhs_col(comm, glob_nrows_constrained, rows_c);
         HypreParVector sol_col(comm, glob_nrows_constrained, rows_c);
         rhs_col = 0.0;
         sol_col = 0.0;

         // Extract column j of A_cf
         for (int i = 0; i < A_cf_diag.Height(); ++i)
         {
            for (int k = A_cf_diag.GetI()[i]; k < A_cf_diag.GetI()[i+1]; ++k)
            {
               if (A_cf_diag.GetJ()[k] == j)
               {
                  rhs_col(i) = A_cf_diag.GetData()[k];
               }
            }
         }

         // Solve A_cc * sol = rhs_col
         A_cc_inv_solver.Mult(rhs_col, sol_col);

         // Store in dense matrix
         for (int i = 0; i < A_cc->Height(); ++i)
         {
            A_cc_inv_A_cf_dense(i, j) = sol_col(i);
         }
      }

      // Now compute A_fc * (A_cc^{-1} * A_cf) and subtract from A_ff
      SparseMatrix S_correction(S->Height(), S->Width());

      for (int i = 0; i < A_fc_diag.Height(); ++i)
      {
         for (int j = 0; j < A_cf->Width(); ++j)
         {
            real_t val = 0.0;
            // Compute dot product of row i of A_fc with column j of A_cc^{-1} * A_cf
            for (int k = A_fc_diag.GetI()[i]; k < A_fc_diag.GetI()[i+1]; ++k)
            {
               int col_k = A_fc_diag.GetJ()[k];
               val += A_fc_diag.GetData()[k] * A_cc_inv_A_cf_dense(col_k, j);
            }
            if (std::abs(val) > 1e-14)
            {
               S_correction.Add(i, j, -val);
            }
         }
      }
      S_correction.Finalize();

      // Add correction to S
      HypreParMatrix *S_correction_par = BuildHypreParMatrixFromMerged(S_correction, *S);
      HypreParMatrix *S_final = ParAdd(S, S_correction_par);

      // Setup AMG preconditioner on Schur complement
      HypreBoomerAMG *schur_prec = new HypreBoomerAMG(*S_final);
      schur_prec->SetPrintLevel(1);
      schur_prec->SetRelaxType(8);
      if (one_level_amg)
      {
         schur_prec->SetMaxLevels(1);
      }

      // Solve reduced system
      CGSolver cg_reduced(comm);
      cg_reduced.SetRelTol(1e-12);
      cg_reduced.SetMaxIter(pcg_max_iters);
      cg_reduced.SetPrintLevel(1);
      cg_reduced.SetPreconditioner(*schur_prec);
      cg_reduced.SetOperator(*S_final);

      HypreParVector *x_free = new HypreParVector(comm, glob_nrows_free, rows_f);
      *x_free = 0.0;
      cg_reduced.Mult(*b_schur, *x_free);

      // Back-substitute to get constrained DOFs: x_c = A_cc^{-1} * (b_c - A_cf * x_f)
      HypreParVector *A_cf_x_f = new HypreParVector(comm, glob_nrows_constrained, rows_c);
      A_cf->Mult(*x_free, *A_cf_x_f);

      HypreParVector *rhs_constrained = new HypreParVector(comm, glob_nrows_constrained, rows_c);
      add(*b_c, -1.0, *A_cf_x_f, *rhs_constrained);

      HypreParVector *x_constrained = new HypreParVector(comm, glob_nrows_constrained, rows_c);
      *x_constrained = 0.0;
      A_cc_inv_solver.Mult(*rhs_constrained, *x_constrained);

      // Assemble full solution
      HypreParVector X_free_full(comm, global_size, fespace.GetTrueDofOffsets());
      HypreParVector X_constrained_full(comm, global_size, fespace.GetTrueDofOffsets());
      X_free_full = 0.0;
      X_constrained_full = 0.0;

      P_free->Mult(*x_free, X_free_full);
      P_constrained->Mult(*x_constrained, X_constrained_full);

      add(X_free_full, 1.0, X_constrained_full, X);

      // Clean up
      delete x_free;
      delete x_constrained;
      delete rhs_constrained;
      delete A_cf_x_f;
      delete b_schur;
      delete A_fc_A_cc_inv_b_c;
      delete A_cc_inv_b_c;
      delete b_f;
      delete b_c;
      delete schur_prec;
      delete S_final;
      delete S_correction_par;
      delete S;
      delete A_cc_inv_prec;
      delete A_ff;
      delete A_fc;
      delete A_cf;
      delete A_cc;
      delete A_Pf;
      delete A_Pc;
      delete P_free;
      delete P_free_T;
      delete P_constrained;
      delete P_constrained_T;
#ifdef HYPRE_BIGINT
      delete [] J_f;
      delete [] J_c;
#endif
   }
   else
   {
      CGSolver cg(comm);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(pcg_max_iters);
      cg.SetPrintLevel(1);
      cg.SetPreconditioner(*prec);
      cg.SetOperator(*tiedA);
      cg.Mult(B, X);
   }

   a.RecoverFEMSolution(X, b, x);

   if (visualize_spectrum)
   {
      DenseMatrix Adense(PTAP->Height(), PTAP->Width());
      DenseMatrix MinvAdense(PTAP->Height(), PTAP->Width());
      Adense = 0.0;
      MinvAdense = 0.0;

      for (int i = 0; i < PTAP->Height(); i++)
      {
         for (int j = PTAP->GetDiagMemoryI()[i]; j < PTAP->GetDiagMemoryI()[i + 1]; j++)
         {
            Adense(i, PTAP->GetDiagMemoryJ()[j]) = PTAP->GetDiagMemoryData()[j];
         }
      }

      for (int i = 0; i < PTAP->Height(); i++)
      {
         Vector Acol, MinvAcol;
         Adense.GetColumn(i, Acol);
         HypreParVector HypreAcol(MPI_COMM_WORLD, PTAP->Height(), Acol, 0, PTAP->GetColStarts());
         HypreParVector HypreMinvAcol(HypreAcol);
         subspacesolver->Mult(Acol, HypreMinvAcol);
         MinvAdense.SetCol(i, HypreMinvAcol);
      }

      std::vector<double> precevals = {1, 1};
      SymmetricEigenDecomposition(MinvAdense, precevals);

      std::vector<double> evals;
      SymmetricEigenDecomposition(Adense, evals);

      auto min_abs_eval = *std::min_element(evals.begin(), evals.end(),
         [](auto a, auto b) { return std::abs(a) < std::abs(b); });

      auto min_abs_preceval = *std::min_element(precevals.begin(), precevals.end(),
         [](auto a, auto b) { return std::abs(a) < std::abs(b); });

      mfem::out << "Smallest eigenvalue of subspace: " << evals.front() << endl;
      mfem::out << "Largest eigenvalue of subspace: " << evals.back() << endl;
      mfem::out << "Smallest absolute eigenvalue of subspace: " << min_abs_eval << endl;
      mfem::out << "Condition number of subspace: " << evals.back() / evals.front() << endl;

      mfem::out << "Smallest eigenvalue of preconditioned subspace: " << precevals.front() << endl;
      mfem::out << "Largest eigenvalue of preconditioned subspace: " << precevals.back() << endl;
      mfem::out << "Smallest absolute eigenvalue of preconditioned subspace: " << min_abs_preceval << endl;
      mfem::out << "Condition number of preconditioned subspace: " << precevals.back() / precevals.front() << endl;

      return 0;

      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      
      for (int eigi = evals.size() - 1; eigi >= 0; --eigi)
      {
         const real_t eval = evals[eigi];
         Vector subX;
         Adense.GetColumn(eigi, subX);

         P_tied->Mult(subX, x);

         sol_sock << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() << "\n"
                  << "solution\n" << pmesh << x
                  << "window_title 'Eigenmode " << eigi+1 
                  << ", Lambda = " << evals[eigi] << "'" << endl << flush;

         if (eigi == evals.size() - 1)
         {
            char tempc;
            cin >> tempc;
         }

         std::ostringstream name;
         name << "eigenmode_" << eigi << "_lambda_" << eval << ".png";
         sol_sock << "screenshot " << name.str() << "\n";

         
         continue;

         char c;
         if (Mpi::Root())
         {
            cout << "press (q)uit or (c)ontinue --> " << endl;
            cin >> c;
         }
         MPI_Bcast(&c, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

         if (c != 'c')
         {
            break;
         }
      }

      return 0;
   }

   // --------------------------------------------------------------------------
   // Visualization
   // --------------------------------------------------------------------------
   if (visualization)
   {
      //x.Save("tied-poisson-sol");
      //pmesh.Save("tied-poisson-mesh");

      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);

      sol_sock << "parallel " << Mpi::WorldSize() << " "
               << Mpi::WorldRank() << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << x << flush;
   }

   delete tiedA;

   if (!use_schur_complement)
   {
      delete prec;
      delete PTAP;
      delete P_tied;
      delete P_tied_T;

      if (do_amgf)
      {
         delete subspacesolver;
      }
   }

   return 0;
}