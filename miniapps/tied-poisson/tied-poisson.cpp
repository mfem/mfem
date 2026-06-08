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

using namespace std;
using namespace mfem;

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
                                        MPI_Comm comm)
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

      const HYPRE_BigInt paired_gdof = it->second;
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
         emit(global_row, vk, -value * w);
         emit(vk, global_row, -value * w);
      }

      for (int k = 0; k < m; ++k)
      {
         const HYPRE_BigInt vk = group[k];
         for (int l = 0; l < m; ++l)
         {
            const HYPRE_BigInt vl = group[l];
            emit(vk, vl, value * w * w);
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
   bool do_amgf = false;
   bool iterative_filter = false;
   bool one_level_amg = false;
   bool symmetric_tie = false;

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
   args.AddOption(&iterative_filter, "-iterative-filter", "--iterative-filter",
                  "-no-iterative-filter", "--no-iterative-filter",
                  "Enable or disable use of Schwarz solver on subspace in AMGF.");
   args.AddOption(&one_level_amg, "-one-level-amg", "--one-level-amg",
                  "-no-one-level-amg", "--no-one-level-amg",
                  "Enable or disable using a non multigrid preconditioner.");
   args.AddOption(&symmetric_tie, "-symmetric-tie", "-no-symmetric-tie",
                  "--symmetric-tie", "--no-symmetric-tie",
                  "Tie both faces to each other.");
   args.ParseCheck();

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
                                 comm);

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

   if (do_amgf)
   {
      prec = new AMGFSolver();
      auto *amgfprec = dynamic_cast<AMGFSolver *>(prec);
      amgfprec->GetAMG().SetPrintLevel(0);
      amgfprec->GetAMG().SetStrengthThresh(0.5);

      if (one_level_amg)
      {
         amgfprec->GetAMG().SetMaxLevels(1);
      }

      if (iterative_filter)
      {
         subspacesolver = new HypreBoomerAMG();
         HypreBoomerAMG *subspaceamgsolver =
            dynamic_cast<HypreBoomerAMG *>(subspacesolver);
         subspaceamgsolver->SetMaxLevels(1);
         subspaceamgsolver->SetRelaxType(0);
         subspaceamgsolver->SetPrintLevel(0);
         HYPRE_BoomerAMGSetNumFunctions(*subspaceamgsolver, 1);
         HYPRE_BoomerAMGSetDomainType(*subspaceamgsolver, 0);
         HYPRE_BoomerAMGSetOverlap(*subspaceamgsolver, 1);
         HYPRE_BoomerAMGSetVariant(*subspaceamgsolver, 0);
         HYPRE_BoomerAMGSetSchwarzRlxWeight(*subspaceamgsolver, 0);
         HYPRE_BoomerAMGSetSmoothType(*subspaceamgsolver, 6);
         HYPRE_BoomerAMGSetSmoothNumLevels(*subspaceamgsolver, 1);
      }
      else
      {
         subspacesolver = new ParallelDirectSolver(comm, "superlu");
         ParallelDirectSolver *subspacedirectsolver =
            dynamic_cast<ParallelDirectSolver *>(subspacesolver);
         subspacedirectsolver->SetPrintLevel(0);
      }

      amgfprec->SetFilteredSubspaceSolver(*subspacesolver);

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
      amgfprec->SetFilteredSubspaceTransferOperator(*P_tied);

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

   // --------------------------------------------------------------------------
   // Solve
   // --------------------------------------------------------------------------
   CGSolver cg(comm);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(pcg_max_iters);
   cg.SetPrintLevel(1);
   cg.SetPreconditioner(*prec);
   cg.SetOperator(*tiedA);
   cg.Mult(B, X);

   a.RecoverFEMSolution(X, b, x);

   if (Mpi::Root())
   {
      mfem::out << "Solutions saved to tied-poisson-sol and tied-poisson-mesh"
                << endl;
   }

   // --------------------------------------------------------------------------
   // Visualization
   // --------------------------------------------------------------------------
   if (visualization)
   {
      x.Save("tied-poisson-sol");
      pmesh.Save("tied-poisson-mesh");

      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << Mpi::WorldSize() << " "
               << Mpi::WorldRank() << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << x << flush;
   }

   delete tiedA;
   delete prec;

   if (do_amgf)
   {
      delete P_tied_T;
      delete P_tied;
      delete subspacesolver;
   }

   return 0;
}