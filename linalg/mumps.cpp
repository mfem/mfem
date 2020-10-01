// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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

#ifdef MFEM_USE_MUMPS
#ifdef MFEM_USE_MPI

#include "mumps.hpp"

namespace mfem
{
MUMPSSolver::~MUMPSSolver()
{
   if (id)
   {
      id->job = -2;
      dmumps_c(id);
      delete[] J;
      delete[] I;
      delete [] data;
   }
}

void MUMPSSolver::SetParameters()
{
   // output stream for error messages
   id->ICNTL(1) = 6;

   // output stream for diagnosting printing local to each proc
   id->ICNTL(2) = 6;

   // output stream for global info
   id->ICNTL(3) = 6;

   // Level of error printing
   id->ICNTL(4) = print_level;

   //input matrix format (assembled)
   id->ICNTL(5) = 0;

   // Use A or A^T
   id->ICNTL(9) = 1;

   // Iterative refinement (disabled)
   id->ICNTL(10) = 0;

   // Error analysis-statistics (disabled)
   id->ICNTL(11) = 0;

   // Use of ScaLAPACK (Parallel factorization on root)
   id->ICNTL(13) = 0;

   // Percentage increase of estimated workspace (default = 20%)
   id->ICNTL(14) = 20;

   // Number of OpenMP threads (default)
   id->ICNTL(16) = 0;

   // Matrix input format (distributed)
   id->ICNTL(18) = 3;

   // Schur complement (no Schur complement matrix returned)
   id->ICNTL(19) = 0;

#if MFEM_MUMPS_VERSION >= 530
   // Distributed RHS and Sol
   id->ICNTL(20) = 10;

   id->ICNTL(21) = 1;
#else
   // Centralized RHS and Sol
   id->ICNTL(20) = 0;

   id->ICNTL(21) = 0;
#endif
   // Out of core factorization and solve (disabled)
   id->ICNTL(22) = 0;

   // Max size of working memory (default = based on estimates)
   id->ICNTL(23) = 0;
}

void MUMPSSolver::SetOperator(const Operator &op)
{
   // Verify that the operator is a HypreParMatrix
   auto APtr = dynamic_cast<const HypreParMatrix *>(&op);
   MFEM_VERIFY(APtr, "Not compatible matrix type");
   height = op.Height();
   width = op.Width();

   comm = APtr->GetComm();
   MPI_Comm_size(comm, &numProcs);
   MPI_Comm_rank(comm, &myid);

   hypre_ParCSRMatrix *parcsr_op
      = (hypre_ParCSRMatrix *) const_cast<HypreParMatrix &>(*APtr);
   hypre_CSRMatrix *csr_op = hypre_MergeDiagAndOffd(parcsr_op);
#if MFEM_HYPRE_VERSION >= 21600
   hypre_CSRMatrixBigJtoJ(csr_op);
#endif

   int *Iptr = csr_op->i;
   int *Jptr = csr_op->j;
   int n_loc = csr_op->num_rows;

   row_start = parcsr_op->first_row_index;

   int nnz;
   if (sym)
   {
      // count nnz;
      nnz = 0;
      int k = 0;
      for (int i = 0; i < n_loc; i++)
      {
         for (int j = Iptr[i]; j < Iptr[i + 1]; j++)
         {
            int ii = row_start + i + 1;
            int jj = Jptr[k] + 1;
            k++;
            if (ii>=jj) { nnz++; }
         }
      }
   }
   else
   {
      nnz = csr_op->num_nonzeros;
   }

   I = new int[nnz];
   J = new int[nnz];

   int k = 0;
   if (sym)
   {
      int l = 0;
      data = new double[nnz];
      for (int i = 0; i < n_loc; i++)
      {
         for (int j = Iptr[i]; j < Iptr[i + 1]; j++)
         {
            // Global I and J indices in 1-based index (for fortran)
            int ii = row_start + i + 1;
            int jj = Jptr[k] + 1;
            if (ii>=jj)
            {
               I[l] = ii;
               J[l] = jj;
               data[l++] = csr_op->data[k];
            }
            k++;
         }
      }
   }
   else
   {
      for (int i = 0; i < n_loc; i++)
      {
         for (int j = Iptr[i]; j < Iptr[i + 1]; j++)
         {
            // Global I and J indices in 1-based index (for fortran)
            I[k] = row_start + i + 1;
            J[k] = Jptr[k] + 1;
            k++;
         }
      }
      data = csr_op->data;
   }

   // new MUMPS object
   id = new DMUMPS_STRUC_C;
   // C to Fortran communicator
   id->comm_fortran = (MUMPS_INT) MPI_Comm_c2f(comm);

   // Host is involved in computation
   id->par = 1;

   // Unsymmetric matrix
   id->sym = sym;

   // Mumps init
   id->job = -1;
   dmumps_c(id);

   // Set MUMPS default parameters
   SetParameters();

   // Global number of rows
   id->n = parcsr_op->global_num_rows;

   // Number of non zeros on the processor
   id->nnz_loc = nnz;

   // Distributed row array
   id->irn_loc = I;

   // Distributed column array
   id->jcn_loc = J;

   // Distributed data array
   id->a_loc = data;

   // MUMPS Analysis
   id->job = 1;
   dmumps_c(id);

   // MUMPS Factorization
   id->job = 2;
   dmumps_c(id);

   // matrix can be destroyed now
   hypre_CSRMatrixDestroy(csr_op);
   if (!sym) { data = nullptr; }

#if MFEM_MUMPS_VERSION >= 530
   irhs_loc.SetSize(n_loc);
   for (int i = 0; i < n_loc; i++)
   {
      irhs_loc[i] = row_start + i + 1;
   }
   row_starts.SetSize(numProcs);
   MPI_Allgather(&row_start, 1, MPI_INT, row_starts, 1, MPI_INT, comm);
   sol_loc.SetSize(id->INFO(23));
   isol_loc.SetSize(id->INFO(23));
#else
   if (myid == 0)
   {
      rhs_glob.SetSize(parcsr_op->global_num_rows);
      recv_counts.SetSize(numProcs);
   }
   MPI_Gather(&n_loc, 1, MPI_INT, recv_counts, 1, MPI_INT, 0, comm);
   if (myid == 0)
   {
      displs.SetSize(numProcs); displs[0] = 0;
      int s = 0;
      for (int k = 0; k < numProcs-1; k++)
      {
         s += recv_counts[k];
         displs[k+1] = s;
      }
   }
#endif
}

void MUMPSSolver::Mult(const Vector &x, Vector &y) const
{
#if MFEM_MUMPS_VERSION >= 530
   id->nloc_rhs = x.Size();
   id->lrhs_loc = x.Size();
   id->rhs_loc = x.GetData();
   id->irhs_loc = const_cast<int *>(irhs_loc.GetData());
   id->sol_loc = sol_loc.GetData();
   id->lsol_loc = id->INFO(23);
   id->isol_loc = const_cast<int *>(isol_loc.GetData());
   id->job = 3;
   dmumps_c(id);
   RedistributeSol(isol_loc, sol_loc, y);
#else
   MPI_Gatherv(x.GetData(), x.Size(), MPI_DOUBLE,
               rhs_glob.GetData(), recv_counts,
               displs, MPI_DOUBLE, 0, comm);
   if (myid == 0)
   {
      id->rhs = rhs_glob.GetData();
   }
   id->job = 3;
   dmumps_c(id);
   MPI_Scatterv(rhs_glob.GetData(), recv_counts, displs,
                MPI_DOUBLE, y.GetData(), y.Size(),
                MPI_DOUBLE, 0, comm);
#endif
}

void MUMPSSolver::MultTranspose(const Vector &x, Vector &y) const
{
   id->ICNTL(9) = 0;
   Mult(x,y);
}

#if MFEM_MUMPS_VERSION >= 530
int MUMPSSolver::GetRowRank(int i, const Array<int> &row_starts_) const
{
   if (row_starts_.Size() == 1)
   {
      return 0;
   }
   auto up = std::upper_bound(row_starts_.begin(), row_starts_.end(), i);
   return std::distance(row_starts_.begin(), up) - 1;
}

void MUMPSSolver::RedistributeSol(const Array<int> &row_map,
                                  const Vector &x,
                                  Vector &y) const
{
   MFEM_VERIFY(row_map.Size() == x.Size(), "Inconcistent sizes");
   int size = x.Size();

   // compute send_count
   Array<int> send_count(numProcs);
   send_count = 0;
   for (int i = 0; i < size; i++)
   {
      int j = row_map[i] - 1; //fix to 0-based indexing
      int row_rank = GetRowRank(j, row_starts);
      send_count[row_rank]++; // both for val and global index
   }

   // compute recv_count
   Array<int> recv_count(numProcs);
   MPI_Alltoall(send_count, 1, MPI_INT, recv_count, 1, MPI_INT, comm);

   // compute offsets
   Array<int> send_displ(numProcs);
   send_displ[0] = 0;
   Array<int> recv_displ(numProcs);
   recv_displ[0] = 0;
   for (int k = 0; k < numProcs - 1; k++)
   {
      send_displ[k + 1] = send_displ[k] + send_count[k];
      recv_displ[k + 1] = recv_displ[k] + recv_count[k];
   }
   int sbuff_size = send_count.Sum();
   int rbuff_size = recv_count.Sum();

   Array<int> sendbuf_index(sbuff_size);
   sendbuf_index = 0;
   Array<double> sendbuf_value(sbuff_size);
   sendbuf_value = 0;
   Array<int> soffs(numProcs);
   soffs = 0;

   // Fill in send buffers
   for (int i = 0; i < size; i++)
   {
      int j = row_map[i] - 1; //fix to 0-based indexing
      int row_rank = GetRowRank(j, row_starts);
      int k = send_displ[row_rank] + soffs[row_rank];
      sendbuf_index[k] = j;
      sendbuf_value[k] = x(i);
      soffs[row_rank]++;
   }

   // communicate
   Array<int> recvbuf_index(rbuff_size);
   Array<double> recvbuf_value(rbuff_size);
   MPI_Alltoallv(sendbuf_index,
                 send_count,
                 send_displ,
                 MPI_INT,
                 recvbuf_index,
                 recv_count,
                 recv_displ,
                 MPI_INT,
                 comm);
   MPI_Alltoallv(sendbuf_value,
                 send_count,
                 send_displ,
                 MPI_DOUBLE,
                 recvbuf_value,
                 recv_count,
                 recv_displ,
                 MPI_DOUBLE,
                 comm);

   // Unpack recv buffer
   for (int i = 0; i < rbuff_size; i++)
   {
      int local_index = recvbuf_index[i] - row_start;
      y(local_index) = recvbuf_value[i];
   }
}
#endif

} // namespace mfem

#endif // MFEM_USE_MPI
#endif // MFEM_USE_MUMPS
