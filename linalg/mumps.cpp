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

#ifdef HYPRE_BIGINT
#error "MUMPSSolver requires HYPRE_Int == int, for now."
#endif

// macro s.t. indices match MUMPS documentation
#define MUMPS_ICNTL(I) icntl[(I) -1]
#define MUMPS_INFO(I) info[(I) -1]

namespace mfem
{

void MUMPSSolver::SetOperator(const Operator &op)
{
   auto APtr = dynamic_cast<const HypreParMatrix *>(&op);

   MFEM_VERIFY(APtr, "Not compatible matrix type");

   height = op.Height();
   width = op.Width();

   comm = APtr->GetComm();
   MPI_Comm_size(comm, &numProcs);
   MPI_Comm_rank(comm, &myid);

   auto parcsr_op = (hypre_ParCSRMatrix *) const_cast<HypreParMatrix &>(*APtr);

   hypre_CSRMatrix *csr_op = hypre_MergeDiagAndOffd(parcsr_op);
#if MFEM_HYPRE_VERSION >= 21600
   hypre_CSRMatrixBigJtoJ(csr_op);
#endif

   int *Iptr = csr_op->i;
   int *Jptr = csr_op->j;
   int n_loc = csr_op->num_rows;

   row_start = parcsr_op->first_row_index;

   MUMPS_INT8 nnz = 0;
   if (mat_type)
   {
      // count nnz in case of symmetric mode
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

   int * I = new int[nnz];
   int * J = new int[nnz];

   // Fill in I and J arrays for
   // COO format in 1-based indexing
   int k = 0;
   double * data;
   if (mat_type)
   {
      int l = 0;
      data = new double[nnz];
      for (int i = 0; i < n_loc; i++)
      {
         for (int j = Iptr[i]; j < Iptr[i + 1]; j++)
         {
            int ii = row_start + i + 1;
            int jj = Jptr[k] + 1;
            if (ii >= jj)
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
            I[k] = row_start + i + 1;
            J[k] = Jptr[k] + 1;
            k++;
         }
      }
      data = csr_op->data;
   }

   // new MUMPS object
   if (id)
   {
      id->job = -2;
      dmumps_c(id);
      delete id;
   }
   id = new DMUMPS_STRUC_C;
   // C to Fortran communicator
   id->comm_fortran = (MUMPS_INT) MPI_Comm_c2f(comm);

   // Host is involved in computation
   id->par = 1;

   id->sym = mat_type;

   // MUMPS init
   id->job = -1;
   dmumps_c(id);

   // Set MUMPS default parameters
   SetParameters();

   id->n = parcsr_op->global_num_rows;

   id->nnz_loc = nnz;

   id->irn_loc = I;

   id->jcn_loc = J;

   id->a_loc = data;

   // MUMPS Analysis
   id->job = 1;
   dmumps_c(id);

   // MUMPS Factorization
   id->job = 2;
   dmumps_c(id);

   hypre_CSRMatrixDestroy(csr_op);
   delete [] I;
   delete [] J;
   if (mat_type) { delete [] data; }

#if MFEM_MUMPS_VERSION >= 530
   delete [] irhs_loc;
   irhs_loc = new int[n_loc];
   for (int i = 0; i < n_loc; i++)
   {
      irhs_loc[i] = row_start + i + 1;
   }
   row_starts.SetSize(numProcs);
   MPI_Allgather(&row_start, 1, MPI_INT, row_starts, 1, MPI_INT, comm);
#else
   if (myid == 0)
   {
      delete [] rhs_glob;
      delete [] recv_counts;
      rhs_glob = new double[parcsr_op->global_num_rows];
      recv_counts = new int[numProcs];
   }
   MPI_Gather(&n_loc, 1, MPI_INT, recv_counts, 1, MPI_INT, 0, comm);
   if (myid == 0)
   {
      delete [] displs;
      displs = new int[numProcs];
      displs[0] = 0;
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
   id->irhs_loc = irhs_loc;

   id->lsol_loc = id->MUMPS_INFO(23);
   id->isol_loc = new int[id->MUMPS_INFO(23)];
   id->sol_loc = new double[id->MUMPS_INFO(23)];

   // MUMPS solve
   id->job = 3;
   dmumps_c(id);

   RedistributeSol(id->isol_loc, id->sol_loc, y.GetData());

   delete [] id->sol_loc;
   delete [] id->isol_loc;
#else
   MPI_Gatherv(x.GetData(), x.Size(), MPI_DOUBLE,
               rhs_glob, recv_counts,
               displs, MPI_DOUBLE, 0, comm);

   if (myid == 0) { id->rhs = rhs_glob; }

   // MUMPS solve
   id->job = 3;
   dmumps_c(id);

   MPI_Scatterv(rhs_glob, recv_counts, displs,
                MPI_DOUBLE, y.GetData(), y.Size(),
                MPI_DOUBLE, 0, comm);
#endif
}

void MUMPSSolver::MultTranspose(const Vector &x, Vector &y) const
{
   // Set flag for Transpose Solve
   id->MUMPS_ICNTL(9) = 0;
   Mult(x,y);
   // Reset the flag
   id->MUMPS_ICNTL(9) = 1;

}

void MUMPSSolver::SetPrintLevel(int print_lvl)
{
   print_level = print_lvl;
}

void MUMPSSolver::SetMatrixSymType(MatType mtype)
{
   mat_type = mtype;
}

MUMPSSolver::~MUMPSSolver()
{
   if (id)
   {
#if MFEM_MUMPS_VERSION >= 530
      delete [] irhs_loc;
#else
      delete [] recv_counts;
      delete [] displs;
      delete [] rhs_glob;
#endif
      id->job = -2;
      dmumps_c(id);
      delete id;
   }
}

void MUMPSSolver::SetParameters()
{
   // output stream for error messages
   id->MUMPS_ICNTL(1) = 6;
   // output stream for diagnosting printing local to each proc
   id->MUMPS_ICNTL(2) = 6;
   // output stream for global info
   id->MUMPS_ICNTL(3) = 6;
   // Level of error printing
   id->MUMPS_ICNTL(4) = print_level;
   //input matrix format (assembled)
   id->MUMPS_ICNTL(5) = 0;
   // Use A or A^T
   id->MUMPS_ICNTL(9) = 1;
   // Iterative refinement (disabled)
   id->MUMPS_ICNTL(10) = 0;
   // Error analysis-statistics (disabled)
   id->MUMPS_ICNTL(11) = 0;
   // Use of ScaLAPACK (Parallel factorization on root)
   id->MUMPS_ICNTL(13) = 0;
   // Percentage increase of estimated workspace (default = 20%)
   id->MUMPS_ICNTL(14) = 20;
   // Number of OpenMP threads (default)
   id->MUMPS_ICNTL(16) = 0;
   // Matrix input format (distributed)
   id->MUMPS_ICNTL(18) = 3;
   // Schur complement (no Schur complement matrix returned)
   id->MUMPS_ICNTL(19) = 0;

#if MFEM_MUMPS_VERSION >= 530
   // Distributed RHS
   id->MUMPS_ICNTL(20) = 10;
   // Distributed Sol
   id->MUMPS_ICNTL(21) = 1;
#else
   // Centralized RHS
   id->MUMPS_ICNTL(20) = 0;
   // Centralized Sol
   id->MUMPS_ICNTL(21) = 0;
#endif
   // Out of core factorization and solve (disabled)
   id->MUMPS_ICNTL(22) = 0;
   // Max size of working memory (default = based on estimates)
   id->MUMPS_ICNTL(23) = 0;
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

void MUMPSSolver::RedistributeSol(const int * row_map,
                                  const double * x, double * y) const
{
   int size = id->MUMPS_INFO(23);
   int * send_count = new int[numProcs]();
   for (int i = 0; i < size; i++)
   {
      int j = row_map[i] - 1;
      int row_rank = GetRowRank(j, row_starts);
      if (myid == row_rank) { continue; }
      send_count[row_rank]++;
   }

   int * recv_count = new int[numProcs];
   MPI_Alltoall(send_count, 1, MPI_INT, recv_count, 1, MPI_INT, comm);

   int * send_displ = new int [numProcs]; send_displ[0] = 0;
   int * recv_displ = new int [numProcs]; recv_displ[0] = 0;
   int sbuff_size = send_count[numProcs-1];
   int rbuff_size = recv_count[numProcs-1];
   for (int k = 0; k < numProcs - 1; k++)
   {
      send_displ[k + 1] = send_displ[k] + send_count[k];
      recv_displ[k + 1] = recv_displ[k] + recv_count[k];
      sbuff_size += send_count[k];
      rbuff_size += recv_count[k];
   }

   int * sendbuf_index = new int[sbuff_size];
   double * sendbuf_values = new double[sbuff_size];
   int * soffs = new int[numProcs]();

   for (int i = 0; i < size; i++)
   {
      int j = row_map[i] - 1;
      int row_rank = GetRowRank(j, row_starts);
      if (myid == row_rank)
      {
         int local_index = j - row_start;
         y[local_index] = x[i];
      }
      else
      {
         int k = send_displ[row_rank] + soffs[row_rank];
         sendbuf_index[k] = j;
         sendbuf_values[k] = x[i];
         soffs[row_rank]++;
      }
   }

   int * recvbuf_index = new int[rbuff_size];
   double * recvbuf_values = new double[rbuff_size];
   MPI_Alltoallv(sendbuf_index,
                 send_count,
                 send_displ,
                 MPI_INT,
                 recvbuf_index,
                 recv_count,
                 recv_displ,
                 MPI_INT,
                 comm);
   MPI_Alltoallv(sendbuf_values,
                 send_count,
                 send_displ,
                 MPI_DOUBLE,
                 recvbuf_values,
                 recv_count,
                 recv_displ,
                 MPI_DOUBLE,
                 comm);

   // Unpack recv buffer
   for (int i = 0; i < rbuff_size; i++)
   {
      int local_index = recvbuf_index[i] - row_start;
      y[local_index] = recvbuf_values[i];
   }

   delete [] recvbuf_values;
   delete [] recvbuf_index;
   delete [] soffs;
   delete [] sendbuf_values;
   delete [] sendbuf_index;
   delete [] recv_displ;
   delete [] send_displ;
   delete [] recv_count;
   delete [] send_count;
}
#endif

} // namespace mfem

#endif // MFEM_USE_MPI
#endif // MFEM_USE_MUMPS
