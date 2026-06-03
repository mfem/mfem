// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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
#include "../general/communication.hpp"

#if defined(MFEM_USE_MUMPS) || defined(MFEM_USE_COMPLEX_MUMPS)
#include "mumps.hpp"
#include <unordered_map>
#include <algorithm>

#if MFEM_MUMPS_VERSION >= 530
#ifdef MUMPS_INTSIZE64
#error "Full 64-bit MUMPS is not yet supported"
#endif
#else
#ifdef INTSIZE64
#error "Full 64-bit MUMPS is not yet supported"
#endif
#endif

// Macro s.t. indices match MUMPS documentation
#define MUMPS_ICNTL(I) icntl[(I) -1]
#define MUMPS_CNTL(I) cntl[(I) -1]
#define MUMPS_INFO(I) info[(I) -1]
#define MUMPS_INFOG(I) infog[(I) -1]

#endif // MFEM_USE_MUMPS || MFEM_USE_COMPLEX_MUMPS


namespace mfem
{

#ifdef MFEM_USE_MUMPS

MUMPSSolver::MUMPSSolver(MPI_Comm comm_)
{
   Init(comm_);
}

MUMPSSolver::MUMPSSolver(const Operator &op)
{
   auto APtr = dynamic_cast<const HypreParMatrix *>(&op);
   MFEM_VERIFY(APtr, "Not a compatible matrix type");
   Init(APtr->GetComm());
   SetOperator(op);
}

void MUMPSSolver::Init(MPI_Comm comm_)
{
   id = nullptr;
   comm = comm_;
   MPI_Comm_size(comm, &numProcs);
   MPI_Comm_rank(comm, &myid);

   mat_type = MatType::UNSYMMETRIC;
   print_level = 0;
   reorder_method = ReorderingStrategy::AUTOMATIC;
   reorder_reuse = false;
   blr_tol = 0.0;

#if MFEM_MUMPS_VERSION >= 530
   irhs_loc = nullptr;
   rhs_loc = nullptr;
   isol_loc = nullptr;
   sol_loc = nullptr;
#else
   recv_counts = nullptr;
   displs = nullptr;
   rhs_glob = nullptr;
#endif
}

MUMPSSolver::~MUMPSSolver()
{
#if MFEM_MUMPS_VERSION >= 530
   delete [] irhs_loc;
   delete [] rhs_loc;
   delete [] isol_loc;
   delete [] sol_loc;
#else
   delete [] recv_counts;
   delete [] displs;
   delete [] rhs_glob;
#endif
   if (id)
   {
      id->job = -2;
#ifdef MFEM_USE_SINGLE
      smumps_c(id);
#else
      dmumps_c(id);
#endif
      delete id;
   }
}

void MUMPSSolver::SetOperator(const Operator &op)
{
   auto APtr = dynamic_cast<const HypreParMatrix *>(&op);
   MFEM_VERIFY(APtr, "Not a compatible matrix type");

   height = op.Height();
   width = op.Width();

   auto parcsr_op = (hypre_ParCSRMatrix *)const_cast<HypreParMatrix &>(*APtr);
   APtr->HostRead();
   hypre_CSRMatrix *csr_op = hypre_MergeDiagAndOffd(parcsr_op);
   APtr->HypreRead();
   HYPRE_Int       *Iptr   = csr_op->i;
#if MFEM_HYPRE_VERSION >= 21600
   HYPRE_BigInt    *Jptr   = csr_op->big_j;
#else
   HYPRE_Int       *Jptr   = csr_op->j;
#endif

   int n_loc = internal::to_int(csr_op->num_rows);
   row_start = internal::to_int(parcsr_op->first_row_index);

   MUMPS_INT8 nnz = 0, k = 0;
   if (mat_type)
   {
      // Count nnz in case of symmetric mode
      for (int i = 0; i < n_loc; i++)
      {
         for (HYPRE_Int j = Iptr[i]; j < Iptr[i + 1]; j++)
         {
            int ii = row_start + i + 1;
#if MFEM_HYPRE_VERSION >= 21600
            HYPRE_BigInt jj = Jptr[k] + 1;
#else
            HYPRE_Int jj = Jptr[k] + 1;
#endif
            if (ii >= jj) { nnz++; }
            k++;
         }
      }
   }
   else
   {
      nnz = csr_op->num_nonzeros;
   }
   int *I = new int[nnz];
   int *J = new int[nnz];

   // Fill in I and J arrays for
   // COO format in 1-based indexing
   k = 0;
   real_t *data;
   if (mat_type)
   {
      MUMPS_INT8 l = 0;
      data = new real_t[nnz];
      for (int i = 0; i < n_loc; i++)
      {
         for (HYPRE_Int j = Iptr[i]; j < Iptr[i + 1]; j++)
         {
            int ii = row_start + i + 1;
#if MFEM_HYPRE_VERSION >= 21600
            HYPRE_BigInt jj = Jptr[k] + 1;
#else
            HYPRE_Int jj = Jptr[k] + 1;
#endif
            if (ii >= jj)
            {
               I[l] = ii;
               J[l] = internal::to_int(jj);
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
         for (HYPRE_Int j = Iptr[i]; j < Iptr[i + 1]; j++)
         {
            I[k] = row_start + i + 1;
            J[k] = internal::to_int(Jptr[k] + 1);
            k++;
         }
      }
      data = csr_op->data;
   }

   // New MUMPS object or reuse the one from a previous matrix
   if (!id || !reorder_reuse)
   {
      if (id)
      {
         id->job = -2;
#ifdef MFEM_USE_SINGLE
         smumps_c(id);
#else
         dmumps_c(id);
#endif
         delete id;
      }
#ifdef MFEM_USE_SINGLE
      id = new SMUMPS_STRUC_C();
#else
      id = new DMUMPS_STRUC_C();
#endif
      id->sym = mat_type;

      // C to Fortran communicator
      id->comm_fortran = (MUMPS_INT)MPI_Comm_c2f(comm);

      // Host is involved in computation
      id->par = 1;

      // MUMPS init
      id->job = -1;
#ifdef MFEM_USE_SINGLE
      smumps_c(id);
#else
      dmumps_c(id);
#endif

      // Set MUMPS default parameters
      SetParameters();

      id->n = internal::to_int(parcsr_op->global_num_rows);
      id->nnz_loc = nnz;
      id->irn_loc = I;
      id->jcn_loc = J;
      id->a_loc = data;

      // MUMPS analysis
      id->job = 1;
#ifdef MFEM_USE_SINGLE
      smumps_c(id);
#else
      dmumps_c(id);
#endif
   }
   else
   {
      id->irn_loc = I;
      id->jcn_loc = J;
      id->a_loc = data;
   }

   // MUMPS factorization
   id->job = 2;
   {
      const int mem_relax_lim = 200;
      while (true)
      {
#ifdef MFEM_USE_SINGLE
         smumps_c(id);
#else
         dmumps_c(id);
#endif
         if (id->MUMPS_INFOG(1) < 0)
         {
            if (id->MUMPS_INFOG(1) == -8 || id->MUMPS_INFOG(1) == -9)
            {
               id->MUMPS_ICNTL(14) += 20;
               MFEM_VERIFY(id->MUMPS_ICNTL(14) <= mem_relax_lim,
                           "Memory relaxation limit reached for MUMPS factorization");
               if (myid == 0 && print_level > 0)
               {
                  mfem::out << "Re-running MUMPS factorization with memory relaxation "
                            << id->MUMPS_ICNTL(14) << '\n';
               }
            }
            else
            {
               MFEM_ABORT("Error during MUMPS numerical factorization");
            }
         }
         else { break; }
      }
   }

   hypre_CSRMatrixDestroy(csr_op);
   delete [] I;
   delete [] J;
   if (mat_type) { delete [] data; }

   id->nrhs = -1;  // Set up solution storage on first call to Mult
#if MFEM_MUMPS_VERSION >= 530
   delete [] irhs_loc;
   delete [] isol_loc;
   id->nloc_rhs = n_loc;
   id->lrhs_loc = n_loc;
   id->lsol_loc = id->MUMPS_INFO(23);
   irhs_loc = new int[id->lrhs_loc];
   isol_loc = new int[id->lsol_loc];
   for (int i = 0; i < n_loc; i++)
   {
      irhs_loc[i] = row_start + i + 1;
   }
   id->irhs_loc = irhs_loc;
   id->isol_loc = isol_loc;

   row_starts.SetSize(numProcs);
   MPI_Allgather(&row_start, 1, MPI_INT, row_starts, 1, MPI_INT, comm);
#else
   id->lrhs = id->n;
   if (myid == 0)
   {
      delete [] recv_counts;
      delete [] displs;
      recv_counts = new int[numProcs];
      displs = new int[numProcs];
   }
   MPI_Gather(&n_loc, 1, MPI_INT, recv_counts, 1, MPI_INT, 0, comm);
   if (myid == 0)
   {
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

void MUMPSSolver::InitRhsSol(int nrhs) const
{
   if (id->nrhs != nrhs)
   {
#if MFEM_MUMPS_VERSION >= 530
      delete [] rhs_loc;
      delete [] sol_loc;
      rhs_loc = (nrhs > 1) ? new real_t[nrhs * id->lrhs_loc] : nullptr;
      sol_loc = new real_t[nrhs * id->lsol_loc];
      id->rhs_loc = rhs_loc;
      id->sol_loc = sol_loc;
#else
      if (myid == 0)
      {
         delete [] rhs_glob;
         rhs_glob = new real_t[nrhs * id->lrhs];
         id->rhs = rhs_glob;
      }
#endif
   }
   id->nrhs = nrhs;
}

void MUMPSSolver::Mult(const Vector &x, Vector &y) const
{
   Array<const Vector *> X(1);
   Array<Vector *> Y(1);
   X[0] = &x;
   Y[0] = &y;
   ArrayMult(X, Y);
}

void MUMPSSolver::ArrayMult(const Array<const Vector *> &X,
                            Array<Vector *> &Y) const
{
   MFEM_ASSERT(X.Size() == Y.Size(),
               "Number of columns mismatch in MUMPSSolver::Mult!");
   InitRhsSol(X.Size());
#if MFEM_MUMPS_VERSION >= 530
   if (id->nrhs == 1)
   {
      MFEM_ASSERT(X.Size() == 1 && X[0], "Missing Vector in MUMPSSolver::Mult!");
      X[0]->HostRead();
      id->rhs_loc = X[0]->GetData();
   }
   else
   {
      for (int i = 0; i < id->nrhs; i++)
      {
         MFEM_ASSERT(X[i], "Missing Vector in MUMPSSolver::Mult!");
         X[i]->HostRead();
         std::copy(X[i]->GetData(), X[i]->GetData() + X[i]->Size(),
                   id->rhs_loc + i * id->lrhs_loc);
      }
   }

   // MUMPS solve
   id->job = 3;
#ifdef MFEM_USE_SINGLE
   smumps_c(id);
#else
   dmumps_c(id);
#endif

   RedistributeSol(id->isol_loc, id->sol_loc, id->lsol_loc, Y);
#else
   for (int i = 0; i < id->nrhs; i++)
   {
      MFEM_ASSERT(X[i], "Missing Vector in MUMPSSolver::Mult!");
      X[i]->HostRead();
      MPI_Gatherv(X[i]->GetData(), X[i]->Size(), MPITypeMap<real_t>::mpi_type,
                  id->rhs + i * id->lrhs, recv_counts, displs, MPITypeMap<real_t>::mpi_type, 0,
                  comm);
   }

   // MUMPS solve
   id->job = 3;
#ifdef MFEM_USE_SINGLE
   smumps_c(id);
#else
   dmumps_c(id);
#endif

   for (int i = 0; i < id->nrhs; i++)
   {
      MFEM_ASSERT(Y[i], "Missing Vector in MUMPSSolver::Mult!");
      Y[i]->HostWrite();
      MPI_Scatterv(id->rhs + i * id->lrhs, recv_counts, displs,
                   MPITypeMap<real_t>::mpi_type,
                   Y[i]->GetData(), Y[i]->Size(), MPITypeMap<real_t>::mpi_type, 0, comm);
   }
#endif
}

void MUMPSSolver::MultTranspose(const Vector &x, Vector &y) const
{
   // Set flag for transpose solve
   id->MUMPS_ICNTL(9) = 0;
   Mult(x, y);

   // Reset the flag
   id->MUMPS_ICNTL(9) = 1;
}

void MUMPSSolver::ArrayMultTranspose(const Array<const Vector *> &X,
                                     Array<Vector *> &Y) const
{
   // Set flag for transpose solve
   id->MUMPS_ICNTL(9) = 0;
   ArrayMult(X, Y);

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

void MUMPSSolver::SetReorderingStrategy(ReorderingStrategy method)
{
   reorder_method = method;
}

void MUMPSSolver::SetReorderingReuse(bool reuse)
{
   reorder_reuse = reuse;
}

#if MFEM_MUMPS_VERSION >= 510
void MUMPSSolver::SetBLRTol(double tol)
{
   blr_tol = tol;
}
#endif

void MUMPSSolver::SetParameters()
{
   // Output stream for error messages
   id->MUMPS_ICNTL(1) = 6;
   // Output stream for diagnosting printing local to each proc
   id->MUMPS_ICNTL(2) = 0;
   // Output stream for global info
   id->MUMPS_ICNTL(3) = 6;
   // Level of error printing
   id->MUMPS_ICNTL(4) = print_level;
   // Input matrix format (assembled)
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
   // Configure reordering
   switch (reorder_method)
   {
      case ReorderingStrategy::AUTOMATIC:
         id->MUMPS_ICNTL(28) = 0;
         id->MUMPS_ICNTL(7) = 7;
         id->MUMPS_ICNTL(29) = 0;
         break;
      case ReorderingStrategy::AMD:
         id->MUMPS_ICNTL(28) = 1;
         id->MUMPS_ICNTL(7) = 0;
         break;
      case ReorderingStrategy::AMF:
         id->MUMPS_ICNTL(28) = 1;
         id->MUMPS_ICNTL(7) = 2;
         break;
      case ReorderingStrategy::PORD:
         id->MUMPS_ICNTL(28) = 1;
         id->MUMPS_ICNTL(7) = 4;
         break;
      case ReorderingStrategy::METIS:
         id->MUMPS_ICNTL(28) = 1;
         id->MUMPS_ICNTL(7) = 5;
         break;
      case ReorderingStrategy::PARMETIS:
         id->MUMPS_ICNTL(28) = 2;
         id->MUMPS_ICNTL(29) = 2;
         break;
      case ReorderingStrategy::SCOTCH:
         id->MUMPS_ICNTL(28) = 1;
         id->MUMPS_ICNTL(7) = 3;
         break;
      case ReorderingStrategy::PTSCOTCH:
         id->MUMPS_ICNTL(28) = 2;
         id->MUMPS_ICNTL(29) = 1;
         break;
      default:
         break; // This should be unreachable
   }
   // Option to activate BLR factorization
#if MFEM_MUMPS_VERSION >= 510
   if (blr_tol > 0.0)
   {
      id->MUMPS_ICNTL(35) = 1;
      id->MUMPS_CNTL(7) = blr_tol;
   }
#endif
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

void MUMPSSolver::RedistributeSol(const int *rmap, const real_t *x,
                                  const int lx_loc, Array<Vector *> &Y) const
{
   int *send_count = new int[numProcs]();
   for (int i = 0; i < lx_loc; i++)
   {
      int j = rmap[i] - 1;
      int row_rank = GetRowRank(j, row_starts);
      if (myid == row_rank) { continue; }
      send_count[row_rank]++;
   }

   int *recv_count = new int[numProcs];
   MPI_Alltoall(send_count, 1, MPI_INT, recv_count, 1, MPI_INT, comm);

   int *send_displ = new int[numProcs]; send_displ[0] = 0;
   int *recv_displ = new int[numProcs]; recv_displ[0] = 0;
   int sbuff_size = send_count[numProcs-1];
   int rbuff_size = recv_count[numProcs-1];
   for (int k = 0; k < numProcs - 1; k++)
   {
      send_displ[k + 1] = send_displ[k] + send_count[k];
      recv_displ[k + 1] = recv_displ[k] + recv_count[k];
      sbuff_size += send_count[k];
      rbuff_size += recv_count[k];
   }

   int *sendbuf_index = new int[sbuff_size];
   real_t *sendbuf_values = new real_t[sbuff_size];
   int *recvbuf_index = new int[rbuff_size];
   real_t *recvbuf_values = new real_t[rbuff_size];
   int *soffs = new int[numProcs]();

   for (int i = 0; i < lx_loc; i++)
   {
      int j = rmap[i] - 1;
      int row_rank = GetRowRank(j, row_starts);
      if (myid != row_rank)
      {
         int k = send_displ[row_rank] + soffs[row_rank];
         sendbuf_index[k] = j;
         soffs[row_rank]++;
      }
   }

   MPI_Alltoallv(sendbuf_index, send_count, send_displ, MPI_INT,
                 recvbuf_index, recv_count, recv_displ, MPI_INT, comm);

   for (int rhs = 0; rhs < Y.Size(); rhs++)
   {
      MFEM_ASSERT(Y[rhs], "Missing Vector in MUMPSSolver::Mult!");
      Y[rhs]->HostWrite();

      std::fill(soffs, soffs + numProcs, 0);
      for (int i = 0; i < lx_loc; i++)
      {
         int j = rmap[i] - 1;
         int row_rank = GetRowRank(j, row_starts);
         if (myid == row_rank)
         {
            int local_index = j - row_start;
            (*Y[rhs])(local_index) = x[rhs * lx_loc + i];
         }
         else
         {
            int k = send_displ[row_rank] + soffs[row_rank];
            sendbuf_values[k] = x[rhs * lx_loc + i];
            soffs[row_rank]++;
         }
      }

      MPI_Alltoallv(sendbuf_values, send_count, send_displ,
                    MPITypeMap<real_t>::mpi_type,
                    recvbuf_values, recv_count, recv_displ, MPITypeMap<real_t>::mpi_type, comm);

      // Unpack recv buffer
      for (int i = 0; i < rbuff_size; i++)
      {
         int local_index = recvbuf_index[i] - row_start;
         (*Y[rhs])(local_index) = recvbuf_values[i];
      }
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

#endif // MFEM_USE_MUMPS


#ifdef MFEM_USE_COMPLEX_MUMPS

ComplexMUMPSSolver::ComplexMUMPSSolver(MPI_Comm comm_)
{
   Init(comm_);
}

ComplexMUMPSSolver::ComplexMUMPSSolver(const Operator &op)
{
   auto APtr = dynamic_cast<const ComplexHypreParMatrix *>(&op);
   MFEM_VERIFY(APtr,
               "ComplexMUMPSSolver requires a ComplexHypreParMatrix operator");
   SetOperator(op);
}

void ComplexMUMPSSolver::Init(MPI_Comm comm_)
{
   comm = comm_;
   MPI_Comm_size(comm, &numProcs);
   MPI_Comm_rank(comm, &myid);

   print_level = 2;
   row_start = 0;

   id = nullptr;

#if MFEM_MUMPS_VERSION >= 530
   irhs_loc = nullptr;
   isol_loc = nullptr;
   rhs_loc  = nullptr;
   sol_loc  = nullptr;
#else
   global_num_rows = 0;
   recv_counts = nullptr;
   displs = nullptr;
   rhs_glob = nullptr;
   rhs_glob_r = nullptr;
   rhs_glob_i = nullptr;
#endif
}

ComplexMUMPSSolver::~ComplexMUMPSSolver()
{
#if MFEM_MUMPS_VERSION >= 530
   delete [] irhs_loc;
   delete [] isol_loc;
   delete [] rhs_loc;
   delete [] sol_loc;
#else
   delete [] recv_counts;
   delete [] displs;
   delete [] rhs_glob;
   delete [] rhs_glob_r;
   delete [] rhs_glob_i;
#endif

   if (id)
   {
      id->job = -2;
      mumps_call();
      delete id;
      id = nullptr;
   }
}

void ComplexMUMPSSolver::SetOperator(const Operator &op)
{
   auto APtr = dynamic_cast<const ComplexHypreParMatrix *>(&op);
   MFEM_VERIFY(APtr,
               "ComplexMUMPSSolver requires a ComplexHypreParMatrix operator");

   height = op.Height();
   width  = op.Width();

   const HypreParMatrix *Ar = (APtr->hasRealPart()) ? &APtr->real() : nullptr;
   const HypreParMatrix *Ai = (APtr->hasImagPart()) ? &APtr->imag() : nullptr;

   MFEM_VERIFY(Ar || Ai, "ComplexMUMPSSolver: both real and imag parts are null.");

   // Pick communicator from the non-null part
   MPI_Comm op_comm = (Ar ? Ar->GetComm() : Ai->GetComm());

   // Comm setup/check
   if (comm == MPI_COMM_NULL) { Init(op_comm); }
   else
   {
      int cmp = MPI_UNEQUAL;
      MPI_Comm_compare(comm, op_comm, &cmp);
      MFEM_VERIFY(cmp != MPI_UNEQUAL, "MPI Comm mismatch");
   }

   // HostRead only if non-null
   if (Ar) { Ar->HostRead(); }
   if (Ai) { Ai->HostRead(); }

   // hypre parcsr pointers
   hypre_ParCSRMatrix *parcsr_op_r = nullptr;
   hypre_ParCSRMatrix *parcsr_op_i = nullptr;

   if (Ar) { parcsr_op_r = (hypre_ParCSRMatrix*) const_cast<HypreParMatrix&>(*Ar); }
   if (Ai) { parcsr_op_i = (hypre_ParCSRMatrix*) const_cast<HypreParMatrix&>(*Ai); }

   // Merge diag+offd for whichever exists
   hypre_CSRMatrix *csr_op_r = nullptr;
   hypre_CSRMatrix *csr_op_i = nullptr;

   if (parcsr_op_r) { csr_op_r = hypre_MergeDiagAndOffd(parcsr_op_r); }
   if (parcsr_op_i) { csr_op_i = hypre_MergeDiagAndOffd(parcsr_op_i); }

#if MFEM_HYPRE_VERSION >= 21600
   if (csr_op_r) { hypre_CSRMatrixBigJtoJ(csr_op_r); }
   if (csr_op_i) { hypre_CSRMatrixBigJtoJ(csr_op_i); }
#endif

   // Determine local/global sizes and row_start from an existing part
   const int n_loc = internal::to_int((csr_op_r ? csr_op_r->num_rows :
                                       csr_op_i->num_rows));
   row_start = internal::to_int((parcsr_op_r ? parcsr_op_r->first_row_index
                                 : parcsr_op_i->first_row_index));
   const int global_n = internal::to_int((parcsr_op_r ?
                                          parcsr_op_r->global_num_rows
                                          : parcsr_op_i->global_num_rows));

   // Use nullptr checks
   const int *Ir = csr_op_r ? csr_op_r->i : nullptr;
   const int *Jr = csr_op_r ? csr_op_r->j : nullptr;
   const real_t *Vr = csr_op_r ? (const real_t*)csr_op_r->data : nullptr;

   const int *Ii = csr_op_i ? csr_op_i->i : nullptr;
   const int *Ji = csr_op_i ? csr_op_i->j : nullptr;
   const real_t *Vi = csr_op_i ? (const real_t*)csr_op_i->data : nullptr;

   // Build union COO
   std::vector<int> Icoo, Jcoo;
   std::vector<mumps_complex_t> Zcoo;

   size_t nnz_r = csr_op_r ? (size_t)csr_op_r->num_nonzeros : 0;
   size_t nnz_i = csr_op_i ? (size_t)csr_op_i->num_nonzeros : 0;
   Icoo.reserve(nnz_r + nnz_i);
   Jcoo.reserve(nnz_r + nnz_i);
   Zcoo.reserve(nnz_r + nnz_i);

   BuildUnionCOO(n_loc, row_start, Ir, Jr, Vr, Ii, Ji, Vi, Icoo, Jcoo, Zcoo);

   const int nnz = (int)Icoo.size();
   int *I = new int[nnz];
   int *J = new int[nnz];
   mumps_complex_t *A = new mumps_complex_t[nnz];

   std::copy(Icoo.begin(), Icoo.end(), I);
   std::copy(Jcoo.begin(), Jcoo.end(), J);
   std::copy(Zcoo.begin(), Zcoo.end(), A);

   // New ComplexMUMPS object or reuse an existing one
   if (!id || !reorder_reuse)
   {
      if (id)
      {
         id->job = -2;
         mumps_call();
         delete id;
         id = nullptr;
      }

#ifdef MFEM_USE_SINGLE
      id = new CMUMPS_STRUC_C();
#else
      id = new ZMUMPS_STRUC_C();
#endif

      id->sym = 0; // general complex
      id->par = 1;
      id->comm_fortran = (MUMPS_INT)MPI_Comm_c2f(comm);

      // Init
      id->job = -1;
      mumps_call();

      // Set parameters
      SetParameters();

      // Attach matrix
      id->n       = global_n;
      id->nnz_loc = nnz;
      id->irn_loc = I;
      id->jcn_loc = J;
      id->a_loc   = A;

      // Analysis (ordering + symbolic)
      id->job = 1;
      mumps_call();
   }
   else
   {
      // Reuse symbolic factorization / ordering
      MFEM_VERIFY(id->n == global_n,
                  "ReorderingReuse requires same global size (id->n mismatch)");

      // Update matrix pointers (pattern is assumed compatible)
      id->nnz_loc = nnz;
      id->irn_loc = I;
      id->jcn_loc = J;
      id->a_loc   = A;
   }

   // Factorization
   id->job = 2;
   {
      const int mem_relax_lim = 200;
      while (true)
      {
         mumps_call();
         if (id->MUMPS_INFOG(1) < 0)
         {
            if (id->MUMPS_INFOG(1) == -8 || id->MUMPS_INFOG(1) == -9)
            {
               id->MUMPS_ICNTL(14) += 20;
               MFEM_VERIFY(id->MUMPS_ICNTL(14) <= mem_relax_lim,
                           "Memory relaxation limit reached for ComplexMUMPSSolver factorization");
               if (myid == 0 && print_level > 0)
               {
                  out << "Re-running ComplexMUMPSSolver factorization with memory relaxation "
                      << id->MUMPS_ICNTL(14) << '\n';
               }
            }
            else
            {
               MFEM_ABORT("Error during ComplexMUMPSSolver numerical factorization");
            }
         }
         else { break; }
      }
   }

   // Done with input storage
   if (csr_op_r) { hypre_CSRMatrixDestroy(csr_op_r);}
   if (csr_op_i) { hypre_CSRMatrixDestroy(csr_op_i);}
   delete [] I;
   delete [] J;
   delete [] A;

   // Post-factorization RHS/SOL setup
   id->nrhs = -1;

#if MFEM_MUMPS_VERSION >= 530
   // Distributed RHS/SOL sizes
   id->nloc_rhs = n_loc;
   id->lrhs_loc = n_loc;
   id->lsol_loc = id->MUMPS_INFO(23);

   delete [] irhs_loc;
   irhs_loc = new int[id->lrhs_loc];
   for (int i = 0; i < n_loc; i++)
   {
      irhs_loc[i] = row_start + i + 1;
   }
   id->irhs_loc = irhs_loc;

   delete [] isol_loc;
   isol_loc = new int[id->lsol_loc];
   id->isol_loc = isol_loc;

   row_starts.SetSize(numProcs);
   MPI_Allgather(&row_start, 1, MPI_INT, row_starts, 1, MPI_INT, comm);

   // Reset cached buffers
   delete [] rhs_loc; rhs_loc = nullptr;
   delete [] sol_loc; sol_loc = nullptr;
   rhs1_buf.clear();

#else
   // Centralized RHS/SOL on root
   id->lrhs = id->n;

   global_num_rows = id->n;

   if (myid == 0)
   {
      delete [] recv_counts;
      delete [] displs;
      recv_counts = new int[numProcs];
      displs = new int[numProcs];

      delete [] rhs_glob;   rhs_glob = nullptr;
      delete [] rhs_glob_r; rhs_glob_r = nullptr;
      delete [] rhs_glob_i; rhs_glob_i = nullptr;
   }

   MPI_Gather(&n_loc, 1, MPI_INT, recv_counts, 1, MPI_INT, 0, comm);

   if (myid == 0)
   {
      displs[0] = 0;
      int s = 0;
      for (int k = 0; k < numProcs - 1; k++)
      {
         s += recv_counts[k];
         displs[k+1] = s;
      }
   }
#endif
}

void ComplexMUMPSSolver::InitRhsSol(int nrhs) const
{
#if MFEM_MUMPS_VERSION >= 530

   MFEM_VERIFY(id, "InitRhsSol called before SetOperator");

   if (id->nrhs != nrhs)
   {
      delete [] rhs_loc;
      delete [] sol_loc;

      rhs_loc = new mumps_complex_t[(size_t)nrhs * (size_t)id->lrhs_loc];
      sol_loc = new mumps_complex_t[(size_t)nrhs * (size_t)id->lsol_loc];

      id->rhs_loc = rhs_loc;
      id->sol_loc = sol_loc;
   }
   id->nrhs = nrhs;

#else
   MFEM_VERIFY(id, "InitRhsSol called before SetOperator");

   id->nrhs = nrhs;
   id->lrhs = id->n;

   if (myid == 0)
   {
      const size_t N = (size_t)nrhs * (size_t)global_num_rows;

      delete [] rhs_glob;
      delete [] rhs_glob_r;
      delete [] rhs_glob_i;

      rhs_glob   = new mumps_complex_t[N];
      rhs_glob_r = new real_t[N];
      rhs_glob_i = new real_t[N];

      id->rhs = rhs_glob;
   }
#endif
}

void ComplexMUMPSSolver::Mult(const Vector &x, Vector &y) const
{
   Array<const Vector *> X(1);
   Array<Vector *> Y(1);
   X[0] = &x;
   Y[0] = &y;
   ArrayMult(X, Y);
}

void ComplexMUMPSSolver::ArrayMult(const Array<const Vector *> &X,
                                   Array<Vector *> &Y) const
{
   MFEM_ASSERT(X.Size() == Y.Size(),
               "Number of columns mismatch in ComplexMUMPSSolver::Mult!");
   MFEM_VERIFY(id, "ComplexMUMPSSolver::ArrayMult called before SetOperator");

   InitRhsSol(X.Size());

#if MFEM_MUMPS_VERSION >= 530
   MFEM_VERIFY(irhs_loc && isol_loc, "RHS/SOL maps not initialized");
   MFEM_VERIFY(rhs_loc && sol_loc, "RHS/SOL buffers not initialized");
   const int n_loc = id->lrhs_loc;
   const int nrhs  = id->nrhs;

   // Pack all RHS
   int xisign = (conv == ComplexOperator::BLOCK_SYMMETRIC) ? -1 : 1;
   for (int i = 0; i < nrhs; i++)
   {
      MFEM_ASSERT(X[i], "Missing Vector in Mult!");
      X[i]->HostRead();
      MFEM_VERIFY(X[i]->Size() == 2*n_loc, "RHS size mismatch");

      const real_t *xdata = X[i]->GetData();
      const real_t *xr = xdata;
      const real_t *xi = xdata + n_loc;

      mumps_complex_t *dst = rhs_loc + i * n_loc;
      for (int j = 0; j < n_loc; j++)
      {
         dst[j].r = xr[j];
         dst[j].i = xisign * xi[j];
      }
   }

   id->rhs_loc  = rhs_loc;
   id->sol_loc  = sol_loc;
   id->irhs_loc = irhs_loc;
   id->isol_loc = isol_loc;

   // MUMPS solve
   id->job = 3;
   mumps_call();

   const int lsol = id->lsol_loc;

   // Redistribute each solution column into Y
   for (int i = 0; i < nrhs; i++)
   {
      MFEM_ASSERT(Y[i], "Missing output Vector in Mult!");
      Y[i]->HostWrite();
      MFEM_VERIFY(Y[i]->Size() == 2*n_loc, "Output size mismatch");

      const mumps_complex_t *xcol = sol_loc + i * lsol;
      RedistributeSol(isol_loc, xcol, Y[i]->GetData(), n_loc, lsol);
   }

#else // MFEM_MUMPS_VERSION < 530

   const int nrhs = id->nrhs;

   MFEM_VERIFY(X.Size() > 0 && X[0], "Missing RHS");
   const int n_loc = X[0]->Size()/2;

   for (int i = 0; i < nrhs; i++)
   {
      MFEM_ASSERT(X[i], "Missing Vector in Mult!");
      X[i]->HostRead();
      MFEM_VERIFY(X[i]->Size() == 2*n_loc, "RHS size mismatch");
   }

   // Gather each RHS column (real+imag separately) into root staging
   for (int i = 0; i < nrhs; i++)
   {
      const real_t *xdata = X[i]->GetData();

      MPI_Gatherv(xdata, n_loc, MPITypeMap<real_t>::mpi_type,
                  rhs_glob_r + i * global_num_rows,
                  recv_counts, displs, MPITypeMap<real_t>::mpi_type,
                  0, comm);

      MPI_Gatherv(xdata + n_loc, n_loc, MPITypeMap<real_t>::mpi_type,
                  rhs_glob_i + i * global_num_rows,
                  recv_counts, displs, MPITypeMap<real_t>::mpi_type,
                  0, comm);
   }

   // Pack into MUMPS complex RHS on root: id->rhs is in-place
   if (myid == 0)
   {
      for (int i = 0; i < nrhs; i++)
      {
         mumps_complex_t *dst = rhs_glob + i * global_num_rows;
         const real_t *rr = rhs_glob_r + i * global_num_rows;
         const real_t *ri = rhs_glob_i + i * global_num_rows;

         for (int j = 0; j < global_num_rows; j++)
         {
            dst[j].r = rr[j];
            dst[j].i = ri[j];
         }
      }
      id->rhs = rhs_glob;
   }

   // Solve
   id->job = 3;
   mumps_call();

   // Unpack to real/imag
   if (myid == 0)
   {
      for (int i = 0; i < nrhs; i++)
      {
         const mumps_complex_t *src = rhs_glob + i * global_num_rows;
         real_t *rr = rhs_glob_r + i * global_num_rows;
         real_t *ri = rhs_glob_i + i * global_num_rows;

         for (int j = 0; j < global_num_rows; j++)
         {
            rr[j] = src[j].r;
            ri[j] = src[j].i;
         }
      }
   }

   // Scatter each RHS solution
   for (int i = 0; i < nrhs; i++)
   {
      MFEM_ASSERT(Y[i], "Missing Vector in Mult!");
      Y[i]->HostWrite();
      MFEM_VERIFY(Y[i]->Size() == 2*n_loc, "Output size mismatch");

      real_t *ydata = Y[i]->GetData();

      MPI_Scatterv(rhs_glob_r + i * global_num_rows,
                   recv_counts, displs, MPITypeMap<real_t>::mpi_type,
                   ydata, n_loc, MPITypeMap<real_t>::mpi_type,
                   0, comm);

      MPI_Scatterv(rhs_glob_i + i * global_num_rows,
                   recv_counts, displs, MPITypeMap<real_t>::mpi_type,
                   ydata + n_loc, n_loc, MPITypeMap<real_t>::mpi_type,
                   0, comm);
   }

#endif
}

void ComplexMUMPSSolver::MultTranspose(const Vector &x, Vector &y) const
{
   MFEM_VERIFY(id, "MultTranspose called before SetOperator");

   // Transpose solve
   id->MUMPS_ICNTL(9) = 0;
   Mult(x, y);
   id->MUMPS_ICNTL(9) = 1;
}

void ComplexMUMPSSolver::ArrayMultTranspose(const Array<const Vector *> &X,
                                            Array<Vector *> &Y) const
{
   MFEM_VERIFY(id, "ArrayMultTranspose called before SetOperator");

   // Transpose solve
   id->MUMPS_ICNTL(9) = 0;
   ArrayMult(X, Y);
   id->MUMPS_ICNTL(9) = 1;
}

void ComplexMUMPSSolver::SetParameters()
{
   // Output stream for error messages
   id->MUMPS_ICNTL(1) = 6;
   // Output stream for diagnostic printing local to each proc
   id->MUMPS_ICNTL(2) = 0;
   // Output stream for global info
   id->MUMPS_ICNTL(3) = 6;
   // Level of error printing
   id->MUMPS_ICNTL(4) = print_level;

   // Input matrix format (assembled)
   id->MUMPS_ICNTL(5) = 0;
   // Use A or A^T
   id->MUMPS_ICNTL(9) = 1;
   // Iterative refinement (disabled)
   id->MUMPS_ICNTL(10) = 0;
   // Error analysis-statistics (disabled)
   id->MUMPS_ICNTL(11) = 0;
   // Use of ScaLAPACK (disabled)
   id->MUMPS_ICNTL(13) = 0;
   // Workspace relaxation (% increase)
   id->MUMPS_ICNTL(14) = 20;
   // OpenMP threads (default)
   id->MUMPS_ICNTL(16) = 0;
   // Matrix input format (distributed)
   id->MUMPS_ICNTL(18) = 3;
   // Schur complement (none)
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

   // Out-of-core (disabled)
   id->MUMPS_ICNTL(22) = 0;
   // Max size of working memory (default)
   id->MUMPS_ICNTL(23) = 0;

   switch (reorder_method)
   {
      case ReorderingStrategy::AUTOMATIC:
         id->MUMPS_ICNTL(28) = 0;
         id->MUMPS_ICNTL(7) = 7;
         id->MUMPS_ICNTL(29) = 0;
         break;
      case ReorderingStrategy::AMD:
         id->MUMPS_ICNTL(28) = 1;
         id->MUMPS_ICNTL(7) = 0;
         break;
      case ReorderingStrategy::AMF:
         id->MUMPS_ICNTL(28) = 1;
         id->MUMPS_ICNTL(7) = 2;
         break;
      case ReorderingStrategy::PORD:
         id->MUMPS_ICNTL(28) = 1;
         id->MUMPS_ICNTL(7) = 4;
         break;
      case ReorderingStrategy::METIS:
         id->MUMPS_ICNTL(28) = 1;
         id->MUMPS_ICNTL(7) = 5;
         break;
      case ReorderingStrategy::PARMETIS:
         id->MUMPS_ICNTL(28) = 2;
         id->MUMPS_ICNTL(29) = 2;
         break;
      case ReorderingStrategy::SCOTCH:
         id->MUMPS_ICNTL(28) = 1;
         id->MUMPS_ICNTL(7) = 3;
         break;
      case ReorderingStrategy::PTSCOTCH:
         id->MUMPS_ICNTL(28) = 2;
         id->MUMPS_ICNTL(29) = 1;
         break;
      default:
         break; // This should be unreachable
   }

}

void ComplexMUMPSSolver::BuildUnionCOO(const int n_loc,
                                       const int row_start_,
                                       const int *Ir, const int *Jr, const real_t *Vr,
                                       const int *Ii, const int *Ji, const real_t *Vi,
                                       std::vector<int> &Icoo,
                                       std::vector<int> &Jcoo,
                                       std::vector<mumps_complex_t> &Zcoo) const
{
   for (int r = 0; r < n_loc; ++r)
   {
      std::unordered_map<int, std::pair<real_t, real_t>> row;

      const int rr0 = Ir ? Ir[r]   : 0;
      const int rr1 = Ir ? Ir[r+1] : 0;
      const int ii0 = Ii ? Ii[r]   : 0;
      const int ii1 = Ii ? Ii[r+1] : 0;

      row.reserve((rr1 - rr0) + (ii1 - ii0));

      if (Ir)
      {
         for (int p = rr0; p < rr1; ++p) { row[Jr[p]].first += Vr[p]; }
      }
      if (Ii)
      {
         for (int p = ii0; p < ii1; ++p) { row[Ji[p]].second += Vi[p]; }
      }

      for (const auto &kv : row)
      {
         Icoo.push_back(row_start_ + r + 1);
         Jcoo.push_back(kv.first + 1);
         Zcoo.push_back(mumps_complex_t{kv.second.first, kv.second.second});
      }
   }
}

#if MFEM_MUMPS_VERSION >= 530
int ComplexMUMPSSolver::GetRowRank(int i, const Array<int> &row_starts_) const
{
   if (row_starts_.Size() == 1) { return 0; }
   auto up = std::upper_bound(row_starts_.begin(), row_starts_.end(), i);
   return (int)std::distance(row_starts_.begin(), up) - 1;
}

void ComplexMUMPSSolver::RedistributeSol(const int *row_map,
                                         const mumps_complex_t *x,
                                         real_t *y_ri,
                                         int n_loc,
                                         int lsol_loc) const
{
   int *send_count = new int[numProcs]();
   for (int i = 0; i < lsol_loc; i++)
   {
      const int j = row_map[i] - 1;
      const int row_rank = GetRowRank(j, row_starts);
      if (myid == row_rank) { continue; }
      send_count[row_rank]++;
   }

   int *recv_count = new int[numProcs];
   MPI_Alltoall(send_count, 1, MPI_INT, recv_count, 1, MPI_INT, comm);

   int *send_displ = new int[numProcs]; send_displ[0] = 0;
   int *recv_displ = new int[numProcs]; recv_displ[0] = 0;

   int sbuff_size = send_count[numProcs-1];
   int rbuff_size = recv_count[numProcs-1];
   for (int k = 0; k < numProcs - 1; k++)
   {
      send_displ[k+1] = send_displ[k] + send_count[k];
      recv_displ[k+1] = recv_displ[k] + recv_count[k];
      sbuff_size += send_count[k];
      rbuff_size += recv_count[k];
   }

   int   *sendbuf_index = new int[sbuff_size];
   real_t *sendbuf_r    = new real_t[sbuff_size];
   real_t *sendbuf_i    = new real_t[sbuff_size];
   int   *soffs         = new int[numProcs]();

   for (int i = 0; i < lsol_loc; i++)
   {
      const int j = row_map[i] - 1;
      const int row_rank = GetRowRank(j, row_starts);

      const real_t xr = (real_t)x[i].r;
      const real_t xi = (real_t)x[i].i;

      if (myid == row_rank)
      {
         const int local_index = j - row_start;
         y_ri[local_index]        = xr;
         y_ri[local_index+n_loc]  = xi;
      }
      else
      {
         const int k = send_displ[row_rank] + soffs[row_rank];
         sendbuf_index[k] = j;
         sendbuf_r[k] = xr;
         sendbuf_i[k] = xi;
         soffs[row_rank]++;
      }
   }

   int    *recvbuf_index = new int[rbuff_size];
   real_t *recvbuf_r     = new real_t[rbuff_size];
   real_t *recvbuf_i     = new real_t[rbuff_size];

   MPI_Alltoallv(sendbuf_index, send_count, send_displ, MPI_INT,
                 recvbuf_index, recv_count, recv_displ, MPI_INT, comm);

   MPI_Alltoallv(sendbuf_r, send_count, send_displ, MPITypeMap<real_t>::mpi_type,
                 recvbuf_r, recv_count, recv_displ, MPITypeMap<real_t>::mpi_type, comm);

   MPI_Alltoallv(sendbuf_i, send_count, send_displ, MPITypeMap<real_t>::mpi_type,
                 recvbuf_i, recv_count, recv_displ, MPITypeMap<real_t>::mpi_type, comm);

   for (int i = 0; i < rbuff_size; i++)
   {
      const int local_index = recvbuf_index[i] - row_start;
      y_ri[local_index]       = recvbuf_r[i];
      y_ri[local_index+n_loc] = recvbuf_i[i];
   }

   delete [] recvbuf_i;
   delete [] recvbuf_r;
   delete [] recvbuf_index;
   delete [] soffs;
   delete [] sendbuf_i;
   delete [] sendbuf_r;
   delete [] sendbuf_index;
   delete [] recv_displ;
   delete [] send_displ;
   delete [] recv_count;
   delete [] send_count;
}
#endif // MFEM_MUMPS_VERSION >= 530

#endif // MFEM_USE_COMPLEX_MUMPS

} // namespace mfem
