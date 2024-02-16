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

#include "../config/config.hpp"

#ifdef MFEM_USE_STRUMPACK
#ifdef MFEM_USE_MPI

#include "strumpack.hpp"

namespace mfem
{

STRUMPACKRowLocMatrix::STRUMPACKRowLocMatrix(MPI_Comm comm,
                                             int num_loc_rows,
                                             HYPRE_BigInt first_loc_row,
                                             HYPRE_BigInt glob_nrows,
                                             HYPRE_BigInt glob_ncols,
                                             int *I, HYPRE_BigInt *J,
                                             double *data, bool sym_sparse)
{
   // Set mfem::Operator member data
   height = num_loc_rows;
   width  = num_loc_rows;

   // Allocate STRUMPACK's CSRMatrixMPI (copies all inputs)
   int rank, nprocs;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &nprocs);
   Array<HYPRE_BigInt> dist(nprocs + 1);
   dist[0] = 0;
   dist[rank + 1] = first_loc_row + (HYPRE_BigInt)num_loc_rows;
   MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                 dist.GetData() + 1, 1, HYPRE_MPI_BIG_INT, comm);

#if !(defined(HYPRE_BIGINT) || defined(HYPRE_MIXEDINT))
   A_ = new strumpack::CSRMatrixMPI<double, HYPRE_BigInt>(
      (HYPRE_BigInt)num_loc_rows, I, J, data, dist.GetData(),
      comm, sym_sparse);
#else
   Array<HYPRE_BigInt> II(num_loc_rows+1);
   for (int i = 0; i <= num_loc_rows; i++) { II[i] = (HYPRE_BigInt)I[i]; }
   A_ = new strumpack::CSRMatrixMPI<double, HYPRE_BigInt>(
      (HYPRE_BigInt)num_loc_rows, II.GetData(), J, data, dist.GetData(),
      comm, sym_sparse);
#endif
}

STRUMPACKRowLocMatrix::STRUMPACKRowLocMatrix(const Operator &op,
                                             bool sym_sparse)
{
   const HypreParMatrix *APtr = dynamic_cast<const HypreParMatrix *>(&op);
   MFEM_VERIFY(APtr, "Not a compatible matrix type");
   MPI_Comm comm = APtr->GetComm();

   // Set mfem::Operator member data
   height = op.Height();
   width  = op.Width();

   // First cast the parameter to a hypre_ParCSRMatrix
   hypre_ParCSRMatrix *parcsr_op =
      (hypre_ParCSRMatrix *)const_cast<HypreParMatrix &>(*APtr);

   // Create the CSRMatrixMPI A by taking the internal data from a
   // hypre_CSRMatrix
   APtr->HostRead();
   hypre_CSRMatrix *csr_op = hypre_MergeDiagAndOffd(parcsr_op);
   APtr->HypreRead();
   HYPRE_Int       *Iptr   = csr_op->i;
#if MFEM_HYPRE_VERSION >= 21600
   HYPRE_BigInt    *Jptr   = csr_op->big_j;
#else
   HYPRE_Int       *Jptr   = csr_op->j;
#endif
   double          *data   = csr_op->data;

   HYPRE_BigInt fst_row = parcsr_op->first_row_index;
   HYPRE_Int    m_loc   = csr_op->num_rows;

   // Allocate STRUMPACK's CSRMatrixMPI
   int rank, nprocs;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &nprocs);
   Array<HYPRE_BigInt> dist(nprocs + 1);
   dist[0] = 0;
   dist[rank + 1] = fst_row + (HYPRE_BigInt)m_loc;
   MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                 dist.GetData() + 1, 1, HYPRE_MPI_BIG_INT, comm);

#if !defined(HYPRE_MIXEDINT)
   A_ = new strumpack::CSRMatrixMPI<double, HYPRE_BigInt>(
      (HYPRE_BigInt)m_loc, Iptr, Jptr, data, dist.GetData(),
      comm, sym_sparse);
#else
   Array<HYPRE_BigInt> II(m_loc+1);
   for (int i = 0; i <= m_loc; i++) { II[i] = (HYPRE_BigInt)Iptr[i]; }
   A_ = new strumpack::CSRMatrixMPI<double, HYPRE_BigInt>(
      (HYPRE_BigInt)m_loc, II.GetData(), Jptr, data, dist.GetData(),
      comm, sym_sparse);
#endif

   // Everything has been copied so delete the structure
   hypre_CSRMatrixDestroy(csr_op);
}

STRUMPACKRowLocMatrix::~STRUMPACKRowLocMatrix()
{
   delete A_;
}

template <typename STRUMPACKSolverType>
STRUMPACKSolverBase<STRUMPACKSolverType>::
STRUMPACKSolverBase(MPI_Comm comm, int argc, char *argv[])
   : APtr_(NULL),
     factor_verbose_(false),
     solve_verbose_(false),
     reorder_reuse_(false),
     nrhs_(-1)
{
   solver_ = new STRUMPACKSolverType(comm, argc, argv, false);
}

template <typename STRUMPACKSolverType>
STRUMPACKSolverBase<STRUMPACKSolverType>::
STRUMPACKSolverBase(STRUMPACKRowLocMatrix &A, int argc, char *argv[])
   : APtr_(&A),
     factor_verbose_(false),
     solve_verbose_(false),
     reorder_reuse_(false),
     nrhs_(-1)
{
   solver_ = new STRUMPACKSolverType(A.GetComm(), argc, argv, false);
   SetOperator(A);
}

template <typename STRUMPACKSolverType>
STRUMPACKSolverBase<STRUMPACKSolverType>::
~STRUMPACKSolverBase()
{
   delete solver_;
}

template <typename STRUMPACKSolverType>
void STRUMPACKSolverBase<STRUMPACKSolverType>::
SetFromCommandLine()
{
   solver_->options().set_from_command_line();
}

template <typename STRUMPACKSolverType>
void STRUMPACKSolverBase<STRUMPACKSolverType>::
SetPrintFactorStatistics(bool print_stat)
{
   factor_verbose_ = print_stat;
}

template <typename STRUMPACKSolverType>
void STRUMPACKSolverBase<STRUMPACKSolverType>::
SetPrintSolveStatistics(bool print_stat)
{
   solve_verbose_ = print_stat;
}

template <typename STRUMPACKSolverType>
void STRUMPACKSolverBase<STRUMPACKSolverType>
::SetRelTol(double rtol)
{
   solver_->options().set_rel_tol(rtol);
}

template <typename STRUMPACKSolverType>
void STRUMPACKSolverBase<STRUMPACKSolverType>
::SetAbsTol(double atol)
{
   solver_->options().set_abs_tol(atol);
}

template <typename STRUMPACKSolverType>
void STRUMPACKSolverBase<STRUMPACKSolverType>
::SetMaxIter(int max_it)
{
   solver_->options().set_maxit(max_it);
}

template <typename STRUMPACKSolverType>
void STRUMPACKSolverBase<STRUMPACKSolverType>
::SetReorderingReuse(bool reuse)
{
   reorder_reuse_ = reuse;
}

template <typename STRUMPACKSolverType>
void STRUMPACKSolverBase<STRUMPACKSolverType>
::EnableGPU()
{
   solver_->options().enable_gpu();
}

template <typename STRUMPACKSolverType>
void STRUMPACKSolverBase<STRUMPACKSolverType>
::DisableGPU()
{
   solver_->options().disable_gpu();
}

template <typename STRUMPACKSolverType>
void STRUMPACKSolverBase<STRUMPACKSolverType>::
SetKrylovSolver(strumpack::KrylovSolver method)
{
   solver_->options().set_Krylov_solver(method);
}

template <typename STRUMPACKSolverType>
void STRUMPACKSolverBase<STRUMPACKSolverType>::
SetReorderingStrategy(strumpack::ReorderingStrategy method)
{
   solver_->options().set_reordering_method(method);
}

template <typename STRUMPACKSolverType>
void STRUMPACKSolverBase<STRUMPACKSolverType>::
SetMatching(strumpack::MatchingJob job)
{
   solver_->options().set_matching(job);
}

template <typename STRUMPACKSolverType>
void STRUMPACKSolverBase<STRUMPACKSolverType>::
SetCompression(strumpack::CompressionType type)
{
#if STRUMPACK_VERSION_MAJOR >= 5
   solver_->options().set_compression(type);
#else
   switch (type)
   {
      case strumpack::NONE:
         solver_->options().disable_BLR();
         solver_->options().disable_HSS();
         break;
      case strumpack::BLR:
         solver_->options().enable_BLR();
         break;
      case strumpack::HSS:
         solver_->options().enable_HSS();
         break;
      default:
         MFEM_ABORT("Invalid compression type for STRUMPACK version " <<
                    STRUMPACK_VERSION_MAJOR << "!");
         break;
   }
#endif
}

template <typename STRUMPACKSolverType>
void STRUMPACKSolverBase<STRUMPACKSolverType>::
SetCompressionRelTol(double rtol)
{
#if STRUMPACK_VERSION_MAJOR >= 5
   solver_->options().set_compression_rel_tol(rtol);
#else
   solver_->options().BLR_options().set_rel_tol(rtol);
   solver_->options().HSS_options().set_rel_tol(rtol);
#endif
}

template <typename STRUMPACKSolverType>
void STRUMPACKSolverBase<STRUMPACKSolverType>::
SetCompressionAbsTol(double atol)
{
#if STRUMPACK_VERSION_MAJOR >= 5
   solver_->options().set_compression_abs_tol(atol);
#else
   solver_->options().BLR_options().set_abs_tol(atol);
   solver_->options().HSS_options().set_abs_tol(atol);
#endif
}

#if STRUMPACK_VERSION_MAJOR >= 5
template <typename STRUMPACKSolverType>
void STRUMPACKSolverBase<STRUMPACKSolverType>::
SetCompressionLossyPrecision(int precision)
{
   solver_->options().set_lossy_precision(precision);
}

template <typename STRUMPACKSolverType>
void STRUMPACKSolverBase<STRUMPACKSolverType>::
SetCompressionButterflyLevels(int levels)
{
   solver_->options().HODLR_options().set_butterfly_levels(levels);
}
#endif

template <typename STRUMPACKSolverType>
void STRUMPACKSolverBase<STRUMPACKSolverType>::
SetOperator(const Operator &op)
{
   // Verify that we have a compatible operator
   bool first_mat = !APtr_;
   APtr_ = dynamic_cast<const STRUMPACKRowLocMatrix *>(&op);
   MFEM_VERIFY(APtr_,
               "STRUMPACK: Operator is not a STRUMPACKRowLocMatrix!");

   // Set mfem::Operator member data
   height = op.Height();
   width  = op.Width();

   if (first_mat || !reorder_reuse_)
   {
      solver_->set_matrix(*(APtr_->GetA()));
   }
   else
   {
      solver_->update_matrix_values(*(APtr_->GetA()));
   }
}

template <typename STRUMPACKSolverType>
void STRUMPACKSolverBase<STRUMPACKSolverType>::
FactorInternal() const
{
   MFEM_ASSERT(APtr_,
               "STRUMPACK: Operator must be set before the system can be "
               "solved!");
   solver_->options().set_verbose(factor_verbose_);
   strumpack::ReturnCode ret = solver_->factor();
   if (ret != strumpack::ReturnCode::SUCCESS)
   {
#if STRUMPACK_VERSION_MAJOR >= 7
      MFEM_ABORT("STRUMPACK: Factor failed with return code " << ret << "!");
#else
      MFEM_ABORT("STRUMPACK: Factor failed!");
#endif
   }
}

template <typename STRUMPACKSolverType>
void STRUMPACKSolverBase<STRUMPACKSolverType>::
Mult(const Vector &x, Vector &y) const
{
   MFEM_ASSERT(x.Size() == Width(),
               "STRUMPACK: Invalid x.Size() = " << x.Size() <<
               ", expected size = " << Width() << "!");
   MFEM_ASSERT(y.Size() == Height(),
               "STRUMPACK: Invalid y.Size() = " << y.Size() <<
               ", expected size = " << Height() << "!");

   const double *xPtr = x.HostRead();
   double *yPtr       = y.HostReadWrite();

   FactorInternal();
   solver_->options().set_verbose(solve_verbose_);
   strumpack::ReturnCode ret = solver_->solve(xPtr, yPtr, false);
   if (ret != strumpack::ReturnCode::SUCCESS)
   {
#if STRUMPACK_VERSION_MAJOR >= 7
      MFEM_ABORT("STRUMPACK: Solve failed with return code " << ret << "!");
#else
      MFEM_ABORT("STRUMPACK: Solve failed!");
#endif
   }
}

template <typename STRUMPACKSolverType>
void STRUMPACKSolverBase<STRUMPACKSolverType>::
ArrayMult(const Array<const Vector *> &X, Array<Vector *> &Y) const
{
   MFEM_ASSERT(X.Size() == Y.Size(),
               "Number of columns mismatch in STRUMPACK solve!");
   if (X.Size() == 1)
   {
      nrhs_ = 1;
      MFEM_ASSERT(X[0] && Y[0], "Missing Vector in STRUMPACK solve!");
      Mult(*X[0], *Y[0]);
      return;
   }

   // Multiple RHS case
   int ldx = Height();
   if (nrhs_ != X.Size())
   {
      rhs_.SetSize(X.Size() * ldx);
      sol_.SetSize(X.Size() * ldx);
      nrhs_ = X.Size();
   }
   for (int i = 0; i < nrhs_; i++)
   {
      MFEM_ASSERT(X[i] && X[i]->Size() == Width(),
                  "STRUMPACK: Missing or invalid sized RHS Vector in solve!");
      Vector s(rhs_, i * ldx, ldx);
      s = *X[i];
      rhs_.SyncMemory(s);  // Update flags for rhs_ if updated on device
   }
   const double *xPtr = rhs_.HostRead();
   double *yPtr       = sol_.HostReadWrite();

   FactorInternal();
   solver_->options().set_verbose(solve_verbose_);
   strumpack::ReturnCode ret = solver_->solve(nrhs_, xPtr, ldx, yPtr, ldx,
                                              false);
   if (ret != strumpack::ReturnCode::SUCCESS)
   {
#if STRUMPACK_VERSION_MAJOR >= 7
      MFEM_ABORT("STRUMPACK: Solve failed with return code " << ret << "!");
#else
      MFEM_ABORT("STRUMPACK: Solve failed!");
#endif
   }

   for (int i = 0; i < nrhs_; i++)
   {
      MFEM_ASSERT(Y[i] && Y[i]->Size() == Width(),
                  "STRUMPACK: Missing or invalid sized solution Vector in solve!");
      Vector s(sol_, i * ldx, ldx);
      *Y[i] = s;
   }
}

STRUMPACKSolver::
STRUMPACKSolver(MPI_Comm comm)
   : STRUMPACKSolverBase<strumpack::
     SparseSolverMPIDist<double, HYPRE_BigInt>>
     (comm, 0, NULL) {}

STRUMPACKSolver::
STRUMPACKSolver(STRUMPACKRowLocMatrix &A)
   : STRUMPACKSolverBase<strumpack::
     SparseSolverMPIDist<double, HYPRE_BigInt>>
     (A, 0, NULL) {}

STRUMPACKSolver::
STRUMPACKSolver(MPI_Comm comm, int argc, char *argv[])
   : STRUMPACKSolverBase<strumpack::
     SparseSolverMPIDist<double, HYPRE_BigInt>>
     (comm, argc, argv) {}

STRUMPACKSolver::
STRUMPACKSolver(STRUMPACKRowLocMatrix &A, int argc, char *argv[])
   : STRUMPACKSolverBase<strumpack::
     SparseSolverMPIDist<double, HYPRE_BigInt>>
     (A, argc, argv) {}

#if STRUMPACK_VERSION_MAJOR >= 7
STRUMPACKMixedPrecisionSolver::
STRUMPACKMixedPrecisionSolver(MPI_Comm comm)
   : STRUMPACKSolverBase<strumpack::
     SparseSolverMixedPrecisionMPIDist<float, double, HYPRE_BigInt>>
     (comm, 0, NULL) {}

STRUMPACKMixedPrecisionSolver::
STRUMPACKMixedPrecisionSolver(STRUMPACKRowLocMatrix &A)
   : STRUMPACKSolverBase<strumpack::
     SparseSolverMixedPrecisionMPIDist<float, double, HYPRE_BigInt>>
     (A, 0, NULL) {}

STRUMPACKMixedPrecisionSolver::
STRUMPACKMixedPrecisionSolver(MPI_Comm comm, int argc, char *argv[])
   : STRUMPACKSolverBase<strumpack::
     SparseSolverMixedPrecisionMPIDist<float, double, HYPRE_BigInt>>
     (comm, argc, argv) {}

STRUMPACKMixedPrecisionSolver::
STRUMPACKMixedPrecisionSolver(STRUMPACKRowLocMatrix &A, int argc, char *argv[])
   : STRUMPACKSolverBase<strumpack::
     SparseSolverMixedPrecisionMPIDist<float, double, HYPRE_BigInt>>
     (A, argc, argv) {}
#endif

template class STRUMPACKSolverBase<strumpack::
                                   SparseSolverMPIDist<double, HYPRE_BigInt>>;
#if STRUMPACK_VERSION_MAJOR >= 7
template class STRUMPACKSolverBase<strumpack::
                                   SparseSolverMixedPrecisionMPIDist<float, double, HYPRE_BigInt>>;
#endif

} // mfem namespace

#endif // MFEM_USE_MPI
#endif // MFEM_USE_STRUMPACK
