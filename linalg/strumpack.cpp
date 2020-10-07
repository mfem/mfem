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

#ifdef MFEM_USE_STRUMPACK
#ifdef MFEM_USE_MPI

#include "strumpack.hpp"

using namespace std;
using namespace strumpack;

namespace mfem
{

STRUMPACKRowLocMatrix::STRUMPACKRowLocMatrix(MPI_Comm comm,
                                             int num_loc_rows, int first_loc_row,
                                             int glob_nrows, int glob_ncols,
                                             int *I, int *J, double *data)
   : comm_(comm), A_(NULL)
{
   // Set mfem::Operator member data
   height = num_loc_rows;
   width  = num_loc_rows;

   // Allocate STRUMPACK's CSRMatrixMPI
   int nprocs, rank;
   MPI_Comm_rank(comm_, &rank);
   MPI_Comm_size(comm_, &nprocs);
   int * dist = new int[nprocs + 1];
   dist[rank + 1] = first_loc_row + num_loc_rows;
   dist[0] = 0;
   MPI_Allgather(MPI_IN_PLACE, 0, MPI_INT, dist + 1, 1, MPI_INT, comm_);
   A_ = new CSRMatrixMPI<double,int>(num_loc_rows, I, J, data, dist, comm_, false);
   delete[] dist;
}

STRUMPACKRowLocMatrix::STRUMPACKRowLocMatrix(const HypreParMatrix & hypParMat)
   : comm_(hypParMat.GetComm()),
     A_(NULL)
{
   // First cast the parameter to a hypre_ParCSRMatrix
   hypre_ParCSRMatrix * parcsr_op =
      (hypre_ParCSRMatrix *)const_cast<HypreParMatrix&>(hypParMat);

   MFEM_ASSERT(parcsr_op != NULL,"STRUMPACK: const_cast failed in SetOperator");

   // Create the CSRMatrixMPI A_ by borrowing the internal data from a
   // hypre_CSRMatrix.
   hypre_CSRMatrix * csr_op = hypre_MergeDiagAndOffd(parcsr_op);
   hypre_CSRMatrixSetDataOwner(csr_op,0);

   height = csr_op->num_rows;
   width  = csr_op->num_rows;

   int nprocs, rank;
   MPI_Comm_rank(comm_, &rank);
   MPI_Comm_size(comm_, &nprocs);
   int * dist = new int[nprocs + 1];
   dist[rank + 1] = parcsr_op->first_row_index + csr_op->num_rows;
   dist[0] = 0;
   MPI_Allgather(MPI_IN_PLACE, 0, MPI_INT, dist + 1, 1, MPI_INT, comm_);
   A_ = new CSRMatrixMPI<double,int>(csr_op->num_rows, csr_op->i, csr_op->j,
                                     csr_op->data, dist, comm_, false);
   delete[] dist;

   // Everything has been copied or abducted so delete the structure
   hypre_CSRMatrixDestroy(csr_op);
}

STRUMPACKRowLocMatrix::~STRUMPACKRowLocMatrix()
{
   // Delete the struct
   if ( A_ != NULL ) { delete A_; }
}

STRUMPACKSolver::STRUMPACKSolver( int argc, char* argv[], MPI_Comm comm )
   : comm_(comm),
     APtr_(NULL),
     solver_(NULL)
{
   this->Init(argc, argv);
}

STRUMPACKSolver::STRUMPACKSolver( STRUMPACKRowLocMatrix & A )
   : comm_(A.GetComm()),
     APtr_(&A),
     solver_(NULL)
{
   height = A.Height();
   width  = A.Width();

   this->Init(0, NULL);
}

STRUMPACKSolver::~STRUMPACKSolver()
{
   if ( solver_ != NULL ) { delete solver_; }
}

void STRUMPACKSolver::Init( int argc, char* argv[] )
{
   MPI_Comm_size(comm_, &numProcs_);
   MPI_Comm_rank(comm_, &myid_);

   factor_verbose_ = false;
   solve_verbose_ = false;

   solver_ = new StrumpackSparseSolverMPIDist<double,int>(comm_, argc, argv,
                                                          false);
}

void STRUMPACKSolver::SetFromCommandLine( )
{
   solver_->options().set_from_command_line( );
}

void STRUMPACKSolver::SetPrintFactorStatistics( bool print_stat )
{
   factor_verbose_ = print_stat;
}

void STRUMPACKSolver::SetPrintSolveStatistics( bool print_stat )
{
   solve_verbose_ = print_stat;
}

void STRUMPACKSolver::SetKrylovSolver( strumpack::KrylovSolver method )
{
   solver_->options().set_Krylov_solver( method );
}

void STRUMPACKSolver::SetReorderingStrategy( strumpack::ReorderingStrategy
                                             method )
{
   solver_->options().set_reordering_method( method );
}

void STRUMPACKSolver::DisableMatching( )
{
#if STRUMPACK_VERSION_MAJOR >= 3
   solver_->options().set_matching( strumpack::MatchingJob::NONE );
#else
   solver_->options().set_mc64job( strumpack::MC64Job::NONE );
#endif
}

void STRUMPACKSolver::EnableMatching( )
{
#if STRUMPACK_VERSION_MAJOR >= 3
   solver_->options().set_matching
   ( strumpack::MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING );
#else
   solver_->options().set_mc64job
   ( strumpack::MC64Job::MAX_DIAGONAL_PRODUCT_SCALING );
#endif
}

#if STRUMPACK_VERSION_MAJOR >= 3
void STRUMPACKSolver::EnableParallelMatching( )
{
   solver_->options().set_matching
   ( strumpack::MatchingJob::COMBBLAS );
}
#endif

void STRUMPACKSolver::SetRelTol( double rtol )
{
   solver_->options().set_rel_tol( rtol );
}

void STRUMPACKSolver::SetAbsTol( double atol )
{
   solver_->options().set_abs_tol( atol );
}


void STRUMPACKSolver::Mult( const Vector & x, Vector & y ) const
{
   MFEM_ASSERT(APtr_ != NULL,
               "STRUMPACK Error: The operator must be set before"
               " the system can be solved.");
   MFEM_ASSERT(x.Size() == Width(), "invalid x.Size() = " << x.Size()
               << ", expected size = " << Width());
   MFEM_ASSERT(y.Size() == Height(), "invalid y.Size() = " << y.Size()
               << ", expected size = " << Height());

   double*  yPtr = (double*)y;
   double*  xPtr = (double*)(const_cast<Vector&>(x));

   solver_->options().set_verbose( factor_verbose_ );
   ReturnCode ret = solver_->factor();
   switch (ret)
   {
      case ReturnCode::SUCCESS: break;
      case ReturnCode::MATRIX_NOT_SET:
      {
         MFEM_ABORT("STRUMPACK:  Matrix was not set!");
      }
      break;
      case ReturnCode::REORDERING_ERROR:
      {
         MFEM_ABORT("STRUMPACK:  Matrix reordering failed!");
      }
      break;
   }
   solver_->options().set_verbose( solve_verbose_ );
   solver_->solve(xPtr, yPtr);

}

void STRUMPACKSolver::SetOperator( const Operator & op )
{
   // Verify that we have a compatible operator
   APtr_ = dynamic_cast<const STRUMPACKRowLocMatrix*>(&op);
   if ( APtr_ == NULL )
   {
      mfem_error("STRUMPACKSolver::SetOperator : not STRUMPACKRowLocMatrix!");
   }

   solver_->set_matrix( *(APtr_->getA()) );

   // Set mfem::Operator member data
   height = op.Height();
   width  = op.Width();

}

} // mfem namespace

#endif // MFEM_USE_MPI
#endif // MFEM_USE_STRUMPACK
