// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../config/config.hpp"

#ifdef MFEM_USE_STRUMPACK
#ifdef MFEM_USE_MPI

#include "strumpack.hpp"

using namespace std;
using namespace strumpack;

namespace mfem
{

STRUMPACKRowLocMatrix::STRUMPACKRowLocMatrix(MPI_Comm comm,
                                             int num_loc_rows,
                                             int first_loc_row,
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
   A_ = new CSRMatrixMPI<double,int>(num_loc_rows, I, J, data,
                                     dist, comm_, false);
   delete [] dist;
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
   // hypre_CSRMatrixSetDataOwner(csr_op,0);

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
   delete [] dist;

   // Everything has been copied or abducted so delete the structure
   hypre_CSRMatrixDestroy(csr_op);
}

STRUMPACKRowLocMatrix::~STRUMPACKRowLocMatrix()
{
   // Delete the struct
   delete A_;
}

STRUMPACKRowLocCmplxMatrix::STRUMPACKRowLocCmplxMatrix(MPI_Comm comm,
                                                       int num_loc_rows,
                                                       int first_loc_row,
                                                       int glob_nrows,
                                                       int glob_ncols,
                                                       int *I, int *J,
                                                       complex<double> *data)
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
   A_ = new CSRMatrixMPI<complex<double>,int>(num_loc_rows, I, J, data,
                                              dist, comm_, false);
   delete [] dist;
}

STRUMPACKRowLocCmplxMatrix::STRUMPACKRowLocCmplxMatrix(
   const HypreParMatrix & hypParMat_R,
   const HypreParMatrix & hypParMat_I)
   : comm_(hypParMat_R.GetComm()),
     A_(NULL)
{
   // Check for compatible dimensions
   MFEM_ASSERT(hypParMat_R.Width() == hypParMat_I.Width(),
               "STRUMPACK: "
               "real and imaginary operators have different widths");
   MFEM_ASSERT(hypParMat_R.Height() == hypParMat_I.Height(),
               "STRUMPACK: "
               "real and imaginary operators have different heights");

   // First cast the parameters to hypre_ParCSRMatrices
   hypre_ParCSRMatrix * parcsr_op_R =
      (hypre_ParCSRMatrix *)const_cast<HypreParMatrix&>(hypParMat_R);
   hypre_ParCSRMatrix * parcsr_op_I =
      (hypre_ParCSRMatrix *)const_cast<HypreParMatrix&>(hypParMat_I);

   MFEM_ASSERT(parcsr_op_R != NULL,"STRUMPACK: "
               "const_cast failed for real part in SetOperator");
   MFEM_ASSERT(parcsr_op_I != NULL,"STRUMPACK: "
               "const_cast failed for imaginary part in SetOperator");

   // Create the CSRMatrixMPI A_ by borrowing the internal data from a
   // hypre_CSRMatrix.
   hypre_CSRMatrix * csr_op_R = hypre_MergeDiagAndOffd(parcsr_op_R);
   hypre_CSRMatrix * csr_op_I = hypre_MergeDiagAndOffd(parcsr_op_I);

   height = csr_op_R->num_rows;
   width  = csr_op_I->num_rows;

   int nprocs, rank;
   MPI_Comm_rank(comm_, &rank);
   MPI_Comm_size(comm_, &nprocs);
   int * dist = new int[nprocs + 1];
   dist[rank + 1] = parcsr_op_R->first_row_index + csr_op_R->num_rows;
   dist[0] = 0;
   MPI_Allgather(MPI_IN_PLACE, 0, MPI_INT, dist + 1, 1, MPI_INT, comm_);

   // Merge the sparsity patterns of the two matrices
   double * data_tmp = NULL;
   double * zeros = NULL;

   zeros = new double[csr_op_R->num_nonzeros];
   for (int i=0; i<csr_op_R->num_nonzeros; i++) { zeros[i] = 0.0; }

   data_tmp = csr_op_R->data;
   csr_op_R->data = zeros;
   hypre_CSRMatrix * csr_op_I_new = hypre_CSRMatrixAdd(csr_op_R, csr_op_I);
   csr_op_R->data = data_tmp;

   if (csr_op_I->num_nonzeros > csr_op_R->num_nonzeros)
   {
      delete [] zeros;
      zeros = new double[csr_op_I->num_nonzeros];
      for (int i=0; i<csr_op_I->num_nonzeros; i++) { zeros[i] = 0.0; }
   }

   data_tmp = csr_op_I->data;
   csr_op_I->data = zeros;
   hypre_CSRMatrix * csr_op_R_new = hypre_CSRMatrixAdd(csr_op_R, csr_op_I);
   csr_op_I->data = data_tmp;

   delete [] zeros;

   // Everything has been copied or abducted so delete the structure
   hypre_CSRMatrixDestroy(csr_op_R);
   hypre_CSRMatrixDestroy(csr_op_I);

   // At this point the two csr_op_?_new operators should have
   // identical sparsity patterns.  Even the order of nonzeros should
   // match.
   complex<double> * data = new complex<double>[csr_op_R_new->num_nonzeros];
   for (int i=0; i<csr_op_R_new->num_nonzeros; i++)
   {
      data[i] = complex<double>(csr_op_R_new->data[i], csr_op_I_new->data[i]);
   }
   A_ = new CSRMatrixMPI<complex<double>,int>(csr_op_R_new->num_rows,
                                              csr_op_R_new->i, csr_op_R_new->j,
                                              data, dist, comm_, false);
   delete [] data;
   delete [] dist;

   // Everything has been copied or abducted so delete the structure
   hypre_CSRMatrixDestroy(csr_op_R_new);
   hypre_CSRMatrixDestroy(csr_op_I_new);
}

STRUMPACKRowLocCmplxMatrix::~STRUMPACKRowLocCmplxMatrix()
{
   // Delete the struct
   delete A_;
}

template<typename scalar_t, typename integer_t>
STRUMPACKBaseSolver<scalar_t,integer_t>::
STRUMPACKBaseSolver(int argc, char* argv[], MPI_Comm comm )
   : comm_(comm),
     solver_(NULL)
{
   this->Init(argc, argv);
}

template<typename scalar_t, typename integer_t>
STRUMPACKBaseSolver<scalar_t,integer_t>::~STRUMPACKBaseSolver()
{
   if ( solver_ != NULL ) { delete solver_; }
}

template<typename scalar_t, typename integer_t>
void STRUMPACKBaseSolver<scalar_t,integer_t>::Init( int argc, char* argv[] )
{
   MPI_Comm_size(comm_, &numProcs_);
   MPI_Comm_rank(comm_, &myid_);

   factor_verbose_ = false;
   solve_verbose_ = false;

   solver_ = new StrumpackSparseSolverMPIDist<scalar_t,integer_t>(comm_,
                                                                  argc,
                                                                  argv,
                                                                  false);
}

template<typename scalar_t, typename integer_t>
void STRUMPACKBaseSolver<scalar_t,integer_t>::SetFromCommandLine( )
{
   solver_->options().set_from_command_line( );
}

template<typename scalar_t, typename integer_t>
void STRUMPACKBaseSolver<scalar_t,integer_t>::
SetPrintFactorStatistics( bool print_stat )
{
   factor_verbose_ = print_stat;
}

template<typename scalar_t, typename integer_t>
void STRUMPACKBaseSolver<scalar_t,integer_t>::
SetPrintSolveStatistics( bool print_stat )
{
   solve_verbose_ = print_stat;
}

template<typename scalar_t, typename integer_t>
void STRUMPACKBaseSolver<scalar_t,integer_t>::
SetKrylovSolver( strumpack::KrylovSolver method )
{
   solver_->options().set_Krylov_solver( method );
}

template<typename scalar_t, typename integer_t>
void STRUMPACKBaseSolver<scalar_t,integer_t>::
SetReorderingStrategy( strumpack::ReorderingStrategy
                       method )
{
   solver_->options().set_reordering_method( method );
}

template<typename scalar_t, typename integer_t>
void STRUMPACKBaseSolver<scalar_t,integer_t>::
SetMC64Job( strumpack::MC64Job job )
{
   solver_->options().set_mc64job( job );
}

template<typename scalar_t, typename integer_t>
void STRUMPACKBaseSolver<scalar_t,integer_t>::SetRelTol( double rtol )
{
   solver_->options().set_rel_tol( rtol );
}

template<typename scalar_t, typename integer_t>
void STRUMPACKBaseSolver<scalar_t,integer_t>::SetAbsTol( double atol )
{
   solver_->options().set_abs_tol( atol );
}


STRUMPACKSolver::STRUMPACKSolver(int argc, char* argv[], MPI_Comm comm)
   : STRUMPACKBaseSolver<double,int>(argc, argv, comm)
{
}

STRUMPACKSolver::STRUMPACKSolver( STRUMPACKRowLocMatrix & A )
   : STRUMPACKBaseSolver<double,int>(0, NULL, A.GetComm()),
     APtr_(&A)
{
   height = A.Height();
   width  = A.Width();
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

STRUMPACKCmplxSolver::STRUMPACKCmplxSolver(int argc, char* argv[],
                                           MPI_Comm comm)
   : STRUMPACKBaseSolver<complex<double>,int>(argc, argv, comm),
     xPtr_(NULL),
     yPtr_(NULL)
{
}

STRUMPACKCmplxSolver::STRUMPACKCmplxSolver( STRUMPACKRowLocCmplxMatrix & A )
   : STRUMPACKBaseSolver<complex<double>,int>(0, NULL, A.GetComm()),
     APtr_(&A)
{
   height = A.Height();
   width  = A.Width();

   xPtr_ = new complex<double>[width];
   yPtr_ = new complex<double>[height];
}

STRUMPACKCmplxSolver::~STRUMPACKCmplxSolver()
{
   delete [] xPtr_;
   delete [] yPtr_;
}

void STRUMPACKCmplxSolver::Mult( const Vector & x, Vector & y ) const
{
   MFEM_ASSERT(APtr_ != NULL,
               "STRUMPACK Error: The operator must be set before"
               " the system can be solved.");
   MFEM_ASSERT(x.Size() == 2 * Width(), "invalid x.Size() = " << x.Size()
               << ", expected size = " << 2 * Width());
   MFEM_ASSERT(y.Size() == 2 * Height(), "invalid y.Size() = " << y.Size()
               << ", expected size = " << 2 * Height());

   for (int i=0; i<width; i++)
   {
      xPtr_[i] = complex<double>(x[i], x[i + width]);
   }

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
   solver_->solve(xPtr_, yPtr_);

   for (int i=0; i<height; i++)
   {
      y[i]          = yPtr_[i].real();
      y[i + height] = yPtr_[i].imag();
   }
}

void STRUMPACKCmplxSolver::SetOperator( const Operator & op )
{
   // Verify that we have a compatible operator
   APtr_ = dynamic_cast<const STRUMPACKRowLocCmplxMatrix*>(&op);
   if ( APtr_ == NULL )
   {
      mfem_error("STRUMPACKCmplxSolver::SetOperator "
                 ": not STRUMPACKRowLocCmplxMatrix!");
   }

   solver_->set_matrix( *(APtr_->getA()) );

   // Set mfem::Operator member data
   height = op.Height();
   width  = op.Width();

}

} // mfem namespace

#endif // MFEM_USE_MPI
#endif // MFEM_USE_STRUMPACK
