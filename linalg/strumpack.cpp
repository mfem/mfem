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
					     int num_loc_rows, int first_loc_row,
					     int glob_nrows, int glob_ncols,
					     int *I, int *J, double *data)
  : comm_(comm),
    A_(NULL)
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

   // for (int p=0; p<rank; p++) MPI_Barrier(comm_);
   // std::cout << "p=" << rank << " first_row=" << parcsr_op->first_row_index << std::endl;
   // std::cout << " I=[";
   // for (int i=0; i<csr_op->num_rows; i++) std::cout << csr_op->i[i] << " ";
   // std::cout << "];" << std::endl;
   // std::cout << " J=[";
   // for (int i=0; i<csr_op->num_nonzeros; i++) std::cout << csr_op->j[i] << " ";
   // std::cout << "];" << std::endl;
   // std::cout << " v=[";
   // for (int i=0; i<csr_op->num_nonzeros; i++) std::cout << csr_op->data[i] << " ";
   // std::cout << "];" << std::endl;
   // std::cout << " dist=[";
   // for (int i=0; i<=nprocs; i++) std::cout << dist[i] << " ";
   // std::cout << "];" << std::endl;
   // for (int p=rank; p<=nprocs; p++) MPI_Barrier(comm_);

   A_ = new CSRMatrixMPI<double,int>(csr_op->num_rows, csr_op->i, csr_op->j, csr_op->data, dist, comm_, false);
   delete[] dist;

   // Everything has been copied or abducted so delete the structure
   hypre_CSRMatrixDestroy(csr_op);
}

STRUMPACKRowLocMatrix::~STRUMPACKRowLocMatrix()
{
   // Delete the struct
   if ( A_ != NULL ) { delete A_; }
}

STRUMPACKSolver::STRUMPACKSolver( MPI_Comm comm )
   : comm_(comm),
     APtr_(NULL),
     solver_(NULL)
{
   this->Init();
}

STRUMPACKSolver::STRUMPACKSolver( STRUMPACKRowLocMatrix & A )
   : comm_(A.GetComm()),
     APtr_(&A),
     solver_(NULL)
{
   height = A.Height();
   width  = A.Width();

   this->Init();
}

STRUMPACKSolver::~STRUMPACKSolver()
{
  if ( solver_ != NULL ) { delete solver_; }
}

void STRUMPACKSolver::Init()
{
   MPI_Comm_size(comm_, &numProcs_);
   MPI_Comm_rank(comm_, &myid_);

   // TODO can we get the command line arguments in here??
   solver_ = new StrumpackSparseSolverMPIDist<double,int>(comm_, false);
}

void STRUMPACKSolver::SetPrintStatistics( bool print_stat )
{
  solver_->options().set_verbose(print_stat);
}

void STRUMPACKSolver::Mult( const Vector & x, Vector & y ) const
{
   MFEM_ASSERT(APtr_ != NULL,
               "STRUMPACK Error: The operator must be set before"
               " the system can be solved.");

   double*  yPtr = (double*)y;
   double*  xPtr = (double*)(const_cast<Vector&>(x));
   int      locSize = y.Size();

   ReturnCode ret = solver_->solve(xPtr, yPtr);
   switch (ret) {
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
