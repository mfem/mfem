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

#ifndef MFEM_STRUMPACK
#define MFEM_STRUMPACK

#include "../config/config.hpp"

#ifdef MFEM_USE_STRUMPACK
#ifdef MFEM_USE_MPI
#include "operator.hpp"
#include "hypre.hpp"

#include <mpi.h>

#include "StrumpackSparseSolverMPIDist.hpp"

namespace mfem
{

class STRUMPACKRowLocMatrix : public Operator
{
public:
   /** Creates a general parallel matrix from a local CSR matrix on each
       processor described by the I, J and data arrays. The local matrix should
       be of size (local) nrows by (global) glob_ncols. The new parallel matrix
       contains copies of all input arrays (so they can be deleted). */
   STRUMPACKRowLocMatrix(MPI_Comm comm,
			 int num_loc_rows, int first_loc_row,
			 int glob_nrows, int glob_ncols,
			 int *I, int *J, double *data);

   /** Creates a copy of the parallel matrix hypParMat in STRUMPACK's RowLoc
       format. All data is copied so the original matrix may be deleted. */
  STRUMPACKRowLocMatrix(const HypreParMatrix & hypParMat);

   ~STRUMPACKRowLocMatrix();

   void Mult(const Vector &x, Vector &y) const
   {
      mfem_error("STRUMPACKRowLocMatrix::Mult(...)\n"
                 "  matrix vector products are not supported.");
   }

   MPI_Comm GetComm() const { return comm_; }

   strumpack::CSRMatrixMPI<double,int>* getA() const { return A_; }

private:
   MPI_Comm   comm_;
   strumpack::CSRMatrixMPI<double,int>* A_;

}; // mfem::STRUMPACKRowLocMatrix

/** The MFEM STRUMPACK Direct Solver class.

    The mfem::STRUMPACKSolver class uses the STRUMPACK library to perform LU
    factorization of a parallel sparse matrix. The solver is capable of handling
    double precision types. See http://portal.nersc.gov/project/sparse/strumpack/.
*/
class STRUMPACKSolver : public mfem::Solver
{
public:
   // Constructor with MPI_Comm parameter.
   STRUMPACKSolver( MPI_Comm comm );

   // Constructor with STRUMPACK Matrix Object.
   STRUMPACKSolver( STRUMPACKRowLocMatrix & A);

   // Default destructor.
   ~STRUMPACKSolver( void );

   // Factor and solve the linear system y = Op^{-1} x.
   void Mult( const Vector & x, Vector & y ) const;

   // Set the operator.
   void SetOperator( const Operator & op );

   // TODO
   // Set various solver options. Refer to STRUMPACK documentation for details.
   void SetPrintStatistics( bool print_stat );

private:
   void Init();

protected:

   MPI_Comm      comm_;
   int           numProcs_;
   int           myid_;

   const STRUMPACKRowLocMatrix * APtr_;
   strumpack::StrumpackSparseSolverMPIDist<double,int> * solver_;

}; // mfem::STRUMPACKSolver class

} // mfem namespace

#endif // MFEM_USE_MPI
#endif // MFEM_USE_STRUMPACK
#endif // MFEM_STRUMPACK
