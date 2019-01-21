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
    double precision types. See http://portal.nersc.gov/project/sparse/strumpack
*/
class STRUMPACKSolver : public mfem::Solver
{
public:
   // Constructor with MPI_Comm parameter.
   STRUMPACKSolver( int argc, char* argv[], MPI_Comm comm );

   // Constructor with STRUMPACK Matrix Object.
   STRUMPACKSolver( STRUMPACKRowLocMatrix & A);

   // Default destructor.
   ~STRUMPACKSolver( void );

   // Factor and solve the linear system y = Op^{-1} x.
   void Mult( const Vector & x, Vector & y ) const;

   // Set the operator.
   void SetOperator( const Operator & op );

   // Set various solver options. Refer to STRUMPACK documentation for
   // details.
   void SetFromCommandLine( );
   void SetPrintFactorStatistics( bool print_stat );
   void SetPrintSolveStatistics( bool print_stat );
   void SetRelTol( double rtol );
   void SetAbsTol( double atol );

   /**
    * STRUMPACK is an (approximate) direct solver. It can be used as a direct
    * solver or as a preconditioner. To use STRUMPACK as only a preconditioner,
    * set the Krylov solver to DIRECT. STRUMPACK also provides iterative solvers
    * which can use the preconditioner, and these iterative solvers can also be
    * used without preconditioner.
    *
    * Supported values are:
    *    AUTO:           Use iterative refinement if no HSS compression is used,
    *                    otherwise use GMRes.
    *    DIRECT:         No outer iterative solver, just a single application of
    *                    the multifrontal solver.
    *    REFINE:         Iterative refinement.
    *    PREC_GMRES:     Preconditioned GMRes.
    *                    The preconditioner is the (approx) multifrontal solver.
    *    GMRES:          UN-preconditioned GMRes. (for testing mainly)
    *    PREC_BICGSTAB:  Preconditioned BiCGStab.
    *                    The preconditioner is the (approx) multifrontal solver.
    *    BICGSTAB:       UN-preconditioned BiCGStab. (for testing mainly)
    */
   void SetKrylovSolver( strumpack::KrylovSolver method );

   /**
    * Supported reorderings are:
    *    METIS, PARMETIS, SCOTCH, PTSCOTCH, RCM
    */
   void SetReorderingStrategy( strumpack::ReorderingStrategy method );

   /**
    * Disable static pivoting for stability. The static pivoting in strumpack
    * permutes the sparse input matrix in order to get large (nonzero) elements
    * on the diagonal. If the input matrix is already diagonally dominant, this
    * reordering can be disabled.
    */
   void DisableMatching();

   /**
    * Enable static pivoting for stability using the MC64 algorithm with
    * job=5. Using a matching algorithm, this will permute the sparse input
    * matrix in order to get nonzero elements (as large as possible) on the
    * diagonal. And will also scale the rows and columns of the matrix.
    */
   void EnableMatching();

#if STRUMPACK_VERSION_MAJOR >= 3
   /**
    * Use the AWPM (approximate weight perfect matching) algorithm from the
    * Combinatorial BLAS library for static pivoting, i.e. getting large
    * nonzeros on the diagonal. This requires that strumpack was compiled with
    * support for Combinatorial BLAS.
    */
   void EnableParallelMatching();
#endif

private:
   void Init( int argc, char* argv[] );

protected:

   MPI_Comm      comm_;
   int           numProcs_;
   int           myid_;

   bool factor_verbose_;
   bool solve_verbose_;

   const STRUMPACKRowLocMatrix * APtr_;
   strumpack::StrumpackSparseSolverMPIDist<double,int> * solver_;

}; // mfem::STRUMPACKSolver class

} // mfem namespace

#endif // MFEM_USE_MPI
#endif // MFEM_USE_STRUMPACK
#endif // MFEM_STRUMPACK
