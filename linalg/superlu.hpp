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

#ifndef MFEM_SUPERLU
#define MFEM_SUPERLU

#include "../config/config.hpp"

#ifdef MFEM_USE_SUPERLU
#ifdef MFEM_USE_MPI
#include "operator.hpp"
#include "hypre.hpp"

#include <mpi.h>

namespace mfem
{

namespace superlu_internal
{
unsigned int sqrti(const unsigned int & a);
}

namespace superlu
{
// Copy selected enumerations from SuperLU
#ifdef MFEM_USE_SUPERLU5
typedef enum {NOROWPERM, LargeDiag, MY_PERMR}                       RowPerm;
#else
typedef enum {NOROWPERM, LargeDiag_MC64, LargeDiag_HWPM, MY_PERMR}  RowPerm;
#endif
typedef enum {NATURAL, MMD_ATA, MMD_AT_PLUS_A, COLAMD,
              METIS_AT_PLUS_A, PARMETIS, ZOLTAN, MY_PERMC
             }          ColPerm;
typedef enum {NOTRANS, TRANS, CONJ}                                 Trans;
typedef enum {NOREFINE, SLU_SINGLE=1, SLU_DOUBLE, SLU_EXTRA}        IterRefine;
}

class SuperLURowLocMatrix : public Operator
{
public:
   /** Creates a general parallel matrix from a local CSR matrix on each
       processor described by the I, J and data arrays. The local matrix should
       be of size (local) nrows by (global) glob_ncols. The new parallel matrix
       contains copies of all input arrays (so they can be deleted). */
   SuperLURowLocMatrix(MPI_Comm comm,
                       int num_loc_rows, int first_loc_row,
                       int glob_nrows, int glob_ncols,
                       int *I, int *J, double *data);

   /** Creates a copy of the parallel matrix hypParMat in SuperLU's RowLoc
       format. All data is copied so the original matrix may be deleted. */
   SuperLURowLocMatrix(const HypreParMatrix & hypParMat);

   ~SuperLURowLocMatrix();

   void Mult(const Vector &x, Vector &y) const
   {
      mfem_error("SuperLURowLocMatrix::Mult(...)\n"
                 "  matrix vector products are not supported.");
   }

   MPI_Comm GetComm() const { return comm_; }

   void * InternalData() const { return rowLocPtr_; }

private:
   MPI_Comm   comm_;
   void     * rowLocPtr_;

}; // mfem::SuperLURowLocMatrix

/** The MFEM SuperLU Direct Solver class.

    The mfem::SuperLUSolver class uses the SuperLU_DIST library to perform LU
    factorization of a parallel sparse matrix. The solver is capable of handling
    double precision types. It is currently maintained by Xiaoye Sherry Li at
    NERSC, see http://crd-legacy.lbl.gov/~xiaoye/SuperLU/.
*/
class SuperLUSolver : public mfem::Solver
{
public:
   // Constructor with MPI_Comm parameter.
   SuperLUSolver( MPI_Comm comm );

   // Constructor with SuperLU Matrix Object.
   SuperLUSolver( SuperLURowLocMatrix & A);

   // Default destructor.
   ~SuperLUSolver( void );

   // Allocate and deallocate the MPI communicators. This routine is called
   // internally by SetOperator().
   void SetupGrid();
   // This routing must be called after the solve, but before destruction.
   void DismantleGrid();

   // Factor and solve the linear system y = Op^{-1} x.
   void Mult( const Vector & x, Vector & y ) const;

   // Set the operator.
   void SetOperator( const Operator & op );

   // Set various solver options. Refer to SuperLU documentation for details.
   void SetPrintStatistics  ( bool              print_stat );
   void SetEquilibriate     ( bool                   equil );
   void SetColumnPermutation( superlu::ColPerm    col_perm );
   void SetRowPermutation   ( superlu::RowPerm    row_perm,
                              Array<int> *     perm = NULL );
   void SetTranspose        ( superlu::Trans         trans );
   void SetIterativeRefine  ( superlu::IterRefine iter_ref );
   void SetReplaceTinyPivot ( bool                     rtp );
   void SetNumLookAheads    ( int           num_lookaheads );
   void SetLookAheadElimTree( bool                   etree );
   void SetSymmetricPattern ( bool                     sym );

private:
   void Init();

protected:

   MPI_Comm      comm_;
   int           numProcs_;
   int           myid_;

   const SuperLURowLocMatrix * APtr_;

   // The actual types of the following pointers are hidden to avoid exposing
   // the SuperLU header files to the entire library. Their types are given in
   // the trailing comments. The reason that this is necessary is that SuperLU
   // defines these structs differently for use with its real and complex
   // solvers. If we want to add support for SuperLU's complex solvers one day
   // we will need to hide these types to avoid name conflicts.
   void*         optionsPtr_;         // superlu_options_t *
   void*         statPtr_;            //     SuperLUStat_t *
   void*         ScalePermstructPtr_; //  ScalePermsruct_t *
   void*         LUstructPtr_;        //        LUstruct_t *
   void*         SOLVEstructPtr_;     //     SOLVEstruct_t *
   void*         gridPtr_;            //        gridinfo_t *

   double*       berr_;
   mutable int*  perm_r_;
   int           nrhs_;
   int           nprow_;
   int           npcol_;
   mutable bool  firstSolveWithThisA_;
   bool          gridInitialized_;
   mutable bool  LUStructInitialized_;

}; // mfem::SuperLUSolver class

} // mfem namespace

#endif // MFEM_USE_MPI
#endif // MFEM_USE_SUPERLU
#endif // MFEM_SUPERLU
