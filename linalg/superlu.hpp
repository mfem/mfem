// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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

namespace superlu
{

// Copy selected enumerations from SuperLU (from superlu_enum_consts.h)
#ifdef MFEM_USE_SUPERLU5
typedef enum {NOROWPERM, LargeDiag, MY_PERMR}                      RowPerm;
#else
typedef enum {NOROWPERM, LargeDiag_MC64, LargeDiag_HWPM, MY_PERMR} RowPerm;
#endif
typedef enum {NATURAL, MMD_ATA, MMD_AT_PLUS_A, COLAMD,
              METIS_AT_PLUS_A, PARMETIS, ZOLTAN, MY_PERMC
             } ColPerm;
typedef enum {NOREFINE, SLU_SINGLE=1, SLU_DOUBLE, SLU_EXTRA} IterRefine;
typedef enum {DOFACT, SamePattern, SamePattern_SameRowPerm, FACTORED} Fact;

} // namespace superlu

class SuperLURowLocMatrix : public Operator
{
public:
   /** Creates a general parallel matrix from a local CSR matrix on each
       processor described by the I, J and data arrays. The local matrix should
       be of size (local) nrows by (global) glob_ncols. The new parallel matrix
       contains copies of all input arrays (so they can be deleted). */
   SuperLURowLocMatrix(MPI_Comm comm,
                       int num_loc_rows, HYPRE_BigInt first_loc_row,
                       HYPRE_BigInt glob_nrows, HYPRE_BigInt glob_ncols,
                       int *I, HYPRE_BigInt *J, double *data);

   /** Creates a copy of the parallel matrix hypParMat in SuperLU's RowLoc
       format. All data is copied so the original matrix may be deleted. */
   SuperLURowLocMatrix(const Operator &op);

   ~SuperLURowLocMatrix();

   void Mult(const Vector &x, Vector &y) const
   {
      MFEM_ABORT("SuperLURowLocMatrix::Mult: Matrix vector products are not "
                 "supported!");
   }

   void *InternalData() const { return rowLocPtr_; }

   MPI_Comm GetComm() const { return comm_; }

   HYPRE_BigInt GetGlobalNumRows() const { return num_global_rows_; }

   HYPRE_BigInt GetGlobalNumColumns() const { return num_global_cols_; }

private:
   MPI_Comm     comm_;
   void        *rowLocPtr_;
   HYPRE_BigInt num_global_rows_, num_global_cols_;
};

/** The MFEM SuperLU Direct Solver class.

    The mfem::SuperLUSolver class uses the SuperLU_DIST library to perform LU
    factorization of a parallel sparse matrix. The solver is capable of handling
    double precision types. It is currently maintained by Xiaoye Sherry Li at
    NERSC, see http://crd-legacy.lbl.gov/~xiaoye/SuperLU/.
*/
class SuperLUSolver : public Solver
{
public:
   // Constructor with MPI_Comm parameter.
   SuperLUSolver(MPI_Comm comm, int npdep = 1);

   // Constructor with SuperLU matrix object.
   SuperLUSolver(SuperLURowLocMatrix &A, int npdep = 1);

   // Default destructor.
   ~SuperLUSolver();

   // Set the operator.
   void SetOperator(const Operator &op);

   // Factor and solve the linear system y = Op^{-1} x.
   // Note: Factorization modifies the operator matrix.
   void Mult(const Vector &x, Vector &y) const;
   void ArrayMult(const Array<const Vector *> &X, Array<Vector *> &Y) const;

   // Factor and solve the linear system y = Op^{-T} x.
   // Note: Factorization modifies the operator matrix.
   void MultTranspose(const Vector &x, Vector &y) const;
   void ArrayMultTranspose(const Array<const Vector *> &X,
                           Array<Vector *> &Y) const;

   // Set various solver options. Refer to SuperLU_DIST documentation for
   // details.
   void SetPrintStatistics(bool print_stat);
   void SetEquilibriate(bool equil);
   void SetColumnPermutation(superlu::ColPerm col_perm);
   void SetRowPermutation(superlu::RowPerm row_perm);
   void SetIterativeRefine(superlu::IterRefine iter_ref);
   void SetReplaceTinyPivot(bool rtp);
   void SetNumLookAheads(int num_lookaheads);
   void SetLookAheadElimTree(bool etree);
   void SetSymmetricPattern(bool sym);
   void SetParSymbFact(bool par);
   void SetFact(superlu::Fact fact);

   // Processor grid for SuperLU_DIST.
   const int nprow_, npcol_, npdep_;

private:
   // Initialize the solver.
   void Init(MPI_Comm comm);

   // Handle error message from call to SuperLU solver.
   void HandleError(int info) const;

protected:
   const SuperLURowLocMatrix *APtr_;
   mutable Vector             sol_;
   mutable int                nrhs_;

   /** The actual types of the following pointers are hidden to avoid exposing
       the SuperLU header files to the entire library. Their types are given in
       the trailing comments. The reason that this is necessary is that SuperLU
       defines these structs differently for use with its real and complex
       solvers. If we want to add support for SuperLU's complex solvers one day
       we will need to hide these types to avoid name conflicts. */
   void *optionsPtr_;          // superlu_options_t *
   void *ScalePermstructPtr_;  //  ScalePermsruct_t *
   void *LUstructPtr_;         //        LUstruct_t *
   void *SOLVEstructPtr_;      //     SOLVEstruct_t *
   void *gridPtr_;             //        gridinfo_t * or gridinfo3d_t *
};

} // namespace mfem

#endif // MFEM_USE_MPI
#endif // MFEM_USE_SUPERLU
#endif // MFEM_SUPERLU
