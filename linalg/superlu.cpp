// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
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

#ifdef MFEM_USE_SUPERLU
#ifdef MFEM_USE_MPI

#include "superlu.hpp"

// SuperLU header
#include "superlu_ddefs.h"

#if XSDK_INDEX_SIZE == 64 && !(defined(HYPRE_BIGINT) || defined(HYPRE_MIXEDINT))
#error "Mismatch between HYPRE (32bit) and SuperLU (64bit) integer types"
#endif
#if XSDK_INDEX_SIZE == 32 && (defined(HYPRE_BIGINT) || defined(HYPRE_MIXEDINT))
#error "Mismatch between HYPRE (64bit) and SuperLU (32bit) integer types"
#endif

#if SUPERLU_DIST_MAJOR_VERSION > 6 || \
   (SUPERLU_DIST_MAJOR_VERSION == 6 && SUPERLU_DIST_MINOR_VERSION >= 3)
#define ScalePermstruct_t dScalePermstruct_t
#define LUstruct_t dLUstruct_t
#define SOLVEstruct_t dSOLVEstruct_t
#define ZeroLblocks dZeroLblocks
#define ZeroUblocks dZeroUblocks
#define Destroy_LU dDestroy_LU
#define SolveFinalize dSolveFinalize
#define ScalePermstructInit dScalePermstructInit
#define ScalePermstructFree dScalePermstructFree
#define LUstructFree dLUstructFree
#define LUstructInit dLUstructInit
#endif

#if SUPERLU_DIST_MAJOR_VERSION > 7 || \
   (SUPERLU_DIST_MAJOR_VERSION == 7 && SUPERLU_DIST_MINOR_VERSION >= 2)
#define DeAllocLlu_3d dDeAllocLlu_3d
#define DeAllocGlu_3d dDeAllocGlu_3d
#define Destroy_A3d_gathered_on_2d dDestroy_A3d_gathered_on_2d
#endif

unsigned int sqrti(unsigned int a)
{
   unsigned int rem     = 0;
   unsigned int root    = 0;
   unsigned short len   = sizeof(int); len <<= 2;
   unsigned short shift = (unsigned short)((len << 1) - 2);

   for (int i = 0; i < len; i++)
   {
      root <<= 1;
      rem = ((rem << 2) + (a >> shift));
      a <<= 2;
      root ++;
      if (root <= rem)
      {
         rem -= root;
         root++;
      }
      else
      {
         root--;
      }
   }
   return (root >> 1);
}

void GetSquare(int np, int &nr, int &nc)
{
   nr = (int)sqrti((unsigned int)np);
   while (np % nr != 0 && nr > 0)
   {
      nr--;
   }
   nc = (int)(np / nr);
   MFEM_VERIFY(nr * nc == np, "Impossible processor partition!");
}

namespace mfem
{

SuperLURowLocMatrix::SuperLURowLocMatrix(MPI_Comm comm,
                                         int num_loc_rows,
                                         HYPRE_BigInt first_loc_row,
                                         HYPRE_BigInt glob_nrows,
                                         HYPRE_BigInt glob_ncols,
                                         int *I, HYPRE_BigInt *J,
                                         double *data)
   : comm_(comm)
{
   // Set mfem::Operator member data
   height = num_loc_rows;
   width  = num_loc_rows;

   // Allocate SuperLU's SuperMatrix struct
   rowLocPtr_     = new SuperMatrix;
   SuperMatrix *A = (SuperMatrix *)rowLocPtr_;
   A->Store       = NULL;

   int_t m       = glob_nrows;
   int_t n       = glob_ncols;
   int_t nnz_loc = I[num_loc_rows];
   int_t m_loc   = num_loc_rows;
   int_t fst_row = first_loc_row;

   double *nzval  = NULL;
   int_t  *colind = NULL;
   int_t  *rowptr = NULL;

   if (!(nzval = doubleMalloc_dist(nnz_loc)))
   {
      MFEM_ABORT("SuperLURowLocMatrix: Malloc failed for nzval!");
   }
   for (int_t i = 0; i < nnz_loc; i++)
   {
      nzval[i] = data[i];
   }

   if (!(colind = intMalloc_dist(nnz_loc)))
   {
      MFEM_ABORT("SuperLURowLocMatrix: Malloc failed for colind!")
   }
   for (int_t i = 0; i < nnz_loc; i++)
   {
      colind[i] = J[i];
   }

   if (!(rowptr = intMalloc_dist(m_loc+1)))
   {
      MFEM_ABORT("SuperLURowLocMatrix: Malloc failed for rowptr!")
   }
   for (int_t i = 0; i <= m_loc; i++)
   {
      rowptr[i] = I[i];
   }

   // Assign the matrix data to SuperLU's SuperMatrix structure
   dCreate_CompRowLoc_Matrix_dist(A, m, n, nnz_loc, m_loc, fst_row,
                                  nzval, colind, rowptr,
                                  SLU_NR_loc, SLU_D, SLU_GE);

   // Save global number of rows and columns of the matrix
   num_global_rows_ = m;
   num_global_cols_ = n;
}

SuperLURowLocMatrix::SuperLURowLocMatrix(const Operator &op)
{
   const HypreParMatrix *APtr = dynamic_cast<const HypreParMatrix *>(&op);
   MFEM_VERIFY(APtr, "Not a compatible matrix type");
   comm_ = APtr->GetComm();

   // Set mfem::Operator member data
   height = op.Height();
   width  = op.Width();

   // Allocate SuperLU's SuperMatrix struct
   rowLocPtr_     = new SuperMatrix;
   SuperMatrix *A = (SuperMatrix *)rowLocPtr_;
   A->Store       = NULL;

   // First cast the parameter to a hypre_ParCSRMatrix
   hypre_ParCSRMatrix *parcsr_op =
      (hypre_ParCSRMatrix *)const_cast<HypreParMatrix &>(*APtr);

   // Create the SuperMatrix A by taking the internal data from a
   // hypre_CSRMatrix
   hypre_CSRMatrix *csr_op = hypre_MergeDiagAndOffd(parcsr_op);
   HYPRE_Int       *Iptr   = csr_op->i;
#if MFEM_HYPRE_VERSION >= 21600
   HYPRE_BigInt    *Jptr   = csr_op->big_j;
#else
   HYPRE_Int       *Jptr   = csr_op->j;
#endif
   hypre_CSRMatrixSetDataOwner(csr_op, 0);

   int_t m       = parcsr_op->global_num_rows;
   int_t n       = parcsr_op->global_num_cols;
   int_t fst_row = parcsr_op->first_row_index;
   int_t nnz_loc = csr_op->num_nonzeros;
   int_t m_loc   = csr_op->num_rows;

   double *nzval  = csr_op->data;
   int_t  *colind = Jptr;
   int_t  *rowptr = NULL;

   // The "i" array cannot be stolen from the hypre_CSRMatrix so we'll copy it
   if (!(rowptr = intMalloc_dist(m_loc+1)))
   {
      MFEM_ABORT("SuperLURowLocMatrix: Malloc failed for rowptr!")
   }
   for (int i = 0; i <= m_loc; i++)
   {
      rowptr[i] = (int_t)Iptr[i];  // Promotion for HYPRE_MIXEDINT
   }

   // Assign he matrix data to SuperLU's SuperMatrix structure
   dCreate_CompRowLoc_Matrix_dist(A, m, n, nnz_loc, m_loc, fst_row,
                                  nzval, colind, rowptr,
                                  SLU_NR_loc, SLU_D, SLU_GE);

   // Everything has been copied or abducted so delete the structure
   hypre_CSRMatrixDestroy(csr_op);

   // Save global number of rows and columns of the matrix
   num_global_rows_ = m;
   num_global_cols_ = n;
}

SuperLURowLocMatrix::~SuperLURowLocMatrix()
{
   SuperMatrix *A = (SuperMatrix *)rowLocPtr_;
   Destroy_CompRowLoc_Matrix_dist(A);
   delete A;
}

SuperLUSolver::SuperLUSolver(MPI_Comm comm, int npdep)
   : APtr_(NULL),
     npdep_(npdep),
     LUStructInitialized_(false),
     firstSolveWithThisA_(true)
{
   Init(comm);
}

SuperLUSolver::SuperLUSolver(SuperLURowLocMatrix &A, int npdep)
   : APtr_(&A),
     npdep_(npdep),
     LUStructInitialized_(false),
     firstSolveWithThisA_(true)
{
   Init(A.GetComm());
   SetOperator(A);
}

SuperLUSolver::~SuperLUSolver()
{
   superlu_dist_options_t *options = (superlu_dist_options_t *)optionsPtr_;

   ScalePermstruct_t *ScalePermstruct = (ScalePermstruct_t *)ScalePermstructPtr_;
   LUstruct_t        *LUstruct        = (LUstruct_t *)LUstructPtr_;
   SOLVEstruct_t     *SOLVEstruct     = (SOLVEstruct_t *)SOLVEstructPtr_;

#if SUPERLU_DIST_MAJOR_VERSION > 7 || \
   (SUPERLU_DIST_MAJOR_VERSION == 7 && SUPERLU_DIST_MINOR_VERSION >= 2)
   if (npdep_ > 1)
   {
      gridinfo3d_t *grid3d = (gridinfo3d_t *)gridPtr_;

      if (LUStructInitialized_)
      {
         if (grid3d->zscp.Iam == 0)
         {
            // Process layer 0
            Destroy_LU(APtr_->GetGlobalNumColumns(), &(grid3d->grid2d),
                       LUstruct);
            if (options->SolveInitialized == YES)
            {
               SolveFinalize(options, SOLVEstruct);
            }
         }
         else
         {
            // Process layers not equal 0
            DeAllocLlu_3d(APtr_->GetGlobalNumColumns(), LUstruct, grid3d);
            DeAllocGlu_3d(LUstruct);
         }
         Destroy_A3d_gathered_on_2d(SOLVEstruct, grid3d);
         ScalePermstructFree(ScalePermstruct);
         LUstructFree(LUstruct);
      }

      superlu_gridexit3d(grid3d);
      delete grid3d;
   }
   else
#endif
   {
      gridinfo_t *grid = (gridinfo_t *)gridPtr_;

      if (LUStructInitialized_)
      {
         Destroy_LU(APtr_->GetGlobalNumColumns(), grid, LUstruct);
         if (options->SolveInitialized == YES)
         {
            SolveFinalize(options, SOLVEstruct);
         }
         ScalePermstructFree(ScalePermstruct);
         LUstructFree(LUstruct);
      }

      superlu_gridexit(grid);
      delete grid;
   }

   delete options;
   delete ScalePermstruct;
   delete LUstruct;
   delete SOLVEstruct;
}

void SuperLUSolver::Init(MPI_Comm comm)
{
   optionsPtr_         = new superlu_dist_options_t;
   ScalePermstructPtr_ = new ScalePermstruct_t;
   LUstructPtr_        = new LUstruct_t;
   SOLVEstructPtr_     = new SOLVEstruct_t;
#if SUPERLU_DIST_MAJOR_VERSION > 7 || \
   (SUPERLU_DIST_MAJOR_VERSION == 7 && SUPERLU_DIST_MINOR_VERSION >= 2)
   if (npdep_ > 1)
   {
      gridPtr_         = new gridinfo3d_t;
   }
   else
#endif
   {
      gridPtr_         = new gridinfo_t;
   }

   // Initialize process grid
   SetupGrid(comm);

   // Set default options:
   //    options.Fact = DOFACT;
   //    options.Equil = YES;
   //    options.ColPerm = METIS_AT_PLUS_A;
   //    options.RowPerm = LargeDiag_MC64;
   //    options.ReplaceTinyPivot = NO;
   //    options.Trans = NOTRANS;
   //    options.IterRefine = SLU_DOUBLE;
   //    options.SolveInitialized = NO;
   //    options.RefineInitialized = NO;
   //    options.PrintStat = YES;
   superlu_dist_options_t *options = (superlu_dist_options_t *)optionsPtr_;
   set_default_options_dist(options);
}

void SuperLUSolver::SetupGrid(MPI_Comm comm)
{
   int nprocs;
   MPI_Comm_size(comm, &nprocs);

#if SUPERLU_DIST_MAJOR_VERSION > 7 || \
   (SUPERLU_DIST_MAJOR_VERSION == 7 && SUPERLU_DIST_MINOR_VERSION >= 2)
   if (npdep_ > 1)
   {
      gridinfo3d_t *grid3d = (gridinfo3d_t *)gridPtr_;

      int mask = npdep_ - 1;
      MFEM_VERIFY(!(npdep_ & mask),
                  "SuperLUSolver::SetupGrid: 3D partition depth must be a power "
                  "of two!");
      MFEM_VERIFY(nprocs % npdep_ == 0,
                  "SuperLUSolver::SetupGrid: Number of processors must be "
                  "evenly divisible by 3D partition depth!");
      GetSquare(nprocs / npdep_, nprow_, npcol_);

      superlu_gridinit3d(comm, nprow_, npcol_, npdep_, grid3d);
   }
   else
#endif
   {
      gridinfo_t *grid = (gridinfo_t *)gridPtr_;

      MFEM_VERIFY(npdep_ == 1,
                  "SuperLUSolver::SetupGrid: 3D partitioning is only available for "
                  "version >= 7.2.0!");
      GetSquare(nprocs, nprow_, npcol_);

      superlu_gridinit(comm, nprow_, npcol_, grid);
   }
}

void SuperLUSolver::SetPrintStatistics(bool print_stat)
{
   superlu_dist_options_t *options = (superlu_dist_options_t *)optionsPtr_;
   yes_no_t opt = print_stat ? YES : NO;
   options->PrintStat = opt;
}

void SuperLUSolver::SetEquilibriate(bool equil)
{
   superlu_dist_options_t *options = (superlu_dist_options_t *)optionsPtr_;
   yes_no_t opt = equil ? YES : NO;
   options->Equil = opt;
}

void SuperLUSolver::SetColumnPermutation(superlu::ColPerm col_perm)
{
   superlu_dist_options_t *options = (superlu_dist_options_t *)optionsPtr_;
   colperm_t opt = (colperm_t)col_perm;
   if (opt == MY_PERMC)
   {
      MFEM_ABORT("SuperLUSolver::SetColumnPermutation does not yet support "
                 "MY_PERMC!");
   }
   options->ColPerm = opt;
}

void SuperLUSolver::SetRowPermutation(superlu::RowPerm row_perm)
{
   superlu_dist_options_t *options = (superlu_dist_options_t *)optionsPtr_;
   rowperm_t opt = (rowperm_t)row_perm;
   if (opt == MY_PERMR)
   {
      MFEM_ABORT("SuperLUSolver::SetRowPermutation does not yet support "
                 "MY_PERMR!");
   }
   options->RowPerm = opt;
}

void SuperLUSolver::SetIterativeRefine(superlu::IterRefine iter_ref)
{
   superlu_dist_options_t *options = (superlu_dist_options_t *)optionsPtr_;
   IterRefine_t opt = (IterRefine_t)iter_ref;
   options->IterRefine = opt;
}

void SuperLUSolver::SetReplaceTinyPivot(bool rtp)
{
   superlu_dist_options_t *options = (superlu_dist_options_t *)optionsPtr_;
   yes_no_t opt = rtp ? YES : NO;
   options->ReplaceTinyPivot = opt;
}

void SuperLUSolver::SetNumLookAheads(int num_lookaheads)
{
   superlu_dist_options_t *options = (superlu_dist_options_t *)optionsPtr_;
   options->num_lookaheads = num_lookaheads;
}

void SuperLUSolver::SetLookAheadElimTree(bool etree)
{
   superlu_dist_options_t *options = (superlu_dist_options_t *)optionsPtr_;
   yes_no_t opt = etree ? YES : NO;
   options->lookahead_etree = opt;
}

void SuperLUSolver::SetSymmetricPattern(bool sym)
{
   superlu_dist_options_t *options = (superlu_dist_options_t *)optionsPtr_;
   yes_no_t opt = sym ? YES : NO;
   options->SymPattern = opt;
}

void SuperLUSolver::SetFact(superlu::Fact fact)
{
   superlu_dist_options_t *options = (superlu_dist_options_t *)optionsPtr_;
   fact_t opt = (fact_t)fact;
   options->Fact = opt;
}

void SuperLUSolver::SetOperator(const Operator &op)
{
   superlu_dist_options_t *options = (superlu_dist_options_t *)optionsPtr_;

   ScalePermstruct_t *ScalePermstruct = (ScalePermstruct_t *)ScalePermstructPtr_;
   LUstruct_t        *LUstruct        = (LUstruct_t *)LUstructPtr_;

   // Verify that we have a compatible operator
   APtr_ = dynamic_cast<const SuperLURowLocMatrix *>(&op);
   MFEM_VERIFY(APtr_, "SuperLUSolver::SetOperator: Not a SuperLURowLocMatrix!");

   // Set mfem::Operator member data
   MFEM_VERIFY(!LUStructInitialized_ ||
               (height == op.Height() && width == op.Width()),
               "SuperLUSolver::SetOperator: Inconsistent new matrix size!");
   height = op.Height();
   width  = op.Width();

   if (!LUStructInitialized_)
   {
      // Initialize ScalePermstruct and LUstruct once for all operators (must
      // have same dimensions)
      ScalePermstructInit(APtr_->GetGlobalNumRows(),
                          APtr_->GetGlobalNumColumns(), ScalePermstruct);
      LUstructInit(APtr_->GetGlobalNumColumns(), LUstruct);
      LUStructInitialized_ = true;
   }
   else
   {
      if (firstSolveWithThisA_)
      {
         // Previous matrix was never factored, so no cleanup to do
         MFEM_VERIFY(options->Fact == DOFACT,
                     "SuperLUSolver::SetOperator: Unexpected value for "
                     "options->Fact!");
      }
      else
      {
         // A previous matrix has already been set and factored
         switch (options->Fact)
         {
            case SamePattern_SameRowPerm:
               {
                  // Just zero the LU factors
#if SUPERLU_DIST_MAJOR_VERSION > 7 || \
   (SUPERLU_DIST_MAJOR_VERSION == 7 && SUPERLU_DIST_MINOR_VERSION >= 2)
                  if (npdep_ > 1)
                  {
                     gridinfo3d_t *grid3d = (gridinfo3d_t *)gridPtr_;

                     if (grid3d->zscp.Iam == 0)
                     {
                        ZeroLblocks(grid3d->iam, APtr_->GetGlobalNumColumns(),
                                    &(grid3d->grid2d), LUstruct);
                        ZeroUblocks(grid3d->iam, APtr_->GetGlobalNumColumns(),
                                    &(grid3d->grid2d), LUstruct);
                     }
                  }
                  else
#endif
                  {
                     gridinfo_t *grid = (gridinfo_t *)gridPtr_;

                     ZeroLblocks(grid->iam, APtr_->GetGlobalNumColumns(),
                                 grid, LUstruct);
                     ZeroUblocks(grid->iam, APtr_->GetGlobalNumColumns(),
                                 grid, LUstruct);
                  }
               }
               break;
            case SamePattern:
            case DOFACT:
            case FACTORED:
               {
                  // Delete factors from the prior factorization
#if SUPERLU_DIST_MAJOR_VERSION > 7 || \
   (SUPERLU_DIST_MAJOR_VERSION == 7 && SUPERLU_DIST_MINOR_VERSION >= 2)
                  if (npdep_ > 1)
                  {
                     gridinfo3d_t *grid3d = (gridinfo3d_t *)gridPtr_;

                     if (grid3d->zscp.Iam == 0)
                     {
                        Destroy_LU(APtr_->GetGlobalNumColumns(),
                                   &(grid3d->grid2d), LUstruct);
                     }
                     else
                     {
                        DeAllocLlu_3d(APtr_->GetGlobalNumColumns(), LUstruct,
                                      grid3d);
                        DeAllocGlu_3d(LUstruct);
                     }
                  }
                  else
#endif
                  {
                     gridinfo_t *grid = (gridinfo_t *)gridPtr_;

                     Destroy_LU(APtr_->GetGlobalNumColumns(), grid, LUstruct);
                  }
               }
               break;
            default:
               MFEM_ABORT("SuperLUSolver::SetOperator: Unexpected value for "
                          "options->Fact!");
               break;
         }
         if (options->Fact == FACTORED) { options->Fact = DOFACT; }
      }
   }
   firstSolveWithThisA_ = true;
}

void SuperLUSolver::Mult(const Vector &x, Vector &y) const
{
   MFEM_ASSERT(APtr_ != NULL,
               "SuperLU Error: The operator must be set before"
               " the system can be solved.");
   SuperMatrix            *A       = (SuperMatrix *)APtr_->InternalData();
   superlu_dist_options_t *options = (superlu_dist_options_t *)optionsPtr_;

   ScalePermstruct_t *ScalePermstruct = (ScalePermstruct_t *)ScalePermstructPtr_;
   LUstruct_t        *LUstruct        = (LUstruct_t *)LUstructPtr_;
   SOLVEstruct_t     *SOLVEstruct     = (SOLVEstruct_t *)SOLVEstructPtr_;

   gridinfo_t        *grid;
#if SUPERLU_DIST_MAJOR_VERSION > 7 || \
   (SUPERLU_DIST_MAJOR_VERSION == 7 && SUPERLU_DIST_MINOR_VERSION >= 2)
   gridinfo3d_t      *grid3d = NULL;
   if (npdep_ > 1)
   {
      grid3d = (gridinfo3d_t *)gridPtr_;
      grid = NULL;
   }
   else
#endif
   {
      grid = (gridinfo_t *)gridPtr_;
   }

   // SuperLU overwrites x with y, so copy x to y and pass that to the solve
   // routine.
   y = x;
   double *yPtr = y.HostReadWrite();
   int locSize = y.Size();
   int nrhs = 1;

   // Initialize the statistics variables
   SuperLUStat_t stat;
   PStatInit(&stat);

   double *berr;
   if (!(berr = doubleMalloc_dist(nrhs)))
   {
      MFEM_ABORT("SuperLUSolver::Mult: Malloc failed for berr!");
   }

   // Solve the system
   int info = -1;
#if SUPERLU_DIST_MAJOR_VERSION > 7 || \
   (SUPERLU_DIST_MAJOR_VERSION == 7 && SUPERLU_DIST_MINOR_VERSION >= 2)
   if (npdep_ > 1)
   {
      pdgssvx3d(options, A, ScalePermstruct, yPtr, locSize, nrhs, grid3d,
                LUstruct, SOLVEstruct, berr, &stat, &info);
   }
   else
#endif
   {
      pdgssvx(options, A, ScalePermstruct, yPtr, locSize, nrhs, grid,
              LUstruct, SOLVEstruct, berr, &stat, &info);
   }

   if (info != 0)
   {
      if (info < 0)
      {
         switch (-info)
         {
            case 1:
               MFEM_ABORT("SuperLUSolver: SuperLU options are invalid!");
               break;
            case 2:
               MFEM_ABORT("SuperLUSolver: Matrix A (in Ax=b) is invalid!");
               break;
            case 5:
               MFEM_ABORT("SuperLUSolver: Vector b dimension (in Ax=b) is "
                          "invalid!");
               break;
            case 6:
               MFEM_ABORT("SuperLUSolver: Number of right-hand sides is "
                          "invalid!");
               break;
            default:
               MFEM_ABORT("SuperLUSolver: Parameter with index "
                          << -info << "invalid (1-indexed)!");
               break;
         }
      }
      else if (info <= A->ncol)
      {
         MFEM_ABORT("SuperLUSolver: Found a singular matrix, U("
                    << info << "," << info << ") is exactly zero!");
      }
      else if (info > A->ncol)
      {
         MFEM_ABORT("SuperLUSolver: Memory allocation error with "
                    << info - A->ncol << " bytes already allocated!");
      }
      else
      {
         MFEM_ABORT("Unknown SuperLU error: info = " << info << "!");
      }
   }

   SUPERLU_FREE(berr);
   PStatFree(&stat);

   // Reuse factorization for future solves
   options->Fact = FACTORED;
   firstSolveWithThisA_ = false;
}

void SuperLUSolver::MultTranspose(const Vector & x, Vector & y) const
{
   // Set flag for transpose solve
   superlu_dist_options_t *options = (superlu_dist_options_t *)optionsPtr_;
   options->Trans = TRANS;
   Mult(x, y);

   // Reset the flag
   options->Trans = NOTRANS;
}

} // namespace mfem

#endif // MFEM_USE_MPI
#endif // MFEM_USE_SUPERLU
