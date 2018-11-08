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

// Author: Stefano Zampini <stefano.zampini@gmail.com>

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI
#ifdef MFEM_USE_PETSC

#include "linalg.hpp"
#include "../fem/fem.hpp"
#if defined(PETSC_HAVE_HYPRE)
#include "petscmathypre.h"
#endif

#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
// Note: there are additional #include statements below.

// Error handling
// Prints PETSc's stacktrace and then calls MFEM_ABORT
// We cannot use PETSc's CHKERRQ since it returns a PetscErrorCode
#define PCHKERRQ(obj,err) do {                                                   \
     if ((err))                                                                  \
     {                                                                           \
        PetscError(PetscObjectComm((PetscObject)(obj)),__LINE__,_MFEM_FUNC_NAME, \
                   __FILE__,(err),PETSC_ERROR_REPEAT,NULL);                      \
        MFEM_ABORT("Error in PETSc. See stacktrace above.");                     \
     }                                                                           \
  } while(0);
#define CCHKERRQ(comm,err) do {                                \
     if ((err))                                                \
     {                                                         \
        PetscError(comm,__LINE__,_MFEM_FUNC_NAME,              \
                   __FILE__,(err),PETSC_ERROR_REPEAT,NULL);    \
        MFEM_ABORT("Error in PETSc. See stacktrace above.");   \
     }                                                         \
  } while(0);

// Callback functions: these functions will be called by PETSc
static PetscErrorCode __mfem_ts_monitor(TS,PetscInt,PetscReal,Vec,void*);
static PetscErrorCode __mfem_ts_rhsfunction(TS,PetscReal,Vec,Vec,void*);
static PetscErrorCode __mfem_ts_rhsjacobian(TS,PetscReal,Vec,Mat,Mat,
                                            void*);
static PetscErrorCode __mfem_ts_ifunction(TS,PetscReal,Vec,Vec,Vec,void*);
static PetscErrorCode __mfem_ts_ijacobian(TS,PetscReal,Vec,Vec,
                                          PetscReal,Mat,
                                          Mat,void*);
static PetscErrorCode __mfem_snes_monitor(SNES,PetscInt,PetscReal,void*);
static PetscErrorCode __mfem_snes_jacobian(SNES,Vec,Mat,Mat,void*);
static PetscErrorCode __mfem_snes_function(SNES,Vec,Vec,void*);
static PetscErrorCode __mfem_ksp_monitor(KSP,PetscInt,PetscReal,void*);
static PetscErrorCode __mfem_pc_shell_apply(PC,Vec,Vec);
static PetscErrorCode __mfem_pc_shell_apply_transpose(PC,Vec,Vec);
static PetscErrorCode __mfem_pc_shell_setup(PC);
static PetscErrorCode __mfem_pc_shell_destroy(PC);
static PetscErrorCode __mfem_pc_shell_view(PC,PetscViewer);
static PetscErrorCode __mfem_mat_shell_apply(Mat,Vec,Vec);
static PetscErrorCode __mfem_mat_shell_apply_transpose(Mat,Vec,Vec);
static PetscErrorCode __mfem_mat_shell_destroy(Mat);
static PetscErrorCode __mfem_array_container_destroy(void*);
static PetscErrorCode __mfem_matarray_container_destroy(void*);

// auxiliary functions
static PetscErrorCode Convert_Array_IS(MPI_Comm,bool,const mfem::Array<int>*,
                                       PetscInt,IS*);
static PetscErrorCode Convert_Vmarks_IS(MPI_Comm,mfem::Array<Mat>&,
                                        const mfem::Array<int>*,PetscInt,IS*);
static PetscErrorCode MakeShellPC(PC,mfem::Solver&,bool);
static PetscErrorCode MakeShellPCWithFactory(PC,
                                             mfem::PetscPreconditionerFactory*);

// Equivalent functions are present in PETSc source code
// if PETSc has been compiled with hypre support
// We provide them here in case PETSC_HAVE_HYPRE is not defined
#if !defined(PETSC_HAVE_HYPRE)
static PetscErrorCode MatConvert_hypreParCSR_AIJ(hypre_ParCSRMatrix*,Mat*);
static PetscErrorCode MatConvert_hypreParCSR_IS(hypre_ParCSRMatrix*,Mat*);
#endif

// structs used by PETSc code
typedef struct
{
   mfem::Operator *op;
} __mfem_mat_shell_ctx;

typedef struct
{
   mfem::Solver                     *op;
   mfem::PetscPreconditionerFactory *factory;
   bool                             ownsop;
   unsigned long int                numprec;
} __mfem_pc_shell_ctx;

typedef struct
{
   mfem::Operator        *op;          // The nonlinear operator
   mfem::PetscBCHandler  *bchandler;   // Handling of essential bc
   mfem::Vector          *work;        // Work vector
   mfem::Operator::Type  jacType;      // OperatorType for the Jacobian
} __mfem_snes_ctx;

typedef struct
{
   mfem::TimeDependentOperator     *op;        // The time-dependent operator
   mfem::PetscBCHandler            *bchandler; // Handling of essential bc
   mfem::Vector                    *work;      // Work vector
   mfem::Operator::Type            jacType;    // OperatorType for the Jacobian
   enum mfem::PetscODESolver::Type type;
   PetscReal                       cached_shift;
   PetscObjectState                cached_ijacstate;
   PetscObjectState                cached_rhsjacstate;
} __mfem_ts_ctx;

// use global scope ierr to check PETSc errors inside mfem calls
static PetscErrorCode ierr;

using namespace std;

namespace mfem
{

// PetscParVector methods

void PetscParVector::_SetDataAndSize_()
{
   const PetscScalar *array;
   PetscInt           n;

   ierr = VecGetArrayRead(x,&array); PCHKERRQ(x,ierr);
   ierr = VecGetLocalSize(x,&n); PCHKERRQ(x,ierr);
   SetDataAndSize((PetscScalar*)array,n);
   ierr = VecRestoreArrayRead(x,&array); PCHKERRQ(x,ierr);
}

PetscInt PetscParVector::GlobalSize() const
{
   PetscInt N;
   ierr = VecGetSize(x,&N); PCHKERRQ(x,ierr);
   return N;
}

PetscParVector::PetscParVector(MPI_Comm comm, const Vector &_x) : Vector()
{
   ierr = VecCreate(comm,&x); CCHKERRQ(comm,ierr);
   ierr = VecSetSizes(x,_x.Size(),PETSC_DECIDE); PCHKERRQ(x,ierr);
   ierr = VecSetType(x,VECSTANDARD); PCHKERRQ(x,ierr);
   _SetDataAndSize_();
}

PetscParVector::PetscParVector(MPI_Comm comm, PetscInt glob_size,
                               PetscInt *col) : Vector()
{
   ierr = VecCreate(comm,&x); CCHKERRQ(comm,ierr);
   if (col)
   {
      PetscMPIInt myid;
      MPI_Comm_rank(comm, &myid);
      ierr = VecSetSizes(x,col[myid+1]-col[myid],PETSC_DECIDE); PCHKERRQ(x,ierr);
   }
   else
   {
      ierr = VecSetSizes(x,PETSC_DECIDE,glob_size); PCHKERRQ(x,ierr);
   }
   ierr = VecSetType(x,VECSTANDARD); PCHKERRQ(x,ierr);
   _SetDataAndSize_();
}

PetscParVector::~PetscParVector()
{
   MPI_Comm comm = PetscObjectComm((PetscObject)x);
   ierr = VecDestroy(&x); CCHKERRQ(comm,ierr);
}

PetscParVector::PetscParVector(MPI_Comm comm, PetscInt glob_size,
                               PetscScalar *_data, PetscInt *col) : Vector()
{
   MFEM_VERIFY(col,"Missing distribution");
   PetscMPIInt myid;
   MPI_Comm_rank(comm, &myid);
   ierr = VecCreateMPIWithArray(comm,1,col[myid+1]-col[myid],glob_size,_data,
                                &x); CCHKERRQ(comm,ierr)
   _SetDataAndSize_();
}

PetscParVector::PetscParVector(const PetscParVector &y) : Vector()
{
   ierr = VecDuplicate(y.x,&x); PCHKERRQ(x,ierr);
   _SetDataAndSize_();
}

PetscParVector::PetscParVector(MPI_Comm comm, const Operator &op,
                               bool transpose, bool allocate) : Vector()
{
   PetscInt loc = transpose ? op.Height() : op.Width();
   if (allocate)
   {
      ierr = VecCreate(comm,&x);
      CCHKERRQ(comm,ierr);
      ierr = VecSetSizes(x,loc,PETSC_DECIDE);
      PCHKERRQ(x,ierr);
      ierr = VecSetType(x,VECSTANDARD);
      PCHKERRQ(x,ierr);
      ierr = VecSetUp(x);
      PCHKERRQ(x,ierr);
   }
   else
   {
      ierr = VecCreateMPIWithArray(comm,1,loc,PETSC_DECIDE,NULL,
                                   &x); CCHKERRQ(comm,ierr);
   }
   _SetDataAndSize_();
}

PetscParVector::PetscParVector(const PetscParMatrix &A,
                               bool transpose, bool allocate) : Vector()
{
   Mat pA = const_cast<PetscParMatrix&>(A);
   if (!transpose)
   {
      ierr = MatCreateVecs(pA,&x,NULL); PCHKERRQ(pA,ierr);
   }
   else
   {
      ierr = MatCreateVecs(pA,NULL,&x); PCHKERRQ(pA,ierr);
   }
   if (!allocate)
   {
      ierr = VecReplaceArray(x,NULL); PCHKERRQ(x,ierr);
   }
   _SetDataAndSize_();
}

PetscParVector::PetscParVector(Vec y, bool ref) : Vector()
{
   if (ref)
   {
      ierr = PetscObjectReference((PetscObject)y); PCHKERRQ(y,ierr);
   }
   x = y;
   _SetDataAndSize_();
}

PetscParVector::PetscParVector(ParFiniteElementSpace *pfes) : Vector()
{

   HYPRE_Int* offsets = pfes->GetTrueDofOffsets();
   MPI_Comm  comm = pfes->GetComm();
   ierr = VecCreate(comm,&x); CCHKERRQ(comm,ierr);

   PetscMPIInt myid = 0;
   if (!HYPRE_AssumedPartitionCheck())
   {
      MPI_Comm_rank(comm,&myid);
   }
   ierr = VecSetSizes(x,offsets[myid+1]-offsets[myid],PETSC_DECIDE);
   PCHKERRQ(x,ierr);
   ierr = VecSetType(x,VECSTANDARD); PCHKERRQ(x,ierr);
   _SetDataAndSize_();
}

Vector * PetscParVector::GlobalVector() const
{
   VecScatter   scctx;
   Vec          vout;
   PetscScalar *array;
   PetscInt     size;

   ierr = VecScatterCreateToAll(x,&scctx,&vout); PCHKERRQ(x,ierr);
   ierr = VecScatterBegin(scctx,x,vout,INSERT_VALUES,SCATTER_FORWARD);
   PCHKERRQ(x,ierr);
   ierr = VecScatterEnd(scctx,x,vout,INSERT_VALUES,SCATTER_FORWARD);
   PCHKERRQ(x,ierr);
   ierr = VecScatterDestroy(&scctx); PCHKERRQ(x,ierr);
   ierr = VecGetArray(vout,&array); PCHKERRQ(x,ierr);
   ierr = VecGetLocalSize(vout,&size); PCHKERRQ(x,ierr);
   Array<PetscScalar> data(size);
   data.Assign(array);
   ierr = VecRestoreArray(vout,&array); PCHKERRQ(x,ierr);
   ierr = VecDestroy(&vout); PCHKERRQ(x,ierr);
   Vector *v = new Vector(data, internal::to_int(size));
   v->MakeDataOwner();
   data.LoseData();
   return v;
}

PetscParVector& PetscParVector::operator=(PetscScalar d)
{
   ierr = VecSet(x,d); PCHKERRQ(x,ierr);
   return *this;
}

PetscParVector& PetscParVector::operator=(const PetscParVector &y)
{
   ierr = VecCopy(y.x,x); PCHKERRQ(x,ierr);
   return *this;
}

void PetscParVector::PlaceArray(PetscScalar *temp_data)
{
   ierr = VecPlaceArray(x,temp_data); PCHKERRQ(x,ierr);
}

void PetscParVector::ResetArray()
{
   ierr = VecResetArray(x); PCHKERRQ(x,ierr);
}

void PetscParVector::Randomize(PetscInt seed)
{
   PetscRandom rctx;

   ierr = PetscRandomCreate(PetscObjectComm((PetscObject)x),&rctx);
   PCHKERRQ(x,ierr);
   ierr = PetscRandomSetSeed(rctx,(unsigned long)seed); PCHKERRQ(x,ierr);
   ierr = PetscRandomSeed(rctx); PCHKERRQ(x,ierr);
   ierr = VecSetRandom(x,rctx); PCHKERRQ(x,ierr);
   ierr = PetscRandomDestroy(&rctx); PCHKERRQ(x,ierr);
}

void PetscParVector::Print(const char *fname, bool binary) const
{
   if (fname)
   {
      PetscViewer view;

      if (binary)
      {
         ierr = PetscViewerBinaryOpen(PetscObjectComm((PetscObject)x),fname,
                                      FILE_MODE_WRITE,&view);
      }
      else
      {
         ierr = PetscViewerASCIIOpen(PetscObjectComm((PetscObject)x),fname,&view);
      }
      PCHKERRQ(x,ierr);
      ierr = VecView(x,view); PCHKERRQ(x,ierr);
      ierr = PetscViewerDestroy(&view); PCHKERRQ(x,ierr);
   }
   else
   {
      ierr = VecView(x,NULL); PCHKERRQ(x,ierr);
   }
}

// PetscParMatrix methods

PetscInt PetscParMatrix::GetNumRows() const
{
   PetscInt N;
   ierr = MatGetLocalSize(A,&N,NULL); PCHKERRQ(A,ierr);
   return N;
}

PetscInt PetscParMatrix::GetNumCols() const
{
   PetscInt N;
   ierr = MatGetLocalSize(A,NULL,&N); PCHKERRQ(A,ierr);
   return N;
}

PetscInt PetscParMatrix::M() const
{
   PetscInt N;
   ierr = MatGetSize(A,&N,NULL); PCHKERRQ(A,ierr);
   return N;
}

PetscInt PetscParMatrix::N() const
{
   PetscInt N;
   ierr = MatGetSize(A,NULL,&N); PCHKERRQ(A,ierr);
   return N;
}

PetscInt PetscParMatrix::NNZ() const
{
   MatInfo info;
   ierr = MatGetInfo(A,MAT_GLOBAL_SUM,&info); PCHKERRQ(A,ierr);
   return (PetscInt)info.nz_used;
}

void PetscParMatrix::Init()
{
   A = NULL;
   X = Y = NULL;
   height = width = 0;
}

PetscParMatrix::PetscParMatrix()
{
   Init();
}

PetscParMatrix::PetscParMatrix(const HypreParMatrix *ha, Operator::Type tid)
{
   Init();
   height = ha->Height();
   width  = ha->Width();
   ConvertOperator(ha->GetComm(),*ha,&A,tid);
}

PetscParMatrix::PetscParMatrix(MPI_Comm comm, const Operator *op,
                               Operator::Type tid)
{
   Init();
   height = op->Height();
   width  = op->Width();
   ConvertOperator(comm,*op,&A,tid);
}

PetscParMatrix::PetscParMatrix(MPI_Comm comm, PetscInt glob_size,
                               PetscInt *row_starts, SparseMatrix *diag,
                               Operator::Type tid)
{
   Init();
   BlockDiagonalConstructor(comm,row_starts,row_starts,diag,
                            tid==PETSC_MATAIJ,&A);
   // update base class
   height = GetNumRows();
   width  = GetNumCols();
}

PetscParMatrix::PetscParMatrix(MPI_Comm comm, PetscInt global_num_rows,
                               PetscInt global_num_cols, PetscInt *row_starts,
                               PetscInt *col_starts, SparseMatrix *diag,
                               Operator::Type tid)
{
   Init();
   BlockDiagonalConstructor(comm,row_starts,col_starts,diag,
                            tid==PETSC_MATAIJ,&A);
   // update base class
   height = GetNumRows();
   width  = GetNumCols();
}

PetscParMatrix& PetscParMatrix::operator=(const HypreParMatrix& B)
{
   if (A)
   {
      MPI_Comm comm = PetscObjectComm((PetscObject)A);
      ierr = MatDestroy(&A); CCHKERRQ(comm,ierr);
      if (X) { delete X; }
      if (Y) { delete Y; }
      X = Y = NULL;
   }
   height = B.Height();
   width  = B.Width();
#if defined(PETSC_HAVE_HYPRE)
   ierr = MatCreateFromParCSR(B,MATAIJ,PETSC_USE_POINTER,&A);
#else
   ierr = MatConvert_hypreParCSR_AIJ(B,&A); CCHKERRQ(B.GetComm(),ierr);
#endif
   return *this;
}

PetscParMatrix& PetscParMatrix::operator=(const PetscParMatrix& B)
{
   if (A)
   {
      MPI_Comm comm = PetscObjectComm((PetscObject)A);
      ierr = MatDestroy(&A); CCHKERRQ(comm,ierr);
      if (X) { delete X; }
      if (Y) { delete Y; }
      X = Y = NULL;
   }
   height = B.Height();
   width  = B.Width();
   ierr   = MatDuplicate(B,MAT_COPY_VALUES,&A); CCHKERRQ(B.GetComm(),ierr);
   return *this;
}

PetscParMatrix& PetscParMatrix::operator+=(const PetscParMatrix& B)
{
   if (!A)
   {
      ierr = MatDuplicate(B,MAT_COPY_VALUES,&A); CCHKERRQ(B.GetComm(),ierr);
   }
   else
   {
      MFEM_VERIFY(height == B.Height(),"Invalid number of local rows");
      MFEM_VERIFY(width  == B.Width(), "Invalid number of local columns");
      ierr = MatAXPY(A,1.0,B,DIFFERENT_NONZERO_PATTERN); CCHKERRQ(B.GetComm(),ierr);
   }
   return *this;
}

void PetscParMatrix::
BlockDiagonalConstructor(MPI_Comm comm,
                         PetscInt *row_starts, PetscInt *col_starts,
                         SparseMatrix *diag, bool assembled, Mat* Ad)
{
   Mat      A;
   PetscInt lrsize,lcsize,rstart,cstart;
   PetscMPIInt myid = 0,commsize;

   ierr = MPI_Comm_size(comm,&commsize); CCHKERRQ(comm,ierr);
   if (!HYPRE_AssumedPartitionCheck())
   {
      ierr = MPI_Comm_rank(comm,&myid); CCHKERRQ(comm,ierr);
   }
   lrsize = row_starts[myid+1]-row_starts[myid];
   rstart = row_starts[myid];
   lcsize = col_starts[myid+1]-col_starts[myid];
   cstart = col_starts[myid];

   if (!assembled)
   {
      IS is;
      ierr = ISCreateStride(comm,diag->Height(),rstart,1,&is); CCHKERRQ(comm,ierr);
      ISLocalToGlobalMapping rl2g,cl2g;
      ierr = ISLocalToGlobalMappingCreateIS(is,&rl2g); PCHKERRQ(is,ierr);
      ierr = ISDestroy(&is); CCHKERRQ(comm,ierr);
      if (row_starts != col_starts)
      {
         ierr = ISCreateStride(comm,diag->Width(),cstart,1,&is);
         CCHKERRQ(comm,ierr);
         ierr = ISLocalToGlobalMappingCreateIS(is,&cl2g); PCHKERRQ(is,ierr);
         ierr = ISDestroy(&is); CCHKERRQ(comm,ierr);
      }
      else
      {
         ierr = PetscObjectReference((PetscObject)rl2g); PCHKERRQ(rl2g,ierr);
         cl2g = rl2g;
      }

      // Create the PETSc object (MATIS format)
      ierr = MatCreate(comm,&A); CCHKERRQ(comm,ierr);
      ierr = MatSetSizes(A,lrsize,lcsize,PETSC_DECIDE,PETSC_DECIDE);
      PCHKERRQ(A,ierr);
      ierr = MatSetType(A,MATIS); PCHKERRQ(A,ierr);
      ierr = MatSetLocalToGlobalMapping(A,rl2g,cl2g); PCHKERRQ(A,ierr);
      ierr = ISLocalToGlobalMappingDestroy(&rl2g); PCHKERRQ(A,ierr)
      ierr = ISLocalToGlobalMappingDestroy(&cl2g); PCHKERRQ(A,ierr)

      // Copy SparseMatrix into PETSc SeqAIJ format
      Mat lA;
      ierr = MatISGetLocalMat(A,&lA); PCHKERRQ(A,ierr);
      if (sizeof(PetscInt) == sizeof(int))
      {
         ierr = MatSeqAIJSetPreallocationCSR(lA,diag->GetI(),diag->GetJ(),
                                             diag->GetData()); PCHKERRQ(lA,ierr);
      }
      else
      {
         MFEM_ABORT("64bit indices not yet supported");
      }
   }
   else
   {
      PetscScalar *da;
      PetscInt    *dii,*djj,*oii,
                  m = diag->Height()+1, nnz = diag->NumNonZeroElems();

      diag->SortColumnIndices();
      // if we can take ownership of the SparseMatrix arrays, we can avoid this
      // step
      ierr = PetscMalloc1(m,&dii); CCHKERRQ(PETSC_COMM_SELF,ierr);
      ierr = PetscMalloc1(nnz,&djj); CCHKERRQ(PETSC_COMM_SELF,ierr);
      ierr = PetscMalloc1(nnz,&da); CCHKERRQ(PETSC_COMM_SELF,ierr);
      if (sizeof(PetscInt) == sizeof(int))
      {
         ierr = PetscMemcpy(dii,diag->GetI(),m*sizeof(PetscInt));
         CCHKERRQ(PETSC_COMM_SELF,ierr);
         ierr = PetscMemcpy(djj,diag->GetJ(),nnz*sizeof(PetscInt));
         CCHKERRQ(PETSC_COMM_SELF,ierr);
         ierr = PetscMemcpy(da,diag->GetData(),nnz*sizeof(PetscScalar));
         CCHKERRQ(PETSC_COMM_SELF,ierr);
      }
      else
      {
         MFEM_ABORT("64bit indices not yet supported");
      }
      ierr = PetscCalloc1(m,&oii);
      CCHKERRQ(PETSC_COMM_SELF,ierr);
      if (commsize > 1)
      {
         ierr = MatCreateMPIAIJWithSplitArrays(comm,lrsize,lcsize,PETSC_DECIDE,
                                               PETSC_DECIDE,
                                               dii,djj,da,oii,NULL,NULL,&A);
         CCHKERRQ(comm,ierr);
      }
      else
      {
         ierr = MatCreateSeqAIJWithArrays(comm,lrsize,lcsize,dii,djj,da,&A);
         CCHKERRQ(comm,ierr);
      }

      void *ptrs[4] = {dii,djj,da,oii};
      const char *names[4] = {"_mfem_csr_dii",
                              "_mfem_csr_djj",
                              "_mfem_csr_da",
                              "_mfem_csr_oii",
                             };
      for (PetscInt i=0; i<4; i++)
      {
         PetscContainer c;

         ierr = PetscContainerCreate(comm,&c); CCHKERRQ(comm,ierr);
         ierr = PetscContainerSetPointer(c,ptrs[i]); CCHKERRQ(comm,ierr);
         ierr = PetscContainerSetUserDestroy(c,__mfem_array_container_destroy);
         CCHKERRQ(comm,ierr);
         ierr = PetscObjectCompose((PetscObject)A,names[i],(PetscObject)c);
         CCHKERRQ(comm,ierr);
         ierr = PetscContainerDestroy(&c); CCHKERRQ(comm,ierr);
      }
   }

   // Tell PETSc the matrix is ready to be used
   ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); PCHKERRQ(A,ierr);
   ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); PCHKERRQ(A,ierr);

   *Ad = A;
}

// TODO ADD THIS CONSTRUCTOR
//PetscParMatrix::PetscParMatrix(MPI_Comm comm, int nrows, PetscInt glob_nrows,
//                  PetscInt glob_ncols, int *I, PetscInt *J,
//                  double *data, PetscInt *rows, PetscInt *cols)
//{
//}

// TODO This should take a reference on op but how?
void PetscParMatrix::MakeWrapper(MPI_Comm comm, const Operator* op, Mat *A)
{
   __mfem_mat_shell_ctx *ctx = new __mfem_mat_shell_ctx;
   ierr = MatCreate(comm,A); CCHKERRQ(comm,ierr);
   ierr = MatSetSizes(*A,op->Height(),op->Width(),
                      PETSC_DECIDE,PETSC_DECIDE); PCHKERRQ(A,ierr);
   ierr = MatSetType(*A,MATSHELL); PCHKERRQ(A,ierr);
   ierr = MatShellSetContext(*A,(void *)ctx); PCHKERRQ(A,ierr);
   ierr = MatShellSetOperation(*A,MATOP_MULT,
                               (void (*)())__mfem_mat_shell_apply);
   PCHKERRQ(A,ierr);
   ierr = MatShellSetOperation(*A,MATOP_MULT_TRANSPOSE,
                               (void (*)())__mfem_mat_shell_apply_transpose);
   PCHKERRQ(A,ierr);
   ierr = MatShellSetOperation(*A,MATOP_DESTROY,
                               (void (*)())__mfem_mat_shell_destroy);
   PCHKERRQ(A,ierr);
   ierr = MatSetUp(*A); PCHKERRQ(*A,ierr);
   ctx->op = const_cast<Operator *>(op);
}

void PetscParMatrix::ConvertOperator(MPI_Comm comm, const Operator &op, Mat* A,
                                     Operator::Type tid)
{
   PetscParMatrix   *pA = const_cast<PetscParMatrix *>
                          (dynamic_cast<const PetscParMatrix *>(&op));
   HypreParMatrix   *pH = const_cast<HypreParMatrix *>
                          (dynamic_cast<const HypreParMatrix *>(&op));
   BlockOperator    *pB = const_cast<BlockOperator *>
                          (dynamic_cast<const BlockOperator *>(&op));
   IdentityOperator *pI = const_cast<IdentityOperator *>
                          (dynamic_cast<const IdentityOperator *>(&op));

   if (pA)
   {
      Mat       At = NULL;
      PetscBool ismatis,istrans;

      ierr = PetscObjectTypeCompare((PetscObject)(pA->A),MATTRANSPOSEMAT,&istrans);
      CCHKERRQ(pA->GetComm(),ierr);
      if (!istrans)
      {
         if (tid == pA->GetType()) // use same object and return
         {
            ierr = PetscObjectReference((PetscObject)(pA->A));
            CCHKERRQ(pA->GetComm(),ierr);
            *A = pA->A;
            return;
         }
         ierr = PetscObjectTypeCompare((PetscObject)(pA->A),MATIS,&ismatis);
         CCHKERRQ(pA->GetComm(),ierr);
      }
      else
      {
         ierr = MatTransposeGetMat(pA->A,&At); CCHKERRQ(pA->GetComm(),ierr);
         ierr = PetscObjectTypeCompare((PetscObject)(At),MATIS,&ismatis);
         CCHKERRQ(pA->GetComm(),ierr);
      }

      // Try to convert
      if (tid == PETSC_MATAIJ)
      {
         if (ismatis)
         {
            if (istrans)
            {
               Mat B;

               ierr = MatISGetMPIXAIJ(At,MAT_INITIAL_MATRIX,&B); PCHKERRQ(pA->A,ierr);
               ierr = MatCreateTranspose(B,A); PCHKERRQ(pA->A,ierr);
               ierr = MatDestroy(&B); PCHKERRQ(pA->A,ierr);
            }
            else
            {
               ierr = MatISGetMPIXAIJ(pA->A,MAT_INITIAL_MATRIX,A);
               PCHKERRQ(pA->A,ierr);
            }
         }
         else
         {
            PetscMPIInt size;
            ierr = MPI_Comm_size(comm,&size); CCHKERRQ(comm,ierr);

            // call MatConvert and see if a converter is available
            if (istrans)
            {
               Mat B;
               ierr = MatConvert(At,size > 1 ? MATMPIAIJ : MATSEQAIJ,MAT_INITIAL_MATRIX,&B);
               PCHKERRQ(pA->A,ierr);
               ierr = MatCreateTranspose(B,A); PCHKERRQ(pA->A,ierr);
               ierr = MatDestroy(&B); PCHKERRQ(pA->A,ierr);
            }
            else
            {
               ierr = MatConvert(pA->A, size > 1 ? MATMPIAIJ : MATSEQAIJ,MAT_INITIAL_MATRIX,A);
               PCHKERRQ(pA->A,ierr);
            }
         }
      }
      else if (tid == PETSC_MATIS)
      {
         if (istrans)
         {
            Mat B;
            ierr = MatConvert(At,MATIS,MAT_INITIAL_MATRIX,&B); PCHKERRQ(pA->A,ierr);
            ierr = MatCreateTranspose(B,A); PCHKERRQ(pA->A,ierr);
            ierr = MatDestroy(&B); PCHKERRQ(pA->A,ierr);
         }
         else
         {
            ierr = MatConvert(pA->A,MATIS,MAT_INITIAL_MATRIX,A); PCHKERRQ(pA->A,ierr);
         }
      }
#if defined(PETSC_HAVE_HYPRE)
      else if (tid == PETSC_MATHYPRE)
      {
         if (istrans)
         {
            Mat B;
            ierr = MatConvert(At,MATHYPRE,MAT_INITIAL_MATRIX,&B); PCHKERRQ(pA->A,ierr);
            ierr = MatCreateTranspose(B,A); PCHKERRQ(pA->A,ierr);
            ierr = MatDestroy(&B); PCHKERRQ(pA->A,ierr);
         }
         else
         {
            ierr = MatConvert(pA->A,MATHYPRE,MAT_INITIAL_MATRIX,A); PCHKERRQ(pA->A,ierr);
         }
      }
#endif
      else if (tid == PETSC_MATSHELL)
      {
         MakeWrapper(comm,&op,A);
      }
      else
      {
         MFEM_ABORT("Unsupported operator type conversion " << tid)
      }
   }
   else if (pH)
   {
      if (tid == PETSC_MATAIJ)
      {
#if defined(PETSC_HAVE_HYPRE)
         ierr = MatCreateFromParCSR(const_cast<HypreParMatrix&>(*pH),MATAIJ,
                                    PETSC_USE_POINTER,A);
#else
         ierr = MatConvert_hypreParCSR_AIJ(const_cast<HypreParMatrix&>(*pH),A);
#endif
         CCHKERRQ(pH->GetComm(),ierr);
      }
      else if (tid == PETSC_MATIS)
      {
#if defined(PETSC_HAVE_HYPRE)
         ierr = MatCreateFromParCSR(const_cast<HypreParMatrix&>(*pH),MATIS,
                                    PETSC_USE_POINTER,A);
#else
         ierr = MatConvert_hypreParCSR_IS(const_cast<HypreParMatrix&>(*pH),A);
#endif
         CCHKERRQ(pH->GetComm(),ierr);
      }
#if defined(PETSC_HAVE_HYPRE)
      else if (tid == PETSC_MATHYPRE)
      {
         ierr = MatCreateFromParCSR(const_cast<HypreParMatrix&>(*pH),MATHYPRE,
                                    PETSC_USE_POINTER,A);
         CCHKERRQ(pH->GetComm(),ierr);
      }
#endif
      else if (tid == PETSC_MATSHELL)
      {
         MakeWrapper(comm,&op,A);
      }
      else
      {
         MFEM_ABORT("Conversion from HypreParCSR to operator type = " << tid <<
                    " is not implemented");
      }
   }
   else if (pB)
   {
      Mat      *mats,*matsl2l = NULL;
      PetscInt i,j,nr,nc;

      nr = pB->NumRowBlocks();
      nc = pB->NumColBlocks();
      ierr = PetscCalloc1(nr*nc,&mats); CCHKERRQ(PETSC_COMM_SELF,ierr);
      if (tid == PETSC_MATIS)
      {
         ierr = PetscCalloc1(nr,&matsl2l); CCHKERRQ(PETSC_COMM_SELF,ierr);
      }
      for (i=0; i<nr; i++)
      {
         PetscBool needl2l = PETSC_TRUE;

         for (j=0; j<nc; j++)
         {
            if (!pB->IsZeroBlock(i,j))
            {
               ConvertOperator(comm,pB->GetBlock(i,j),&mats[i*nc+j],tid);
               if (tid == PETSC_MATIS && needl2l)
               {
                  PetscContainer c;
                  ierr = PetscObjectQuery((PetscObject)mats[i*nc+j],"_MatIS_PtAP_l2l",
                                          (PetscObject*)&c);
                  PCHKERRQ(mats[i*nc+j],ierr);
                  // special case for block operators: the local Vdofs should be
                  // ordered as:
                  // [f1_1,...f1_N1,f2_1,...,f2_N2,...,fm_1,...,fm_Nm]
                  // with m fields, Ni the number of Vdofs for the i-th field
                  if (c)
                  {
                     Array<Mat> *l2l = NULL;
                     ierr = PetscContainerGetPointer(c,(void**)&l2l);
                     PCHKERRQ(c,ierr);
                     MFEM_VERIFY(l2l->Size() == 1,"Unexpected size "
                                 << l2l->Size() << " for block row " << i );
                     ierr = PetscObjectReference((PetscObject)(*l2l)[0]);
                     PCHKERRQ(c,ierr);
                     matsl2l[i] = (*l2l)[0];
                     needl2l = PETSC_FALSE;
                  }
               }
            }
         }
      }
      ierr = MatCreateNest(comm,nr,NULL,nc,NULL,mats,A); CCHKERRQ(comm,ierr);
      if (tid == PETSC_MATIS)
      {
         ierr = MatConvert(*A,MATIS,MAT_INPLACE_MATRIX,A); CCHKERRQ(comm,ierr);

         mfem::Array<Mat> *vmatsl2l = new mfem::Array<Mat>(nr);
         for (PetscInt i=0; i<nr; i++) { (*vmatsl2l)[i] = matsl2l[i]; }
         ierr = PetscFree(matsl2l); CCHKERRQ(PETSC_COMM_SELF,ierr);

         PetscContainer c;
         ierr = PetscContainerCreate(comm,&c); CCHKERRQ(comm,ierr);
         ierr = PetscContainerSetPointer(c,vmatsl2l); PCHKERRQ(c,ierr);
         ierr = PetscContainerSetUserDestroy(c,__mfem_matarray_container_destroy);
         PCHKERRQ(c,ierr);
         ierr = PetscObjectCompose((PetscObject)(*A),"_MatIS_PtAP_l2l",(PetscObject)c);
         PCHKERRQ((*A),ierr);
         ierr = PetscContainerDestroy(&c); CCHKERRQ(comm,ierr);
      }
      for (i=0; i<nr*nc; i++) { ierr = MatDestroy(&mats[i]); CCHKERRQ(comm,ierr); }
      ierr = PetscFree(mats); CCHKERRQ(PETSC_COMM_SELF,ierr);
   }
   else if (pI)
   {
      MFEM_VERIFY(tid == PETSC_MATAIJ,"Unsupported operation");
      PetscInt rst;

      ierr = MatCreate(comm,A); CCHKERRQ(comm,ierr);
      ierr = MatSetSizes(*A,pI->Height(),pI->Width(),PETSC_DECIDE,PETSC_DECIDE);
      PCHKERRQ(A,ierr);
      ierr = MatSetType(*A,MATAIJ); PCHKERRQ(*A,ierr);
      ierr = MatMPIAIJSetPreallocation(*A,1,NULL,0,NULL); PCHKERRQ(*A,ierr);
      ierr = MatSeqAIJSetPreallocation(*A,1,NULL); PCHKERRQ(*A,ierr);
      ierr = MatSetOption(*A,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE); PCHKERRQ(*A,ierr);
      ierr = MatGetOwnershipRange(*A,&rst,NULL); PCHKERRQ(*A,ierr);
      for (PetscInt i = rst; i < rst+pI->Height(); i++)
      {
         ierr = MatSetValue(*A,i,i,1.,INSERT_VALUES); PCHKERRQ(*A,ierr);
      }
      ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY); PCHKERRQ(*A,ierr);
      ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY); PCHKERRQ(*A,ierr);
   }
   else // fallback to general operator
   {
      MakeWrapper(comm,&op,A);
   }
}

void PetscParMatrix::Destroy()
{
   if (A != NULL)
   {
      MPI_Comm comm = MPI_COMM_NULL;
      ierr = PetscObjectGetComm((PetscObject)A,&comm); PCHKERRQ(A,ierr);
      ierr = MatDestroy(&A); CCHKERRQ(comm,ierr);
   }
   delete X;
   delete Y;
   X = Y = NULL;
}

PetscParMatrix::PetscParMatrix(Mat a, bool ref)
{
   if (ref)
   {
      ierr = PetscObjectReference((PetscObject)a); PCHKERRQ(a,ierr);
   }
   Init();
   A = a;
   height = GetNumRows();
   width = GetNumCols();
}

// Computes y = alpha * A  * x + beta * y
//       or y = alpha * A^T* x + beta * y
static void MatMultKernel(Mat A,PetscScalar a,Vec X,PetscScalar b,Vec Y,
                          bool transpose)
{
   PetscErrorCode (*f)(Mat,Vec,Vec);
   PetscErrorCode (*fadd)(Mat,Vec,Vec,Vec);
   if (transpose)
   {
      f = MatMultTranspose;
      fadd = MatMultTransposeAdd;
   }
   else
   {
      f = MatMult;
      fadd = MatMultAdd;
   }
   if (a != 0.)
   {
      if (b == 1.)
      {
         ierr = VecScale(X,a); PCHKERRQ(A,ierr);
         ierr = (*fadd)(A,X,Y,Y); PCHKERRQ(A,ierr);
         ierr = VecScale(X,1./a); PCHKERRQ(A,ierr);
      }
      else if (b != 0.)
      {
         ierr = VecScale(X,a); PCHKERRQ(A,ierr);
         ierr = VecScale(Y,b); PCHKERRQ(A,ierr);
         ierr = (*fadd)(A,X,Y,Y); PCHKERRQ(A,ierr);
         ierr = VecScale(X,1./a); PCHKERRQ(A,ierr);
      }
      else
      {
         ierr = (*f)(A,X,Y); PCHKERRQ(A,ierr);
         if (a != 1.)
         {
            ierr = VecScale(Y,a); PCHKERRQ(A,ierr);
         }
      }
   }
   else
   {
      if (b == 1.)
      {
         // do nothing
      }
      else if (b != 0.)
      {
         ierr = VecScale(Y,b); PCHKERRQ(A,ierr);
      }
      else
      {
         ierr = VecSet(Y,0.); PCHKERRQ(A,ierr);
      }
   }
}

void PetscParMatrix::MakeRef(const PetscParMatrix &master)
{
   ierr = PetscObjectReference((PetscObject)master.A); PCHKERRQ(master.A,ierr);
   Destroy();
   Init();
   A = master.A;
   height = master.height;
   width = master.width;
}

PetscParVector * PetscParMatrix::GetX() const
{
   if (!X)
   {
      MFEM_VERIFY(A,"Mat not present");
      X = new PetscParVector(*this,false); PCHKERRQ(A,ierr);
   }
   return X;
}

PetscParVector * PetscParMatrix::GetY() const
{
   if (!Y)
   {
      MFEM_VERIFY(A,"Mat not present");
      Y = new PetscParVector(*this,true); PCHKERRQ(A,ierr);
   }
   return Y;
}

PetscParMatrix * PetscParMatrix::Transpose(bool action)
{
   Mat B;
   if (action)
   {
      ierr = MatCreateTranspose(A,&B); PCHKERRQ(A,ierr);
   }
   else
   {
      ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&B); PCHKERRQ(A,ierr);
   }
   return new PetscParMatrix(B,false);
}

void PetscParMatrix::operator*=(double s)
{
   ierr = MatScale(A,s); PCHKERRQ(A,ierr);
}

void PetscParMatrix::Mult(double a, const Vector &x, double b, Vector &y) const
{
   MFEM_ASSERT(x.Size() == Width(), "invalid x.Size() = " << x.Size()
               << ", expected size = " << Width());
   MFEM_ASSERT(y.Size() == Height(), "invalid y.Size() = " << y.Size()
               << ", expected size = " << Height());

   PetscParVector *XX = GetX();
   PetscParVector *YY = GetY();
   XX->PlaceArray(x.GetData());
   YY->PlaceArray(y.GetData());
   MatMultKernel(A,a,XX->x,b,YY->x,false);
   XX->ResetArray();
   YY->ResetArray();
}

void PetscParMatrix::MultTranspose(double a, const Vector &x, double b,
                                   Vector &y) const
{
   MFEM_ASSERT(x.Size() == Height(), "invalid x.Size() = " << x.Size()
               << ", expected size = " << Height());
   MFEM_ASSERT(y.Size() == Width(), "invalid y.Size() = " << y.Size()
               << ", expected size = " << Width());

   PetscParVector *XX = GetX();
   PetscParVector *YY = GetY();
   YY->PlaceArray(x.GetData());
   XX->PlaceArray(y.GetData());
   MatMultKernel(A,a,YY->x,b,XX->x,true);
   XX->ResetArray();
   YY->ResetArray();
}

void PetscParMatrix::Print(const char *fname, bool binary) const
{
   if (fname)
   {
      PetscViewer view;

      if (binary)
      {
         ierr = PetscViewerBinaryOpen(PetscObjectComm((PetscObject)A),fname,
                                      FILE_MODE_WRITE,&view);
      }
      else
      {
         ierr = PetscViewerASCIIOpen(PetscObjectComm((PetscObject)A),fname,&view);
      }
      PCHKERRQ(A,ierr);
      ierr = MatView(A,view); PCHKERRQ(A,ierr);
      ierr = PetscViewerDestroy(&view); PCHKERRQ(A,ierr);
   }
   else
   {
      ierr = MatView(A,NULL); PCHKERRQ(A,ierr);
   }
}


PetscParMatrix * RAP(PetscParMatrix *Rt, PetscParMatrix *A, PetscParMatrix *P)
{
   Mat       pA = *A,pP = *P,pRt = *Rt;
   Mat       B;
   PetscBool Aismatis,Pismatis,Rtismatis;

   MFEM_VERIFY(A->Width() == P->Height(),
               "Petsc RAP: Number of local cols of A " << A->Width() <<
               " differs from number of local rows of P " << P->Height());
   MFEM_VERIFY(A->Height() == Rt->Height(),
               "Petsc RAP: Number of local rows of A " << A->Height() <<
               " differs from number of local rows of Rt " << Rt->Height());
   ierr = PetscObjectTypeCompare((PetscObject)pA,MATIS,&Aismatis);
   PCHKERRQ(pA,ierr);
   ierr = PetscObjectTypeCompare((PetscObject)pP,MATIS,&Pismatis);
   PCHKERRQ(pA,ierr);
   ierr = PetscObjectTypeCompare((PetscObject)pRt,MATIS,&Rtismatis);
   PCHKERRQ(pA,ierr);
   if (Aismatis &&
       Pismatis &&
       Rtismatis) // handle special case (this code will eventually go into PETSc)
   {
      Mat                    lA,lP,lB,lRt;
      ISLocalToGlobalMapping cl2gP,cl2gRt;
      PetscInt               rlsize,clsize,rsize,csize;

      ierr = MatGetLocalToGlobalMapping(pP,NULL,&cl2gP); PCHKERRQ(pA,ierr);
      ierr = MatGetLocalToGlobalMapping(pRt,NULL,&cl2gRt); PCHKERRQ(pA,ierr);
      ierr = MatGetLocalSize(pP,NULL,&clsize); PCHKERRQ(pP,ierr);
      ierr = MatGetLocalSize(pRt,NULL,&rlsize); PCHKERRQ(pRt,ierr);
      ierr = MatGetSize(pP,NULL,&csize); PCHKERRQ(pP,ierr);
      ierr = MatGetSize(pRt,NULL,&rsize); PCHKERRQ(pRt,ierr);
      ierr = MatCreate(A->GetComm(),&B); PCHKERRQ(pA,ierr);
      ierr = MatSetSizes(B,rlsize,clsize,rsize,csize); PCHKERRQ(B,ierr);
      ierr = MatSetType(B,MATIS); PCHKERRQ(B,ierr);
      ierr = MatSetLocalToGlobalMapping(B,cl2gRt,cl2gP); PCHKERRQ(B,ierr);
      ierr = MatISGetLocalMat(pA,&lA); PCHKERRQ(pA,ierr);
      ierr = MatISGetLocalMat(pP,&lP); PCHKERRQ(pA,ierr);
      ierr = MatISGetLocalMat(pRt,&lRt); PCHKERRQ(pA,ierr);
      if (lRt == lP)
      {
         ierr = MatPtAP(lA,lP,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&lB);
         PCHKERRQ(lA,ierr);
      }
      else
      {
         Mat lR;
         ierr = MatTranspose(lRt,MAT_INITIAL_MATRIX,&lR); PCHKERRQ(lRt,ierr);
         ierr = MatMatMatMult(lR,lA,lP,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&lB);
         PCHKERRQ(lRt,ierr);
         ierr = MatDestroy(&lR); PCHKERRQ(lRt,ierr);
      }

      // attach lRt matrix to the subdomain local matrix
      // it may be used if markers on vdofs have to be mapped on
      // subdomain true dofs
      {
         mfem::Array<Mat> *vmatsl2l = new mfem::Array<Mat>(1);
         ierr = PetscObjectReference((PetscObject)lRt); PCHKERRQ(lRt,ierr);
         (*vmatsl2l)[0] = lRt;

         PetscContainer c;
         ierr = PetscContainerCreate(PetscObjectComm((PetscObject)B),&c);
         PCHKERRQ(B,ierr);
         ierr = PetscContainerSetPointer(c,vmatsl2l); PCHKERRQ(c,ierr);
         ierr = PetscContainerSetUserDestroy(c,__mfem_matarray_container_destroy);
         PCHKERRQ(c,ierr);
         ierr = PetscObjectCompose((PetscObject)B,"_MatIS_PtAP_l2l",(PetscObject)c);
         PCHKERRQ(B,ierr);
         ierr = PetscContainerDestroy(&c); PCHKERRQ(B,ierr);
      }

      // Set local problem
      ierr = MatISSetLocalMat(B,lB); PCHKERRQ(lB,ierr);
      ierr = MatDestroy(&lB); PCHKERRQ(lA,ierr);
      ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY); PCHKERRQ(B,ierr);
      ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY); PCHKERRQ(B,ierr);
   }
   else // it raises an error if the PtAP is not supported in PETSc
   {
      if (pP == pRt)
      {
         ierr = MatPtAP(pA,pP,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B);
         PCHKERRQ(pA,ierr);
      }
      else
      {
         Mat pR;
         ierr = MatTranspose(pRt,MAT_INITIAL_MATRIX,&pR); PCHKERRQ(Rt,ierr);
         ierr = MatMatMatMult(pR,pA,pP,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B);
         PCHKERRQ(pRt,ierr);
         ierr = MatDestroy(&pR); PCHKERRQ(pRt,ierr);
      }
   }
   return new PetscParMatrix(B);
}

PetscParMatrix * RAP(PetscParMatrix *A, PetscParMatrix *P)
{
   PetscParMatrix *out = RAP(P,A,P);
   return out;
}

PetscParMatrix* PetscParMatrix::EliminateRowsCols(const Array<int> &rows_cols)
{
   Mat Ae;

   PetscParVector dummy(GetComm(),0);
   ierr = MatDuplicate(A,MAT_COPY_VALUES,&Ae); PCHKERRQ(A,ierr);
   EliminateRowsCols(rows_cols,dummy,dummy);
   ierr = MatAXPY(Ae,-1.,A,SAME_NONZERO_PATTERN); PCHKERRQ(A,ierr);
   return new PetscParMatrix(Ae);
}

void PetscParMatrix::EliminateRowsCols(const Array<int> &rows_cols,
                                       const HypreParVector &X,
                                       HypreParVector &B)
{
   MFEM_ABORT("PetscParMatrix::EliminateRowsCols(). To be implemented");
}

void PetscParMatrix::EliminateRowsCols(const Array<int> &rows_cols,
                                       const PetscParVector &X,
                                       PetscParVector &B)
{
   PetscInt M,N;
   ierr = MatGetSize(A,&M,&N); PCHKERRQ(A,ierr);
   MFEM_VERIFY(M == N,"Rectangular case unsupported");

   // TODO: what if a diagonal term is not present?
   ierr = MatSetOption(A,MAT_NO_OFF_PROC_ZERO_ROWS,PETSC_TRUE); PCHKERRQ(A,ierr);

   // rows need to be in global numbering
   PetscInt rst;
   ierr = MatGetOwnershipRange(A,&rst,NULL); PCHKERRQ(A,ierr);

   IS dir;
   ierr = Convert_Array_IS(GetComm(),true,&rows_cols,rst,&dir); PCHKERRQ(A,ierr);
   if (!X.GlobalSize() && !B.GlobalSize())
   {
      ierr = MatZeroRowsColumnsIS(A,dir,1.,NULL,NULL); PCHKERRQ(A,ierr);
   }
   else
   {
      ierr = MatZeroRowsColumnsIS(A,dir,1.,X,B); PCHKERRQ(A,ierr);
   }
   ierr = ISDestroy(&dir); PCHKERRQ(A,ierr);
}

Mat PetscParMatrix::ReleaseMat(bool dereference)
{

   Mat B = A;
   if (dereference)
   {
      MPI_Comm comm = GetComm();
      ierr = PetscObjectDereference((PetscObject)A); CCHKERRQ(comm,ierr);
   }
   A = NULL;
   return B;
}

Operator::Type PetscParMatrix::GetType() const
{
   PetscBool ok;
   MFEM_VERIFY(A, "no associated PETSc Mat object");
   PetscObject oA = (PetscObject)(this->A);
   // map all of MATAIJ, MATSEQAIJ, and MATMPIAIJ to -> PETSC_MATAIJ
   ierr = PetscObjectTypeCompare(oA, MATAIJ, &ok); PCHKERRQ(A,ierr);
   if (ok == PETSC_TRUE) { return PETSC_MATAIJ; }
   ierr = PetscObjectTypeCompare(oA, MATSEQAIJ, &ok); PCHKERRQ(A,ierr);
   if (ok == PETSC_TRUE) { return PETSC_MATAIJ; }
   ierr = PetscObjectTypeCompare(oA, MATMPIAIJ, &ok); PCHKERRQ(A,ierr);
   if (ok == PETSC_TRUE) { return PETSC_MATAIJ; }
   ierr = PetscObjectTypeCompare(oA, MATIS, &ok); PCHKERRQ(A,ierr);
   if (ok == PETSC_TRUE) { return PETSC_MATIS; }
   ierr = PetscObjectTypeCompare(oA, MATSHELL, &ok); PCHKERRQ(A,ierr);
   if (ok == PETSC_TRUE) { return PETSC_MATSHELL; }
   ierr = PetscObjectTypeCompare(oA, MATNEST, &ok); PCHKERRQ(A,ierr);
   if (ok == PETSC_TRUE) { return PETSC_MATNEST; }
#if defined(PETSC_HAVE_HYPRE)
   ierr = PetscObjectTypeCompare(oA, MATHYPRE, &ok); PCHKERRQ(A,ierr);
   if (ok == PETSC_TRUE) { return PETSC_MATHYPRE; }
#endif
   return PETSC_MATGENERIC;
}

void EliminateBC(PetscParMatrix &A, PetscParMatrix &Ae,
                 const Array<int> &ess_dof_list,
                 const Vector &X, Vector &B)
{
   const PetscScalar *array;
   Mat pA = const_cast<PetscParMatrix&>(A);

   // B -= Ae*X
   Ae.Mult(-1.0, X, 1.0, B);

   Vec diag = const_cast<PetscParVector&>((*A.GetX()));
   ierr = MatGetDiagonal(pA,diag); PCHKERRQ(pA,ierr);
   ierr = VecGetArrayRead(diag,&array); PCHKERRQ(diag,ierr);
   for (int i = 0; i < ess_dof_list.Size(); i++)
   {
      int r = ess_dof_list[i];
      B(r) = array[r] * X(r);
   }
   ierr = VecRestoreArrayRead(diag,&array); PCHKERRQ(diag,ierr);
}

// PetscSolver methods

PetscSolver::PetscSolver() : clcustom(false)
{
   obj = NULL;
   B = X = NULL;
   cid         = -1;
   operatorset = false;
   bchandler   = NULL;
   private_ctx = NULL;
}

PetscSolver::~PetscSolver()
{
   delete B;
   delete X;
   FreePrivateContext();
}

void PetscSolver::SetTol(double tol)
{
   SetRelTol(tol);
}

void PetscSolver::SetRelTol(double tol)
{
   if (cid == KSP_CLASSID)
   {
      KSP ksp = (KSP)obj;
      ierr = KSPSetTolerances(ksp,tol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
   }
   else if (cid == SNES_CLASSID)
   {
      SNES snes = (SNES)obj;
      ierr = SNESSetTolerances(snes,PETSC_DEFAULT,tol,PETSC_DEFAULT,PETSC_DEFAULT,
                               PETSC_DEFAULT);
   }
   else if (cid == TS_CLASSID)
   {
      TS ts = (TS)obj;
      ierr = TSSetTolerances(ts,PETSC_DECIDE,NULL,tol,NULL);
   }
   else
   {
      MFEM_ABORT("CLASSID = " << cid << " is not implemented!");
   }
   PCHKERRQ(obj,ierr);
}

void PetscSolver::SetAbsTol(double tol)
{
   if (cid == KSP_CLASSID)
   {
      KSP ksp = (KSP)obj;
      ierr = KSPSetTolerances(ksp,PETSC_DEFAULT,tol,PETSC_DEFAULT,PETSC_DEFAULT);
   }
   else if (cid == SNES_CLASSID)
   {
      SNES snes = (SNES)obj;
      ierr = SNESSetTolerances(snes,tol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,
                               PETSC_DEFAULT);
   }
   else if (cid == TS_CLASSID)
   {
      TS ts = (TS)obj;
      ierr = TSSetTolerances(ts,tol,NULL,PETSC_DECIDE,NULL);
   }
   else
   {
      MFEM_ABORT("CLASSID = " << cid << " is not implemented!");
   }
   PCHKERRQ(obj,ierr);
}

void PetscSolver::SetMaxIter(int max_iter)
{
   if (cid == KSP_CLASSID)
   {
      KSP ksp = (KSP)obj;
      ierr = KSPSetTolerances(ksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,
                              max_iter);
   }
   else if (cid == SNES_CLASSID)
   {
      SNES snes = (SNES)obj;
      ierr = SNESSetTolerances(snes,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,
                               max_iter,PETSC_DEFAULT);
   }
   else if (cid == TS_CLASSID)
   {
      TS ts = (TS)obj;
      ierr = TSSetMaxSteps(ts,max_iter);
   }
   else
   {
      MFEM_ABORT("CLASSID = " << cid << " is not implemented!");
   }
   PCHKERRQ(obj,ierr);
}


void PetscSolver::SetPrintLevel(int plev)
{
   typedef PetscErrorCode (*myPetscFunc)(void**);
   PetscViewerAndFormat *vf = NULL;
   PetscViewer viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm(obj));

   if (plev > 0)
   {
      ierr = PetscViewerAndFormatCreate(viewer,PETSC_VIEWER_DEFAULT,&vf);
      PCHKERRQ(obj,ierr);
   }
   if (cid == KSP_CLASSID)
   {
      // there are many other options, see the function KSPSetFromOptions() in
      // src/ksp/ksp/interface/itcl.c
      typedef PetscErrorCode (*myMonitor)(KSP,PetscInt,PetscReal,void*);
      KSP ksp = (KSP)obj;
      if (plev >= 0)
      {
         ierr = KSPMonitorCancel(ksp); PCHKERRQ(ksp,ierr);
      }
      if (plev == 1)
      {
         ierr = KSPMonitorSet(ksp,(myMonitor)KSPMonitorDefault,vf,
                              (myPetscFunc)PetscViewerAndFormatDestroy);
         PCHKERRQ(ksp,ierr);
      }
      else if (plev > 1)
      {
         ierr = KSPSetComputeSingularValues(ksp,PETSC_TRUE); PCHKERRQ(ksp,ierr);
         ierr = KSPMonitorSet(ksp,(myMonitor)KSPMonitorSingularValue,vf,
                              (myPetscFunc)PetscViewerAndFormatDestroy);
         PCHKERRQ(ksp,ierr);
         if (plev > 2)
         {
            ierr = PetscViewerAndFormatCreate(viewer,PETSC_VIEWER_DEFAULT,&vf);
            PCHKERRQ(viewer,ierr);
            ierr = KSPMonitorSet(ksp,(myMonitor)KSPMonitorTrueResidualNorm,vf,
                                 (myPetscFunc)PetscViewerAndFormatDestroy);
            PCHKERRQ(ksp,ierr);
         }
      }
   }
   else if (cid == SNES_CLASSID)
   {
      typedef PetscErrorCode (*myMonitor)(SNES,PetscInt,PetscReal,void*);
      SNES snes = (SNES)obj;
      if (plev >= 0)
      {
         ierr = SNESMonitorCancel(snes); PCHKERRQ(snes,ierr);
      }
      if (plev > 0)
      {
         ierr = SNESMonitorSet(snes,(myMonitor)SNESMonitorDefault,vf,
                               (myPetscFunc)PetscViewerAndFormatDestroy);
         PCHKERRQ(snes,ierr);
      }
   }
   else if (cid == TS_CLASSID)
   {
      TS ts = (TS)obj;
      if (plev >= 0)
      {
         ierr = TSMonitorCancel(ts); PCHKERRQ(ts,ierr);
      }
   }
   else
   {
      MFEM_ABORT("CLASSID = " << cid << " is not implemented!");
   }
}

void PetscSolver::SetMonitor(PetscSolverMonitor *ctx)
{
   if (cid == KSP_CLASSID)
   {
      ierr = KSPMonitorSet((KSP)obj,__mfem_ksp_monitor,ctx,NULL);
      PCHKERRQ(obj,ierr);
   }
   else if (cid == SNES_CLASSID)
   {
      ierr = SNESMonitorSet((SNES)obj,__mfem_snes_monitor,ctx,NULL);
      PCHKERRQ(obj,ierr);
   }
   else if (cid == TS_CLASSID)
   {
      ierr = TSMonitorSet((TS)obj,__mfem_ts_monitor,ctx,NULL);
      PCHKERRQ(obj,ierr);
   }
   else
   {
      MFEM_ABORT("CLASSID = " << cid << " is not implemented!");
   }
}

void PetscSolver::SetBCHandler(PetscBCHandler *bch)
{
   bchandler = bch;
   if (cid == SNES_CLASSID)
   {
      __mfem_snes_ctx* snes_ctx = (__mfem_snes_ctx*)private_ctx;
      snes_ctx->bchandler = bchandler;
   }
   else if (cid == TS_CLASSID)
   {
      __mfem_ts_ctx* ts_ctx = (__mfem_ts_ctx*)private_ctx;
      ts_ctx->bchandler = bchandler;
   }
   else
   {
      MFEM_ABORT("Handling of essential bc only implemented for nonlinear and time-dependent solvers");
   }
}

void PetscSolver::SetPreconditionerFactory(PetscPreconditionerFactory *factory)
{
   PC pc = NULL;
   if (cid == TS_CLASSID)
   {
      SNES snes;
      KSP  ksp;

      ierr = TSGetSNES((TS)obj,&snes); PCHKERRQ(obj,ierr);
      ierr = SNESGetKSP(snes,&ksp); PCHKERRQ(obj,ierr);
      ierr = KSPGetPC(ksp,&pc); PCHKERRQ(obj,ierr);
   }
   else if (cid == SNES_CLASSID)
   {
      KSP ksp;

      ierr = SNESGetKSP((SNES)obj,&ksp); PCHKERRQ(obj,ierr);
      ierr = KSPGetPC(ksp,&pc); PCHKERRQ(obj,ierr);
   }
   else if (cid == KSP_CLASSID)
   {
      ierr = KSPGetPC((KSP)obj,&pc); PCHKERRQ(obj,ierr);
   }
   else if (cid == PC_CLASSID)
   {
      pc = (PC)obj;
   }
   else
   {
      MFEM_ABORT("No support for PetscPreconditionerFactory for this object");
   }
   ierr = MakeShellPCWithFactory(pc,factory); PCHKERRQ(pc,ierr);
}

void PetscSolver::Customize(bool customize) const
{
   if (!customize) { clcustom = true; }
   if (!clcustom)
   {
      if (cid == PC_CLASSID)
      {
         PC pc = (PC)obj;
         ierr = PCSetFromOptions(pc); PCHKERRQ(pc, ierr);
      }
      else if (cid == KSP_CLASSID)
      {
         KSP ksp = (KSP)obj;
         ierr = KSPSetFromOptions(ksp); PCHKERRQ(ksp, ierr);
      }
      else if (cid == SNES_CLASSID)
      {
         SNES snes = (SNES)obj;
         ierr = SNESSetFromOptions(snes); PCHKERRQ(snes, ierr);
      }
      else if (cid == TS_CLASSID)
      {
         TS ts = (TS)obj;
         ierr = TSSetFromOptions(ts); PCHKERRQ(ts, ierr);
      }
      else
      {
         MFEM_ABORT("CLASSID = " << cid << " is not implemented!");
      }
   }
   clcustom = true;
}

int PetscSolver::GetConverged()
{
   if (cid == KSP_CLASSID)
   {
      KSP ksp = (KSP)obj;
      KSPConvergedReason reason;
      ierr = KSPGetConvergedReason(ksp,&reason);
      PCHKERRQ(ksp,ierr);
      return reason > 0 ? 1 : 0;
   }
   else if (cid == SNES_CLASSID)
   {
      SNES snes = (SNES)obj;
      SNESConvergedReason reason;
      ierr = SNESGetConvergedReason(snes,&reason);
      PCHKERRQ(snes,ierr);
      return reason > 0 ? 1 : 0;
   }
   else if (cid == TS_CLASSID)
   {
      TS ts = (TS)obj;
      TSConvergedReason reason;
      ierr = TSGetConvergedReason(ts,&reason);
      PCHKERRQ(ts,ierr);
      return reason > 0 ? 1 : 0;
   }
   else
   {
      MFEM_ABORT("CLASSID = " << cid << " is not implemented!");
      return -1;
   }
}

int PetscSolver::GetNumIterations()
{
   if (cid == KSP_CLASSID)
   {
      KSP ksp = (KSP)obj;
      PetscInt its;
      ierr = KSPGetIterationNumber(ksp,&its);
      PCHKERRQ(ksp,ierr);
      return its;
   }
   else if (cid == SNES_CLASSID)
   {
      SNES snes = (SNES)obj;
      PetscInt its;
      ierr = SNESGetIterationNumber(snes,&its);
      PCHKERRQ(snes,ierr);
      return its;
   }
   else if (cid == TS_CLASSID)
   {
      TS ts = (TS)obj;
      PetscInt its;
      ierr = TSGetStepNumber(ts,&its);
      PCHKERRQ(ts,ierr);
      return its;
   }
   else
   {
      MFEM_ABORT("CLASSID = " << cid << " is not implemented!");
      return -1;
   }
}

double PetscSolver::GetFinalNorm()
{
   if (cid == KSP_CLASSID)
   {
      KSP ksp = (KSP)obj;
      PetscReal norm;
      ierr = KSPGetResidualNorm(ksp,&norm);
      PCHKERRQ(ksp,ierr);
      return norm;
   }
   if (cid == SNES_CLASSID)
   {
      SNES snes = (SNES)obj;
      PetscReal norm;
      ierr = SNESGetFunctionNorm(snes,&norm);
      PCHKERRQ(snes,ierr);
      return norm;
   }
   else
   {
      MFEM_ABORT("CLASSID = " << cid << " is not implemented!");
      return PETSC_MAX_REAL;
   }
}

void PetscSolver::CreatePrivateContext()
{
   FreePrivateContext();
   if (cid == SNES_CLASSID)
   {
      __mfem_snes_ctx *snes_ctx;
      ierr = PetscNew(&snes_ctx); CCHKERRQ(PETSC_COMM_SELF,ierr);
      snes_ctx->op = NULL;
      snes_ctx->bchandler = NULL;
      snes_ctx->work = NULL;
      snes_ctx->jacType = Operator::PETSC_MATAIJ;
      private_ctx = (void*) snes_ctx;
   }
   else if (cid == TS_CLASSID)
   {
      __mfem_ts_ctx *ts_ctx;
      ierr = PetscNew(&ts_ctx); CCHKERRQ(PETSC_COMM_SELF,ierr);
      ts_ctx->op = NULL;
      ts_ctx->bchandler = NULL;
      ts_ctx->work = NULL;
      ts_ctx->cached_shift = std::numeric_limits<PetscReal>::min();
      ts_ctx->cached_ijacstate = -1;
      ts_ctx->cached_rhsjacstate = -1;
      ts_ctx->type = PetscODESolver::ODE_SOLVER_GENERAL;
      ts_ctx->jacType = Operator::PETSC_MATAIJ;
      private_ctx = (void*) ts_ctx;
   }
}

void PetscSolver::FreePrivateContext()
{
   if (!private_ctx) { return; }
   // free private context's owned objects
   if (cid == SNES_CLASSID)
   {
      __mfem_snes_ctx *snes_ctx = (__mfem_snes_ctx *)private_ctx;
      delete snes_ctx->work;
   }
   else if (cid == TS_CLASSID)
   {
      __mfem_ts_ctx *ts_ctx = (__mfem_ts_ctx *)private_ctx;
      delete ts_ctx->work;
   }
   ierr = PetscFree(private_ctx); CCHKERRQ(PETSC_COMM_SELF,ierr);
}

// PetscBCHandler methods

PetscBCHandler::PetscBCHandler(Array<int>& ess_tdof_list,
                               enum PetscBCHandler::Type _type)
   : bctype(_type), setup(false), eval_t(0.0),
     eval_t_cached(std::numeric_limits<double>::min())
{
   SetTDofs(ess_tdof_list);
}

void PetscBCHandler::SetTDofs(Array<int>& list)
{
   ess_tdof_list.SetSize(list.Size());
   ess_tdof_list.Assign(list);
   setup = false;
}

void PetscBCHandler::SetUp(PetscInt n)
{
   if (setup) { return; }
   if (bctype == CONSTANT)
   {
      eval_g.SetSize(n);
      this->Eval(eval_t,eval_g);
      eval_t_cached = eval_t;
   }
   else if (bctype == TIME_DEPENDENT)
   {
      eval_g.SetSize(n);
   }
   setup = true;
}

void PetscBCHandler::ApplyBC(const Vector &x, Vector &y)
{
   if (!setup) { MFEM_ABORT("PetscBCHandler not yet setup"); }
   y = x;
   if (bctype == ZERO)
   {
      for (PetscInt i = 0; i < ess_tdof_list.Size(); ++i)
      {
         y[ess_tdof_list[i]] = 0.0;
      }
   }
   else
   {
      if (bctype != CONSTANT && eval_t != eval_t_cached)
      {
         Eval(eval_t,eval_g);
         eval_t_cached = eval_t;
      }
      for (PetscInt i = 0; i < ess_tdof_list.Size(); ++i)
      {
         y[ess_tdof_list[i]] = eval_g[ess_tdof_list[i]];
      }
   }
}

void PetscBCHandler::FixResidualBC(const Vector& x, Vector& y)
{
   if (!setup) { MFEM_ABORT("PetscBCHandler not yet setup"); }
   if (bctype == ZERO)
   {
      for (PetscInt i = 0; i < ess_tdof_list.Size(); ++i)
      {
         y[ess_tdof_list[i]] = x[ess_tdof_list[i]];
      }
   }
   else
   {
      for (PetscInt i = 0; i < ess_tdof_list.Size(); ++i)
      {
         y[ess_tdof_list[i]] = x[ess_tdof_list[i]] - eval_g[ess_tdof_list[i]];
      }
   }
}

// PetscLinearSolver methods

PetscLinearSolver::PetscLinearSolver(MPI_Comm comm, const std::string &prefix,
                                     bool wrapin)
   : PetscSolver(), Solver(), wrap(wrapin)
{
   KSP ksp;
   ierr = KSPCreate(comm,&ksp); CCHKERRQ(comm,ierr);
   obj  = (PetscObject)ksp;
   ierr = PetscObjectGetClassId(obj,&cid); PCHKERRQ(obj,ierr);
   ierr = KSPSetOptionsPrefix(ksp, prefix.c_str()); PCHKERRQ(ksp, ierr);
}

PetscLinearSolver::PetscLinearSolver(const PetscParMatrix &A,
                                     const std::string &prefix)
   : PetscSolver(), Solver(), wrap(false)
{
   KSP ksp;
   ierr = KSPCreate(A.GetComm(),&ksp); CCHKERRQ(A.GetComm(),ierr);
   obj  = (PetscObject)ksp;
   ierr = PetscObjectGetClassId(obj,&cid); PCHKERRQ(obj,ierr);
   ierr = KSPSetOptionsPrefix(ksp, prefix.c_str()); PCHKERRQ(ksp, ierr);
   SetOperator(A);
}

PetscLinearSolver::PetscLinearSolver(const HypreParMatrix &A, bool wrapin,
                                     const std::string &prefix)
   : PetscSolver(), Solver(), wrap(wrapin)
{
   KSP ksp;
   ierr = KSPCreate(A.GetComm(),&ksp); CCHKERRQ(A.GetComm(),ierr);
   obj  = (PetscObject)ksp;
   ierr = PetscObjectGetClassId(obj, &cid); PCHKERRQ(obj, ierr);
   ierr = KSPSetOptionsPrefix(ksp, prefix.c_str()); PCHKERRQ(ksp, ierr);
   SetOperator(A);
}

void PetscLinearSolver::SetOperator(const Operator &op)
{
   const HypreParMatrix *hA = dynamic_cast<const HypreParMatrix *>(&op);
   PetscParMatrix       *pA = const_cast<PetscParMatrix *>
                              (dynamic_cast<const PetscParMatrix *>(&op));
   const Operator       *oA = dynamic_cast<const Operator *>(&op);

   // Preserve Pmat if already set
   KSP ksp = (KSP)obj;
   Mat P = NULL;
   PetscBool pmat;
   ierr = KSPGetOperatorsSet(ksp,NULL,&pmat); PCHKERRQ(ksp,ierr);
   if (pmat)
   {
      ierr = KSPGetOperators(ksp,NULL,&P); PCHKERRQ(ksp,ierr);
      ierr = PetscObjectReference((PetscObject)P); PCHKERRQ(ksp,ierr);
   }

   // update base classes: Operator, Solver, PetscLinearSolver
   bool delete_pA = false;
   if (!pA)
   {
      if (hA)
      {
         // Create MATSHELL object or convert into a format suitable to construct preconditioners
         pA = new PetscParMatrix(hA, wrap ? PETSC_MATSHELL : PETSC_MATAIJ);
         delete_pA = true;
      }
      else if (oA) // fallback to general operator
      {
         // Create MATSHELL or MATNEST (if oA is a BlockOperator) object
         // If oA is a BlockOperator, Operator::Type is relevant to the subblocks
         pA = new PetscParMatrix(PetscObjectComm(obj),oA,
                                 wrap ? PETSC_MATSHELL : PETSC_MATAIJ);
         delete_pA = true;
      }
   }
   MFEM_VERIFY(pA, "Unsupported operation!");

   // Set operators into PETSc KSP
   Mat A = pA->A;
   if (operatorset)
   {
      Mat C;
      PetscInt nheight,nwidth,oheight,owidth;

      ierr = KSPGetOperators(ksp,&C,NULL); PCHKERRQ(ksp,ierr);
      ierr = MatGetSize(A,&nheight,&nwidth); PCHKERRQ(A,ierr);
      ierr = MatGetSize(C,&oheight,&owidth); PCHKERRQ(A,ierr);
      if (nheight != oheight || nwidth != owidth)
      {
         // reinit without destroying the KSP
         // communicator remains the same
         ierr = KSPReset(ksp); PCHKERRQ(ksp,ierr);
         delete X;
         delete B;
         X = B = NULL;
         wrap = false;
      }
   }
   if (P)
   {
      ierr = KSPSetOperators(ksp,A,P); PCHKERRQ(ksp,ierr);
      ierr = MatDestroy(&P); PCHKERRQ(ksp,ierr);
   }
   else
   {
      ierr = KSPSetOperators(ksp,A,A); PCHKERRQ(ksp,ierr);
   }

   // Update PetscSolver
   operatorset = true;

   // Update the Operator fields.
   height = pA->Height();
   width  = pA->Width();

   if (delete_pA) { delete pA; }
}

void PetscLinearSolver::SetOperator(const Operator &op, const Operator &pop)
{
   const HypreParMatrix *hA = dynamic_cast<const HypreParMatrix *>(&op);
   PetscParMatrix       *pA = const_cast<PetscParMatrix *>
                              (dynamic_cast<const PetscParMatrix *>(&op));
   const Operator       *oA = dynamic_cast<const Operator *>(&op);

   PetscParMatrix       *ppA = const_cast<PetscParMatrix *>
                               (dynamic_cast<const PetscParMatrix *>(&pop));
   const Operator       *poA = dynamic_cast<const Operator *>(&pop);

   // Convert Operator for linear system
   bool delete_pA = false;
   if (!pA)
   {
      if (hA)
      {
         // Create MATSHELL object or convert into a format suitable to construct preconditioners
         pA = new PetscParMatrix(hA, wrap ? PETSC_MATSHELL : PETSC_MATAIJ);
         delete_pA = true;
      }
      else if (oA) // fallback to general operator
      {
         // Create MATSHELL or MATNEST (if oA is a BlockOperator) object
         // If oA is a BlockOperator, Operator::Type is relevant to the subblocks
         pA = new PetscParMatrix(PetscObjectComm(obj),oA,
                                 wrap ? PETSC_MATSHELL : PETSC_MATAIJ);
         delete_pA = true;
      }
   }
   MFEM_VERIFY(pA, "Unsupported operation!");

   // Convert Operator to be preconditioned
   bool delete_ppA = false;
   if (!ppA)
   {
      if (oA == poA && !wrap) // Same operator, already converted
      {
         ppA = pA;
      }
      else
      {
         ppA = new PetscParMatrix(PetscObjectComm(obj), poA, PETSC_MATAIJ);
         delete_ppA = true;
      }
   }
   MFEM_VERIFY(ppA, "Unsupported operation!");

   // Set operators into PETSc KSP
   KSP ksp = (KSP)obj;
   Mat A = pA->A;
   Mat P = ppA->A;
   if (operatorset)
   {
      Mat C;
      PetscInt nheight,nwidth,oheight,owidth;

      ierr = KSPGetOperators(ksp,&C,NULL); PCHKERRQ(ksp,ierr);
      ierr = MatGetSize(A,&nheight,&nwidth); PCHKERRQ(A,ierr);
      ierr = MatGetSize(C,&oheight,&owidth); PCHKERRQ(A,ierr);
      if (nheight != oheight || nwidth != owidth)
      {
         // reinit without destroying the KSP
         // communicator remains the same
         ierr = KSPReset(ksp); PCHKERRQ(ksp,ierr);
         delete X;
         delete B;
         X = B = NULL;
         wrap = false;
      }
   }
   ierr = KSPSetOperators(ksp,A,P); PCHKERRQ(ksp,ierr);

   // Update PetscSolver
   operatorset = true;

   // Update the Operator fields.
   height = pA->Height();
   width  = pA->Width();

   if (delete_pA) { delete pA; }
   if (delete_ppA) { delete ppA; }
}

void PetscLinearSolver::SetPreconditioner(Solver &precond)
{
   KSP ksp = (KSP)obj;

   // Preserve Amat if already set
   Mat A = NULL;
   PetscBool amat;
   ierr = KSPGetOperatorsSet(ksp,&amat,NULL); PCHKERRQ(ksp,ierr);
   if (amat)
   {
      ierr = KSPGetOperators(ksp,&A,NULL); PCHKERRQ(ksp,ierr);
      ierr = PetscObjectReference((PetscObject)A); PCHKERRQ(ksp,ierr);
   }
   PetscPreconditioner *ppc = dynamic_cast<PetscPreconditioner *>(&precond);
   if (ppc)
   {
      ierr = KSPSetPC(ksp,*ppc); PCHKERRQ(ksp,ierr);
   }
   else
   {
      // wrap the Solver action
      // Solver is assumed to be already setup
      // ownership of precond is not tranferred,
      // consistently with other MFEM's linear solvers
      PC pc;
      ierr = KSPGetPC(ksp,&pc); PCHKERRQ(ksp,ierr);
      ierr = MakeShellPC(pc,precond,false); PCHKERRQ(ksp,ierr);
   }
   if (A)
   {
      Mat P;

      ierr = KSPGetOperators(ksp,NULL,&P); PCHKERRQ(ksp,ierr);
      ierr = PetscObjectReference((PetscObject)P); PCHKERRQ(ksp,ierr);
      ierr = KSPSetOperators(ksp,A,P); PCHKERRQ(ksp,ierr);
      ierr = MatDestroy(&A); PCHKERRQ(ksp,ierr);
      ierr = MatDestroy(&P); PCHKERRQ(ksp,ierr);
   }
}

void PetscLinearSolver::Mult(const Vector &b, Vector &x) const
{
   KSP ksp = (KSP)obj;

   if (!B || !X)
   {
      Mat pA = NULL;
      ierr = KSPGetOperators(ksp, &pA, NULL); PCHKERRQ(obj, ierr);
      if (!B)
      {
         PetscParMatrix A = PetscParMatrix(pA, true);
         B = new PetscParVector(A, true, false);
      }
      if (!X)
      {
         PetscParMatrix A = PetscParMatrix(pA, true);
         X = new PetscParVector(A, false, false);
      }
   }
   B->PlaceArray(b.GetData());
   X->PlaceArray(x.GetData());

   Customize();

   ierr = KSPSetInitialGuessNonzero(ksp, (PetscBool)iterative_mode);
   PCHKERRQ(ksp, ierr);

   // Solve the system.
   ierr = KSPSolve(ksp, B->x, X->x); PCHKERRQ(ksp,ierr);
   B->ResetArray();
   X->ResetArray();
}

PetscLinearSolver::~PetscLinearSolver()
{
   MPI_Comm comm;
   KSP ksp = (KSP)obj;
   ierr = PetscObjectGetComm((PetscObject)ksp,&comm); PCHKERRQ(ksp,ierr);
   ierr = KSPDestroy(&ksp); CCHKERRQ(comm,ierr);
}

// PetscPCGSolver methods

PetscPCGSolver::PetscPCGSolver(MPI_Comm comm, const std::string &prefix)
   : PetscLinearSolver(comm,prefix)
{
   KSP ksp = (KSP)obj;
   ierr = KSPSetType(ksp,KSPCG); PCHKERRQ(ksp,ierr);
   // this is to obtain a textbook PCG
   ierr = KSPSetNormType(ksp,KSP_NORM_NATURAL); PCHKERRQ(ksp,ierr);
}

PetscPCGSolver::PetscPCGSolver(PetscParMatrix& A, const std::string &prefix)
   : PetscLinearSolver(A,prefix)
{
   KSP ksp = (KSP)obj;
   ierr = KSPSetType(ksp,KSPCG); PCHKERRQ(ksp,ierr);
   // this is to obtain a textbook PCG
   ierr = KSPSetNormType(ksp,KSP_NORM_NATURAL); PCHKERRQ(ksp,ierr);
}

PetscPCGSolver::PetscPCGSolver(HypreParMatrix& A, bool wrap,
                               const std::string &prefix)
   : PetscLinearSolver(A,wrap,prefix)
{
   KSP ksp = (KSP)obj;
   ierr = KSPSetType(ksp,KSPCG); PCHKERRQ(ksp,ierr);
   // this is to obtain a textbook PCG
   ierr = KSPSetNormType(ksp,KSP_NORM_NATURAL); PCHKERRQ(ksp,ierr);
}

// PetscPreconditioner methods

PetscPreconditioner::PetscPreconditioner(MPI_Comm comm,
                                         const std::string &prefix)
   : PetscSolver(), Solver()
{
   PC pc;
   ierr = PCCreate(comm,&pc); CCHKERRQ(comm,ierr);
   obj  = (PetscObject)pc;
   ierr = PetscObjectGetClassId(obj,&cid); PCHKERRQ(obj,ierr);
   ierr = PCSetOptionsPrefix(pc, prefix.c_str()); PCHKERRQ(pc, ierr);
}

PetscPreconditioner::PetscPreconditioner(PetscParMatrix &A,
                                         const string &prefix)
   : PetscSolver(), Solver()
{
   PC pc;
   ierr = PCCreate(A.GetComm(),&pc); CCHKERRQ(A.GetComm(),ierr);
   obj  = (PetscObject)pc;
   ierr = PetscObjectGetClassId(obj,&cid); PCHKERRQ(obj,ierr);
   ierr = PCSetOptionsPrefix(pc, prefix.c_str()); PCHKERRQ(pc, ierr);
   SetOperator(A);
}

PetscPreconditioner::PetscPreconditioner(MPI_Comm comm, Operator &op,
                                         const string &prefix)
   : PetscSolver(), Solver()
{
   PC pc;
   ierr = PCCreate(comm,&pc); CCHKERRQ(comm,ierr);
   obj  = (PetscObject)pc;
   ierr = PetscObjectGetClassId(obj,&cid); PCHKERRQ(obj,ierr);
   ierr = PCSetOptionsPrefix(pc, prefix.c_str()); PCHKERRQ(pc, ierr);
   SetOperator(op);
}

void PetscPreconditioner::SetOperator(const Operator &op)
{
   bool delete_pA = false;
   PetscParMatrix *pA = const_cast<PetscParMatrix *>
                        (dynamic_cast<const PetscParMatrix *>(&op));

   if (!pA)
   {
      const Operator *cop = dynamic_cast<const Operator *>(&op);
      pA = new PetscParMatrix(PetscObjectComm(obj),cop,PETSC_MATAIJ);
      delete_pA = true;
   }

   // Set operators into PETSc PC
   PC pc = (PC)obj;
   Mat A = pA->A;
   if (operatorset)
   {
      Mat C;
      PetscInt nheight,nwidth,oheight,owidth;

      ierr = PCGetOperators(pc,&C,NULL); PCHKERRQ(pc,ierr);
      ierr = MatGetSize(A,&nheight,&nwidth); PCHKERRQ(A,ierr);
      ierr = MatGetSize(C,&oheight,&owidth); PCHKERRQ(A,ierr);
      if (nheight != oheight || nwidth != owidth)
      {
         // reinit without destroying the PC
         // communicator remains the same
         ierr = PCReset(pc); PCHKERRQ(pc,ierr);
         delete X;
         delete B;
         X = B = NULL;
      }
   }
   ierr = PCSetOperators(pc,pA->A,pA->A); PCHKERRQ(obj,ierr);

   // Update PetscSolver
   operatorset = true;

   // Update the Operator fields.
   height = pA->Height();
   width  = pA->Width();

   if (delete_pA) { delete pA; };
}

void PetscPreconditioner::Mult(const Vector &b, Vector &x) const
{
   PC pc = (PC)obj;

   if (!B || !X)
   {
      Mat pA = NULL;
      ierr = PCGetOperators(pc, NULL, &pA); PCHKERRQ(obj, ierr);
      if (!B)
      {
         PetscParMatrix A(pA, true);
         B = new PetscParVector(A, true, false);
      }
      if (!X)
      {
         PetscParMatrix A(pA, true);
         X = new PetscParVector(A, false, false);
      }
   }
   B->PlaceArray(b.GetData());
   X->PlaceArray(x.GetData());

   Customize();

   // Apply the preconditioner.
   ierr = PCApply(pc, B->x, X->x); PCHKERRQ(pc, ierr);
   B->ResetArray();
   X->ResetArray();
}

PetscPreconditioner::~PetscPreconditioner()
{
   MPI_Comm comm;
   PC pc = (PC)obj;
   ierr = PetscObjectGetComm((PetscObject)pc,&comm); PCHKERRQ(pc,ierr);
   ierr = PCDestroy(&pc); CCHKERRQ(comm,ierr);
}

// PetscBDDCSolver methods

void PetscBDDCSolver::BDDCSolverConstructor(const PetscBDDCSolverParams &opts)
{
   MPI_Comm comm = PetscObjectComm(obj);

   // get PETSc object
   PC pc = (PC)obj;
   Mat pA;
   ierr = PCGetOperators(pc,NULL,&pA); PCHKERRQ(pc,ierr);

   // matrix type should be of type MATIS
   PetscBool ismatis;
   ierr = PetscObjectTypeCompare((PetscObject)pA,MATIS,&ismatis);
   PCHKERRQ(pA,ierr);
   MFEM_VERIFY(ismatis,"PetscBDDCSolver needs the matrix in unassembled format");

   // set PETSc PC type to PCBDDC
   ierr = PCSetType(pc,PCBDDC); PCHKERRQ(obj,ierr);

   // index sets for fields splitting
   IS *fields = NULL;
   PetscInt nf = 0;

   // index sets for boundary dofs specification (Essential = dir, Natural = neu)
   IS dir = NULL, neu = NULL;
   PetscInt rst;

   // Extract l2l matrices
   Array<Mat> *l2l = NULL;
   if (opts.ess_dof_local || opts.nat_dof_local)
   {
      PetscContainer c;

      ierr = PetscObjectQuery((PetscObject)pA,"_MatIS_PtAP_l2l",(PetscObject*)&c);
      MFEM_VERIFY(c,"Local-to-local PETSc container not present");
      ierr = PetscContainerGetPointer(c,(void**)&l2l); PCHKERRQ(c,ierr);
   }

   // check information about index sets (essential dofs, fields, etc.)
#ifdef MFEM_DEBUG
   {
      // make sure ess/nat_dof have been collectively set
      PetscBool lpr = PETSC_FALSE,pr;
      if (opts.ess_dof) { lpr = PETSC_TRUE; }
      ierr = MPI_Allreduce(&lpr,&pr,1,MPIU_BOOL,MPI_LOR,comm);
      PCHKERRQ(pA,ierr);
      MFEM_VERIFY(lpr == pr,"ess_dof should be collectively set");
      lpr = PETSC_FALSE;
      if (opts.nat_dof) { lpr = PETSC_TRUE; }
      ierr = MPI_Allreduce(&lpr,&pr,1,MPIU_BOOL,MPI_LOR,comm);
      PCHKERRQ(pA,ierr);
      MFEM_VERIFY(lpr == pr,"nat_dof should be collectively set");
      // make sure fields have been collectively set
      PetscInt ms[2],Ms[2];
      ms[0] = -nf; ms[1] = nf;
      ierr = MPI_Allreduce(&ms,&Ms,2,MPIU_INT,MPI_MAX,comm);
      PCHKERRQ(pA,ierr);
      MFEM_VERIFY(-Ms[0] == Ms[1],
                  "number of fields should be the same across processes");
   }
#endif

   // boundary sets
   ierr = MatGetOwnershipRange(pA,&rst,NULL); PCHKERRQ(pA,ierr);
   if (opts.ess_dof)
   {
      PetscInt st = opts.ess_dof_local ? 0 : rst;
      if (!opts.ess_dof_local)
      {
         // need to compute the boundary dofs in global ordering
         ierr = Convert_Array_IS(comm,true,opts.ess_dof,st,&dir);
         CCHKERRQ(comm,ierr);
         ierr = PCBDDCSetDirichletBoundaries(pc,dir); PCHKERRQ(pc,ierr);
      }
      else
      {
         // need to compute a list for the marked boundary dofs in local ordering
         ierr = Convert_Vmarks_IS(comm,*l2l,opts.ess_dof,st,&dir);
         CCHKERRQ(comm,ierr);
         ierr = PCBDDCSetDirichletBoundariesLocal(pc,dir); PCHKERRQ(pc,ierr);
      }
   }
   if (opts.nat_dof)
   {
      PetscInt st = opts.nat_dof_local ? 0 : rst;
      if (!opts.nat_dof_local)
      {
         // need to compute the boundary dofs in global ordering
         ierr = Convert_Array_IS(comm,true,opts.nat_dof,st,&neu);
         CCHKERRQ(comm,ierr);
         ierr = PCBDDCSetNeumannBoundaries(pc,neu); PCHKERRQ(pc,ierr);
      }
      else
      {
         // need to compute a list for the marked boundary dofs in local ordering
         ierr = Convert_Vmarks_IS(comm,*l2l,opts.nat_dof,st,&neu);
         CCHKERRQ(comm,ierr);
         ierr = PCBDDCSetNeumannBoundariesLocal(pc,neu); PCHKERRQ(pc,ierr);
      }
   }

   // field splitting
   if (nf)
   {
      ierr = PCBDDCSetDofsSplitting(pc,nf,fields); PCHKERRQ(pc,ierr);
      for (int i = 0; i < nf; i++)
      {
         ierr = ISDestroy(&fields[i]); CCHKERRQ(comm,ierr);
      }
      ierr = PetscFree(fields); PCHKERRQ(pc,ierr);
   }

   // code for block size is disabled since we cannot change the matrix
   // block size after it has been setup
   // int bs = 1;

   // Customize using the finite element space (if any)
   ParFiniteElementSpace *fespace = opts.fespace;
   if (fespace)
   {
      const     FiniteElementCollection *fec = fespace->FEColl();
      bool      edgespace, rtspace;
      bool      needint = false;
      bool      tracespace, rt_tracespace, edge_tracespace;
      int       dim, p;
      PetscBool B_is_Trans = PETSC_FALSE;

      ParMesh *pmesh = (ParMesh *) fespace->GetMesh();
      dim = pmesh->Dimension();
      // bs = fec->DofForGeometry(Geometry::POINT);
      // bs = bs ? bs : 1;
      rtspace = dynamic_cast<const RT_FECollection*>(fec);
      edgespace = dynamic_cast<const ND_FECollection*>(fec);
      edge_tracespace = dynamic_cast<const ND_Trace_FECollection*>(fec);
      rt_tracespace = dynamic_cast<const RT_Trace_FECollection*>(fec);
      tracespace = edge_tracespace || rt_tracespace;

      p = 1;
      if (fespace->GetNE() > 0)
      {
         if (!tracespace)
         {
            p = fespace->GetOrder(0);
         }
         else
         {
            p = fespace->GetFaceOrder(0);
            if (dim == 2) { p++; }
         }
      }

      if (edgespace) // H(curl)
      {
         if (dim == 2)
         {
            needint = true;
            if (tracespace)
            {
               MFEM_WARNING("Tracespace case doesn't work for H(curl) and p=2,"
                            " not using auxiliary quadrature");
               needint = false;
            }
         }
         else
         {
            FiniteElementCollection *vfec;
            if (tracespace)
            {
               vfec = new H1_Trace_FECollection(p,dim);
            }
            else
            {
               vfec = new H1_FECollection(p,dim);
            }
            ParFiniteElementSpace *vfespace = new ParFiniteElementSpace(pmesh,vfec);
            ParDiscreteLinearOperator *grad;
            grad = new ParDiscreteLinearOperator(vfespace,fespace);
            if (tracespace)
            {
               grad->AddTraceFaceInterpolator(new GradientInterpolator);
            }
            else
            {
               grad->AddDomainInterpolator(new GradientInterpolator);
            }
            grad->Assemble();
            grad->Finalize();
            HypreParMatrix *hG = grad->ParallelAssemble();
            PetscParMatrix *G = new PetscParMatrix(hG,PETSC_MATAIJ);
            delete hG;
            delete grad;

            PetscBool conforming = PETSC_TRUE;
            if (pmesh->Nonconforming()) { conforming = PETSC_FALSE; }
            ierr = PCBDDCSetDiscreteGradient(pc,*G,p,0,PETSC_TRUE,conforming);
            PCHKERRQ(pc,ierr);
            delete vfec;
            delete vfespace;
            delete G;
         }
      }
      else if (rtspace) // H(div)
      {
         needint = true;
         if (tracespace)
         {
            MFEM_WARNING("Tracespace case doesn't work for H(div), not using"
                         " auxiliary quadrature");
            needint = false;
         }
      }
      //else if (bs == dim) // Elasticity?
      //{
      //   needint = true;
      //}

      PetscParMatrix *B = NULL;
      if (needint)
      {
         // Generate bilinear form in unassembled format which is used to
         // compute the net-flux across subdomain boundaries for H(div) and
         // Elasticity, and the line integral \int u x n of 2D H(curl) fields
         FiniteElementCollection *auxcoll;
         if (tracespace) { auxcoll = new RT_Trace_FECollection(p,dim); }
         else { auxcoll = new L2_FECollection(p,dim); };
         ParFiniteElementSpace *pspace = new ParFiniteElementSpace(pmesh,auxcoll);
         ParMixedBilinearForm *b = new ParMixedBilinearForm(fespace,pspace);

         if (edgespace)
         {
            if (tracespace)
            {
               b->AddTraceFaceIntegrator(new VectorFECurlIntegrator);
            }
            else
            {
               b->AddDomainIntegrator(new VectorFECurlIntegrator);
            }
         }
         else
         {
            if (tracespace)
            {
               b->AddTraceFaceIntegrator(new VectorFEDivergenceIntegrator);
            }
            else
            {
               b->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
            }
         }
         b->Assemble();
         b->Finalize();
         OperatorHandle Bh(Operator::PETSC_MATIS);
         b->ParallelAssemble(Bh);
         Bh.Get(B);
         Bh.SetOperatorOwner(false);

         if (dir) // if essential dofs are present, we need to zero the columns
         {
            Mat pB = *B;
            ierr = MatTranspose(pB,MAT_INPLACE_MATRIX,&pB); PCHKERRQ(pA,ierr);
            if (!opts.ess_dof_local)
            {
               ierr = MatZeroRowsIS(pB,dir,0.,NULL,NULL); PCHKERRQ(pA,ierr);
            }
            else
            {
               ierr = MatZeroRowsLocalIS(pB,dir,0.,NULL,NULL); PCHKERRQ(pA,ierr);
            }
            B_is_Trans = PETSC_TRUE;
         }
         delete b;
         delete pspace;
         delete auxcoll;
      }

      if (B)
      {
         ierr = PCBDDCSetDivergenceMat(pc,*B,B_is_Trans,NULL); PCHKERRQ(pc,ierr);
      }
      delete B;
   }
   ierr = ISDestroy(&dir); PCHKERRQ(pc,ierr);
   ierr = ISDestroy(&neu); PCHKERRQ(pc,ierr);
}

PetscBDDCSolver::PetscBDDCSolver(PetscParMatrix &A,
                                 const PetscBDDCSolverParams &opts,
                                 const std::string &prefix)
   : PetscPreconditioner(A,prefix)
{
   BDDCSolverConstructor(opts);
   Customize();
}

PetscBDDCSolver::PetscBDDCSolver(MPI_Comm comm, Operator &op,
                                 const PetscBDDCSolverParams &opts,
                                 const std::string &prefix)
   : PetscPreconditioner(comm,op,prefix)
{
   BDDCSolverConstructor(opts);
   Customize();
}

PetscFieldSplitSolver::PetscFieldSplitSolver(MPI_Comm comm, Operator &op,
                                             const string &prefix)
   : PetscPreconditioner(comm,op,prefix)
{
   PC pc = (PC)obj;

   Mat pA;
   ierr = PCGetOperators(pc,&pA,NULL); PCHKERRQ(pc,ierr);

   // Check if pA is of type MATNEST
   // (this requirement can be removed when we can pass fields).
   PetscBool isnest;
   ierr = PetscObjectTypeCompare((PetscObject)pA,MATNEST,&isnest);
   PCHKERRQ(pA,ierr);
   MFEM_VERIFY(isnest,
               "PetscFieldSplitSolver needs the matrix in nested format.");

   PetscInt nr;
   IS  *isrow;
   ierr = PCSetType(pc,PCFIELDSPLIT); PCHKERRQ(pc,ierr);
   ierr = MatNestGetSize(pA,&nr,NULL); PCHKERRQ(pc,ierr);
   ierr = PetscCalloc1(nr,&isrow); CCHKERRQ(PETSC_COMM_SELF,ierr);
   ierr = MatNestGetISs(pA,isrow,NULL); PCHKERRQ(pc,ierr);

   // We need to customize here, before setting the index sets.
   Customize();

   for (PetscInt i=0; i<nr; i++)
   {
      ierr = PCFieldSplitSetIS(pc,NULL,isrow[i]); PCHKERRQ(pc,ierr);
   }
   ierr = PetscFree(isrow); CCHKERRQ(PETSC_COMM_SELF,ierr);
}

// PetscNonlinearSolver methods

PetscNonlinearSolver::PetscNonlinearSolver(MPI_Comm comm,
                                           const std::string &prefix)
   : PetscSolver(), Solver()
{
   // Create the actual solver object
   SNES snes;
   ierr = SNESCreate(comm, &snes); CCHKERRQ(comm, ierr);
   obj  = (PetscObject)snes;
   ierr = PetscObjectGetClassId(obj, &cid); PCHKERRQ(obj, ierr);
   ierr = SNESSetOptionsPrefix(snes, prefix.c_str()); PCHKERRQ(snes, ierr);

   // Allocate private solver context
   CreatePrivateContext();
}

PetscNonlinearSolver::PetscNonlinearSolver(MPI_Comm comm, Operator &op,
                                           const std::string &prefix)
   : PetscSolver(), Solver()
{
   // Create the actual solver object
   SNES snes;
   ierr = SNESCreate(comm, &snes); CCHKERRQ(comm, ierr);
   obj  = (PetscObject)snes;
   ierr = PetscObjectGetClassId(obj, &cid); PCHKERRQ(obj, ierr);
   ierr = SNESSetOptionsPrefix(snes, prefix.c_str()); PCHKERRQ(snes, ierr);

   // Allocate private solver context
   CreatePrivateContext();

   SetOperator(op);
}

PetscNonlinearSolver::~PetscNonlinearSolver()
{
   MPI_Comm comm;
   SNES snes = (SNES)obj;
   ierr = PetscObjectGetComm(obj,&comm); PCHKERRQ(obj, ierr);
   ierr = SNESDestroy(&snes); CCHKERRQ(comm, ierr);
}

void PetscNonlinearSolver::SetOperator(const Operator &op)
{
   SNES snes = (SNES)obj;

   if (operatorset)
   {
      PetscBool ls,gs;
      void     *fctx,*jctx;

      ierr = SNESGetFunction(snes, NULL, NULL, &fctx);
      PCHKERRQ(snes, ierr);
      ierr = SNESGetJacobian(snes, NULL, NULL, NULL, &jctx);
      PCHKERRQ(snes, ierr);

      ls = (PetscBool)(height == op.Height() && width  == op.Width() &&
                       (void*)&op == fctx &&
                       (void*)&op == jctx);
      ierr = MPI_Allreduce(&ls,&gs,1,MPIU_BOOL,MPI_LAND,
                           PetscObjectComm((PetscObject)snes));
      PCHKERRQ(snes,ierr);
      if (!gs)
      {
         ierr = SNESReset(snes); PCHKERRQ(snes,ierr);
         delete X;
         delete B;
         X = B = NULL;
      }
   }

   __mfem_snes_ctx *snes_ctx = (__mfem_snes_ctx*)private_ctx;
   snes_ctx->op = (mfem::Operator*)&op;
   ierr = SNESSetFunction(snes, NULL, __mfem_snes_function, (void *)snes_ctx);
   PCHKERRQ(snes, ierr);
   ierr = SNESSetJacobian(snes, NULL, NULL, __mfem_snes_jacobian,
                          (void *)snes_ctx);
   PCHKERRQ(snes, ierr);

   // Update PetscSolver
   operatorset = true;

   // Update the Operator fields.
   height = op.Height();
   width  = op.Width();
}

void PetscNonlinearSolver::SetJacobianType(Operator::Type jacType)
{
   __mfem_snes_ctx *snes_ctx = (__mfem_snes_ctx*)private_ctx;
   snes_ctx->jacType = jacType;
}

void PetscNonlinearSolver::Mult(const Vector &b, Vector &x) const
{
   SNES snes = (SNES)obj;

   bool b_nonempty = b.Size();
   if (!B) { B = new PetscParVector(PetscObjectComm(obj), *this, true); }
   if (!X) { X = new PetscParVector(PetscObjectComm(obj), *this, false, false); }
   X->PlaceArray(x.GetData());
   if (b_nonempty) { B->PlaceArray(b.GetData()); }
   else { *B = 0.0; }

   Customize();

   if (!iterative_mode) { *X = 0.; }

   if (bchandler) { bchandler->SetUp(X->Size()); }

   // Solve the system.
   ierr = SNESSolve(snes, B->x, X->x); PCHKERRQ(snes, ierr);
   X->ResetArray();
   if (b_nonempty) { B->ResetArray(); }
}

// PetscODESolver methods

PetscODESolver::PetscODESolver(MPI_Comm comm, const string &prefix)
   : PetscSolver(), ODESolver()
{
   // Create the actual solver object
   TS ts;
   ierr = TSCreate(comm,&ts); CCHKERRQ(comm,ierr);
   obj  = (PetscObject)ts;
   ierr = PetscObjectGetClassId(obj,&cid); PCHKERRQ(obj,ierr);
   ierr = TSSetOptionsPrefix(ts, prefix.c_str()); PCHKERRQ(ts, ierr);

   // Allocate private solver context
   CreatePrivateContext();

   // Default options, to comply with the current interface to ODESolver.
   ierr = TSSetMaxSteps(ts,PETSC_MAX_INT-1);
   PCHKERRQ(ts,ierr);
   ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);
   PCHKERRQ(ts,ierr);
   TSAdapt tsad;
   ierr = TSGetAdapt(ts,&tsad);
   PCHKERRQ(ts,ierr);
   ierr = TSAdaptSetType(tsad,TSADAPTNONE);
   PCHKERRQ(ts,ierr);
}

PetscODESolver::~PetscODESolver()
{
   MPI_Comm comm;
   TS ts = (TS)obj;
   ierr = PetscObjectGetComm(obj,&comm); PCHKERRQ(obj,ierr);
   ierr = TSDestroy(&ts); CCHKERRQ(comm,ierr);
}

void PetscODESolver::Init(TimeDependentOperator &f_,
                          enum PetscODESolver::Type type)
{
   TS ts = (TS)obj;

   __mfem_ts_ctx *ts_ctx = (__mfem_ts_ctx*)private_ctx;
   if (operatorset)
   {
      PetscBool ls,gs;
      void *fctx = NULL,*jctx = NULL,*rfctx = NULL,*rjctx = NULL;

      if (f->isImplicit())
      {
         ierr = TSGetIFunction(ts, NULL, NULL, &fctx);
         PCHKERRQ(ts, ierr);
         ierr = TSGetIJacobian(ts, NULL, NULL, NULL, &jctx);
         PCHKERRQ(ts, ierr);
      }
      if (!f->isHomogeneous())
      {
         ierr = TSGetRHSFunction(ts, NULL, NULL, &rfctx);
         PCHKERRQ(ts, ierr);
         ierr = TSGetRHSJacobian(ts, NULL, NULL, NULL, &rjctx);
         PCHKERRQ(ts, ierr);
      }
      ls = (PetscBool)(f->Height() == f_.Height() &&
                       f->Width() == f_.Width() &&
                       f->isExplicit() == f_.isExplicit() &&
                       f->isImplicit() == f_.isImplicit() &&
                       f->isHomogeneous() == f_.isHomogeneous());
      if (ls && f_.isImplicit())
      {
         ls = (PetscBool)(ls && (void*)&f_ == fctx && (void*)&f_ == jctx);
      }
      if (ls && !f_.isHomogeneous())
      {
         ls = (PetscBool)(ls && (void*)&f_ == rfctx && (void*)&f_ == rjctx);
      }
      ierr = MPI_Allreduce(&ls,&gs,1,MPIU_BOOL,MPI_LAND,
                           PetscObjectComm((PetscObject)ts));
      PCHKERRQ(ts,ierr);
      if (!gs)
      {
         ierr = TSReset(ts); PCHKERRQ(ts,ierr);
         delete X;
         X = NULL;
         ts_ctx->cached_shift = std::numeric_limits<PetscReal>::min();
         ts_ctx->cached_ijacstate = -1;
         ts_ctx->cached_rhsjacstate = -1;
      }
   }
   f = &f_;

   // Set functions in TS
   ts_ctx->op = &f_;
   if (f_.isImplicit())
   {
      ierr = TSSetIFunction(ts, NULL, __mfem_ts_ifunction, (void *)ts_ctx);
      PCHKERRQ(ts, ierr);
      ierr = TSSetIJacobian(ts, NULL, NULL, __mfem_ts_ijacobian, (void *)ts_ctx);
      PCHKERRQ(ts, ierr);
      ierr = TSSetEquationType(ts, TS_EQ_IMPLICIT);
      PCHKERRQ(ts, ierr);
   }
   if (!f_.isHomogeneous())
   {
      if (!f_.isImplicit())
      {
         ierr = TSSetEquationType(ts, TS_EQ_EXPLICIT);
         PCHKERRQ(ts, ierr);
      }
      ierr = TSSetRHSFunction(ts, NULL, __mfem_ts_rhsfunction, (void *)ts_ctx);
      PCHKERRQ(ts, ierr);
      ierr = TSSetRHSJacobian(ts, NULL, NULL, __mfem_ts_rhsjacobian, (void *)ts_ctx);
      PCHKERRQ(ts, ierr);
   }
   operatorset = true;

   ts_ctx->type = type;
   if (type == ODE_SOLVER_LINEAR)
   {
      ierr = TSSetProblemType(ts, TS_LINEAR);
      PCHKERRQ(ts, ierr);
   }
   else
   {
      ierr = TSSetProblemType(ts, TS_NONLINEAR);
      PCHKERRQ(ts, ierr);
   }
}

void PetscODESolver::SetJacobianType(Operator::Type jacType)
{
   __mfem_ts_ctx *ts_ctx = (__mfem_ts_ctx*)private_ctx;
   ts_ctx->jacType = jacType;
}

void PetscODESolver::Step(Vector &x, double &t, double &dt)
{
   // Pass the parameters to PETSc.
   TS ts = (TS)obj;
   ierr = TSSetTime(ts, t); PCHKERRQ(ts, ierr);
   ierr = TSSetTimeStep(ts, dt); PCHKERRQ(ts, ierr);

   if (!X) { X = new PetscParVector(PetscObjectComm(obj), *f, false, false); }
   X->PlaceArray(x.GetData());

   Customize();

   if (bchandler) { bchandler->SetUp(x.Size()); }

   // Take the step.
   ierr = TSSetSolution(ts, *X); PCHKERRQ(ts, ierr);
   ierr = TSStep(ts); PCHKERRQ(ts, ierr);
   X->ResetArray();

   // Get back current time and the time step used to caller.
   // We cannot use TSGetTimeStep() as it returns the next candidate step
   PetscReal pt;
   ierr = TSGetTime(ts,&pt); PCHKERRQ(ts,ierr);
   dt = pt - (PetscReal)t;
   t = pt;
}

void PetscODESolver::Run(Vector &x, double &t, double &dt, double t_final)
{
   // Give the parameters to PETSc.
   TS ts = (TS)obj;
   ierr = TSSetTime(ts, t); PCHKERRQ(ts, ierr);
   ierr = TSSetTimeStep(ts, dt); PCHKERRQ(ts, ierr);
   ierr = TSSetMaxTime(ts, t_final); PCHKERRQ(ts, ierr);
   ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);
   PCHKERRQ(ts, ierr);

   if (!X) { X = new PetscParVector(PetscObjectComm(obj), *f, false, false); }
   X->PlaceArray(x.GetData());

   Customize();

   if (bchandler) { bchandler->SetUp(x.Size()); }

   // Take the steps.
   ierr = TSSolve(ts, X->x); PCHKERRQ(ts, ierr);
   X->ResetArray();

   // Get back final time and time step to caller.
   PetscReal pt;
   ierr = TSGetTime(ts, &pt); PCHKERRQ(ts,ierr);
   t = pt;
   ierr = TSGetTimeStep(ts,&pt); PCHKERRQ(ts,ierr);
   dt = pt;
}

}  // namespace mfem

#include "petsc/private/petscimpl.h"

// auxiliary functions
static PetscErrorCode __mfem_ts_monitor(TS ts, PetscInt it, PetscReal t, Vec x,
                                        void* ctx)
{
   mfem::PetscSolverMonitor *monitor_ctx = (mfem::PetscSolverMonitor *)ctx;

   PetscFunctionBeginUser;
   if (!ctx)
   {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER, "No monitor context provided");
   }
   if (monitor_ctx->mon_sol)
   {
      mfem::PetscParVector V(x,true);
      monitor_ctx->MonitorSolution(it,t,V);
   }
   if (monitor_ctx->mon_res)
   {
      SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,
              "Cannot monitor the residual with TS");
   }
   PetscFunctionReturn(0);
}

static PetscErrorCode __mfem_ksp_monitor(KSP ksp, PetscInt it, PetscReal res,
                                         void* ctx)
{
   mfem::PetscSolverMonitor *monitor_ctx = (mfem::PetscSolverMonitor *)ctx;
   Vec x;
   PetscErrorCode ierr;

   PetscFunctionBeginUser;
   if (!ctx)
   {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"No monitor context provided");
   }
   if (monitor_ctx->mon_sol)
   {
      ierr = KSPBuildSolution(ksp,NULL,&x); CHKERRQ(ierr);
      mfem::PetscParVector V(x,true);
      monitor_ctx->MonitorSolution(it,res,V);
   }
   if (monitor_ctx->mon_res)
   {
      ierr = KSPBuildResidual(ksp,NULL,NULL,&x); CHKERRQ(ierr);
      mfem::PetscParVector V(x,true);
      monitor_ctx->MonitorResidual(it,res,V);
   }
   PetscFunctionReturn(0);
}

static PetscErrorCode __mfem_ts_ifunction(TS ts, PetscReal t, Vec x, Vec xp,
                                          Vec f,void *ctx)
{
   __mfem_ts_ctx* ts_ctx = (__mfem_ts_ctx*)ctx;
   PetscErrorCode ierr;

   PetscFunctionBeginUser;
   mfem::PetscParVector xx(x,true);
   mfem::PetscParVector yy(xp,true);
   mfem::PetscParVector ff(f,true);

   mfem::TimeDependentOperator *op = ts_ctx->op;
   op->SetTime(t);

   if (ts_ctx->bchandler)
   {
      // we evaluate the ImplicitMult method with the correct bc
      if (!ts_ctx->work) { ts_ctx->work = new mfem::Vector(xx.Size()); }
      mfem::PetscBCHandler *bchandler = ts_ctx->bchandler;
      mfem::Vector* txx = ts_ctx->work;
      bchandler->SetTime(t);
      bchandler->ApplyBC(xx,*txx);
      op->ImplicitMult(*txx,yy,ff);
      // and fix the residual (i.e. f_\partial\Omega = u - g(t))
      bchandler->FixResidualBC(xx,ff);
   }
   else
   {
      // use the ImplicitMult method of the class
      op->ImplicitMult(xx,yy,ff);
   }

   // need to tell PETSc the Vec has been updated
   ierr = PetscObjectStateIncrease((PetscObject)f); CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

static PetscErrorCode __mfem_ts_rhsfunction(TS ts, PetscReal t, Vec x, Vec f,
                                            void *ctx)
{
   __mfem_ts_ctx* ts_ctx = (__mfem_ts_ctx*)ctx;
   PetscErrorCode ierr;

   PetscFunctionBeginUser;
   if (ts_ctx->bchandler) { MFEM_ABORT("RHS evaluation with bc not implemented"); } // TODO
   mfem::PetscParVector xx(x,true);
   mfem::PetscParVector ff(f,true);
   mfem::TimeDependentOperator *top = ts_ctx->op;
   top->SetTime(t);

   // use the ExplicitMult method - compute the RHS function
   top->ExplicitMult(xx,ff);

   // need to tell PETSc the Vec has been updated
   ierr = PetscObjectStateIncrease((PetscObject)f); CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

static PetscErrorCode __mfem_ts_ijacobian(TS ts, PetscReal t, Vec x,
                                          Vec xp, PetscReal shift, Mat A, Mat P,
                                          void *ctx)
{
   __mfem_ts_ctx*   ts_ctx = (__mfem_ts_ctx*)ctx;
   mfem::Vector     *xx;
   PetscScalar      *array;
   PetscReal        eps = 0.001; /* 0.1% difference */
   PetscInt         n;
   PetscObjectState state;
   PetscErrorCode   ierr;

   PetscFunctionBeginUser;
   // update time
   mfem::TimeDependentOperator *op = ts_ctx->op;
   op->SetTime(t);

   // prevent to recompute a Jacobian if we already did so
   // the relative tolerance comparison should be fine given the fact
   // that two consecutive shifts should have similar magnitude
   ierr = PetscObjectStateGet((PetscObject)A,&state); CHKERRQ(ierr);
   if (ts_ctx->type == mfem::PetscODESolver::ODE_SOLVER_LINEAR &&
       std::abs(ts_ctx->cached_shift/shift - 1.0) < eps &&
       state == ts_ctx->cached_ijacstate) { PetscFunctionReturn(0); }

   // wrap Vecs with Vectors
   ierr = VecGetLocalSize(x,&n); CHKERRQ(ierr);
   ierr = VecGetArrayRead(xp,(const PetscScalar**)&array); CHKERRQ(ierr);
   mfem::Vector yy(array,n);
   ierr = VecRestoreArrayRead(xp,(const PetscScalar**)&array); CHKERRQ(ierr);
   ierr = VecGetArrayRead(x,(const PetscScalar**)&array); CHKERRQ(ierr);
   if (!ts_ctx->bchandler)
   {
      xx = new mfem::Vector(array,n);
   }
   else
   {
      // make sure we compute a Jacobian with the correct boundary values
      if (!ts_ctx->work) { ts_ctx->work = new mfem::Vector(n); }
      mfem::Vector txx(array,n);
      mfem::PetscBCHandler *bchandler = ts_ctx->bchandler;
      xx = ts_ctx->work;
      bchandler->SetTime(t);
      bchandler->ApplyBC(txx,*xx);
   }
   ierr = VecRestoreArrayRead(x,(const PetscScalar**)&array); CHKERRQ(ierr);

   // Use TimeDependentOperator::GetImplicitGradient(x,y,s)
   mfem::Operator& J = op->GetImplicitGradient(*xx,yy,shift);
   if (!ts_ctx->bchandler) { delete xx; }
   ts_ctx->cached_shift = shift;

   // Convert to the operator type requested if needed
   bool delete_pA = false;
   mfem::PetscParMatrix *pA = const_cast<mfem::PetscParMatrix *>
                              (dynamic_cast<const mfem::PetscParMatrix *>(&J));
   if (!pA || pA->GetType() != ts_ctx->jacType)
   {
      pA = new mfem::PetscParMatrix(PetscObjectComm((PetscObject)ts),&J,
                                    ts_ctx->jacType);
      delete_pA = true;
   }

   // Eliminate essential dofs
   if (ts_ctx->bchandler)
   {
      mfem::PetscBCHandler *bchandler = ts_ctx->bchandler;
      mfem::PetscParVector dummy(PetscObjectComm((PetscObject)ts),0);
      pA->EliminateRowsCols(bchandler->GetTDofs(),dummy,dummy);
   }

   // Avoid unneeded copy of the matrix by hacking
   Mat B;
   B = pA->ReleaseMat(false);
   ierr = MatHeaderReplace(A,&B); CHKERRQ(ierr);
   if (delete_pA) { delete pA; }

   // Jacobian reusage
   ierr = PetscObjectStateGet((PetscObject)A,&ts_ctx->cached_ijacstate);
   CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

static PetscErrorCode __mfem_ts_rhsjacobian(TS ts, PetscReal t, Vec x,
                                            Mat A, Mat P, void *ctx)
{
   __mfem_ts_ctx*   ts_ctx = (__mfem_ts_ctx*)ctx;
   mfem::Vector     *xx;
   PetscScalar      *array;
   PetscInt         n;
   PetscObjectState state;
   PetscErrorCode   ierr;

   PetscFunctionBeginUser;
   // update time
   mfem::TimeDependentOperator *op = ts_ctx->op;
   op->SetTime(t);

   // prevent to recompute a Jacobian if we already did so
   ierr = PetscObjectStateGet((PetscObject)A,&state); CHKERRQ(ierr);
   if (ts_ctx->type == mfem::PetscODESolver::ODE_SOLVER_LINEAR &&
       state == ts_ctx->cached_rhsjacstate) { PetscFunctionReturn(0); }

   // wrap Vec with Vector
   ierr = VecGetLocalSize(x,&n); CHKERRQ(ierr);
   ierr = VecGetArrayRead(x,(const PetscScalar**)&array); CHKERRQ(ierr);
   if (!ts_ctx->bchandler)
   {
      xx = new mfem::Vector(array,n);
   }
   else
   {
      // make sure we compute a Jacobian with the correct boundary values
      if (!ts_ctx->work) { ts_ctx->work = new mfem::Vector(n); }
      mfem::Vector txx(array,n);
      mfem::PetscBCHandler *bchandler = ts_ctx->bchandler;
      xx = ts_ctx->work;
      bchandler->SetTime(t);
      bchandler->ApplyBC(txx,*xx);
   }
   ierr = VecRestoreArrayRead(x,(const PetscScalar**)&array); CHKERRQ(ierr);

   // Use TimeDependentOperator::GetExplicitGradient(x)
   mfem::Operator& J = op->GetExplicitGradient(*xx);
   if (!ts_ctx->bchandler) { delete xx; }

   // Convert to the operator type requested if needed
   bool delete_pA = false;
   mfem::PetscParMatrix *pA = const_cast<mfem::PetscParMatrix *>
                              (dynamic_cast<const mfem::PetscParMatrix *>(&J));
   if (!pA || pA->GetType() != ts_ctx->jacType)
   {
      pA = new mfem::PetscParMatrix(PetscObjectComm((PetscObject)ts),&J,
                                    ts_ctx->jacType);
      delete_pA = true;
   }

   // Eliminate essential dofs
   if (ts_ctx->bchandler)
   {
      mfem::PetscBCHandler *bchandler = ts_ctx->bchandler;
      mfem::PetscParVector dummy(PetscObjectComm((PetscObject)ts),0);
      pA->EliminateRowsCols(bchandler->GetTDofs(),dummy,dummy);
   }

   // Avoid unneeded copy of the matrix by hacking
   Mat B;
   B = pA->ReleaseMat(false);
   ierr = MatHeaderReplace(A,&B); CHKERRQ(ierr);
   if (delete_pA) { delete pA; }

   // Jacobian reusage
   if (ts_ctx->type == mfem::PetscODESolver::ODE_SOLVER_LINEAR)
   {
      ierr = TSRHSJacobianSetReuse(ts,PETSC_TRUE); PCHKERRQ(ts,ierr);
   }
   ierr = PetscObjectStateGet((PetscObject)A,&ts_ctx->cached_rhsjacstate);
   CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

static PetscErrorCode __mfem_snes_monitor(SNES snes, PetscInt it, PetscReal res,
                                          void* ctx)
{
   mfem::PetscSolverMonitor *monitor_ctx = (mfem::PetscSolverMonitor *)ctx;
   Vec x;
   PetscErrorCode ierr;

   PetscFunctionBeginUser;
   if (!ctx)
   {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"No monitor context provided");
   }
   if (monitor_ctx->mon_sol)
   {
      ierr = SNESGetSolution(snes,&x); CHKERRQ(ierr);
      mfem::PetscParVector V(x,true);
      monitor_ctx->MonitorSolution(it,res,V);
   }
   if (monitor_ctx->mon_res)
   {
      ierr = SNESGetFunction(snes,&x,NULL,NULL); CHKERRQ(ierr);
      mfem::PetscParVector V(x,true);
      monitor_ctx->MonitorResidual(it,res,V);
   }
   PetscFunctionReturn(0);
}

static PetscErrorCode __mfem_snes_jacobian(SNES snes, Vec x, Mat A, Mat P,
                                           void *ctx)
{
   PetscScalar     *array;
   PetscInt         n;
   PetscErrorCode   ierr;
   mfem::Vector    *xx;
   __mfem_snes_ctx *snes_ctx = (__mfem_snes_ctx*)ctx;

   PetscFunctionBeginUser;
   ierr = VecGetArrayRead(x,(const PetscScalar**)&array); CHKERRQ(ierr);
   ierr = VecGetLocalSize(x,&n); CHKERRQ(ierr);
   if (!snes_ctx->bchandler)
   {
      xx = new mfem::Vector(array,n);
   }
   else
   {
      // make sure we compute a Jacobian with the correct boundary values
      if (!snes_ctx->work) { snes_ctx->work = new mfem::Vector(n); }
      mfem::Vector txx(array,n);
      mfem::PetscBCHandler *bchandler = snes_ctx->bchandler;
      xx = snes_ctx->work;
      bchandler->ApplyBC(txx,*xx);
   }

   // Use Operator::GetGradient(x)
   mfem::Operator& J = snes_ctx->op->GetGradient(*xx);
   ierr = VecRestoreArrayRead(x,(const PetscScalar**)&array); CHKERRQ(ierr);
   if (!snes_ctx->bchandler) { delete xx; }

   // Convert to the operator type requested if needed
   bool delete_pA = false;
   mfem::PetscParMatrix *pA = const_cast<mfem::PetscParMatrix *>
                              (dynamic_cast<const mfem::PetscParMatrix *>(&J));
   if (!pA || pA->GetType() != snes_ctx->jacType)
   {
      pA = new mfem::PetscParMatrix(PetscObjectComm((PetscObject)snes),&J,
                                    snes_ctx->jacType);
      delete_pA = true;
   }

   // Eliminate essential dofs
   if (snes_ctx->bchandler)
   {
      mfem::PetscBCHandler *bchandler = snes_ctx->bchandler;
      mfem::PetscParVector dummy(PetscObjectComm((PetscObject)snes),0);
      pA->EliminateRowsCols(bchandler->GetTDofs(),dummy,dummy);
   }

   // Avoid unneeded copy of the matrix by hacking
   Mat B = pA->ReleaseMat(false);
   ierr = MatHeaderReplace(A,&B); CHKERRQ(ierr);
   if (delete_pA) { delete pA; }
   PetscFunctionReturn(0);
}

static PetscErrorCode __mfem_snes_function(SNES snes, Vec x, Vec f, void *ctx)
{
   __mfem_snes_ctx* snes_ctx = (__mfem_snes_ctx*)ctx;

   PetscFunctionBeginUser;
   mfem::PetscParVector xx(x,true);
   mfem::PetscParVector ff(f,true);
   if (snes_ctx->bchandler)
   {
      // we evaluate the Mult method with the correct bc
      if (!snes_ctx->work) { snes_ctx->work = new mfem::Vector(xx.Size()); }
      mfem::PetscBCHandler *bchandler = snes_ctx->bchandler;
      mfem::Vector* txx = snes_ctx->work;
      bchandler->ApplyBC(xx,*txx);
      snes_ctx->op->Mult(*txx,ff);
      // and fix the residual (i.e. f_\partial\Omega = u - g)
      bchandler->FixResidualBC(xx,ff);
   }
   else
   {
      // use the Mult method of the class
      snes_ctx->op->Mult(xx,ff);
   }
   // need to tell PETSc the Vec has been updated
   ierr = PetscObjectStateIncrease((PetscObject)f); CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

static PetscErrorCode __mfem_mat_shell_apply(Mat A, Vec x, Vec y)
{
   __mfem_mat_shell_ctx *ctx;
   PetscErrorCode       ierr;

   PetscFunctionBeginUser;
   ierr = MatShellGetContext(A,(void **)&ctx); CHKERRQ(ierr);
   mfem::PetscParVector xx(x,true);
   mfem::PetscParVector yy(y,true);
   ctx->op->Mult(xx,yy);
   // need to tell PETSc the Vec has been updated
   ierr = PetscObjectStateIncrease((PetscObject)y); CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

static PetscErrorCode __mfem_mat_shell_apply_transpose(Mat A, Vec x, Vec y)
{
   __mfem_mat_shell_ctx *ctx;
   PetscErrorCode       ierr;

   PetscFunctionBeginUser;
   ierr = MatShellGetContext(A,(void **)&ctx); CHKERRQ(ierr);
   mfem::PetscParVector xx(x,true);
   mfem::PetscParVector yy(y,true);
   ctx->op->MultTranspose(xx,yy);
   // need to tell PETSc the Vec has been updated
   ierr = PetscObjectStateIncrease((PetscObject)y); CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

static PetscErrorCode __mfem_mat_shell_destroy(Mat A)
{
   __mfem_mat_shell_ctx *ctx;
   PetscErrorCode       ierr;

   PetscFunctionBeginUser;
   ierr = MatShellGetContext(A,(void **)&ctx); CHKERRQ(ierr);
   delete ctx;
   PetscFunctionReturn(0);
}

static PetscErrorCode __mfem_pc_shell_view(PC pc, PetscViewer viewer)
{
   __mfem_pc_shell_ctx *ctx;
   PetscErrorCode      ierr;

   PetscFunctionBeginUser;
   ierr = PCShellGetContext(pc,(void **)&ctx); CHKERRQ(ierr);
   if (ctx->op)
   {
      PetscBool isascii;
      ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);
      CHKERRQ(ierr);

      mfem::PetscPreconditioner *ppc = dynamic_cast<mfem::PetscPreconditioner *>
                                       (ctx->op);
      if (ppc)
      {
         ierr = PCView(*ppc,viewer); CHKERRQ(ierr);
      }
      else
      {
         if (isascii)
         {
            ierr = PetscViewerASCIIPrintf(viewer,
                                          "No information available on the mfem::Solver\n");
            CHKERRQ(ierr);
         }
      }
      if (isascii && ctx->factory)
      {
         ierr = PetscViewerASCIIPrintf(viewer,
                                       "Number of preconditioners created by the factory %lu\n",ctx->numprec);
         CHKERRQ(ierr);
      }
   }
   PetscFunctionReturn(0);
}

static PetscErrorCode __mfem_pc_shell_apply(PC pc, Vec x, Vec y)
{
   __mfem_pc_shell_ctx *ctx;
   PetscErrorCode      ierr;

   PetscFunctionBeginUser;
   mfem::PetscParVector xx(x,true);
   mfem::PetscParVector yy(y,true);
   ierr = PCShellGetContext(pc,(void **)&ctx); CHKERRQ(ierr);
   if (ctx->op)
   {
      ctx->op->Mult(xx,yy);
      // need to tell PETSc the Vec has been updated
      ierr = PetscObjectStateIncrease((PetscObject)y); CHKERRQ(ierr);
   }
   else // operator is not present, copy x
   {
      yy = xx;
   }
   PetscFunctionReturn(0);
}

static PetscErrorCode __mfem_pc_shell_apply_transpose(PC pc, Vec x, Vec y)
{
   __mfem_pc_shell_ctx *ctx;
   PetscErrorCode      ierr;

   PetscFunctionBeginUser;
   mfem::PetscParVector xx(x,true);
   mfem::PetscParVector yy(y,true);
   ierr = PCShellGetContext(pc,(void **)&ctx); CHKERRQ(ierr);
   if (ctx->op)
   {
      ctx->op->MultTranspose(xx,yy);
      // need to tell PETSc the Vec has been updated
      ierr = PetscObjectStateIncrease((PetscObject)y); CHKERRQ(ierr);
   }
   else // operator is not present, copy x
   {
      yy = xx;
   }
   PetscFunctionReturn(0);
}

static PetscErrorCode __mfem_pc_shell_setup(PC pc)
{
   __mfem_pc_shell_ctx *ctx;

   PetscFunctionBeginUser;
   ierr = PCShellGetContext(pc,(void **)&ctx); CHKERRQ(ierr);
   if (ctx->factory)
   {
      // Delete any owned operator
      if (ctx->ownsop)
      {
         delete ctx->op;
      }

      // Get current preconditioning Mat
      Mat B;
      ierr = PCGetOperators(pc,NULL,&B); CHKERRQ(ierr);

      // Call user-defined setup
      mfem::OperatorHandle hB(new mfem::PetscParMatrix(B,true),true);
      mfem::PetscPreconditionerFactory *factory = ctx->factory;
      ctx->op = factory->NewPreconditioner(hB);
      ctx->ownsop = true;
      ctx->numprec++;
   }
   PetscFunctionReturn(0);
}

static PetscErrorCode __mfem_pc_shell_destroy(PC pc)
{
   __mfem_pc_shell_ctx *ctx;
   PetscErrorCode      ierr;

   PetscFunctionBeginUser;
   ierr = PCShellGetContext(pc,(void **)&ctx); CHKERRQ(ierr);
   if (ctx->ownsop)
   {
      delete ctx->op;
   }
   delete ctx;
   PetscFunctionReturn(0);
}

static PetscErrorCode __mfem_array_container_destroy(void *ptr)
{
   PetscErrorCode ierr;

   PetscFunctionBeginUser;
   ierr = PetscFree(ptr); CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

static PetscErrorCode __mfem_matarray_container_destroy(void *ptr)
{
   mfem::Array<Mat> *a = (mfem::Array<Mat>*)ptr;
   PetscErrorCode   ierr;

   PetscFunctionBeginUser;
   for (int i=0; i<a->Size(); i++)
   {
      Mat M = (*a)[i];
      MPI_Comm comm = PetscObjectComm((PetscObject)M);
      ierr = MatDestroy(&M); CCHKERRQ(comm,ierr);
   }
   delete a;
   PetscFunctionReturn(0);
}

// Sets the type of PC to PCSHELL and wraps the solver action
// if ownsop is true, ownership of precond is transferred to the PETSc object
PetscErrorCode MakeShellPC(PC pc, mfem::Solver &precond, bool ownsop)
{
   PetscFunctionBeginUser;
   __mfem_pc_shell_ctx *ctx = new __mfem_pc_shell_ctx;
   ctx->op      = &precond;
   ctx->ownsop  = ownsop;
   ctx->factory = NULL;
   ctx->numprec = 0;

   ierr = PCSetType(pc,PCSHELL); CHKERRQ(ierr);
   ierr = PCShellSetName(pc,"MFEM Solver"); CHKERRQ(ierr);
   ierr = PCShellSetContext(pc,(void *)ctx); CHKERRQ(ierr);
   ierr = PCShellSetApply(pc,__mfem_pc_shell_apply); CHKERRQ(ierr);
   ierr = PCShellSetApplyTranspose(pc,__mfem_pc_shell_apply_transpose);
   CHKERRQ(ierr);
   ierr = PCShellSetSetUp(pc,__mfem_pc_shell_setup); CHKERRQ(ierr);
   ierr = PCShellSetView(pc,__mfem_pc_shell_view); CHKERRQ(ierr);
   ierr = PCShellSetDestroy(pc,__mfem_pc_shell_destroy); CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

// Sets the type of PC to PCSHELL. Uses a PetscPreconditionerFactory to construct the solver
// Takes ownership of the solver created by the factory
PetscErrorCode MakeShellPCWithFactory(PC pc,
                                      mfem::PetscPreconditionerFactory *factory)
{
   PetscFunctionBeginUser;
   __mfem_pc_shell_ctx *ctx = new __mfem_pc_shell_ctx;
   ctx->op      = NULL;
   ctx->ownsop  = true;
   ctx->factory = factory;
   ctx->numprec = 0;

   ierr = PCSetType(pc,PCSHELL); CHKERRQ(ierr);
   ierr = PCShellSetName(pc,factory->GetName()); CHKERRQ(ierr);
   ierr = PCShellSetContext(pc,(void *)ctx); CHKERRQ(ierr);
   ierr = PCShellSetApply(pc,__mfem_pc_shell_apply); CHKERRQ(ierr);
   ierr = PCShellSetApplyTranspose(pc,__mfem_pc_shell_apply_transpose);
   CHKERRQ(ierr);
   ierr = PCShellSetSetUp(pc,__mfem_pc_shell_setup); CHKERRQ(ierr);
   ierr = PCShellSetView(pc,__mfem_pc_shell_view); CHKERRQ(ierr);
   ierr = PCShellSetDestroy(pc,__mfem_pc_shell_destroy); CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

// Converts from a list (or a marked Array if islist is false) to an IS
// st indicates the offset where to start numbering
static PetscErrorCode Convert_Array_IS(MPI_Comm comm, bool islist,
                                       const mfem::Array<int> *list,
                                       PetscInt st, IS* is)
{
   PetscInt       n = list->Size(),*idxs;
   const int      *data = list->GetData();
   PetscErrorCode ierr;

   PetscFunctionBeginUser;
   ierr = PetscMalloc1(n,&idxs); CHKERRQ(ierr);
   if (islist)
   {
      for (PetscInt i=0; i<n; i++) { idxs[i] = data[i] + st; }
   }
   else
   {
      PetscInt cum = 0;
      for (PetscInt i=0; i<n; i++)
      {
         if (data[i]) { idxs[cum++] = i+st; }
      }
      n = cum;
   }
   ierr = ISCreateGeneral(comm,n,idxs,PETSC_OWN_POINTER,is);
   CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

// Converts from a marked Array of Vdofs to an IS
// st indicates the offset where to start numbering
// l2l is a vector of matrices generated during RAP
static PetscErrorCode Convert_Vmarks_IS(MPI_Comm comm,
                                        mfem::Array<Mat> &pl2l,
                                        const mfem::Array<int> *mark,
                                        PetscInt st, IS* is)
{
   mfem::Array<int> sub_dof_marker;
   mfem::Array<mfem::SparseMatrix*> l2l(pl2l.Size());
   PetscInt         nl;
   PetscErrorCode   ierr;

   PetscFunctionBeginUser;
   for (int i = 0; i < pl2l.Size(); i++)
   {
      PetscInt  m,n,*ii,*jj;
      PetscBool done;
      ierr = MatGetRowIJ(pl2l[i],0,PETSC_FALSE,PETSC_FALSE,&m,(const PetscInt**)&ii,
                         (const PetscInt**)&jj,&done); CHKERRQ(ierr);
      MFEM_VERIFY(done,"Unable to perform MatGetRowIJ on " << i << " l2l matrix");
      ierr = MatGetSize(pl2l[i],NULL,&n); CHKERRQ(ierr);
      l2l[i] = new mfem::SparseMatrix(ii,jj,NULL,m,n,false,true,true);
   }
   nl = 0;
   for (int i = 0; i < l2l.Size(); i++) { nl += l2l[i]->Width(); }
   sub_dof_marker.SetSize(nl);
   const int* vdata = mark->GetData();
   int* sdata = sub_dof_marker.GetData();
   int cumh = 0, cumw = 0;
   for (int i = 0; i < l2l.Size(); i++)
   {
      const mfem::Array<int> vf_marker(const_cast<int*>(vdata)+cumh,
                                       l2l[i]->Height());
      mfem::Array<int> sf_marker(sdata+cumw,l2l[i]->Width());
      l2l[i]->BooleanMultTranspose(vf_marker,sf_marker);
      cumh += l2l[i]->Height();
      cumw += l2l[i]->Width();
   }
   ierr = Convert_Array_IS(comm,false,&sub_dof_marker,st,is); CCHKERRQ(comm,ierr);
   for (int i = 0; i < pl2l.Size(); i++)
   {
      PetscInt  m = l2l[i]->Height();
      PetscInt  *ii = l2l[i]->GetI(),*jj = l2l[i]->GetJ();
      PetscBool done;
      ierr = MatRestoreRowIJ(pl2l[i],0,PETSC_FALSE,PETSC_FALSE,&m,
                             (const PetscInt**)&ii,
                             (const PetscInt**)&jj,&done); CHKERRQ(ierr);
      MFEM_VERIFY(done,"Unable to perform MatRestoreRowIJ on "
                  << i << " l2l matrix");
      delete l2l[i];
   }
   PetscFunctionReturn(0);
}

#if !defined(PETSC_HAVE_HYPRE)

#include "_hypre_parcsr_mv.h"
static PetscErrorCode MatConvert_hypreParCSR_AIJ(hypre_ParCSRMatrix* hA,Mat* pA)
{
   MPI_Comm        comm = hypre_ParCSRMatrixComm(hA);
   hypre_CSRMatrix *hdiag,*hoffd;
   PetscScalar     *da,*oa,*aptr;
   PetscInt        *dii,*djj,*oii,*ojj,*iptr;
   PetscInt        i,dnnz,onnz,m,n;
   PetscMPIInt     size;
   PetscErrorCode  ierr;

   PetscFunctionBeginUser;
   hdiag = hypre_ParCSRMatrixDiag(hA);
   hoffd = hypre_ParCSRMatrixOffd(hA);
   m     = hypre_CSRMatrixNumRows(hdiag);
   n     = hypre_CSRMatrixNumCols(hdiag);
   dnnz  = hypre_CSRMatrixNumNonzeros(hdiag);
   onnz  = hypre_CSRMatrixNumNonzeros(hoffd);
   ierr  = PetscMalloc1(m+1,&dii); CHKERRQ(ierr);
   ierr  = PetscMalloc1(dnnz,&djj); CHKERRQ(ierr);
   ierr  = PetscMalloc1(dnnz,&da); CHKERRQ(ierr);
   ierr  = PetscMemcpy(dii,hypre_CSRMatrixI(hdiag),(m+1)*sizeof(PetscInt));
   CHKERRQ(ierr);
   ierr  = PetscMemcpy(djj,hypre_CSRMatrixJ(hdiag),dnnz*sizeof(PetscInt));
   CHKERRQ(ierr);
   ierr  = PetscMemcpy(da,hypre_CSRMatrixData(hdiag),dnnz*sizeof(PetscScalar));
   CHKERRQ(ierr);
   iptr  = djj;
   aptr  = da;
   for (i=0; i<m; i++)
   {
      PetscInt nc = dii[i+1]-dii[i];
      ierr = PetscSortIntWithScalarArray(nc,iptr,aptr); CHKERRQ(ierr);
      iptr += nc;
      aptr += nc;
   }
   ierr = MPI_Comm_size(comm,&size); CHKERRQ(ierr);
   if (size > 1)
   {
      PetscInt *offdj,*coffd;

      ierr  = PetscMalloc1(m+1,&oii); CHKERRQ(ierr);
      ierr  = PetscMalloc1(onnz,&ojj); CHKERRQ(ierr);
      ierr  = PetscMalloc1(onnz,&oa); CHKERRQ(ierr);
      ierr  = PetscMemcpy(oii,hypre_CSRMatrixI(hoffd),(m+1)*sizeof(PetscInt));
      CHKERRQ(ierr);
      offdj = hypre_CSRMatrixJ(hoffd);
      coffd = hypre_ParCSRMatrixColMapOffd(hA);
      for (i=0; i<onnz; i++) { ojj[i] = coffd[offdj[i]]; }
      ierr  = PetscMemcpy(oa,hypre_CSRMatrixData(hoffd),onnz*sizeof(PetscScalar));
      CHKERRQ(ierr);
      iptr  = ojj;
      aptr  = oa;
      for (i=0; i<m; i++)
      {
         PetscInt nc = oii[i+1]-oii[i];
         ierr = PetscSortIntWithScalarArray(nc,iptr,aptr); CHKERRQ(ierr);
         iptr += nc;
         aptr += nc;
      }
      ierr = MatCreateMPIAIJWithSplitArrays(comm,m,n,PETSC_DECIDE,PETSC_DECIDE,dii,
                                            djj,da,oii,ojj,oa,pA); CHKERRQ(ierr);
   }
   else
   {
      oii = ojj = NULL;
      oa = NULL;
      ierr = MatCreateSeqAIJWithArrays(comm,m,n,dii,djj,da,pA); CHKERRQ(ierr);
   }
   /* We are responsible to free the CSR arrays.  However, since we can take
      references of a PetscParMatrix but we cannot take reference of PETSc
      arrays, we need to create a PetscContainer object to take reference of
      these arrays in reference objects */
   void *ptrs[6] = {dii,djj,da,oii,ojj,oa};
   const char *names[6] = {"_mfem_csr_dii",
                           "_mfem_csr_djj",
                           "_mfem_csr_da",
                           "_mfem_csr_oii",
                           "_mfem_csr_ojj",
                           "_mfem_csr_oa"
                          };
   for (i=0; i<6; i++)
   {
      PetscContainer c;

      ierr = PetscContainerCreate(comm,&c); CHKERRQ(ierr);
      ierr = PetscContainerSetPointer(c,ptrs[i]); CHKERRQ(ierr);
      ierr = PetscContainerSetUserDestroy(c,__mfem_array_container_destroy);
      CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)(*pA),names[i],(PetscObject)c);
      CHKERRQ(ierr);
      ierr = PetscContainerDestroy(&c); CHKERRQ(ierr);
   }
   PetscFunctionReturn(0);
}

static PetscErrorCode MatConvert_hypreParCSR_IS(hypre_ParCSRMatrix* hA,Mat* pA)
{
   Mat                    lA;
   ISLocalToGlobalMapping rl2g,cl2g;
   IS                     is;
   hypre_CSRMatrix        *hdiag,*hoffd;
   MPI_Comm               comm = hypre_ParCSRMatrixComm(hA);
   void                   *ptrs[2];
   const char             *names[2] = {"_mfem_csr_aux",
                                       "_mfem_csr_data"
                                      };
   PetscScalar            *hdd,*hod,*aa,*data;
   PetscInt               *col_map_offd,*hdi,*hdj,*hoi,*hoj;
   PetscInt               *aux,*ii,*jj;
   PetscInt               cum,dr,dc,oc,str,stc,nnz,i,jd,jo;
   PetscErrorCode         ierr;

   PetscFunctionBeginUser;
   /* access relevant information in ParCSR */
   str   = hypre_ParCSRMatrixFirstRowIndex(hA);
   stc   = hypre_ParCSRMatrixFirstColDiag(hA);
   hdiag = hypre_ParCSRMatrixDiag(hA);
   hoffd = hypre_ParCSRMatrixOffd(hA);
   dr    = hypre_CSRMatrixNumRows(hdiag);
   dc    = hypre_CSRMatrixNumCols(hdiag);
   nnz   = hypre_CSRMatrixNumNonzeros(hdiag);
   hdi   = hypre_CSRMatrixI(hdiag);
   hdj   = hypre_CSRMatrixJ(hdiag);
   hdd   = hypre_CSRMatrixData(hdiag);
   oc    = hypre_CSRMatrixNumCols(hoffd);
   nnz  += hypre_CSRMatrixNumNonzeros(hoffd);
   hoi   = hypre_CSRMatrixI(hoffd);
   hoj   = hypre_CSRMatrixJ(hoffd);
   hod   = hypre_CSRMatrixData(hoffd);

   /* generate l2g maps for rows and cols */
   ierr = ISCreateStride(comm,dr,str,1,&is); CHKERRQ(ierr);
   ierr = ISLocalToGlobalMappingCreateIS(is,&rl2g); CHKERRQ(ierr);
   ierr = ISDestroy(&is); CHKERRQ(ierr);
   col_map_offd = hypre_ParCSRMatrixColMapOffd(hA);
   ierr = PetscMalloc1(dc+oc,&aux); CHKERRQ(ierr);
   for (i=0; i<dc; i++) { aux[i]    = i+stc; }
   for (i=0; i<oc; i++) { aux[i+dc] = col_map_offd[i]; }
   ierr = ISCreateGeneral(comm,dc+oc,aux,PETSC_OWN_POINTER,&is); CHKERRQ(ierr);
   ierr = ISLocalToGlobalMappingCreateIS(is,&cl2g); CHKERRQ(ierr);
   ierr = ISDestroy(&is); CHKERRQ(ierr);

   /* create MATIS object */
   ierr = MatCreate(comm,pA); CHKERRQ(ierr);
   ierr = MatSetSizes(*pA,dr,dc,PETSC_DECIDE,PETSC_DECIDE); CHKERRQ(ierr);
   ierr = MatSetType(*pA,MATIS); CHKERRQ(ierr);
   ierr = MatSetLocalToGlobalMapping(*pA,rl2g,cl2g); CHKERRQ(ierr);
   ierr = ISLocalToGlobalMappingDestroy(&rl2g); CHKERRQ(ierr);
   ierr = ISLocalToGlobalMappingDestroy(&cl2g); CHKERRQ(ierr);

   /* merge local matrices */
   ierr = PetscMalloc1(nnz+dr+1,&aux); CHKERRQ(ierr);
   ierr = PetscMalloc1(nnz,&data); CHKERRQ(ierr);
   ii   = aux;
   jj   = aux+dr+1;
   aa   = data;
   *ii  = *(hdi++) + *(hoi++);
   for (jd=0,jo=0,cum=0; *ii<nnz; cum++)
   {
      PetscScalar *aold = aa;
      PetscInt    *jold = jj,nc = jd+jo;
      for (; jd<*hdi; jd++) { *jj++ = *hdj++;      *aa++ = *hdd++; }
      for (; jo<*hoi; jo++) { *jj++ = *hoj++ + dc; *aa++ = *hod++; }
      *(++ii) = *(hdi++) + *(hoi++);
      ierr = PetscSortIntWithScalarArray(jd+jo-nc,jold,aold); CHKERRQ(ierr);
   }
   for (; cum<dr; cum++) { *(++ii) = nnz; }
   ii   = aux;
   jj   = aux+dr+1;
   aa   = data;
   ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,dr,dc+oc,ii,jj,aa,&lA);
   CHKERRQ(ierr);
   ptrs[0] = aux;
   ptrs[1] = data;
   for (i=0; i<2; i++)
   {
      PetscContainer c;

      ierr = PetscContainerCreate(PETSC_COMM_SELF,&c); CHKERRQ(ierr);
      ierr = PetscContainerSetPointer(c,ptrs[i]); CHKERRQ(ierr);
      ierr = PetscContainerSetUserDestroy(c,__mfem_array_container_destroy);
      CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)lA,names[i],(PetscObject)c);
      CHKERRQ(ierr);
      ierr = PetscContainerDestroy(&c); CHKERRQ(ierr);
   }
   ierr = MatISSetLocalMat(*pA,lA); CHKERRQ(ierr);
   ierr = MatDestroy(&lA); CHKERRQ(ierr);
   ierr = MatAssemblyBegin(*pA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
   ierr = MatAssemblyEnd(*pA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
   PetscFunctionReturn(0);
}
#endif

#endif  // MFEM_USE_PETSC
#endif  // MFEM_USE_MPI
