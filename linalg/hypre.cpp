// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifdef MFEM_USE_MPI

#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <stdlib.h>

#include "linalg.hpp"

HypreParVector::HypreParVector(int glob_size, int *col) : Vector()
{
   x = hypre_ParVectorCreate(MPI_COMM_WORLD,glob_size,col);
   hypre_ParVectorInitialize(x);
   hypre_ParVectorSetPartitioningOwner(x,0);
   // The data will be destroyed by hypre (this is the default)
   hypre_ParVectorSetDataOwner(x,1);
   hypre_SeqVectorSetDataOwner(hypre_ParVectorLocalVector(x),1);
   SetDataAndSize(hypre_VectorData(hypre_ParVectorLocalVector(x)),
                  hypre_VectorSize(hypre_ParVectorLocalVector(x)));
   own_ParVector = 1;
}

HypreParVector::HypreParVector(int glob_size, double *_data, int *col)
   : Vector()
{
   x = hypre_ParVectorCreate(MPI_COMM_WORLD,glob_size,col);
   hypre_ParVectorSetDataOwner(x,1); // owns the seq vector
   hypre_SeqVectorSetDataOwner(hypre_ParVectorLocalVector(x),0);
   hypre_ParVectorSetPartitioningOwner(x,0);
   hypre_VectorData(hypre_ParVectorLocalVector(x)) = _data;
   // If hypre_ParVectorLocalVector(x) is non-NULL, hypre_ParVectorInitialize(x)
   // does not allocate memory!
   hypre_ParVectorInitialize(x);
   SetDataAndSize(hypre_VectorData(hypre_ParVectorLocalVector(x)),
                  hypre_VectorSize(hypre_ParVectorLocalVector(x)));
   own_ParVector = 1;
}

HypreParVector::HypreParVector(const HypreParVector &y) : Vector()
{
   x = hypre_ParVectorCreate(MPI_COMM_WORLD, y.x -> global_size,
                             y.x -> partitioning);
   hypre_ParVectorInitialize(x);
   hypre_ParVectorSetPartitioningOwner(x,0);
   hypre_ParVectorSetDataOwner(x,1);
   hypre_SeqVectorSetDataOwner(hypre_ParVectorLocalVector(x),1);
   SetDataAndSize(hypre_VectorData(hypre_ParVectorLocalVector(x)),
                  hypre_VectorSize(hypre_ParVectorLocalVector(x)));
   own_ParVector = 1;
}

HypreParVector::HypreParVector(HYPRE_ParVector y) : Vector()
{
   x = (hypre_ParVector *) y;
   SetDataAndSize(hypre_VectorData(hypre_ParVectorLocalVector(x)),
                  hypre_VectorSize(hypre_ParVectorLocalVector(x)));
   own_ParVector = 0;
}

HypreParVector::operator hypre_ParVector*() const
{
   return x;
}

HypreParVector::operator HYPRE_ParVector() const
{
   return (HYPRE_ParVector) x;
}

Vector * HypreParVector::GlobalVector()
{
   hypre_Vector *hv = hypre_ParVectorToVectorAll(*this);
   Vector *v = new Vector(hv->data, hv->size);
   v->MakeDataOwner();
   hypre_SeqVectorSetDataOwner(hv,0);
   hypre_SeqVectorDestroy(hv);
   return v;
}

HypreParVector& HypreParVector::operator=(double d)
{
   hypre_ParVectorSetConstantValues(x,d);
   return *this;
}

HypreParVector& HypreParVector::operator=(const HypreParVector &y)
{
#ifdef MFEM_DEBUG
   if (size != y.Size())
      cerr << "HypreParVector::operator=" << endl;
#endif

   for (int i = 0; i < size; i++)
      data[i] = y.data[i];
   return *this;
}

void HypreParVector::SetData(double *_data)
{
   Vector::data = hypre_VectorData(hypre_ParVectorLocalVector(x)) = _data;
}

int HypreParVector::Randomize(int seed)
{
   return hypre_ParVectorSetRandomValues(x,seed);
}

void HypreParVector::Print(const char *fname)
{
   hypre_ParVectorPrint(x,fname);
}

HypreParVector::~HypreParVector()
{
   if (own_ParVector)
      hypre_ParVectorDestroy(x);
}


double InnerProduct(HypreParVector *x, HypreParVector *y)
{
   return hypre_ParVectorInnerProd(*x, *y);
}

double InnerProduct(HypreParVector &x, HypreParVector &y)
{
   return hypre_ParVectorInnerProd(x, y);
}


HypreParMatrix::HypreParMatrix(int size, int *row, SparseMatrix *diag)
   : Operator(size)
{
   A = hypre_ParCSRMatrixCreate(MPI_COMM_WORLD, size, size, row, row,
                                0, diag->NumNonZeroElems(), 0);
   hypre_ParCSRMatrixSetDataOwner(A,0);
   hypre_ParCSRMatrixSetRowStartsOwner(A,0);
   hypre_ParCSRMatrixSetColStartsOwner(A,0);

   hypre_CSRMatrixSetDataOwner(A->diag,0);
   hypre_CSRMatrixI(A->diag)    = diag->GetI();
   hypre_CSRMatrixJ(A->diag)    = diag->GetJ();
   hypre_CSRMatrixData(A->diag) = diag->GetData();
   hypre_CSRMatrixSetRownnz(A->diag);

   hypre_CSRMatrixSetDataOwner(A->offd,1);
   hypre_CSRMatrixI(A->offd)    = new int[diag->Size()+1];
   hypre_CSRMatrixJ(A->offd)    = NULL;
   for (int k = 0; k < diag->Size()+1; k++)
      (A->offd)->i[k] = 0;

   /* Don't need to call these, since they allocate memory only
      if it was not already allocated */
   // hypre_CSRMatrixInitialize(A->diag);
   // hypre_ParCSRMatrixInitialize(A);

   hypre_ParCSRMatrixSetNumNonzeros(A);

   hypre_MatvecCommPkgCreate(A);

   CommPkg = NULL;
   X = Y = NULL;
}


HypreParMatrix::HypreParMatrix(int M, int N, int *row, int *col,
                               SparseMatrix *diag)
{
   A = hypre_ParCSRMatrixCreate(MPI_COMM_WORLD, M, N, row, col,
                                0, diag->NumNonZeroElems(), 0);
   hypre_ParCSRMatrixSetDataOwner(A,0);
   hypre_ParCSRMatrixSetRowStartsOwner(A,0);
   hypre_ParCSRMatrixSetColStartsOwner(A,0);

   hypre_CSRMatrixSetDataOwner(A->diag,0);
   hypre_CSRMatrixI(A->diag)    = diag->GetI();
   hypre_CSRMatrixJ(A->diag)    = diag->GetJ();
   hypre_CSRMatrixData(A->diag) = diag->GetData();
   hypre_CSRMatrixSetRownnz(A->diag);

   hypre_CSRMatrixSetDataOwner(A->offd,1);
   hypre_CSRMatrixI(A->offd) = new int[diag->Size()+1];
   for (int k = 0; k < diag->Size()+1; k++) (A->offd)->i[k] = 0;

   hypre_ParCSRMatrixSetNumNonzeros(A);

   hypre_MatvecCommPkgCreate(A);

   CommPkg = NULL;
   X = Y = NULL;

   size = GetNumRows();
}

HypreParMatrix::HypreParMatrix(int M, int N, int *row, int *col,
                               SparseMatrix *diag, SparseMatrix *offd,
                               int *cmap)
{
   A = hypre_ParCSRMatrixCreate(MPI_COMM_WORLD, M, N, row, col,
                                offd->Width(), diag->NumNonZeroElems(),
                                offd->NumNonZeroElems());
   hypre_ParCSRMatrixSetDataOwner(A,0);
   hypre_ParCSRMatrixSetRowStartsOwner(A,0);
   hypre_ParCSRMatrixSetColStartsOwner(A,0);

   hypre_CSRMatrixSetDataOwner(A->diag,0);
   hypre_CSRMatrixI(A->diag)    = diag->GetI();
   hypre_CSRMatrixJ(A->diag)    = diag->GetJ();
   hypre_CSRMatrixData(A->diag) = diag->GetData();
   hypre_CSRMatrixSetRownnz(A->diag);

   hypre_CSRMatrixSetDataOwner(A->offd,0);
   hypre_CSRMatrixI(A->offd)    = offd->GetI();
   hypre_CSRMatrixJ(A->offd)    = offd->GetJ();
   hypre_CSRMatrixData(A->offd) = offd->GetData();
   hypre_CSRMatrixSetRownnz(A->offd);

   hypre_ParCSRMatrixColMapOffd(A) = cmap;

   hypre_ParCSRMatrixSetNumNonzeros(A);

   hypre_MatvecCommPkgCreate(A);

   CommPkg = NULL;
   X = Y = NULL;

   size = GetNumRows();
}

HypreParMatrix::HypreParMatrix(int *row, int *col, SparseMatrix *sm_a)
{
#ifdef MFEM_DEBUG
   if (sm_a == NULL)
      mfem_error("HypreParMatrix::HypreParMatrix: sm_a==NULL");
#endif

   hypre_CSRMatrix *csr_a;

   csr_a = hypre_CSRMatrixCreate(sm_a -> Size(), sm_a -> Width(),
                                 sm_a -> NumNonZeroElems());

   hypre_CSRMatrixSetDataOwner(csr_a,0);
   hypre_CSRMatrixI(csr_a)    = sm_a -> GetI();
   hypre_CSRMatrixJ(csr_a)    = sm_a -> GetJ();
   hypre_CSRMatrixData(csr_a) = sm_a -> GetData();
   hypre_CSRMatrixSetRownnz(csr_a);

   A = hypre_CSRMatrixToParCSRMatrix(MPI_COMM_WORLD,csr_a,row,col);

   CommPkg = NULL;
   X = Y = NULL;

   size = GetNumRows();

   hypre_MatvecCommPkgCreate(A);
}

HypreParMatrix::HypreParMatrix(int M, int N, int *row, int *col,
                               Table *diag)
{
   int nnz = diag->Size_of_connections();
   A = hypre_ParCSRMatrixCreate(MPI_COMM_WORLD, M, N, row, col,
                                0, nnz, 0);
   hypre_ParCSRMatrixSetDataOwner(A,1);
   hypre_ParCSRMatrixSetRowStartsOwner(A,0);
   hypre_ParCSRMatrixSetColStartsOwner(A,0);

   hypre_CSRMatrixSetDataOwner(A->diag,1);
   hypre_CSRMatrixI(A->diag)    = diag->GetI();
   hypre_CSRMatrixJ(A->diag)    = diag->GetJ();

   hypre_CSRMatrixData(A->diag) = new double[nnz];
   for (int k = 0; k < nnz; k++)
      (hypre_CSRMatrixData(A->diag))[k] = 1.0;

   hypre_CSRMatrixSetRownnz(A->diag);

   hypre_CSRMatrixSetDataOwner(A->offd,1);
   hypre_CSRMatrixI(A->offd) = new int[diag->Size()+1];
   for (int k = 0; k < diag->Size()+1; k++) (A->offd)->i[k] = 0;

   hypre_ParCSRMatrixSetNumNonzeros(A);

   hypre_MatvecCommPkgCreate(A);

   CommPkg = NULL;
   X = Y = NULL;

   size = GetNumRows();
}

HypreParMatrix::HypreParMatrix(MPI_Comm comm, int id, int np,
                               int *row, int *col,
                               int *i_diag, int *j_diag,
                               int *i_offd, int *j_offd,
                               int *cmap, int cmap_size)
{
   int diag_col, offd_col;

   if (HYPRE_AssumedPartitionCheck())
   {
      diag_col = i_diag[row[1]-row[0]];
      offd_col = i_offd[row[1]-row[0]];

      A = hypre_ParCSRMatrixCreate(comm, row[2], col[2], row, col,
                                   cmap_size, diag_col, offd_col);
   }
   else
   {
      diag_col = i_diag[row[id+1]-row[id]];
      offd_col = i_offd[row[id+1]-row[id]];

      A = hypre_ParCSRMatrixCreate(comm, row[np], col[np], row, col,
                                   cmap_size, diag_col, offd_col);
   }

   hypre_ParCSRMatrixSetDataOwner(A,1);
   hypre_ParCSRMatrixSetRowStartsOwner(A,0);
   hypre_ParCSRMatrixSetColStartsOwner(A,0);

   int i;

   double * a_diag = new double[diag_col];
   for (i = 0; i < diag_col; i++)
      a_diag[i] = 1.0;

   double * a_offd = new double[offd_col];
   for (i = 0; i < offd_col; i++)
      a_offd[i] = 1.0;

   hypre_CSRMatrixSetDataOwner(A->diag,1);
   hypre_CSRMatrixI(A->diag)    = i_diag;
   hypre_CSRMatrixJ(A->diag)    = j_diag;
   hypre_CSRMatrixData(A->diag) = a_diag;
   hypre_CSRMatrixSetRownnz(A->diag);

   hypre_CSRMatrixSetDataOwner(A->offd,1);
   hypre_CSRMatrixI(A->offd)    = i_offd;
   hypre_CSRMatrixJ(A->offd)    = j_offd;
   hypre_CSRMatrixData(A->offd) = a_offd;
   hypre_CSRMatrixSetRownnz(A->offd);

   hypre_ParCSRMatrixColMapOffd(A) = cmap;

   hypre_ParCSRMatrixSetNumNonzeros(A);

   hypre_MatvecCommPkgCreate(A);

   CommPkg = NULL;
   X = Y = NULL;

   size = GetNumRows();
}

void HypreParMatrix::SetCommPkg(hypre_ParCSRCommPkg *comm_pkg)
{
   CommPkg = comm_pkg;

   if (hypre_ParCSRMatrixCommPkg(A))
      hypre_MatvecCommPkgDestroy(hypre_ParCSRMatrixCommPkg(A));

   hypre_ParCSRMatrixCommPkg(A) = comm_pkg;
}

void HypreParMatrix::CheckCommPkg()
{
#ifdef MFEM_DEBUG
   if (CommPkg == NULL || CommPkg != hypre_ParCSRMatrixCommPkg(A))
      cerr << endl << "HypreParMatrix::CheckCommPkg()" << endl;
#endif
}

void HypreParMatrix::DestroyCommPkg()
{
   if (CommPkg == NULL)
      return;
   hypre_TFree(CommPkg->send_procs);
   hypre_TFree(CommPkg->send_map_starts);
   hypre_TFree(CommPkg->send_map_elmts);
   hypre_TFree(CommPkg->recv_procs);
   hypre_TFree(CommPkg->recv_vec_starts);
   if (CommPkg->send_mpi_types)
      hypre_TFree(CommPkg->send_mpi_types);
   if (CommPkg->recv_mpi_types)
      hypre_TFree(CommPkg->recv_mpi_types);
   if (hypre_ParCSRMatrixCommPkg(A) == CommPkg)
      hypre_ParCSRMatrixCommPkg(A) = NULL;
   delete CommPkg;
   CommPkg = NULL;
}

HypreParMatrix::operator hypre_ParCSRMatrix*()
{
   return A;
}

HypreParMatrix::operator HYPRE_ParCSRMatrix()
{
   return (HYPRE_ParCSRMatrix) A;
}

hypre_ParCSRMatrix* HypreParMatrix::StealData()
{
   hypre_ParCSRMatrix *R = A;
   A = NULL;
   return R;
}

HypreParMatrix * HypreParMatrix::Transpose()
{
   hypre_ParCSRMatrix * At;
   hypre_ParCSRMatrixTranspose(A, &At, 1);
   hypre_ParCSRMatrixSetNumNonzeros(At);

   hypre_MatvecCommPkgCreate(At);

   return new HypreParMatrix(At);
}

int HypreParMatrix::Mult(HypreParVector &x, HypreParVector &y,
                         double a, double b)
{
   return hypre_ParCSRMatrixMatvec(a, A, x, b, y);
}

void HypreParMatrix::Mult(const Vector &x, Vector &y) const
{
   if (X == NULL)
   {
      X = new HypreParVector(GetGlobalNumCols(),
                             x.GetData(),
                             GetColStarts());
      Y = new HypreParVector(GetGlobalNumRows(),
                             y.GetData(),
                             GetRowStarts());
   }
   else
   {
      X -> SetData(x.GetData());
      Y -> SetData(y.GetData());
   }

   hypre_ParCSRMatrixMatvec(1.0, A, *X, 0.0, *Y);
}

int HypreParMatrix::Mult(HYPRE_ParVector x, HYPRE_ParVector y,
                         double a, double b)
{
   return hypre_ParCSRMatrixMatvec(a,A,(hypre_ParVector *)x,b,(hypre_ParVector *)y);
}

int HypreParMatrix::MultTranspose(HypreParVector & x, HypreParVector & y,
                                  double a, double b)
{
   return hypre_ParCSRMatrixMatvecT(a,A,x,b,y);
}

void HypreParMatrix::Print(const char *fname, int offi, int offj)
{
   hypre_ParCSRMatrixPrintIJ(A,offi,offj,fname);
}

void HypreParMatrix::Read(const char *fname)
{
   if (A) hypre_ParCSRMatrixDestroy(A);
   int io,jo;
   hypre_ParCSRMatrixReadIJ(MPI_COMM_WORLD, fname, &io, &jo, &A);
   hypre_ParCSRMatrixSetNumNonzeros(A);

   hypre_MatvecCommPkgCreate(A);
}

HypreParMatrix::~HypreParMatrix()
{
   DestroyCommPkg();

   if (A)
   {
      if (hypre_ParCSRMatrixCommPkg(A))
         hypre_MatvecCommPkgDestroy(hypre_ParCSRMatrixCommPkg(A));
      hypre_ParCSRMatrixCommPkg(A) = NULL;

      if (hypre_CSRMatrixOwnsData(A->diag))
      {
         hypre_CSRMatrixDestroy(A->diag);
         A->diag = NULL;
      }
      else
      {
         hypre_TFree(A->diag);
         A->diag = NULL;
      }

      if (hypre_CSRMatrixOwnsData(A->offd))
      {
         hypre_CSRMatrixDestroy(A->offd);
         A->offd = NULL;
      }
      hypre_ParCSRMatrixDestroy(A);
   }

   delete X;
   delete Y;
}

HypreParMatrix * ParMult(HypreParMatrix *A, HypreParMatrix *B)
{
   hypre_ParCSRMatrix * ab;
   ab = hypre_ParMatmul(*A,*B);

   hypre_MatvecCommPkgCreate(ab);

   return new HypreParMatrix(ab);
}

HypreParMatrix * RAP(HypreParMatrix *A, HypreParMatrix *P)
{
   int P_owns_its_col_starts =
      hypre_ParCSRMatrixOwnsColStarts((hypre_ParCSRMatrix*)(*P));
   hypre_ParCSRMatrix * rap;
   hypre_BoomerAMGBuildCoarseOperator(*P,*A,*P,&rap);

   hypre_ParCSRMatrixSetNumNonzeros(rap);
   // hypre_MatvecCommPkgCreate(rap);
   if (!P_owns_its_col_starts)
   {
      /* Warning: hypre_BoomerAMGBuildCoarseOperator steals the col_starts
         from P (even if it does not own them)! */
      hypre_ParCSRMatrixSetRowStartsOwner(rap,0);
      hypre_ParCSRMatrixSetColStartsOwner(rap,0);
   }
   return new HypreParMatrix(rap);
}


HypreSolver::HypreSolver()
{
   size = 0;

   A = NULL;
   setup_called = 0;
   B = X = NULL;
}

HypreSolver::HypreSolver(HypreParMatrix *_A)
{
   size = _A -> GetNumRows();

   A = _A;
   setup_called = 0;
   B = X = NULL;
}

void HypreSolver::Mult(const HypreParVector &b, HypreParVector &x) const
{
   if (A == NULL)
   {
      cerr << "HypreSolver::Mult (...) : HypreParMatrix A is missing" << endl;
      return;
   }
   if (!setup_called)
   {
      SetupFcn()(*this, *A, b, x);
      setup_called = 1;
   }

   SolveFcn()(*this, *A, b, x);
}

void HypreSolver::Mult(const Vector &b, Vector &x) const
{
   if (A == NULL)
   {
      cerr << "HypreSolver::Mult (...) : HypreParMatrix A is missing" << endl;
      return;
   }
   if (B == NULL)
   {
      B = new HypreParVector(A -> GetGlobalNumRows(),
                             b.GetData(),
                             A -> GetRowStarts());
      X = new HypreParVector(A -> GetGlobalNumCols(),
                             x.GetData(),
                             A -> GetColStarts());
   }
   else
   {
      B -> SetData(b.GetData());
      X -> SetData(x.GetData());
   }

   Mult(*B, *X);
}

HypreSolver::~HypreSolver()
{
   if (B) delete B;
   if (X) delete X;
}


HyprePCG::HyprePCG(HypreParMatrix &_A) : HypreSolver(&_A)
{
   MPI_Comm comm;

   print_level = 0;
   use_zero_initial_iterate = 0;

   HYPRE_ParCSRMatrixGetComm(*A, &comm);

   HYPRE_ParCSRPCGCreate(comm, &pcg_solver);
}

void HyprePCG::SetTol(double tol)
{
   HYPRE_ParCSRPCGSetTol(pcg_solver, tol);
}

void HyprePCG::SetMaxIter(int max_iter)
{
   HYPRE_ParCSRPCGSetMaxIter(pcg_solver, max_iter);
}

void HyprePCG::SetLogging(int logging)
{
   HYPRE_ParCSRPCGSetLogging(pcg_solver, logging);
}

void HyprePCG::SetPrintLevel(int print_lvl)
{
   print_level = print_lvl;
   HYPRE_ParCSRPCGSetPrintLevel(pcg_solver, print_level);
}

void HyprePCG::SetPreconditioner(HypreSolver &precond)
{
   HYPRE_ParCSRPCGSetPrecond(pcg_solver,
                             precond.SolveFcn(),
                             precond.SetupFcn(),
                             precond);
}

void HyprePCG::Mult(const HypreParVector &b, HypreParVector &x) const
{
   int myid;
   int time_index;
   int num_iterations;
   double final_res_norm;
   MPI_Comm comm;

   HYPRE_ParCSRMatrixGetComm(*A, &comm);

   if (!setup_called)
   {
      HYPRE_ParCSRPCGSetup(pcg_solver, *A, b, x);
      setup_called = 1;
   }

   if (print_level > 0)
   {
      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_BeginTiming(time_index);
   }

   if (use_zero_initial_iterate)
      x = 0.0;

   HYPRE_ParCSRPCGSolve(pcg_solver, *A, b, x);

   if (print_level > 0)
   {
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_ParCSRPCGGetNumIterations(pcg_solver, &num_iterations);
      HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(pcg_solver,
                                                  &final_res_norm);

      MPI_Comm_rank(comm, &myid);

      if (myid == 0)
      {
         cout << "PCG Iterations = " << num_iterations << endl
              << "Final PCG Relative Residual Norm = " << final_res_norm
              << endl;
      }
   }
}

HyprePCG::~HyprePCG()
{
   HYPRE_ParCSRPCGDestroy(pcg_solver);
}


HypreGMRES::HypreGMRES(HypreParMatrix &_A) : HypreSolver(&_A)
{
   MPI_Comm comm;

   int k_dim    = 50;
   int max_iter = 100;
   double tol   = 1e-6;

   print_level = 0;
   use_zero_initial_iterate = 0;

   HYPRE_ParCSRMatrixGetComm(*A, &comm);

   HYPRE_ParCSRGMRESCreate(comm, &gmres_solver);
   HYPRE_ParCSRGMRESSetKDim(gmres_solver, k_dim);
   HYPRE_ParCSRGMRESSetMaxIter(gmres_solver, max_iter);
   HYPRE_ParCSRGMRESSetTol(gmres_solver, tol);
}

void HypreGMRES::SetTol(double tol)
{
   HYPRE_ParCSRGMRESSetTol(gmres_solver, tol);
}

void HypreGMRES::SetMaxIter(int max_iter)
{
   HYPRE_ParCSRGMRESSetMaxIter(gmres_solver, max_iter);
}

void HypreGMRES::SetKDim(int k_dim)
{
   HYPRE_ParCSRGMRESSetKDim(gmres_solver, k_dim);
}

void HypreGMRES::SetLogging(int logging)
{
   HYPRE_ParCSRGMRESSetLogging(gmres_solver, logging);
}

void HypreGMRES::SetPrintLevel(int print_lvl)
{
   print_level = print_lvl;
   HYPRE_ParCSRGMRESSetPrintLevel(gmres_solver, print_level);
}

void HypreGMRES::SetPreconditioner(HypreSolver &precond)
{
   HYPRE_ParCSRGMRESSetPrecond(gmres_solver,
                               precond.SolveFcn(),
                               precond.SetupFcn(),
                               precond);
}

void HypreGMRES::Mult(const HypreParVector &b, HypreParVector &x) const
{
   int myid;
   int time_index;
   int num_iterations;
   double final_res_norm;
   MPI_Comm comm;

   HYPRE_ParCSRMatrixGetComm(*A, &comm);

   if (!setup_called)
   {
      HYPRE_ParCSRGMRESSetup(gmres_solver, *A, b, x);
      setup_called = 1;
   }

   if (print_level > 0)
   {
      time_index = hypre_InitializeTiming("GMRES Solve");
      hypre_BeginTiming(time_index);
   }

   if (use_zero_initial_iterate)
      x = 0.0;

   HYPRE_ParCSRGMRESSolve(gmres_solver, *A, b, x);

   if (print_level > 0)
   {
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_ParCSRGMRESGetNumIterations(gmres_solver, &num_iterations);
      HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm(gmres_solver,
                                                    &final_res_norm);

      MPI_Comm_rank(comm, &myid);

      if (myid == 0)
      {
         cout << "GMRES Iterations = " << num_iterations << endl
              << "Final GMRES Relative Residual Norm = " << final_res_norm
              << endl;
      }
   }
}

HypreGMRES::~HypreGMRES()
{
   HYPRE_ParCSRGMRESDestroy(gmres_solver);
}


HypreParaSails::HypreParaSails(HypreParMatrix &A) : HypreSolver(&A)
{
   MPI_Comm comm;

   int    sai_max_levels = 1;
   double sai_threshold  = 0.1;
   double sai_filter     = 0.1;
   int    sai_sym        = 0;
   double sai_loadbal    = 0.0;
   int    sai_reuse      = 0;
   int    sai_logging    = 1;

   HYPRE_ParCSRMatrixGetComm(A, &comm);

   HYPRE_ParaSailsCreate(comm, &sai_precond);
   HYPRE_ParaSailsSetParams(sai_precond, sai_threshold, sai_max_levels);
   HYPRE_ParaSailsSetFilter(sai_precond, sai_filter);
   HYPRE_ParaSailsSetSym(sai_precond, sai_sym);
   HYPRE_ParaSailsSetLoadbal(sai_precond, sai_loadbal);
   HYPRE_ParaSailsSetReuse(sai_precond, sai_reuse);
   HYPRE_ParaSailsSetLogging(sai_precond, sai_logging);
}

void HypreParaSails::SetSymmetry(int sym)
{
   HYPRE_ParaSailsSetSym(sai_precond, sym);
}

HypreParaSails::~HypreParaSails()
{
   HYPRE_ParaSailsDestroy(sai_precond);
}


HypreBoomerAMG::HypreBoomerAMG(HypreParMatrix &A) : HypreSolver(&A)
{
   int coarsen_type = 10;
   int agg_levels   = 1;
   int relax_type   = 8;
   int relax_sweeps = 1;
   double theta     = 0.25;
   int interp_type  = 6;
   int Pmax         = 4;
   int print_level  = 1;

   HYPRE_BoomerAMGCreate(&amg_precond);

   HYPRE_BoomerAMGSetCoarsenType(amg_precond, coarsen_type);
   HYPRE_BoomerAMGSetAggNumLevels(amg_precond, agg_levels);
   HYPRE_BoomerAMGSetRelaxType(amg_precond, relax_type);
   HYPRE_BoomerAMGSetNumSweeps(amg_precond, relax_sweeps);
   HYPRE_BoomerAMGSetMaxLevels(amg_precond, 25);
   HYPRE_BoomerAMGSetTol(amg_precond, 0.0);
   HYPRE_BoomerAMGSetMaxIter(amg_precond, 1); // one V-cycle
   HYPRE_BoomerAMGSetStrongThreshold(amg_precond, theta);
   HYPRE_BoomerAMGSetInterpType(amg_precond, interp_type);
   HYPRE_BoomerAMGSetPMaxElmts(amg_precond, Pmax);
   HYPRE_BoomerAMGSetPrintLevel(amg_precond, print_level);
}

void HypreBoomerAMG::SetSystemsOptions(int dim)
{
   HYPRE_BoomerAMGSetNumFunctions(amg_precond, dim);

   // More robust options with respect to convergence
   HYPRE_BoomerAMGSetAggNumLevels(amg_precond, 0);
   HYPRE_BoomerAMGSetStrongThreshold(amg_precond, 0.5);
}

HypreBoomerAMG::~HypreBoomerAMG()
{
   HYPRE_BoomerAMGDestroy(amg_precond);
}


#include "../fem/fem.hpp"

HypreAMS::HypreAMS(HypreParMatrix &A, ParFiniteElementSpace *edge_fespace)
   : HypreSolver(&A)
{
   int cycle_type       = 13;
   int rlx_type         = 2;
   int rlx_sweeps       = 1;
   double rlx_weight    = 1.0;
   double rlx_omega     = 1.0;
   int amg_coarsen_type = 10;
   int amg_agg_levels   = 1;
   int amg_agg_npaths   = 1;
   int amg_rlx_type     = 8;
   double theta         = 0.25;
   int amg_interp_type  = 6;
   int amg_Pmax         = 4;
   int print_level      = 1;

   HYPRE_AMSCreate(&ams);

   HYPRE_AMSSetDimension(ams, 3); // 3D problems
   HYPRE_AMSSetTol(ams, 0.0);
   HYPRE_AMSSetMaxIter(ams, 1); // use as a preconditioner
   HYPRE_AMSSetCycleType(ams, cycle_type);
   HYPRE_AMSSetPrintLevel(ams, 1);

   /// define the nodal linear finite element space associated with edge_fespace
   ParMesh *pmesh = (ParMesh *) edge_fespace->GetMesh();
   FiniteElementCollection *vert_fec = new LinearFECollection;
   ParFiniteElementSpace *vert_fespace = new ParFiniteElementSpace(pmesh, vert_fec);

   // generate and set the vertex coordinates
   ParGridFunction x_coord(vert_fespace);
   ParGridFunction y_coord(vert_fespace);
   ParGridFunction z_coord(vert_fespace);
   double *coord;
   for (int i = 0; i < pmesh->GetNV(); i++)
   {
      coord = pmesh -> GetVertex(i);
      x_coord(i) = coord[0];
      y_coord(i) = coord[1];
      z_coord(i) = coord[2];
   }
   x = x_coord.ParallelAverage();
   y = y_coord.ParallelAverage();
   z = z_coord.ParallelAverage();
   HYPRE_AMSSetCoordinateVectors(ams, *x, *y, *z);

   // generate and set the discrete gradient
   HYPRE_ParCSRMatrix Gh;
   {
      int *edge_vertex = new int[2*edge_fespace->TrueVSize()];
      Array<int> vert;
      // set orientation of locally owned edges (tdofs in edge_fespace)
      for (int i = 0; i < pmesh->GetNEdges(); i++)
      {
         int j = edge_fespace->GetLocalTDofNumber(i);
         if (j >= 0)
         {
            pmesh->GetEdgeVertices(i,vert);
            if (vert[0] < vert[1])
            {
               edge_vertex[2*j] = vert_fespace->GetGlobalTDofNumber(vert[0]);
               edge_vertex[2*j+1] = vert_fespace->GetGlobalTDofNumber(vert[1]);
            }
            else
            {
               edge_vertex[2*j] = vert_fespace->GetGlobalTDofNumber(vert[1]);
               edge_vertex[2*j+1] = vert_fespace->GetGlobalTDofNumber(vert[0]);
            }
         }
      }
      // fix the orientation of shared edges
      for (int gr = 1; gr < pmesh->GetNGroups(); gr++)
      {
         if (pmesh->groupmaster_lproc[gr] == 0)
            for (int j = 0; j < pmesh->GroupNEdges(gr); j++)
            {
               int k, o;
               pmesh->GroupEdge(gr, j, k, o);
               if (edge_fespace->GetDofSign(k) < 0)
               {
                  k = edge_fespace->GetLocalTDofNumber(k);
                  int tmp = edge_vertex[2*k];
                  edge_vertex[2*k] = edge_vertex[2*k+1];
                  edge_vertex[2*k+1] = tmp;
               }
            }
      }
      HYPRE_AMSConstructDiscreteGradient(A, *x, edge_vertex, 1, &Gh);
      delete edge_vertex;
   }
   G = new HypreParMatrix((hypre_ParCSRMatrix *)Gh);
   HYPRE_AMSSetDiscreteGradient(ams, *G);

   delete vert_fec;
   delete vert_fespace;

   // set additional AMS options
   HYPRE_AMSSetSmoothingOptions(ams, rlx_type, rlx_sweeps, rlx_weight, rlx_omega);
   HYPRE_AMSSetAlphaAMGOptions(ams, amg_coarsen_type, amg_agg_levels, amg_rlx_type,
                               theta, amg_interp_type, amg_Pmax);
   HYPRE_AMSSetBetaAMGOptions(ams, amg_coarsen_type, amg_agg_levels, amg_rlx_type,
                              theta, amg_interp_type, amg_Pmax);

}

HypreAMS::~HypreAMS()
{
   HYPRE_AMSDestroy(ams);

   delete x;
   delete y;
   delete z;

   delete G;
}

#endif
