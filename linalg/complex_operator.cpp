// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "complex_operator.hpp"
#include <set>
#include <map>

namespace mfem
{

ComplexOperator::ComplexOperator(Operator * Op_Real, Operator * Op_Imag,
                                 bool ownReal, bool ownImag,
                                 Convention convention)
   : Operator(2*((Op_Real)?Op_Real->Height():Op_Imag->Height()),
              2*((Op_Real)?Op_Real->Width():Op_Imag->Width()))
   , Op_Real_(Op_Real)
   , Op_Imag_(Op_Imag)
   , ownReal_(ownReal)
   , ownImag_(ownImag)
   , convention_(convention)
   , x_r_()
   , x_i_()
   , y_r_()
   , y_i_()
   , u_(NULL)
   , v_(NULL)
{}

ComplexOperator::~ComplexOperator()
{
   if (ownReal_) { delete Op_Real_; }
   if (ownImag_) { delete Op_Imag_; }
   delete u_;
   delete v_;
}

Operator & ComplexOperator::real()
{
   MFEM_ASSERT(Op_Real_, "ComplexOperator has no real part!");
   return *Op_Real_;
}

Operator & ComplexOperator::imag()
{
   MFEM_ASSERT(Op_Imag_, "ComplexOperator has no imaginary part!");
   return *Op_Imag_;
}

const Operator & ComplexOperator::real() const
{
   MFEM_ASSERT(Op_Real_, "ComplexOperator has no real part!");
   return *Op_Real_;
}

const Operator & ComplexOperator::imag() const
{
   MFEM_ASSERT(Op_Imag_, "ComplexOperator has no imaginary part!");
   return *Op_Imag_;
}

void ComplexOperator::Mult(const Vector &x, Vector &y) const
{
   x.Read();
   y.UseDevice(true); y = 0.0;

   x_r_.MakeRef(const_cast<Vector&>(x), 0, width/2);
   x_i_.MakeRef(const_cast<Vector&>(x), width/2, width/2);

   y_r_.MakeRef(y, 0, height/2);
   y_i_.MakeRef(y, height/2, height/2);

   this->Mult(x_r_, x_i_, y_r_, y_i_);

   y_r_.SyncAliasMemory(y);
   y_i_.SyncAliasMemory(y);

   // Destroy alias vectors to prevent dangling aliases when the base vectors
   // are deleted
   x_r_.Destroy();
   x_i_.Destroy();
   y_r_.Destroy();
   y_i_.Destroy();
}

void ComplexOperator::Mult(const Vector &x_r, const Vector &x_i,
                           Vector &y_r, Vector &y_i) const
{
   if (Op_Real_)
   {
      Op_Real_->Mult(x_r, y_r);
      Op_Real_->Mult(x_i, y_i);
   }
   else
   {
      y_r = 0.0;
      y_i = 0.0;
   }

   if (Op_Imag_)
   {
      if (!v_) { v_ = new Vector(); }
      v_->UseDevice(true);
      v_->SetSize(Op_Imag_->Height());

      Op_Imag_->Mult(x_i, *v_);
      y_r.Add(-1.0, *v_);
      Op_Imag_->Mult(x_r, *v_);
      y_i.Add(1.0, *v_);
   }

   if (convention_ == BLOCK_SYMMETRIC)
   {
      y_i *= -1.0;
   }
}

void ComplexOperator::MultTranspose(const Vector &x, Vector &y) const
{
   x.Read();
   y.UseDevice(true); y = 0.0;

   x_r_.MakeRef(const_cast<Vector&>(x), 0, height/2);
   x_i_.MakeRef(const_cast<Vector&>(x), height/2, height/2);

   y_r_.MakeRef(y, 0, width/2);
   y_i_.MakeRef(y, width/2, width/2);

   this->MultTranspose(x_r_, x_i_, y_r_, y_i_);

   y_r_.SyncAliasMemory(y);
   y_i_.SyncAliasMemory(y);

   // Destroy alias vectors to prevent dangling aliases when the base vectors
   // are deleted
   x_r_.Destroy();
   x_i_.Destroy();
   y_r_.Destroy();
   y_i_.Destroy();
}

void ComplexOperator::MultTranspose(const Vector &x_r, const Vector &x_i,
                                    Vector &y_r, Vector &y_i) const
{
   if (Op_Real_)
   {
      Op_Real_->MultTranspose(x_r, y_r);
      Op_Real_->MultTranspose(x_i, y_i);

      if (convention_ == BLOCK_SYMMETRIC)
      {
         y_i *= -1.0;
      }
   }
   else
   {
      y_r = 0.0;
      y_i = 0.0;
   }

   if (Op_Imag_)
   {
      if (!u_) { u_ = new Vector(); }
      u_->UseDevice(true);
      u_->SetSize(Op_Imag_->Width());

      Op_Imag_->MultTranspose(x_i, *u_);
      y_r.Add(convention_ == BLOCK_SYMMETRIC ? -1.0 : 1.0, *u_);
      Op_Imag_->MultTranspose(x_r, *u_);
      y_i.Add(-1.0, *u_);
   }
}


SparseMatrix & ComplexSparseMatrix::real()
{
   MFEM_ASSERT(Op_Real_, "ComplexSparseMatrix has no real part!");
   return dynamic_cast<SparseMatrix &>(*Op_Real_);
}

SparseMatrix & ComplexSparseMatrix::imag()
{
   MFEM_ASSERT(Op_Imag_, "ComplexSparseMatrix has no imaginary part!");
   return dynamic_cast<SparseMatrix &>(*Op_Imag_);
}

const SparseMatrix & ComplexSparseMatrix::real() const
{
   MFEM_ASSERT(Op_Real_, "ComplexSparseMatrix has no real part!");
   return dynamic_cast<const SparseMatrix &>(*Op_Real_);
}

const SparseMatrix & ComplexSparseMatrix::imag() const
{
   MFEM_ASSERT(Op_Imag_, "ComplexSparseMatrix has no imaginary part!");
   return dynamic_cast<const SparseMatrix &>(*Op_Imag_);
}

SparseMatrix * ComplexSparseMatrix::GetSystemMatrix() const
{
   SparseMatrix * A_r = dynamic_cast<SparseMatrix*>(Op_Real_);
   SparseMatrix * A_i = dynamic_cast<SparseMatrix*>(Op_Imag_);

   const int  nrows_r = (A_r)?A_r->Height():0;
   const int  nrows_i = (A_i)?A_i->Height():0;
   const int    nrows = std::max(nrows_r, nrows_i);

   const int     *I_r = (A_r)?A_r->GetI():NULL;
   const int     *I_i = (A_i)?A_i->GetI():NULL;

   const int     *J_r = (A_r)?A_r->GetJ():NULL;
   const int     *J_i = (A_i)?A_i->GetJ():NULL;

   const double  *D_r = (A_r)?A_r->GetData():NULL;
   const double  *D_i = (A_i)?A_i->GetData():NULL;

   const int    nnz_r = (I_r)?I_r[nrows]:0;
   const int    nnz_i = (I_i)?I_i[nrows]:0;
   const int    nnz   = 2 * (nnz_r + nnz_i);

   int    *I = Memory<int>(this->Height()+1);
   int    *J = Memory<int>(nnz);
   double *D = Memory<double>(nnz);

   const double factor = (convention_ == HERMITIAN) ? 1.0 : -1.0;

   I[0] = 0;
   I[nrows] = nnz_r + nnz_i;
   for (int i=0; i<nrows; i++)
   {
      I[i + 1]         = ((I_r)?I_r[i+1]:0) + ((I_i)?I_i[i+1]:0);
      I[i + nrows + 1] = I[i+1] + nnz_r + nnz_i;

      if (I_r)
      {
         const int off_i = (I_i)?(I_i[i+1] - I_i[i]):0;
         for (int j=0; j<I_r[i+1] - I_r[i]; j++)
         {
            J[I[i] + j] = J_r[I_r[i] + j];
            D[I[i] + j] = D_r[I_r[i] + j];

            J[I[i+nrows] + off_i + j] = J_r[I_r[i] + j] + nrows;
            D[I[i+nrows] + off_i + j] = factor*D_r[I_r[i] + j];
         }
      }
      if (I_i)
      {
         const int off_r = (I_r)?(I_r[i+1] - I_r[i]):0;
         for (int j=0; j<I_i[i+1] - I_i[i]; j++)
         {
            J[I[i] + off_r + j] =  J_i[I_i[i] + j] + nrows;
            D[I[i] + off_r + j] = -D_i[I_i[i] + j];

            J[I[i+nrows] + j] = J_i[I_i[i] + j];
            D[I[i+nrows] + j] = factor*D_i[I_i[i] + j];
         }
      }
   }

   return new SparseMatrix(I, J, D, this->Height(), this->Width());
}


#ifdef MFEM_USE_SUITESPARSE

void ComplexUMFPackSolver::Init()
{
   mat = NULL;
   Numeric = NULL;
   AI = AJ = NULL;
   if (!use_long_ints)
   {
      umfpack_zi_defaults(Control);
   }
   else
   {
      umfpack_zl_defaults(Control);
   }
}

void ComplexUMFPackSolver::SetOperator(const Operator &op)
{
   void *Symbolic;

   if (Numeric)
   {
      if (!use_long_ints)
      {
         umfpack_zi_free_numeric(&Numeric);
      }
      else
      {
         umfpack_zl_free_numeric(&Numeric);
      }
   }

   mat = const_cast<ComplexSparseMatrix *>
         (dynamic_cast<const ComplexSparseMatrix *>(&op));
   MFEM_VERIFY(mat, "not a ComplexSparseMatrix");

   MFEM_VERIFY(mat->real().NumNonZeroElems() == mat->imag().NumNonZeroElems(),
               "Real and imag Sparsity pattern mismatch: Try setting Assemble (skip_zeros = 0)");

   // UMFPack requires that the column-indices in mat corresponding to each
   // row be sorted.
   // Generally, this will modify the ordering of the entries of mat.

   mat->real().SortColumnIndices();
   mat->imag().SortColumnIndices();

   height = mat->real().Height();
   width = mat->real().Width();
   MFEM_VERIFY(width == height, "not a square matrix");

   const int * Ap =
      mat->real().HostReadI(); // assuming real and imag have the same sparsity
   const int * Ai = mat->real().HostReadJ();
   const double * Ax = mat->real().HostReadData();
   const double * Az = mat->imag().HostReadData();

   if (!use_long_ints)
   {
      int status = umfpack_zi_symbolic(width,width,Ap,Ai,Ax,Az,&Symbolic,
                                       Control,Info);
      if (status < 0)
      {
         umfpack_zi_report_info(Control, Info);
         umfpack_zi_report_status(Control, status);
         mfem_error("ComplexUMFPackSolver::SetOperator :"
                    " umfpack_zi_symbolic() failed!");
      }

      status = umfpack_zi_numeric(Ap, Ai, Ax, Az, Symbolic, &Numeric,
                                  Control, Info);
      if (status < 0)
      {
         umfpack_zi_report_info(Control, Info);
         umfpack_zi_report_status(Control, status);
         mfem_error("ComplexUMFPackSolver::SetOperator :"
                    " umfpack_zi_numeric() failed!");
      }
      umfpack_zi_free_symbolic(&Symbolic);
   }
   else
   {
      SuiteSparse_long status;

      delete [] AJ;
      delete [] AI;
      AI = new SuiteSparse_long[width + 1];
      AJ = new SuiteSparse_long[Ap[width]];
      for (int i = 0; i <= width; i++)
      {
         AI[i] = (SuiteSparse_long)(Ap[i]);
      }
      for (int i = 0; i < Ap[width]; i++)
      {
         AJ[i] = (SuiteSparse_long)(Ai[i]);
      }

      status = umfpack_zl_symbolic(width, width, AI, AJ, Ax, Az, &Symbolic,
                                   Control, Info);
      if (status < 0)
      {
         umfpack_zl_report_info(Control, Info);
         umfpack_zl_report_status(Control, status);
         mfem_error("ComplexUMFPackSolver::SetOperator :"
                    " umfpack_zl_symbolic() failed!");
      }

      status = umfpack_zl_numeric(AI, AJ, Ax, Az, Symbolic, &Numeric,
                                  Control, Info);
      if (status < 0)
      {
         umfpack_zl_report_info(Control, Info);
         umfpack_zl_report_status(Control, status);
         mfem_error("ComplexUMFPackSolver::SetOperator :"
                    " umfpack_zl_numeric() failed!");
      }
      umfpack_zl_free_symbolic(&Symbolic);
   }
}

void ComplexUMFPackSolver::Mult(const Vector &b, Vector &x) const
{
   if (mat == NULL)
      mfem_error("ComplexUMFPackSolver::Mult : matrix is not set!"
                 " Call SetOperator first!");

   b.HostRead();
   x.HostReadWrite();

   int n = b.Size()/2;
   double * datax = x.GetData();
   double * datab = b.GetData();

   // For the Block Symmetric case data the imaginary part
   // has to be scaled by -1
   ComplexOperator::Convention conv = mat->GetConvention();
   Vector bimag;
   if (conv == ComplexOperator::Convention::BLOCK_SYMMETRIC)
   {
      bimag.SetDataAndSize(&datab[n],n);
      bimag *=-1.0;
   }

   // Solve the transpose, since UMFPack expects CCS instead of CRS format
   if (!use_long_ints)
   {
      int status =
         umfpack_zi_solve(UMFPACK_Aat, mat->real().HostReadI(), mat->real().HostReadJ(),
                          mat->real().HostReadData(), mat->imag().HostReadData(),
                          datax, &datax[n], datab, &datab[n], Numeric, Control, Info);
      umfpack_zi_report_info(Control, Info);
      if (status < 0)
      {
         umfpack_zi_report_status(Control, status);
         mfem_error("ComplexUMFPackSolver::Mult : umfpack_zi_solve() failed!");
      }
   }
   else
   {
      SuiteSparse_long status =
         umfpack_zl_solve(UMFPACK_Aat,AI,AJ,mat->real().HostReadData(),
                          mat->imag().HostReadData(),
                          datax,&datax[n],datab,&datab[n],Numeric,Control,Info);

      umfpack_zl_report_info(Control, Info);
      if (status < 0)
      {
         umfpack_zl_report_status(Control, status);
         mfem_error("ComplexUMFPackSolver::Mult : umfpack_zl_solve() failed!");
      }
   }
   if (conv == ComplexOperator::Convention::BLOCK_SYMMETRIC)
   {
      bimag *=-1.0;
   }
}

void ComplexUMFPackSolver::MultTranspose(const Vector &b, Vector &x) const
{
   if (mat == NULL)
      mfem_error("ComplexUMFPackSolver::Mult : matrix is not set!"
                 " Call SetOperator first!");
   b.HostRead();
   x.HostReadWrite();
   int n = b.Size()/2;
   double * datax = x.GetData();
   double * datab = b.GetData();

   ComplexOperator::Convention conv = mat->GetConvention();
   Vector bimag;
   bimag.SetDataAndSize(&datab[n],n);

   // Solve the Adjoint A^H x = b by solving
   // the conjugate problem A^T \bar{x} = \bar{b}
   if ((!transa && conv == ComplexOperator::HERMITIAN) ||
       ( transa && conv == ComplexOperator::BLOCK_SYMMETRIC))
   {
      bimag *=-1.0;
   }

   if (!use_long_ints)
   {
      int status =
         umfpack_zi_solve(UMFPACK_A, mat->real().HostReadI(), mat->real().HostReadJ(),
                          mat->real().HostReadData(), mat->imag().HostReadData(),
                          datax, &datax[n], datab, &datab[n], Numeric, Control, Info);
      umfpack_zi_report_info(Control, Info);
      if (status < 0)
      {
         umfpack_zi_report_status(Control, status);
         mfem_error("ComplexUMFPackSolver::Mult : umfpack_zi_solve() failed!");
      }
   }
   else
   {
      SuiteSparse_long status =
         umfpack_zl_solve(UMFPACK_A,AI,AJ,mat->real().HostReadData(),
                          mat->imag().HostReadData(),
                          datax,&datax[n],datab,&datab[n],Numeric,Control,Info);

      umfpack_zl_report_info(Control, Info);
      if (status < 0)
      {
         umfpack_zl_report_status(Control, status);
         mfem_error("ComplexUMFPackSolver::Mult : umfpack_zl_solve() failed!");
      }
   }
   if (!transa)
   {
      Vector ximag;
      ximag.SetDataAndSize(&datax[n],n);
      ximag *=-1.0;
   }
   if ((!transa && conv == ComplexOperator::HERMITIAN) ||
       ( transa && conv == ComplexOperator::BLOCK_SYMMETRIC))
   {
      bimag *=-1.0;
   }
}

ComplexUMFPackSolver::~ComplexUMFPackSolver()
{
   delete [] AJ;
   delete [] AI;
   if (Numeric)
   {
      if (!use_long_ints)
      {
         umfpack_zi_free_numeric(&Numeric);
      }
      else
      {
         umfpack_zl_free_numeric(&Numeric);
      }
   }
}

#endif

#ifdef MFEM_USE_MPI

ComplexHypreParMatrix::ComplexHypreParMatrix(HypreParMatrix * A_Real,
                                             HypreParMatrix * A_Imag,
                                             bool ownReal, bool ownImag,
                                             Convention convention)
   : ComplexOperator(A_Real, A_Imag, ownReal, ownImag, convention)
{
   comm_ = (A_Real) ? A_Real->GetComm() :
           ((A_Imag) ? A_Imag->GetComm() : MPI_COMM_WORLD);

   MPI_Comm_rank(comm_, &myid_);
   MPI_Comm_size(comm_, &nranks_);
}

HypreParMatrix & ComplexHypreParMatrix::real()
{
   MFEM_ASSERT(Op_Real_, "ComplexHypreParMatrix has no real part!");
   return dynamic_cast<HypreParMatrix &>(*Op_Real_);
}

HypreParMatrix & ComplexHypreParMatrix::imag()
{
   MFEM_ASSERT(Op_Imag_, "ComplexHypreParMatrix has no imaginary part!");
   return dynamic_cast<HypreParMatrix &>(*Op_Imag_);
}

const HypreParMatrix & ComplexHypreParMatrix::real() const
{
   MFEM_ASSERT(Op_Real_, "ComplexHypreParMatrix has no real part!");
   return dynamic_cast<const HypreParMatrix &>(*Op_Real_);
}

const HypreParMatrix & ComplexHypreParMatrix::imag() const
{
   MFEM_ASSERT(Op_Imag_, "ComplexHypreParMatrix has no imaginary part!");
   return dynamic_cast<const HypreParMatrix &>(*Op_Imag_);
}

HypreParMatrix * ComplexHypreParMatrix::GetSystemMatrix() const
{
   HypreParMatrix * A_r = dynamic_cast<HypreParMatrix*>(Op_Real_);
   HypreParMatrix * A_i = dynamic_cast<HypreParMatrix*>(Op_Imag_);

   if ( A_r == NULL && A_i == NULL ) { return NULL; }

   HYPRE_Int global_num_rows_r = (A_r) ? A_r->GetGlobalNumRows() : 0;
   HYPRE_Int global_num_rows_i = (A_i) ? A_i->GetGlobalNumRows() : 0;
   HYPRE_Int global_num_rows = std::max(global_num_rows_r, global_num_rows_i);

   HYPRE_Int global_num_cols_r = (A_r) ? A_r->GetGlobalNumCols() : 0;
   HYPRE_Int global_num_cols_i = (A_i) ? A_i->GetGlobalNumCols() : 0;
   HYPRE_Int global_num_cols = std::max(global_num_cols_r, global_num_cols_i);

   int row_starts_size = (HYPRE_AssumedPartitionCheck()) ? 2 : nranks_ + 1;
   HYPRE_Int * row_starts = mfem_hypre_CTAlloc(HYPRE_Int, row_starts_size);
   HYPRE_Int * col_starts = mfem_hypre_CTAlloc(HYPRE_Int, row_starts_size);

   const HYPRE_Int * row_starts_z = (A_r) ? A_r->RowPart() :
                                    ((A_i) ? A_i->RowPart() : NULL);
   const HYPRE_Int * col_starts_z = (A_r) ? A_r->ColPart() :
                                    ((A_i) ? A_i->ColPart() : NULL);

   for (int i = 0; i < row_starts_size; i++)
   {
      row_starts[i] = 2 * row_starts_z[i];
      col_starts[i] = 2 * col_starts_z[i];
   }

   SparseMatrix diag_r, diag_i, offd_r, offd_i;
   HYPRE_Int * cmap_r, * cmap_i;

   int nrows_r = 0, nrows_i = 0, ncols_r = 0, ncols_i = 0;
   int ncols_offd_r = 0, ncols_offd_i = 0;
   if (A_r)
   {
      A_r->GetDiag(diag_r);
      A_r->GetOffd(offd_r, cmap_r);
      nrows_r = diag_r.Height();
      ncols_r = diag_r.Width();
      ncols_offd_r = offd_r.Width();
   }
   if (A_i)
   {
      A_i->GetDiag(diag_i);
      A_i->GetOffd(offd_i, cmap_i);
      nrows_i = diag_i.Height();
      ncols_i = diag_i.Width();
      ncols_offd_i = offd_i.Width();
   }
   int nrows = std::max(nrows_r, nrows_i);
   int ncols = std::max(ncols_r, ncols_i);

   // Determine the unique set of off-diagonal columns global indices
   std::set<int> cset;
   for (int i=0; i<ncols_offd_r; i++)
   {
      cset.insert(cmap_r[i]);
   }
   for (int i=0; i<ncols_offd_i; i++)
   {
      cset.insert(cmap_i[i]);
   }
   int num_cols_offd = (int)cset.size();

   // Extract pointers to the various CSR arrays of the diagonal blocks
   const int * diag_r_I = (A_r) ? diag_r.GetI() : NULL;
   const int * diag_i_I = (A_i) ? diag_i.GetI() : NULL;

   const int * diag_r_J = (A_r) ? diag_r.GetJ() : NULL;
   const int * diag_i_J = (A_i) ? diag_i.GetJ() : NULL;

   const double * diag_r_D = (A_r) ? diag_r.GetData() : NULL;
   const double * diag_i_D = (A_i) ? diag_i.GetData() : NULL;

   int diag_r_nnz = (diag_r_I) ? diag_r_I[nrows] : 0;
   int diag_i_nnz = (diag_i_I) ? diag_i_I[nrows] : 0;
   int diag_nnz = 2 * (diag_r_nnz + diag_i_nnz);

   // Extract pointers to the various CSR arrays of the off-diagonal blocks
   const int * offd_r_I = (A_r) ? offd_r.GetI() : NULL;
   const int * offd_i_I = (A_i) ? offd_i.GetI() : NULL;

   const int * offd_r_J = (A_r) ? offd_r.GetJ() : NULL;
   const int * offd_i_J = (A_i) ? offd_i.GetJ() : NULL;

   const double * offd_r_D = (A_r) ? offd_r.GetData() : NULL;
   const double * offd_i_D = (A_i) ? offd_i.GetData() : NULL;

   int offd_r_nnz = (offd_r_I) ? offd_r_I[nrows] : 0;
   int offd_i_nnz = (offd_i_I) ? offd_i_I[nrows] : 0;
   int offd_nnz = 2 * (offd_r_nnz + offd_i_nnz);

   // Allocate CSR arrays for the combined matrix
   HYPRE_Int * diag_I = mfem_hypre_CTAlloc(HYPRE_Int, 2 * nrows + 1);
   HYPRE_Int * diag_J = mfem_hypre_CTAlloc(HYPRE_Int, diag_nnz);
   double    * diag_D = mfem_hypre_CTAlloc(double, diag_nnz);

   HYPRE_Int * offd_I = mfem_hypre_CTAlloc(HYPRE_Int, 2 * nrows + 1);
   HYPRE_Int * offd_J = mfem_hypre_CTAlloc(HYPRE_Int, offd_nnz);
   double    * offd_D = mfem_hypre_CTAlloc(double, offd_nnz);
   HYPRE_Int * cmap   = mfem_hypre_CTAlloc(HYPRE_Int, 2 * num_cols_offd);

   // Fill the CSR arrays for the diagonal portion of the matrix
   const double factor = (convention_ == HERMITIAN) ? 1.0 : -1.0;

   diag_I[0] = 0;
   diag_I[nrows] = diag_r_nnz + diag_i_nnz;
   for (int i=0; i<nrows; i++)
   {
      diag_I[i + 1]         = ((diag_r_I)?diag_r_I[i+1]:0) +
                              ((diag_i_I)?diag_i_I[i+1]:0);
      diag_I[i + nrows + 1] = diag_I[i+1] + diag_r_nnz + diag_i_nnz;

      if (diag_r_I)
      {
         for (int j=0; j<diag_r_I[i+1] - diag_r_I[i]; j++)
         {
            diag_J[diag_I[i] + j] = diag_r_J[diag_r_I[i] + j];
            diag_D[diag_I[i] + j] = diag_r_D[diag_r_I[i] + j];

            diag_J[diag_I[i+nrows] + j] =
               diag_r_J[diag_r_I[i] + j] + ncols;
            diag_D[diag_I[i+nrows] + j] =
               factor * diag_r_D[diag_r_I[i] + j];
         }
      }
      if (diag_i_I)
      {
         const int off_r = (diag_r_I)?(diag_r_I[i+1] - diag_r_I[i]):0;
         for (int j=0; j<diag_i_I[i+1] - diag_i_I[i]; j++)
         {
            diag_J[diag_I[i] + off_r + j] =  diag_i_J[diag_i_I[i] + j] + ncols;
            diag_D[diag_I[i] + off_r + j] = -diag_i_D[diag_i_I[i] + j];

            diag_J[diag_I[i+nrows] + off_r + j] = diag_i_J[diag_i_I[i] + j];
            diag_D[diag_I[i+nrows] + off_r + j] =
               factor * diag_i_D[diag_i_I[i] + j];
         }
      }
   }

   // Determine the mappings describing the layout of off-diagonal columns
   int num_recv_procs = 0;
   HYPRE_Int * offd_col_start_stop = NULL;
   this->getColStartStop(A_r, A_i, num_recv_procs, offd_col_start_stop);

   std::set<int>::iterator sit;
   std::map<int,int> cmapa, cmapb, cinvmap;
   for (sit=cset.begin(); sit!=cset.end(); sit++)
   {
      int col_orig = *sit;
      int col_2x2  = -1;
      int col_size = 0;
      for (int i=0; i<num_recv_procs; i++)
      {
         if (offd_col_start_stop[2*i] <= col_orig &&
             col_orig < offd_col_start_stop[2*i+1])
         {
            col_2x2 = offd_col_start_stop[2*i] + col_orig;
            col_size = offd_col_start_stop[2*i+1] - offd_col_start_stop[2*i];
            break;
         }
      }
      cmapa[*sit] = col_2x2;
      cmapb[*sit] = col_2x2 + col_size;
      cinvmap[col_2x2] = -1;
      cinvmap[col_2x2 + col_size] = -1;
   }
   delete [] offd_col_start_stop;

   std::map<int, int>::iterator mit;
   int i = 0;
   for (mit=cinvmap.begin(); mit!=cinvmap.end(); mit++, i++)
   {
      mit->second = i;
      cmap[i] = mit->first;
   }

   // Fill the CSR arrays for the off-diagonal portion of the matrix
   offd_I[0] = 0;
   offd_I[nrows] = offd_r_nnz + offd_i_nnz;
   for (int i=0; i<nrows; i++)
   {
      offd_I[i + 1]         = ((offd_r_I)?offd_r_I[i+1]:0) +
                              ((offd_i_I)?offd_i_I[i+1]:0);
      offd_I[i + nrows + 1] = offd_I[i+1] + offd_r_nnz + offd_i_nnz;

      if (offd_r_I)
      {
         const int off_i = (offd_i_I)?(offd_i_I[i+1] - offd_i_I[i]):0;
         for (int j=0; j<offd_r_I[i+1] - offd_r_I[i]; j++)
         {
            offd_J[offd_I[i] + j] =
               cinvmap[cmapa[cmap_r[offd_r_J[offd_r_I[i] + j]]]];
            offd_D[offd_I[i] + j] = offd_r_D[offd_r_I[i] + j];

            offd_J[offd_I[i+nrows] + off_i + j] =
               cinvmap[cmapb[cmap_r[offd_r_J[offd_r_I[i] + j]]]];
            offd_D[offd_I[i+nrows] + off_i + j] =
               factor * offd_r_D[offd_r_I[i] + j];
         }
      }
      if (offd_i_I)
      {
         const int off_r = (offd_r_I)?(offd_r_I[i+1] - offd_r_I[i]):0;
         for (int j=0; j<offd_i_I[i+1] - offd_i_I[i]; j++)
         {
            offd_J[offd_I[i] + off_r + j] =
               cinvmap[cmapb[cmap_i[offd_i_J[offd_i_I[i] + j]]]];
            offd_D[offd_I[i] + off_r + j] = -offd_i_D[offd_i_I[i] + j];

            offd_J[offd_I[i+nrows] + j] =
               cinvmap[cmapa[cmap_i[offd_i_J[offd_i_I[i] + j]]]];
            offd_D[offd_I[i+nrows] + j] = factor * offd_i_D[offd_i_I[i] + j];
         }
      }
   }

   // Construct the combined matrix
   HypreParMatrix * A = new HypreParMatrix(comm_,
                                           2 * global_num_rows,
                                           2 * global_num_cols,
                                           row_starts, col_starts,
                                           diag_I, diag_J, diag_D,
                                           offd_I, offd_J, offd_D,
                                           2 * num_cols_offd, cmap);

   // Give the new matrix ownership of its internal arrays
   A->SetOwnerFlags(-1,-1,-1);
   hypre_CSRMatrixSetDataOwner(((hypre_ParCSRMatrix*)(*A))->diag,1);
   hypre_CSRMatrixSetDataOwner(((hypre_ParCSRMatrix*)(*A))->offd,1);
   hypre_ParCSRMatrixSetRowStartsOwner((hypre_ParCSRMatrix*)(*A),1);
   hypre_ParCSRMatrixSetColStartsOwner((hypre_ParCSRMatrix*)(*A),1);

   return A;
}

void
ComplexHypreParMatrix::getColStartStop(const HypreParMatrix * A_r,
                                       const HypreParMatrix * A_i,
                                       int & num_recv_procs,
                                       HYPRE_Int *& offd_col_start_stop) const
{
   hypre_ParCSRCommPkg * comm_pkg_r =
      (A_r) ? hypre_ParCSRMatrixCommPkg((hypre_ParCSRMatrix*)(*A_r)) : NULL;
   hypre_ParCSRCommPkg * comm_pkg_i =
      (A_i) ? hypre_ParCSRMatrixCommPkg((hypre_ParCSRMatrix*)(*A_i)) : NULL;

   std::set<HYPRE_Int> send_procs, recv_procs;
   if ( comm_pkg_r )
   {
      for (HYPRE_Int i=0; i<comm_pkg_r->num_sends; i++)
      {
         send_procs.insert(comm_pkg_r->send_procs[i]);
      }
      for (HYPRE_Int i=0; i<comm_pkg_r->num_recvs; i++)
      {
         recv_procs.insert(comm_pkg_r->recv_procs[i]);
      }
   }
   if ( comm_pkg_i )
   {
      for (HYPRE_Int i=0; i<comm_pkg_i->num_sends; i++)
      {
         send_procs.insert(comm_pkg_i->send_procs[i]);
      }
      for (HYPRE_Int i=0; i<comm_pkg_i->num_recvs; i++)
      {
         recv_procs.insert(comm_pkg_i->recv_procs[i]);
      }
   }

   num_recv_procs = (int)recv_procs.size();

   HYPRE_Int loc_start_stop[2];
   offd_col_start_stop = new HYPRE_Int[2 * num_recv_procs];

   const HYPRE_Int * row_part = (A_r) ? A_r->RowPart() :
                                ((A_i) ? A_i->RowPart() : NULL);

   int row_part_ind = (HYPRE_AssumedPartitionCheck()) ? 0 : myid_;
   loc_start_stop[0] = row_part[row_part_ind];
   loc_start_stop[1] = row_part[row_part_ind+1];

   MPI_Request * req = new MPI_Request[send_procs.size()+recv_procs.size()];
   MPI_Status * stat = new MPI_Status[send_procs.size()+recv_procs.size()];
   int send_count = 0;
   int recv_count = 0;
   int tag = 0;

   std::set<HYPRE_Int>::iterator sit;
   for (sit=send_procs.begin(); sit!=send_procs.end(); sit++)
   {
      MPI_Isend(loc_start_stop, 2, HYPRE_MPI_INT,
                *sit, tag, comm_, &req[send_count]);
      send_count++;
   }
   for (sit=recv_procs.begin(); sit!=recv_procs.end(); sit++)
   {
      MPI_Irecv(&offd_col_start_stop[2*recv_count], 2, HYPRE_MPI_INT,
                *sit, tag, comm_, &req[send_count+recv_count]);
      recv_count++;
   }

   MPI_Waitall(send_count+recv_count, req, stat);

   delete [] req;
   delete [] stat;
}


#ifdef MFEM_USE_MUMPS

void ComplexMUMPSSolver::SetOperator(const Operator &op)
{
   auto APtr = dynamic_cast<const ComplexHypreParMatrix *>(&op);

   MFEM_VERIFY(APtr, "Not compatible matrix type");
   height = op.Height();
   width = op.Width();

   conv = APtr->GetConvention();
   comm = APtr->real().GetComm();
   MPI_Comm_size(comm, &numProcs);
   MPI_Comm_rank(comm, &myid);

   auto parcsr_op_r = (hypre_ParCSRMatrix *) const_cast<HypreParMatrix &>
                      (APtr->real());
   auto parcsr_op_i = (hypre_ParCSRMatrix *) const_cast<HypreParMatrix &>
                      (APtr->imag());

   hypre_CSRMatrix *csr_op_r = hypre_MergeDiagAndOffd(parcsr_op_r);
   hypre_CSRMatrix *csr_op_i = hypre_MergeDiagAndOffd(parcsr_op_i);
#if MFEM_HYPRE_VERSION >= 21600
   hypre_CSRMatrixBigJtoJ(csr_op_r);
   hypre_CSRMatrixBigJtoJ(csr_op_i);
#endif
   MFEM_VERIFY(csr_op_r->num_nonzeros == csr_op_i->num_nonzeros,
               "Incompatible sparsity partters");

   int *Iptr = csr_op_r->i;
   int *Jptr = csr_op_r->j;
   int n_loc = csr_op_r->num_rows;
   row_start = parcsr_op_i->first_row_index;
   MUMPS_INT8 nnz = csr_op_r->num_nonzeros;

   int * I = new int[nnz];
   int * J = new int[nnz];

   // Fill in I and J arrays for
   // COO format in 1-based indexing
   int k = 0;
   double * data_r = csr_op_r->data;
   double * data_i = csr_op_i->data;
   mumps_double_complex *zdata = new mumps_double_complex[nnz];
   for (int i = 0; i < n_loc; i++)
   {
      for (int j = Iptr[i]; j < Iptr[i + 1]; j++)
      {
         I[k] = row_start + i + 1;
         J[k] = Jptr[k] + 1;
         zdata[k].r = data_r[k];
         zdata[k].i = data_i[k];
         k++;
      }
   }

   // new MUMPS object
   if (id)
   {
      id->job = -2;
      zmumps_c(id);
      delete id;
   }
   id = new ZMUMPS_STRUC_C;
   // C to Fortran communicator
   id->comm_fortran = (MUMPS_INT) MPI_Comm_c2f(comm);

   // Host is involved in computation
   id->par = 1;

   id->sym = 0;

   // MUMPS init
   id->job = -1;
   zmumps_c(id);

   // Set MUMPS default parameters
   SetParameters();

   id->n = parcsr_op_r->global_num_rows;

   id->nnz_loc = nnz;

   id->irn_loc = I;

   id->jcn_loc = J;

   id->a_loc = zdata;

   // MUMPS Analysis
   id->job = 1;
   zmumps_c(id);

   // MUMPS Factorization
   id->job = 2;
   zmumps_c(id);

   hypre_CSRMatrixDestroy(csr_op_r);
   hypre_CSRMatrixDestroy(csr_op_i);
   delete [] I;
   delete [] J;
   delete [] zdata;

#if MFEM_MUMPS_VERSION >= 530
   delete [] irhs_loc;
   irhs_loc = new int[n_loc];
   for (int i = 0; i < n_loc; i++)
   {
      irhs_loc[i] = row_start + i + 1;
   }
   row_starts.SetSize(numProcs);
   MPI_Allgather(&row_start, 1, MPI_INT, row_starts, 1, MPI_INT, comm);
#else
   if (myid == 0)
   {
      delete [] rhs_glob;
      delete [] recv_counts;
      global_num_rows = parcsr_op_r->global_num_rows;
      rhs_glob = new mumps_double_complex[global_num_rows];
      recv_counts = new int[numProcs];
   }
   MPI_Gather(&n_loc, 1, MPI_INT, recv_counts, 1, MPI_INT, 0, comm);
   if (myid == 0)
   {
      delete [] displs;
      displs = new int[numProcs];
      displs[0] = 0;
      int s = 0;
      for (int k = 0; k < numProcs-1; k++)
      {
         s += recv_counts[k];
         displs[k+1] = s;
      }
   }
#endif
}

void ComplexMUMPSSolver::Mult(const Vector &x, Vector &y) const
{
   int n = x.Size()/2;
   double * datax = x.GetData();
   double * datay = y.GetData();
   Vector ximag;
   if (conv == ComplexOperator::Convention::BLOCK_SYMMETRIC)
   {
      ximag.SetDataAndSize(&datax[n],n);
      ximag *=-1.0;
   }
#if MFEM_MUMPS_VERSION >= 530
   id->nloc_rhs = n;
   id->lrhs_loc = n;
   mumps_double_complex *zx = new mumps_double_complex[n];
   for (int i = 0; i<n; i++)
   {
      zx[i].r = x[i];
      zx[i].i = x[n+i];
   }
   id->rhs_loc = zx;
   id->irhs_loc = irhs_loc;

   id->lsol_loc = id->MUMPSC_INFO(23);
   id->isol_loc = new int[id->MUMPSC_INFO(23)];
   id->sol_loc = new mumps_double_complex[id->MUMPSC_INFO(23)];

   // MUMPS solve
   id->job = 3;
   zmumps_c(id);

   double *zy = new double[2*id->MUMPSC_INFO(23)];
   for (int i = 0; i<id->MUMPSC_INFO(23); i++)
   {
      zy[i] = id->sol_loc[i].r;
      zy[id->MUMPSC_INFO(23)+i] = id->sol_loc[i].i;
   }

   RedistributeSol(id->isol_loc, zy, y.GetData());

   delete [] zy;
   delete [] zx;
   delete [] id->sol_loc;
   delete [] id->isol_loc;
#else
   // real
   MFEM_ABORT("Not implemented yet");
   double * rhs_glob_r = nullptr;
   double * rhs_glob_i = nullptr;
   if (myid == 0)
   {
      rhs_glob_r = new double[global_num_rows];
      rhs_glob_i = new double[global_num_rows];
   }
   MPI_Gatherv(datax, n, MPI_DOUBLE,
               rhs_glob_r, recv_counts,
               displs, MPI_DOUBLE, 0, comm);
   MPI_Gatherv(&datax[n], n, MPI_DOUBLE,
               rhs_glob_i, recv_counts,
               displs, MPI_DOUBLE, 0, comm);

   if (myid == 0)
   {
      for (int i = 0; i<global_num_rows; i++)
      {
         rhs_glob[i].r = rhs_glob_r[i];
         rhs_glob[i].i = rhs_glob_i[i];
      }
      id->rhs = rhs_glob;
   }
   // MUMPS solve
   id->job = 3;
   zmumps_c(id);
   if (myid == 0)
   {
      for (int i = 0; i<global_num_rows; i++)
      {
         rhs_glob_r[i] = rhs_glob[i].r;
         rhs_glob_i[i] = rhs_glob[i].i;
      }
   }
   MPI_Scatterv(rhs_glob_r, recv_counts, displs,
                MPI_DOUBLE, datay, n,
                MPI_DOUBLE, 0, comm);
   MPI_Scatterv(rhs_glob_i, recv_counts, displs,
                MPI_DOUBLE, &datay[n], n,
                MPI_DOUBLE, 0, comm);

   if (myid == 0)
   {
      delete [] rhs_glob_r;
      delete [] rhs_glob_i;
   }

#endif

   if (conv == ComplexOperator::Convention::BLOCK_SYMMETRIC)
   {
      ximag *=-1.0;
   }
}

void ComplexMUMPSSolver::SetPrintLevel(int print_lvl)
{
   print_level = print_lvl;
}

ComplexMUMPSSolver::~ComplexMUMPSSolver()
{
   if (id)
   {
#if MFEM_MUMPS_VERSION >= 530
      delete [] irhs_loc;
#else
      delete [] recv_counts;
      delete [] displs;
      delete [] rhs_glob;
#endif
      id->job = -2;
      zmumps_c(id);
      delete id;
   }
}

void ComplexMUMPSSolver::SetParameters()
{
   // output stream for error messages
   id->ICNTL(1) = 6;
   // output stream for diagnosting printing local to each proc
   id->ICNTL(2) = 6;
   // output stream for global info
   id->ICNTL(3) = 6;
   // Level of error printing
   id->ICNTL(4) = print_level;
   //input matrix format (assembled)
   id->ICNTL(5) = 0;
   // Use A or A^T
   id->ICNTL(9) = 1;
   // Iterative refinement (disabled)
   id->ICNTL(10) = 0;
   // Error analysis-statistics (disabled)
   id->ICNTL(11) = 0;
   // Use of ScaLAPACK (Parallel factorization on root)
   id->ICNTL(13) = 0;
   // Percentage increase of estimated workspace (default = 20%)
   id->ICNTL(14) = 20;
   // Number of OpenMP threads (default)
   id->ICNTL(16) = 0;
   // Matrix input format (distributed)
   id->ICNTL(18) = 3;
   // Schur complement (no Schur complement matrix returned)
   id->ICNTL(19) = 0;

#if MFEM_MUMPS_VERSION >= 530
   // Distributed RHS
   id->ICNTL(20) = 10;
   // Distributed Sol
   id->ICNTL(21) = 1;
#else
   // Centralized RHS
   id->ICNTL(20) = 0;
   // Centralized Sol
   id->ICNTL(21) = 0;
#endif
   // Out of core factorization and solve (disabled)
   id->ICNTL(22) = 0;
   // Max size of working memory (default = based on estimates)
   id->ICNTL(23) = 0;
}

#if MFEM_MUMPS_VERSION >= 530
int ComplexMUMPSSolver::GetRowRank(int i, const Array<int> &row_starts_) const
{
   if (row_starts_.Size() == 1)
   {
      return 0;
   }
   auto up = std::upper_bound(row_starts_.begin(), row_starts_.end(), i);
   return std::distance(row_starts_.begin(), up) - 1;
}

void ComplexMUMPSSolver::RedistributeSol(const int * row_map,
                                         const double * x, double * y) const
{
   int size = id->MUMPSC_INFO(23);
   int n = id->nloc_rhs;
   int * send_count = new int[numProcs]();
   for (int i = 0; i < size; i++)
   {
      int j = row_map[i] - 1;
      int row_rank = GetRowRank(j, row_starts);
      if (myid == row_rank) { continue; }
      send_count[row_rank]++;
   }

   int * recv_count = new int[numProcs];
   MPI_Alltoall(send_count, 1, MPI_INT, recv_count, 1, MPI_INT, comm);

   int * send_displ = new int [numProcs]; send_displ[0] = 0;
   int * recv_displ = new int [numProcs]; recv_displ[0] = 0;
   int sbuff_size = send_count[numProcs-1];
   int rbuff_size = recv_count[numProcs-1];
   for (int k = 0; k < numProcs - 1; k++)
   {
      send_displ[k + 1] = send_displ[k] + send_count[k];
      recv_displ[k + 1] = recv_displ[k] + recv_count[k];
      sbuff_size += send_count[k];
      rbuff_size += recv_count[k];
   }

   int * sendbuf_index = new int[sbuff_size];
   double * sendbuf_values_r = new double[sbuff_size];
   double * sendbuf_values_i = new double[sbuff_size];
   int * soffs = new int[numProcs]();

   for (int i = 0; i < size; i++)
   {
      int j = row_map[i] - 1;
      int row_rank = GetRowRank(j, row_starts);
      if (myid == row_rank)
      {
         int local_index = j - row_start;
         y[local_index] = x[i];
         y[local_index+n] = x[i+size];
      }
      else
      {
         int k = send_displ[row_rank] + soffs[row_rank];
         sendbuf_index[k] = j;
         sendbuf_values_r[k] = x[i];
         sendbuf_values_i[k] = x[i+size];
         soffs[row_rank]++;
      }
   }

   int * recvbuf_index = new int[rbuff_size];
   double * recvbuf_values_r = new double[rbuff_size];
   double * recvbuf_values_i = new double[rbuff_size];
   MPI_Alltoallv(sendbuf_index,
                 send_count,
                 send_displ,
                 MPI_INT,
                 recvbuf_index,
                 recv_count,
                 recv_displ,
                 MPI_INT,
                 comm);
   MPI_Alltoallv(sendbuf_values_r,
                 send_count,
                 send_displ,
                 MPI_DOUBLE,
                 recvbuf_values_r,
                 recv_count,
                 recv_displ,
                 MPI_DOUBLE,
                 comm);
   MPI_Alltoallv(sendbuf_values_i,
                 send_count,
                 send_displ,
                 MPI_DOUBLE,
                 recvbuf_values_i,
                 recv_count,
                 recv_displ,
                 MPI_DOUBLE,
                 comm);

   // Unpack recv buffer
   for (int i = 0; i < rbuff_size; i++)
   {
      int local_index = recvbuf_index[i] - row_start;
      y[local_index] = recvbuf_values_r[i];
      y[local_index+n] = recvbuf_values_i[i];
   }

   delete [] recvbuf_values_r;
   delete [] recvbuf_values_i;
   delete [] recvbuf_index;
   delete [] soffs;
   delete [] sendbuf_values_r;
   delete [] sendbuf_values_i;
   delete [] sendbuf_index;
   delete [] recv_displ;
   delete [] send_displ;
   delete [] recv_count;
   delete [] send_count;
}
#endif // MUMPS VERSION
#endif // MFEM_USE_MUMPS
#endif // MFEM_USE_MPI

}
