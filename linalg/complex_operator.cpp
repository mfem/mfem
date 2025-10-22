// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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

   const real_t  *D_r = (A_r)?A_r->GetData():NULL;
   const real_t  *D_i = (A_i)?A_i->GetData():NULL;

   const int    nnz_r = (I_r)?I_r[nrows]:0;
   const int    nnz_i = (I_i)?I_i[nrows]:0;
   const int    nnz   = 2 * (nnz_r + nnz_i);

   int    *I = Memory<int>(this->Height()+1);
   int    *J = Memory<int>(nnz);
   real_t *D = Memory<real_t>(nnz);

   const real_t factor = (convention_ == HERMITIAN) ? 1.0 : -1.0;

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
   const real_t * Ax = mat->real().HostReadData();
   const real_t * Az = mat->imag().HostReadData();

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
   real_t * datax = x.GetData();
   real_t * datab = b.GetData();

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
   real_t * datax = x.GetData();
   real_t * datab = b.GetData();

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

   HYPRE_BigInt global_num_rows_r = (A_r) ? A_r->GetGlobalNumRows() : 0;
   HYPRE_BigInt global_num_rows_i = (A_i) ? A_i->GetGlobalNumRows() : 0;
   HYPRE_BigInt global_num_rows = std::max(global_num_rows_r,
                                           global_num_rows_i);

   HYPRE_BigInt global_num_cols_r = (A_r) ? A_r->GetGlobalNumCols() : 0;
   HYPRE_BigInt global_num_cols_i = (A_i) ? A_i->GetGlobalNumCols() : 0;
   HYPRE_BigInt global_num_cols = std::max(global_num_cols_r,
                                           global_num_cols_i);

   int row_starts_size = (HYPRE_AssumedPartitionCheck()) ? 2 : nranks_ + 1;
   HYPRE_BigInt * row_starts = mfem_hypre_CTAlloc_host(HYPRE_BigInt,
                                                       row_starts_size);
   HYPRE_BigInt * col_starts = mfem_hypre_CTAlloc_host(HYPRE_BigInt,
                                                       row_starts_size);

   const HYPRE_BigInt * row_starts_z = (A_r) ? A_r->RowPart() :
                                       ((A_i) ? A_i->RowPart() : NULL);
   const HYPRE_BigInt * col_starts_z = (A_r) ? A_r->ColPart() :
                                       ((A_i) ? A_i->ColPart() : NULL);

   for (int i = 0; i < row_starts_size; i++)
   {
      row_starts[i] = 2 * row_starts_z[i];
      col_starts[i] = 2 * col_starts_z[i];
   }

   SparseMatrix diag_r, diag_i, offd_r, offd_i;
   HYPRE_BigInt * cmap_r = NULL, * cmap_i = NULL;

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
   std::set<HYPRE_BigInt> cset;
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

   const real_t * diag_r_D = (A_r) ? diag_r.GetData() : NULL;
   const real_t * diag_i_D = (A_i) ? diag_i.GetData() : NULL;

   int diag_r_nnz = (diag_r_I) ? diag_r_I[nrows] : 0;
   int diag_i_nnz = (diag_i_I) ? diag_i_I[nrows] : 0;
   int diag_nnz = 2 * (diag_r_nnz + diag_i_nnz);

   // Extract pointers to the various CSR arrays of the off-diagonal blocks
   const int * offd_r_I = (A_r) ? offd_r.GetI() : NULL;
   const int * offd_i_I = (A_i) ? offd_i.GetI() : NULL;

   const int * offd_r_J = (A_r) ? offd_r.GetJ() : NULL;
   const int * offd_i_J = (A_i) ? offd_i.GetJ() : NULL;

   const real_t * offd_r_D = (A_r) ? offd_r.GetData() : NULL;
   const real_t * offd_i_D = (A_i) ? offd_i.GetData() : NULL;

   int offd_r_nnz = (offd_r_I) ? offd_r_I[nrows] : 0;
   int offd_i_nnz = (offd_i_I) ? offd_i_I[nrows] : 0;
   int offd_nnz = 2 * (offd_r_nnz + offd_i_nnz);

   // Allocate CSR arrays for the combined matrix
   HYPRE_Int * diag_I = mfem_hypre_CTAlloc_host(HYPRE_Int, 2 * nrows + 1);
   HYPRE_Int * diag_J = mfem_hypre_CTAlloc_host(HYPRE_Int, diag_nnz);
   real_t    * diag_D = mfem_hypre_CTAlloc_host(real_t, diag_nnz);

   HYPRE_Int * offd_I = mfem_hypre_CTAlloc_host(HYPRE_Int, 2 * nrows + 1);
   HYPRE_Int * offd_J = mfem_hypre_CTAlloc_host(HYPRE_Int, offd_nnz);
   real_t    * offd_D = mfem_hypre_CTAlloc_host(real_t, offd_nnz);
   HYPRE_BigInt * cmap = mfem_hypre_CTAlloc_host(HYPRE_BigInt,
                                                 2 * num_cols_offd);

   // Fill the CSR arrays for the diagonal portion of the matrix
   const real_t factor = (convention_ == HERMITIAN) ? 1.0 : -1.0;

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
   HYPRE_BigInt * offd_col_start_stop = NULL;
   this->getColStartStop(A_r, A_i, num_recv_procs, offd_col_start_stop);

   std::set<HYPRE_BigInt>::iterator sit;
   std::map<HYPRE_BigInt,HYPRE_BigInt> cmapa, cmapb, cinvmap;
   for (sit=cset.begin(); sit!=cset.end(); sit++)
   {
      HYPRE_BigInt col_orig = *sit;
      HYPRE_BigInt col_2x2  = -1;
      HYPRE_BigInt col_size = 0;
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

   {
      std::map<HYPRE_BigInt, HYPRE_BigInt>::iterator mit;
      HYPRE_BigInt i = 0;
      for (mit=cinvmap.begin(); mit!=cinvmap.end(); mit++, i++)
      {
         mit->second = i;
         cmap[i] = mit->first;
      }
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
                                           2 * num_cols_offd, cmap,
                                           true);

#if MFEM_HYPRE_VERSION <= 22200
   // Give the new matrix ownership of row_starts and col_starts
   hypre_ParCSRMatrix *hA = (hypre_ParCSRMatrix*)(*A);

   hypre_ParCSRMatrixSetRowStartsOwner(hA,1);
   hypre_ParCSRMatrixSetColStartsOwner(hA,1);
#else
   mfem_hypre_TFree_host(row_starts);
   mfem_hypre_TFree_host(col_starts);
#endif

   return A;
}

void
ComplexHypreParMatrix::getColStartStop(const HypreParMatrix * A_r,
                                       const HypreParMatrix * A_i,
                                       int & num_recv_procs,
                                       HYPRE_BigInt *& offd_col_start_stop
                                      ) const
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

   HYPRE_BigInt loc_start_stop[2];
   offd_col_start_stop = new HYPRE_BigInt[2 * num_recv_procs];

   const HYPRE_BigInt * row_part = (A_r) ? A_r->RowPart() :
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
      MPI_Isend(loc_start_stop, 2, HYPRE_MPI_BIG_INT,
                *sit, tag, comm_, &req[send_count]);
      send_count++;
   }
   for (sit=recv_procs.begin(); sit!=recv_procs.end(); sit++)
   {
      MPI_Irecv(&offd_col_start_stop[2*recv_count], 2, HYPRE_MPI_BIG_INT,
                *sit, tag, comm_, &req[send_count+recv_count]);
      recv_count++;
   }

   MPI_Waitall(send_count+recv_count, req, stat);

   delete [] req;
   delete [] stat;
}

#endif // MFEM_USE_MPI

}
