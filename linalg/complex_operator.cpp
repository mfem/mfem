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
#include "../general/communication.hpp"
#ifdef MFEM_USE_MPI
#include "blockoperator.hpp"
#endif
#include <set>
#include <map>
#include <unordered_map>
#include <vector>
#include <limits>

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

#ifdef MFEM_USE_MPI
ComplexHypreParMatrix * ComplexOperator::AsComplexHypreParMatrix() const
{
   HypreParMatrix *Ar = nullptr;
   HypreParMatrix *Ai = nullptr;
   bool own_r = false;
   bool own_i = false;

   if (auto *Ahr = dynamic_cast<const HypreParMatrix*>(&real()))
   {
      Ar = const_cast<HypreParMatrix*>(Ahr);
   }
   else if (auto *Br = dynamic_cast<const BlockOperator*>(&real()))
   {
      Ar = Br->GetMonolithicHypreParMatrix();
      own_r = true;
   }
   else
   {
      MFEM_ABORT("Real part is neither HypreParMatrix nor BlockOperator.");
   }

   if (auto *Ahi = dynamic_cast<const HypreParMatrix*>(&imag()))
   {
      Ai = const_cast<HypreParMatrix*>(Ahi);
   }
   else if (auto *Bi = dynamic_cast<const BlockOperator*>(&imag()))
   {
      Ai = Bi->GetMonolithicHypreParMatrix();
      own_i = true;
   }
   else
   {
      MFEM_ABORT("Imag part is neither HypreParMatrix nor BlockOperator.");
   }

   return new ComplexHypreParMatrix(Ar, Ai, own_r, own_i, GetConvention());
}



#endif




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

static Array<int> Twice(const Array<int> &offs)
{
   Array<int> arrayout(offs.Size());
   for (int i = 0; i < offs.Size(); i++) { arrayout[i] = 2 * offs[i]; }
   return arrayout;
}


ComplexBlockOperator::ComplexBlockOperator(const ComplexOperator &A)
   : BlockOperator(
        Twice(dynamic_cast<const BlockOperator&>(A.real()).RowOffsets()),
        Twice(dynamic_cast<const BlockOperator&>(A.real()).ColOffsets()))
{
   const BlockOperator *Ar = dynamic_cast<const BlockOperator*>(&A.real());
   const BlockOperator *Ai = dynamic_cast<const BlockOperator*>(&A.imag());

   MFEM_VERIFY(Ar && Ai,
               "ComplexBlockOperator: expected ComplexOperator with BlockOperator Re/Im.");
   MFEM_VERIFY(Ar->NumRowBlocks() == Ai->NumRowBlocks() &&
               Ar->NumColBlocks() == Ai->NumColBlocks(),
               "ComplexBlockOperator: Re/Im block layouts mismatch.");

   // Populate this BlockOperator (base) with ComplexOperator blocks.
   for (int i = 0; i < Ar->NumRowBlocks(); ++i)
   {
      for (int j = 0; j < Ar->NumColBlocks(); ++j)
      {
         HypreParMatrix *Rij = nullptr;
         if (!Ar->IsZeroBlock(i, j))
         {
            Rij = const_cast<HypreParMatrix*>
                  (dynamic_cast<const HypreParMatrix*>(&Ar->GetBlock(i, j)));
         }

         HypreParMatrix *Iij = nullptr;
         if (!Ai->IsZeroBlock(i, j))
         {
            Iij = const_cast<HypreParMatrix*>
                  (dynamic_cast<const HypreParMatrix*>(&Ai->GetBlock(i, j)));
         }

         // MFEM_VERIFY((Rij && Iij) || (!Rij && !Iij),
         //             "ComplexBlockOperator: inconsistent sparsity at block ("
         //             << i << "," << j << ").");

         if (Rij)
         {
            auto *Cij = new ComplexHypreParMatrix(Rij, Iij, false, false,
                                                  A.GetConvention());
            SetBlock(i, j, Cij);
         }
      }
   }
}

void ComplexBlockOperator::BlockComplexToComplexBlock(const Vector &xin,
                                                      Vector &xout) const
{
   MFEM_VERIFY(xout.Size() == xin.Size(),
               "BlockComplexToComplexBlock: size mismatch (xout != xin).");
   MFEM_VERIFY(xin.Size() % 2 == 0,
               "BlockComplexToComplexBlock: expected even-sized vector (2*N).");

   // Decide whether xin is domain-sized or range-sized by matching doubled offsets.
   const int twoNcols = ColOffsets().Last();
   const int twoNrows = RowOffsets().Last();
   const Array<int> *doffs = nullptr; // doubled offsets to use

   if (xin.Size() == twoNcols) { doffs = &ColOffsets(); }
   else if (xin.Size() == twoNrows) { doffs = &RowOffsets(); }
   else
   {
      MFEM_ABORT("BlockComplexToComplexBlock: vector size does not match "
                 "either doubled domain or range size.");
   }

   // xin layout: [ Re(all 0..N-1), Im(all 0..N-1) ], where N = doffs->Last()/2
   const int N = doffs->Last() / 2;

   int pos = 0; // position in per-block layout (xout), but note xout is also size 2*N
   for (int b = 0; b < doffs->Size() - 1; b++)
   {
      // Doubled block segment in this BlockOperator is [(*doffs)[b], (*doffs)[b+1])
      // The *base* (undoubled) start/length are:
      const int s_base = (*doffs)[b] / 2;
      const int len    = ((*doffs)[b+1] - (*doffs)[b]) / 2;

      // Write per-block Re then Im, contiguous
      for (int k = 0; k < len; k++) { xout[pos + k]       = xin[s_base + k]; }
      for (int k = 0; k < len; k++) { xout[pos + len + k] = xin[N + s_base + k]; }

      pos += 2 * len;
   }
}

void ComplexBlockOperator::ComplexBlockToBlockComplex(const Vector &xin,
                                                      Vector &xout) const
{
   MFEM_VERIFY(xout.Size() == xin.Size(),
               "ComplexBlockToBlockComplex: size mismatch (xout != xin).");
   MFEM_VERIFY(xin.Size() % 2 == 0,
               "ComplexBlockToBlockComplex: expected even-sized vector (2*N).");

   const int Ncols = ColOffsets().Last();
   const int Nrows = RowOffsets().Last();
   const Array<int> *doffs = nullptr;

   if (xin.Size() == Ncols) { doffs = &ColOffsets(); }
   else if (xin.Size() == Nrows) { doffs = &RowOffsets(); }
   else
   {
      MFEM_ABORT("ComplexBlockToBlockComplex: vector size does not match "
                 "either doubled domain or range size.");
   }

   const int N = doffs->Last() / 2;

   int pos = 0; // position in per-block layout (xin)
   for (int b = 0; b < doffs->Size() - 1; b++)
   {
      const int s_base = (*doffs)[b] / 2;
      const int len    = ((*doffs)[b+1] - (*doffs)[b]) / 2;

      // Read per-block Re then Im, scatter to stacked [Re(all); Im(all)]
      for (int k = 0; k < len; k++) { xout[s_base + k]     = xin[pos + k]; }
      for (int k = 0; k < len; k++) { xout[N + s_base + k] = xin[pos + len + k]; }

      pos += 2 * len;
   }
}


#ifdef MFEM_USE_COMPLEX_MUMPS

// Macro so indices match MUMPS documentation
#define MUMPS_ICNTL(I) icntl[(I) - 1]
#define MUMPS_CNTL(I)  cntl[(I) - 1]
#define MUMPS_INFO(I)  info[(I) - 1]
#define MUMPS_INFOG(I) infog[(I) - 1]

ComplexMUMPSSolver::ComplexMUMPSSolver(MPI_Comm comm_)
{
   Init(comm_);
}

ComplexMUMPSSolver::ComplexMUMPSSolver(const Operator &op)
{
   auto APtr = dynamic_cast<const ComplexHypreParMatrix *>(&op);
   MFEM_VERIFY(APtr, "Not a compatible matrix type for ComplexMUMPSSolver");
   SetOperator(op);
}

void ComplexMUMPSSolver::Init(MPI_Comm comm_)
{
   comm = comm_;
   MPI_Comm_size(comm, &numProcs);
   MPI_Comm_rank(comm, &myid);

   print_level = 2;
   row_start = 0;

   id = nullptr;

#if MFEM_MUMPS_VERSION >= 530
   irhs_loc = nullptr;
   isol_loc = nullptr;
   rhs_loc  = nullptr;
   sol_loc  = nullptr;
#else
   global_num_rows = 0;
   recv_counts = nullptr;
   displs = nullptr;
   rhs_glob = nullptr;
   rhs_glob_r = nullptr;
   rhs_glob_i = nullptr;
#endif
}

ComplexMUMPSSolver::~ComplexMUMPSSolver()
{
#if MFEM_MUMPS_VERSION >= 530
   delete [] irhs_loc;
   delete [] isol_loc;
   delete [] rhs_loc;
   delete [] sol_loc;
#else
   delete [] recv_counts;
   delete [] displs;
   delete [] rhs_glob;
   delete [] rhs_glob_r;
   delete [] rhs_glob_i;
#endif

   if (id)
   {
      id->job = -2;
      mumps_call();
      delete id;
      id = nullptr;
   }
}

void ComplexMUMPSSolver::SetOperator(const Operator &op)
{
   auto APtr = dynamic_cast<const ComplexHypreParMatrix *>(&op);
   MFEM_VERIFY(APtr, "Not compatible matrix type for ComplexMUMPSSolver");

   height = op.Height();
   width  = op.Width();

   const HypreParMatrix *Ar = (APtr->hasRealPart()) ? &APtr->real() : nullptr;
   const HypreParMatrix *Ai = (APtr->hasImagPart()) ? &APtr->imag() : nullptr;

   MFEM_VERIFY(Ar || Ai, "ComplexMUMPSSolver: both real and imag parts are null.");

   // Pick communicator from the non-null part
   MPI_Comm op_comm = (Ar ? Ar->GetComm() : Ai->GetComm());

   // Comm setup/check
   if (comm == MPI_COMM_NULL) { Init(op_comm); }
   else
   {
      int cmp = MPI_UNEQUAL;
      MPI_Comm_compare(comm, op_comm, &cmp);
      MFEM_VERIFY(cmp != MPI_UNEQUAL, "MPI Comm mismatch");
   }

   // HostRead only if non-null
   if (Ar) { Ar->HostRead(); }
   if (Ai) { Ai->HostRead(); }

   // hypre parcsr pointers
   hypre_ParCSRMatrix *parcsr_op_r = nullptr;
   hypre_ParCSRMatrix *parcsr_op_i = nullptr;

   if (Ar) { parcsr_op_r = (hypre_ParCSRMatrix*) const_cast<HypreParMatrix&>(*Ar); }
   if (Ai) { parcsr_op_i = (hypre_ParCSRMatrix*) const_cast<HypreParMatrix&>(*Ai); }

   // Merge diag+offd for whichever exists
   hypre_CSRMatrix *csr_op_r = nullptr;
   hypre_CSRMatrix *csr_op_i = nullptr;

   if (parcsr_op_r) { csr_op_r = hypre_MergeDiagAndOffd(parcsr_op_r); }
   if (parcsr_op_i) { csr_op_i = hypre_MergeDiagAndOffd(parcsr_op_i); }

#if MFEM_HYPRE_VERSION >= 21600
   if (csr_op_r) { hypre_CSRMatrixBigJtoJ(csr_op_r); }
   if (csr_op_i) { hypre_CSRMatrixBigJtoJ(csr_op_i); }
#endif

   // Determine local/global sizes and row_start from an existing part
   const int n_loc = internal::to_int((csr_op_r ? csr_op_r->num_rows :
                                       csr_op_i->num_rows));
   row_start = internal::to_int((parcsr_op_r ? parcsr_op_r->first_row_index
                                 : parcsr_op_i->first_row_index));
   const int global_n = internal::to_int((parcsr_op_r ?
                                          parcsr_op_r->global_num_rows
                                          : parcsr_op_i->global_num_rows));

   // Use nullptr checks
   const int *Ir = csr_op_r ? csr_op_r->i : nullptr;
   const int *Jr = csr_op_r ? csr_op_r->j : nullptr;
   const real_t *Vr = csr_op_r ? (const real_t*)csr_op_r->data : nullptr;

   const int *Ii = csr_op_i ? csr_op_i->i : nullptr;
   const int *Ji = csr_op_i ? csr_op_i->j : nullptr;
   const real_t *Vi = csr_op_i ? (const real_t*)csr_op_i->data : nullptr;

   // Build union COO
   std::vector<int> Icoo, Jcoo;
   std::vector<mumps_complex_t> Zcoo;

   size_t nnz_r = csr_op_r ? (size_t)csr_op_r->num_nonzeros : 0;
   size_t nnz_i = csr_op_i ? (size_t)csr_op_i->num_nonzeros : 0;
   Icoo.reserve(nnz_r + nnz_i);
   Jcoo.reserve(nnz_r + nnz_i);
   Zcoo.reserve(nnz_r + nnz_i);

   BuildUnionCOO(n_loc, row_start, Ir, Jr, Vr, Ii, Ji, Vi, Icoo, Jcoo, Zcoo);

   const int nnz = (int)Icoo.size();
   int *I = new int[nnz];
   int *J = new int[nnz];
   mumps_complex_t *A = new mumps_complex_t[nnz];

   std::copy(Icoo.begin(), Icoo.end(), I);
   std::copy(Jcoo.begin(), Jcoo.end(), J);
   std::copy(Zcoo.begin(), Zcoo.end(), A);

   // New ComplexMUMPS object or reuse an existing one
   if (!id || !reorder_reuse)
   {
      if (id)
      {
         id->job = -2;
         mumps_call();
         delete id;
         id = nullptr;
      }

#ifdef MFEM_USE_SINGLE
      id = new CMUMPS_STRUC_C();
#else
      id = new ZMUMPS_STRUC_C();
#endif

      id->sym = 0; // general complex
      id->par = 1;
      id->comm_fortran = (MUMPS_INT)MPI_Comm_c2f(comm);

      // Init
      id->job = -1;
      mumps_call();

      // Set parameters
      SetParameters();

      // Attach matrix
      id->n       = global_n;
      id->nnz_loc = nnz;
      id->irn_loc = I;
      id->jcn_loc = J;
      id->a_loc   = A;

      // Analysis (ordering + symbolic)
      id->job = 1;
      mumps_call();
   }
   else
   {
      // Reuse symbolic factorization / ordering
      MFEM_VERIFY(id->n == global_n,
                  "ReorderingReuse requires same global size (id->n mismatch)");

      // Update matrix pointers (pattern is assumed compatible)
      id->nnz_loc = nnz;
      id->irn_loc = I;
      id->jcn_loc = J;
      id->a_loc   = A;
   }

   // Factorization
   id->job = 2;
   {
      const int mem_relax_lim = 200;
      while (true)
      {
         mumps_call();
         if (id->MUMPS_INFOG(1) < 0)
         {
            if (id->MUMPS_INFOG(1) == -8 || id->MUMPS_INFOG(1) == -9)
            {
               id->MUMPS_ICNTL(14) += 20;
               MFEM_VERIFY(id->MUMPS_ICNTL(14) <= mem_relax_lim,
                           "Memory relaxation limit reached for MUMPS factorization");
               if (myid == 0 && print_level > 0)
               {
                  out << "Re-running MUMPS factorization with memory relaxation "
                      << id->MUMPS_ICNTL(14) << '\n';
               }
            }
            else
            {
               MFEM_ABORT("Error during MUMPS numerical factorization");
            }
         }
         else { break; }
      }
   }

   // Done with input storage
   if (csr_op_r) { hypre_CSRMatrixDestroy(csr_op_r);}
   if (csr_op_i) { hypre_CSRMatrixDestroy(csr_op_i);}
   delete [] I;
   delete [] J;
   delete [] A;

   // Post-factorization RHS/SOL setup
   id->nrhs = -1;

#if MFEM_MUMPS_VERSION >= 530
   // Distributed RHS/SOL sizes
   id->nloc_rhs = n_loc;
   id->lrhs_loc = n_loc;
   id->lsol_loc = id->MUMPS_INFO(23);

   delete [] irhs_loc;
   irhs_loc = new int[id->lrhs_loc];
   for (int i = 0; i < n_loc; i++)
   {
      irhs_loc[i] = row_start + i + 1;
   }
   id->irhs_loc = irhs_loc;

   delete [] isol_loc;
   isol_loc = new int[id->lsol_loc];
   id->isol_loc = isol_loc;

   row_starts.SetSize(numProcs);
   MPI_Allgather(&row_start, 1, MPI_INT, row_starts, 1, MPI_INT, comm);

   // Reset cached buffers
   delete [] rhs_loc; rhs_loc = nullptr;
   delete [] sol_loc; sol_loc = nullptr;
   rhs1_buf.clear();

#else
   // Centralized RHS/SOL on root
   id->lrhs = id->n;

   global_num_rows = id->n;

   if (myid == 0)
   {
      delete [] recv_counts;
      delete [] displs;
      recv_counts = new int[numProcs];
      displs = new int[numProcs];

      delete [] rhs_glob;   rhs_glob = nullptr;
      delete [] rhs_glob_r; rhs_glob_r = nullptr;
      delete [] rhs_glob_i; rhs_glob_i = nullptr;
   }

   MPI_Gather(&n_loc, 1, MPI_INT, recv_counts, 1, MPI_INT, 0, comm);

   if (myid == 0)
   {
      displs[0] = 0;
      int s = 0;
      for (int k = 0; k < numProcs - 1; k++)
      {
         s += recv_counts[k];
         displs[k+1] = s;
      }
   }
#endif
}

void ComplexMUMPSSolver::InitRhsSol(int nrhs) const
{
#if MFEM_MUMPS_VERSION >= 530

   MFEM_VERIFY(id, "InitRhsSol called before SetOperator");

   if (id->nrhs != nrhs)
   {
      delete [] rhs_loc;
      delete [] sol_loc;

      rhs_loc = new mumps_complex_t[(size_t)nrhs * (size_t)id->lrhs_loc];
      sol_loc = new mumps_complex_t[(size_t)nrhs * (size_t)id->lsol_loc];

      id->rhs_loc = rhs_loc;
      id->sol_loc = sol_loc;
   }
   id->nrhs = nrhs;

#else
   MFEM_VERIFY(id, "InitRhsSol called before SetOperator");

   id->nrhs = nrhs;
   id->lrhs = id->n;

   if (myid == 0)
   {
      const size_t N = (size_t)nrhs * (size_t)global_num_rows;

      delete [] rhs_glob;
      delete [] rhs_glob_r;
      delete [] rhs_glob_i;

      rhs_glob   = new mumps_complex_t[N];
      rhs_glob_r = new real_t[N];
      rhs_glob_i = new real_t[N];

      id->rhs = rhs_glob;
   }
#endif
}

void ComplexMUMPSSolver::Mult(const Vector &x, Vector &y) const
{
   Array<const Vector *> X(1);
   Array<Vector *> Y(1);
   X[0] = &x;
   Y[0] = &y;
   ArrayMult(X, Y);
}

void ComplexMUMPSSolver::ArrayMult(const Array<const Vector *> &X,
                                   Array<Vector *> &Y) const
{
   MFEM_ASSERT(X.Size() == Y.Size(),
               "Number of columns mismatch in ComplexMUMPSSolver::Mult!");
   MFEM_VERIFY(id, "ComplexMUMPSSolver::ArrayMult called before SetOperator");

   InitRhsSol(X.Size());

#if MFEM_MUMPS_VERSION >= 530
   MFEM_VERIFY(irhs_loc && isol_loc, "RHS/SOL maps not initialized");
   MFEM_VERIFY(rhs_loc && sol_loc, "RHS/SOL buffers not initialized");
   const int n_loc = id->lrhs_loc;
   const int nrhs  = id->nrhs;

   // Pack all RHS
   for (int i = 0; i < nrhs; i++)
   {
      MFEM_ASSERT(X[i], "Missing Vector in Mult!");
      X[i]->HostRead();
      MFEM_VERIFY(X[i]->Size() == 2*n_loc, "RHS size mismatch");

      const real_t *xdata = X[i]->GetData();
      const real_t *xr = xdata;
      const real_t *xi = xdata + n_loc;

      mumps_complex_t *dst = rhs_loc + i * n_loc;
      for (int j = 0; j < n_loc; j++)
      {
         dst[j].r = xr[j];
         dst[j].i = xi[j];
      }
   }

   id->rhs_loc  = rhs_loc;
   id->sol_loc  = sol_loc;
   id->irhs_loc = irhs_loc;
   id->isol_loc = isol_loc;

   // MUMPS solve
   id->job = 3;
   mumps_call();

   const int lsol = id->lsol_loc;

   // Redistribute each solution column into Y
   for (int i = 0; i < nrhs; i++)
   {
      MFEM_ASSERT(Y[i], "Missing output Vector in Mult!");
      Y[i]->HostWrite();
      MFEM_VERIFY(Y[i]->Size() == 2*n_loc, "Output size mismatch");

      const mumps_complex_t *xcol = sol_loc + i * lsol;
      RedistributeSol(isol_loc, xcol, Y[i]->GetData(), n_loc, lsol);
   }

#else // MFEM_MUMPS_VERSION < 530

   const int nrhs = id->nrhs;

   MFEM_VERIFY(X.Size() > 0 && X[0], "Missing RHS");
   const int n_loc = X[0]->Size()/2;

   for (int i = 0; i < nrhs; i++)
   {
      MFEM_ASSERT(X[i], "Missing Vector in Mult!");
      X[i]->HostRead();
      MFEM_VERIFY(X[i]->Size() == 2*n_loc, "RHS size mismatch");
   }

   // Gather each RHS column (real+imag separately) into root staging
   for (int i = 0; i < nrhs; i++)
   {
      const real_t *xdata = X[i]->GetData();

      MPI_Gatherv(xdata, n_loc, MPITypeMap<real_t>::mpi_type,
                  rhs_glob_r + i * global_num_rows,
                  recv_counts, displs, MPITypeMap<real_t>::mpi_type,
                  0, comm);

      MPI_Gatherv(xdata + n_loc, n_loc, MPITypeMap<real_t>::mpi_type,
                  rhs_glob_i + i * global_num_rows,
                  recv_counts, displs, MPITypeMap<real_t>::mpi_type,
                  0, comm);
   }

   // Pack into MUMPS complex RHS on root: id->rhs is in-place
   if (myid == 0)
   {
      for (int i = 0; i < nrhs; i++)
      {
         mumps_complex_t *dst = rhs_glob + i * global_num_rows;
         const real_t *rr = rhs_glob_r + i * global_num_rows;
         const real_t *ri = rhs_glob_i + i * global_num_rows;

         for (int j = 0; j < global_num_rows; j++)
         {
            dst[j].r = rr[j];
            dst[j].i = ri[j];
         }
      }
      id->rhs = rhs_glob;
   }

   // Solve
   id->job = 3;
   mumps_call();

   // Unpack to real/imag
   if (myid == 0)
   {
      for (int i = 0; i < nrhs; i++)
      {
         const mumps_complex_t *src = rhs_glob + i * global_num_rows;
         real_t *rr = rhs_glob_r + i * global_num_rows;
         real_t *ri = rhs_glob_i + i * global_num_rows;

         for (int j = 0; j < global_num_rows; j++)
         {
            rr[j] = src[j].r;
            ri[j] = src[j].i;
         }
      }
   }

   // Scatter each RHS solution
   for (int i = 0; i < nrhs; i++)
   {
      MFEM_ASSERT(Y[i], "Missing Vector in Mult!");
      Y[i]->HostWrite();
      MFEM_VERIFY(Y[i]->Size() == 2*n_loc, "Output size mismatch");

      real_t *ydata = Y[i]->GetData();

      MPI_Scatterv(rhs_glob_r + i * global_num_rows,
                   recv_counts, displs, MPITypeMap<real_t>::mpi_type,
                   ydata, n_loc, MPITypeMap<real_t>::mpi_type,
                   0, comm);

      MPI_Scatterv(rhs_glob_i + i * global_num_rows,
                   recv_counts, displs, MPITypeMap<real_t>::mpi_type,
                   ydata + n_loc, n_loc, MPITypeMap<real_t>::mpi_type,
                   0, comm);
   }

#endif
}

void ComplexMUMPSSolver::MultTranspose(const Vector &x, Vector &y) const
{
   MFEM_VERIFY(id, "MultTranspose called before SetOperator");

   // Transpose solve
   id->MUMPS_ICNTL(9) = 0;
   Mult(x, y);
   id->MUMPS_ICNTL(9) = 1;
}

void ComplexMUMPSSolver::ArrayMultTranspose(const Array<const Vector *> &X,
                                            Array<Vector *> &Y) const
{
   MFEM_VERIFY(id, "ArrayMultTranspose called before SetOperator");

   // Transpose solve
   id->MUMPS_ICNTL(9) = 0;
   ArrayMult(X, Y);
   id->MUMPS_ICNTL(9) = 1;
}

void ComplexMUMPSSolver::SetParameters()
{
   // Output stream for error messages
   id->MUMPS_ICNTL(1) = 6;
   // Output stream for diagnostic printing local to each proc
   id->MUMPS_ICNTL(2) = 0;
   // Output stream for global info
   id->MUMPS_ICNTL(3) = 6;
   // Level of error printing
   id->MUMPS_ICNTL(4) = print_level;

   // Input matrix format (assembled)
   id->MUMPS_ICNTL(5) = 0;
   // Use A or A^T
   id->MUMPS_ICNTL(9) = 1;
   // Iterative refinement (disabled)
   id->MUMPS_ICNTL(10) = 0;
   // Error analysis-statistics (disabled)
   id->MUMPS_ICNTL(11) = 0;
   // Use of ScaLAPACK (disabled)
   id->MUMPS_ICNTL(13) = 0;
   // Workspace relaxation (% increase)
   id->MUMPS_ICNTL(14) = 20;
   // OpenMP threads (default)
   id->MUMPS_ICNTL(16) = 0;
   // Matrix input format (distributed)
   id->MUMPS_ICNTL(18) = 3;
   // Schur complement (none)
   id->MUMPS_ICNTL(19) = 0;

#if MFEM_MUMPS_VERSION >= 530
   // Distributed RHS
   id->MUMPS_ICNTL(20) = 10;
   // Distributed Sol
   id->MUMPS_ICNTL(21) = 1;
#else
   // Centralized RHS
   id->MUMPS_ICNTL(20) = 0;
   // Centralized Sol
   id->MUMPS_ICNTL(21) = 0;
#endif

   // Out-of-core (disabled)
   id->MUMPS_ICNTL(22) = 0;
   // Max size of working memory (default)
   id->MUMPS_ICNTL(23) = 0;

   switch (reorder_method)
   {
      case ReorderingStrategy::AUTOMATIC:
         id->MUMPS_ICNTL(28) = 0;
         id->MUMPS_ICNTL(7) = 7;
         id->MUMPS_ICNTL(29) = 0;
         break;
      case ReorderingStrategy::AMD:
         id->MUMPS_ICNTL(28) = 1;
         id->MUMPS_ICNTL(7) = 0;
         break;
      case ReorderingStrategy::AMF:
         id->MUMPS_ICNTL(28) = 1;
         id->MUMPS_ICNTL(7) = 2;
         break;
      case ReorderingStrategy::PORD:
         id->MUMPS_ICNTL(28) = 1;
         id->MUMPS_ICNTL(7) = 4;
         break;
      case ReorderingStrategy::METIS:
         id->MUMPS_ICNTL(28) = 1;
         id->MUMPS_ICNTL(7) = 5;
         break;
      case ReorderingStrategy::PARMETIS:
         id->MUMPS_ICNTL(28) = 2;
         id->MUMPS_ICNTL(29) = 2;
         break;
      case ReorderingStrategy::SCOTCH:
         id->MUMPS_ICNTL(28) = 1;
         id->MUMPS_ICNTL(7) = 3;
         break;
      case ReorderingStrategy::PTSCOTCH:
         id->MUMPS_ICNTL(28) = 2;
         id->MUMPS_ICNTL(29) = 1;
         break;
      default:
         break; // This should be unreachable
   }

}

void ComplexMUMPSSolver::BuildUnionCOO(const int n_loc,
                                       const int row_start_,
                                       const int *Ir, const int *Jr, const real_t *Vr,
                                       const int *Ii, const int *Ji, const real_t *Vi,
                                       std::vector<int> &Icoo,
                                       std::vector<int> &Jcoo,
                                       std::vector<mumps_complex_t> &Zcoo) const
{
   for (int r = 0; r < n_loc; ++r)
   {
      std::unordered_map<int, std::pair<real_t, real_t>> row;

      const int rr0 = Ir ? Ir[r]   : 0;
      const int rr1 = Ir ? Ir[r+1] : 0;
      const int ii0 = Ii ? Ii[r]   : 0;
      const int ii1 = Ii ? Ii[r+1] : 0;

      row.reserve((rr1 - rr0) + (ii1 - ii0));

      if (Ir)
      {
         for (int p = rr0; p < rr1; ++p) { row[Jr[p]].first += Vr[p]; }
      }
      if (Ii)
      {
         for (int p = ii0; p < ii1; ++p) { row[Ji[p]].second += Vi[p]; }
      }

      for (const auto &kv : row)
      {
         Icoo.push_back(row_start_ + r + 1);
         Jcoo.push_back(kv.first + 1);
         Zcoo.push_back(mumps_complex_t{kv.second.first, kv.second.second});
      }
   }
}

#if MFEM_MUMPS_VERSION >= 530
int ComplexMUMPSSolver::GetRowRank(int i, const Array<int> &row_starts_) const
{
   if (row_starts_.Size() == 1) { return 0; }
   auto up = std::upper_bound(row_starts_.begin(), row_starts_.end(), i);
   return (int)std::distance(row_starts_.begin(), up) - 1;
}

void ComplexMUMPSSolver::RedistributeSol(const int *row_map,
                                         const mumps_complex_t *x,
                                         real_t *y_ri,
                                         int n_loc,
                                         int lsol_loc) const
{
   int *send_count = new int[numProcs]();
   for (int i = 0; i < lsol_loc; i++)
   {
      const int j = row_map[i] - 1;
      const int row_rank = GetRowRank(j, row_starts);
      if (myid == row_rank) { continue; }
      send_count[row_rank]++;
   }

   int *recv_count = new int[numProcs];
   MPI_Alltoall(send_count, 1, MPI_INT, recv_count, 1, MPI_INT, comm);

   int *send_displ = new int[numProcs]; send_displ[0] = 0;
   int *recv_displ = new int[numProcs]; recv_displ[0] = 0;

   int sbuff_size = send_count[numProcs-1];
   int rbuff_size = recv_count[numProcs-1];
   for (int k = 0; k < numProcs - 1; k++)
   {
      send_displ[k+1] = send_displ[k] + send_count[k];
      recv_displ[k+1] = recv_displ[k] + recv_count[k];
      sbuff_size += send_count[k];
      rbuff_size += recv_count[k];
   }

   int   *sendbuf_index = new int[sbuff_size];
   real_t *sendbuf_r    = new real_t[sbuff_size];
   real_t *sendbuf_i    = new real_t[sbuff_size];
   int   *soffs         = new int[numProcs]();

   for (int i = 0; i < lsol_loc; i++)
   {
      const int j = row_map[i] - 1;
      const int row_rank = GetRowRank(j, row_starts);

      const real_t xr = (real_t)x[i].r;
      const real_t xi = (real_t)x[i].i;

      if (myid == row_rank)
      {
         const int local_index = j - row_start;
         y_ri[local_index]        = xr;
         y_ri[local_index+n_loc]  = xi;
      }
      else
      {
         const int k = send_displ[row_rank] + soffs[row_rank];
         sendbuf_index[k] = j;
         sendbuf_r[k] = xr;
         sendbuf_i[k] = xi;
         soffs[row_rank]++;
      }
   }

   int    *recvbuf_index = new int[rbuff_size];
   real_t *recvbuf_r     = new real_t[rbuff_size];
   real_t *recvbuf_i     = new real_t[rbuff_size];

   MPI_Alltoallv(sendbuf_index, send_count, send_displ, MPI_INT,
                 recvbuf_index, recv_count, recv_displ, MPI_INT, comm);

   MPI_Alltoallv(sendbuf_r, send_count, send_displ, MPITypeMap<real_t>::mpi_type,
                 recvbuf_r, recv_count, recv_displ, MPITypeMap<real_t>::mpi_type, comm);

   MPI_Alltoallv(sendbuf_i, send_count, send_displ, MPITypeMap<real_t>::mpi_type,
                 recvbuf_i, recv_count, recv_displ, MPITypeMap<real_t>::mpi_type, comm);

   for (int i = 0; i < rbuff_size; i++)
   {
      const int local_index = recvbuf_index[i] - row_start;
      y_ri[local_index]       = recvbuf_r[i];
      y_ri[local_index+n_loc] = recvbuf_i[i];
   }

   delete [] recvbuf_i;
   delete [] recvbuf_r;
   delete [] recvbuf_index;
   delete [] soffs;
   delete [] sendbuf_i;
   delete [] sendbuf_r;
   delete [] sendbuf_index;
   delete [] recv_displ;
   delete [] send_displ;
   delete [] recv_count;
   delete [] send_count;
}
#endif // MFEM_MUMPS_VERSION >= 530
#endif // MFEM_USE_COMPLEX_MUMPS
#endif // MFEM_USE_MPI

} // namespace mfem
