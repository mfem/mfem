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

#include "complex_operator.hpp"

namespace mfem
{

ComplexOperator::ComplexOperator(Operator * Op_Real, Operator * Op_Imag,
                                 bool ownReal, bool ownImag,
                                 Convention convention)
   : Operator(2*Op_Real->Height(), 2*Op_Real->Width())
   , Op_Real_(Op_Real)
   , Op_Imag_(Op_Imag)
   , ownReal_(ownReal)
   , ownImag_(ownImag)
   , convention_(convention)
   , x_r_(NULL, Op_Real->Width())
   , x_i_(NULL, Op_Real->Width())
   , y_r_(NULL, Op_Real->Height())
   , y_i_(NULL, Op_Real->Height())
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

void ComplexOperator::Mult(const Vector &x, Vector &y) const
{
   double * x_data = x.GetData();
   x_r_.SetData(x_data);
   x_i_.SetData(&x_data[Op_Real_->Width()]);

   y_r_.SetData(&y[0]);
   y_i_.SetData(&y[Op_Real_->Height()]);

   this->Mult(x_r_, x_i_, y_r_, y_i_);
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
      if (!v_) { v_ = new Vector(Op_Imag_->Height()); }
      Op_Imag_->Mult(x_i, *v_);
      y_r_ -= *v_;
      Op_Imag_->Mult(x_r, *v_);
      y_i_ += *v_;
   }

   if (convention_ == BLOCK_SYMMETRIC)
   {
      y_i_ *= -1.0;
   }
}

void ComplexOperator::MultTranspose(const Vector &x, Vector &y) const
{
   double * x_data = x.GetData();
   y_r_.SetData(x_data);
   y_i_.SetData(&x_data[Op_Real_->Height()]);

   x_r_.SetData(&y[0]);
   x_i_.SetData(&y[Op_Real_->Width()]);

   this->MultTranspose(y_r_, y_i_, x_r_, x_i_);
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
      if (!u_) { u_ = new Vector(Op_Imag_->Width()); }
      Op_Imag_->MultTranspose(x_i, *u_);
      y_r_.Add(convention_ == BLOCK_SYMMETRIC ? -1.0 : 1.0, *u_);
      Op_Imag_->MultTranspose(x_r, *u_);
      y_i_ -= *u_;
   }
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

   int    *I = new int[this->Height()+1];
   int    *J = new int[nnz];
   double *D = new double[nnz];

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

}
