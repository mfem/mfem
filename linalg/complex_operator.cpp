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

}
