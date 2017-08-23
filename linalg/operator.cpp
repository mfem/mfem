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

#include <iostream>
#include <iomanip>

#include "vector.hpp"
#include "operator.hpp"

namespace mfem
{

void Operator::FormLinearSystem(const Array<int> &ess_tdof_list,
                                Vector &x, Vector &b,
                                Operator* &Aout, Vector &X, Vector &B,
                                int copy_interior)
{
   const Operator *P = this->GetProlongation();
   const Operator *R = this->GetRestriction();
   Operator *rap;

   if (P)
   {
      // Variational restriction with P
      B.SetSize(P->Width());
      P->MultTranspose(b, B);
      X.SetSize(R->Height());
      R->Mult(x, X);
      rap = new RAPOperator(*P, *this, *P);
   }
   else
   {
      // rap, X and B point to the same data as this, x and b
      X.NewDataAndSize(x.GetData(), x.Size());
      B.NewDataAndSize(b.GetData(), b.Size());
      rap = this;
   }

   if (!copy_interior) { X.SetSubVectorComplement(ess_tdof_list, 0.0); }

   // Impose the boundary conditions through a ConstrainedOperator, which owns
   // the rap operator when P and R are non-trivial
   ConstrainedOperator *A = new ConstrainedOperator(rap, ess_tdof_list,
                                                    rap != this);
   A->EliminateRHS(X, B);
   Aout = A;
}

void Operator::RecoverFEMSolution(const Vector &X, const Vector &b, Vector &x)
{
   const Operator *P = this->GetProlongation();
   if (P)
   {
      // Apply conforming prolongation
      x.SetSize(P->Height());
      P->Mult(X, x);
   }
   else
   {
      // X and x point to the same data
   }
}

void Operator::PrintMatlab(std::ostream & out, int n, int m) const
{
   using namespace std;
   if (n == 0) { n = width; }
   if (m == 0) { m = height; }

   Vector x(n), y(m);
   x = 0.0;

   out << setiosflags(ios::scientific | ios::showpos);
   for (int i = 0; i < n; i++)
   {
      x(i) = 1.0;
      Mult(x, y);
      for (int j = 0; j < m; j++)
      {
         if (y(j))
         {
            out << j+1 << " " << i+1 << " " << y(j) << '\n';
         }
      }
      x(i) = 0.0;
   }
}


ConstrainedOperator::ConstrainedOperator(Operator *A, const Array<int> &list,
                                         bool _own_A)
   : Operator(A->Height(), A->Width()), A(A), own_A(_own_A)
{
   constraint_list.MakeRef(list);
   z.SetSize(height);
   w.SetSize(height);
}

void ConstrainedOperator::EliminateRHS(const Vector &x, Vector &b) const
{
   w = 0.0;

   for (int i = 0; i < constraint_list.Size(); i++)
   {
      w(constraint_list[i]) = x(constraint_list[i]);
   }

   A->Mult(w, z);

   b -= z;

   for (int i = 0; i < constraint_list.Size(); i++)
   {
      b(constraint_list[i]) = x(constraint_list[i]);
   }
}

void ConstrainedOperator::Mult(const Vector &x, Vector &y) const
{
   if (constraint_list.Size() == 0)
   {
      A->Mult(x, y);
      return;
   }

   z = x;

   for (int i = 0; i < constraint_list.Size(); i++)
   {
      z(constraint_list[i]) = 0.0;
   }

   A->Mult(z, y);

   for (int i = 0; i < constraint_list.Size(); i++)
   {
      y(constraint_list[i]) = x(constraint_list[i]);
   }
}

ComplexOperator::ComplexOperator(Operator * Op_Real, Operator * Op_Imag,
                                 bool ownReal, bool ownImag)
   : Operator(2*Op_Real->Height(), 2*Op_Real->Width())
   , Op_Real_(Op_Real)
   , Op_Imag_(Op_Imag)
   , ownReal_(ownReal)
   , ownImag_(ownImag)
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

void
ComplexOperator::Mult(const Vector &x, Vector &y) const
{
   double * x_data = x.GetData();
   x_r_.SetData(x_data);
   x_i_.SetData(&x_data[Op_Real_->Width()]);

   y_r_.SetData(&y[0]);
   y_i_.SetData(&y[Op_Real_->Height()]);

   this->Mult(x_r_, x_i_, y_r_, y_i_);
}

void
ComplexOperator::Mult(const Vector &x_r, const Vector &x_i,
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
      Op_Imag_->Mult(x_i, *v_); y_r_ -= *v_;
      Op_Imag_->Mult(x_r, *v_); y_i_ += *v_;
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
   }
   else
   {
      y_r = 0.0;
      y_i = 0.0;
   }
   if (Op_Imag_)
   {
      if (!u_) { u_ = new Vector(Op_Imag_->Width()); }
      Op_Imag_->MultTranspose(x_i, *u_); y_r_ += *u_;
      Op_Imag_->MultTranspose(x_r, *u_); y_i_ -= *u_;
   }
}

}
