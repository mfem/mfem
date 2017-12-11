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

#include "vector.hpp"
#include "operator.hpp"

#include <iostream>
#include <iomanip>

namespace mfem
{

template <class TVector>
void TOperator<TVector>::FormLinearSystem(const Array<int> &ess_tdof_list,
                                TVector &x, TVector &b,
                                TOperator<TVector>* &Aout, TVector &X, TVector &B,
                                int copy_interior)
{
   const TOperator<TVector> *P = this->GetProlongation();
   const TOperator<TVector> *R = this->GetRestriction();
   TOperator<TVector> *rap;

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

template <class TVector>
void TOperator<TVector>::RecoverFEMSolution(const TVector &X,
                                            const TVector &b,
                                            TVector &x)
{
   const TOperator<TVector> *P = this->GetProlongation();
   if (P)
   {
      // Apply conforming prolongation
      x.SetSize(P->Height());
      P->Mult(X, x);
   }
   // Otherwise X and x point to the same data
}

template <class TVector>
void TOperator<TVector>::PrintMatlab(std::ostream & out, int n, int m) const
{
   using namespace std;
   if (n == 0) { n = width; }
   if (m == 0) { m = height; }

   TVector x(n), y(m);
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


template <class TVector>
TConstrainedOperator<TVector>::TConstrainedOperator(TOperator<TVector> *A,
                                                    const Array<int> &list,
                                                    bool _own_A)
   : TOperator<TVector>(A->Height(), A->Width()), A(A), own_A(_own_A)
{
   constraint_list.MakeRef(list);
   z.SetSize(TOperator<TVector>::height);
   w.SetSize(TOperator<TVector>::height);
}

template <class TVector>
void TConstrainedOperator<TVector>::EliminateRHS(const TVector &x,
                                                 TVector &b) const
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
  
template <class TVector>
void TConstrainedOperator<TVector>::Mult(const TVector &x, TVector &y) const
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

}
