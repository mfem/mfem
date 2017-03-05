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
  TFormLinearSystem<Vector>(ess_tdof_list,
                            x, b, Aout, X, B,
                            copy_interior);
}

void Operator::ImposeBoundaryConditions(const Array<int> &ess_tdof_list,
                                        Operator *rap,
                                        Operator* &Aout, Vector &X, Vector &B)
{
  ConstrainedOperator *A = new ConstrainedOperator(rap, ess_tdof_list,
                                                   rap != this);
  A->EliminateRHS(X, B);
  Aout = A;
}

void Operator::RecoverFEMSolution(const Vector &X, const Vector &b, Vector &x)
{
  TRecoverFEMSolution(X, b, x);
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

}
