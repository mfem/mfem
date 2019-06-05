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
#include "dtensor.hpp"
#include "operator.hpp"
#include "../general/forall.hpp"

#include <iostream>
#include <iomanip>

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

void Operator::CheckJacobian(Vector &x, Array<int> &ess_tdof, int print_level)
{
   double eps = 1e-8;
   Vector xpert(height), one(height), fx(height),
          fxp(height), basis(height), jac_col(height);

   basis = 0.0;
   xpert = x;

   Operator &jac = this->GetGradient(x);

   this->Mult(x, fx);

   for (int j = 0; j < height; j++)
   {
      // Check if column is associated with an essential boundary
      if (ess_tdof.Find(j) >= 0)
      {
         jac_col = 0.0;
      }
      else
      {
         basis(j) = 1.0;

         // Forward finite differences
         xpert(j) += eps;
         this->Mult(xpert, fxp);
         fxp -= fx;
         fxp /= eps;

         // Extract column from the operator using the j'th basis
         jac.Mult(basis, jac_col);
         jac_col -= fxp;

         basis(j) = 0.0;
         xpert(j) = x(j);
      }

      double norm = jac_col.Norml2();
      // Check if there is an error in the column
      if (norm >= sqrt(eps))
      {
         if (print_level == 1)
         {
            std::cout << "Possible error in jacobian column " << j << std::endl;
            std::cout << "||J(:," << j << ")||_l2 = " << jac_col.Norml2() << std::endl;
         }
         else if (print_level == 2)
         {
            for (int i = 0; i < height; i++)
            {
               if (jac_col(i) >= sqrt(eps))
               {
                  std::cout << "Possible error in jacobian column " << j << std::endl;
                  std::cout << "dJ/du(" << i << "," << j << ") = " << jac_col(i) << std::endl;
                  std::cout << "FD dJ/du(" << i << "," << j << ") = " << fxp(i) << std::endl;
               }
            }
         }
      }
   }
}


ProductOperator::ProductOperator(const Operator *A, const Operator *B,
                                 bool ownA, bool ownB)
   : Operator(A->Height(), B->Width()),
     A(A), B(B), ownA(ownA), ownB(ownB), z(A->Width())
{
   MFEM_VERIFY(A->Width() == B->Height(),
               "incompatible Operators: A->Width() = " << A->Width()
               << ", B->Height() = " << B->Height());
}

ProductOperator::~ProductOperator()
{
   if (ownA) { delete A; }
   if (ownB) { delete B; }
}


RAPOperator::RAPOperator(const Operator &Rt_, const Operator &A_,
                         const Operator &P_)
   : Operator(Rt_.Width(), P_.Width()), Rt(Rt_), A(A_), P(P_),
     Px(P.Height()), APx(A.Height())
{
   MFEM_VERIFY(Rt.Height() == A.Height(),
               "incompatible Operators: Rt.Height() = " << Rt.Height()
               << ", A.Height() = " << A.Height());
   MFEM_VERIFY(A.Width() == P.Height(),
               "incompatible Operators: A.Width() = " << A.Width()
               << ", P.Height() = " << P.Height());
}


TripleProductOperator::TripleProductOperator(
   const Operator *A, const Operator *B, const Operator *C,
   bool ownA, bool ownB, bool ownC)
   : Operator(A->Height(), C->Width())
   , A(A), B(B), C(C)
   , ownA(ownA), ownB(ownB), ownC(ownC)
   , t1(C->Height()), t2(B->Height())
{
   MFEM_VERIFY(A->Width() == B->Height(),
               "incompatible Operators: A->Width() = " << A->Width()
               << ", B->Height() = " << B->Height());
   MFEM_VERIFY(B->Width() == C->Height(),
               "incompatible Operators: B->Width() = " << B->Width()
               << ", C->Height() = " << C->Height());
}

TripleProductOperator::~TripleProductOperator()
{
   if (ownA) { delete A; }
   if (ownB) { delete B; }
   if (ownC) { delete C; }
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
   const int csz = constraint_list.Size();
   const DeviceArray idx(constraint_list, csz);
   const DeviceVector d_x(x, x.Size());
   DeviceVector d_w(w, w.Size());
   MFEM_FORALL(i, csz,
   {
      const int id = idx[i];
      d_w[id] = d_x[id];
   });

   A->Mult(w, z);

   b -= z;
   DeviceVector d_b(b, b.Size());
   MFEM_FORALL(i, csz,
   {
      const int id = idx[i];
      d_b[id] = d_x[id];
   });
}

void ConstrainedOperator::Mult(const Vector &x, Vector &y) const
{
   const int csz = constraint_list.Size();
   if (csz == 0)
   {
      A->Mult(x, y);
      return;
   }

   z = x;

   const DeviceArray idx(constraint_list, csz);
   DeviceVector d_z(z, z.Size());
   MFEM_FORALL(i, csz, d_z[idx[i]] = 0.0;);

   A->Mult(z, y);

   const DeviceVector d_x(x, x.Size());
   DeviceVector d_y(y, y.Size());
   MFEM_FORALL(i, csz,
   {
      const int id = idx[i];
      d_y[id] = d_x[id];
   });
}

}
