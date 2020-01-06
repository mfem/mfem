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
   ConstrainedOperator *constrainedA;
   FormConstrainedSystemOperator(ess_tdof_list, constrainedA);

   const Operator *P = this->GetProlongation();
   const Operator *R = this->GetRestriction();

   if (P)
   {
      // Variational restriction with P
      B.SetSize(P->Width(), b);
      P->MultTranspose(b, B);
      X.SetSize(R->Height(), x);
      R->Mult(x, X);
   }
   else
   {
      // rap, X and B point to the same data as this, x and b, respectively
      X.NewMemoryAndSize(x.GetMemory(), x.Size(), false);
      B.NewMemoryAndSize(b.GetMemory(), b.Size(), false);
   }

   if (!copy_interior) { X.SetSubVectorComplement(ess_tdof_list, 0.0); }

   constrainedA->EliminateRHS(X, B);
   Aout = constrainedA;
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

      // If the validity flags of X's Memory were changed (e.g. if it was moved
      // to device memory) then we need to tell x about that.
      x.SyncMemory(X);
   }
}

void Operator::FormConstrainedSystemOperator(
   const Array<int> &ess_tdof_list, ConstrainedOperator* &Aout)
{
   const Operator *P = this->GetProlongation();
   Operator *rap;

   if (P)
   {
      // Variational restriction with P
      rap = new RAPOperator(*P, *this, *P);
   }
   else
   {
      rap = this;
   }

   // Impose the boundary conditions through a ConstrainedOperator, which owns
   // the rap operator when P and R are non-trivial
   ConstrainedOperator *A = new ConstrainedOperator(rap, ess_tdof_list,
                                                    rap != this);
   Aout = A;
}

void Operator::FormSystemOperator(const Array<int> &ess_tdof_list,
                                  Operator* &Aout)
{
   ConstrainedOperator *A;
   FormConstrainedSystemOperator(ess_tdof_list, A);
   Aout = A;
}

void Operator::FormDiscreteOperator(Operator* &Aout)
{
   const Operator *Pin  = this->GetProlongation();
   const Operator *Rout = this->GetOutputRestriction();
   Aout = new TripleProductOperator(Rout, this, Pin,false, false, false);
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


void TimeDependentOperator::ExplicitMult(const Vector &, Vector &) const
{
   mfem_error("TimeDependentOperator::ExplicitMult() is not overridden!");
}

void TimeDependentOperator::ImplicitMult(const Vector &, const Vector &,
                                         Vector &) const
{
   mfem_error("TimeDependentOperator::ImplicitMult() is not overridden!");
}

void TimeDependentOperator::Mult(const Vector &, Vector &) const
{
   mfem_error("TimeDependentOperator::Mult() is not overridden!");
}

void TimeDependentOperator::ImplicitSolve(const double, const Vector &,
                                          Vector &)
{
   mfem_error("TimeDependentOperator::ImplicitSolve() is not overridden!");
}

Operator &TimeDependentOperator::GetImplicitGradient(
   const Vector &, const Vector &, double) const
{
   mfem_error("TimeDependentOperator::GetImplicitGradient() is "
              "not overridden!");
   return const_cast<Operator &>(dynamic_cast<const Operator &>(*this));
}

Operator &TimeDependentOperator::GetExplicitGradient(const Vector &) const
{
   mfem_error("TimeDependentOperator::GetExplicitGradient() is "
              "not overridden!");
   return const_cast<Operator &>(dynamic_cast<const Operator &>(*this));
}

int TimeDependentOperator::SUNImplicitSetup(const Vector &,
                                            const Vector &,
                                            int, int *, double)
{
   mfem_error("TimeDependentOperator::SUNImplicitSetup() is not overridden!");
   return (-1);
}

int TimeDependentOperator::SUNImplicitSolve(const Vector &, Vector &, double)
{
   mfem_error("TimeDependentOperator::SUNImplicitSolve() is not overridden!");
   return (-1);
}

int TimeDependentOperator::SUNMassSetup()
{
   mfem_error("TimeDependentOperator::SUNMassSetup() is not overridden!");
   return (-1);
}

int TimeDependentOperator::SUNMassSolve(const Vector &, Vector &, double)
{
   mfem_error("TimeDependentOperator::SUNMassSolve() is not overridden!");
   return (-1);
}

int TimeDependentOperator::SUNMassMult(const Vector &, Vector &)
{
   mfem_error("TimeDependentOperator::SUNMassMult() is not overridden!");
   return (-1);
}


ProductOperator::ProductOperator(const Operator *A, const Operator *B,
                                 bool ownA, bool ownB)
   : Operator(A->Height(), B->Width()),
     A(A), B(B), ownA(ownA), ownB(ownB), z(A->Width())
{
   MFEM_VERIFY(A->Width() == B->Height(),
               "incompatible Operators: A->Width() = " << A->Width()
               << ", B->Height() = " << B->Height());

   {
      const Solver* SolverB = dynamic_cast<const Solver*>(B);
      if (SolverB)
      {
         MFEM_VERIFY(!(SolverB->iterative_mode),
                     "Operator B of a ProductOperator should not be in iterative mode");
      }
   }
}

ProductOperator::~ProductOperator()
{
   if (ownA) { delete A; }
   if (ownB) { delete B; }
}


RAPOperator::RAPOperator(const Operator &Rt_, const Operator &A_,
                         const Operator &P_)
   : Operator(Rt_.Width(), P_.Width()), Rt(Rt_), A(A_), P(P_)
{
   MFEM_VERIFY(Rt.Height() == A.Height(),
               "incompatible Operators: Rt.Height() = " << Rt.Height()
               << ", A.Height() = " << A.Height());
   MFEM_VERIFY(A.Width() == P.Height(),
               "incompatible Operators: A.Width() = " << A.Width()
               << ", P.Height() = " << P.Height());

   {
      const Solver* SolverA = dynamic_cast<const Solver*>(&A);
      if (SolverA)
      {
         MFEM_VERIFY(!(SolverA->iterative_mode),
                     "Operator A of an RAPOperator should not be in iterative mode");
      }

      const Solver* SolverP = dynamic_cast<const Solver*>(&P);
      if (SolverP)
      {
         MFEM_VERIFY(!(SolverP->iterative_mode),
                     "Operator P of an RAPOperator should not be in iterative mode");
      }
   }

   mem_class = Rt.GetMemoryClass()*P.GetMemoryClass();
   MemoryType mem_type = GetMemoryType(A.GetMemoryClass()*mem_class);
   Px.SetSize(P.Height(), mem_type);
   APx.SetSize(A.Height(), mem_type);
}


TripleProductOperator::TripleProductOperator(
   const Operator *A, const Operator *B, const Operator *C,
   bool ownA, bool ownB, bool ownC)
   : Operator(A->Height(), C->Width())
   , A(A), B(B), C(C)
   , ownA(ownA), ownB(ownB), ownC(ownC)
{
   MFEM_VERIFY(A->Width() == B->Height(),
               "incompatible Operators: A->Width() = " << A->Width()
               << ", B->Height() = " << B->Height());
   MFEM_VERIFY(B->Width() == C->Height(),
               "incompatible Operators: B->Width() = " << B->Width()
               << ", C->Height() = " << C->Height());

   {
      const Solver* SolverB = dynamic_cast<const Solver*>(B);
      if (SolverB)
      {
         MFEM_VERIFY(!(SolverB->iterative_mode),
                     "Operator B of a TripleProductOperator should not be in iterative mode");
      }

      const Solver* SolverC = dynamic_cast<const Solver*>(C);
      if (SolverC)
      {
         MFEM_VERIFY(!(SolverC->iterative_mode),
                     "Operator C of a TripleProductOperator should not be in iterative mode");
      }
   }

   mem_class = A->GetMemoryClass()*C->GetMemoryClass();
   MemoryType mem_type = GetMemoryType(mem_class*B->GetMemoryClass());
   t1.SetSize(C->Height(), mem_type);
   t2.SetSize(B->Height(), mem_type);
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
   // 'mem_class' should work with A->Mult() and MFEM_FORALL():
   mem_class = A->GetMemoryClass()*Device::GetMemoryClass();
   MemoryType mem_type = GetMemoryType(mem_class);
   list.Read(); // TODO: just ensure 'list' is registered, no need to copy it
   constraint_list.MakeRef(list);
   // typically z and w are large vectors, so store them on the device
   z.SetSize(height, mem_type); z.UseDevice(true);
   w.SetSize(height, mem_type); w.UseDevice(true);
}

void ConstrainedOperator::EliminateRHS(const Vector &x, Vector &b) const
{
   w = 0.0;
   const int csz = constraint_list.Size();
   auto idx = constraint_list.Read();
   auto d_x = x.Read();
   // Use read+write access - we are modifying sub-vector of w
   auto d_w = w.ReadWrite();
   MFEM_FORALL(i, csz,
   {
      const int id = idx[i];
      d_w[id] = d_x[id];
   });

   A->Mult(w, z);

   b -= z;
   // Use read+write access - we are modifying sub-vector of b
   auto d_b = b.ReadWrite();
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

   auto idx = constraint_list.Read();
   // Use read+write access - we are modifying sub-vector of z
   auto d_z = z.ReadWrite();
   MFEM_FORALL(i, csz, d_z[idx[i]] = 0.0;);

   A->Mult(z, y);

   auto d_x = x.Read();
   // Use read+write access - we are modifying sub-vector of y
   auto d_y = y.ReadWrite();
   MFEM_FORALL(i, csz,
   {
      const int id = idx[i];
      d_y[id] = d_x[id];
   });
}

}
