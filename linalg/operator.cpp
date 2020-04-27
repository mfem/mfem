// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "vector.hpp"
#include "operator.hpp"
#include "../general/forall.hpp"

#include <iostream>
#include <iomanip>

namespace mfem
{

void Operator::InitTVectors(const Operator *Po, const Operator *Ri,
                            const Operator *Pi,
                            Vector &x, Vector &b,
                            Vector &X, Vector &B) const
{
   if (!IsIdentityProlongation(Po))
   {
      // Variational restriction with Po
      B.SetSize(Po->Width(), b);
      Po->MultTranspose(b, B);
   }
   else
   {
      // B points to same data as b
      B.NewMemoryAndSize(b.GetMemory(), b.Size(), false);
   }
   if (!IsIdentityProlongation(Pi))
   {
      // Variational restriction with Ri
      X.SetSize(Ri->Height(), x);
      Ri->Mult(x, X);
   }
   else
   {
      // X points to same data as x
      X.NewMemoryAndSize(x.GetMemory(), x.Size(), false);
   }
}

void Operator::FormLinearSystem(const Array<int> &ess_tdof_list,
                                Vector &x, Vector &b,
                                Operator* &Aout, Vector &X, Vector &B,
                                int copy_interior)
{
   const Operator *P = this->GetProlongation();
   const Operator *R = this->GetRestriction();
   InitTVectors(P, R, P, x, b, X, B);

   if (!copy_interior) { X.SetSubVectorComplement(ess_tdof_list, 0.0); }

   ConstrainedOperator *constrainedA;
   FormConstrainedSystemOperator(ess_tdof_list, constrainedA);
   constrainedA->EliminateRHS(X, B);
   Aout = constrainedA;
}

void Operator::FormRectangularLinearSystem(
   const Array<int> &trial_tdof_list,
   const Array<int> &test_tdof_list, Vector &x, Vector &b,
   Operator* &Aout, Vector &X, Vector &B)
{
   const Operator *Pi = this->GetProlongation();
   const Operator *Po = this->GetOutputProlongation();
   const Operator *Ri = this->GetRestriction();
   InitTVectors(Po, Ri, Pi, x, b, X, B);

   RectangularConstrainedOperator *constrainedA;
   FormRectangularConstrainedSystemOperator(trial_tdof_list, test_tdof_list,
                                            constrainedA);
   constrainedA->EliminateRHS(X, B);
   Aout = constrainedA;
}

void Operator::RecoverFEMSolution(const Vector &X, const Vector &b, Vector &x)
{
   // Same for Rectangular and Square operators
   const Operator *P = this->GetProlongation();
   if (!IsIdentityProlongation(P))
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

Operator * Operator::SetupRAP(const Operator *Pi, const Operator *Po)
{
   Operator *rap;
   if (!IsIdentityProlongation(Pi))
   {
      if (!IsIdentityProlongation(Po))
      {
         rap = new RAPOperator(*Po, *this, *Pi);
      }
      else
      {
         rap = new ProductOperator(this, Pi, false,false);
      }
   }
   else
   {
      if (!IsIdentityProlongation(Po))
      {
         TransposeOperator * PoT = new TransposeOperator(Po);
         rap = new ProductOperator(PoT, this, true,false);
      }
      else
      {
         rap = this;
      }
   }
   return rap;
}

void Operator::FormConstrainedSystemOperator(
   const Array<int> &ess_tdof_list, ConstrainedOperator* &Aout)
{
   const Operator *P = this->GetProlongation();
   Operator *rap = SetupRAP(P, P);

   // Impose the boundary conditions through a ConstrainedOperator, which owns
   // the rap operator when P and R are non-trivial
   ConstrainedOperator *A = new ConstrainedOperator(rap, ess_tdof_list,
                                                    rap != this);
   Aout = A;
}

void Operator::FormRectangularConstrainedSystemOperator(
   const Array<int> &trial_tdof_list, const Array<int> &test_tdof_list,
   RectangularConstrainedOperator* &Aout)
{
   const Operator *Pi = this->GetProlongation();
   const Operator *Po = this->GetOutputProlongation();
   Operator *rap = SetupRAP(Pi, Po);

   // Impose the boundary conditions through a RectangularConstrainedOperator,
   // which owns the rap operator when P and R are non-trivial
   RectangularConstrainedOperator *A
      = new RectangularConstrainedOperator(rap,
                                           trial_tdof_list, test_tdof_list,
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

void Operator::FormRectangularSystemOperator(const Array<int> &trial_tdof_list,
                                             const Array<int> &test_tdof_list,
                                             Operator* &Aout)
{
   RectangularConstrainedOperator *A;
   FormRectangularConstrainedSystemOperator(trial_tdof_list, test_tdof_list, A);
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


void SecondOrderTimeDependentOperator::Mult(const Vector &x,
                                            const Vector &dxdt,
                                            Vector &y) const
{
   mfem_error("SecondOrderTimeDependentOperator::Mult() is not overridden!");
}

void SecondOrderTimeDependentOperator::ImplicitSolve(const double dt0,
                                                     const double dt1,
                                                     const Vector &x,
                                                     const Vector &dxdt,
                                                     Vector &k)
{
   mfem_error("SecondOrderTimeDependentOperator::ImplicitSolve() is not overridden!");
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
   mem_class = A->GetMemoryClass()*Device::GetDeviceMemoryClass();
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

   // A.AddMult(w, b, -1.0); // if available to all Operators
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

RectangularConstrainedOperator::RectangularConstrainedOperator(
   Operator *A,
   const Array<int> &trial_list,
   const Array<int> &test_list,
   bool _own_A)
   : Operator(A->Height(), A->Width()), A(A), own_A(_own_A)
{
   // 'mem_class' should work with A->Mult() and MFEM_FORALL():
   mem_class = A->GetMemoryClass()*Device::GetMemoryClass();
   MemoryType mem_type = GetMemoryType(mem_class);
   trial_list.Read(); // TODO: just ensure 'list' is registered, no need to copy it
   test_list.Read(); // TODO: just ensure 'list' is registered, no need to copy it
   trial_constraints.MakeRef(trial_list);
   test_constraints.MakeRef(test_list);
   // typically z and w are large vectors, so store them on the device
   z.SetSize(height, mem_type); z.UseDevice(true);
   w.SetSize(width, mem_type); w.UseDevice(true);
}

void RectangularConstrainedOperator::EliminateRHS(const Vector &x,
                                                  Vector &b) const
{
   w = 0.0;
   const int trial_csz = trial_constraints.Size();
   auto trial_idx = trial_constraints.Read();
   auto d_x = x.Read();
   // Use read+write access - we are modifying sub-vector of w
   auto d_w = w.ReadWrite();
   MFEM_FORALL(i, trial_csz,
   {
      const int id = trial_idx[i];
      d_w[id] = d_x[id];
   });

   // A.AddMult(w, b, -1.0); // if available to all Operators
   A->Mult(w, z);
   b -= z;

   const int test_csz = test_constraints.Size();
   auto test_idx = test_constraints.Read();
   auto d_b = b.ReadWrite();
   MFEM_FORALL(i, test_csz, d_b[test_idx[i]] = 0.0;);
}

void RectangularConstrainedOperator::Mult(const Vector &x, Vector &y) const
{
   const int trial_csz = trial_constraints.Size();
   const int test_csz = test_constraints.Size();
   if (trial_csz == 0)
   {
      A->Mult(x, y);
   }
   else
   {
      w = x;

      auto idx = trial_constraints.Read();
      // Use read+write access - we are modifying sub-vector of w
      auto d_w = w.ReadWrite();
      MFEM_FORALL(i, trial_csz, d_w[idx[i]] = 0.0;);

      A->Mult(w, y);
   }

   if (test_csz != 0)
   {
      auto idx = test_constraints.Read();
      auto d_y = y.ReadWrite();
      MFEM_FORALL(i, test_csz, d_y[idx[i]] = 0.0;);
   }
}

double PowerMethod::EstimateLargestEigenvalue(Operator& opr, Vector& v0,
                                              int numSteps, double tolerance, int seed)
{
   v1.SetSize(v0.Size());
   v0.Randomize(seed);

   double eigenvalue = 1.0;

   for (int iter = 0; iter < numSteps; ++iter)
   {
      double normV0;

#ifdef MFEM_USE_MPI
      if (comm != MPI_COMM_NULL)
      {
         normV0 = InnerProduct(comm, v0, v0);
      }
      else
      {
         normV0 = InnerProduct(v0, v0);
      }
#else
      normV0 = InnerProduct(v0, v0);
#endif

      v0 /= sqrt(normV0);
      opr.Mult(v0, v1);

      double eigenvalueNew;
#ifdef MFEM_USE_MPI
      if (comm != MPI_COMM_NULL)
      {
         eigenvalueNew = InnerProduct(comm, v0, v1);
      }
      else
      {
         eigenvalueNew = InnerProduct(v0, v1);
      }
#else
      eigenvalueNew = InnerProduct(v0, v1);
#endif
      double diff = std::abs((eigenvalueNew - eigenvalue) / eigenvalue);

      eigenvalue = eigenvalueNew;
      std::swap(v0, v1);

      if (diff < tolerance)
      {
         break;
      }
   }

   return eigenvalue;
}

}
