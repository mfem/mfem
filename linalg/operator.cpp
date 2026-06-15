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
      B.MakeRef(b, 0, b.Size());
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
      X.MakeRef(x, 0, x.Size());
   }
}

void Operator::AddMult(const Vector &x, Vector &y, const real_t a) const
{
   mfem::Vector z(y.Size());
   Mult(x, z);
   y.Add(a, z);
}

void Operator::AddMultTranspose(const Vector &x, Vector &y,
                                const real_t a) const
{
   mfem::Vector z(y.Size());
   MultTranspose(x, z);
   y.Add(a, z);
}

void Operator::ArrayMult(const Array<const Vector *> &X,
                         Array<Vector *> &Y) const
{
   MFEM_ASSERT(X.Size() == Y.Size(),
               "Number of columns mismatch in Operator::Mult!");
   for (int i = 0; i < X.Size(); i++)
   {
      MFEM_ASSERT(X[i] && Y[i], "Missing Vector in Operator::Mult!");
      Mult(*X[i], *Y[i]);
   }
}

void Operator::ArrayMultTranspose(const Array<const Vector *> &X,
                                  Array<Vector *> &Y) const
{
   MFEM_ASSERT(X.Size() == Y.Size(),
               "Number of columns mismatch in Operator::MultTranspose!");
   for (int i = 0; i < X.Size(); i++)
   {
      MFEM_ASSERT(X[i] && Y[i], "Missing Vector in Operator::MultTranspose!");
      MultTranspose(*X[i], *Y[i]);
   }
}

void Operator::ArrayAddMult(const Array<const Vector *> &X, Array<Vector *> &Y,
                            const real_t a) const
{
   MFEM_ASSERT(X.Size() == Y.Size(),
               "Number of columns mismatch in Operator::AddMult!");
   for (int i = 0; i < X.Size(); i++)
   {
      MFEM_ASSERT(X[i] && Y[i], "Missing Vector in Operator::AddMult!");
      AddMult(*X[i], *Y[i], a);
   }
}

void Operator::ArrayAddMultTranspose(const Array<const Vector *> &X,
                                     Array<Vector *> &Y, const real_t a) const
{
   MFEM_ASSERT(X.Size() == Y.Size(),
               "Number of columns mismatch in Operator::AddMultTranspose!");
   for (int i = 0; i < X.Size(); i++)
   {
      MFEM_ASSERT(X[i] && Y[i], "Missing Vector in Operator::AddMultTranspose!");
      AddMultTranspose(*X[i], *Y[i], a);
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

void Operator::PrintMatlab(std::ostream & os, int n, int m) const
{
   using namespace std;
   if (n == 0) { n = width; }
   if (m == 0) { m = height; }

   Vector x(n), y(m);
   x = 0.0;

   os << setiosflags(ios::scientific | ios::showpos);
   for (int i = 0; i < n; i++)
   {
      x(i) = 1.0;
      Mult(x, y);
      for (int j = 0; j < m; j++)
      {
         if (y(j) != 0)
         {
            os << j+1 << " " << i+1 << " " << y(j) << '\n';
         }
      }
      x(i) = 0.0;
   }
}

void Operator::PrintMatlab(std::ostream &os) const
{
   PrintMatlab(os, width, height);
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

void TimeDependentOperator::ImplicitSolve(const real_t, const Vector &,
                                          Vector &)
{
   mfem_error("TimeDependentOperator::ImplicitSolve() is not overridden!");
}

Operator &TimeDependentOperator::GetImplicitGradient(
   const Vector &, const Vector &, real_t) const
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
                                            int, int *, real_t)
{
   mfem_error("TimeDependentOperator::SUNImplicitSetup() is not overridden!");
   return (-1);
}

int TimeDependentOperator::SUNImplicitSolve(const Vector &, Vector &, real_t)
{
   mfem_error("TimeDependentOperator::SUNImplicitSolve() is not overridden!");
   return (-1);
}

int TimeDependentOperator::SUNMassSetup()
{
   mfem_error("TimeDependentOperator::SUNMassSetup() is not overridden!");
   return (-1);
}

int TimeDependentOperator::SUNMassSolve(const Vector &, Vector &, real_t)
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

void SecondOrderTimeDependentOperator::ImplicitSolve(const real_t dt0,
                                                     const real_t dt1,
                                                     const Vector &x,
                                                     const Vector &dxdt,
                                                     Vector &k)
{
   mfem_error("SecondOrderTimeDependentOperator::ImplicitSolve() is not overridden!");
}

SumOperator::SumOperator(const Operator *A, const real_t alpha,
                         const Operator *B, const real_t beta,
                         bool ownA, bool ownB)
   : Operator(A->Height(), A->Width()),
     A(A), B(B), alpha(alpha), beta(beta), ownA(ownA), ownB(ownB),
     z(A->Height())
{
   MFEM_VERIFY(A->Width() == B->Width(),
               "incompatible Operators: different widths\n"
               << "A->Width() = " << A->Width()
               << ", B->Width() = " << B->Width() );
   MFEM_VERIFY(A->Height() == B->Height(),
               "incompatible Operators: different heights\n"
               << "A->Height() = " << A->Height()
               << ", B->Height() = " << B->Height() );

   {
      const Solver* SolverA = dynamic_cast<const Solver*>(A);
      const Solver* SolverB = dynamic_cast<const Solver*>(B);
      if (SolverA)
      {
         MFEM_VERIFY(!(SolverA->iterative_mode),
                     "Operator A of a SumOperator should not be in iterative mode");
      }
      if (SolverB)
      {
         MFEM_VERIFY(!(SolverB->iterative_mode),
                     "Operator B of a SumOperator should not be in iterative mode");
      }
   }

}

SumOperator::~SumOperator()
{
   if (ownA) { delete A; }
   if (ownB) { delete B; }
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
                                         bool own_A_,
                                         DiagonalPolicy diag_policy_)
   : Operator(A->Height(), A->Width()), A(A), own_A(own_A_),
     diag_policy(diag_policy_)
{
   // 'mem_class' should work with A->Mult() and mfem::forall():
   mem_class = A->GetMemoryClass()*Device::GetDeviceMemoryClass();
   MemoryType mem_type = GetMemoryType(mem_class);
   list.Read(); // TODO: just ensure 'list' is registered, no need to copy it
   constraint_list.MakeRef(list);
   // typically z and w are large vectors, so use the device (GPU) to perform
   // operations on them
   z.SetSize(height, mem_type); z.UseDevice(true);
   w.SetSize(height, mem_type); w.UseDevice(true);
}

void ConstrainedOperator::AssembleDiagonal(Vector &diag) const
{
   A->AssembleDiagonal(diag);

   if (diag_policy == DIAG_KEEP) { return; }

   const int csz = constraint_list.Size();
   auto d_diag = diag.ReadWrite();
   auto idx = constraint_list.Read();
   switch (diag_policy)
   {
      case DIAG_ONE:
         mfem::forall(csz, [=] MFEM_HOST_DEVICE (int i)
         {
            const int id = idx[i];
            d_diag[id] = 1.0;
         });
         break;
      case DIAG_ZERO:
         mfem::forall(csz, [=] MFEM_HOST_DEVICE (int i)
         {
            const int id = idx[i];
            d_diag[id] = 0.0;
         });
         break;
      default:
         MFEM_ABORT("unknown diagonal policy");
         break;
   }
}

void ConstrainedOperator::EliminateRHS(const Vector &x, Vector &b) const
{
   w = 0.0;
   const int csz = constraint_list.Size();
   auto idx = constraint_list.Read();
   auto d_x = x.Read();
   // Use read+write access - we are modifying sub-vector of w
   auto d_w = w.ReadWrite();
   mfem::forall(csz, [=] MFEM_HOST_DEVICE (int i)
   {
      const int id = idx[i];
      d_w[id] = d_x[id];
   });

   // A.AddMult(w, b, -1.0); // if available to all Operators
   A->Mult(w, z);
   b -= z;

   // Use read+write access - we are modifying sub-vector of b
   auto d_b = b.ReadWrite();
   mfem::forall(csz, [=] MFEM_HOST_DEVICE (int i)
   {
      const int id = idx[i];
      d_b[id] = d_x[id];
   });
}

void ConstrainedOperator::ConstrainedMult(const Vector &x, Vector &y,
                                          const bool transpose) const
{
   const int csz = constraint_list.Size();
   if (csz == 0)
   {
      if (transpose)
      {
         A->MultTranspose(x, y);
      }
      else
      {
         A->Mult(x, y);
      }
      return;
   }

   z = x;

   auto idx = constraint_list.Read();
   // Use read+write access - we are modifying sub-vector of z
   auto d_z = z.ReadWrite();
   mfem::forall(csz, [=] MFEM_HOST_DEVICE (int i) { d_z[idx[i]] = 0.0; });

   if (transpose)
   {
      A->MultTranspose(z, y);
   }
   else
   {
      A->Mult(z, y);
   }

   auto d_x = x.Read();
   // Use read+write access - we are modifying sub-vector of y
   auto d_y = y.ReadWrite();
   switch (diag_policy)
   {
      case DIAG_ONE:
         mfem::forall(csz, [=] MFEM_HOST_DEVICE (int i)
         {
            const int id = idx[i];
            d_y[id] = d_x[id];
         });
         break;
      case DIAG_ZERO:
         mfem::forall(csz, [=] MFEM_HOST_DEVICE (int i)
         {
            const int id = idx[i];
            d_y[id] = 0.0;
         });
         break;
      case DIAG_KEEP:
         // Needs action of the operator diagonal on vector
         mfem_error("ConstrainedOperator::Mult #1");
         break;
      default:
         mfem_error("ConstrainedOperator::Mult #2");
         break;
   }
}

void ConstrainedOperator::ConstrainedAbsMult(const Vector &x, Vector &y,
                                             const bool transpose) const
{
   const int csz = constraint_list.Size();
   if (csz == 0)
   {
      if (transpose)
      {
         A->AbsMultTranspose(x, y);
      }
      else
      {
         A->AbsMult(x, y);
      }
      return;
   }

   z = x;

   auto idx = constraint_list.Read();
   // Use read+write access - we are modifying sub-vector of z
   auto d_z = z.ReadWrite();
   mfem::forall(csz, [=] MFEM_HOST_DEVICE (int i) { d_z[idx[i]] = 0.0; });

   if (transpose)
   {
      A->AbsMultTranspose(z, y);
   }
   else
   {
      A->AbsMult(z, y);
   }

   auto d_x = x.Read();
   // Use read+write access - we are modifying sub-vector of y
   auto d_y = y.ReadWrite();
   switch (diag_policy)
   {
      case DIAG_ONE:
         mfem::forall(csz, [=] MFEM_HOST_DEVICE (int i)
         {
            const int id = idx[i];
            d_y[id] = d_x[id];
         });
         break;
      case DIAG_ZERO:
         mfem::forall(csz, [=] MFEM_HOST_DEVICE (int i)
         {
            const int id = idx[i];
            d_y[id] = 0.0;
         });
         break;
      case DIAG_KEEP:
         // Needs action of the operator diagonal on vector
         mfem_error("ConstrainedOperator::AbsMult #1");
         break;
      default:
         mfem_error("ConstrainedOperator::AbsMult #2");
         break;
   }
}

void ConstrainedOperator::Mult(const Vector &x, Vector &y) const
{
   constexpr bool transpose = false;
   ConstrainedMult(x, y, transpose);
}

void ConstrainedOperator::AbsMult(const Vector &x, Vector &y) const
{
   constexpr bool transpose = false;
   ConstrainedAbsMult(x, y, transpose);
}

void ConstrainedOperator::MultTranspose(const Vector &x, Vector &y) const
{
   constexpr bool transpose = true;
   ConstrainedMult(x, y, transpose);
}

void ConstrainedOperator::AbsMultTranspose(const Vector &x, Vector &y) const
{
   constexpr bool transpose = true;
   ConstrainedAbsMult(x, y, transpose);
}

void ConstrainedOperator::AddMult(const Vector &x, Vector &y,
                                  const real_t a) const
{
   Mult(x, w);
   y.Add(a, w);
}

RectangularConstrainedOperator::RectangularConstrainedOperator(
   Operator *A,
   const Array<int> &trial_list,
   const Array<int> &test_list,
   bool own_A_)
   : Operator(A->Height(), A->Width()), A(A), own_A(own_A_)
{
   // 'mem_class' should work with A->Mult() and mfem::forall():
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
   mfem::forall(trial_csz, [=] MFEM_HOST_DEVICE (int i)
   {
      const int id = trial_idx[i];
      d_w[id] = d_x[id];
   });

   A->AddMult(w, b, -1.0);

   const int test_csz = test_constraints.Size();
   auto test_idx = test_constraints.Read();
   auto d_b = b.ReadWrite();
   mfem::forall(test_csz, [=] MFEM_HOST_DEVICE (int i)
   {
      d_b[test_idx[i]] = 0.0;
   });
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
      mfem::forall(trial_csz, [=] MFEM_HOST_DEVICE (int i)
      {
         d_w[idx[i]] = 0.0;
      });

      A->Mult(w, y);
   }

   if (test_csz != 0)
   {
      auto idx = test_constraints.Read();
      auto d_y = y.ReadWrite();
      mfem::forall(test_csz, [=] MFEM_HOST_DEVICE (int i)
      {
         d_y[idx[i]] = 0.0;
      });
   }
}

void RectangularConstrainedOperator::MultTranspose(const Vector &x,
                                                   Vector &y) const
{
   const int trial_csz = trial_constraints.Size();
   const int test_csz = test_constraints.Size();
   if (test_csz == 0)
   {
      A->MultTranspose(x, y);
   }
   else
   {
      z = x;

      auto idx = test_constraints.Read();
      // Use read+write access - we are modifying sub-vector of z
      auto d_z = z.ReadWrite();
      mfem::forall(test_csz, [=] MFEM_HOST_DEVICE (int i)
      {
         d_z[idx[i]] = 0.0;
      });

      A->MultTranspose(z, y);
   }

   if (trial_csz != 0)
   {
      auto idx = trial_constraints.Read();
      auto d_y = y.ReadWrite();
      mfem::forall(trial_csz, [=] MFEM_HOST_DEVICE (int i)
      {
         d_y[idx[i]] = 0.0;
      });
   }
}

real_t InnerProductOperator::Dot(const Vector &x, const Vector &y) const
{
#ifndef MFEM_USE_MPI
   return (x * y);
#else
   if (dot_prod_type == 0)
   {
      return (x * y);
   }
   else
   {
      return InnerProduct(comm, x, y);
   }
#endif
}

real_t PowerMethod::EstimateLargestEigenvalue(Operator& opr, Vector& v0,
                                              int numSteps, real_t tolerance,
                                              int seed)
{
   v1.SetSize(v0.Size());
   if (seed != 0)
   {
      v0.Randomize(seed);
   }

   real_t eigenvalue = 1.0;

   for (int iter = 0; iter < numSteps; ++iter)
   {
      real_t normV0;

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

      real_t eigenvalueNew;
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
      real_t diff = std::abs((eigenvalueNew - eigenvalue) / eigenvalue);

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
