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

#ifndef MFEM_OPERATOR
#define MFEM_OPERATOR

#include "vector.hpp"

namespace mfem
{

/// Abstract operator
class Operator
{
protected:
   int height, width;

public:
   /// Construct a square Operator with given size s (default 0)
   explicit Operator(int s = 0) { height = width = s; }

   /** Construct an Operator with the given height (output size) and width
       (input size). */
   Operator(int h, int w) { height = h; width = w; }

   /// Get the height (size of output) of the Operator. Synonym with NumRows.
   inline int Height() const { return height; }
   /** Get the number of rows (size of output) of the Operator. Synonym with
       Height. */
   inline int NumRows() const { return height; }

   /// Get the width (size of input) of the Operator. Synonym with NumCols.
   inline int Width() const { return width; }
   /** Get the number of columns (size of input) of the Operator. Synonym with
       Width. */
   inline int NumCols() const { return width; }

   /// Operator application
   virtual void Mult(const Vector &x, Vector &y) const = 0;

   /// Action of the transpose operator
   virtual void MultTranspose(const Vector &x, Vector &y) const
   { mfem_error ("Operator::MultTranspose() is not overloaded!"); }

   /// Evaluate the gradient operator at the point x
   virtual Operator &GetGradient(const Vector &x) const
   {
      mfem_error("Operator::GetGradient() is not overloaded!");
      return const_cast<Operator &>(*this);
   }

   /// Prints operator with input size n and output size m in matlab format.
   void PrintMatlab (std::ostream & out, int n = 0, int m = 0) const;

   virtual ~Operator() { }
};


/// Base abstract class for time dependent operators: (x,t) -> f(x,t)
class TimeDependentOperator : public Operator
{
protected:
   double t;

public:
   /** Construct a "square" time dependent Operator y = f(x,t), where x and y
       have the same dimension 'n'. */
   explicit TimeDependentOperator(int n = 0, double _t = 0.0)
      : Operator(n) { t = _t; }

   /** Construct a time dependent Operator y = f(x,t), where x and y have
       dimensions 'w' and 'h', respectively. */
   TimeDependentOperator(int h, int w, double _t = 0.0)
      : Operator(h, w) { t = _t; }

   virtual double GetTime() const { return t; }

   virtual void SetTime(const double _t) { t = _t; }

   /** Solve the equation: k = f(x + dt*k, t), for the unknown k.
       This method allows for the abstract implementation of some time
       integration methods, including diagonal implicit Runge-Kutta (DIRK)
       methods and the backward Euler method in particular. */
   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k)
   {
      mfem_error("TimeDependentOperator::ImplicitSolve() is not overloaded!");
   }

   virtual ~TimeDependentOperator() { }
};


/// Base class for solvers
class Solver : public Operator
{
public:
   /// If true, use the second argument of Mult as an initial guess
   bool iterative_mode;

   /** Initialize a square Solver with size 's'.
       WARNING: use a boolean expression for the second parameter (not an int)
       to distinguish this call from the general rectangular constructor. */
   explicit Solver(int s = 0, bool iter_mode = false)
      : Operator(s) { iterative_mode = iter_mode; }

   /// Initialize a Solver with height 'h' and width 'w'
   Solver(int h, int w, bool iter_mode = false)
      : Operator(h, w) { iterative_mode = iter_mode; }

   /// Set/update the solver for the given operator
   virtual void SetOperator(const Operator &op) = 0;
};


/// Operator I: x -> x
class IdentityOperator : public Operator
{
public:
   /// Creates I_{nxn}
   explicit IdentityOperator(int n) : Operator(n) { }

   /// Operator application
   virtual void Mult(const Vector &x, Vector &y) const { y = x; }

   ~IdentityOperator() { }
};


/// The transpose of a given operator
class TransposeOperator : public Operator
{
private:
   const Operator &A;

public:
   /// Construct the transpose of a given operator
   TransposeOperator(const Operator *a)
      : Operator(a->Width(), a->Height()), A(*a) { }

   /// Construct the transpose of a given operator
   TransposeOperator(const Operator &a)
      : Operator(a.Width(), a.Height()), A(a) { }

   /// Operator application
   virtual void Mult(const Vector &x, Vector &y) const
   { A.MultTranspose(x, y); }

   virtual void MultTranspose(const Vector &x, Vector &y) const
   { A.Mult(x, y); }

   ~TransposeOperator() { }
};


/// The operator x -> R*A*P*x
class RAPOperator : public Operator
{
private:
   Operator & Rt;
   Operator & A;
   Operator & P;
   mutable Vector Px;
   mutable Vector APx;

public:
   /// Construct the RAP operator given R^T, A and P
   RAPOperator(Operator &Rt_, Operator &A_, Operator &P_)
      : Operator(Rt_.Width(), P_.Width()), Rt(Rt_), A(A_), P(P_),
        Px(P.Height()), APx(A.Height()) { }

   /// Operator application
   virtual void Mult(const Vector & x, Vector & y) const
   { P.Mult(x, Px); A.Mult(Px, APx); Rt.MultTranspose(APx, y); }

   virtual void MultTranspose(const Vector & x, Vector & y) const
   { Rt.Mult(x, APx); A.MultTranspose(APx, Px); P.MultTranspose(Px, y); }

   virtual ~RAPOperator() { }
};


/// General triple product operator x -> A*B*C*x, with ownership of the factors
class TripleProductOperator : public Operator
{
   Operator *A;
   Operator *B;
   Operator *C;
   bool ownA, ownB, ownC;
   mutable Vector t1, t2;

public:
   TripleProductOperator(Operator *A, Operator *B, Operator *C,
                         bool ownA, bool ownB, bool ownC)
      : Operator(A->Height(), C->Width())
      , A(A), B(B), C(C)
      , ownA(ownA), ownB(ownB), ownC(ownC)
      , t1(C->Height()), t2(B->Height())
   {}

   virtual void Mult(const Vector &x, Vector &y) const
   { C->Mult(x, t1); B->Mult(t1, t2); A->Mult(t2, y); }

   virtual void MultTranspose(const Vector &x, Vector &y) const
   { A->MultTranspose(x, t2); B->MultTranspose(t2, t1); C->MultTranspose(t1, y); }

   virtual ~TripleProductOperator()
   {
      if (ownA) { delete A; }
      if (ownB) { delete B; }
      if (ownC) { delete C; }
   }
};

}

#endif
