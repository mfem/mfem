// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_OPERATOR
#define MFEM_OPERATOR

#include "vector.hpp"

/// Abstract operator
class Operator
{
protected:
   int size;

public:
   /// Construct Operator with given size s (default 0)
   explicit Operator (int s = 0) { size = s; }

   /// Returns the size of the input
   inline int Size() const { return size; }

   /// Operator application
   virtual void Mult (const Vector & x, Vector & y) const = 0;

   /// Action of the transpose operator
   virtual void MultTranspose (const Vector & x, Vector & y) const
   { mfem_error ("Operator::MultTranspose() is not overloaded!"); }

   /// Prints operator with input size n and output size m in matlab format.
   void PrintMatlab (ostream & out, int n = 0, int m = 0);

   virtual ~Operator() { }
};


/// Operator I: x -> x
class IdentityOperator : public Operator
{
public:
   /// Creates I_{nxn}
   explicit IdentityOperator (int n) { size = n; }

   /// Operator application
   virtual void Mult (const Vector & x, Vector & y) const { y = x; }

   ~IdentityOperator() { }
};


/// The transpose of a given operator (square matrix)
class TransposeOperator : public Operator
{
private:
   Operator * A;

public:
   /// Saves the operator
   TransposeOperator (Operator * a) : A(a) { size = A -> Size(); }

   /// Operator application
   virtual void Mult (const Vector & x, Vector & y) const
   { A -> MultTranspose(x,y); }

   virtual void MultTranspose (const Vector & x, Vector & y) const
   { A -> Mult(x,y); }

   ~TransposeOperator() { }
};

#endif
