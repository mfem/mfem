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

#ifndef MFEM_MATRIX
#define MFEM_MATRIX

// Abstract data types matrix, inverse matrix

#include "../general/array.hpp"
#include "operator.hpp"

class  MatrixInverse;

/// Abstract data type matrix
class Matrix : public Operator
{
   friend class MatrixInverse;
public:
   /// Creates matrix of width s.
   Matrix (int s) { size=s; }

   /// Returns reference to a_{ij}.  Index i, j = 0 .. size-1
   virtual double& Elem (int i, int j) = 0;

   /// Returns constant reference to a_{ij}.  Index i, j = 0 .. size-1
   virtual const double& Elem (int i, int j) const = 0;

   /// Returns a pointer to (approximation) of the matrix inverse.
   virtual MatrixInverse * Inverse() const = 0;

   /// Finalizes the matrix initialization.
   virtual void Finalize(int) { }

   /// Prints matrix to stream out.
   virtual void Print (ostream & out = cout, int width = 4) const;

   /// Destroys matrix.
   virtual ~Matrix() { }
};


/// Abstract data type for matrix inverse
class MatrixInverse : public Operator
{
protected:
   const Matrix *a;

public:
   /// Creates approximation of the inverse of square matrix
   MatrixInverse (const Matrix &mat) { size = mat.size; a = &mat; }

   /// Destroys inverse matrix.
   virtual ~MatrixInverse() { }
};

#endif
