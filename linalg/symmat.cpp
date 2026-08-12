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


// Implementation of data type DenseSymmetricMatrix

#include "symmat.hpp"

namespace mfem
{

DenseSymmetricMatrix::DenseSymmetricMatrix() : Matrix(0) { }

DenseSymmetricMatrix::DenseSymmetricMatrix(int s) : Matrix(s)
{
   MFEM_ASSERT(s >= 0, "invalid DenseSymmetricMatrix size: " << s);
   if (s > 0)
   {
      data.SetSize((s*(s+1))/2);
      *this = 0.0; // init with zeroes
   }
}

void DenseSymmetricMatrix::SetSize(int s)
{
   MFEM_ASSERT(s >= 0,
               "invalid DenseSymmetricMatrix size: " << s);
   if (Height() == s)
   {
      return;
   }
   height = s;
   width = s;
   data.SetSize((s*(s+1))/2);
   *this = 0.0; // init with zeroes
}

DenseSymmetricMatrix &DenseSymmetricMatrix::operator=(real_t c)
{
   const int s = (Height()*(Height()+1))/2;
   for (int i = 0; i < s; i++)
   {
      data[i] = c;
   }
   return *this;
}

real_t &DenseSymmetricMatrix::Elem(int i, int j)
{
   return (*this)(i,j);
}

const real_t &DenseSymmetricMatrix::Elem(int i, int j) const
{
   return (*this)(i,j);
}

DenseSymmetricMatrix &DenseSymmetricMatrix::operator*=(real_t c)
{
   int s = GetStoredSize();
   for (int i = 0; i < s; i++)
   {
      data[i] *= c;
   }
   return *this;
}

void DenseSymmetricMatrix::Mult(const Vector &x, Vector &y) const
{
   mfem_error("DenseSymmetricMatrix::Mult() not implemented!");
}

MatrixInverse *DenseSymmetricMatrix::Inverse() const
{
   mfem_error("DenseSymmetricMatrix::Inverse() not implemented!");
   return nullptr;
}

}
