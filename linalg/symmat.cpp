// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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

DenseSymmetricMatrix::DenseSymmetricMatrix() : Matrix(0)
{
   data.Reset();
}

DenseSymmetricMatrix::DenseSymmetricMatrix(int s)
: Matrix(s), data((s*(s+1))/2)
{
   MFEM_ASSERT(s >= 0, "invalid DenseSymmetricMatrix size: " << s);
   *this = 0.0; // init with zeroes
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
   const int s2 = (s*(s+1))/2;
   if (s2 > data.Capacity())
   {
      data.Delete();
      data.New(s2);
      *this = 0.0; // init with zeroes
   }
}

DenseSymmetricMatrix &DenseSymmetricMatrix::operator=(double c)
{
   const int s = (Height()*(Height()+1))/2;
   for (int i = 0; i < s; i++)
   {
      data[i] = c;
   }
   return *this;
}

double &DenseSymmetricMatrix::Elem(int i, int j)
{
   return (*this)(i,j);
}

const double &DenseSymmetricMatrix::Elem(int i, int j) const
{
   return (*this)(i,j);
}

DenseSymmetricMatrix &DenseSymmetricMatrix::operator*=(double c)
{
   int s = Height()*(Height()+1)/2;
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

void DenseSymmetricMatrix::Print (std::ostream & out, int width_) const
{
   mfem_error("DenseSymmetricMatrix::Print() not implemented!");
}

DenseSymmetricMatrix::~DenseSymmetricMatrix()
{
   data.Delete();
}

}
