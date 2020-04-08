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

#ifndef MFEM_ELEMMAT
#define MFEM_ELEMMAT

#include "../general/cuda.hpp"
#include "vector.hpp"
#include <iostream>

namespace mfem
{

class ElementMatrix
{
private:
   const int ne;
   const int ndofs;
   const Vector &data;

public:
   ElementMatrix(const Vector &vec, const int ne, const int ndofs)
      : ne(ne), ndofs(ndofs), data(vec)
   {
   }

   void AddMult(const Vector &x, Vector &y) const;

   void AddMultTranspose(const Vector &x, Vector &y) const;

   friend std::ostream& operator<<(std::ostream& os, const ElementMatrix& data);
};

std::ostream& operator<<(std::ostream& os, const ElementMatrix& mat);

class FaceMatrixInt
{
private:
   const int nf;
   const int ndofs;// on the face
   const Vector &data;

public:
   FaceMatrixInt(const Vector &vec, const int nf, const int ndofs)
      : nf(nf), ndofs(ndofs), data(vec)
   {
   }

   void AddMult(const Vector &x, Vector &y) const;

   void AddMultTranspose(const Vector &x, Vector &y) const;

   friend std::ostream& operator<<(std::ostream& os, const FaceMatrixInt& mat);
};

std::ostream& operator<<(std::ostream& os, const FaceMatrixInt& mat);

class FaceMatrixExt
{
private:
   const int nf;
   const int ndofs;// on the face
   const Vector &data;

public:
   FaceMatrixExt(const Vector &vec, const int nf, const int ndofs)
      : nf(nf), ndofs(ndofs), data(vec)
   {
   }

   void AddMult(const Vector &x, Vector &y) const;

   void AddMultTranspose(const Vector &x, Vector &y) const;

   friend std::ostream& operator<<(std::ostream& os, const FaceMatrixExt& mat);
};

std::ostream& operator<<(std::ostream& os, const FaceMatrixExt& mat);

class FaceMatrixBdr
{
private:
   const int nf;
   const int ndofs;// on the face
   const Vector &data;

public:
   FaceMatrixBdr(const Vector &vec, const int nf, const int ndofs)
      : nf(nf), ndofs(ndofs), data(vec)
   {
   }

   void AddMult(const Vector &x, Vector &y) const;

   void AddMultTranspose(const Vector &x, Vector &y) const;

   friend std::ostream& operator<<(std::ostream& os, const FaceMatrixBdr& mat);
};

std::ostream& operator<<(std::ostream& os, const FaceMatrixBdr& mat);

}

#endif
