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
#include "../general/forall.hpp"
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

   void AddMult(const Vector &x, Vector &y) const
   {
      const int NDOFS = ndofs;
      auto X = Reshape(x.Read(), ndofs, ne);
      auto Y = Reshape(y.ReadWrite(), ndofs, ne);
      auto A = Reshape(data.Read(), ndofs, ndofs, ne);
      MFEM_FORALL(glob_j, ne*ndofs,
      {
         const int e = glob_j/NDOFS;
         const int j = glob_j%NDOFS;
         double res = 0.0;
         for (int i = 0; i < NDOFS; i++)
         {
            res += A(i, j, e)*X(i, e);
         }
         Y(j, e) += res;
      });
   }

   void AddMultTranspose(const Vector &x, Vector &y) const
   {
      const int NDOFS = ndofs;
      auto X = Reshape(x.Read(), ndofs, ne);
      auto Y = Reshape(y.ReadWrite(), ndofs, ne);
      auto A = Reshape(data.Read(), ndofs, ndofs, ne);
      MFEM_FORALL(glob_j, ne*ndofs,
      {
         const int e = glob_j/NDOFS;
         const int j = glob_j%NDOFS;
         double res = 0.0;
         for (int i = 0; i < NDOFS; i++)
         {
            res += A(j, i, e)*X(i, e);
         }
         Y(j, e) += res;
      });
   }

   friend std::ostream& operator<<(std::ostream& os, const ElementMatrix& data);
};

std::ostream& operator<<(std::ostream& os, const ElementMatrix& mat)
{
   for (size_t e = 0; e < mat.ne; e++)
   {
      os << "Element "<< e <<std::endl;
      for (size_t i = 0; i < mat.ndofs; i++)
      {
         for (size_t j = 0; j < mat.ndofs; j++)
         {
            os << mat.data[i+j*mat.ndofs+e*mat.ndofs*mat.ndofs] << ", ";
         }
         os << std::endl;
      }
      os << std::endl;
      os << std::endl;
   }
   return os;
}

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

   void AddMult(const Vector &x, Vector &y) const
   {
      auto X = Reshape(x.Read(), ndofs, 2, nf);
      auto Y = Reshape(y.ReadWrite(), ndofs, 2, nf);
      auto A = Reshape(data.Read(), ndofs, ndofs, 2, nf);
      const int NDOFS = ndofs;
      MFEM_FORALL(glob_j, nf*ndofs,
      {
         const int f = glob_j/NDOFS;
         const int j = glob_j%NDOFS;
         double res = 0.0;
         for (int i = 0; i < NDOFS; i++)
         {
            res += A(i, j, 0, f)*X(i, 0, f);
         }
         Y(j, 0, f) += res;
         res = 0.0;
         for (int i = 0; i < NDOFS; i++)
         {
            res += A(i, j, 1, f)*X(i, 1, f);
         }
         Y(j, 1, f) += res;
      });
   }

   void AddMultTranspose(const Vector &x, Vector &y) const
   {
      auto X = Reshape(x.Read(), ndofs, 2, nf);
      auto Y = Reshape(y.ReadWrite(), ndofs, 2, nf);
      auto A = Reshape(data.Read(), ndofs, ndofs, 2, nf);
      const int NDOFS = ndofs;
      MFEM_FORALL(glob_j, nf*ndofs,
      {
         const int f = glob_j/NDOFS;
         const int j = glob_j%NDOFS;
         double res = 0.0;
         for (int i = 0; i < NDOFS; i++)
         {
            res += A(j, i, 0, f)*X(i, 0, f);
         }
         Y(j, 0, f) += res;
         res = 0.0;
         for (int i = 0; i < NDOFS; i++)
         {
            res += A(j, i, 1, f)*X(i, 1, f);
         }
         Y(j, 1, f) += res;
      });
   }

   friend std::ostream& operator<<(std::ostream& os, const FaceMatrixInt& mat);
};

std::ostream& operator<<(std::ostream& os, const FaceMatrixInt& mat)
{
   for (size_t f = 0; f < mat.nf; f++)
   {
      os << "Face "<<f <<std::endl;
      for (size_t i = 0; i < mat.ndofs; i++)
      {
         for (size_t j = 0; j < mat.ndofs; j++)
         {
            os << mat.data[i+j*mat.ndofs+2*f*mat.ndofs*mat.ndofs] << ", ";
         }
         os << std::endl;
      }
      os << std::endl;
      for (size_t i = 0; i < mat.ndofs; i++)
      {
         for (size_t j = 0; j < mat.ndofs; j++)
         {
            os << mat.data[i+j*mat.ndofs+(2*f+1)*mat.ndofs*mat.ndofs] << ", ";
         }
         os << std::endl;
      }
      os << std::endl;
      os << std::endl;
   }   
   return os;
}

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

   void AddMult(const Vector &x, Vector &y) const
   {
      auto X = Reshape(x.Read(), ndofs, 2, nf);
      auto Y = Reshape(y.ReadWrite(), ndofs, 2, nf);
      auto A = Reshape(data.Read(), ndofs, ndofs, 2, nf);
      const int NDOFS = ndofs;
      MFEM_FORALL(glob_j, nf*ndofs,
      {
         const int f = glob_j/NDOFS;
         const int j = glob_j%NDOFS;
         double res = 0.0;
         for (int i = 0; i < NDOFS; i++)
         {
            res += A(i, j, 0, f)*X(i, 0, f);
         }
         Y(j, 1, f) += res;
         res = 0.0;
         for (int i = 0; i < NDOFS; i++)
         {
            res += A(i, j, 1, f)*X(i, 1, f);
         }
         Y(j, 0, f) += res;
      });
   }

   void AddMultTranspose(const Vector &x, Vector &y) const
   {
      auto X = Reshape(x.Read(), ndofs, 2, nf);
      auto Y = Reshape(y.ReadWrite(), ndofs, 2, nf);
      auto A = Reshape(data.Read(), ndofs, ndofs, 2, nf);
      const int NDOFS = ndofs;
      MFEM_FORALL(glob_j, nf*ndofs,
      {
         const int f = glob_j/NDOFS;
         const int j = glob_j%NDOFS;
         double res = 0.0;
         for (int i = 0; i < NDOFS; i++)
         {
            res += A(j, i, 0, f)*X(i, 0, f);
         }
         Y(j, 1, f) += res;
         res = 0.0;
         for (int i = 0; i < NDOFS; i++)
         {
            res += A(j, i, 1, f)*X(i, 1, f);
         }
         Y(j, 0, f) += res;
      });
   }

   friend std::ostream& operator<<(std::ostream& os, const FaceMatrixExt& mat);
};

std::ostream& operator<<(std::ostream& os, const FaceMatrixExt& mat)
{
   for (size_t f = 0; f < mat.nf; f++)
   {
      os << "Face "<<f <<std::endl;
      for (size_t i = 0; i < mat.ndofs; i++)
      {
         for (size_t j = 0; j < mat.ndofs; j++)
         {
            os << mat.data[i+j*mat.ndofs+2*f*mat.ndofs*mat.ndofs] << ", ";
         }
         os << std::endl;
      }
      os << std::endl;
      for (size_t i = 0; i < mat.ndofs; i++)
      {
         for (size_t j = 0; j < mat.ndofs; j++)
         {
            os << mat.data[i+j*mat.ndofs+(2*f+1)*mat.ndofs*mat.ndofs] << ", ";
         }
         os << std::endl;
      }
      os << std::endl;
      os << std::endl;
   }   
   return os;
}

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

   void AddMult(const Vector &x, Vector &y) const
   {
      auto X = Reshape(x.Read(), ndofs, nf);
      auto Y = Reshape(y.ReadWrite(), ndofs, nf);
      auto A = Reshape(data.Read(), ndofs, ndofs, nf);
      const int NDOFS = ndofs;
      MFEM_FORALL(glob_j, nf*ndofs,
      {
         const int f = glob_j/NDOFS;
         const int j = glob_j%NDOFS;
         double res = 0.0;
         for (int i = 0; i < NDOFS; i++)
         {
            res += A(i, j, f)*X(i, f);
         }
         Y(j, f) += res;
      });
   }

   void AddMultTranspose(const Vector &x, Vector &y) const
   {
      auto X = Reshape(x.Read(), ndofs, nf);
      auto Y = Reshape(y.ReadWrite(), ndofs, nf);
      auto A = Reshape(data.Read(), ndofs, ndofs, nf);
      const int NDOFS = ndofs;
      MFEM_FORALL(glob_j, nf*ndofs,
      {
         const int f = glob_j/NDOFS;
         const int j = glob_j%NDOFS;
         double res = 0.0;
         for (int i = 0; i < NDOFS; i++)
         {
            res += A(j, i, f)*X(i, f);
         }
         Y(j, f) += res;
      });
   }

   friend std::ostream& operator<<(std::ostream& os, const FaceMatrixBdr& mat);
};

std::ostream& operator<<(std::ostream& os, const FaceMatrixBdr& mat)
{
   for (size_t f = 0; f < mat.nf; f++)
   {
      os << "Face "<<f <<std::endl;
      for (size_t i = 0; i < mat.ndofs; i++)
      {
         for (size_t j = 0; j < mat.ndofs; j++)
         {
            os << mat.data[i+j*mat.ndofs+f*mat.ndofs*mat.ndofs] << ", ";
         }
         os << std::endl;
      }
      os << std::endl;
      os << std::endl;
   }   
   return os;
}

}

#endif
