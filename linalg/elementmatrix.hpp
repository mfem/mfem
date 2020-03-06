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

#ifndef MFEM_ELEMMAT
#define MFEM_ELEMMAT

#include "../general/cuda.hpp"
#include "../general/forall.hpp"
#include "vector.hpp"

namespace mfem
{

template <typename Scalar>
class EMat
{
private:
   const int ne;
   const int ndofs;
   const Vector &data;

public:
   EMat(const Vector &vec, const int ne, const int ndofs)
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
         Scalar res = 0.0;
         for (int i = 0; i < NDOFS; i++)
         {
            res += A(i, j, e)*X(i, e);
         }
         Y(j, e) += res;
      });
   }

   // EMat<Scalar>& operator+=(const EMat<Scalar> &rhs)
   // {
   //    auto A = Reshape(data, ndofs, ndofs, ne);
   //    MFEM_FORALL_2D(e, ne, ndofs, ndofs, 1,
   //    {
   //       MFEM_FOREACH_THREAD(i,x,ndofs)
   //       {
   //          MFEM_FOREACH_THREAD(j,y,ndofs)
   //          {
   //             (*this)(e,i,j) += rhs(e,i,j);
   //          }
   //       }
   //    }
   //    return *this;
   // }

   // MFEM_HOST_DEVICE inline
   // Scalar& operator()(const int e, const int i, const int j)
   // {
   //    return data[e*ndofs*ndofs + j*ndofs + i];
   // }

   // // bofbof
   // MFEM_HOST_DEVICE
   // void SetEMat(const int e, const Vector e_mat)
   // {
   //    auto A = Reshape(data, ndofs, ndofs, ne);
   //    auto mat = Reshape(e_mat.Read(), ndofs, ndofs);
   //    for (int i = 0; i < ndofs; i++)
   //    {
   //       for (int j = 0; j < ndofs; j++)
   //       {
   //          A(i, j, e) = mat(i, j);
   //       }
   //    }
   // }
};

using ElementMatrix = EMat<double>;

// template <typename Scalar>
// void L1Smoother(EMat<Scalar> &A, Vector &s)
// {
//    auto S = Reshape(s.Write(), ndofs, ne);
//    auto A = Reshape(data.Read(), ndofs, ndofs, ne);
//    MFEM_FORALL(glob_j, ne*ndofs,
//    {
//       const int e = glob_j/ndofs;
//       const int j = glob_j%ndofs;
//       Scalar res = 0.0;
//       for (int i = 0; i < ndofs; i++)
//       {
//          const Scalar val = A(i, j, e);
//          res += val>0?val:-val;
//       }
//       S(glob_j, e) = res;
//    }
// }

}

#endif
