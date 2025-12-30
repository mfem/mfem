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

#include "../kernels.hpp"
#include "native.hpp"
#include "../dtensor.hpp"
#include "../../general/forall.hpp"

namespace mfem
{

void NativeBatchedLinAlg::AddMult(const DenseTensor &A, const Vector &x,
                                  Vector &y, real_t alpha, real_t beta,
                                  Op op) const
{
   const bool tr = (op == Op::T);

   const int m = A.SizeI();
   const int n = A.SizeJ();
   const int n_mat = A.SizeK();
   const int k = x.Size() / (tr ? m : n) / n_mat;

   auto d_A = Reshape(A.Read(), m, n, n_mat);
   auto d_x = Reshape(x.Read(), (tr ? m : n), k, n_mat);
   auto d_y = Reshape(beta == 0.0 ? y.Write() : y.ReadWrite(),
                      (tr ? n : m), k, n_mat);

   if (tr)
   {
      mfem::forall(n_mat, [=] MFEM_HOST_DEVICE (int i)
      {
         kernels::AddMultAtB(m, n, k, &d_A(0,0,i), &d_x(0,0,i), &d_y(0,0,i),
                             alpha, beta);
      });
   }
   else
   {
      mfem::forall(n_mat, [=] MFEM_HOST_DEVICE (int i)
      {
         kernels::AddMult(m, k, n, &d_A(0,0,i), &d_x(0,0,i), &d_y(0,0,i),
                          alpha, beta);
      });
   }

   // Alternative approach, threading also over the second index. Which one is
   // better?

   // mfem::forall(n_mat * k, [=] MFEM_HOST_DEVICE (int idx)
   // {
   //    const int i = idx % k;
   //    const int j = idx / k;
   //    kernels::Mult(m, n, &d_A(0,0,j), &d_x(0,i,j), &d_y(0,i,j));
   // });
}

void NativeBatchedLinAlg::Invert(DenseTensor &A) const
{
   const int m = A.SizeI();
   const int NE = A.SizeK();
   DenseTensor LU = A;
   Array<int> P(m*NE);

   LUFactor(LU, P);

   auto data_all = Reshape(LU.Read(), m, m, NE);
   auto piv_all  = Reshape(P.Read(), m, NE);
   auto inv_all  = Reshape(A.Write(), m, m, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      // A^{-1} = U^{-1} L^{-1} P
      // X <- U^{-1} (set only the upper triangular part of X)
      real_t *X          = &inv_all(0, 0, e);
      real_t *x          = X;
      const real_t *data = &data_all(0, 0, e);
      const int *ipiv    = &piv_all(0, e);

      for (int k = 0; k < m; k++)
      {
         const real_t minus_x_k = -(x[k] = 1.0 / data[k + k * m]);
         for (int i = 0; i < k; i++)
         {
            x[i] = data[i + k * m] * minus_x_k;
         }
         for (int j = k - 1; j >= 0; j--)
         {
            const real_t x_j = (x[j] /= data[j + j * m]);
            for (int i = 0; i < j; i++)
            {
               x[i] -= data[i + j * m] * x_j;
            }
         }
         x += m;
      }

      // X <- X L^{-1} (use input only from the upper triangular part of X)
      {
         int k = m - 1;
         for (int j = 0; j < k; j++)
         {
            const real_t minus_L_kj = -data[k + j * m];
            for (int i = 0; i <= j; i++)
            {
               X[i + j * m] += X[i + k * m] * minus_L_kj;
            }
            for (int i = j + 1; i < m; i++)
            {
               X[i + j * m] = X[i + k * m] * minus_L_kj;
            }
         }
      }
      for (int k = m - 2; k >= 0; k--)
      {
         for (int j = 0; j < k; j++)
         {
            const real_t L_kj = data[k + j * m];
            for (int i = 0; i < m; i++)
            {
               X[i + j * m] -= X[i + k * m] * L_kj;
            }
         }
      }

      // X <- X P
      for (int k = m - 1; k >= 0; k--)
      {
         const int piv_k = ipiv[k] - 1;
         if (k != piv_k)
         {
            for (int i = 0; i < m; i++)
            {
               kernels::internal::Swap(X[i + k * m], X[i + piv_k * m]);
            }
         }
      }
   });
}

void NativeBatchedLinAlg::LUFactor(DenseTensor &A, Array<int> &P) const
{
   const int m = A.SizeI();
   const int NE = A.SizeK();
   P.SetSize(m*NE);

   auto data_all = Reshape(A.ReadWrite(), m, m, NE);
   auto ipiv_all = Reshape(P.Write(), m, NE);
   Array<bool> pivot_flag(1);
   pivot_flag[0] = true;
   bool *d_pivot_flag = pivot_flag.ReadWrite();

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      const bool flag = kernels::LUFactor(&data_all(0,0,e), m, &ipiv_all(0,e));
      if (!flag) { d_pivot_flag[0] = false; }
   });

   MFEM_VERIFY(pivot_flag.HostRead()[0], "Batch LU factorization failed");
}

void NativeBatchedLinAlg::LUSolve(const DenseTensor &LU, const Array<int> &P,
                                  Vector &x) const
{
   const int m = LU.SizeI();
   const int n_mat = LU.SizeK();
   const int n_rhs = x.Size() / m / n_mat;

   auto d_LU = Reshape(LU.Read(), m, m, n_mat);
   auto d_P = Reshape(P.Read(), m, n_mat);
   auto d_x = Reshape(x.Write(), m, n_rhs, n_mat);

   mfem::forall(n_mat * n_rhs, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int i_rhs = idx % n_rhs;
      const int i_mat = idx / n_rhs;

      kernels::LUSolve(&d_LU(0,0,i_mat), m, &d_P(0,i_mat), &d_x(0,i_rhs,i_mat));
   });
}

}
