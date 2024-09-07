// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
                                  Vector &y, real_t alpha, real_t beta) const
{
   const int m = A.SizeI();
   const int n = A.SizeJ();
   const int n_mat = A.SizeK();
   const int k = x.Size() / n / n_mat;

   auto d_A = Reshape(A.Read(), m, n, n_mat);
   auto d_x = Reshape(x.Read(), n, k, n_mat);
   auto d_y = Reshape(beta == 0.0 ? y.Write() : y.ReadWrite(), m, k, n_mat);

   mfem::forall(n_mat, [=] MFEM_HOST_DEVICE (int i)
   {
      kernels::AddMult(m, k, n, &d_A(0,0,i), &d_x(0,0,i), &d_y(0,0,i),
                       alpha, beta);
   });

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
   MFEM_ABORT("");
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
