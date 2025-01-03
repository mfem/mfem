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

#include "magma.hpp"
#include "../lapack.hpp"
#include "../../general/forall.hpp"

#ifdef MFEM_USE_MAGMA

#ifdef MFEM_USE_SINGLE
#define MFEM_MAGMA_PREFIX(stub) magma_s ## stub
#define MFEM_MAGMABLAS_PREFIX(stub) magmablas_s ## stub
#elif defined(MFEM_USE_DOUBLE)
#define MFEM_MAGMA_PREFIX(stub) magma_d ## stub
#define MFEM_MAGMABLAS_PREFIX(stub) magmablas_d ## stub
#endif

namespace mfem
{

Magma::Magma()
{
   const magma_int_t status = magma_init();
   MFEM_VERIFY(status == MAGMA_SUCCESS, "Error initializing MAGMA.");
   magma_device_t dev;
   magma_getdevice(&dev);
   magma_queue_create(dev, &queue);
}

Magma::~Magma()
{
   magma_queue_destroy(queue);
   const magma_int_t status = magma_finalize();
   MFEM_VERIFY(status == MAGMA_SUCCESS, "Error finalizing MAGMA.");
}

Magma &Magma::Instance()
{
   static Magma magma;
   return magma;
}

magma_queue_t Magma::Queue()
{
   return Instance().queue;
}

void MagmaBatchedLinAlg::AddMult(const DenseTensor &A, const Vector &x,
                                 Vector &y, real_t alpha, real_t beta,
                                 Op op) const
{
   const bool tr = (op == Op::T);

   const int m = tr ? A.SizeJ() : A.SizeI();
   const int n = tr ? A.SizeI() : A.SizeJ();
   const int n_mat = A.SizeK();
   const int k = x.Size() / n / n_mat;

   auto d_A = A.Read();
   auto d_x = x.Read(); // Shape (n, k, n_mat);
   auto d_y = beta == 0.0 ? y.Write() : y.ReadWrite(); // Shape (m, k, n_mat);

   magma_trans_t magma_op = tr ? MagmaNoTrans : MagmaTrans;

   MFEM_MAGMABLAS_PREFIX(gemm_batched_strided)(
      magma_op, MagmaNoTrans, m, k, n, alpha, d_A, m, m*n, d_x, n, n*k,
      beta, d_y, m, m*k, n_mat, Magma::Queue());
}

void MagmaBatchedLinAlg::LUFactor(DenseTensor &A, Array<int> &P) const
{
   const int n = A.SizeI();
   const int n_mat = A.SizeK();

   P.SetSize(n*n_mat);

   real_t *A_base = A.ReadWrite();
   int *P_base = P.ReadWrite();

   Array<real_t*> A_ptrs(n_mat);
   Array<int*> P_ptrs(n_mat);
   real_t **d_A_ptrs = A_ptrs.Write();
   int **d_P_ptrs = P_ptrs.Write();
   mfem::forall(n_mat, [=] MFEM_HOST_DEVICE (int i)
   {
      d_A_ptrs[i] = A_base + i*n*n;
      d_P_ptrs[i] = P_base + i*n;
   });

   Array<int> info_array(n_mat);
   const magma_int_t status = MFEM_MAGMA_PREFIX(getrf_batched)(
                                 n, n, d_A_ptrs, n, d_P_ptrs,
                                 info_array.Write(), n_mat, Magma::Queue());
   MFEM_VERIFY(status == MAGMA_SUCCESS, "");
}

void MagmaBatchedLinAlg::LUSolve(
   const DenseTensor &LU, const Array<int> &P, Vector &x) const
{
   const int n = LU.SizeI();
   const int n_mat = LU.SizeK();
   const int n_rhs = x.Size() / n / n_mat;

   Array<real_t*> A_ptrs(n_mat);
   Array<real_t*> B_ptrs(n_mat);
   Array<int*> P_ptrs(n_mat);
   real_t **d_A_ptrs = A_ptrs.Write();
   real_t **d_B_ptrs = B_ptrs.Write();
   int **d_P_ptrs = P_ptrs.Write();

   {
      real_t *A_base = const_cast<real_t*>(LU.Read());
      real_t *B_base = x.ReadWrite();
      int *P_base = const_cast<int*>(P.Read());
      mfem::forall(n_mat, [=] MFEM_HOST_DEVICE (int i)
      {
         d_A_ptrs[i] = A_base + i*n*n;
         d_B_ptrs[i] = B_base + i*n*n_rhs;
         d_P_ptrs[i] = P_base + i*n;
      });
   }

   const magma_int_t status = MFEM_MAGMA_PREFIX(getrs_batched)(
                                 MagmaNoTrans, n, n_rhs, d_A_ptrs, n, d_P_ptrs,
                                 d_B_ptrs, n, n_mat, Magma::Queue());
   MFEM_VERIFY(status == MAGMA_SUCCESS, "");
}

void MagmaBatchedLinAlg::Invert(DenseTensor &A) const
{
   const int n = A.SizeI();
   const int n_mat = A.SizeK();

   DenseTensor LU(A.SizeI(), A.SizeJ(), A.SizeK());
   LU.Write();
   LU.GetMemory().CopyFrom(A.GetMemory(), A.TotalSize());

   Array<int> P(n*n_mat);

   Array<real_t*> LU_ptrs(n_mat);
   Array<real_t*> A_ptrs(n_mat);
   Array<int*> P_ptrs(n_mat);
   real_t **d_A_ptrs = A_ptrs.Write();
   real_t **d_LU_ptrs = LU_ptrs.Write();
   int **d_P_ptrs = P_ptrs.Write();
   {
      real_t *A_base = A.ReadWrite();
      real_t *LU_base = LU.Write();
      int *P_base = P.Write();
      mfem::forall(n_mat, [=] MFEM_HOST_DEVICE (int i)
      {
         d_A_ptrs[i] = A_base + i*n*n;
         d_LU_ptrs[i] = LU_base + i*n*n;
         d_P_ptrs[i] = P_base + i*n;
      });
   }

   Array<int> info_array(n_mat);
   magma_int_t status;

   status = MFEM_MAGMA_PREFIX(getrf_batched)(
               n, n, d_A_ptrs, n, d_P_ptrs, info_array.Write(), n_mat,
               Magma::Queue());
   MFEM_VERIFY(status == MAGMA_SUCCESS, "");

   status = MFEM_MAGMA_PREFIX(getri_outofplace_batched)(
               n, d_LU_ptrs, n, d_P_ptrs, d_A_ptrs, n, info_array.Write(),
               n_mat, Magma::Queue());
   MFEM_VERIFY(status == MAGMA_SUCCESS, "");
}

} // namespace mfem

#endif
