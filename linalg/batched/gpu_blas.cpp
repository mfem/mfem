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

#include "gpu_blas.hpp"
#include "../../general/forall.hpp"

#if defined(MFEM_USE_CUDA)
#define MFEM_cu_or_hip(stub) cu##stub
#define MFEM_CU_or_HIP(stub) CU##stub
#elif defined(MFEM_USE_HIP)
#define MFEM_cu_or_hip(stub) hip##stub
#define MFEM_CU_or_HIP(stub) HIP##stub
#endif

#define MFEM_CONCAT(x, y, z) MFEM_CONCAT_(x, y, z)
#define MFEM_CONCAT_(x, y, z) x ## y ## z

#ifdef MFEM_USE_SINGLE
#define MFEM_GPUBLAS_PREFIX(stub) MFEM_CONCAT(MFEM_cu_or_hip(blas), S, stub)
#elif defined(MFEM_USE_DOUBLE)
#define MFEM_GPUBLAS_PREFIX(stub) MFEM_CONCAT(MFEM_cu_or_hip(blas), D, stub)
#endif

#define MFEM_BLAS_SUCCESS MFEM_CU_or_HIP(BLAS_STATUS_SUCCESS)

namespace mfem
{

GPUBlas &GPUBlas::Instance()
{
   static GPUBlas instance;
   return instance;
}

GPUBlas::HandleType GPUBlas::Handle()
{
   return Instance().handle;
}

#ifndef MFEM_USE_CUDA_OR_HIP

GPUBlas::GPUBlas() { }
GPUBlas::~GPUBlas() { }
void GPUBlas::EnableAtomics() { }
void GPUBlas::DisableAtomics() { }

#else

using blasStatus_t = MFEM_cu_or_hip(blasStatus_t);

GPUBlas::GPUBlas()
{
   blasStatus_t status = MFEM_cu_or_hip(blasCreate)(&handle);
   MFEM_VERIFY(status == MFEM_BLAS_SUCCESS, "Cannot initialize GPU BLAS.");
}

GPUBlas::~GPUBlas()
{
   MFEM_cu_or_hip(blasDestroy)(handle);
}

void GPUBlas::EnableAtomics()
{
   const blasStatus_t status = MFEM_cu_or_hip(blasSetAtomicsMode)(
                                  Handle(), MFEM_CU_or_HIP(BLAS_ATOMICS_ALLOWED));
   MFEM_VERIFY(status == MFEM_BLAS_SUCCESS, "GPU BLAS error.");
}

void GPUBlas::DisableAtomics()
{
   const blasStatus_t status = MFEM_cu_or_hip(blasSetAtomicsMode)(
                                  Handle(), MFEM_CU_or_HIP(BLAS_ATOMICS_NOT_ALLOWED));
   MFEM_VERIFY(status == MFEM_BLAS_SUCCESS, "GPU BLAS error.");
}

void GPUBlasBatchedLinAlg::AddMult(const DenseTensor &A, const Vector &x,
                                   Vector &y, real_t alpha, real_t beta,
                                   Op op) const
{
   const bool tr = (op == Op::T);

   const int m = tr ? A.SizeJ() : A.SizeI();
   const int n = tr ? A.SizeI() : A.SizeJ();
   const int n_mat = A.SizeK();
   const int k = x.Size() / n / n_mat;

   auto d_A = A.Read();
   auto d_x = x.Read(); // Shape: (n, k, n_mat)
   auto d_y = beta == 0.0 ? y.Write() : y.ReadWrite(); // Shape (m, k, n_mat)

   const auto op_A = tr ? MFEM_CU_or_HIP(BLAS_OP_T) : MFEM_CU_or_HIP(BLAS_OP_N);
   const auto op_B = MFEM_CU_or_HIP(BLAS_OP_N);

   const blasStatus_t status = MFEM_GPUBLAS_PREFIX(gemmStridedBatched)(
                                  GPUBlas::Handle(), op_A, op_B, m, k, n,
                                  &alpha, d_A, m, m*n, d_x, n, n*k, &beta, d_y,
                                  m, m*k, n_mat);
   MFEM_VERIFY(status == MFEM_BLAS_SUCCESS, "GPU BLAS error.");
}

void GPUBlasBatchedLinAlg::LUFactor(DenseTensor &A, Array<int> &P) const
{
   const int n = A.SizeI();
   const int n_mat = A.SizeK();

   P.SetSize(n*n_mat);

   Array<int> info_array(n_mat);

   real_t *A_base = A.ReadWrite();
   Array<real_t*> A_ptrs(n_mat);
   real_t **d_A_ptrs = A_ptrs.Write();
   mfem::forall(n_mat, [=] MFEM_HOST_DEVICE (int i)
   {
      d_A_ptrs[i] = A_base + i*n*n;
   });

   const blasStatus_t status = MFEM_GPUBLAS_PREFIX(getrfBatched)(
                                  GPUBlas::Handle(), n, d_A_ptrs, n, P.Write(),
                                  info_array.Write(), n_mat);
   MFEM_VERIFY(status == MFEM_BLAS_SUCCESS, "");
}

void GPUBlasBatchedLinAlg::LUSolve(
   const DenseTensor &LU, const Array<int> &P, Vector &x) const
{
   const int n = LU.SizeI();
   const int n_mat = LU.SizeK();
   const int n_rhs = x.Size() / n / n_mat;

   Array<real_t*> A_ptrs(n_mat);
   real_t **d_A_ptrs = A_ptrs.Write();
   Array<real_t*> B_ptrs(n_mat);
   real_t **d_B_ptrs = B_ptrs.Write();

   {
      real_t *A_base = const_cast<real_t*>(LU.Read());
      real_t *B_base = x.ReadWrite();
      mfem::forall(n_mat, [=] MFEM_HOST_DEVICE (int i)
      {
         d_A_ptrs[i] = A_base + i*n*n;
         d_B_ptrs[i] = B_base + i*n*n_rhs;
      });
   }

   int info = 0;
   const blasStatus_t status = MFEM_GPUBLAS_PREFIX(getrsBatched)(
                                  GPUBlas::Handle(), MFEM_CU_or_HIP(BLAS_OP_N),
                                  n, n_rhs, d_A_ptrs, n, P.Read(), d_B_ptrs, n,
                                  &info, n_mat);
   MFEM_VERIFY(status == MFEM_BLAS_SUCCESS, "");
}

void GPUBlasBatchedLinAlg::Invert(DenseTensor &A) const
{
   const int n = A.SizeI();
   const int n_mat = A.SizeK();

   DenseTensor LU(A.SizeI(), A.SizeJ(), A.SizeK());
   LU.Write();
   LU.GetMemory().CopyFrom(A.GetMemory(), A.TotalSize());

   Array<real_t*> LU_ptrs(n_mat);
   Array<real_t*> A_ptrs(n_mat);
   real_t **d_A_ptrs = A_ptrs.Write();
   real_t **d_LU_ptrs = LU_ptrs.Write();
   {
      real_t *A_base = A.ReadWrite();
      real_t *LU_base = LU.Write();
      mfem::forall(n_mat, [=] MFEM_HOST_DEVICE (int i)
      {
         d_A_ptrs[i] = A_base + i*n*n;
         d_LU_ptrs[i] = LU_base + i*n*n;
      });
   }

   Array<int> P(n*n_mat);
   Array<int> info_array(n_mat);
   blasStatus_t status;

   status = MFEM_GPUBLAS_PREFIX(getrfBatched)(
               GPUBlas::Handle(), n, d_LU_ptrs, n, P.Write(),
               info_array.Write(), n_mat);
   MFEM_VERIFY(status == MFEM_BLAS_SUCCESS, "");

   status = MFEM_GPUBLAS_PREFIX(getriBatched)(
               GPUBlas::Handle(), n, d_LU_ptrs, n, P.ReadWrite(), d_A_ptrs, n,
               info_array.Write(), n_mat);
   MFEM_VERIFY(status == MFEM_BLAS_SUCCESS, "");
}

#endif

} // namespace mfem
