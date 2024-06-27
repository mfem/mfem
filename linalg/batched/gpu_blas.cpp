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

#include "gpu_blas.hpp"
#include "../../general/forall.hpp"

#ifdef MFEM_USE_CUDA_OR_HIP

#include <cublas_v2.h>

namespace mfem
{

GPUBlas::GPUBlas()
{
   cublasStatus_t status = cublasCreate(&handle);
   MFEM_VERIFY(status == CUBLAS_STATUS_SUCCESS, "Cannot initialize GPU BLAS.");
}

GPUBlas::~GPUBlas()
{
   cublasDestroy(handle);
}

GPUBlas &GPUBlas::Instance()
{
   static GPUBlas instance;
   return instance;
}

cublasHandle_t GPUBlas::Handle()
{
   return Instance().handle;
}

void GPUBlas::EnableAtomics()
{
   cublasStatus_t status = cublasSetAtomicsMode(Handle(),
                                                CUBLAS_ATOMICS_ALLOWED);
   MFEM_VERIFY(status == CUBLAS_STATUS_SUCCESS, "cuBLAS error.");
}

void GPUBlas::DisableAtomics()
{
   cublasStatus_t status = cublasSetAtomicsMode(Handle(),
                                                CUBLAS_ATOMICS_NOT_ALLOWED);
   MFEM_VERIFY(status == CUBLAS_STATUS_SUCCESS, "cuBLAS error.");
}

void GPUBlasBatchedLinAlg::Mult(
   const DenseTensor &A, const Vector &x, Vector &y) const
{
   MFEM_ABORT("");
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

   cublasStatus_t status = cublasDgetrfBatched(
                              GPUBlas::Handle(),
                              n,
                              d_A_ptrs,
                              n,
                              P.Write(),
                              info_array.Write(),
                              n_mat);

   MFEM_VERIFY(status == CUBLAS_STATUS_SUCCESS, "");
}

void GPUBlasBatchedLinAlg::LUSolve(
   const DenseTensor &LU, const Array<int> &P, Vector &x) const
{
   const int n = LU.SizeI();
   const int n_mat = LU.SizeK();
   const int n_rhs = x.Size() / n / n_mat;

   Array<const real_t*> A_ptrs(n_mat);
   const real_t **d_A_ptrs = A_ptrs.Write();
   Array<real_t*> B_ptrs(n_mat);
   real_t **d_B_ptrs = B_ptrs.Write();

   {
      const real_t *A_base = LU.Read();
      real_t *B_base = x.ReadWrite();
      mfem::forall(n_mat, [=] MFEM_HOST_DEVICE (int i)
      {
         d_A_ptrs[i] = A_base + i*n*n;
         d_B_ptrs[i] = B_base + i*n*n_rhs;
      });
   }

   int info = 0;

   cublasStatus_t status = cublasDgetrsBatched(GPUBlas::Handle(),
                                               CUBLAS_OP_N,
                                               n,
                                               n_rhs,
                                               d_A_ptrs,
                                               n,
                                               P.Read(),
                                               d_B_ptrs,
                                               n,
                                               &info,
                                               n_mat);
   MFEM_VERIFY(status == CUBLAS_STATUS_SUCCESS, "");
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
   cublasStatus_t status;

   status = cublasDgetrfBatched(GPUBlas::Handle(),
                                n,
                                d_LU_ptrs,
                                n,
                                P.Write(),
                                info_array.Write(),
                                n_mat);
   MFEM_VERIFY(status == CUBLAS_STATUS_SUCCESS, "");

   status = cublasDgetriBatched(GPUBlas::Handle(),
                                n,
                                d_LU_ptrs,
                                n,
                                P.ReadWrite(),
                                d_A_ptrs,
                                n,
                                info_array.Write(),
                                n_mat);
   MFEM_VERIFY(status == CUBLAS_STATUS_SUCCESS, "");
}

} // namespace mfem

#endif
