// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "libBatchSolver.hpp"

namespace mfem
{

void LibBatchMult::Mult(const Vector &b, Vector &x)
{
#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
   if (Device::Allows(Backend::DEVICE_MASK))
   {
#if defined(MFEM_USE_MAGMA)
      const double alpha = 1.0, beta = 0.0;
#else
      const double alpha_t = 1.0, beta_t = 0.0;
      const double* alpha = &alpha_t;
      const double* beta = &beta_t;
#endif
     
#if defined(MFEM_USE_HIP) || (defined(MFEM_USE_CUDA) && CUDA_VERSION >= 11070) || defined(MFEM_USE_MAGMA)
#if defined(MFEM_USE_MAGMA)
      magmablas_dgemv_batched_strided(MagmaNoTrans,
#else
      MFEM_SUB_cu_or_hip(blasStatus_t)
      status = MFEM_SUB_cu_or_hip(blasDgemvStridedBatched)(MFEM_SUB_Cuda_or_Hip(
                                                           BLAS::Handle)(),
                                                           MFEM_SUB_CU_or_HIP(BLAS_OP_N),
#endif
                                                           mat_size,
                                                           mat_size,
                                                           alpha,
                                                           MatrixBatch.Read(),
                                                           mat_size,
                                                           mat_size * mat_size,
                                                           b.Read(),
                                                           1,
                                                           mat_size,
                                                           beta,
                                                           x.Write(),
                                                           1,
                                                           mat_size,
                                                           num_mats
#if defined(MFEM_USE_MAGMA)
                                                           , queue
#endif
							   );
#else
      MFEM_SUB_cu_or_hip(blasStatus_t)
      status = MFEM_SUB_cu_or_hip(blasDgemmStridedBatched)(MFEM_SUB_Cuda_or_Hip(
                                                           BLAS::Handle)(),
                                                           MFEM_SUB_CU_or_HIP(BLAS_OP_N),
                                                           MFEM_SUB_CU_or_HIP(BLAS_OP_N),
                                                           mat_size,
                                                           1,
                                                           mat_size,
                                                           alpha,
                                                           MatrixBatch.Read(),
                                                           mat_size,
                                                           mat_size * mat_size,
                                                           b.Read(),
                                                           mat_size,
                                                           mat_size,
                                                           beta,
                                                           x.Write(),
                                                           mat_size,
                                                           mat_size,
                                                           num_mats);


#endif

#if !defined(MFEM_USE_MAGMA)
      MFEM_VERIFY(status == MFEM_SUB_CU_or_HIP(BLAS_STATUS_SUCCESS),
                  "blasDgemvStridedBatched");
#endif
   }
   else
#endif
   {
      const int rows = MatrixBatch.SizeI();
      const int cols = MatrixBatch.SizeJ();
      const int N = MatrixBatch.SizeK();

      for (int e=0; e<N; ++e)
      {
         //matvec
         for (int c=0; c<cols; ++c)
         {
            double dot=0.0;
            for (int r=0; r<rows; ++r)
            {
               int idx = r  +  rows * e;
               dot += MatrixBatch(c, r, e) * b(idx);
            }
            int idx = c + cols * e;
            x(idx) = dot;
         }
      }
   }
}
} //mfem namespace
