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

   const double alpha = 1.0, beta = 0.0;
   MFEM_SUB_cu_or_hip(blasStatus_t)
   status = MFEM_SUB_cu_or_hip(blasDgemmStridedBatched)(MFEM_SUB_Cuda_or_Hip(
                                                           BLAS::Handle)(),
                                                        MFEM_SUB_CU_or_HIP(BLAS_OP_N),
                                                        MFEM_SUB_CU_or_HIP(BLAS_OP_N),
                                                        mat_size,
                                                        1,
                                                        mat_size,
                                                        &alpha,
                                                        MatrixBatch.Read(),
                                                        mat_size,
                                                        mat_size * mat_size,
                                                        b.Read(),
                                                        mat_size,
                                                        mat_size,
                                                        &beta,
                                                        x.Write(),
                                                        mat_size,
                                                        mat_size,
                                                        num_mats);

   MFEM_VERIFY(status == MFEM_SUB_CU_or_HIP(BLAS_STATUS_SUCCESS),
               "blasDgemmStridedBatched");
}


} //mfem namespace
