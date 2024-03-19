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


// Implementation of batchlinalg class

#include "batchlinalg.hpp"
#include "../general/forall.hpp"

#if defined(MFEM_USE_UMPIRE)
#define MFEM_RELEASE_MEM(o) o.GetMemory().DeleteDevice(false)
#else
#define MFEM_RELEASE_MEM(o)
#endif

#if defined(MFEM_USE_CUDA)
#include <cublas_v2.h>
#include <cusolverDn.h>
#define MFEM_cu_or_hip(stub) cu##stub
#define MFEM_Cu_or_Hip(stub) Cu##stub
#define MFEM_CU_or_HIP(stub) CU##stub
#define MFEM_CUDA_or_HIP(stub) CUDA##stub
#elif defined(MFEM_USE_HIP)
#include <hipblas/hipblas.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#define MFEM_cu_or_hip(stub) hip##stub
#define MFEM_Cu_or_Hip(stub) Hip##stub
#define MFEM_CU_or_HIP(stub) HIP##stub
#define MFEM_CUDA_or_HIP(stub) HIP##stub
#else
#define MFEM_cu_or_hip(stub)
#define MFEM_Cu_or_Hip(stub)
#define MFEM_CU_or_HIP(stub)
#define MFEM_CUDA_or_HIP(stub)
#endif

namespace mfem
{

#if defined(MFEM_USE_CUDA_OR_HIP)
const MFEM_cu_or_hip(blasHandle_t) & DeviceBlasHandle()
{
   static MFEM_cu_or_hip(blasHandle_t) handle = nullptr;
   static bool init                               = true;

   if (init)
   {
      init        = false;
      auto status = MFEM_cu_or_hip(blasCreate)(&handle);
      MFEM_VERIFY(status == MFEM_CU_or_HIP(BLAS_STATUS_SUCCESS),
                  "Cannot initialize GPU BLAS");
      atexit([]()
      {
         MFEM_cu_or_hip(blasDestroy)(handle);
         handle = nullptr;
      });
   }
   return handle;
}
#endif


LibBatchSolver::LibBatchSolver(const DenseTensor &MatrixBatch,
                               const SolveMode mode)
   : mode_(mode)
     // TODO: should this really be a copy?
   , LUMatrixBatch_(MatrixBatch)
{
   if (!setup_)
   {
      Setup();
   }
}

LibBatchSolver::LibBatchSolver(const SolveMode mode) : mode_(mode) {}

void LibBatchSolver::AssignMatrices(const mfem::DenseTensor &MatrixBatch)
{
   // TODO: should this really be a copy?
   LUMatrixBatch_ = MatrixBatch;
   setup_         = false;  //Setup is now false as the matrices have changed
   lu_valid_      = false;

   //Need to always call setup since the matrices have changed
   if (!setup_)
   {
      Setup();
   }
}

void LibBatchSolver::AssignMatrices(const mfem::Vector &vMatrixBatch,
                                    const int ndofs,
                                    const int num_matrices)
{
   const int totalSize = ndofs * ndofs * num_matrices;
   LUMatrixBatch_.SetSize(ndofs, ndofs,
                          num_matrices); //needs to be temporary memory
   double *d_LUMatrixBatch      = LUMatrixBatch_.Write();
   const double *d_vMatrixBatch = vMatrixBatch.Read();

   mfem::forall(totalSize, [=] MFEM_HOST_DEVICE (int i) { d_LUMatrixBatch[i] = d_vMatrixBatch[i]; });

   AssignMatrices(LUMatrixBatch_);
}

void LibBatchSolver::GetInverse(mfem::DenseTensor &InvMatBatch) const
{

   /*
    if (mode_ == SolveMode::INVERSE)
    {
       MFEM_WARNING("GetInverse with SolveMode::Inverse involves and extra memory copy, consider "
                    "GetInverse(M, M_inv) instead");
    }
   */

   if (!setup_)
   {
      mfem_error("LibBatchSolver has not been setup");
   }

   // use existing inverse
   if (mode_ == SolveMode::INVERSE)
   {
      MFEM_VERIFY(InvMatrixBatch_.TotalSize() == InvMatBatch.TotalSize(),
                  "Internal error, InvMatrixBatch_.TotalSize() != InvMatBatch.TotalSize()");

      const double *d_M_inv = InvMatrixBatch_.Read();
      double *d_out         = InvMatBatch.Write();

      mfem::forall(InvMatrixBatch_.TotalSize(), [=] MFEM_HOST_DEVICE (int i) { d_out[i] = d_M_inv[i]; });
   }
   else if (mode_ == SolveMode::LU)
   {
      return ComputeInverse(InvMatBatch);
   }
   else
   {
      mfem_error("unsupported mode");
   }
}

void LibBatchSolver::ComputeLU()
{
   if (lu_valid_)
   {
      return;
   }

#if defined(MFEM_USE_CUDA_OR_HIP)
   if (Device::Allows(Backend::DEVICE_MASK))
   {

      mfem::Array<int> info_array(num_matrices_); // need to move to temp mem

      MFEM_cu_or_hip(blasStatus_t)
      status = MFEM_cu_or_hip(blasDgetrfBatched)(DeviceBlasHandle(),
                                                 matrix_size_,
                                                 lu_ptr_array_.ReadWrite(),
                                                 matrix_size_,
                                                 P_.Write(),
                                                 info_array.Write(),
                                                 num_matrices_);

      MFEM_VERIFY(status == MFEM_CU_or_HIP(BLAS_STATUS_SUCCESS),
                  "Failed at blasDgetrfBatched");

   }
   else
#endif
   {
      //Hand written version
      BatchLUFactor(LUMatrixBatch_, P_);
   }
   lu_valid_ = true;
}


void LibBatchSolver::ComputeInverse(mfem::DenseTensor &InvMatBatch) const
{

   MFEM_VERIFY(lu_valid_, "LU must be valid");

#if defined(MFEM_USE_CUDA_OR_HIP)
   if (Device::Allows(Backend::DEVICE_MASK))
   {

      //Must be moved to temporary memory
      Array<double *> inv_ptr_array(num_matrices_);

      double *inv_ptr_base = InvMatBatch.Write();
      double **d_inv_ptr_array = inv_ptr_array.Write();
      const int matrix_size    = matrix_size_;
      mfem::forall(num_matrices_, [=] MFEM_HOST_DEVICE (int i)
      {
         d_inv_ptr_array[i] = inv_ptr_base + i * matrix_size * matrix_size;
      });

      //move to temporary memory
      mfem::Array<int> info_array(num_matrices_);

      //Invert matrices
      MFEM_cu_or_hip(blasStatus_t) status =
         MFEM_cu_or_hip(blasDgetriBatched)(DeviceBlasHandle(),
                                           matrix_size_,
                                           lu_ptr_array_.Read(),
                                           matrix_size_,
                                           // from hipblas.h: @param[in] ipiv
                                           // we can const_cast safely because it's an "in" variable
                                           const_cast<int *>(P_.Read()),
                                           inv_ptr_array.ReadWrite(),
                                           matrix_size_,
                                           info_array.Write(),
                                           num_matrices_);

      MFEM_VERIFY(status == MFEM_CU_or_HIP(BLAS_STATUS_SUCCESS),
                  "Failed at blasDgetriBatched");
   }
   else
#endif
   {
      BatchInverseMatrix(LUMatrixBatch_, P_, InvMatBatch);
   }
}

void LibBatchSolver::SolveLU(const mfem::Vector &b, mfem::Vector &x) const
{

   x = b;
#if defined(MFEM_USE_CUDA)
   if (Device::Allows(Backend::DEVICE_MASK))
   {
      mfem::Array<double *> vector_array(num_matrices_); //need to move to TEMP memory

      // TODO: can this be Write? does `blasDtrsmBatched' just overwrite what's in x?
      double *x_ptr_base = x.ReadWrite();

      double alpha = 1.0;

      double **d_vector_array = vector_array.Write();
      const int matrix_size   = matrix_size_;
      mfem::forall(num_matrices_, [=] MFEM_HOST_DEVICE (int i) { d_vector_array[i] = x_ptr_base + i * matrix_size; });


      MFEM_cu_or_hip(blasStatus_t)
      status_lo = MFEM_cu_or_hip(blasDtrsmBatched)(DeviceBlasHandle(),
                                                   MFEM_CU_or_HIP(BLAS_SIDE_LEFT),
                                                   MFEM_CU_or_HIP(BLAS_FILL_MODE_LOWER),
                                                   MFEM_CU_or_HIP(BLAS_OP_N),
                                                   MFEM_CU_or_HIP(BLAS_DIAG_UNIT),
                                                   matrix_size_,
                                                   1,
                                                   &alpha,
                                                   lu_ptr_array_.Read(),
                                                   matrix_size_,
                                                   vector_array.ReadWrite(),
                                                   matrix_size_,
                                                   num_matrices_);

      MFEM_VERIFY(status_lo == MFEM_CU_or_HIP(BLAS_STATUS_SUCCESS),
                  "Failed at blasDtrsmBatched lo");

      MFEM_cu_or_hip(blasStatus_t)
      status_upp = MFEM_cu_or_hip(blasDtrsmBatched)(DeviceBlasHandle(),
                                                    MFEM_CU_or_HIP(BLAS_SIDE_LEFT),
                                                    MFEM_CU_or_HIP(BLAS_FILL_MODE_UPPER),
                                                    MFEM_CU_or_HIP(BLAS_OP_N),
                                                    MFEM_CU_or_HIP(BLAS_DIAG_NON_UNIT),
                                                    matrix_size_,
                                                    1,
                                                    &alpha,
                                                    lu_ptr_array_.Read(),
                                                    matrix_size_,
                                                    vector_array.ReadWrite(),
                                                    matrix_size_,
                                                    num_matrices_);

      MFEM_VERIFY(status_upp == MFEM_CU_or_HIP(BLAS_STATUS_SUCCESS),
                  "Failed at blasDtrsmBatched upper");
   }
   else
#endif
   {
      BatchLUSolve(LUMatrixBatch_, P_, x);
   }

}

//hand rolled block mult
void ApplyBlkMult(const mfem::DenseTensor &Mat, const mfem::Vector &x,
                  mfem::Vector &y)
{
   const int ndof = Mat.SizeI();
   const int NE   = Mat.SizeK();
   auto X         = mfem::Reshape(x.Read(), ndof, NE);
   auto Y         = mfem::Reshape(y.Write(), ndof, NE);
   auto Me        = mfem::Reshape(Mat.Read(), ndof, ndof, NE);

   //Takes row major format
   mfem::forall(ndof* NE, [=] MFEM_HOST_DEVICE (int tid)
   {

      const int c = tid % ndof;
      const int e = tid / ndof;

      {
         double dot = 0;
         for (int r = 0; r < ndof; ++r)
         {
            dot += Me(r, c, e) * X(r, e);
         }
         Y(c, e) = dot;
      }
   });
}


void LibBatchSolver::ApplyInverse(const mfem::Vector &b, mfem::Vector &x) const
{
   //Extend with vendor library capabilities
   ApplyBlkMult(InvMatrixBatch_, b, x);
}


void LibBatchSolver::Setup()
{
   matrix_size_  = LUMatrixBatch_.SizeI();
   num_matrices_ = LUMatrixBatch_.SizeK();

   P_.SetSize(matrix_size_ * num_matrices_); //move to temp mem
   lu_ptr_array_.SetSize(num_matrices_); //move to temp memory

   // TODO: can this just be a Write?
   double *lu_ptr_base = LUMatrixBatch_.ReadWrite();

   const int matrix_size   = matrix_size_;
   double **d_lu_ptr_array = lu_ptr_array_.Write();

   mfem::forall(num_matrices_, [=] MFEM_HOST_DEVICE (int i)
   {
      d_lu_ptr_array[i] = lu_ptr_base + i * matrix_size * matrix_size;
   });

   switch (mode_)
   {
      case SolveMode::LU: ComputeLU(); break;

      case SolveMode::INVERSE:
         ComputeLU();
         InvMatrixBatch_.SetSize(matrix_size_,
                                 matrix_size_,
                                 num_matrices_); //move to temporary memory
         ComputeInverse(InvMatrixBatch_);
         break;

      default: mfem_error("Case not supported");
   }
   setup_ = true;
}

void LibBatchSolver::Mult(const mfem::Vector &b, mfem::Vector &x) const
{
   switch (mode_)
   {
      case SolveMode::LU: return SolveLU(b, x);

      case SolveMode::INVERSE: return ApplyInverse(b, x);

      default: mfem_error("Case not supported");
   }
}

void LibBatchSolver::ReleaseMemory()
{
   MFEM_RELEASE_MEM(LUMatrixBatch_);
   MFEM_RELEASE_MEM(InvMatrixBatch_);
   MFEM_RELEASE_MEM(P_);
   MFEM_RELEASE_MEM(lu_ptr_array_);
}


} // namespace mfem
