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
#include "../general/backends.hpp"

namespace mfem
{

#if defined(MFEM_USE_CUDA_OR_HIP)
static MFEM_cu_or_hip(blasHandle_t) device_blas_handle = nullptr;

const MFEM_cu_or_hip(blasHandle_t) & DeviceBlasHandle()
{
   if (!device_blas_handle)
   {
      auto status = MFEM_cu_or_hip(blasCreate)(&device_blas_handle);
      MFEM_VERIFY(status == MFEM_CU_or_HIP(BLAS_STATUS_SUCCESS),
                  "Cannot initialize GPU BLAS");
      atexit([]()
      {
         MFEM_cu_or_hip(blasDestroy)(device_blas_handle);
         device_blas_handle = nullptr;
      });
   }
   return device_blas_handle;
}
#endif


BatchSolver::BatchSolver(const DenseTensor &MatrixBatch,
                         const SolveMode mode, MemoryType d_mt)
   : mode_(mode)
     // TODO: should this really be a copy?
   , LUMatrixBatch_(MatrixBatch)
   , d_mt_(d_mt)
{
   if (!setup_)
   {
      Setup();
   }
}

BatchSolver::BatchSolver(const SolveMode mode, MemoryType d_mt) : mode_(mode),
   d_mt_(d_mt) {}

void BatchSolver::AssignMatrices(const DenseTensor &MatrixBatch)
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

void BatchSolver::AssignMatrices(const Vector &vMatrixBatch,
                                 const int size,
                                 const int num_matrices)
{
   const int totalSize = size * size * num_matrices;
   LUMatrixBatch_.SetSize(size, size,
                          num_matrices, d_mt_);
   double *d_LUMatrixBatch      = LUMatrixBatch_.Write();
   const double *d_vMatrixBatch = vMatrixBatch.Read();

   mfem::forall(totalSize, [=] MFEM_HOST_DEVICE (int i) { d_LUMatrixBatch[i] = d_vMatrixBatch[i]; });

   AssignMatrices(LUMatrixBatch_);
}

void BatchSolver::GetInverse(DenseTensor &InvMatBatch) const
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
      mfem_error("BatchSolver has not been setup");
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

void BatchSolver::ComputeLU()
{
   if (lu_valid_)
   {
      return;
   }

#if defined(MFEM_USE_CUDA_OR_HIP)
   if (Device::Allows(Backend::DEVICE_MASK))
   {

      Array<int> info_array(num_matrices_); // need to move to temp mem

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


void BatchSolver::ComputeInverse(DenseTensor &InvMatBatch) const
{

   MFEM_VERIFY(lu_valid_, "LU must be valid");

#if defined(MFEM_USE_CUDA_OR_HIP)
   if (Device::Allows(Backend::DEVICE_MASK))
   {

      Array<double *> inv_ptr_array(num_matrices_, d_mt_);

      double *inv_ptr_base = InvMatBatch.Write();
      double **d_inv_ptr_array = inv_ptr_array.Write();
      const int matrix_size    = matrix_size_;
      mfem::forall(num_matrices_, [=] MFEM_HOST_DEVICE (int i)
      {
         d_inv_ptr_array[i] = inv_ptr_base + i * matrix_size * matrix_size;
      });

      Array<int> info_array(num_matrices_, d_mt_);

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

void BatchSolver::SolveLU(const Vector &b, Vector &x) const
{

   x = b;
#if defined(MFEM_USE_CUDA_OR_HIP)
   if (Device::Allows(Backend::DEVICE_MASK))
   {
      Array<double *> vector_array(num_matrices_, d_mt_);

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

//Hand rolled -- TODO replace with vendor call
void ApplyBlkMult(const DenseTensor &Mat, const Vector &x,
                  Vector &y)
{
   const int ndof = Mat.SizeI();
   MFEM_VERIFY(Mat.SizeI() == Mat.SizeJ(), "Batcched matrices are not square: not invertible");
   const int NE = Mat.SizeK();
   
#if defined(MFEM_USE_CUDA_OR_HIP)
   if (Device::Allows(Backend::DEVICE_MASK))
   {
      Array<double *> Mat_ptr(NE); 
      Array<double *> x_ptr(NE);
      Array<double *> y_ptr(NE);
      for (int k = 0; k < NE; k++) {
         Mat_ptr[k] = &const_cast<DenseTensor&>(Mat).ReadWrite()[ndof*ndof*k];
         x_ptr[k] = &const_cast<Vector&>(x).ReadWrite()[ndof*k];
         y_ptr[k] = &y.ReadWrite()[ndof*k];
      }

      double alpha = 1.0;
      double beta = 0.0; 
      MFEM_cu_or_hip(blasStatus_t)
      status = MFEM_cu_or_hip(blasDgemvBatched)(DeviceBlasHandle(), 
                                                MFEM_CU_or_HIP(BLAS_OP_N), 
                                                ndof, ndof,
                                                &alpha, 
                                                Mat_ptr.Read(), ndof, 
                                                x_ptr.Read(), 1,
                                                &beta, 
                                                y_ptr.ReadWrite(), 1, 
                                                NE);
      //free variables? 
      MFEM_VERIFY(status == MFEM_CU_or_HIP(BLAS_STATUS_SUCCESS), 
                  "Failed at blasDgemvBatched");
   }
   else
#endif
   {
      // Hand written version
      auto X         = Reshape(x.Read(), ndof, NE);
      auto Y         = Reshape(y.Write(), ndof, NE);
      auto Me        = Reshape(Mat.Read(), ndof, ndof, NE);

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
}


void BatchSolver::ApplyInverse(const Vector &b, Vector &x) const
{
   //Extend with vendor library capabilities
   ApplyBlkMult(InvMatrixBatch_, b, x);
}


void BatchSolver::Setup()
{
   matrix_size_  = LUMatrixBatch_.SizeI();
   num_matrices_ = LUMatrixBatch_.SizeK();

   P_.SetSize(matrix_size_ * num_matrices_, d_mt_);
   lu_ptr_array_.SetSize(num_matrices_, d_mt_);

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
                                 num_matrices_, d_mt_);
         ComputeInverse(InvMatrixBatch_);
         break;

      default: mfem_error("Case not supported");
   }
   setup_ = true;
}

void BatchSolver::Mult(const Vector &b, Vector &x) const
{
   switch (mode_)
   {
      case SolveMode::LU: return SolveLU(b, x);

      case SolveMode::INVERSE: return ApplyInverse(b, x);

      default: mfem_error("Case not supported");
   }
}

void BatchSolver::ReleaseMemory()
{
   LUMatrixBatch_.GetMemory().ReleaseDeviceMemory(false);
   InvMatrixBatch_.GetMemory().ReleaseDeviceMemory(false);
   P_.GetMemory().ReleaseDeviceMemory(false);
   lu_ptr_array_.GetMemory().ReleaseDeviceMemory(false);
}


} // namespace mfem
