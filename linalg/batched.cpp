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

#include "batched.hpp"
#include "kernels.hpp"
#include "dtensor.hpp"
#include "../general/forall.hpp"

#ifdef MFEM_USE_CUDA
#include <cublas.h>
#include <cusolverDn.h>
#endif

namespace mfem
{

#ifdef MFEM_USE_CUDA

class CuSolver
{
protected:
   cusolverDnHandle_t handle = nullptr;
   CuSolver()
   {
      cusolverStatus_t status = cusolverDnCreate(&handle);
      MFEM_VERIFY(status == CUSOLVER_STATUS_SUCCESS,
                  "Cannot initialize CuSolver.");
   }
   ~CuSolver()
   {
      cusolverDnDestroy(handle);
   }
   static CuSolver &Instance()
   {
      static CuSolver instance;
      return instance;
   }
public:
   static cusolverDnHandle_t Handle()
   {
      return Instance().handle;
   }
};

class CuBLAS
{
protected:
   cublasHandle_t handle = nullptr;
   CuBLAS()
   {
      cublasStatus_t status = cublasCreate(&handle);
      MFEM_VERIFY(status == CUBLAS_STATUS_SUCCESS, "Cannot initialize cuBLAS.");
   }
   ~CuBLAS()
   {
      cublasDestroy(handle);
   }
   static CuBLAS &Instance()
   {
      static CuBLAS instance;
      return instance;
   }
public:
   static cublasHandle_t Handle()
   {
      return Instance().handle;
   }

   static void EnableAtomics()
   {
      cublasStatus_t status = cublasSetAtomicsMode(Handle(),
                                                   CUBLAS_ATOMICS_ALLOWED);
      MFEM_VERIFY(status == CUBLAS_STATUS_SUCCESS, "cuBLAS error.");
   }

   static void DisableAtomics()
   {
      cublasStatus_t status = cublasSetAtomicsMode(Handle(),
                                                   CUBLAS_ATOMICS_NOT_ALLOWED);
      MFEM_VERIFY(status == CUBLAS_STATUS_SUCCESS, "cuBLAS error.");
   }
};

#endif

void BatchLUFactor_CuBLAS(DenseTensor &A, Array<int> &P, const real_t tol)
{
#ifdef MFEM_USE_CUDA
   const int n = A.SizeI();
   const int n_mats = A.SizeK();

   P.SetSize(n*n_mats);

   Array<int> info_array(n_mats);

   real_t *A_base = A.ReadWrite();
   Array<real_t*> A_ptrs(n_mats);
   real_t **d_A_ptrs = A_ptrs.Write();
   mfem::forall(n_mats, [=] MFEM_HOST_DEVICE (int i)
   {
      d_A_ptrs[i] = A_base + i*n*n;
   });

   cublasStatus_t status = cublasDgetrfBatched(
                              CuBLAS::Handle(),
                              n,
                              d_A_ptrs,
                              n,
                              P.Write(),
                              info_array.Write(),
                              n_mats);

   MFEM_VERIFY(status == CUBLAS_STATUS_SUCCESS, "");
#else
   MFEM_ABORT("MFEM must be compiled with CUDA support.")
#endif
}

void BatchLUSolve_CuBLAS(const DenseTensor &A, const Array<int> &P, Vector &B,
                         int nrhs)
{
#ifdef MFEM_USE_CUDA
   const int n = A.SizeI();
   const int n_mats = A.SizeK();

   Array<const real_t*> A_ptrs(n_mats);
   const real_t **d_A_ptrs = A_ptrs.Write();
   Array<real_t*> B_ptrs(n_mats);
   real_t **d_B_ptrs = B_ptrs.Write();

   {
      const real_t *A_base = A.Read();
      real_t *B_base = B.ReadWrite();
      mfem::forall(n_mats, [=] MFEM_HOST_DEVICE (int i)
      {
         d_A_ptrs[i] = A_base + i*n*n;
         d_B_ptrs[i] = B_base + i*n*nrhs;
      });
   }

   int info = 0;

   cublasStatus_t status = cublasDgetrsBatched(CuBLAS::Handle(),
                                               CUBLAS_OP_N,
                                               n,
                                               nrhs,
                                               d_A_ptrs,
                                               n,
                                               P.Read(),
                                               d_B_ptrs,
                                               n,
                                               &info,
                                               n_mats);
   MFEM_VERIFY(status == CUBLAS_STATUS_SUCCESS, "");
#else
   MFEM_ABORT("MFEM must be compiled with CUDA support.")
#endif
}

void BatchLUFactor_Fallback(DenseTensor &A, Array<int> &P, const real_t tol)
{
   const int m = A.SizeI();
   const int NE = A.SizeK();
   P.SetSize(m*NE);

   auto data_all = mfem::Reshape(A.ReadWrite(), m, m, NE);
   auto ipiv_all = mfem::Reshape(P.Write(), m, NE);
   Array<bool> pivot_flag(1);
   pivot_flag[0] = true;
   bool *d_pivot_flag = pivot_flag.ReadWrite();

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      for (int i = 0; i < m; i++)
      {
         // pivoting
         {
            int piv = i;
            real_t a = fabs(data_all(piv,i,e));
            for (int j = i+1; j < m; j++)
            {
               const real_t b = fabs(data_all(j,i,e));
               if (b > a)
               {
                  a = b;
                  piv = j;
               }
            }
            ipiv_all(i,e) = piv;
            if (piv != i)
            {
               // swap rows i and piv in both L and U parts
               for (int j = 0; j < m; j++)
               {
                  mfem::kernels::internal::Swap<real_t>(data_all(i,j,e), data_all(piv,j,e));
               }
            }
         } // pivot end

         if (abs(data_all(i,i,e)) <= tol)
         {
            d_pivot_flag[0] = false;
         }

         const real_t a_ii_inv = 1.0 / data_all(i,i,e);
         for (int j = i+1; j < m; j++)
         {
            data_all(j,i,e) *= a_ii_inv;
         }

         for (int k = i+1; k < m; k++)
         {
            const real_t a_ik = data_all(i,k,e);
            for (int j = i+1; j < m; j++)
            {
               data_all(j,k,e) -= a_ik * data_all(j,i,e);
            }
         }
      } // m loop
   });

   MFEM_VERIFY(pivot_flag.HostRead()[0], "Batch LU factorization failed");
}

void BatchLUSolve_Fallback(const DenseTensor &A, const Array<int> &P,
                           Vector &X, int nrhs)
{
   const int m = A.SizeI();
   const int ne = A.SizeK();

   auto d_LU = mfem::Reshape(A.Read(), m, m, ne);
   auto d_piv = mfem::Reshape(P.Read(), m, ne);
   auto d_x = mfem::Reshape(X.ReadWrite(), m, nrhs, ne);

   mfem::forall(ne*nrhs, [=] MFEM_HOST_DEVICE (int i)
   {
      const int k = i % nrhs;
      const int e = i / nrhs;
      kernels::LUSolve(&d_LU(0,0,e), m, &d_piv(0,e), &d_x(0,k,e));
   });
}

void BatchInverse_CuBLAS(DenseTensor &A)
{
#ifdef MFEM_USE_CUDA
   const int n = A.SizeI();
   const int n_mats = A.SizeK();

   DenseTensor LU(A.SizeI(), A.SizeJ(), A.SizeK());
   LU.Write();
   LU.GetMemory().CopyFrom(A.GetMemory(), A.TotalSize());

   Array<real_t*> LU_ptrs(n_mats);
   Array<real_t*> A_ptrs(n_mats);
   real_t **d_A_ptrs = A_ptrs.Write();
   real_t **d_LU_ptrs = LU_ptrs.Write();
   {
      real_t *A_base = A.ReadWrite();
      real_t *LU_base = LU.Write();
      mfem::forall(n_mats, [=] MFEM_HOST_DEVICE (int i)
      {
         d_A_ptrs[i] = A_base + i*n*n;
         d_LU_ptrs[i] = LU_base + i*n*n;
      });
   }

   Array<int> P(n*n_mats);
   Array<int> info_array(n_mats);
   cublasStatus_t status;

   status = cublasDgetrfBatched(CuBLAS::Handle(),
                                n,
                                d_LU_ptrs,
                                n,
                                P.Write(),
                                info_array.Write(),
                                n_mats);
   MFEM_VERIFY(status == CUBLAS_STATUS_SUCCESS, "");

   status = cublasDgetriBatched(CuBLAS::Handle(),
                                n,
                                d_LU_ptrs,
                                n,
                                P.ReadWrite(),
                                d_A_ptrs,
                                n,
                                info_array.Write(),
                                n_mats);
   MFEM_VERIFY(status == CUBLAS_STATUS_SUCCESS, "");

#else
   MFEM_ABORT("");
#endif
}

void BatchInverse_Fallback(DenseTensor &A)
{
   MFEM_ABORT("");
}

void BatchLUFactor(DenseTensor &A, Array<int> &P, const real_t tol)
{
   if (Device::Allows(Backend::CUDA_MASK))
   {
      BatchLUFactor_CuBLAS(A, P, tol);
   }
   else
   {
      BatchLUFactor_Fallback(A, P, tol);
   }
}

void BatchLUSolve(const DenseTensor &A, const Array<int> &P, Vector &X,
                  int nrhs)
{
   if (Device::Allows(Backend::CUDA_MASK))
   {
      BatchLUSolve_CuBLAS(A, P, X, nrhs);
   }
   else
   {
      BatchLUSolve_Fallback(A, P, X, nrhs);
   }
}

void BatchInverse(DenseTensor &A)
{
   if (Device::Allows(Backend::CUDA_MASK))
   {
      BatchInverse_CuBLAS(A);
   }
   else
   {
      BatchInverse_Fallback(A);
   }
}

void BatchSetup()
{
   if (Device::Allows(Backend::CUDA_MASK))
   {
#ifdef MFEM_USE_CUDA
      CuBLAS::Handle();
      CuSolver::Handle();
#endif
   }
}

}
