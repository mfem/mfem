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

#include "tripack_magma.hpp"

#ifdef MFEM_USE_MAGMA

#include "../general/forall.hpp"

namespace mfem
{
namespace
{

#ifdef MFEM_USE_SINGLE
#define MFEM_TRIPACK_MAGMA_PREFIX(stub) magma_s##stub
#define MFEM_TRIPACK_MAGMA_SET_POINTER magma_sset_pointer
#elif defined(MFEM_USE_DOUBLE)
#define MFEM_TRIPACK_MAGMA_PREFIX(stub) magma_d##stub
#define MFEM_TRIPACK_MAGMA_SET_POINTER magma_dset_pointer
#else
#error "Unsupported MFEM precision for MAGMA packed routines."
#endif

real_t **SetPackedPointerArray(Array<real_t *> &ptrs,
                               real_t *data,
                               const int stride,
                               const int batch_size,
                               const magma_queue_t queue)
{
   if (ptrs.Size() != batch_size)
   {
      if (ptrs.Size() != 0) { magma_queue_sync(queue); }
      ptrs.SetSize(batch_size, Device::GetDeviceMemoryType());
   }

   real_t **d_ptrs = ptrs.Write();
   MFEM_TRIPACK_MAGMA_SET_POINTER(d_ptrs, data, 1, 0, 0, stride,
                                  batch_size, queue);
   return d_ptrs;
}

} // namespace

MagmaPackedLowerCholesky::MagmaPackedLowerCholesky()
{
   queue = Magma::Queue();
}

void MagmaPackedLowerCholesky::Factor(
   const TriPackLowerMatrix &A,
   TriPackLowerMatrix &L)
{
   MFEM_VERIFY(queue != nullptr, "MAGMA queue is not set.");

   n = A.GetNumRows();
   batch_size = A.GetNumMatrices();
   packed_size = A.GetPackedSize();

   L.SetSize(n, batch_size);
   L.UseDevice(true);

   if (batch_size == 0) { return; }

   L.Data() = A.Data();

   real_t *factor_data = L.Data().ReadWrite();
   real_t **d_factor_ptrs =
      SetPackedPointerArray(factor_ptrs, factor_data, packed_size,
                            batch_size, queue);

   info.SetSize(batch_size, Device::GetDeviceMemoryType());
   magma_int_t *d_info = info.Write();
   magma_memset(d_info, 0, batch_size*sizeof(magma_int_t));

   const magma_int_t status =
      (n <= 8) ?
      MFEM_TRIPACK_MAGMA_PREFIX(pptrf_batched_small)(
         MagmaLower, n, d_factor_ptrs, d_info, batch_size, queue) :
      MFEM_TRIPACK_MAGMA_PREFIX(pptf2_batched_small)(
         MagmaLower, n, d_factor_ptrs, d_info, batch_size, queue);

   MFEM_VERIFY(status == MAGMA_SUCCESS,
               "MAGMA packed Cholesky factorization failed.");

   magma_queue_sync(queue);

   const magma_int_t *h_info = info.HostRead();
   for (int e = 0; e < batch_size; ++e)
   {
      MFEM_VERIFY(h_info[e] == 0,
                  "MAGMA packed Cholesky factorization failed on matrix "
                  << e << '.');
   }
}

void MagmaPackedLowerCholesky::SolveInPlace(
   const TriPackLowerMatrix &L,
   Vector &rhs_sol) const
{
   MFEM_VERIFY(queue != nullptr, "MAGMA queue is not set.");
   MFEM_VERIFY(L.GetNumRows() > 0 || L.GetNumMatrices() == 0,
               "Invalid factor dimensions.");

   const int solve_n = L.GetNumRows();
   const int solve_batch = L.GetNumMatrices();
   const int solve_packed = L.GetPackedSize();

   MFEM_VERIFY(rhs_sol.Size() == solve_batch*solve_n,
               "Right-hand side has the wrong size.");
   if (solve_batch == 0) { return; }

   real_t *factor_data = const_cast<real_t *>(L.Data().Read());
   real_t **d_factor_ptrs =
      SetPackedPointerArray(factor_ptrs, factor_data, solve_packed,
                            solve_batch, queue);

   real_t *rhs_data = rhs_sol.ReadWrite();
   real_t **d_rhs_ptrs =
      SetPackedPointerArray(rhs_ptrs, rhs_data, solve_n, solve_batch, queue);

   // On HIP, MAGMA's packed 1-RHS batched solve kernel only supports n <= 32.
   // Fail fast with a clear error for larger element sizes (e.g., (p+1)^3 = 64).
   if (Device::Allows(Backend::HIP_MASK))
   {
      MFEM_VERIFY(solve_n <= 32,
                  "MAGMA packed Cholesky solve (1 RHS) supports n <= 32 on HIP; "
                  "got n = " << solve_n << ". "
                  "Use a smaller element size (lower order), or disable MAGMA "
                  "solve in the benchmark, or use the full (dense) MAGMA path.");
   }

   const magma_int_t status =
      MFEM_TRIPACK_MAGMA_PREFIX(pptrs_1rhs_batched_small)(
         solve_n, 1, d_factor_ptrs, d_rhs_ptrs, solve_n, solve_batch, queue);

   MFEM_VERIFY(status == MAGMA_SUCCESS,
               "MAGMA packed Cholesky solve failed.");
}

MagmaPackedLowerInverse::MagmaPackedLowerInverse()
{
   queue = Magma::Queue();
}

void MagmaPackedLowerInverse::Compute(
   const TriPackLowerMatrix &A,
   TriPackLowerMatrix &A_inv)
{
   MFEM_VERIFY(queue != nullptr, "MAGMA queue is not set.");

   n = A.GetNumRows();
   batch_size = A.GetNumMatrices();
   packed_size = A.GetPackedSize();

   MFEM_VERIFY(n <= 64, "MAGMA packed inverse supports n <= 64.");

   A_inv.SetSize(n, batch_size);
   A_inv.UseDevice(true);

   if (batch_size == 0) { return; }

   A_inv.Data() = A.Data();

   real_t *inv_data = A_inv.Data().ReadWrite();
   real_t **d_inv_ptrs =
      SetPackedPointerArray(inv_ptrs, inv_data, packed_size, batch_size, queue);

   info.SetSize(batch_size, Device::GetDeviceMemoryType());
   magma_int_t *d_info = info.Write();
   magma_memset(d_info, 0, batch_size*sizeof(magma_int_t));

   // MAGMA currently expects a valid pointer for device_lwork even when the
   // required workspace is 0 bytes.
   int64_t device_lwork[1] = {0};
   const magma_int_t status =
      MFEM_TRIPACK_MAGMA_PREFIX(ppinv_batched)(
         MagmaLower, n, d_inv_ptrs,
         /*device_work*/ nullptr, device_lwork,
         d_info, batch_size, queue);
   MFEM_VERIFY(status == MAGMA_SUCCESS, "MAGMA packed inverse failed.");

   magma_queue_sync(queue);

   const magma_int_t *h_info = info.HostRead();
   for (int e = 0; e < batch_size; ++e)
   {
      MFEM_VERIFY(h_info[e] == 0,
                  "MAGMA packed inverse failed on matrix " << e << '.');
   }
}

void MagmaPackedLowerInverse::ApplyInPlace(
   const TriPackLowerMatrix &A_inv,
   Vector &rhs_sol) const
{
   MFEM_VERIFY(queue != nullptr, "MAGMA queue is not set.");
   MFEM_VERIFY(A_inv.GetNumRows() > 0 || A_inv.GetNumMatrices() == 0,
               "Invalid inverse dimensions.");

   const int apply_n = A_inv.GetNumRows();
   const int apply_batch = A_inv.GetNumMatrices();
   const int apply_packed = A_inv.GetPackedSize();

   MFEM_VERIFY(rhs_sol.Size() == apply_batch*apply_n,
               "Right-hand side has the wrong size.");

   if (apply_batch == 0) { return; }

   // Prefer MAGMA's tuned packed-symmetric matvec when available (n <= 32).
   // Fall back to an MFEM device kernel for larger n.
   if (apply_n <= 32)
   {
      real_t *inv_data = const_cast<real_t *>(A_inv.Data().Read());
      real_t **d_inv_ptrs =
         SetPackedPointerArray(inv_ptrs, inv_data, apply_packed, apply_batch,
                               queue);

      real_t *rhs_data = rhs_sol.ReadWrite();
      real_t **d_rhs_ptrs =
         SetPackedPointerArray(rhs_ptrs, rhs_data, apply_n, apply_batch, queue);

      // Note: MAGMA's symv_packed_inplace_batched_small returns void; it will
      // report argument errors via magma_xerbla.
      MFEM_TRIPACK_MAGMA_PREFIX(symv_packed_inplace_batched_small)(
         MagmaLower, apply_n, d_inv_ptrs, d_rhs_ptrs, apply_n, apply_batch,
         queue);
      return;
   }

   work.SetSize(apply_batch*apply_n);
   work.UseDevice(true);

   const real_t *AP = A_inv.Data().Read();
   const real_t *X = rhs_sol.Read();
   real_t *Y = work.Write();

   mfem::forall(apply_batch*apply_n, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int i = idx % apply_n;
      const int e = idx / apply_n;
      const real_t *APe = AP + e*apply_packed;
      const real_t *Xe = X + e*apply_n;
      real_t sum = 0.0;
      for (int j = 0; j < apply_n; ++j)
      {
         const real_t aij =
            (i >= j) ?
            APe[TriPackLowerMatrix::LowerIndex(i, j, apply_n)] :
            APe[TriPackLowerMatrix::LowerIndex(j, i, apply_n)];
         sum += aij * Xe[j];
      }
      Y[idx] = sum;
   });

   const real_t *Y_in = work.Read();
   real_t *X_out = rhs_sol.Write();
   mfem::forall(apply_batch*apply_n, [=] MFEM_HOST_DEVICE (int idx)
   {
      X_out[idx] = Y_in[idx];
   });
}

#undef MFEM_TRIPACK_MAGMA_SET_POINTER
#undef MFEM_TRIPACK_MAGMA_PREFIX

} // namespace mfem

#endif // MFEM_USE_MAGMA
