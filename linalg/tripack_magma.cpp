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
   const TriPackMatrix<TriangularPart::LOWER> &A,
   TriPackMatrix<TriangularPart::LOWER> &L)
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
   const TriPackMatrix<TriangularPart::LOWER> &L,
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

   const magma_int_t status =
      MFEM_TRIPACK_MAGMA_PREFIX(pptrs_batched)(
         MagmaLower, solve_n, 1, d_factor_ptrs, d_rhs_ptrs, solve_n,
         solve_batch, queue);

   MFEM_VERIFY(status == MAGMA_SUCCESS,
               "MAGMA packed Cholesky solve failed.");
}

#undef MFEM_TRIPACK_MAGMA_SET_POINTER
#undef MFEM_TRIPACK_MAGMA_PREFIX

} // namespace mfem

#endif // MFEM_USE_MAGMA
