// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
#pragma once

#include "../../general/array.hpp"

#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
#include "../../general/backends.hpp" // IWYU pragma: keep
#endif

#include <algorithm>
#include <cstddef>
#include <cstring>

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

namespace internal
{

#if defined(MFEM_USE_CUDA_OR_HIP_LANG)
template<class Tag, int Rows, int Cols>
MFEM_DEVICE_CONSTANT real_t ConstantMatrixSymbol[Rows][Cols];
#endif

/** @brief Rows-by-Cols basis table stored for both host and device kernels. */
template<class Tag, int Rows, int Cols>
struct ConstantMatrix
{
   static_assert(Rows > 0 && Cols > 0);

   // Host-side values
   static inline real_t host[Rows][Cols];

   // Values visible to the kernel
   MFEM_HOST_DEVICE static const real_t (&Read())[Rows][Cols]
   {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      return ConstantMatrixSymbol<Tag, Rows, Cols>;
#else
      return host;
#endif
   }

   // Copy into the host buffer and on device into the symbol
   static void Upload(const real_t *src, std::size_t bytes)
   {
      MFEM_ASSERT(bytes == sizeof(host), "Invalid constant basis upload");
      std::memcpy(host, src, bytes);
#if defined(MFEM_USE_CUDA_OR_HIP_LANG)
      if (Device::Allows(Backend::DEVICE_MASK & ~Backend::DEBUG_DEVICE))
      {
         MFEM_GPU_CHECK(
            MFEM_cuda_or_hip(MemcpyToSymbol)
            (MFEM_DEVICE_SYMBOL(ConstantMatrixSymbol<Tag, Rows, Cols>),
             src, bytes, 0, MFEM_cuda_or_hip(MemcpyHostToDevice)));
      }
#endif
   }
};

MFEM_HOST_DEVICE inline int ij_to_index(const int p, const int i, const int j)
{
   return i * (2 * p - i + 1) / 2 + j;
}

MFEM_HOST_DEVICE inline int ijk_to_index(const int p, const int i, const int j,
                                         const int k)
{
   const int n = p - k;
   const int layer_sum = (p * (p + 1) * (p + 2) - n * (n + 1) * (n + 2)) / 6;
   const int offset_in_layer = j * (2 * n - j + 1) / 2 + i;
   return layer_sum + offset_in_layer;
}

/** @brief Host mirror of the basis values stored in one device constant symbol.

    Each symbol must have its own cache. This cache owns its host copy; it does
    not retain the input array or a pointer to it.

    @warning This cache and its constant symbol are shared mutable state and
    are not concurrency-safe. An operator application may upload new values.
    Callers must serialize updates with all kernels reading the same symbol,
    including across host threads and device streams. Locking only the upload
    does not protect kernels that are still reading the previous values. */
template<int N>
class ConstantBasisCache
{
private:
   /// Owned copy of the values from the last successful upload.
   Array<real_t> uploaded;

public:
   /** @brief Upload changed basis values and then update the host mirror.

       The upload callback receives the host data and its size in bytes. It
       must complete the upload successfully before returning. Empty bases
       (N == 0) need no upload, as in degenerate derivative bases. */
   template<typename Upload>
   void Update(const Array<real_t> &basis, Upload upload)
   {
      static_assert(N >= 0, "Invalid constant basis size");
      MFEM_VERIFY(basis.Size() == N, "Invalid constant basis size");
      if (N == 0) { return; }

      const real_t *values = basis.HostRead();
      if (uploaded.Size() == N &&
          std::equal(values, values + N, uploaded.HostRead()))
      {
         return;
      }

      upload(values, N * sizeof(real_t));
      uploaded.SetSize(N);
      std::copy(values, values + N, uploaded.HostWrite());
   }
};

} // namespace internal

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
