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

#ifndef MFEM_BILININTEG_PA_SIMPLICES_COMMON_HPP
#define MFEM_BILININTEG_PA_SIMPLICES_COMMON_HPP

#include "../../general/array.hpp"
#include "../../general/backends.hpp"

#include <algorithm>
#include <cstddef>

#if defined(MFEM_USE_CUDA)
#define MFEM_cuda_or_hip(stub) cuda##stub
#elif defined(MFEM_USE_HIP)
#define MFEM_cuda_or_hip(stub) hip##stub
#endif

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

namespace internal
{

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

#endif // MFEM_BILININTEG_PA_SIMPLICES_COMMON_HPP
