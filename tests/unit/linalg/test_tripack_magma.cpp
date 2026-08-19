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

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

#if defined(MFEM_USE_MAGMA) && (defined(MFEM_USE_HIP) || defined(MFEM_USE_CUDA))

#include <vector>

#if defined(MFEM_USE_HIP)
#include <hip/hip_runtime.h>
#elif defined(MFEM_USE_CUDA)
#include <cuda_runtime.h>
#endif

namespace
{

bool HasGpuDevice()
{
#if defined(MFEM_USE_HIP)
   int count = 0;
   const hipError_t err = hipGetDeviceCount(&count);
   return err == hipSuccess && count > 0;
#elif defined(MFEM_USE_CUDA)
   int count = 0;
   const cudaError_t err = cudaGetDeviceCount(&count);
   return err == cudaSuccess && count > 0;
#else
   return false;
#endif
}

DenseMatrix MakeSPD(const int n, const int seed)
{
   DenseMatrix B(n), A(n);
   for (int j = 0; j < n; ++j)
   {
      for (int i = 0; i < n; ++i)
      {
         const int v = (17*(i + 1) + 31*(j + 1) + 7*seed) % 23;
         B(i, j) = 0.05 * real_t(v);
      }
   }
   MultAtB(B, B, A);
   for (int i = 0; i < n; ++i) { A(i, i) += 1.0 + 0.1*i; }
   return A;
}

void PackLower(const DenseMatrix &mat, real_t *packed)
{
   const int n = mat.Height();
   for (int j = 0; j < n; ++j)
   {
      for (int i = j; i < n; ++i)
      {
         packed[TriPackMatrix<TriangularPart::LOWER>::LowerIndex(i, j, n)] =
            mat(i, j);
      }
   }
}

real_t MaxResidual(const DenseMatrix &A, const Vector &x, const Vector &b)
{
   const int n = A.Height();
   real_t max_abs = 0.0;
   for (int i = 0; i < n; ++i)
   {
      real_t sum = 0.0;
      for (int j = 0; j < n; ++j) { sum += A(i, j) * x(j); }
      max_abs = std::max(max_abs, std::abs(sum - b(i)));
   }
   return max_abs;
}

} // namespace

TEST_CASE("MAGMA packed-lower Cholesky factor+solve", "[MAGMA][TriPackMatrix]")
{
   if (!HasGpuDevice())
   {
      WARN("No GPU device visible; skipping MAGMA packed-lower tests.");
      return;
   }

   Device device(
#if defined(MFEM_USE_HIP)
      "hip"
#elif defined(MFEM_USE_CUDA)
      "cuda"
#else
      "cpu"
#endif
   );

   constexpr int n = 8;
   constexpr int batch_size = 17;
   constexpr double tol = 5e-9;

   TriPackMatrix<TriangularPart::LOWER> A_packed(n, batch_size);
   A_packed.UseDevice(true);
   A_packed = 0.0;

   std::vector<DenseMatrix> A_dense;
   A_dense.reserve(batch_size);

   real_t *h_packed = A_packed.Data().HostWrite();
   const int ps = A_packed.GetPackedSize();
   for (int e = 0; e < batch_size; ++e)
   {
      A_dense.emplace_back(MakeSPD(n, e + 1));
      PackLower(A_dense.back(), h_packed + e*ps);
   }

   Vector b(batch_size*n);
   Vector x(batch_size*n);
   real_t *h_b = b.HostWrite();
   for (int e = 0; e < batch_size; ++e)
   {
      for (int i = 0; i < n; ++i)
      {
         h_b[e*n + i] = 1.0 + real_t((13*(i + 1) + 7*(e + 1)) % 29)/real_t(29);
      }
   }
   b.UseDevice(true);
   x.UseDevice(true);

   TriPackMatrix<TriangularPart::LOWER> L;
   MagmaPackedLowerCholesky ws;
   ws.Factor(A_packed, L);

   x = b;
   ws.SolveInPlace(L, x);
   MFEM_DEVICE_SYNC;

   const real_t *h_x = x.HostRead();
   const real_t *h_b_ro = b.HostRead();
   for (int e = 0; e < batch_size; ++e)
   {
      Vector xe(const_cast<real_t *>(h_x) + e*n, n);
      Vector be(const_cast<real_t *>(h_b_ro) + e*n, n);
      const real_t res = MaxResidual(A_dense[e], xe, be);
      REQUIRE(res == MFEM_Approx(0.0, tol, tol));
   }
}

#else

TEST_CASE("MAGMA packed-lower tests disabled", "[MAGMA][TriPackMatrix]")
{
   SUCCEED("MFEM was built without MAGMA+GPU support.");
}

#endif
