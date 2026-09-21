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

#include "tripack.hpp"

#include "../general/forall.hpp"

#include <cmath>
#include <limits>

namespace mfem
{
namespace
{

MFEM_HOST_DEVICE inline int SymmetricIndex(const int i,
                                           const int j,
                                           const int n)
{
   return TriPackLowerMatrix::Index(i, j, n);
}

void ComputeCholeskyFactorsLower(
   const TriPackLowerMatrix &packed_lower,
   Vector &factor)
{
   const int n = packed_lower.GetNumRows();
   const int batch_size = packed_lower.GetNumMatrices();
   const int packed_size = packed_lower.GetPackedSize();
   const real_t nan = std::numeric_limits<real_t>::quiet_NaN();

   const real_t *A = packed_lower.Data().Read();
   factor.SetSize(batch_size*packed_size);
   factor.UseDevice(true);
   real_t *L = factor.Write();

   mfem::forall(batch_size, [=] MFEM_HOST_DEVICE (int e)
   {
      const int eoff = e*packed_size;
      const real_t eps = std::numeric_limits<real_t>::epsilon();
      bool bad = false;

      // Copy packed-lower input into factor storage.
      for (int j = 0; j < n; ++j)
      {
         for (int i = j; i < n; ++i)
         {
            const int t = TriPackLowerMatrix::LowerIndex(i, j, n);
            const real_t Aij = A[eoff + t];
            if (!IsFinite(Aij)) { bad = true; }
            L[eoff + t] = Aij;
         }
      }

      if (bad)
      {
         for (int t = 0; t < packed_size; ++t) { L[eoff + t] = nan; }
         return;
      }

      for (int k = 0; k < n; ++k)
      {
         const int kk = eoff + TriPackLowerMatrix::LowerIndex(k, k, n);
         const real_t Lkk0 = L[kk];
         real_t Lkk = Lkk0;

         for (int s = 0; s < k; ++s)
         {
            const real_t Lks = L[eoff + TriPackLowerMatrix::LowerIndex(k, s, n)];
            Lkk -= Lks*Lks;
         }

         const real_t tol = 64.0*eps*fabs(Lkk0);
         if (!IsFinite(Lkk) || Lkk < -tol)
         {
            bad = true;
            break;
         }

         if (Lkk < 0.0) { Lkk = 0.0; }
         L[kk] = sqrt(Lkk);

         const real_t Ldiag = L[kk];
         for (int i = k + 1; i < n; ++i)
         {
            const int ik = eoff + TriPackLowerMatrix::LowerIndex(i, k, n);
            real_t Aik = L[ik];
            for (int s = 0; s < k; ++s)
            {
               Aik -= L[eoff + TriPackLowerMatrix::LowerIndex(i, s, n)] *
                      L[eoff + TriPackLowerMatrix::LowerIndex(k, s, n)];
            }
            L[ik] = Aik/Ldiag;
         }
      }

      if (bad)
      {
         for (int t = 0; t < packed_size; ++t) { L[eoff + t] = nan; }
      }
   });
}

}

namespace tripack
{

bool CompareWithFull(const TriPackLowerMatrix &packed, const Vector &full,
                     real_t tol)
{
   const int n = packed.GetNumRows();
   const int batch_size = packed.GetNumMatrices();
   const int packed_size = packed.GetPackedSize();

   MFEM_VERIFY(full.Size() == batch_size*n*n,
               "Full matrix data has the wrong size.");

   const real_t *packed_data = packed.Data().HostRead();
   const real_t *full_data = full.HostRead();

   if (tol == 0.0)
   {
      tol = 256.0*std::numeric_limits<real_t>::epsilon();
   }

   for (int e = 0; e < batch_size; ++e)
   {
      const int packed_offset = e*packed_size;
      const int full_offset = e*n*n;
      for (int i = 0; i < n; ++i)
      {
         for (int j = 0; j < n; ++j)
         {
            const real_t packed_val =
               packed_data[packed_offset + SymmetricIndex(i, j, n)];
            const real_t full_val = full_data[full_offset + i + n*j];
            if (std::fabs(full_val - packed_val) > tol)
            {
               return false;
            }
         }
      }
      for (int i = 0; i < n; ++i)
      {
         for (int j = i + 1; j < n; ++j)
         {
            const real_t a = full_data[full_offset + i + n*j];
            const real_t b = full_data[full_offset + j + n*i];
            if (std::fabs(a - b) > tol)
            {
               return false;
            }
         }
      }
   }

   return true;
}

void Mult(const TriPackLowerMatrix &packed, const Vector &x, Vector &y)
{
   const int n = packed.GetNumRows();
   const int batch_size = packed.GetNumMatrices();

   MFEM_VERIFY(x.Size() == batch_size*n, "Input vector has the wrong size.");

   y.SetSize(batch_size*n);
   y.UseDevice(true);

   const real_t *A = packed.Data().Read();
   const real_t *X = x.Read();
   real_t *Y = y.Write();

   mfem::forall(batch_size*n, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int i = idx % n;
      const int e = idx / n;
      const real_t *Ae = A + e*TriPackLowerMatrix::PackedSize(n);
      const real_t *Xe = X + e*n;
      real_t sum = 0.0;
      for (int j = 0; j < n; ++j)
      {
         sum += Ae[SymmetricIndex(i, j, n)] * Xe[j];
      }
      Y[idx] = sum;
   });
}

void Lump(const TriPackLowerMatrix &packed, Vector &lump)
{
   const int n = packed.GetNumRows();
   const int batch_size = packed.GetNumMatrices();

   lump.SetSize(batch_size*n);
   lump.UseDevice(true);

   const real_t *A = packed.Data().Read();
   real_t *L = lump.Write();

   mfem::forall(batch_size*n, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int i = idx % n;
      const int e = idx / n;
      const real_t *Ae = A + e*TriPackLowerMatrix::PackedSize(n);
      real_t sum = 0.0;
      for (int j = 0; j < n; ++j)
      {
         sum += Ae[SymmetricIndex(i, j, n)];
      }
      L[idx] = sum;
   });
}

void ComputeCholeskyLower(const TriPackLowerMatrix &packed_lower,
                          TriPackLowerMatrix &lower_factor)
{
   const int n = packed_lower.GetNumRows();
   const int batch_size = packed_lower.GetNumMatrices();

   MFEM_VERIFY(&packed_lower != &lower_factor,
               "Input and output TriPackLowerMatrix objects must be distinct.");
   if (batch_size == 0)
   {
      lower_factor.SetSize(n, batch_size);
      return;
   }

   Vector factored;
   ComputeCholeskyFactorsLower(packed_lower, factored);

   lower_factor.SetSize(n, batch_size);
   lower_factor.UseDevice(true);

   lower_factor.Data() = factored;
}

void ComputeCholeskyLowerInverse(const TriPackLowerMatrix &packed_lower,
                                 TriPackLowerMatrix &lower_inverse)
{
   const int n = packed_lower.GetNumRows();
   const int batch_size = packed_lower.GetNumMatrices();
   const int packed_size = packed_lower.GetPackedSize();
   const real_t nan = std::numeric_limits<real_t>::quiet_NaN();

   MFEM_VERIFY(&packed_lower != &lower_inverse,
               "Input and output TriPackLowerMatrix objects must be distinct.");
   if (batch_size == 0)
   {
      lower_inverse.SetSize(n, batch_size);
      return;
   }

   lower_inverse.SetSize(n, batch_size);
   lower_inverse.UseDevice(true);

   Vector factored;
   Vector work(batch_size*packed_size);
   work.UseDevice(true);

   ComputeCholeskyFactorsLower(packed_lower, factored);

   const real_t *L = factored.Read();
   real_t *X = work.Write();
   real_t *Linv = lower_inverse.Data().Write();

   mfem::forall(batch_size, [=] MFEM_HOST_DEVICE (int e)
   {
      const int eoff = e*packed_size;
      bool bad = false;

      for (int t = 0; t < packed_size; ++t)
      {
         if (!IsFinite(L[eoff + t]))
         {
            bad = true;
            break;
         }
      }

      if (bad)
      {
         for (int t = 0; t < packed_size; ++t) { Linv[eoff + t] = nan; }
         return;
      }

      for (int t = 0; t < packed_size; ++t) { X[eoff + t] = 0.0; }

      // Compute X = L^{-1} (packed lower).
      for (int j = 0; j < n; ++j)
      {
         const int jj = eoff + TriPackLowerMatrix::LowerIndex(j, j, n);
         X[jj] = 1.0/L[jj];

         for (int i = j + 1; i < n; ++i)
         {
            real_t sum = 0.0;
            for (int k = j; k < i; ++k)
            {
               sum += L[eoff + TriPackLowerMatrix::LowerIndex(i, k, n)] *
                      X[eoff + TriPackLowerMatrix::LowerIndex(k, j, n)];
            }
            X[eoff + TriPackLowerMatrix::LowerIndex(i, j, n)] =
               -sum/L[eoff + TriPackLowerMatrix::LowerIndex(i, i, n)];
         }
      }

      for (int t = 0; t < packed_size; ++t)
      {
         Linv[eoff + t] = X[eoff + t];
      }
   });
}

void SolveLower(const TriPackLowerMatrix &lower_factor,
                const Vector &rhs,
                Vector &sol)
{
   const int n = lower_factor.GetNumRows();
   const int batch_size = lower_factor.GetNumMatrices();
   const int packed_size = lower_factor.GetPackedSize();

   MFEM_VERIFY(rhs.Size() == batch_size*n, "Right-hand side has the wrong size.");

   Vector out(batch_size*n);
   out.UseDevice(true);

   const real_t *L = lower_factor.Data().Read();
   const real_t *B = rhs.Read();
   real_t *X = out.Write();

   mfem::forall(batch_size, [=] MFEM_HOST_DEVICE (int e)
   {
      const real_t *Le = L + e*packed_size;
      const real_t *Be = B + e*n;
      real_t *Xe = X + e*n;

      for (int i = 0; i < n; ++i)
      {
         real_t sum = Be[i];
         for (int j = 0; j < i; ++j)
         {
            sum -= Le[TriPackLowerMatrix::LowerIndex(i, j, n)] * Xe[j];
         }
         Xe[i] = sum / Le[TriPackLowerMatrix::LowerIndex(i, i, n)];
      }
   });

   sol.SetSize(batch_size*n);
   sol = out;
}

void SolveLowerTranspose(const TriPackLowerMatrix &lower_factor,
                         const Vector &rhs,
                         Vector &sol)
{
   const int n = lower_factor.GetNumRows();
   const int batch_size = lower_factor.GetNumMatrices();
   const int packed_size = lower_factor.GetPackedSize();

   MFEM_VERIFY(rhs.Size() == batch_size*n, "Right-hand side has the wrong size.");

   Vector out(batch_size*n);
   out.UseDevice(true);

   const real_t *L = lower_factor.Data().Read();
   const real_t *B = rhs.Read();
   real_t *X = out.Write();

   mfem::forall(batch_size, [=] MFEM_HOST_DEVICE (int e)
   {
      const real_t *Le = L + e*packed_size;
      const real_t *Be = B + e*n;
      real_t *Xe = X + e*n;

      for (int i = n - 1; i >= 0; --i)
      {
         real_t sum = Be[i];
         for (int j = i + 1; j < n; ++j)
         {
            sum -= Le[TriPackLowerMatrix::LowerIndex(j, i, n)] * Xe[j];
         }
         Xe[i] = sum / Le[TriPackLowerMatrix::LowerIndex(i, i, n)];
      }
   });

   sol.SetSize(batch_size*n);
   sol = out;
}

void SolveCholesky(const TriPackLowerMatrix &lower_factor,
                   const Vector &rhs,
                   Vector &sol)
{
   Vector tmp;
   SolveLower(lower_factor, rhs, tmp);
   SolveLowerTranspose(lower_factor, tmp, sol);
}

} // namespace tripack
} // namespace mfem
