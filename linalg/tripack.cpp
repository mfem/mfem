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

template <TriangularPart PART>
MFEM_HOST_DEVICE inline int SymmetricIndex(const int i,
                                           const int j,
                                           const int n)
{
   if constexpr (PART == TriangularPart::UPPER)
   {
      return (i <= j) ? TriPackMatrix<PART>::UpperIndex(i, j, n)
                      : TriPackMatrix<PART>::UpperIndex(j, i, n);
   }
   return (i >= j) ? TriPackMatrix<PART>::LowerIndex(i, j)
                   : TriPackMatrix<PART>::LowerIndex(j, i);
}

MFEM_HOST_DEVICE inline int UpperRowFromPackedIndex(const int t, const int n)
{
   int row = 0;
   while (row + 1 < n &&
          TriPackMatrix<TriangularPart::UPPER>::UpperIndex(row + 1, row + 1, n) <= t)
   {
      ++row;
   }
   return row;
}

MFEM_HOST_DEVICE inline int UpperColFromPackedIndex(const int t, const int n)
{
   const int row = UpperRowFromPackedIndex(t, n);
   const int row_start =
      TriPackMatrix<TriangularPart::UPPER>::UpperIndex(row, row, n);
   return row + (t - row_start);
}

template <TriangularPart PART>
MFEM_HOST_DEVICE inline real_t TriPackGet(const real_t *data,
                                          const int i,
                                          const int j,
                                          const int n)
{
   return data[SymmetricIndex<PART>(i, j, n)];
}

MFEM_HOST_DEVICE inline bool TriPackIsFinite(const real_t val)
{
#ifdef isfinite
   return isfinite(val);
#else
   return std::isfinite(val);
#endif
}

void ComputeScaledCholeskyFactors(
                                  const TriPackMatrix<TriangularPart::UPPER> &packed_upper,
                                  Vector &scaled_factor,
                                  Vector &scaling,
                                  bool do_scale)
{
   const int n = packed_upper.GetNumRows();
   const int batch_size = packed_upper.GetNumMatrices();
   const int packed_size = packed_upper.GetPackedSize();
   const real_t nan = std::numeric_limits<real_t>::quiet_NaN();

   scaled_factor.SetSize(batch_size*packed_size);
   scaling.SetSize(batch_size*n);
   scaled_factor.UseDevice(true);
   scaling.UseDevice(true);

   const real_t *A = packed_upper.Data().Read();
   real_t *R = scaled_factor.Write();
   real_t *D = scaling.Write();

   mfem::forall(batch_size, [=] MFEM_HOST_DEVICE (int e)
   {
      const int eoff = e*packed_size;
      const int doff = e*n;
      const real_t eps = std::numeric_limits<real_t>::epsilon();
      bool bad = false;

      for (int i = 0; i < n; ++i)
      {
         const real_t Aii = A[eoff + TriPackMatrix<TriangularPart::UPPER>::UpperIndex(i, i, n)];
         if (!(Aii > 0.0) || !TriPackIsFinite(Aii))
         {
            bad = true;
            break;
         }
         D[doff + i] = do_scale ? 1.0/sqrt(Aii) : 1.0;
      }

      if (bad)
      {
         for (int i = 0; i < n; ++i) { D[doff + i] = nan; }
         for (int t = 0; t < packed_size; ++t) { R[eoff + t] = nan; }
         return;
      }

      for (int j = 0; j < n; ++j)
      {
         const real_t dj = D[doff + j];
         for (int i = 0; i <= j; ++i)
         {
            const int t = TriPackMatrix<TriangularPart::UPPER>::UpperIndex(i, j, n);
            R[eoff + t] = A[eoff + t] * D[doff + i] * dj;
         }
      }

      for (int k = 0; k < n; ++k)
      {
         const int kk = eoff + TriPackMatrix<TriangularPart::UPPER>::UpperIndex(k, k, n);
         const real_t Akk0 = R[kk];
         real_t Akk = Akk0;

         for (int s = 0; s < k; ++s)
         {
            const real_t Usk = R[eoff + TriPackMatrix<TriangularPart::UPPER>::UpperIndex(s, k, n)];
            Akk -= Usk*Usk;
         }

         const real_t tol = 64.0*eps*fabs(Akk0);
         if (!TriPackIsFinite(Akk) || Akk < -tol)
         {
            bad = true;
            break;
         }

         if (Akk < 0.0) { Akk = 0.0; }
         R[kk] = sqrt(Akk);

         const real_t Ukk = R[kk];
         for (int j = k + 1; j < n; ++j)
         {
            const int kj = eoff + TriPackMatrix<TriangularPart::UPPER>::UpperIndex(k, j, n);
            real_t Akj = R[kj];
            for (int s = 0; s < k; ++s)
            {
               Akj -= R[eoff + TriPackMatrix<TriangularPart::UPPER>::UpperIndex(s, k, n)] *
                      R[eoff + TriPackMatrix<TriangularPart::UPPER>::UpperIndex(s, j, n)];
            }
            R[kj] = Akj/Ukk;
         }
      }

      if (bad)
      {
         for (int i = 0; i < n; ++i) { D[doff + i] = nan; }
         for (int t = 0; t < packed_size; ++t) { R[eoff + t] = nan; }
      }
   });
}

}

namespace tripack
{

template <TriangularPart PART>
bool CompareWithFull(const TriPackMatrix<PART> &packed, const Vector &full,
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
               packed_data[packed_offset + SymmetricIndex<PART>(i, j, n)];
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

template <TriangularPart PART>
void Mult(const TriPackMatrix<PART> &packed, const Vector &x, Vector &y)
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
      const real_t *Ae = A + e*TriPackMatrix<PART>::PackedSize(n);
      const real_t *Xe = X + e*n;
      real_t sum = 0.0;
      for (int j = 0; j < n; ++j)
      {
         sum += TriPackGet<PART>(Ae, i, j, n) * Xe[j];
      }
      Y[idx] = sum;
   });
}

void MultUUt(const TriPackMatrix<TriangularPart::UPPER> &packed_upper,
             const Vector &x, Vector &y)
{
   const int n = packed_upper.GetNumRows();
   const int batch_size = packed_upper.GetNumMatrices();
   MFEM_VERIFY(x.Size() == batch_size*n, "Input vector has the wrong size.");

   Vector t(batch_size*n);
   t.UseDevice(true);
   y.SetSize(batch_size*n);
   y.UseDevice(true);

   const int packed_size = packed_upper.GetPackedSize();
   const real_t *U = packed_upper.Data().Read();
   const real_t *X = x.Read();
   real_t *T = t.Write();
   real_t *Y = y.Write();

   mfem::forall(batch_size*n, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int r = idx % n;
      const int e = idx / n;
      const real_t *Ue = U + e*packed_size;
      const real_t *Xe = X + e*n;
      real_t sum = 0.0;
      for (int j = 0; j <= r; ++j)
      {
         sum += Ue[TriPackMatrix<TriangularPart::UPPER>::UpperIndex(j, r, n)] * Xe[j];
      }
      T[idx] = sum;
   });

   mfem::forall(batch_size*n, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int i = idx % n;
      const int e = idx / n;
      const real_t *Ue = U + e*packed_size;
      const real_t *Te = T + e*n;
      real_t sum = 0.0;
      for (int j = i; j < n; ++j)
      {
         sum += Ue[TriPackMatrix<TriangularPart::UPPER>::UpperIndex(i, j, n)] * Te[j];
      }
      Y[idx] = sum;
   });
}

template <TriangularPart PART>
void Lump(const TriPackMatrix<PART> &packed, Vector &lump)
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
      const real_t *Ae = A + e*TriPackMatrix<PART>::PackedSize(n);
      real_t sum = 0.0;
      for (int j = 0; j < n; ++j)
      {
         sum += TriPackGet<PART>(Ae, i, j, n);
      }
      L[idx] = sum;
   });
}

void ComputeJacobiScaledCholeskyUpper(
                                      const TriPackMatrix<TriangularPart::UPPER> &packed_upper,
                                      TriPackMatrix<TriangularPart::UPPER> &upper_factor,
                                      bool do_scale)
{
   const int n = packed_upper.GetNumRows();
   const int batch_size = packed_upper.GetNumMatrices();
   const int packed_size = packed_upper.GetPackedSize();

   MFEM_VERIFY(&packed_upper != &upper_factor,
               "Input and output TriPackMatrix objects must be distinct.");
   if (batch_size == 0)
   {
      upper_factor.SetSize(n, batch_size);
      return;
   }

   Vector scaled_factor, scaling;
   ComputeScaledCholeskyFactors(packed_upper, scaled_factor, scaling, do_scale);

   upper_factor.SetSize(n, batch_size);
   upper_factor.UseDevice(true);

   const real_t *R = scaled_factor.Read();
   const real_t *D = scaling.Read();
   real_t *U = upper_factor.Data().Write();

   mfem::forall(batch_size*packed_size, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int e = idx / packed_size;
      const int t = idx % packed_size;
      const int col = UpperColFromPackedIndex(t, n);

      U[idx] = R[idx] / D[e*n + col];
   });
}

void SolveUpper(const TriPackMatrix<TriangularPart::UPPER> &upper_factor,
                const Vector &rhs,
                Vector &sol)
{
   const int n = upper_factor.GetNumRows();
   const int batch_size = upper_factor.GetNumMatrices();
   const int packed_size = upper_factor.GetPackedSize();

   MFEM_VERIFY(rhs.Size() == batch_size*n, "Right-hand side has the wrong size.");

   Vector out(batch_size*n);
   out.UseDevice(true);

   const real_t *U = upper_factor.Data().Read();
   const real_t *B = rhs.Read();
   real_t *X = out.Write();

   mfem::forall(batch_size, [=] MFEM_HOST_DEVICE (int e)
   {
      const real_t *Ue = U + e*packed_size;
      const real_t *Be = B + e*n;
      real_t *Xe = X + e*n;

      for (int i = n - 1; i >= 0; --i)
      {
         real_t sum = Be[i];
         for (int j = i + 1; j < n; ++j)
         {
            sum -= Ue[TriPackMatrix<TriangularPart::UPPER>::UpperIndex(i, j, n)] * Xe[j];
         }
         Xe[i] = sum / Ue[TriPackMatrix<TriangularPart::UPPER>::UpperIndex(i, i, n)];
      }
   });

   sol.SetSize(batch_size*n);
   sol = out;
}

void SolveUpperTranspose(const TriPackMatrix<TriangularPart::UPPER> &upper_factor,
                         const Vector &rhs,
                         Vector &sol)
{
   const int n = upper_factor.GetNumRows();
   const int batch_size = upper_factor.GetNumMatrices();
   const int packed_size = upper_factor.GetPackedSize();

   MFEM_VERIFY(rhs.Size() == batch_size*n, "Right-hand side has the wrong size.");

   Vector out(batch_size*n);
   out.UseDevice(true);

   const real_t *U = upper_factor.Data().Read();
   const real_t *B = rhs.Read();
   real_t *X = out.Write();

   mfem::forall(batch_size, [=] MFEM_HOST_DEVICE (int e)
   {
      const real_t *Ue = U + e*packed_size;
      const real_t *Be = B + e*n;
      real_t *Xe = X + e*n;

      for (int i = 0; i < n; ++i)
      {
         real_t sum = Be[i];
         for (int j = 0; j < i; ++j)
         {
            sum -= Ue[TriPackMatrix<TriangularPart::UPPER>::UpperIndex(j, i, n)] * Xe[j];
         }
         Xe[i] = sum / Ue[TriPackMatrix<TriangularPart::UPPER>::UpperIndex(i, i, n)];
      }
   });

   sol.SetSize(batch_size*n);
   sol = out;
}

void SolveCholesky(const TriPackMatrix<TriangularPart::UPPER> &upper_factor,
                   const Vector &rhs,
                   Vector &sol)
{
   Vector tmp;
   SolveUpperTranspose(upper_factor, rhs, tmp);
   SolveUpper(upper_factor, tmp, sol);
}

void ComputeJacobiScaledCholeskyUpperInverse(
                                             const TriPackMatrix<TriangularPart::UPPER> &packed_upper,
                                             TriPackMatrix<TriangularPart::UPPER> &upper_inverse,
                                             bool do_scale,
                                             bool do_refine)
{
   const int n = packed_upper.GetNumRows();
   const int batch_size = packed_upper.GetNumMatrices();
   const int packed_size = packed_upper.GetPackedSize();
   const real_t nan = std::numeric_limits<real_t>::quiet_NaN();

   MFEM_VERIFY(&packed_upper != &upper_inverse,
               "Input and output TriPackMatrix objects must be distinct.");
   if (batch_size == 0) { return; }

   upper_inverse.SetSize(n, batch_size);
   upper_inverse.UseDevice(true);

   Vector factored;
   Vector work(batch_size*packed_size);
   Vector scaling;
   work.UseDevice(true);

   Vector refinement;
   if (do_refine)
   {
      refinement.SetSize(batch_size*packed_size);
      refinement.UseDevice(true);
   }

   ComputeScaledCholeskyFactors(packed_upper, factored, scaling, do_scale);

   const real_t *R = factored.Read();
   real_t *X = work.Write();
   const real_t *D = scaling.Read();
   real_t *E = do_refine ? refinement.Write() : nullptr;
   real_t *Uinv = upper_inverse.Data().Write();

   mfem::forall(batch_size, [=] MFEM_HOST_DEVICE (int e)
   {
      const int eoff = e*packed_size;
      const int doff = e*n;
      bool bad = false;

      for (int t = 0; t < packed_size; ++t)
      {
         if (!TriPackIsFinite(R[eoff + t]))
         {
            bad = true;
            break;
         }
      }

      if (bad)
      {
         for (int t = 0; t < packed_size; ++t) { Uinv[eoff + t] = nan; }
         return;
      }

      for (int t = 0; t < packed_size; ++t) { X[eoff + t] = 0.0; }

      for (int i = n - 1; i >= 0; --i)
      {
         const int ii = eoff + TriPackMatrix<TriangularPart::UPPER>::UpperIndex(i, i, n);
         X[ii] = 1.0/R[ii];

         const real_t invUii = X[ii];
         for (int j = i + 1; j < n; ++j)
         {
            real_t sum = 0.0;
            for (int k = i + 1; k <= j; ++k)
            {
               sum += R[eoff + TriPackMatrix<TriangularPart::UPPER>::UpperIndex(i, k, n)] *
                      X[eoff + TriPackMatrix<TriangularPart::UPPER>::UpperIndex(k, j, n)];
            }
            X[eoff + TriPackMatrix<TriangularPart::UPPER>::UpperIndex(i, j, n)] = -invUii*sum;
         }
      }

      if (do_refine)
      {
         for (int j = 0; j < n; ++j)
         {
            for (int i = 0; i <= j; ++i)
            {
               real_t sum = 0.0;
               for (int k = i; k <= j; ++k)
               {
                  sum += R[eoff + TriPackMatrix<TriangularPart::UPPER>::UpperIndex(i, k, n)] *
                         X[eoff + TriPackMatrix<TriangularPart::UPPER>::UpperIndex(k, j, n)];
               }
               E[eoff + TriPackMatrix<TriangularPart::UPPER>::UpperIndex(i, j, n)] =
                  ((i == j) ? 1.0 : 0.0) - sum;
            }
         }

         for (int i = n - 1; i >= 0; --i)
         {
            const real_t invRii =
               1.0/R[eoff + TriPackMatrix<TriangularPart::UPPER>::UpperIndex(i, i, n)];
            E[eoff + TriPackMatrix<TriangularPart::UPPER>::UpperIndex(i, i, n)] *= invRii;

            for (int j = i + 1; j < n; ++j)
            {
               real_t sum = 0.0;
               for (int k = i + 1; k <= j; ++k)
               {
                  sum += R[eoff + TriPackMatrix<TriangularPart::UPPER>::UpperIndex(i, k, n)] *
                         E[eoff + TriPackMatrix<TriangularPart::UPPER>::UpperIndex(k, j, n)];
               }
               E[eoff + TriPackMatrix<TriangularPart::UPPER>::UpperIndex(i, j, n)] =
                  (E[eoff + TriPackMatrix<TriangularPart::UPPER>::UpperIndex(i, j, n)] - sum) * invRii;
            }
         }

         for (int t = 0; t < packed_size; ++t)
         {
            X[eoff + t] += E[eoff + t];
         }
      }

      for (int j = 0; j < n; ++j)
      {
         for (int i = 0; i <= j; ++i)
         {
            const int t = TriPackMatrix<TriangularPart::UPPER>::UpperIndex(i, j, n);
            Uinv[eoff + t] = X[eoff + t] * D[doff + i];
         }
      }
   });
}

template bool CompareWithFull<TriangularPart::LOWER>(
   const TriPackMatrix<TriangularPart::LOWER> &, const Vector &, real_t);
template bool CompareWithFull<TriangularPart::UPPER>(
   const TriPackMatrix<TriangularPart::UPPER> &, const Vector &, real_t);

template void Mult<TriangularPart::LOWER>(
   const TriPackMatrix<TriangularPart::LOWER> &, const Vector &, Vector &);
template void Mult<TriangularPart::UPPER>(
   const TriPackMatrix<TriangularPart::UPPER> &, const Vector &, Vector &);

template void Lump<TriangularPart::LOWER>(
   const TriPackMatrix<TriangularPart::LOWER> &, Vector &);
template void Lump<TriangularPart::UPPER>(
   const TriPackMatrix<TriangularPart::UPPER> &, Vector &);

} // namespace tripack
} // namespace mfem
