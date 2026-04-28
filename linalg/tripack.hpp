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

#ifndef MFEM_TRIPACK
#define MFEM_TRIPACK

#include "../config/config.hpp"
#include "vector.hpp"

namespace mfem
{

/// Select the triangular portion stored in packed matrices.
enum class TriangularPart
{
   LOWER,
   UPPER
};

/// Packed storage for a batch of triangular matrices of fixed size.
template <TriangularPart PART>
class TriPackMatrix
{
private:
   Vector data;
   int nrows = 0;
   int nmats = 0;

public:
   static constexpr TriangularPart Part = PART;

   TriPackMatrix() = default;

   TriPackMatrix(int n, int batch_size)
   {
      SetSize(n, batch_size);
   }

   MFEM_HOST_DEVICE static int PackedSize(const int n)
   {
      return n*(n + 1)/2;
   }

   MFEM_HOST_DEVICE static int LowerIndex(const int i, const int j)
   {
      return i*(i + 1)/2 + j;
   }

   MFEM_HOST_DEVICE static int UpperIndex(const int i, const int j,
                                          const int n)
   {
      return i*n - i*(i - 1)/2 + (j - i);
   }

   MFEM_HOST_DEVICE static int Index(const int i, const int j, const int n,
                                     const TriangularPart p)
   {
      return (p == TriangularPart::LOWER) ?
             LowerIndex(i, j) : UpperIndex(i, j, n);
   }

   void SetSize(const int n, const int batch_size)
   {
      nrows = n;
      nmats = batch_size;
      data.SetSize(batch_size*PackedSize(n));
   }

   static constexpr TriangularPart GetTriangularPart() { return PART; }

   int GetNumRows() const { return nrows; }

   int GetNumMatrices() const { return nmats; }

   int GetPackedSize() const { return PackedSize(nrows); }

   int Size() const { return data.Size(); }

   void UseDevice(bool use_dev) { data.UseDevice(use_dev); }

   TriPackMatrix &operator=(real_t value)
   {
      data = value;
      return *this;
   }

   Vector &Data() { return data; }
   const Vector &Data() const { return data; }
};

namespace tripack
{

template <TriangularPart PART>
bool CompareWithFull(const TriPackMatrix<PART> &packed, const Vector &full,
                     real_t tol = 0.0);

template <TriangularPart PART>
void Mult(const TriPackMatrix<PART> &packed, const Vector &x, Vector &y);

void MultUUt(const TriPackMatrix<TriangularPart::UPPER> &packed_upper,
             const Vector &x, Vector &y);

template <TriangularPart PART>
void Lump(const TriPackMatrix<PART> &packed, Vector &lump);

void ComputeJacobiScaledCholeskyUpper(
   const TriPackMatrix<TriangularPart::UPPER> &packed_upper,
   TriPackMatrix<TriangularPart::UPPER> &upper_factor,
                                      bool do_scale = true);

void SolveUpper(const TriPackMatrix<TriangularPart::UPPER> &upper_factor,
                const Vector &rhs,
                Vector &sol);

void SolveUpperTranspose(const TriPackMatrix<TriangularPart::UPPER> &upper_factor,
                         const Vector &rhs,
                         Vector &sol);

void SolveCholesky(const TriPackMatrix<TriangularPart::UPPER> &upper_factor,
                   const Vector &rhs,
                   Vector &sol);

void ComputeJacobiScaledCholeskyUpperInverse(
                                             const TriPackMatrix<TriangularPart::UPPER> &packed_upper,
                                             TriPackMatrix<TriangularPart::UPPER> &upper_inverse,
                                             bool do_scale = true,
                                             bool do_refine = true);

} // namespace tripack

} // namespace mfem

#endif
