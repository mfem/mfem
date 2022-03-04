// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_LOR_SPARSITY
#define MFEM_LOR_SPARSITY

#include "../bilinearform.hpp"

namespace mfem
{

/// @brief Class to help assembling CSR matrices associated with LOR
/// discretizations.
///
/// Some of the functionality in this class is similar to that of
/// ElementRestriction, but modified for the case of low-order refined elements,
/// where each "element matrix" is a sparse rather than dense matrix.
class LORSparsity
{
protected:
   const FiniteElementSpace &fes_ho; ///< Original high-order space.
public:
   /// Create the LORSparsity object given the high-order FE space.
   LORSparsity(const FiniteElementSpace &fes_ho_) : fes_ho(fes_ho_) { };

   /// Construct the CSR I array.
   int FillI(SparseMatrix &A, const DenseMatrix &sparse_mapping) const;

   /// @brief Construct the CSR J array and fill the matrix entries.
   ///
   /// The matrix entries are given by @a sparse_ij. The array @a sparse_ij is
   /// interpreted to have shape (nnz_per_row, ndof_per_el, nel_ho).
   /// This is essentiall a block diagonal matrix, with blocks corresponding to
   /// the high-order elements (the last index). Each block is itself a sparse
   /// matrix, where the row index is the second index (local DOF index within
   /// the element), and the column is determined by the first index. The first
   /// index maps to a local DOF index through the mapping @a sparse_mapping,
   /// which has shape shape (nnz_per_row, ndof_per_el). For example, for the
   /// entry (i, j, k) in @a sparse_ij, the local DOF index of the row is @a j,
   /// and the local DOF index of the column is sparse_mapping(i, j).
   void FillJAndData(SparseMatrix &A, const Vector &sparse_ij,
                     const DenseMatrix &sparse_mapping) const;

   SparseMatrix *FormCSR(const Vector &sparse_ij,
                         const DenseMatrix &sparse_mapping) const;

   /// @brief Internal method for constructing the sparsity pattern.
   ///
   /// @note This is not part of the public API, it is public only because of
   /// compiler restrictions (it contains MFEM_FORALL kernels).
   void Setup();
};

} // namespace mfem

#endif
