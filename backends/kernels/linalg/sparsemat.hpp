// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_BACKENDS_KERNELS_SPARSE_MAT_HPP
#define MFEM_BACKENDS_KERNELS_SPARSE_MAT_HPP

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

namespace mfem
{

namespace kernels
{

/// TODO: doxygen
class KernelsSparseMatrix : public Operator
{
public:
   kernels::array<int> offsets, indices;
   kernels::array<double> weights;
   kernels::array<int> reorderIndices, mappedIndices;
   //kernels::kernel mapKernel, multKernel;

   /// Construct an empty KernelsSparseMatrix.
   KernelsSparseMatrix(const Operator &orig)
      : Operator(orig) { }

   KernelsSparseMatrix(Layout &in_layout, Layout &out_layout,
                       const mfem::SparseMatrix &m);

   KernelsSparseMatrix(Layout &in_layout, Layout &out_layout,
                       const mfem::SparseMatrix &m,
                       kernels::array<int> reorderIndices_,
                       kernels::array<int> mappedIndices_);

   KernelsSparseMatrix(Layout &in_layout, Layout &out_layout,
                       kernels::array<int> offsets_,
                       kernels::array<int> indices_,
                       kernels::array<double> weights_);

   KernelsSparseMatrix(Layout &in_layout, Layout &out_layout,
                       kernels::array<int> offsets_,
                       kernels::array<int> indices_,
                       kernels::array<double> weights_,
                       kernels::array<int> reorderIndices_,
                       kernels::array<int> mappedIndices_);

   void Setup(kernels::device device, const mfem::SparseMatrix &m);

   void Setup(kernels::device device, const mfem::SparseMatrix &m,
              kernels::array<int> reorderIndices_,
              kernels::array<int> mappedIndices_);

   void SetupKernel(kernels::device device);

   // override
   virtual void Mult_(const Vector &x, Vector &y) const;
};


/// TODO: doxygen
KernelsSparseMatrix* CreateMappedSparseMatrix(
   Layout &in_layout, Layout &out_layout,
   const mfem::SparseMatrix &m);
} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#endif // MFEM_BACKENDS_KERNELS_SPARSE_MAT_HPP
