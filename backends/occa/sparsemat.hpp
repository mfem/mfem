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

#ifndef MFEM_BACKENDS_OCCA_SPARSE_MAT_HPP
#define MFEM_BACKENDS_OCCA_SPARSE_MAT_HPP

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)

#include <occa.hpp>
#include "vector.hpp"
#include "engine.hpp"
#include "operator.hpp"
#include "../../linalg/sparsemat.hpp"

namespace mfem
{

namespace occa
{

/// TODO: doxygen
class OccaSparseMatrix : public Operator
{
protected:
   ::occa::array<int> offsets, indices;
   ::occa::array<double> weights;
   ::occa::array<int> reorderIndices, mappedIndices;
   ::occa::kernel mapKernel, multKernel;

   void Setup(::occa::device device, const mfem::SparseMatrix &m,
              const ::occa::properties &props);

   void Setup(::occa::device device, const mfem::SparseMatrix &m,
              ::occa::array<int> reorderIndices_,
              ::occa::array<int> mappedIndices_,
              const ::occa::properties &props);

   void SetupKernel(::occa::device device,
                    const ::occa::properties &props);

public:
   /// Construct an empty OccaSparseMatrix.
   OccaSparseMatrix(const Operator &orig)
      : Operator(orig) { }

   // Implicitly defined copy constructor.

   OccaSparseMatrix(Layout &in_layout, Layout &out_layout,
                    const mfem::SparseMatrix &m,
                    const ::occa::properties &props = ::occa::properties());

   OccaSparseMatrix(Layout &in_layout, Layout &out_layout,
                    ::occa::array<int> offsets_,
                    ::occa::array<int> indices_,
                    ::occa::array<double> weights_,
                    ::occa::array<int> reorderIndices_,
                    ::occa::array<int> mappedIndices_,
                    const ::occa::properties &props = ::occa::properties());

   const ::occa::array<int> &GetReorderIndices() const
   { return reorderIndices; }

   // override
   virtual void Mult_(const Vector &x, Vector &y) const;
};


/// TODO: doxygen
OccaSparseMatrix* CreateMappedSparseMatrix(
   Layout &in_layout, Layout &out_layout,
   const mfem::SparseMatrix &m,
   const ::occa::properties &props = ::occa::properties());

} // namespace mfem::occa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)

#endif // MFEM_BACKENDS_OCCA_SPARSE_MAT_HPP
