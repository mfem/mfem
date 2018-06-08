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

#ifndef MFEM_BACKENDS_RAJA_SPARSE_MAT_HPP
#define MFEM_BACKENDS_RAJA_SPARSE_MAT_HPP

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#include "../raja.hpp"
#include "../linalg/vector.hpp"
#include "../engine/engine.hpp"
#include "../linalg/operator.hpp"
#include "../../../linalg/sparsemat.hpp"

namespace mfem
{

namespace raja
{

/// TODO: doxygen
class RajaSparseMatrix : public Operator
{
public:
   raja::array<int> offsets, indices;
   raja::array<double> weights;
   raja::array<int> reorderIndices, mappedIndices;
   //raja::kernel mapKernel, multKernel;

   /// Construct an empty RajaSparseMatrix.
   RajaSparseMatrix(const Operator &orig)
      : Operator(orig) { }

   RajaSparseMatrix(Layout &in_layout, Layout &out_layout,
                    const mfem::SparseMatrix &m);

   RajaSparseMatrix(Layout &in_layout, Layout &out_layout,
                    const mfem::SparseMatrix &m,
                    raja::array<int> reorderIndices_,
                    raja::array<int> mappedIndices_);

   RajaSparseMatrix(Layout &in_layout, Layout &out_layout,
                    raja::array<int> offsets_,
                    raja::array<int> indices_,
                    raja::array<double> weights_);

   RajaSparseMatrix(Layout &in_layout, Layout &out_layout,
                    raja::array<int> offsets_,
                    raja::array<int> indices_,
                    raja::array<double> weights_,
                    raja::array<int> reorderIndices_,
                    raja::array<int> mappedIndices_);

   void Setup(raja::device device, const mfem::SparseMatrix &m);

   void Setup(raja::device device, const mfem::SparseMatrix &m,
              raja::array<int> reorderIndices_,
              raja::array<int> mappedIndices_);

   void SetupKernel(raja::device device);

   // override
   virtual void Mult_(const Vector &x, Vector &y) const;
};


/// TODO: doxygen
RajaSparseMatrix* CreateMappedSparseMatrix(
   Layout &in_layout, Layout &out_layout,
   const mfem::SparseMatrix &m);
} // namespace mfem::raja

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#endif // MFEM_BACKENDS_RAJA_SPARSE_MAT_HPP
