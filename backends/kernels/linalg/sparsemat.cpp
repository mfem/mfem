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

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#include "../kernels.hpp"

namespace mfem
{

namespace kernels
{

KernelsSparseMatrix::KernelsSparseMatrix(Layout &in_layout, Layout &out_layout,
                                         const mfem::SparseMatrix &m) :
   Operator(in_layout, out_layout)
{
   push();
   Setup(in_layout.KernelsEngine().GetDevice(), m);
   pop();
}

KernelsSparseMatrix::KernelsSparseMatrix(Layout &in_layout, Layout &out_layout,
                                         const mfem::SparseMatrix &m,
                                         kernels::array<int> reorderIndices_,
                                         kernels::array<int> mappedIndices_) :
   Operator(in_layout, out_layout)
{
   push();
   Setup(in_layout.KernelsEngine().GetDevice(), m,
         reorderIndices, mappedIndices_);
   pop();
}

KernelsSparseMatrix::KernelsSparseMatrix(Layout &in_layout, Layout &out_layout,
                                         kernels::array<int> offsets_,
                                         kernels::array<int> indices_,
                                         kernels::array<double> weights_) :
   Operator(in_layout, out_layout),
   offsets(offsets_),
   indices(indices_),
   weights(weights_)
{
   push();
   SetupKernel(in_layout.KernelsEngine().GetDevice());
   pop();
}

KernelsSparseMatrix::KernelsSparseMatrix(Layout &in_layout, Layout &out_layout,
                                         kernels::array<int> offsets_,
                                         kernels::array<int> indices_,
                                         kernels::array<double> weights_,
                                         kernels::array<int> reorderIndices_,
                                         kernels::array<int> mappedIndices_) :
   Operator(in_layout, out_layout),
   offsets(offsets_),
   indices(indices_),
   weights(weights_),
   reorderIndices(reorderIndices_),
   mappedIndices(mappedIndices_)
{
   push();
   SetupKernel(in_layout.KernelsEngine().GetDevice());
   pop();
}

void KernelsSparseMatrix::Setup(kernels::device device,
                                const mfem::SparseMatrix &m)
{
   push();
   Setup(device, m, kernels::array<int>(), kernels::array<int>());
   pop();
}

void KernelsSparseMatrix::Setup(kernels::device device, const SparseMatrix &m,
                                kernels::array<int> reorderIndices_,
                                kernels::array<int> mappedIndices_)
{
   push();
   assert(false);
   reorderIndices = reorderIndices_;
   mappedIndices  = mappedIndices_;
   SetupKernel(device);
   pop();
}

void KernelsSparseMatrix::SetupKernel(kernels::device device)
{
   push();
   pop();
}

void KernelsSparseMatrix::Mult_(const Vector &x, Vector &y) const
{
   push();
   assert(false);
   if (reorderIndices.isInitialized() ||
       mappedIndices.isInitialized())
   {
      if (reorderIndices.isInitialized())
      {
         assert(false);/*
         mapKernel((int) (reorderIndices.size() / 2),
                   reorderIndices,
                   x.KernelsMem(), y.KernelsMem());*/
      }
      if (mappedIndices.isInitialized())
      {
         assert(false);/*
         multKernel((int) (mappedIndices.size()),
                    offsets, indices, weights,
                    mappedIndices,
                    x.KernelsMem(), y.KernelsMem());*/
      }
   }
   else
   {
      assert(false);/*
      multKernel((int) height,
                 offsets, indices, weights,
                 x.KernelsMem(), y.KernelsMem());*/
   }
   pop();
}


KernelsSparseMatrix* CreateMappedSparseMatrix(Layout &in_layout,
                                              Layout &out_layout,
                                              const mfem::SparseMatrix &m)
{
   push();
   const int mHeight = m.Height();
   // const int mWidth  = m.Width();

   // Count indices that are only reordered (true dofs)
   const int *I = m.GetI();
   const int *J = m.GetJ();
   const double *D = m.GetData();

   int trueCount = 0;
   for (int i = 0; i < mHeight; ++i)
   {
      trueCount += ((I[i + 1] - I[i]) == 1);
   }
   const int dupCount = (mHeight - trueCount);
   dbg(": mHeight=%d trueCount=%d", mHeight,trueCount);

   // Create the reordering map for entries that aren't modified (true dofs)
   kernels::device device(in_layout.KernelsEngine().GetDevice());
   kernels::array<int> reorderIndices(/*device,*/2*trueCount);
   kernels::array<int> mappedIndices, offsets, indices;
   kernels::array<double> weights;

   if (dupCount)
   {
      assert(false);/*
                      mappedIndices.allocate(device,dupCount);*/
   }
   int trueIdx = 0, dupIdx = 0;
   for (int i = 0; i < mHeight; ++i)
   {
      const int i1 = I[i];
      if ((I[i + 1] - i1) == 1)
      {
         reorderIndices[trueIdx++] = J[i1];
         reorderIndices[trueIdx++] = i;
      }
      else
      {
         mappedIndices[dupIdx++] = i;
      }
   }
   //reorderIndices.keepInDevice();

   if (dupCount)
   {
      //mappedIndices.keepInDevice();

      // Extract sparse matrix without reordered identity
      //const int dupNnz = I[mHeight] - trueCount;

      assert(false);/*
      offsets.allocate(device,dupCount + 1);
      indices.allocate(device,dupNnz);
      weights.allocate(device,dupNnz);*/

      int nnz = 0;
      offsets[0] = 0;
      for (int i = 0; i < dupCount; ++i)
      {
         const int idx = mappedIndices[i];
         const int offStart = I[idx];
         const int offEnd   = I[idx + 1];
         offsets[i + 1] = offsets[i] + (offEnd - offStart);
         for (int j = offStart; j < offEnd; ++j)
         {
            indices[nnz] = J[j];
            weights[nnz] = D[j];
            ++nnz;
         }
      }
      //offsets.keepInDevice();
      //indices.keepInDevice();
      //weights.keepInDevice();
   }
   KernelsSparseMatrix *nRSM = new KernelsSparseMatrix(in_layout, out_layout,
                                                       offsets, indices, weights,
                                                       reorderIndices, mappedIndices);
   pop();
   return nRSM;
}

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)
