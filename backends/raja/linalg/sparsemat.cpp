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
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#include "../raja.hpp"

namespace mfem
{

namespace raja
{

RajaSparseMatrix::RajaSparseMatrix(Layout &in_layout, Layout &out_layout,
                                   const mfem::SparseMatrix &m) :
   Operator(in_layout, out_layout)
{

   Setup(in_layout.RajaEngine().GetDevice(), m);
}

RajaSparseMatrix::RajaSparseMatrix(Layout &in_layout, Layout &out_layout,
                                   const mfem::SparseMatrix &m,
                                   raja::array<int> reorderIndices_,
                                   raja::array<int> mappedIndices_) :
   Operator(in_layout, out_layout)
{

   Setup(in_layout.RajaEngine().GetDevice(), m,
         reorderIndices, mappedIndices_);
}

RajaSparseMatrix::RajaSparseMatrix(Layout &in_layout, Layout &out_layout,
                                   raja::array<int> offsets_,
                                   raja::array<int> indices_,
                                   raja::array<double> weights_) :
   Operator(in_layout, out_layout),
   offsets(offsets_),
   indices(indices_),
   weights(weights_)
{

   SetupKernel(in_layout.RajaEngine().GetDevice());
}

RajaSparseMatrix::RajaSparseMatrix(Layout &in_layout, Layout &out_layout,
                                   raja::array<int> offsets_,
                                   raja::array<int> indices_,
                                   raja::array<double> weights_,
                                   raja::array<int> reorderIndices_,
                                   raja::array<int> mappedIndices_) :
   Operator(in_layout, out_layout),
   offsets(offsets_),
   indices(indices_),
   weights(weights_),
   reorderIndices(reorderIndices_),
   mappedIndices(mappedIndices_)
{

   SetupKernel(in_layout.RajaEngine().GetDevice());
}

void RajaSparseMatrix::Setup(raja::device device, const mfem::SparseMatrix &m)
{
   Setup(device, m, raja::array<int>(), raja::array<int>());
}

void RajaSparseMatrix::Setup(raja::device device, const SparseMatrix &m,
                             raja::array<int> reorderIndices_,
                             raja::array<int> mappedIndices_)
{

   //const int nnz = m.GetI()[height];
   assert(false);
   //offsets.allocate(device,height + 1, m.GetI());
   //indices.allocate(device,nnz, m.GetJ());
   //weights.allocate(device,nnz, m.GetData());
   //offsets.keepInDevice();
   //indices.keepInDevice();
   //weights.keepInDevice();

   reorderIndices = reorderIndices_;
   mappedIndices  = mappedIndices_;

   SetupKernel(device);
}

void RajaSparseMatrix::SetupKernel(raja::device device)
{
}

void RajaSparseMatrix::Mult_(const Vector &x, Vector &y) const
{
   assert(false);
   if (reorderIndices.isInitialized() ||
       mappedIndices.isInitialized())
   {
      if (reorderIndices.isInitialized())
      {
         assert(false);/*
         mapKernel((int) (reorderIndices.size() / 2),
                   reorderIndices,
                   x.RajaMem(), y.RajaMem());*/
      }
      if (mappedIndices.isInitialized())
      {
         assert(false);/*
         multKernel((int) (mappedIndices.size()),
                    offsets, indices, weights,
                    mappedIndices,
                    x.RajaMem(), y.RajaMem());*/
      }
   }
   else
   {
      assert(false);/*
      multKernel((int) height,
                 offsets, indices, weights,
                 x.RajaMem(), y.RajaMem());*/
   }
}


RajaSparseMatrix* CreateMappedSparseMatrix(Layout &in_layout,
                                           Layout &out_layout,
                                           const mfem::SparseMatrix &m)
{
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
   dbg("\n[CreateMappedSparseMatrix] mHeight=%d trueCount=%d, dupCount is %s",
       mHeight,trueCount,dupCount?"TRUE":"false");

   // Create the reordering map for entries that aren't modified (true dofs)
   raja::device device(in_layout.RajaEngine().GetDevice());
   raja::array<int> reorderIndices(/*device,*/2*trueCount);
   raja::array<int> mappedIndices, offsets, indices;
   raja::array<double> weights;

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

   return new RajaSparseMatrix(in_layout, out_layout,
                               offsets, indices, weights,
                               reorderIndices, mappedIndices);
}

} // namespace mfem::raja

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)
