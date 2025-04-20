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

#ifndef MFEM_FES_KERNELS_HPP
#define MFEM_FES_KERNELS_HPP

#include "../general/forall.hpp"

namespace mfem
{
/// \cond DO_NOT_DOCUMENT
namespace internal
{

template <Ordering::Type Order, class Base, bool Diag = true>
struct DerefineMatrixOpFunctorBase;

template <class Base>
struct DerefineMatrixOpFunctorBase<Ordering::byNODES, Base, true>
{
   /// block col idcs offsets
   const int *bcptr;
   /// col idcs
   const int *cptr;

   int MFEM_HOST_DEVICE BlockWidth(int k) const
   {
      return bcptr[k + 1] - bcptr[k];
   }

   int MFEM_HOST_DEVICE Col(int j, int k) const { return cptr[bcptr[k] + j]; }

   int MFEM_HOST_DEVICE IndexX(int col, int vdim) const
   {
      return col + vdim * static_cast<const Base *>(this)->width;
   }
   int MFEM_HOST_DEVICE IndexY(int row, int vdim) const
   {
      return row + vdim * static_cast<const Base *>(this)->height;
   }
};

template <class Base>
struct DerefineMatrixOpFunctorBase<Ordering::byVDIM, Base, true>
{
   /// block col idcs offsets
   const int *bcptr;
   /// col idcs
   const int *cptr;

   int MFEM_HOST_DEVICE BlockWidth(int k) const
   {
      return bcptr[k + 1] - bcptr[k];
   }

   int MFEM_HOST_DEVICE Col(int j, int k) const { return cptr[bcptr[k] + j]; }

   int MFEM_HOST_DEVICE IndexX(int col, int vdim) const
   {
      return vdim + col * static_cast<const Base *>(this)->vdims;
   }
   int MFEM_HOST_DEVICE IndexY(int row, int vdim) const
   {
      return vdim + row * static_cast<const Base *>(this)->vdims;
   }
};

/// internally used to implementing the derefinement operator Mult diagonal
/// block
template <Ordering::Type Order, bool Atomic, bool Diag = true>
struct DerefineMatrixOpMultFunctor
   : public DerefineMatrixOpFunctorBase<
     Ordering::byNODES, DerefineMatrixOpMultFunctor<Order, Atomic, Diag>,
     Diag>
{
   const real_t *xptr;
   real_t *yptr;
   /// block storage
   const real_t *bsptr;
   /// block offsets
   const int *boptr;
   /// block row idcs offsets
   const int *brptr;
   /// row idcs
   const int *rptr;

   // number of blocks
   int nblocks;
   // number of components
   int vdims;
   /// overall operator height (for vdim = 1)
   int height;
   /// overall operator width (for vdim = 1)
   int width;
   void MFEM_HOST_DEVICE operator()(int kidx) const
   {
      int k = kidx % nblocks;
      int vdim = kidx / nblocks;

      int block_height = brptr[k + 1] - brptr[k];
      int block_width = this->BlockWidth(k);
      MFEM_FOREACH_THREAD(i, x, block_height)
      {
         int row = rptr[brptr[k] + i];
         int rsign = 1;
         if (row < 0)
         {
            row = -1 - row;
            rsign = -1;
         }
         if (row < height)
         {
            // row not marked as unused
            real_t sum = 0;
            for (int j = 0; j < block_width; ++j)
            {
               int col = this->Col(j, k);
               int sign = rsign;
               if (col < 0)
               {
                  col = -1 - col;
                  sign *= -1;
               }
               sum += sign * bsptr[boptr[k] + i + j * block_height] *
                      xptr[this->IndexX(col, vdim)];
            }
#if defined(__CUDA_ARCH__) or defined(__HIP_DEVICE_COMPILE__)
            if (Atomic)
            {
               atomicAdd(yptr + this->IndexY(row, vdim), sum);
            }
            else
#endif
            {
               yptr[this->IndexY(row, vdim)] += sum;
            }
         }
      }
   }

   /// N is the max block row size (doesn't have to be a power of 2)
   void Run(int N) const
   {
      forall_2D(nblocks * vdims, N, 1, *this);
   }
};

} // namespace internal
}

#endif
