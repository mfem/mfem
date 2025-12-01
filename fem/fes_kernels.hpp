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

#include <climits>

namespace mfem
{
/// \cond DO_NOT_DOCUMENT
namespace internal
{

///
/// Implements matrix-vector multiply $y = A x$ for a sparse matrix composed of
/// a sum of smaller dense blocks. There is additional permutation/sign
/// information associated with each block. The base class only implements
/// helper routines such as computing block widths, index into x, index into y,
/// and column in A given sub-block information.
/// @sa DerefineMatrixOpMultFunctor
///
/// @tparam Order vdim ordering for x and y. Note that for Diag = false this is
/// ignored for x as x has a special interleaved order.
/// @tparam Base used for the curious recurring template pattern (CRTP) so the
/// base class can access child class fields without virtual functions
/// @tparam Diag true if this corresponds to the diagonal block (coarse element
/// and fine element are on our rank), false otherwise (coarse element is on our
/// rank, fine element is on a different rank).
///
template <Ordering::Type Order, class Base, bool Diag = true>
struct DerefineMatrixOpFunctorBase;

template <class Base>
struct DerefineMatrixOpFunctorBase<Ordering::byNODES, Base, true>
{
   /// block column indices offsets
   const int *bcptr;
   /// column indices
   const int *cptr;

   int MFEM_HOST_DEVICE BlockWidth(int k) const
   {
      return bcptr[k + 1] - bcptr[k];
   }

   void MFEM_HOST_DEVICE Col(int j, int k, int &col, int &sign) const
   {
      col = cptr[bcptr[k] + j];
      if (col < 0)
      {
         col = -1 - col;
         sign = -sign;
      }
   }

   int MFEM_HOST_DEVICE IndexX(int col, int vdim, int) const
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
   /// block column indices offsets
   const int *bcptr;
   /// column indices
   const int *cptr;

   int MFEM_HOST_DEVICE BlockWidth(int k) const
   {
      return bcptr[k + 1] - bcptr[k];
   }

   void MFEM_HOST_DEVICE Col(int j, int k, int &col, int &sign) const
   {
      col = cptr[bcptr[k] + j];
      if (col < 0)
      {
         col = -1 - col;
         sign = -sign;
      }
   }

   int MFEM_HOST_DEVICE IndexX(int col, int vdim, int) const
   {
      return vdim + col * static_cast<const Base *>(this)->vdims;
   }
   int MFEM_HOST_DEVICE IndexY(int row, int vdim) const
   {
      return vdim + row * static_cast<const Base *>(this)->vdims;
   }
};

template <class Base>
struct DerefineMatrixOpFunctorBase<Ordering::byNODES, Base, false>
{
   /// receive segment offsets
   const int *segptr;
   /// receive segment index
   const int *rsptr;
   /// off-diagonal block column offsets
   const int *coptr;
   /// off-diagonal block widths
   const int *bwptr;

   int MFEM_HOST_DEVICE BlockWidth(int k) const { return bwptr[k]; }

   void MFEM_HOST_DEVICE Col(int j, int k, int &col, int &sign) const
   {
      col = coptr[k] + j;
   }

   int MFEM_HOST_DEVICE IndexX(int col, int vdim, int k) const
   {
      int tmp = rsptr[k];
      int segwidth = segptr[tmp + 1] - segptr[tmp];
      return segptr[tmp] * static_cast<const Base *>(this)->vdims + col +
             vdim * segwidth;
   }
   int MFEM_HOST_DEVICE IndexY(int row, int vdim) const
   {
      return row + vdim * static_cast<const Base *>(this)->height;
   }
};

template <class Base>
struct DerefineMatrixOpFunctorBase<Ordering::byVDIM, Base, false>
{
   /// receive segment offsets
   const int *segptr;
   /// receive segment index
   const int *rsptr;
   /// off-diagonal block column offsets
   const int *coptr;
   /// off-diagonal block widths
   const int *bwptr;

   int MFEM_HOST_DEVICE BlockWidth(int k) const { return bwptr[k]; }

   void MFEM_HOST_DEVICE Col(int j, int k, int &col, int &sign) const
   {
      col = coptr[k] + j;
   }

   int MFEM_HOST_DEVICE IndexX(int col, int vdim, int k) const
   {
      int tmp = rsptr[k];
      int segwidth = segptr[tmp + 1] - segptr[tmp];
      return segptr[tmp] * static_cast<const Base *>(this)->vdims + col +
             vdim * segwidth;
   }
   int MFEM_HOST_DEVICE IndexY(int row, int vdim) const
   {
      return vdim + row * static_cast<const Base *>(this)->vdims;
   }
};

/// internally used to implement the derefinement operator Mult diagonal
/// block
template <Ordering::Type Order, bool Atomic, bool Diag = true>
struct DerefineMatrixOpMultFunctor
   : public DerefineMatrixOpFunctorBase<
     Order, DerefineMatrixOpMultFunctor<Order, Atomic, Diag>, Diag>
{
   const real_t *xptr;
   real_t *yptr;
   /// block storage
   const real_t *bsptr;
   /// block offsets
   const int *boptr;
   /// block row index offsets
   const int *brptr;
   /// row indices
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
         if (row < INT_MAX)
         {
            // row not marked as unused
            real_t sum = 0;
            for (int j = 0; j < block_width; ++j)
            {
               int col, sign = rsign;
               this->Col(j, k, col, sign);
               sum += sign * bsptr[boptr[k] + i + j * block_height] *
                      xptr[this->IndexX(col, vdim, k)];
            }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
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
   void Run(int N) const { forall_2D(nblocks * vdims, N, 1, *this); }
};

} // namespace internal
/// \endcond DO_NOT_DOCUMENT
} // namespace mfem

#endif
