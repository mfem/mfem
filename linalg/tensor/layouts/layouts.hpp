// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_LAYOUTS
#define MFEM_LAYOUTS

/// A dynamically sized layout
#include "dynamic_layout.hpp"
/// A dynamically sized layout where the first dimension is threaded
#include "dynamic_1dthread_layout.hpp"
/// A dynamically sized layout where the two first dimensions are threaded
#include "dynamic_2dthread_layout.hpp"
/// A dynamically sized layout where the three first dimensions are threaded
#include "dynamic_3dthread_layout.hpp"
/// A statically sized layout
#include "static_layout.hpp"
/// A statically sized layout, except for the last dimension (used for E-vectors)
#include "static_E_layout.hpp"
/// A statically sized layout where the first dimension is threaded
#include "static_1dthread_layout.hpp"
/// A statically sized layout where the two first dimensions are threaded
#include "static_2dthread_layout.hpp"
/// A statically sized layout where the three first dimensions are threaded
#include "static_3dthread_layout.hpp"
/// A layout that removes one chosen dimension to another layout (used in Get)
#include "restricted_layout.hpp"

namespace mfem
{

// /// Strided Layout
// template <int Rank>
// class StridedLayout
// {
// private:
//    int strides[Rank];
//    int offsets[Rank];
//    int sizes[Rank];

// public:
//    template <typename... Idx> MFEM_HOST_DEVICE inline
//    constexpr int index(Idx... idx) const
//    {
//       static_assert(sizeof...(Idx)==Rank,"Wrong number of argumets.");
//       return StridedIndex<1>::eval(offsets, strides, idx...);
//    }

//    // Can be constexpr if Tensor inherit from Layout
//    template <int N> MFEM_HOST_DEVICE inline
//    int Size() const
//    {
//       static_assert(N>=0 && N<Rank,"Accessed size is higher than the rank of the Tensor.");
//       return sizes[N];
//    }

// private:
//    template <int N>
//    struct StridedIndex
//    {
//       template <typename... Idx>
//       static inline int eval(int* offsets, int* strides, int first, Idx... args)
//       {
//          return (offsets[N-1]+first)*strides[N-1] + StridedIndex<N+1>::eval(args...);
//       }
//    };

//    template <>
//    struct StridedIndex<Rank>
//    {
//       template <typename... Idx>
//       static inline int eval(int* offsets, int* strides, int first)
//       {
//          return (offsets[Rank-1]+first)*strides[Rank-1];
//       }
//    };
// };

} // namespace mfem

#endif // MFEM_LAYOUTS
