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

#ifndef MFEM_LAYOUT
#define MFEM_LAYOUT

#include "dynamic_layout.hpp"
#include "dynamic_2dthread_layout.hpp"
#include "dynamic_3dthread_layout.hpp"
#include "static_layout.hpp"
#include "static_E_layout.hpp"
#include "static_2dthread_layout.hpp"
#include "static_3dthread_layout.hpp"
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

/// Layout for Nedelec finite elements dofs TODO
class NDLayout;

/// Layout for Raviart-Thomas finite elements dofs TODO
class RTLayout;

} // namespace mfem

#endif // MFEM_LAYOUT
