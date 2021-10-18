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

#ifndef MFEM_DYNAMIC_LAYOUT
#define MFEM_DYNAMIC_LAYOUT

#include "../details/util.hpp"
#include "layout_traits.hpp"

namespace mfem
{

/// A dynamic layout with first index fastest
template <int Rank>
class DynamicLayout
{
private:
   int sizes[Rank];

public:
   template <typename... Sizes> MFEM_HOST_DEVICE
   DynamicLayout(int arg0, Sizes... args)
   {
      InitDynamicLayout<Rank>::result(sizes, arg0, args...);
   }

   template <typename Layout> MFEM_HOST_DEVICE
   DynamicLayout(const Layout &rhs)
   {
      InitDynamicLayout<Rank>::result(sizes,rhs);
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   int index(Idx... idx) const
   {
      static_assert(Rank==sizeof...(Idx), "Wrong number of arguments.");
      return DynamicLayoutIndex<Rank>::eval(sizes, idx...);
   }

   template <int N> MFEM_HOST_DEVICE inline
   int Size() const
   {
      static_assert(N>=0 && N<Rank,"Accessed size is higher than the rank of the Tensor.");
      return sizes[N];
   }
};

// get_layout_rank
template <int Rank>
struct get_layout_rank_v<DynamicLayout<Rank>>
{
   static constexpr int value = Rank;
};

// is_dynamic_layout
template<int Rank>
struct is_dynamic_layout_v<DynamicLayout<Rank>>
{
   static constexpr bool value = true;
};

// is_serial_layout
template <int Rank>
struct is_serial_layout_v<DynamicLayout<Rank>>
{
   static constexpr bool value = true;
};

// get_layout_size
template <int N, int Rank>
struct get_layout_size_v<N, DynamicLayout<Rank>>
{
   static constexpr int value = Dynamic;
};

// get_layout_sizes
template <int Rank>
struct get_layout_sizes_t<DynamicLayout<Rank>>
{
   using type = int_repeat<Dynamic,Rank>;
};

// get_layout_result_type
template <int Rank>
struct get_layout_result_type<DynamicLayout<Rank>>
{
   template <int myRank>
   using type = DynamicLayout<myRank>;
};

} // namespace mfem

#endif // MFEM_DYNAMIC_LAYOUT
