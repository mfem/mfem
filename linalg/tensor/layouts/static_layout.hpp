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

#ifndef MFEM_STATIC_LAYOUT
#define MFEM_STATIC_LAYOUT

#include "layout_traits.hpp"

namespace mfem
{

/// A static layout
template <int... Sizes>
class StaticLayout
{
public:
   MFEM_HOST_DEVICE
   constexpr StaticLayout() { }

   template <typename... Dims> MFEM_HOST_DEVICE
   constexpr StaticLayout(Dims... args)
   {
      static_assert(
         sizeof...(Dims)==sizeof...(Sizes),
         "Static and dynamic sizes don't match.");
      // TODO verify that Dims == sizes in Debug mode
   }

   template <typename Layout> MFEM_HOST_DEVICE
   constexpr StaticLayout(const Layout& rhs)
   {
      // for (int i = 0; i < Rank; i++)
      // {
      //    MFEM_ASSERT(Sizes...[i] == lhs.Size<i>());
      // }
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   constexpr int index(Idx... idx) const
   {
      static_assert(
         sizeof...(Sizes)==sizeof...(Idx),
         "Wrong number of arguments.");
      return StaticLayoutIndex<Sizes...>::eval(idx...);
   }

   template <int N> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      static_assert(
         N>=0 && N<sizeof...(Sizes),
         "Accessed size is higher than the rank of the Tensor.");
      return get_value<N,Sizes...>;
   }
};

// get_layout_rank
template <int... Dims>
struct get_layout_rank_v<StaticLayout<Dims...>>
{
   static constexpr int value = sizeof...(Dims);
};

// is_static_layout
template<int... Dims>
struct is_static_layout_v<StaticLayout<Dims...>>
{
   static constexpr bool value = true;
};

// is_serial_layout
template<int... Dims>
struct is_serial_layout_v<StaticLayout<Dims...>>
{
   static constexpr bool value = true;
};

// get_layout_size
template <int N, int... Dims>
struct get_layout_size_v<N, StaticLayout<Dims...>>
{
   static constexpr int value = get_value<N, Dims...>;
};

// get_layout_sizes
template <int... Dims>
struct get_layout_sizes_t<StaticLayout<Dims...>>
{
   using type = int_list<Dims...>;
};

// get_layout_capacity
template <int... Sizes>
struct get_layout_capacity_v<StaticLayout<Sizes...>>
{
   static constexpr int value = prod(Sizes...);
};

// get_layout_result_type
template <int... Sizes>
struct get_layout_result_type_t<StaticLayout<Sizes...>>
{
   template <int... Dims>
   using type = StaticLayout<Dims...>;
};

} // namespace mfem

#endif // MFEM_STATIC_LAYOUT
