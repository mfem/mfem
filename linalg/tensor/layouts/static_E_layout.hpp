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

#ifndef MFEM_STATIC_E_LAYOUT
#define MFEM_STATIC_E_LAYOUT

#include "../details/util.hpp"
#include "layout_traits.hpp"
#include "restricted_layout.hpp"

namespace mfem
{

/// A static layout with the last dimension dynamic (typically for element index)
template<int... Sizes>
class StaticELayout
{
private:
   const int last_size;

public:
   template <typename... Dims> MFEM_HOST_DEVICE
   constexpr StaticELayout(int arg0, Dims... args)
   : last_size(GetLast(arg0, args...))
   {
      static_assert(sizeof...(Dims)==sizeof...(Sizes),
         "Static and dynamic sizes don't match.");
      // TODO verify that Dims == sizes in Debug mode
   }

   // MFEM_HOST_DEVICE
   // constexpr StaticELayout(const StaticELayout& rhs)
   // : last_size(rhs.last_size)
   // {
   //    // for (int i = 0; i < Rank; i++)
   //    // {
   //    //    MFEM_ASSERT(Sizes...[i] == lhs.Size<i>());
   //    // }
   // }

   // template <typename Layout> MFEM_HOST_DEVICE
   // constexpr StaticELayout(const Layout& rhs)
   // : last_size(rhs.template Size<sizeof...(Sizes)>())
   // {
   //    // for (int i = 0; i < Rank; i++)
   //    // {
   //    //    MFEM_ASSERT(Sizes...[i] == lhs.Size<i>());
   //    // }
   // }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   constexpr int index(Idx... idx) const
   {
      static_assert(sizeof...(Sizes)+1==sizeof...(Idx), "Wrong number of arguments.");
      return StaticELayoutIndex<Sizes...>::eval(idx...);
   }

   template <int N> MFEM_HOST_DEVICE inline
   int Size() const
   {
      static_assert(N>=0 && N<sizeof...(Sizes)+1,"Accessed size is higher than the rank of the Tensor.");
      // return N==sizeof...(Sizes) ? last_size : Dim<N,Sizes...>::val;
      return StaticELayoutSize<sizeof...(Sizes),N,Sizes...>::eval(last_size);
   }
};

// get_layout_rank
template <int... Dims>
struct get_layout_rank_v<StaticELayout<Dims...>>
{
   static constexpr int value = sizeof...(Dims)+1;
};

// is_static_layout
template<int... Dims>
struct is_static_layout_v<StaticELayout<Dims...>>
{
   static constexpr bool value = true;
};

// is_serial_layout
template<int... Dims>
struct is_serial_layout_v<StaticELayout<Dims...>>
{
   static constexpr bool value = true;
};

// get_layout_size
template <int N, int... Dims>
struct get_layout_size_v<N, StaticELayout<Dims...>>
{
   static constexpr int value = get_value<N, Dims...>;
};

// get_layout_sizes
template <int... Dims>
struct get_layout_sizes_t<StaticELayout<Dims...>>
{
   using type = int_list<Dims...,Dynamic>;
};

// get_layout_capacity
template <int... Sizes>
struct get_layout_capacity_v<
   RestrictedLayout<sizeof...(Sizes),StaticELayout<Sizes...>>>
{
   static constexpr int value = prod(Sizes...);
};

// get_layout_result_type
template <int... Sizes>
struct get_layout_result_type<StaticELayout<Sizes...>>
{
   template <int... Dims>
   using type = StaticLayout<Dims...>;
};

} // namespace mfem

#endif // MFEM_STATIC_E_LAYOUT
