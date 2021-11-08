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

#ifndef MFEM_VIEW_CONTAINER
#define MFEM_VIEW_CONTAINER

#include "container_traits.hpp"

namespace mfem
{

/// A view Container
template <typename Container>
class ViewContainer
{
private:
   using T = get_container_type<Container>;
   Container &data;

public:
   MFEM_HOST_DEVICE
   ViewContainer(Container &data): data(data) { }

   MFEM_HOST_DEVICE
   const T& operator[](int i) const
   {
      return data[ i ];
   }

   MFEM_HOST_DEVICE
   T& operator[](int i)
   {
      return data[ i ];
   }
};

/// A view Container
template <typename Container>
class ConstViewContainer
{
private:
   using T = get_container_type<Container>;
   const Container &data;

public:
   MFEM_HOST_DEVICE
   ConstViewContainer(const Container &data): data(data) { }

   MFEM_HOST_DEVICE
   const T& operator[](int i) const
   {
      return data[ i ];
   }
};

// get_container_type
template <typename Container>
struct get_container_type_t<ViewContainer<Container>>
{
   using type = get_container_type<Container>;
};

template <typename Container>
struct get_container_type_t<ConstViewContainer<Container>>
{
   using type = get_container_type<Container>;
};

// get_container_sizes
template <typename Container>
struct get_container_sizes_t<ViewContainer<Container>>
{
   using type = typename get_container_sizes_t<Container>::type;
};

template <typename Container>
struct get_container_sizes_t<ConstViewContainer<Container>>
{
   using type = typename get_container_sizes_t<Container>::type;
};

// get_unsized_container
template <typename Container>
struct get_unsized_container<ViewContainer<Container>>
{
   template <int... Sizes>
   using type = typename get_unsized_container<Container>::template type<Sizes...>;
};

template <typename Container>
struct get_unsized_container<ConstViewContainer<Container>>
{
   template <int... Sizes>
   using type = typename get_unsized_container<Container>::template type<Sizes...>;
};

// is_pointer_container
template <typename Container>
struct is_pointer_container_v<ViewContainer<Container>>
{
   static constexpr bool value = is_pointer_container<Container>;
};

template <typename Container>
struct is_pointer_container_v<ConstViewContainer<Container>>
{
   static constexpr bool value = is_pointer_container<Container>;
};

} // namespace mfem

#endif // MFEM_VIEW_CONTAINER
