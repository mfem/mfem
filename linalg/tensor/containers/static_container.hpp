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

#ifndef MFEM_STATIC_CONTAINER
#define MFEM_STATIC_CONTAINER

#include "container_traits.hpp"
#include "../utilities/util.hpp"
#include "../utilities/int_list.hpp"
#include "../utilities/prod.hpp"

namespace mfem
{

/// Owning Container statically sized.
template <typename T, int... Dims>
class StaticContainer
{
private:
   T data[prod(Dims...)];

public:
   MFEM_HOST_DEVICE
   StaticContainer() { }

   template <typename... Sizes> MFEM_HOST_DEVICE
   StaticContainer(int size0, Sizes... sizes)
   {
      // static_assert(
      //    sizeof...(Dims)==sizeof...(Sizes)+1,
      //    "Wrong number of dynamic sizes.");
      // TODO verify that Dims == sizes in Debug mode
   }

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

// get_container_type
template <typename T, int... Dims>
struct get_container_type_t<StaticContainer<T,Dims...>>
{
   using type = T;
};

// get_container_sizes
template <typename T, int... Dims>
struct get_container_sizes_t<StaticContainer<T, Dims...>>
{
   using type = int_list<Dims...>;
};

// get_unsized_container
template <typename T, int... Dims>
struct get_unsized_container<StaticContainer<T, Dims...>>
{
   template <int... Sizes>
   using type = StaticContainer<T, Sizes...>;
};

} // namespace mfem

#endif // MFEM_STATIC_CONTAINER
