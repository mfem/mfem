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

#ifndef MFEM_READ_CONTAINER
#define MFEM_READ_CONTAINER

#include "container_traits.hpp"

namespace mfem
{
/// Non-owning const Container that can be moved between host and device.
template <typename T>
class ReadContainer
{
private:
   const T* data;

public:
   template <typename... Sizes> MFEM_HOST_DEVICE
   ReadContainer(int size0, Sizes... sizes) : data(nullptr)
   {
      // static_assert(false,"Read Container are not supposed to be created like this");
   }

   MFEM_HOST_DEVICE
   ReadContainer(const T* data) : data(data)
   { }

   MFEM_HOST_DEVICE
   ReadContainer(const ReadContainer &rhs) : data(rhs.data)
   { }

   MFEM_HOST_DEVICE
   const T& operator[](int i) const
   {
      return data[ i ];
   }
};

// get_container_type
template <typename T>
struct get_container_type_t<ReadContainer<T>>
{
   using type = T;
};

// is_pointer_container
template <typename T>
struct is_pointer_container_v<ReadContainer<T>>
{
   static constexpr bool value = true;
};

} // namespace mfem

#endif // MFEM_READ_CONTAINER
