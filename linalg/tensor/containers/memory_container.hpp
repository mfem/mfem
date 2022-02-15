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

#ifndef MFEM_MEMORY_CONTAINER
#define MFEM_MEMORY_CONTAINER

#include "container_traits.hpp"

namespace mfem
{

/// Owning Memory Container meant for storage on host.
template <typename T>
class MemoryContainer
{
private:
   Memory<T> data;

public:
   template <typename... Sizes>
   MemoryContainer(int size0, Sizes... sizes) : data(prod(size0,sizes...)) { }

   // MemoryContainer(const MemoryContainer &rhs)
   // {
   //    if(rhs.Capacity()>Capacity())
   //    {
   //       data.New(rhs.Capacity(), data.GetMemoryType());
   //    }
   //    auto ptr = data.Write();
   //    auto rhs_ptr = rhs.data.Read();
   //    MFEM_FORALL(i, Capacity(),{
   //       ptr[i] = rhs_ptr[i];
   //    });
   // }

   const T& operator[](int i) const
   {
      return data[ i ];
   }

   T& operator[](int i)
   {
      return data[ i ];
   }

   int Capacity() const
   {
      return data.Capacity();
   }

   ReadContainer<T> ReadData() const
   {
      return ReadContainer<T>(data.Read(), data.Capacity());
   }

   DeviceContainer<T> WriteData()
   {
      return DeviceContainer<T>(data.Write(), data.Capacity());
   }

   DeviceContainer<T> ReadWriteData()
   {
      return DeviceContainer<T>(data.ReadWrite(), data.Capacity());
   }
};

// get_container_type
template <typename T>
struct get_container_type_t<MemoryContainer<T>>
{
   using type = T;
};

// is_pointer_container
template <typename T>
struct is_pointer_container_v<MemoryContainer<T>>
{
   static constexpr bool value = true;
};

} // namespace mfem

#endif // MFEM_MEMORY_CONTAINER
