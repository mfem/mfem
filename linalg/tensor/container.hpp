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

#ifndef MFEM_CONTAINER
#define MFEM_CONTAINER

#include "util.hpp"

namespace mfem
{

/// Non-owning modifiable Container that can be moved between host and device.
template <typename T>
class DeviceContainer
{
protected:
   T* data;

public:
   template <typename... Sizes> MFEM_HOST_DEVICE
   DeviceContainer(int size0, Sizes... sizes) : data(nullptr)
   {
      // static_assert(false,"Read Container are not supposed to be created like this");
   }

   MFEM_HOST_DEVICE
   DeviceContainer(T* data) : data(data)
   { }

   MFEM_HOST_DEVICE
   DeviceContainer(const DeviceContainer &rhs) : data(rhs.data)
   { }

   MFEM_HOST_DEVICE
   T& operator[](int i) const
   {
      return data[ i ];
   }
};

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
      // static_assert(sizeof...(Dims)==sizeof...(Sizes), "Static and dynamic sizes don't match.");
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

/// Statically sized owning Container distributed over a plane of threads.
/// TODO This should only be used on device in combination with BlockLayout.
template <typename T, int... Dims>
class BlockContainer;

/// 1D special case
template <typename T, int DimX>
class BlockContainer<T,DimX>
{
private:
   T data;

public:
   MFEM_HOST_DEVICE
   BlockContainer(int size) { /* TODO Verify that size < threadIdx.x */ }

   MFEM_HOST_DEVICE
   const T& operator[](int i) const
   {
      // TODO Verify in debug that i==0
      return data;
   }

   MFEM_HOST_DEVICE
   T& operator[](int i)
   {
      // TODO Verify in debug that i==0
      return data;
   }
};

/// 2D special case
template <typename T, int DimX, int DimY>
class BlockContainer<T,DimX, DimY>
{
private:
   T data;

public:
   MFEM_HOST_DEVICE
   BlockContainer(int size0, int size1) { }

   MFEM_HOST_DEVICE
   const T& operator[](int i) const
   {
      // TODO Verify in debug that i==0
      return data;
   }

   MFEM_HOST_DEVICE
   T& operator[](int i)
   {
      // TODO Verify in debug that i==0
      return data;
   }
};

/// 3D and more general case
template <typename T, int DimX, int DimY, int... Dims>
class BlockContainer<T,DimX,DimY,Dims...>
{
private:
   StaticContainer<T,Dims...> data;

public:
   template <typename... Sizes> MFEM_HOST_DEVICE
   BlockContainer(int size0, int size1, Sizes... sizes): data(sizes...) { }

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
template <typename T, typename Container>
class ViewContainer
{
private:
   // using T = get_container_type<Container>;
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
template <typename T, typename Container>
class ConstViewContainer
{
private:
   // using T = get_container_type<Container>;
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

////////////////////
// Container Traits

// get_container_type
template <typename Container>
struct get_container_type_t;

template <typename T>
struct get_container_type_t<DeviceContainer<T>>
{
   using type = T;
};

template <typename T>
struct get_container_type_t<ReadContainer<T>>
{
   using type = T;
};

template <typename T>
struct get_container_type_t<MemoryContainer<T>>
{
   using type = T;
};

template <typename T, int... Dims>
struct get_container_type_t<StaticContainer<T,Dims...>>
{
   using type = T;
};

template <typename T, int... Dims>
struct get_container_type_t<BlockContainer<T,Dims...>>
{
   using type = T;
};

template <typename T, typename Container>
struct get_container_type_t<ViewContainer<T,Container>>
{
   using type = T;
};

template <typename T, typename Container>
struct get_container_type_t<ConstViewContainer<T,Container>>
{
   using type = T;
};

template <typename Container>
using get_container_type = typename get_container_type_t<Container>::type;

// get_container_sizes
template <typename Container>
struct get_container_sizes_t;

template <typename T, int... Dims>
struct get_container_sizes_t<StaticContainer<T, Dims...>>
{
   using type = int_list<Dims...>;
};

template <typename T, int... Dims>
struct get_container_sizes_t<BlockContainer<T, Dims...>>
{
   using type = int_list<Dims...>;
};

template <typename T, typename Container>
struct get_container_sizes_t<ViewContainer<T, Container>>
{
   using type = typename get_container_sizes_t<Container>::type;
};

template <typename T, typename Container>
struct get_container_sizes_t<ConstViewContainer<T, Container>>
{
   using type = typename get_container_sizes_t<Container>::type;
};

template <typename Container>
using get_container_sizes = typename get_container_sizes_t<Container>::type;

// get_unsized_container
template <typename Container>
struct get_unsized_container;

template <typename T, int... Dims>
struct get_unsized_container<StaticContainer<T, Dims...>>
{
   template <int... Sizes>
   using type = StaticContainer<T, Sizes...>;
};

template <typename T, int... Dims>
struct get_unsized_container<BlockContainer<T, Dims...>>
{
   template <int... Sizes>
   using type = BlockContainer<T, Sizes...>;
};

template <typename T, typename Container>
struct get_unsized_container<ViewContainer<T, Container>>
{
   template <int... Sizes>
   using type = typename get_unsized_container<Container>::template type<Sizes...>;
};

template <typename T, typename Container>
struct get_unsized_container<ConstViewContainer<T, Container>>
{
   template <int... Sizes>
   using type = typename get_unsized_container<Container>::template type<Sizes...>;
};

// is_pointer_container
template <typename Container>
struct is_pointer_container_v
{
   static constexpr bool value = false;
};

template <typename T>
struct is_pointer_container_v<DeviceContainer<T>>
{
   static constexpr bool value = true;
};

template <typename T>
struct is_pointer_container_v<ReadContainer<T>>
{
   static constexpr bool value = true;
};

template <typename T>
struct is_pointer_container_v<MemoryContainer<T>>
{
   static constexpr bool value = true;
};

template <typename T, typename Container>
struct is_pointer_container_v<ViewContainer<T,Container>>
{
   static constexpr bool value = is_pointer_container_v<Container>::value;
};

template <typename T, typename Container>
struct is_pointer_container_v<ConstViewContainer<T,Container>>
{
   static constexpr bool value = is_pointer_container_v<Container>::value;
};

template <typename Tensor>
constexpr bool is_pointer_container = is_pointer_container_v<Tensor>::value;

} // namespace mfem

#endif // MFEM_CONTAINER
