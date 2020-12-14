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
   int capacity;

public:
   MFEM_HOST_DEVICE
   DeviceContainer(const T* data, int capacity) : data(data), capacity(capacity)
   { }

   MFEM_HOST_DEVICE
   DeviceContainer(const DeviceContainer &rhs) : data(rhs.data), capacity(rhs.capacity)
   { }

   MFEM_HOST_DEVICE
   T& operator[](int i) const
   {
      return data[ i ];
   }

   const int Capacity() const
   {
      return capacity;
   }
};

/// Non-owning const Container that can be moved between host and device.
template <typename T>
class ReadContainer
{
private:
   const T* data;
   int capacity;

public:
   MFEM_HOST_DEVICE
   ReadContainer(const T* data, int capacity) : data(data), capacity(capacity)
   { }

   MFEM_HOST_DEVICE
   ReadContainer(const ReadContainer &rhs) : data(rhs.data), capacity(rhs.capacity)
   { }

   MFEM_HOST_DEVICE
   const T& operator[](int i) const
   {
      return data[ i ];
   }

   const int Capacity() const
   {
      return capacity;
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
   MemoryContainer(Sizes... sizes) : data(prod(sizes...)) { }

   MemoryContainer(const MemoryContainer &rhs)
   {
      if(rhs.Capacity>Capacity())
      {
         data.New(rhs.Capacity(), data.GetMemoryType());
      }
      auto ptr = data.Write();
      auto rhs_ptr = rhs.data.Read();
      MFEM_FORALL(i, Capacity(),{
         ptr[i] = rhs_ptr[i];
      });
   }

   const T& operator[](int i) const
   {
      return data[ i ];
   }

   T& operator[](int i)
   {
      return data[ i ];
   }

   const int Capacity() const
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
   template <typename... Sizes> MFEM_HOST_DEVICE
   StaticContainer(Sizes... sizes)
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

   MFEM_HOST_DEVICE
   constexpr int Capacity() const
   {
      return prod(Dims...);
   }
};

/// Owning Container using shared memory on device and statically sized.
template <typename T, int... Dims>
class StaticSharedContainer
{
private:
// TODO might have to take into account some sort of number of elements
   MFEM_SHARED T data[prod(Dims...)];
public:
   template <typename... Sizes> MFEM_HOST_DEVICE
   StaticSharedContainer(Sizes... sizes)
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

   constexpr int Capacity() const
   {
      return prod(Dims...);
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

   MFEM_HOST_DEVICE
   constexpr int Capacity() const
   {
      return 1;
   }
};

/// 2D special case
template <typename T, int DimX, int DimY>
class BlockContainer<T,DimX, DimY>
{
private:
   StaticContainer<T,1> data;

public:
   MFEM_HOST_DEVICE
   BlockContainer(int size0, int size1) { }

   MFEM_HOST_DEVICE
   const T& operator[](int i) const
   {
      // TODO Verify in debug that i==0
      return data[ 0 ];
   }

   MFEM_HOST_DEVICE
   T& operator[](int i)
   {
      // TODO Verify in debug that i==0
      return data[ 0 ];
   }

   MFEM_HOST_DEVICE
   constexpr int Capacity() const
   {
      return 1;
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

   MFEM_HOST_DEVICE
   constexpr int Capacity() const
   {
      return prod(Dims...);
   }
};

/// A view Container
template <typename T, typename Container>
class ViewContainer
{
private:
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

   MFEM_HOST_DEVICE
   constexpr int Capacity() const
   {
      return data.Capacity();
   }
};

/// A view Container
template <typename T, typename Container>
class ConstViewContainer
{
private:
   const Container &data;

public:
   MFEM_HOST_DEVICE
   ConstViewContainer(const Container &data): data(data) { }

   MFEM_HOST_DEVICE
   const T& operator[](int i) const
   {
      return data[ i ];
   }

   MFEM_HOST_DEVICE
   constexpr int Capacity() const
   {
      return data.Capacity();
   }
};

} // namespace mfem

#endif // MFEM_CONTAINER
