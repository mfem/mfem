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

template <typename T>
class DeviceContainer
{
private:
   int capacity;
   T* data;

public:
   template <typename... Sizes> MFEM_HOST_DEVICE
   DeviceContainer(Sizes... sizes)
   : capacity(prod(sizes...)), data(nullptr)
   {
      data = new T[capacity];
   }

   MFEM_HOST_DEVICE
   T& operator[](const int i) const
   {
      return data[ i ];
   }

   const int size() const
   {
      return capacity;
   }
};

template <typename T, int... Dims>
class StaticSharedContainer
{
private:
   MFEM_SHARED T data[prod(Dims...)];
public:
   template <typename... Sizes> MFEM_HOST_DEVICE
   StaticSharedContainer(Sizes... sizes)
   {
      // static_assert(sizeof...(Dims)==sizeof...(Sizes), "Static and dynamic sizes don't match.");
      // TODO verify that Dims == sizes in Debug mode
   }

   MFEM_HOST_DEVICE
   const T& operator[](const int i) const
   {
      return data[ i ];
   }

   MFEM_HOST_DEVICE
   T& operator[](const int i)
   {
      return data[ i ];
   }

   constexpr int size() const
   {
      return prod(Dims...);
   }
};

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
   const T& operator[](const int i) const
   {
      return data[ i ];
   }

   MFEM_HOST_DEVICE
   T& operator[](const int i)
   {
      return data[ i ];
   }

   MFEM_HOST_DEVICE
   constexpr int size() const
   {
      return prod(Dims...);
   }
};

} // namespace mfem

#endif // MFEM_CONTAINER
