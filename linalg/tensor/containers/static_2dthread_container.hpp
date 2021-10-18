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

#ifndef MFEM_STATIC_2DTHREAD_CONTAINER
#define MFEM_STATIC_2DTHREAD_CONTAINER

namespace mfem
{

/// Statically sized owning Container distributed over a plane of threads.
/// TODO This should only be used on device in combination with Static2dThreadLayout.
template <typename T, int... Dims>
class Static2dThreadContainer;

/// 1D special case
template <typename T, int DimX>
class Static2dThreadContainer<T,DimX>
{
private:
   T data;

public:
   MFEM_HOST_DEVICE
   Static2dThreadContainer()
   {
      // TODO verify that DimX <= threadIdx.x
   }

   MFEM_HOST_DEVICE
   Static2dThreadContainer(int size) { /* TODO Verify that size < threadIdx.x */ }

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
class Static2dThreadContainer<T,DimX, DimY>
{
private:
   T data;

public:
   MFEM_HOST_DEVICE
   Static2dThreadContainer()
   {
      // TODO verify that DimX <= threadIdx.x
      // TODO verify that DimY <= threadIdx.y
   }

   MFEM_HOST_DEVICE
   Static2dThreadContainer(int size0, int size1) { }

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
class Static2dThreadContainer<T,DimX,DimY,Dims...>
{
private:
   StaticContainer<T,Dims...> data;

public:
   MFEM_HOST_DEVICE
   Static2dThreadContainer(): data()
   {
      // TODO verify that DimX <= threadIdx.x
      // TODO verify that DimY <= threadIdx.y
   }

   template <typename... Sizes> MFEM_HOST_DEVICE
   Static2dThreadContainer(int size0, int size1, Sizes... sizes): data(sizes...) { }

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
struct get_container_type_t<Static2dThreadContainer<T,Dims...>>
{
   using type = T;
};

// get_container_sizes
template <typename T, int... Dims>
struct get_container_sizes_t<Static2dThreadContainer<T, Dims...>>
{
   using type = int_list<Dims...>;
};

// get_unsized_container
template <typename T, int... Dims>
struct get_unsized_container<Static2dThreadContainer<T, Dims...>>
{
   template <int... Sizes>
   using type = Static2dThreadContainer<T, Sizes...>;
};

} // namespace mfem

#endif // MFEM_STATIC_2DTHREAD_CONTAINER
