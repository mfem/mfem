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

#ifndef MFEM_TENSOR_TYPES
#define MFEM_TENSOR_TYPES

#include "tensor.hpp"

namespace mfem
{

//////////////////////////
// Behavioral Tensor types

/// Dynamically sized Tensor
template <int Rank, typename T, int MaxSize = pow(DynamicMaxSize,Rank)>
using DynamicTensor = Tensor<StaticContainer<T, MaxSize>,
                             DynamicLayout<Rank> >;

template <int Rank, int MaxSize = pow(DynamicMaxSize,Rank)>
using DynamicDTensor = DynamicTensor<Rank,double,MaxSize>;

/// Statically sized Tensor
template <typename T, int... Sizes>
using StaticTensor = Tensor<StaticContainer<T, Sizes...>,
                            StaticLayout<Sizes...> >;

template <int... Sizes>
using dTensor = StaticTensor<double,Sizes...>; // TODO remove

template <int... Sizes>
using StaticDTensor = StaticTensor<double,Sizes...>;

/// A Tensor dynamically distributed over a plane of threads
constexpr int get_Dynamic2dThreadLayout_size(int MaxSize, int Rank)
{
   return Rank>2 ? pow(MaxSize,Rank-2) : 1;
}

template <int Rank, typename T, int BatchSize, int MaxSize = DynamicMaxSize>
using Dynamic2dThreadTensor = Tensor<
                              StaticContainer<
                                 T,
                                 get_Dynamic2dThreadLayout_size(MaxSize,Rank)
                              >,
                              Dynamic2dThreadLayout<Rank,BatchSize>
                           >;

template <int Rank, int BatchSize, int MaxSize = DynamicMaxSize>
using Dynamic2dThreadDTensor = Dynamic2dThreadTensor<Rank,double,BatchSize,MaxSize>;

/// A Tensor statically distributed over a plane of threads
template <typename T, int BatchSize, int... Sizes>
using Static2dThreadTensor = Tensor<Static2dThreadContainer<T, Sizes...>,
                                 Static2dThreadLayout<BatchSize, Sizes...> >;

template <int BatchSize, int... Sizes>
using Static2dThreadDTensor = Static2dThreadTensor<double,BatchSize,Sizes...>;

/// A Tensor dynamically distributed over a cube of threads
constexpr int get_Dynamic3dThreadLayout_size(int MaxSize, int Rank)
{
   return Rank>3 ? pow(MaxSize,Rank-3) : 1;
}

template <int Rank, typename T, int BatchSize, int MaxSize = DynamicMaxSize>
using Dynamic3dThreadTensor = Tensor<
                                 StaticContainer<
                                    T,
                                    get_Dynamic3dThreadLayout_size(MaxSize,Rank)
                                 >,
                                 Dynamic3dThreadLayout<Rank,BatchSize>
                              >;

template <int Rank, int BatchSize, int MaxSize = DynamicMaxSize>
using Dynamic3dThreadDTensor = Dynamic3dThreadTensor<Rank,double,BatchSize,MaxSize>;

/// A Tensor statically distributed over a plane of threads
constexpr int get_Static3dThreadTensor_size(int Size0)
{
   return 1;
}

constexpr int get_Static3dThreadTensor_size(int Size0, int Size1)
{
   return 1;
}

constexpr int get_Static3dThreadTensor_size(int Size0, int Size1, int Size2)
{
   return 1;
}

template <typename... Sizes>
constexpr int get_Static3dThreadTensor_size(int Size0, int Size1, int Size2,
                                            Sizes... sizes)
{
   return prod(sizes...);
}

template <typename T, int BatchSize, int... Sizes>
using Static3dThreadTensor = Tensor<StaticContainer<
                                       T,
                                       get_Static3dThreadTensor_size(Sizes...)
                                    >,
                                    Static3dThreadLayout<BatchSize, Sizes...>
                             >;

template <int BatchSize, int... Sizes>
using Static3dThreadDTensor = Static3dThreadTensor<double,BatchSize,Sizes...>;

/// A threaded tensor type with Dim threaded dimensions
template <int Dim>
struct ThreadTensor;

template <>
struct ThreadTensor<1>
{
   template <typename T, int BatchSize, int... Sizes>
   using static_type = Static2dThreadTensor<T,BatchSize,Sizes...>;

   template <int Rank, typename T, int BatchSize, int MaxSize = DynamicMaxSize>
   using dynamic_type = Dynamic2dThreadTensor<Rank,T,BatchSize,MaxSize>;
};

template <>
struct ThreadTensor<2>
{
   template <typename T, int BatchSize, int... Sizes>
   using static_type = Static2dThreadTensor<T,BatchSize,Sizes...>;

   template <int Rank, typename T, int BatchSize, int MaxSize = DynamicMaxSize>
   using dynamic_type = Dynamic2dThreadTensor<Rank,T,BatchSize,MaxSize>;
};

template <>
struct ThreadTensor<3>
{
   template <typename T, int BatchSize, int... Sizes>
   using static_type = Static3dThreadTensor<T,BatchSize,Sizes...>;

   template <int Rank, typename T, int BatchSize, int MaxSize = DynamicMaxSize>
   using dynamic_type = Dynamic3dThreadTensor<Rank,T,BatchSize,MaxSize>;
};

template <int Dim, typename T, int BatchSize, int... Sizes>
using StaticThreadTensor = typename ThreadTensor<Dim>
                              ::template static_type<T,BatchSize,Sizes...>;

template <int Dim, int Rank, typename T, int BatchSize, int MaxSize = DynamicMaxSize>
using DynamicThreadTensor = typename ThreadTensor<Dim>
                               ::template dynamic_type<Rank,T,BatchSize,MaxSize>;


/// A tensor using a read write access pointer and a dynamic data layout.
// Backward compatible if renamed in DeviceTensor
template <int Rank, typename T>
using MyDeviceTensor = Tensor<DeviceContainer<T>,
                              DynamicLayout<Rank> >;

template <int Rank>
using DeviceDTensor = MyDeviceTensor<Rank,double>;

template <typename T, int... Sizes>
using StaticPointerTensor = Tensor<DeviceContainer<T>,
                                   StaticLayout<Sizes...> >;

template <int... Sizes>
using StaticPointerDTensor = StaticPointerTensor<double,Sizes...>;

/// A tensor using a read only const pointer and a dynamic data layout.
template <int Rank, typename T>
using ReadTensor = Tensor<ReadContainer<T>,
                          DynamicLayout<Rank> >;

template <int Rank>
using ReadDTensor = ReadTensor<Rank,double>;

} // namespace mfem

#endif // MFEM_TENSOR_TYPES
