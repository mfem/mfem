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

#ifndef MFEM_TENSOR
#define MFEM_TENSOR

#include "../../general/backends.hpp"
#include "container.hpp"
#include "layout.hpp"
#include "util.hpp"
#include "tensor_traits.hpp"
#include "foreach.hpp"
#include <iostream>

namespace mfem
{

/** A tensor class
    @a Container is the type of data container, they can either be statically or
       dynamically allocated,
    @a Layout is a class that represents the data layout
       There is two main sub-categories of Layout, Static and Dynamic layouts.
       Dynamic Layout have the following signature:
       template <int Rank>,
       Static Layout have the following signature:
       template <int... Sizes>,
       where Sizes... is the list of the sizes of the dimensions of the Tensor.
   */
template <typename Container,
          typename Layout>
class Tensor: public Container, public Layout
{
public:
   using T = get_container_type<Container>;
   using container = Container;
   using layout = Layout;

   /// Default Constructor
   MFEM_HOST_DEVICE
   Tensor() : Container(), Layout() { }

   /// Main Constructor
   template <typename... Sizes> MFEM_HOST_DEVICE
   Tensor(int size0, Sizes... sizes)
   : Container(size0,sizes...), Layout(size0,sizes...) { }

   /// Utility Constructor
   MFEM_HOST_DEVICE
   Tensor(Layout index): Container(), Layout(index) { }

   MFEM_HOST_DEVICE
   Tensor(Container data, Layout index): Container(data), Layout(index) { }

   template <typename... Sizes> MFEM_HOST_DEVICE
   Tensor(T* ptr, Sizes... sizes)
   : Container(ptr), Layout(sizes...) { }

   /// Copy Constructor
   MFEM_HOST_DEVICE
   Tensor(const Tensor &rhs): Container(rhs), Layout(rhs) { }

   // TODO default constructor for Container?
   template <typename OtherTensor> MFEM_HOST_DEVICE
   Tensor(const OtherTensor &rhs): Container(), Layout(rhs)
   {
      ForallDims<Tensor>::ApplyBinOp(*this, rhs, [&](auto... idx)
      {
         (*this)(idx...) = rhs(idx...);
      });
   }

   /// Accessor
   template <typename... Idx> MFEM_HOST_DEVICE inline
   T& operator()(Idx... args)
   {
      static_assert(get_layout_rank<Layout> == sizeof...(Idx),
                    "Wrong number of indices");
      return this->operator[]( this->index(args...) );
   }

   /// Const Accessor
   template <typename... Idx> MFEM_HOST_DEVICE inline
   const T& operator()(Idx... args) const
   {
      static_assert(get_layout_rank<Layout> == sizeof...(Idx),
                    "Wrong number of indices");
      return this->operator[]( this->index(args...) );
   }

   /// Initialization of a Tensor to a constant value.
   MFEM_HOST_DEVICE inline
   Tensor<Container,Layout>& operator=(const T &val)
   {
      ForallDims<Tensor>::Apply(*this, [&](auto... idx)
      {
         (*this)(idx...) = val;
      });
      return *this;
   }

   template <typename OtherTensor> MFEM_HOST_DEVICE inline
   Tensor<Container,Layout>& operator=(const OtherTensor &rhs)
   {
      ForallDims<Tensor>::ApplyBinOp(*this, rhs, [&](auto... idx)
      {
         (*this)(idx...) = rhs(idx...);
      });
      return *this;
   }

   template <typename OtherTensor> MFEM_HOST_DEVICE inline
   Tensor<Container,Layout>& operator+=(const OtherTensor &rhs)
   {
      ForallDims<Tensor>::ApplyBinOp(*this, rhs, [&](auto... idx)
      {
         (*this)(idx...) += rhs(idx...);
      });
      return *this;
   }

   // TODO Soft and Hard Get? (lazy accessor or hard copy)
   // Call it Extract?

   /// Lazy accessor for the sub-Tensor extracted from idx in Nth dimension.
   template <int N> MFEM_HOST_DEVICE inline
   auto Get(int idx)
   {
      static_assert(N>=0 && N<get_layout_rank<Layout>,
         "Cannot access this dimension with Get");
      using C = ViewContainer<T,Container>;
      using L = RestrictedLayout<N,Layout>;
      using RestrictedTensor = Tensor<C,L>;
      C data(*this);
      L layout(idx,*this);
      return RestrictedTensor(data,layout);
   }

   template <int N> MFEM_HOST_DEVICE inline
   auto Get(int idx) const
   {
      static_assert(N>=0 && N<get_layout_rank<Layout>,
         "Cannot access this dimension with Get");
      using C = ConstViewContainer<T,Container>;
      using L = RestrictedLayout<N,Layout>;
      using RestrictedTensor = Tensor<C,L>;
      C data(*this);
      L layout(idx,*this);
      return RestrictedTensor(data,layout);
   }

   /// Get the Sub-Tensor extracted from idx in Nth dimension.
   // template <int N, int... Dims, typename... Idx>
   // auto MultiGet(int idx0, Idx... idx)
   // {
   //    return Get<N>(idx0).MultiGet<Dims...,Idx...>(idx...);
   // }

   // template <int N, int... Dims, typename... Idx>
   // auto MultiGet(int idx0, Idx... idx) const
   // {
   //    return Get<N>(idx0).MultiGet<Dims...,Idx...>(idx...);
   // }

   /// Generate a Tensor that be read on device
   auto Read()
   {
      return Tensor<ReadContainer<T>,Layout>(this->ReadData(),*this);
   }

   /// Generate a Tensor that be writen on device (read is unsafe)
   auto Write()
   {
      return Tensor<DeviceContainer<T>,Layout>(this->WriteData(),*this);
   }

   /// Generate a Tensor that be read and writen on device
   auto ReadWrite()
   {
      return Tensor<DeviceContainer<T>,Layout>(this->ReadWriteData(),*this);
   }

   friend std::ostream& operator<<(std::ostream &os, const Tensor &t)
   {
      ForallDims<Tensor>::Apply(t,[&](auto... idx)
      {
         os << t(idx...) << " ";
      });
      os << std::endl;
      return os;
   }
};

//////////////////////////
// Behavioral Tensor types

/// Dynamically sized Tensor
template <int Rank, typename T, int MaxSize = pow(16,Rank)>
using DynamicTensor = Tensor<StaticContainer<T, MaxSize>,
                             DynamicLayout<Rank> >;

template <int Rank, int MaxSize = pow(16,Rank)>
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
constexpr int get_DynamicBlockLayout_size(int MaxSize, int Rank)
{
   return Rank>2 ? pow(MaxSize,Rank-2) : 1;
}

template <int Rank, typename T, int BatchSize, int MaxSize = 16>
using Dynamic2dThreadTensor = Tensor<
                              StaticContainer<
                                 T,
                                 get_DynamicBlockLayout_size(MaxSize,Rank)
                              >,
                              Dynamic2dThreadLayout<Rank,BatchSize>
                           >;

template <int Rank, int BatchSize, int MaxSize = 16>
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

template <int Rank, typename T, int BatchSize, int MaxSize = 16>
using Dynamic3dThreadTensor = Tensor<
                                 StaticContainer<
                                    T,
                                    get_DynamicBlockLayout_size(MaxSize,Rank)
                                 >,
                                 Dynamic3dThreadLayout<Rank,BatchSize>
                              >;

template <int Rank, int BatchSize, int MaxSize = 16>
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

template <typename... Sizes>
constexpr int get_Static3dThreadTensor_size(int Size0, int Size1, Sizes... sizes)
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

////////////////////////////
// Architecture Tensor types

/// Defines the dynamic type of Tensor used for computation on CPU.
template <int Rank, typename T, int BatchSize, int MaxSize = pow(16,Rank)>
using DynamicCPUTensor = DynamicTensor<Rank, T, MaxSize>;

/// Defines the static type of Tensor used for computation on CPU.
template <typename T, int BatchSize, int... Sizes>
using StaticCPUTensor = StaticTensor<T, Sizes...>;

/// Defines the dynamic type of Tensor used for computation on CUDA.
template <int Rank, typename T, int BatchSize, int MaxSize = pow(16,Rank)>
using DynamicCUDATensor = Dynamic2dThreadTensor<Rank, T, BatchSize, MaxSize>;

/// Defines the static type of Tensor used for computation on CUDA.
template <typename T, int BatchSize, int... Sizes>
using StaticCUDATensor = Static2dThreadTensor<T, BatchSize, Sizes...>;

/// Defines the dynamic type of Tensor used for computation on Hip.
template <int Rank, typename T, int BatchSize, int MaxSize = pow(16,Rank)>
using DynamicHipTensor = Dynamic2dThreadTensor<Rank, T, BatchSize, MaxSize>;

/// Defines the static type of Tensor used for computation on Hip.
template <typename T, int BatchSize, int... Sizes>
using StaticHipTensor = Static2dThreadTensor<T, BatchSize, Sizes...>;

/// A structure that defines static and dynamic Tensor types for an architecture
struct DeviceTensorType
{
#ifdef __CUDA_ARCH__
   // CUDA types
   template <int Rank, typename T, int BatchSize, int MaxSize = pow(16,Rank)>
   using dynamic_type = DynamicCUDATensor<Rank,T,BatchSize,MaxSize>;

   template <typename T, int BatchSize, int... Sizes>
   using static_type = StaticCUDATensor<T,BatchSize,Sizes...>;
#elif defined(__HIP_DEVICE_COMPILE__)
   // Hip types
   template <int Rank, typename T, int BatchSize, int MaxSize = pow(16,Rank)>
   using dynamic_type = DynamicHipTensor<Rank,T,BatchSize,MaxSize>;

   template <typename T, int BatchSize, int... Sizes>
   using static_type = StaticHipTensor<T,BatchSize,Sizes...>;
#elif defined(FUGAKU_ARCH) // extension exemple
   template <int Rank, typename T, int BatchSize, int MaxSize = pow(16,Rank)>
   using dynamic_type = DynamicCPUTensor<Rank,T,BatchSize,MaxSize>;

   template <typename T, int BatchSize, int... Sizes>
   using static_type = StaticCPUTensor<T,BatchSize,Sizes...>;
#else
   // CPU types
   template <int Rank, typename T, int BatchSize, int MaxSize = pow(16,Rank)>
   using dynamic_type = DynamicCPUTensor<Rank,T,BatchSize,MaxSize>;

   template <typename T, int BatchSize, int... Sizes>
   using static_type = StaticCPUTensor<T,BatchSize,Sizes...>;
#endif
};

/// Defines the dynamic Tensor type for the compiling architecture
template <int Rank, typename T, int BatchSize, int MaxSize = pow(16,Rank)>
using DynamicDeviceTensor = DeviceTensorType::dynamic_type<Rank,T,BatchSize,MaxSize>;

template <int Rank, int BatchSize, int MaxSize = pow(16,Rank)>
using DynamicDeviceDTensor = DynamicDeviceTensor<Rank,double,BatchSize,MaxSize>;

/// Defines the static Tensor type for the compiling architecture
template <typename T, int BatchSize, int... Sizes>
using StaticDeviceTensor = DeviceTensorType::static_type<T,BatchSize,Sizes...>;

template <int BatchSize, int... Sizes>
using StaticDeviceDTensor = StaticDeviceTensor<double,BatchSize,Sizes...>;

// template <int Rank, typename T>
// using ViewTensor = Tensor<Rank,T,ViewContainer,RestrictedLayout>;

// template <int Rank, typename T>
// using ConstViewTensor = Tensor<Rank,T,ConstViewContainer,RestrictedLayout>;

// template <typename... LayoutParams,
//           template <typename...> typename LayoutOut,
//           int Rank,
//           typename T,
//           typename Container,
//           typename LayoutIn>
// auto Reshape(Tensor<Rank,T,Container,LayoutIn> &t, InitParams... params)
// {
//    return Tensor<Rank,T,Container,LayoutOut>{params};
// }

// template <typename Basis>
// class TensorTypeForBasis;

// template <int Dim>
// class TensorTypeForBasis<Basis<Dim,true,0,0>>
// {
//    using Type = DynamicDTensor<Dim>;

//    template <typename... Sizes> MFEM_HOST_DEVICE
//    Type make()
// };

// template <int Dim>
// class TensorTypeForBasis<Basis<Dim,false,0,0>>
// {
//    using Type = DynamicDTensor<1>;
// };



// template <int Dim, bool IsTensor, int Dofs, int Quads>
// class TensorTypeForBasis<Basis<Dim,true,Dofs,Quads>>
// {
//    using Static2dThreadDTensor<Q>;
// };

// template <int Dim, bool IsTensor, int Dofs, int Quads>
// class TensorTypeForBasis<Basis<Dim,false,Dofs,Quads>>
// {

// };

// template <typename Tensor, class Enable = void>
// struct TypeOf_t;

// // TODO not really what I want... I want the static or dynamic format (<Sizes...> or <Rank>)
// template <int Rank,
//           typename T,
//           typename C,
//           typename L,
//           std::enable_if_t<
//              is_static_tensor<Tensor<Rank,T,C,L>>::value
//           > >
// struct TypeOf_t<Tensor<Rank,T,C,L>>
// {
//    template <typename R,typename T,typename C,typename L>
//    using type = Tensor<R,T,C,L>;
//    // template <int Sizes...>
//    // using typet = Tensor<Sizes...>
// };

// template <typename Tensor>
// using TypeOf = TypeOf_t<Tensor>;

// template <int Rank, typename T, int MaxSize>
// class TypeOf<DynamicTensor<Rank,T,MaxSize>>
// {
//    template <int... Sizes>
//    using type = DynamicTensor<rank(Sizes...),T,MaxSize>;
// };

// template <typename T, int... Sizes>
// class TypeOf<Static2dThreadTensor<T,Sizes...>>
// {
//    template <int... YourSizes>
//    using type = Static2dThreadTensor<T,YourSizes...>;
// };

} // namespace mfem

#endif // MFEM_TENSOR
