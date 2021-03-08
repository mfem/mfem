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

namespace mfem
{

/** A tensor class
    @a Rank is the rank of the Tensor,
    @a T is the type of elements stored,
    @a Container is the type of data container, they can either be statically or
       dynamically allocated,
    @a Layout is a class that represents the data layout
       There is two main sub-categories of Layout, Static and Dynamic layouts.
       Dynamic Layout have the following signature:
       template <int Rank, typename T>,
       Static Layout have the following signature:
       template <typename T, int... Sizes>,
       where Sizes... is the list of the sizes of the dimensions of the Tensor.
   */
template <int Rank,
          typename T = double,
          typename Container = MemoryContainer<T>,
          typename Layout = DynamicLayout<Rank> >
class Tensor: public Container, public Layout
{
public:
   static const int rank = Rank;
   using type = T;
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

   /// Copy Constructor
   MFEM_HOST_DEVICE
   Tensor(const Tensor &rhs): Container(rhs), Layout(rhs) { }

   /// Accessor
   template <typename... Idx> MFEM_HOST_DEVICE inline
   T& operator()(Idx... args)
   {
      // static_assert(Rank==sizeof...(Idx), "Wrong number of indices");
      return this->operator[]( this->index(args...) );
   }

   /// Const Accessor
   template <typename... Idx> MFEM_HOST_DEVICE inline
   const T& operator()(Idx... args) const
   {
      // static_assert(Rank==sizeof...(Idx), "Wrong number of indices");
      return this->operator[]( this->index(args...) );
   }

   /// Return the size of the dimension N
   // TODO remove after inheriting
   // template <int N>
   // int Size() const
   // {
   //    return index.template Size<N>();
   // }

   /// Initialization of a Tensor to a constant value.
   MFEM_HOST_DEVICE inline
   Tensor<Rank,T,Container,Layout>& operator=(const T &val)
   {
      // TODO this doesn't work with all containers
      for (size_t i = 0; i < this->Capacity(); i++)
      {
         this->operator[](i) = val;
      }
      return *this;
   }

   template <typename OtherTensor> MFEM_HOST_DEVICE inline
   Tensor<Rank,T,Container,Layout>& operator=(const OtherTensor &rhs)
   {
      Forall<Rank>::equalize(*this,rhs);
      return *this;
   }

   // TODO Soft and Hard Get? (lazy accessor or hard copy)

   /// Get the Sub-Tensor extracted from idx in Nth dimension.
   template <int N>
   auto Get(int idx)
   {
      static_assert(N>=0 && N<Rank,"Cannot access this dimension with Get");
      using RestrictedTensor = Tensor<Rank,
                                      T,
                                      ViewContainer<T,Container>,
                                      RestrictedLayout<N,Layout>>;
      ViewContainer<T,Container> data(*this);
      RestrictedLayout<N,Layout> layout(idx,*this);
      return RestrictedTensor(data,layout);
   }

   template <int N>
   auto Get(int idx) const
   {
      static_assert(N>=0 && N<Rank,"Cannot access this dimension with Get");
      using RestrictedTensor = Tensor<Rank-1,
                                      T,
                                      ConstViewContainer<T,Container>,
                                      RestrictedLayout<N,Layout>>;
      ConstViewContainer<T,Container> data(*this);
      RestrictedLayout<N,Layout> layout(idx,*this);
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
      return Tensor<Rank,T,ReadContainer<T>,Layout>(this->ReadData(),*this);
   }

   /// Generate a Tensor that be writen on device (read is unsafe)
   auto Write()
   {
      return Tensor<Rank,T,DeviceContainer<T>,Layout>(this->WriteData(),*this);
   }

   /// Generate a Tensor that be read and writen on device
   auto ReadWrite()
   {
      return Tensor<Rank,T,DeviceContainer<T>,Layout>(this->ReadWriteData(),
                                                      *this);
   }

private:
/// A structure that implements an imbricated forall with for loops.
template <int N>
struct Forall
{
   template <typename TensorLHS, typename TensorRHS, typename... Idx>
   static void equalize(TensorLHS &lhs, const TensorRHS &rhs, Idx... idx)
   {
      // TODO replace with iterator
      for(int i = 0; i<lhs.template Size<N-1>(); i++)
      {
         // TODO understand why the recursive form doesn't work
         Forall<N-1>::equalize(lhs,rhs,i,idx...);
      }
   }
};

template <>
struct Forall<2>
{
   template <typename TensorLHS, typename TensorRHS, typename... Idx>
   static void equalize(TensorLHS &lhs, const TensorRHS &rhs, Idx... idx)
   {
      for(int j = 0; j<lhs.template Size<1>(); j++)
      {
         for(int i = 0; i<lhs.template Size<0>(); i++)
         {
            lhs(i,j) = rhs(i,j);
         }
      }
   }
};

template <>
struct Forall<0>
{
   template <typename TensorLHS, typename TensorRHS, typename... Idx>
   static void equalize(TensorLHS &lhs, const TensorRHS &rhs, Idx... idx)
   {
      lhs(idx...) = rhs(idx...);
   }
};

};

//////////////////////////
// Behavioral Tensor types

/// Dynamically sized Tensor
template <int Rank, typename T, int MaxSize = pow(16,Rank)>
using DynamicTensor = Tensor<Rank,
                             T,
                             StaticContainer<T, MaxSize>,
                             DynamicLayout<Rank> >;

template <int Rank, int MaxSize = pow(16,Rank)>
using DynamicDTensor = DynamicTensor<Rank,double,MaxSize>;

/// Statically sized Tensor
template <typename T, int... Sizes>
using StaticTensor = Tensor<sizeof...(Sizes),
                            T,
                            StaticContainer<T, Sizes...>,
                            StaticLayout<Sizes...> >;

template <int... Sizes>
using dTensor = StaticTensor<double,Sizes...>; // TODO remove

template <int... Sizes>
using StaticDTensor = StaticTensor<double,Sizes...>;

// TODO deprecate
/// A dynamically sized Tensor using a static amount of shared memory.
template <int Rank, typename T, int MaxSize = pow(16,Rank)>
using DynamicSharedTensor = Tensor<Rank,
                                   T,
                                   StaticSharedContainer<T, MaxSize>,
                                   DynamicLayout<Rank> >;

template <int Rank, int MaxSize = pow(16,Rank)>
using DynamicSharedDTensor = DynamicSharedTensor<Rank,double,MaxSize>;

/// A statically sized Tensor using shared memory.
template <typename T, int... Sizes>
using StaticSharedTensor = Tensor<sizeof...(Sizes),
                                  T,
                                  StaticSharedContainer<T, Sizes...>,
                                  StaticLayout<Sizes...> >;

template <int... Sizes>
using StaticSharedDTensor = StaticSharedTensor<double,Sizes...>;

/// A Tensor dynamically distributed over a plane of threads
template <int Rank, typename T, int BatchSize, int MaxSize = pow(16,Rank)>
using DynamicBlockTensor = Tensor<Rank,
                                  T,
                                  BlockContainer<T, MaxSize>,
                                  DynamicBlockLayout<Rank,BatchSize> >;

template <int Rank, int BatchSize, int MaxSize = pow(16,Rank)>
using DynamicBlockDTensor = DynamicBlockTensor<Rank,double,BatchSize,MaxSize>;

/// A Tensor statically distributed over a plane of threads
template <typename T, int BatchSize, int... Sizes>
using StaticBlockTensor = Tensor<sizeof...(Sizes),
                                 T,
                                 BlockContainer<T, Sizes...>,
                                 BlockLayout<BatchSize, Sizes...> >;

template <int BatchSize, int... Sizes>
using StaticBlockDTensor = StaticBlockTensor<double,BatchSize,Sizes...>;

/// A tensor using a read write access pointer and a dynamic data layout.
// Backward compatible if renamed in DeviceTensor
template <int Rank, typename T>
using MyDeviceTensor = Tensor<Rank,
                            T,
                            DeviceContainer<T>,
                            DynamicLayout<Rank> >;

template <int Rank>
using DeviceDTensor = MyDeviceTensor<Rank,double>;

/// A tensor using a read only const pointer and a dynamic data layout.
template <int Rank, typename T>
using ReadTensor = Tensor<Rank,
                          T,
                          ReadContainer<T>,
                          DynamicLayout<Rank> >;

template <int Rank>
using ReadDTensor = ReadTensor<Rank,double>;

//////////////////////////
// Convenient Tensor types

/// Defines the dynamic type of Tensor used for computation on CPU.
template <int Rank, typename T, int BatchSize, int MaxSize = pow(16,Rank)>
using DynamicCPUTensor = DynamicTensor<Rank, T, MaxSize>;

/// Defines the static type of Tensor used for computation on CPU.
template <typename T, int BatchSize, int... Sizes>
using StaticCPUTensor = StaticTensor<T, BatchSize, Sizes...>;

/// Defines the dynamic type of Tensor used for computation on CUDA.
template <int Rank, typename T, int BatchSize, int MaxSize = pow(16,Rank)>
using DynamicCUDATensor = DynamicBlockTensor<Rank, T, BatchSize, MaxSize>;

/// Defines the static type of Tensor used for computation on CUDA.
template <typename T, int BatchSize, int... Sizes>
using StaticCUDATensor = StaticBlockTensor<T, BatchSize, Sizes...>;

/// Defines the dynamic type of Tensor used for computation on Hip.
template <int Rank, typename T, int BatchSize, int MaxSize = pow(16,Rank)>
using DynamicHipTensor = DynamicBlockTensor<Rank, T, BatchSize, MaxSize>;

/// Defines the static type of Tensor used for computation on Hip.
template <typename T, int BatchSize, int... Sizes>
using StaticHipTensor = StaticBlockTensor<T, BatchSize, Sizes...>;

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
//    using StaticBlockDTensor<Q>;
// };

// template <int Dim, bool IsTensor, int Dofs, int Quads>
// class TensorTypeForBasis<Basis<Dim,false,Dofs,Quads>>
// {

// };

// template <typename Tensor>
// class TypeOf;

// template <int Rank, typename T, int MaxSize>
// class TypeOf<DynamicTensor<Rank,T,MaxSize>>
// {
//    template <int... Sizes>
//    using type = DynamicTensor<rank(Sizes...),T,MaxSize>;
// };

// template <typename T, int... Sizes>
// class TypeOf<StaticBlockTensor<T,Sizes...>>
// {
//    template <int... YourSizes>
//    using type = StaticBlockTensor<T,YourSizes...>;
// };

} // namespace mfem

#endif // MFEM_TENSOR
