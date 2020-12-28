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
          typename Layout = DynamicLayout<Rank>>
class Tensor: public Container, public Layout
{
public:
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

template <typename T, int... Sizes>
using StaticSharedTensor = Tensor<sizeof...(Sizes),
                                  T,
                                  StaticSharedContainer<T, Sizes...>,
                                  StaticLayout<Sizes...> >;

template <int... Sizes>
using StaticSharedDTensor = StaticSharedTensor<double,Sizes...>;

template <int Rank, typename T, int MaxSize = pow(16,Rank)>
using SharedTensor = Tensor<Rank,
                            T,
                            StaticSharedContainer<T, MaxSize>,
                            DynamicLayout<Rank> >;

template <int Rank, int MaxSize = pow(16,Rank)>
using SharedDTensor = SharedTensor<Rank,double,MaxSize>;

/// Statically sized Tensor
template <typename T, int... Sizes>
using StaticTensor = Tensor<sizeof...(Sizes),
                            T,
                            StaticContainer<T, Sizes...>,
                            StaticLayout<Sizes...> >;

template <int... Sizes>
using dTensor = StaticTensor<double,Sizes...>;

template <int... Sizes>
using StaticDTensor = StaticTensor<double,Sizes...>;

/// Dynamically sized Tensor
template <int Rank, typename T, int MaxSize = pow(16,Rank)>
using DynamicTensor = Tensor<Rank,
                             T,
                             StaticContainer<T, MaxSize>,
                             DynamicLayout<Rank> >;

template <int Rank, int MaxSize = pow(16,Rank)>
using DynamicDTensor = DynamicTensor<Rank,double,MaxSize>;

/// A Tensor distributed statically over a plane of threads
template <typename T, int... Sizes>
using BlockTensor = Tensor<sizeof...(Sizes),
                           T,
                           BlockContainer<T, Sizes...>,
                           BlockLayout<Sizes...> >;

template <int... Sizes>
using BlockDTensor = BlockTensor<double,Sizes...>;

/// A Tensor distributed dynamically over a plane of threads
template <int Rank, typename T, int MaxSize = pow(16,Rank)>
using DynamicBlockTensor = Tensor<Rank,
                           T,
                           BlockContainer<T, MaxSize>,
                           DynamicBlockLayout<Rank> >;

template <int Rank, int MaxSize = pow(16,Rank)>
using DynamicBlockDTensor = DynamicBlockTensor<Rank,double,MaxSize>;

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
//    using BlockDTensor<Q>;
// };

// template <int Dim, bool IsTensor, int Dofs, int Quads>
// class TensorTypeForBasis<Basis<Dim,false,Dofs,Quads>>
// {

// };

template <typename Tensor>
class TypeOf;

template <int Rank, typename T, int MaxSize>
class TypeOf<DynamicTensor<Rank,T,MaxSize>>
{
   template <int... Sizes>
   using type = DynamicTensor<rank(Sizes...),T,MaxSize>;
};

template <typename T, int... Sizes>
class TypeOf<BlockTensor<T,Sizes...>>
{
   template <int... YourSizes>
   using type = BlockTensor<T,YourSizes...>;
};

} // namespace mfem

#endif // MFEM_TENSOR
