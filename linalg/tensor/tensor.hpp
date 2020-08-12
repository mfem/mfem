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

#include "../general/backends.hpp"

namespace mfem
{

// Getter for the N-th dimension value
template <int N, int... Dims>
struct Dim;

template <int Dim0, int... Dims>
struct Dim<0, Dim0, Dims...>
{
   static constexpr int val = Dim0;
};
template <int N, int Dim0, int... T>
struct Dim<N, Dim0, T...>
{
   static constexpr int val = Dim0;
};

// Compute the product of the dimensions
template<int Dim0, int... Dims>
struct Size
{
   static constexpr int val = Dim0 * Size<Dims...>::val;
};

template<int Dim0>
struct Size<Dim0>
{
   static constexpr int val = Dim0;
};

//Compute the index inside a Tensor
template<int Cpt, int rank, int... Dims>
struct Index
{
   template <typename... Idx>
   static inline int eval(int first, Idx... args)
   {
      return first + Dim<Cpt-1,Dims...>::val * Index<Cpt+1, rank, Dims...>::eval(args...);
   }
};

template<int rank, int... Dims>
struct Index<rank,rank,Dims...>
{
   static inline int eval(int first)
   {
      return first;
   }
};

template<int... Dims>
struct TensorIndex
{
   template <typename... Idx>
   static inline int eval(Idx... args)
   {
      return Index<1,sizeof...(Dims),Dims...>::eval(args...);
   }
};

/// A fixed size tensor class
template<typename T, int... Dims>
class Tensor{
private:
   MFEM_SHARED T data[Size<Dims...>::val];

public:
   explicit Tensor(const T &val)
   {
      for (size_t i = 0; i < Size<Dims...>::val; i++)
      {
         data[i] = val;
      }      
   }

   Tensor(const Tensor &rhs)
   {
      for (size_t i = 0; i < Size<Dims...>::val; i++)
      {
         data[i] = rhs[i];
      }
   }

   template<typename... Idx> MFEM_HOST_DEVICE inline
   const T& operator()(Idx... args) const
   {
      static_assert(sizeof...(Dims)==sizeof...(Idx), "Wrong number of indices");
      return data[ TensorIndex<Dims...>::eval(args...) ];
   }

   template<typename... Idx> MFEM_HOST_DEVICE inline
   T& operator()(Idx... args)
   {
      static_assert(sizeof...(Dims)==sizeof...(Idx), "Wrong number of indices");
      return data[ TensorIndex<Dims...>::eval(args...) ];
   }

   Tensor<T,Dims...>& operator= (const T &val)
   {
      for (size_t i = 0; i < Size<Dims...>::val; i++)
      {
         data[i] = val;
      }
      return *this;
   }

};

template <int... Dims>
using dTensor = Tensor<double,Dims...>;


} // namespace mfem

#endif // MFEM_TENSOR
