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

#ifndef MFEM_LAYOUT
#define MFEM_LAYOUT

#include "util.hpp"

namespace mfem
{

template <int Rank>
class DynamicLayout
{
private:
   int sizes[Rank];

public:
   template <typename... Sizes> MFEM_HOST_DEVICE
   DynamicLayout(Sizes... args)
   {
      Init<1, Rank, Sizes...>::result(sizes, args...);
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   int operator()(Idx... idx) const
   {
   #if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)) // TODO add MFEM_DEBUG
      MFEM_VERIFY(Rank==sizeof...(Idx), "Wrong number of arguments.");
   #endif
      return DynamicTensorIndex<1, Rank, Idx...>::eval(sizes, idx...);
   }

   template <int N>
   int Size() const
   {
      static_assert(N>=0 && N<Rank,"Accessed size is higher than the rank of the Tensor.");
      return sizes[N];
   }

private:
   /// A Class to compute the real index from the multi-indices of a tensor
   template <int N, int Dim, typename T, typename... Args>
   class DynamicTensorIndex
   {
   public:
      MFEM_HOST_DEVICE
      static inline int eval(const int* sizes, T first, Args... args)
      {
   #if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)) // TODO add MFEM_DEBUG
         MFEM_VERIFY(first<sizes[N-1],"Trying to access out of boundary.");
   #endif
         return first + sizes[N - 1] * DynamicTensorIndex < N + 1, Dim, Args... >
               ::eval(sizes, args...);
      }
   };

   // Terminal case
   template <int Dim, typename T, typename... Args>
   class DynamicTensorIndex<Dim, Dim, T, Args...>
   {
   public:
      MFEM_HOST_DEVICE
      static inline int eval(const int* sizes, T first, Args... args)
      {
   #if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)) // TODO add MFEM_DEBUG
         MFEM_VERIFY(first<sizes[Dim-1],"Trying to access out of boundary.");
   #endif
         return first;
      }
   };

   /// A class to initialize the size of a Tensor
   template <int N, int Dim, typename T, typename... Args>
   class Init
   {
   public:
      static inline int result(int* sizes, T first, Args... args)
      {
         sizes[N - 1] = first;
         return first * Init < N + 1, Dim, Args... >::result(sizes, args...);
      }
   };

   // Terminal case
   template <int Dim, typename T, typename... Args>
   class Init<Dim, Dim, T, Args...>
   {
   public:
      static inline int result(int* sizes, T first, Args... args)
      {
         sizes[Dim - 1] = first;
         return first;
      }
   };
};

template<int... Sizes>
class StaticLayout
{
public:
   template <typename... Dims> MFEM_HOST_DEVICE
   StaticLayout(Dims... args)
   {
      // static_assert(sizeof...(Dims)==sizeof...(Sizes), "Static and dynamic sizes don't match.");
      // TODO verify that Dims == sizes in Debug mode
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   constexpr int operator()(Idx... idx) const
   {
   #if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
      static_assert(sizeof...(Sizes)==sizeof...(Idx), "Wrong number of arguments.");
   #endif
      return StaticTensorIndex<Sizes...>::eval(idx...);
   }

   // Can be constexpr if Tensor inherit from Layout
   template <int N>
   int Size() const
   {
      static_assert(N>=0 && N<rank(Sizes...),"Accessed size is higher than the rank of the Tensor.");
      return Dim<N,Sizes...>::val;
   }

private:
   // Getter for the N-th dimension value
   template <int N, int... Dims>
   struct Dim;

   template <int Dim0, int... Dims>
   struct Dim<0, Dim0, Dims...>
   {
      static constexpr int val = Dim0;
   };
   template <int N, int Dim0, int... Dims>
   struct Dim<N, Dim0, Dims...>
   {
      static constexpr int val = Dim<N-1,Dims...>::val;
   };

   //Compute the index inside a Tensor
   template<int Cpt, int rank, int... Dims>
   struct StaticIndex
   {
      template <typename... Idx>
      static inline int eval(int first, Idx... args)
      {
         return first + Dim<Cpt-1,Dims...>::val * StaticIndex<Cpt+1, rank, Dims...>::eval(args...);
      }
   };

   template<int rank, int... Dims>
   struct StaticIndex<rank,rank,Dims...>
   {
      static inline int eval(int first)
      {
         return first;
      }
   };

   template<int... Dims>
   struct StaticTensorIndex
   {
      template <typename... Idx>
      static inline int eval(Idx... args)
      {
         return StaticIndex<1,sizeof...(Dims),Dims...>::eval(args...);
      }
   };
};

template <int... Dims>
class BlockLayout;

template <int DimX>
class BlockLayout<DimX>
{
public:
   constexpr int operator()(int idx) const
   {
      // TODO verify that idx < DimX
      return 0;
   }
};

template <int DimX, int DimY>
class BlockLayout<DimX, DimY>
{
public:
   constexpr int operator()(int idx0, int idx1) const
   {
      // TODO verify that idx0 < DimX && idx1 < DimY
      // TODO verify that idx0 == threadIdx.x && idx1 == threadIdx.y
      return 0;
   }
};

template <int DimX, int DimY, int... Dims>
class BlockLayout<DimX,DimY,Dims...>
{
private:
   StaticLayout<Dims...> layout;
public:
   template <typename... Idx> MFEM_HOST_DEVICE inline
   constexpr int operator()(int idx0, int idx1, Idx... idx) const
   {
      // TODO verify that idx0 < DimX && idx1 < DimY && idx2 < DimZ
      // TODO verify that idx0 == threadIdx.x && idx1 == threadIdx.y
      return layout(idx...);
   }
};

} // namespace mfem

#endif // MFEM_LAYOUT
