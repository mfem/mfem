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

/// A dynamic layout with first index fastest
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
   #if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
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
   #if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
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
   #if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
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

/// A static layout
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

/// Layout using a thread plane to distribute data
template <int... Dims>
class BlockLayout;

template <int DimX>
class BlockLayout<DimX>
{
public:
   MFEM_HOST_DEVICE inline
   BlockLayout(int size0)
   {
      // TODO Verify in debug that size0==DimX
      // TODO verify that size0 < BlockSizeX
   }

   MFEM_HOST_DEVICE inline
   constexpr int operator()(int idx0) const
   {
      // TODO verify that idx0 < DimX
      // TODO verify that idx0 == threadIdx.x
      return 0;
   }

   // Can be constexpr if Tensor inherit from Layout
   template <int N> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      static_assert(N==0,"Accessed size is higher than the rank of the Tensor.");
      return DimX;
   }
};

template <int DimX, int DimY>
class BlockLayout<DimX, DimY>
{
public:
   MFEM_HOST_DEVICE inline
   BlockLayout(int size0, int size1)
   {
      // TODO Verify in debug that size0==DimX && size1==DimY
      // TODO verify that size0 < BlockSizeX && size1 < BlockSizeY
   }

   MFEM_HOST_DEVICE inline
   constexpr int operator()(int idx0, int idx1) const
   {
      // TODO verify that idx0 < DimX && idx1 < DimY
      // TODO verify that idx0 == threadIdx.x && idx1 == threadIdx.y
      return 0;
   }

   // Can be constexpr if Tensor inherit from Layout
   template <int N> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      static_assert(N>=0 && N<2,"Accessed size is higher than the rank of the Tensor.");
      return Dim<N,DimX,DimY>::val;
   }
};

template <int DimX, int DimY, int... Dims>
class BlockLayout<DimX,DimY,Dims...>
{
private:
   StaticLayout<Dims...> layout;
public:
   template <typename... Sizes> MFEM_HOST_DEVICE inline
   BlockLayout(int size0, int size1, Sizes... sizes)
   : layout(sizes)
   {
      // TODO Verify in debug that size0==DimX && size1==DimY
      // TODO verify that size0 < BlockSizeX && size1 < BlockSizeY
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   constexpr int operator()(int idx0, int idx1, Idx... idx) const
   {
      // TODO verify that idx0 < DimX && idx1 < DimY && idx2 < DimZ
      // TODO verify that idx0 == threadIdx.x && idx1 == threadIdx.y
      return layout(idx...);
   }

   // Can be constexpr if Tensor inherit from Layout
   template <int N> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      static_assert(N>=0 && N<rank(Sizes...),"Accessed size is higher than the rank of the Tensor.");
      return Dim<N,DimX,DimY,Dims...>::val;
   }
};

template <int Rank>
class DynamicBlockLayout
{
private:
   const int size0;
   const int size1;
   DynamicLayout<Rank-2> layout;
public:
   template <typename... Sizes> MFEM_HOST_DEVICE inline
   DynamicBlockLayout(int size0, int size1,  Sizes... sizes)
   : size0(size0), size1(size1), layout(sizes...)
   {
      // TODO verify that size0 < BlockSizeX && size1 < BlockSizeY
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   constexpr int operator()(int idx0, int idx1, Idx... idx) const
   {
      // TODO verify that idx0 < size0 && idx1 < size1
      // TODO verify that idx0 == threadIdx.x && idx1 == threadIdx.y
      return layout(idx...);
   }

   // Can be constexpr if Tensor inherit from Layout
   template <int N> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      static_assert(N>=0 && N<Rank,"Accessed size is higher than the rank of the Tensor.");
      switch (N)
      {
      case 0:
         return size0;
      case 1:
         return size1;
      default:
         return layout.Size<N-2>();
      }
   }
};

template <>
class DynamicBlockLayout<1>
{
private:
   const int size0;
public:
   MFEM_HOST_DEVICE inline
   DynamicBlockLayout(int size0)
   : size0(size0)
   {
      // TODO verify that size0 < BlockSizeX
   }

   MFEM_HOST_DEVICE inline
   constexpr int operator()(int idx) const
   {
      // TODO verify that idx < DimX
      return 0;
   }

   // Can be constexpr if Tensor inherit from Layout
   template <int N> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      static_assert(N==0,"Accessed size is higher than the rank of the Tensor.")
      return size0;
   }
};

template <>
class DynamicBlockLayout<2>
{
private:
   const int size0;
   const int size1;
public:
   MFEM_HOST_DEVICE inline
   DynamicBlockLayout(int size0, int size1)
   : size0(size0), size1(size1)
   {
      // TODO verify that size0 < BlockSizeX && size1 < BlockSizeY
   }

   MFEM_HOST_DEVICE inline
   constexpr int operator()(int idx0, int idx1) const
   {
      // TODO verify that idx0 < size0 && idx1 < size1
      // TODO verify that idx0 == threadIdx.x && idx1 == threadIdx.y
      return 0;
   }

   // Can be constexpr if Tensor inherit from Layout
   template <int N> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      static_assert(N>=0 && N<2,"Accessed size is higher than the rank of the Tensor.");
      if (N==0)
      {
         return size0;
      }
      else
      {
         return size1;
      }
   }
};

/// Strided Layout
template <int Rank>
class StridedLayout
{
private:
   int strides[Rank];
   int offsets[Rank];
   int sizes[Rank]

public:
   template <typename... Idx> MFEM_HOST_DEVICE inline
   constexpr int operator()(Idx... idx) const
   {
      static_assert(sizeof...(Idx...)==Rank,"Wrong number of argumets.");
      return StridedIndex::eval<1>(idx...);
   }

   // Can be constexpr if Tensor inherit from Layout
   template <int N> MFEM_HOST_DEVICE inline
   int Size() const
   {
      static_assert(N>=0 && N<rank(Sizes...),"Accessed size is higher than the rank of the Tensor.");
      return sizes[N];
   }

private:
   template <int N>
   struct StridedIndex
   {
      template <typename... Idx>
      static inline int eval(int first, Idx... args)
      {
         return (offsets[N-1]+first)*strides[N-1] + StridedIndex<N+1>(args...);
      }
   };

   template <>
   struct StridedIndex<Rank>
   {
      template <typename... Idx>
      static inline int eval(int first)
      {
         return (offsets[N-1]+first)*strides[N-1];
      }
   };
};

// TODO possible to write directly in a generic way?
// TODO throw an error if N>8?
template <int N, typename Layout>
class RestrictedLayout;

template <typename Layout>
class RestrictedLayout<0,Layout>
{
private:
   const int i;
   Layout &layout;

public:
   MFEM_HOST_DEVICE
   RestrictedLayout(int i, Layout &layout): i(i), layout(layout)
   {
      // TODO Check i < layout.Size<0>()
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   int operator()(Idx... idx)
   {
      return layout(i,idx...);
   }
};

template <typename Layout>
class RestrictedLayout<1,Layout>
{
private:
   const int i;
   Layout &layout;

public:
   MFEM_HOST_DEVICE
   RestrictedLayout(int i, Layout &layout): i(i), layout(layout)
   {
      // TODO Check i < layout.Size<1>()
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   int operator()(int idx0, Idx... idx)
   {
      return layout(idx0,i,idx...);
   }
};

template <typename Layout>
class RestrictedLayout<2,Layout>
{
private:
   const int i;
   Layout &layout;

public:
   MFEM_HOST_DEVICE
   RestrictedLayout(int i, Layout &layout): i(i), layout(layout)
   {
      // TODO Check i < layout.Size<2>()
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   int operator()(int idx0, int idx1, Idx... idx)
   {
      return layout(idx0,idx1,i,idx...);
   }
};

template <typename Layout>
class RestrictedLayout<3,Layout>
{
private:
   const int i;
   Layout &layout;

public:
   MFEM_HOST_DEVICE
   RestrictedLayout(int i, Layout &layout): i(i), layout(layout)
   {
      // TODO Check i < layout.Size<3>()
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   int operator()(int idx0, int idx1, int idx2, Idx... idx)
   {
      return layout(idx0,idx1,idx2,i,idx...);
   }
};

template <typename Layout>
class RestrictedLayout<4,Layout>
{
private:
   const int i;
   Layout &layout;

public:
   MFEM_HOST_DEVICE
   RestrictedLayout(int i, Layout &layout): i(i), layout(layout)
   {
      // TODO Check i < layout.Size<4>()
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   int operator()(int idx0, int idx1, int idx2, int idx3, Idx... idx)
   {
      return layout(idx0,idx1,idx2,idx3,i,idx...);
   }
};

template <typename Layout>
class RestrictedLayout<5,Layout>
{
private:
   const int i;
   Layout &layout;

public:
   MFEM_HOST_DEVICE
   RestrictedLayout(int i, Layout &layout): i(i), layout(layout)
   {
      // TODO Check i < layout.Size<5>()
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   int operator()(int idx0, int idx1, int idx2, int idx3, int idx4, Idx... idx)
   {
      return layout(idx0,idx1,idx2,idx3,idx4,i,idx...);
   }
};

template <typename Layout>
class RestrictedLayout<6,Layout>
{
private:
   const int i;
   Layout &layout;

public:
   MFEM_HOST_DEVICE
   RestrictedLayout(int i, Layout &layout): i(i), layout(layout)
   {
      // TODO Check i < layout.Size<6>()
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   int operator()(int idx0, int idx1, int idx2, int idx3, int idx4, int idx5, Idx... idx)
   {
      return layout(idx0,idx1,idx2,idx3,idx4,idx5,i,idx...);
   }
};

template <typename Layout>
class RestrictedLayout<7,Layout>
{
private:
   const int i;
   Layout &layout;

public:
   MFEM_HOST_DEVICE
   RestrictedLayout(int i, Layout &layout): i(i)
   {
      // TODO Check i < layout.Size<7>()
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   int operator()(int idx0, int idx1, int idx2, int idx3, int idx4, int idx5, int idx6, Idx... idx)
   {
      return layout(idx0,idx1,idx2,idx3,idx4,idx5,idx6,i,idx...);
   }
};

template <typename Layout>
class RestrictedLayout<8,Layout>
{
private:
   const int i;
   Layout &layout;

public:
   MFEM_HOST_DEVICE
   RestrictedLayout(int i, Layout &layout): i(i)
   {
      // TODO Check i < layout.Size<8>()
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   int operator()(int idx0, int idx1, int idx2, int idx3,int idx4, int idx5, int idx6, int idx7, Idx... idx)
   {
      return layout(idx0,idx1,idx2,idx3,idx4,idx5,idx6,idx7,i,idx...);
   }
};

/// Layout for Nedelec finite elements dofs TODO
class NDLayout;

/// Layout for Raviart-Thomas finite elements dofs TODO
class RTLayout;

} // namespace mfem

#endif // MFEM_LAYOUT
