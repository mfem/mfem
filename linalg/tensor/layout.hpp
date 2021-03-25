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
   DynamicLayout(int arg0, Sizes... args)
   {
      Init<Rank>::result(sizes, arg0, args...);
   }

   template <typename Layout>
   DynamicLayout(const Layout &rhs)
   {
      Init<Rank>::result(sizes,rhs);
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   int index(Idx... idx) const
   {
      static_assert(Rank==sizeof...(Idx), "Wrong number of arguments.");
      return DynamicTensorIndex<Rank>::eval(sizes, idx...);
   }

   template <int N>
   int Size() const
   {
      static_assert(N>=0 && N<Rank,"Accessed size is higher than the rank of the Tensor.");
      return sizes[N];
   }

private:
   /// A Class to compute the real index from the multi-indices of a tensor
   template <int Dim, int N = 1>
   class DynamicTensorIndex
   {
   public:
      template <typename... Args> MFEM_HOST_DEVICE
      static inline int eval(const int* sizes, int first, Args... args)
      {
   #if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
         MFEM_VERIFY(first<sizes[N-1],"Trying to access out of boundary.");
   #endif
         return first + sizes[N - 1] * DynamicTensorIndex<Dim,N+1>::eval(sizes, args...);
      }
   };

   // Terminal case
   template <int Dim>
   class DynamicTensorIndex<Dim, Dim>
   {
   public:
      MFEM_HOST_DEVICE
      static inline int eval(const int* sizes, int first)
      {
   #if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
         MFEM_VERIFY(first<sizes[Dim-1],"Trying to access out of boundary.");
   #endif
         return first;
      }
   };

   /// A class to initialize the size of a Tensor
   template <int Dim, int N = 1>
   class Init
   {
   public:
      template <typename... Args>
      static inline void result(int* sizes, int first, Args... args)
      {
         sizes[N - 1] = first;
         Init<Dim,N+1>::result(sizes, args...);
      }

      template <typename Layout>
      static inline void result(int* sizes, const Layout &rhs)
      {
         sizes[N - 1] = rhs.template Size<N-1>();
         Init<Dim,N+1>::result(sizes, rhs);
      }
   };

   // Terminal case
   template <int Dim>
   class Init<Dim, Dim>
   {
   public:
      template <typename... Args>
      static inline void result(int* sizes, int first, Args... args)
      {
         sizes[Dim - 1] = first;
      }

      template <typename Layout>
      static inline void result(int* sizes, const Layout &rhs)
      {
         sizes[Dim - 1] = rhs.template Size<Dim-1>();
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

   template <typename Layout> MFEM_HOST_DEVICE
   StaticLayout(const Layout& rhs)
   {
      // for (int i = 0; i < Rank; i++)
      // {
      //    MFEM_VERIFY(Sizes...[i] == lhs.Size<i>());
      // }
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   constexpr int index(Idx... idx) const
   {
   #if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
      static_assert(sizeof...(Sizes)==sizeof...(Idx), "Wrong number of arguments.");
   #endif
      return StaticTensorIndex<Sizes...>::eval(idx...);
   }

   template <int N>
   constexpr int Size() const
   {
      static_assert(N>=0 && N<sizeof...(Sizes),"Accessed size is higher than the rank of the Tensor.");
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
template <int BatchSize, int... Dims>
class BlockLayout;

template <int BatchSize, int DimX>
class BlockLayout<BatchSize, DimX>
{
public:
   static constexpr int batch_size = BatchSize;

   MFEM_HOST_DEVICE inline
   BlockLayout(int size0)
   {
      // TODO Verify in debug that size0==DimX
      // TODO verify that size0 < BlockSizeX
      // TODO verify that BlockSizeZ == BatchSize
   }

   MFEM_HOST_DEVICE inline
   constexpr int index(int idx0) const
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

template <int BatchSize, int DimX, int DimY>
class BlockLayout<BatchSize, DimX, DimY>
{
public:
   static constexpr int batch_size = BatchSize;

   MFEM_HOST_DEVICE inline
   BlockLayout(int size0, int size1)
   {
      // TODO Verify in debug that size0==DimX && size1==DimY
      // TODO verify that size0 < BlockSizeX && size1 < BlockSizeY
      // TODO verify that BlockSizeZ == BatchSize
   }

   MFEM_HOST_DEVICE inline
   constexpr int index(int idx0, int idx1) const
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

template <int BatchSize, int DimX, int DimY, int... Dims>
class BlockLayout<BatchSize, DimX, DimY, Dims...>
{
private:
   StaticLayout<Dims...> layout;
public:
   static constexpr int batch_size = BatchSize;

   template <typename... Sizes> MFEM_HOST_DEVICE inline
   BlockLayout(int size0, int size1, Sizes... sizes)
   : layout(sizes...)
   {
      // TODO Verify in debug that size0==DimX && size1==DimY
      // TODO verify that size0 < BlockSizeX && size1 < BlockSizeY
      // TODO verify that BlockSizeZ == BatchSize
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   constexpr int index(int idx0, int idx1, Idx... idx) const
   {
      // TODO verify that idx0 < DimX && idx1 < DimY && idx2 < DimZ
      // TODO verify that idx0 == threadIdx.x && idx1 == threadIdx.y
      return layout(idx...);
   }

   // Can be constexpr if Tensor inherit from Layout
   template <int N> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      static_assert(N>=0 && N<rank(DimX,DimY,Dims...),"Accessed size is higher than the rank of the Tensor.");
      return Dim<N,DimX,DimY,Dims...>::val;
   }
};

template <int Rank, int BatchSize>
class DynamicBlockLayout
{
private:
   const int size0;
   const int size1;
   DynamicLayout<Rank-2> layout;
public:
   static constexpr int batch_size = BatchSize;

   template <typename... Sizes> MFEM_HOST_DEVICE inline
   DynamicBlockLayout(int size0, int size1,  Sizes... sizes)
   : size0(size0), size1(size1), layout(sizes...)
   {
      // TODO verify that size0 < BlockSizeX && size1 < BlockSizeY
      // TODO verify that BlockSizeZ == BatchSize
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   constexpr int index(int idx0, int idx1, Idx... idx) const
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
      return BlockSize<N>::eval(size0,size1,layout);
   }

private:
   template <int N>
   struct BlockSize
   {
      static int eval(int size0, int size1, const DynamicLayout<Rank-2> &layout)
      {
         return layout.template Size<N-2>();
      }
   };

   template <>
   struct BlockSize<0>
   {
      static int eval(int size0, int size1, const DynamicLayout<Rank-2> &layout)
      {
         return size0;
      }
   };

   template <>
   struct BlockSize<1>
   {
      static int eval(int size0, int size1, const DynamicLayout<Rank-2> &layout)
      {
         return size1;
      }
   };
};

template <int BatchSize>
class DynamicBlockLayout<1,BatchSize>
{
private:
   const int size0;
public:
   static constexpr int batch_size = BatchSize;

   MFEM_HOST_DEVICE inline
   DynamicBlockLayout(int size0)
   : size0(size0)
   {
      // TODO verify that size0 < BlockSizeX
      // TODO verify that BlockSizeZ == BatchSize
   }

   MFEM_HOST_DEVICE inline
   constexpr int index(int idx) const
   {
      // TODO verify that idx < DimX
      return 0;
   }

   // Can be constexpr if Tensor inherit from Layout
   template <int N> MFEM_HOST_DEVICE inline
   constexpr int Size() const
   {
      static_assert(N==0,"Accessed size is higher than the rank of the Tensor.");
      return size0;
   }
};

template <int BatchSize>
class DynamicBlockLayout<2,BatchSize>
{
private:
   const int size0;
   const int size1;
public:
   static constexpr int batch_size = BatchSize;

   MFEM_HOST_DEVICE inline
   DynamicBlockLayout(int size0, int size1)
   : size0(size0), size1(size1)
   {
      // TODO verify that size0 < BlockSizeX && size1 < BlockSizeY
      // TODO verify that BlockSizeZ == BatchSize
   }

   MFEM_HOST_DEVICE inline
   constexpr int index(int idx0, int idx1) const
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
   int sizes[Rank];

public:
   template <typename... Idx> MFEM_HOST_DEVICE inline
   constexpr int index(Idx... idx) const
   {
      static_assert(sizeof...(Idx)==Rank,"Wrong number of argumets.");
      return StridedIndex<1>::eval(offsets, strides, idx...);
   }

   // Can be constexpr if Tensor inherit from Layout
   template <int N> MFEM_HOST_DEVICE inline
   int Size() const
   {
      static_assert(N>=0 && N<Rank,"Accessed size is higher than the rank of the Tensor.");
      return sizes[N];
   }

private:
   template <int N>
   struct StridedIndex
   {
      template <typename... Idx>
      static inline int eval(int* offsets, int* strides, int first, Idx... args)
      {
         return (offsets[N-1]+first)*strides[N-1] + StridedIndex<N+1>::eval(args...);
      }
   };

   template <>
   struct StridedIndex<Rank>
   {
      template <typename... Idx>
      static inline int eval(int* offsets, int* strides, int first)
      {
         return (offsets[Rank-1]+first)*strides[Rank-1];
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
   const Layout &layout;

public:
   MFEM_HOST_DEVICE
   RestrictedLayout(int i, const Layout &layout): i(i), layout(layout)
   {
      // TODO Check i < layout.Size<0>()
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   constexpr int index(Idx... idx) const
   {
      return layout.index(i,idx...);
   }

   template <int M>
   int Size() const
   {
      return layout.template Size<M+1>();
   }
};

template <typename Layout>
class RestrictedLayout<1,Layout>
{
private:
   const int i;
   const Layout &layout;

public:
   MFEM_HOST_DEVICE
   RestrictedLayout(int i, const Layout &layout): i(i), layout(layout)
   {
      // TODO Check i < layout.Size<1>()
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   constexpr int index(int idx0, Idx... idx) const
   {
      return layout.index(idx0,i,idx...);
   }

   template <int M>
   constexpr int Size() const
   {
      return layout.template Size<(M<1?M:M+1)>();
   }
};

template <typename Layout>
class RestrictedLayout<2,Layout>
{
private:
   const int i;
   const Layout &layout;

public:
   MFEM_HOST_DEVICE
   RestrictedLayout(int i, const Layout &layout): i(i), layout(layout)
   {
      // TODO Check i < layout.Size<2>()
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   constexpr int index(int idx0, int idx1, Idx... idx) const
   {
      return layout.index(idx0,idx1,i,idx...);
   }

   template <int M>
   constexpr int Size() const
   {
      return layout.template Size<(M<2?M:M+1)>();
   }
};

template <typename Layout>
class RestrictedLayout<3,Layout>
{
private:
   const int i;
   const Layout &layout;

public:
   MFEM_HOST_DEVICE
   RestrictedLayout(int i, const Layout &layout): i(i), layout(layout)
   {
      // TODO Check i < layout.Size<3>()
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   constexpr int index(int idx0, int idx1, int idx2, Idx... idx) const
   {
      return layout.index(idx0,idx1,idx2,i,idx...);
   }

   template <int M>
   constexpr int Size() const
   {
      return layout.template Size<(M<3?M:M+1)>();
   }
};

template <typename Layout>
class RestrictedLayout<4,Layout>
{
private:
   const int i;
   const Layout &layout;

public:
   MFEM_HOST_DEVICE
   RestrictedLayout(int i, const Layout &layout): i(i), layout(layout)
   {
      // TODO Check i < layout.Size<4>()
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   constexpr int index(int idx0, int idx1, int idx2, int idx3, Idx... idx) const
   {
      return layout.index(idx0,idx1,idx2,idx3,i,idx...);
   }

   template <int M>
   constexpr int Size() const
   {
      return layout.template Size<(M<4?M:M+1)>();
   }
};

template <typename Layout>
class RestrictedLayout<5,Layout>
{
private:
   const int i;
   const Layout &layout;

public:
   MFEM_HOST_DEVICE
   RestrictedLayout(int i, const Layout &layout): i(i), layout(layout)
   {
      // TODO Check i < layout.Size<5>()
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   constexpr int index(int idx0, int idx1, int idx2, int idx3, int idx4, Idx... idx) const
   {
      return layout.index(idx0,idx1,idx2,idx3,idx4,i,idx...);
   }

   template <int M>
   constexpr int Size() const
   {
      return layout.template Size<(M<5?M:M+1)>();
   }
};

template <typename Layout>
class RestrictedLayout<6,Layout>
{
private:
   const int i;
   const Layout &layout;

public:
   MFEM_HOST_DEVICE
   RestrictedLayout(int i, const Layout &layout): i(i), layout(layout)
   {
      // TODO Check i < layout.Size<6>()
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   constexpr int index(int idx0, int idx1, int idx2, int idx3, int idx4, int idx5, Idx... idx) const
   {
      return layout.index(idx0,idx1,idx2,idx3,idx4,idx5,i,idx...);
   }

   template <int M>
   constexpr int Size() const
   {
      return layout.template Size<(M<6?M:M+1)>();
   }
};

template <typename Layout>
class RestrictedLayout<7,Layout>
{
private:
   const int i;
   const Layout &layout;

public:
   MFEM_HOST_DEVICE
   RestrictedLayout(int i, const Layout &layout): i(i)
   {
      // TODO Check i < layout.Size<7>()
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   constexpr int index(int idx0, int idx1, int idx2, int idx3, int idx4, int idx5, int idx6, Idx... idx) const
   {
      return layout.index(idx0,idx1,idx2,idx3,idx4,idx5,idx6,i,idx...);
   }

   template <int M>
   constexpr int Size() const
   {
      return layout.template Size<(M<7?M:M+1)>();
   }
};

template <typename Layout>
class RestrictedLayout<8,Layout>
{
private:
   const int i;
   const Layout &layout;

public:
   MFEM_HOST_DEVICE
   RestrictedLayout(int i, const Layout &layout): i(i)
   {
      // TODO Check i < layout.Size<8>()
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   constexpr int index(int idx0, int idx1, int idx2, int idx3,int idx4, int idx5, int idx6, int idx7, Idx... idx) const
   {
      return layout.index(idx0,idx1,idx2,idx3,idx4,idx5,idx6,idx7,i,idx...);
   }

   template <int M>
   constexpr int Size() const
   {
      return layout.template Size<(M<8?M:M+1)>();
   }
};

/// Layout for Nedelec finite elements dofs TODO
class NDLayout;

/// Layout for Raviart-Thomas finite elements dofs TODO
class RTLayout;

/////////////////
// Layout Traits

// is_dynamic_layout
template <typename Layout>
struct is_dynamic_layout
{
   static constexpr bool value = false;
};

template<int Rank>
struct is_dynamic_layout<DynamicLayout<Rank>>
{
   static constexpr bool value = true;
};

template <int Rank, int BatchSize>
struct is_dynamic_layout<DynamicBlockLayout<Rank,BatchSize>>
{
   static constexpr bool value = true;
};


template <int N, typename Layout>
struct is_dynamic_layout<RestrictedLayout<N,Layout>>
{
   static constexpr bool value = is_dynamic_layout<Layout>::value;
};

// is_static_layout
template <typename Layout>
struct is_static_layout
{
   static constexpr bool value = false;
};

template <int BatchSize, int... Dims>
struct is_static_layout<BlockLayout<BatchSize,Dims...>>
{
   static constexpr bool value = true;
};

template<int... Dims>
struct is_static_layout<StaticLayout<Dims...>>
{
   static constexpr bool value = true;
};

template <int N, typename Layout>
struct is_static_layout<RestrictedLayout<N,Layout>>
{
   static constexpr bool value = is_static_layout<Layout>::value;
};

// is_serial_layout
template <typename Layout>
struct is_serial_layout
{
   static constexpr bool value = false;
};

template<int... Dims>
struct is_serial_layout<StaticLayout<Dims...>>
{
   static constexpr bool value = true;
};

template <int Rank>
struct is_serial_layout<DynamicLayout<Rank>>
{
   static constexpr bool value = true;
};

template <int N, typename Layout>
struct is_serial_layout<RestrictedLayout<N,Layout>>
{
   static constexpr bool value = is_serial_layout<Layout>::value;
};

// is_2d_threaded_layout
template <typename Layout>
struct is_2d_threaded_layout
{
   static constexpr bool value = false;
};

template <int BatchSize, int... Dims>
struct is_2d_threaded_layout<BlockLayout<BatchSize,Dims...>>
{
   static constexpr bool value = true;
};

template <int Rank, int BatchSize>
struct is_2d_threaded_layout<DynamicBlockLayout<Rank,BatchSize>>
{
   static constexpr bool value = true;
};

template <int N, typename Layout>
struct is_2d_threaded_layout<RestrictedLayout<N,Layout>>
{
   static constexpr bool value = is_2d_threaded_layout<Layout>::value;
};

// get_layout_batch_size
template <typename Layout>
struct get_layout_batch_size
{
   static constexpr int value = 1;
};

template <int BatchSize, int... Dims>
struct get_layout_batch_size<BlockLayout<BatchSize, Dims...>>
{
   static constexpr int value = BatchSize;
};

template <int Rank, int BatchSize>
struct get_layout_batch_size<DynamicBlockLayout<Rank, BatchSize>>
{
   static constexpr int value = BatchSize;
};

template <int N, typename Layout>
struct get_layout_batch_size<RestrictedLayout<N, Layout>>
{
   static constexpr int value = get_layout_batch_size<Layout>::value;
};

} // namespace mfem

#endif // MFEM_LAYOUT
