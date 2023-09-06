// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../config/tconfig.hpp" // MFEM_STATIC_ASSERT

#include "backends.hpp"

namespace mfem
{

namespace internal
{

/**
 * @brief The mdsmem class
 */
template <int rank, bool exclusive = true, typename T = double>
struct mdsmem
{
   static constexpr auto iseq = std::make_integer_sequence<int, rank> {};

   MFEM_HOST_DEVICE mdsmem() {}

   template<typename U>
   MFEM_HOST_DEVICE mdsmem(U* &smem, const int (& dimensions)[rank])
   {
      data = reinterpret_cast<T*>(smem);
      size = 1;
      for (int i = 0; i < rank; i++)
      {
         int id = rank - 1 - i;
         size *= dimensions[id];
         shape[id] = dimensions[id];
         strides[id] = (id == rank - 1) ? 1 : strides[id+1] * shape[id+1];
      }
      smem += exclusive ? size * sizeof(T) / sizeof(U) : 0;
   }

   template <int N, int R, typename S, typename... Args>
   struct Layout
   {
      MFEM_HOST_DEVICE
      static inline int ini(int* shape, int* strides, S k, Args... args)
      {
         shape[N - 1] = k;
         strides[N - 1] = Layout<N + 1, R, Args...>::ini(shape, strides, args...);
         return shape[N - 1] * strides[N - 1];
      }
   };

   template <int R, typename S, typename... Args>
   struct Layout<R, R, S, Args...>
   {
      MFEM_HOST_DEVICE
      static inline int ini(int* shape, int *strides, T k, Args...)
      {
         return (strides[R - 1] = 1, shape[R - 1] = k);
      }
   };

   template<typename U, typename... Args>
   MFEM_HOST_DEVICE mdsmem(U* &smem, Args... args)
   {
      data = reinterpret_cast<T*>(smem);
      MFEM_STATIC_ASSERT(sizeof...(args) == rank, "Wrong number of arguments");
      size = Layout<1, rank, Args...>::ini(shape, strides, args...);
      smem += exclusive ? size * sizeof(T) / sizeof(U) : 0;
   }

   template < typename ... index_types >
   MFEM_HOST_DEVICE auto & operator()(index_types ... indices)
   {
      static_assert(sizeof ... (indices) == rank);
      return data[index(iseq, indices...)];
   }

   template < typename ... index_types >
   MFEM_HOST_DEVICE auto & operator()(index_types ... indices) const
   {
      static_assert(sizeof ... (indices) == rank);
      return data[index(iseq, indices...)];
   }

   template < int ... I, typename ... index_types >
   MFEM_HOST_DEVICE auto index(std::integer_sequence<int, I...>, index_types
                               ... indices) const
   {
      return ((indices * strides[I]) + ...);
   }

   MFEM_HOST_DEVICE inline operator T *() const { return data; }

   T* data; // alignas(alignof(T));
   int size, shape[rank], strides[rank];
};

/**
 * @brief The mdsmem class
 */
template <int rank, typename T = double>
struct mdview
{
   static constexpr auto iseq = std::make_integer_sequence<int, rank> {};

   MFEM_HOST_DEVICE mdview() {}

   template <int N, int R, typename S, typename... Args>
   struct Layout
   {
      MFEM_HOST_DEVICE
      static inline int ini(int* shape, int* strides, S k, Args... args)
      {
         shape[N - 1] = k;
         strides[N - 1] = Layout<N + 1, R, Args...>::ini(shape, strides, args...);
         return shape[N - 1] * strides[N - 1];
      }
   };

   template <int R, typename S, typename... Args>
   struct Layout<R, R, S, Args...>
   {
      MFEM_HOST_DEVICE
      static inline int ini(int* shape, int *strides, T k, Args...)
      {
         return (strides[R - 1] = 1, shape[R - 1] = k);
      }
   };

   template<typename... Args>
   MFEM_HOST_DEVICE mdview(T* smem, Args... args)
   {
      data = smem;
      MFEM_STATIC_ASSERT(sizeof...(args) == rank, "Wrong number of arguments");
      size = Layout<1, rank, Args...>::ini(shape, strides, args...);
   }

   template < typename ... index_types >
   MFEM_HOST_DEVICE auto & operator()(index_types ... indices)
   {
      static_assert(sizeof ... (indices) == rank);
      return data[index(iseq, indices...)];
   }

   template < typename ... index_types >
   MFEM_HOST_DEVICE auto & operator()(index_types ... indices) const
   {
      static_assert(sizeof ... (indices) == rank);
      return data[index(iseq, indices...)];
   }

   template < int ... I, typename ... index_types >
   MFEM_HOST_DEVICE auto index(std::integer_sequence<int, I...>, index_types
                               ... indices) const
   {
      return ((indices * strides[I]) + ...);
   }

   MFEM_HOST_DEVICE inline operator T *() const { return data; }

   T* data;
   int size, shape[rank], strides[rank];
};

} // namespace internal

} // namespace mfem
