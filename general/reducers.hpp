// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_REDUCERS_HPP
#define MFEM_REDUCERS_HPP

#include "array.hpp"
#include "forall.hpp"

#include <cmath>
#include <limits>
#include <type_traits>

namespace mfem
{

/// Pair of values which can be used in device code
template <class A, class B> struct DevicePair
{
   A first;
   B second;
};

/// Two pairs for the min/max values and their location indices
template <class A, class B> struct MinMaxLocScalar
{
   A min_val;
   A max_val;
   B min_loc;
   B max_loc;
};

/// @brief a += b
template <class T> struct SumReducer
{
   using value_type = T;
   static MFEM_HOST_DEVICE void Join(value_type &a, const value_type &b)
   {
      a += b;
   }

   static MFEM_HOST_DEVICE void SetInitialValue(value_type &a) { a = T(0); }
};

/// @brief a *= b
template <class T> struct MultReducer
{
   using value_type = T;
   static MFEM_HOST_DEVICE void Join(value_type &a, const value_type &b)
   {
      a *= b;
   }

   static MFEM_HOST_DEVICE void SetInitialValue(value_type &a) { a = T(1); }
};

/// @brief a &= b
template <class T> struct BAndReducer
{
   static_assert(std::is_integral<T>::value, "Only works for integral types");
   using value_type = T;
   static MFEM_HOST_DEVICE void Join(value_type &a, const value_type &b)
   {
      a &= b;
   }

   static MFEM_HOST_DEVICE void SetInitialValue(value_type &a)
   {
      // sets all bits, does not work for floating point types
      // bitwise operators are not defined for floating point types anyways
      a = ~T(0);
   }
};

/// @brief a |= b
template <class T> struct BOrReducer
{
   static_assert(std::is_integral<T>::value, "Only works for integral types");
   using value_type = T;
   static MFEM_HOST_DEVICE void Join(value_type &a, const value_type &b)
   {
      a |= b;
   }

   static MFEM_HOST_DEVICE void SetInitialValue(T &a) { a = T(0); }
};

/// @brief a = min(a,b)
template <class T> struct MinReducer
{
   using value_type = T;

   static MFEM_HOST_DEVICE void Join(value_type &a, value_type b)
   {
      if (b < a)
      {
         a = b;
      }
   }

   // If we use std::numeric_limits<T>::max() in host-device method, Cuda
   // complains about calling host-only constexpr functions in device code
   // without --expt-relaxed-constexpr, so we define the following constant as a
   // workaround for the Cuda warning.
   static constexpr T max_val = std::numeric_limits<T>::max();

   static MFEM_HOST_DEVICE void SetInitialValue(value_type &a)
   {
      a = max_val;
   }
};

template <> struct MinReducer<float>
{
   using value_type = float;
   static MFEM_HOST_DEVICE void Join(value_type &a, value_type b)
   {
      a = fmin(a, b);
   }

   static MFEM_HOST_DEVICE void SetInitialValue(value_type &a) { a = HUGE_VALF; }
};

template <> struct MinReducer<double>
{
   using value_type = double;
   static MFEM_HOST_DEVICE void Join(value_type &a, value_type b)
   {
      a = fmin(a, b);
   }

   static MFEM_HOST_DEVICE void SetInitialValue(value_type &a) { a = HUGE_VAL; }
};

/// @brief a = max(a,b)
template <class T> struct MaxReducer
{
   using value_type = T;
   static MFEM_HOST_DEVICE void Join(value_type &a, value_type b)
   {
      if (a < b)
      {
         a = b;
      }
   }

   // If we use std::numeric_limits<T>::min() in host-device method, Cuda
   // complains about calling host-only constexpr functions in device code
   // without --expt-relaxed-constexpr, so we define the following constant as a
   // workaround for the Cuda warning.
   static constexpr T min_val = std::numeric_limits<T>::min();

   static MFEM_HOST_DEVICE void SetInitialValue(value_type &a)
   {
      a = min_val;
   }
};

template <> struct MaxReducer<float>
{
   using value_type = float;
   static MFEM_HOST_DEVICE void Join(value_type &a, value_type b)
   {
      a = fmax(a, b);
   }

   static MFEM_HOST_DEVICE void SetInitialValue(value_type &a)
   {
      a = -HUGE_VALF;
   }
};

template <> struct MaxReducer<double>
{
   using value_type = double;
   static MFEM_HOST_DEVICE void Join(value_type &a, value_type b)
   {
      a = fmax(a, b);
   }

   static MFEM_HOST_DEVICE void SetInitialValue(value_type &a) { a = -HUGE_VAL; }
};

/// @brief a = minmax(a,b)
template <class T> struct MinMaxReducer
{
   using value_type = DevicePair<T, T>;
   static MFEM_HOST_DEVICE void Join(value_type &a, const value_type &b)
   {
      if (b.first < a.first)
      {
         a.first = b.first;
      }
      if (b.second > a.second)
      {
         a.second = b.second;
      }
   }

   // If we use std::numeric_limits<T>::min() (or max()) in host-device method,
   // Cuda complains about calling host-only constexpr functions in device code
   // without --expt-relaxed-constexpr, so we define the following constants as
   // a workaround for the Cuda warning.
   static constexpr T min_val = std::numeric_limits<T>::min();
   static constexpr T max_val = std::numeric_limits<T>::max();

   static MFEM_HOST_DEVICE void SetInitialValue(value_type &a)
   {
      a = value_type{max_val, min_val};
   }
};

template <> struct MinMaxReducer<float>
{
   using value_type = DevicePair<float, float>;
   static MFEM_HOST_DEVICE void Join(value_type &a, value_type b)
   {
      a.first = fmin(a.first, b.first);
      a.second = fmax(a.second, b.second);
   }

   static MFEM_HOST_DEVICE void SetInitialValue(value_type &a)
   {
      a = value_type{HUGE_VALF, -HUGE_VALF};
   }
};

template <> struct MinMaxReducer<double>
{
   using value_type = DevicePair<double, double>;
   static MFEM_HOST_DEVICE void Join(value_type &a, value_type b)
   {
      a.first = fmin(a.first, b.first);
      a.second = fmax(a.second, b.second);
   }

   static MFEM_HOST_DEVICE void SetInitialValue(value_type &a)
   {
      a = value_type{HUGE_VAL, -HUGE_VAL};
   }
};

/// @brief i = argmin(a[i], a[j])
///
/// Note: for ties the returned index can correspond to any min entry, not
/// necesarily the first one
template <class T, class I> struct ArgMinReducer
{
   using value_type = DevicePair<T, I>;
   static MFEM_HOST_DEVICE void Join(value_type &a, const value_type &b)
   {
      if (b.first <= a.first)
      {
         a = b;
      }
   }
   static MFEM_HOST_DEVICE void SetInitialValue(value_type &a)
   {
      // Cuda complains about calling host-only constexpr functions in device
      // code without --expt-relaxed-constexpr, wrap into integral_constant to
      // get around this
      a = value_type
      {
         std::integral_constant<T, std::numeric_limits<T>::max()>::value, I{0}};
   }
};

template <class I> struct ArgMinReducer<float, I>
{
   using value_type = DevicePair<float, I>;
   static MFEM_HOST_DEVICE void Join(value_type &a, const value_type &b)
   {
      if (b.first <= a.first)
      {
         a = b;
      }
   }
   static MFEM_HOST_DEVICE void SetInitialValue(value_type &a)
   {
      a = value_type{HUGE_VALF, I{0}};
   }
};

template <class I> struct ArgMinReducer<double, I>
{
   using value_type = DevicePair<double, I>;
   static MFEM_HOST_DEVICE void Join(value_type &a, const value_type &b)
   {
      if (b.first <= a.first)
      {
         a = b;
      }
   }
   static MFEM_HOST_DEVICE void SetInitialValue(value_type &a)
   {
      a = value_type{HUGE_VAL, I{0}};
   }
};

/// @brief i = argmax(a[i], a[j])
///
/// Note: for ties the returned index can correspond to any min entry, not
/// necesarily the first one.
template <class T, class I> struct ArgMaxReducer
{
   using value_type = DevicePair<T, I>;
   static MFEM_HOST_DEVICE void Join(value_type &a, const value_type &b)
   {
      if (a.first <= b.first)
      {
         a = b;
      }
   }
   static MFEM_HOST_DEVICE void SetInitialValue(value_type &a)
   {
      // Cuda complains about calling host-only constexpr functions in device
      // code without --expt-relaxed-constexpr, wrap into integral_constant to
      // get around this
      a = value_type
      {
         std::integral_constant<T, std::numeric_limits<T>::max()>::value, I{0}};
   }
};

template <class I> struct ArgMaxReducer<float, I>
{
   using value_type = DevicePair<float, I>;
   static MFEM_HOST_DEVICE void Join(value_type &a, const value_type &b)
   {
      if (a.first <= b.first)
      {
         a = b;
      }
   }
   static MFEM_HOST_DEVICE void SetInitialValue(value_type &a)
   {
      a = value_type{-HUGE_VALF, I{0}};
   }
};

template <class I> struct ArgMaxReducer<double, I>
{
   using value_type = DevicePair<double, I>;
   static MFEM_HOST_DEVICE void Join(value_type &a, const value_type &b)
   {
      if (a.first <= b.first)
      {
         a = b;
      }
   }
   static MFEM_HOST_DEVICE void SetInitialValue(value_type &a)
   {
      a = value_type{-HUGE_VAL, I{0}};
   }
};

template <class T, class I> struct ArgMinMaxReducer
{
   using value_type = MinMaxLocScalar<T, I>;
   static MFEM_HOST_DEVICE void Join(value_type &a, const value_type &b)
   {
      if (b.min_val <= a.min_val)
      {
         a.min_val = b.min_val;
         a.min_loc = b.min_loc;
      }
      if (b.max_val >= a.max_val)
      {
         a.max_val = b.max_val;
         a.max_loc = b.max_loc;
      }
   }

   static MFEM_HOST_DEVICE void SetInitialValue(value_type &a)
   {
      // Cuda complains about calling host-only constexpr functions in device
      // code without --expt-relaxed-constexpr, wrap into integral_constant to
      // get around this
      a = value_type
      {
         std::integral_constant<T, std::numeric_limits<T>::max()>::value,
         std::integral_constant<T, std::numeric_limits<T>::min()>::value, I(0),
         I(0)};
   }
};

template <class I> struct ArgMinMaxReducer<float, I>
{
   using value_type = MinMaxLocScalar<float, I>;
   static MFEM_HOST_DEVICE void Join(value_type &a, const value_type &b)
   {
      if (b.min_val <= a.min_val)
      {
         a.min_val = b.min_val;
         a.min_loc = b.min_loc;
      }
      if (b.max_val >= a.max_val)
      {
         a.max_val = b.max_val;
         a.max_loc = b.max_loc;
      }
   }

   static MFEM_HOST_DEVICE void SetInitialValue(value_type &a)
   {
      a = value_type{HUGE_VALF, -HUGE_VALF, I(0), I(0)};
   }
};

template <class I> struct ArgMinMaxReducer<double, I>
{
   using value_type = MinMaxLocScalar<double, I>;
   static MFEM_HOST_DEVICE void Join(value_type &a, const value_type &b)
   {
      if (b.min_val <= a.min_val)
      {
         a.min_val = b.min_val;
         a.min_loc = b.min_loc;
      }
      if (b.max_val >= a.max_val)
      {
         a.max_val = b.max_val;
         a.max_loc = b.max_loc;
      }
   }

   static MFEM_HOST_DEVICE void SetInitialValue(value_type &a)
   {
      a = value_type{HUGE_VAL, -HUGE_VAL, I(0), I(0)};
   }
};

namespace internal
{

/**
 @brief Device portion of a reduction over a 1D sequence [0, N)
 @tparam B Reduction body. Must be callable with the signature void(int i, value_type&
 v), where i is the index to evaluate and v is the value to update.
 @tparam R Reducer capable of combining values of type value_type. See reducers.hpp for
 pre-defined reducers.
 */
template<class B, class R> struct reduction_kernel
{
   /// value type body and reducer operate on.
   using value_type = typename R::value_type;
   /// workspace for the intermediate reduction results
   mutable value_type *work;
   B body;
   R reducer;
   /// Length of sequence to reduce over.
   int N;
   /// How many items is each thread responsible for during the serial phase
   int items_per_thread;

   constexpr static MFEM_HOST_DEVICE int max_blocksize() { return 256; }

   /// helper for computing the reduction block size
   static int block_log2(unsigned N)
   {
#if defined(__GNUC__) || defined(__clang__)
      return N ? (sizeof(unsigned) * 8 - __builtin_clz(N)) : 0;
#elif defined(_MSC_VER)
      return sizeof(unsigned) * 8 - __lzclz(N);
#else
      int res = 0;
      while (N)
      {
         N >>= 1;
         ++res;
      }
      return res;
#endif
   }

   MFEM_HOST_DEVICE void operator()(int work_idx) const
   {
      MFEM_SHARED value_type buffer[max_blocksize()];
      reducer.SetInitialValue(buffer[MFEM_THREAD_ID(x)]);
      // serial part
      for (int idx = 0; idx < items_per_thread; ++idx)
      {
         int i = MFEM_THREAD_ID(x) +
                 (idx + work_idx * items_per_thread) * MFEM_THREAD_SIZE(x);
         if (i < N)
         {
            body(i, buffer[MFEM_THREAD_ID(x)]);
         }
         else
         {
            break;
         }
      }
      // binary tree reduction
      for (int i = (MFEM_THREAD_SIZE(x) >> 1); i > 0; i >>= 1)
      {
         MFEM_SYNC_THREAD;
         if (MFEM_THREAD_ID(x) < i)
         {
            reducer.Join(buffer[MFEM_THREAD_ID(x)], buffer[MFEM_THREAD_ID(x) + i]);
         }
      }
      if (MFEM_THREAD_ID(x) == 0)
      {
         work[work_idx] = buffer[0];
      }
   }
};
}

/**
 @brief Performs a 1D reduction on the range [0,N).
 @a res initial value and where the result will be written.
 @a body reduction function body.
 @a reducer helper for joining two reduced values.
 @a use_dev true to perform the reduction on the device, if possible.
 @a workspace temporary workspace used for device reductions. May be resized to
 a larger capacity as needed. Preferably should have MemoryType::MANAGED or
 MemoryType::HOST_PINNED. TODO: replace with internal temporary workspace
 vectors once that's added to the memory manager.
 @tparam T value_type to operate on
 */
template <class T, class B, class R>
void reduce(int N, T &res, B &&body, const R &reducer, bool use_dev,
            Array<T> &workspace)
{
   if (N == 0)
   {
      return;
   }

#if defined(MFEM_USE_CUDA_OR_HIP)
   if (use_dev &&
       mfem::Device::Allows(Backend::CUDA | Backend::HIP | Backend::RAJA_CUDA |
                            Backend::RAJA_HIP))
   {
      using red_type = internal::reduction_kernel<typename std::decay<B>::type,
            typename std::decay<R>::type>;
      // max block size is 256, but can be smaller
      int block_size = std::min<int>(red_type::max_blocksize(),
                                     1ll << red_type::block_log2(N));

      int num_mp = Device::NumMultiprocessors(Device::GetId());
#if defined(MFEM_USE_CUDA)
      // good value of mp_sat found experimentally on Lassen
      constexpr int mp_sat = 8;
#elif defined(MFEM_USE_HIP)
      // good value of mp_sat found experimentally on Tuolumne
      constexpr int mp_sat = 4;
#else
      num_mp = 1;
      constexpr int mp_sat = 1;
#endif
      // determine how many items each thread should sum during the serial
      // portion
      int nblocks = std::min(mp_sat * num_mp, (N + block_size - 1) / block_size);
      int items_per_thread =
         (N + block_size * nblocks - 1) / (block_size * nblocks);

      red_type red{nullptr, std::forward<B>(body), reducer, N, items_per_thread};
      // allocate res to fit block_size entries
      auto mt = workspace.GetMemory().GetMemoryType();
      if (mt != MemoryType::HOST_PINNED && mt != MemoryType::MANAGED)
      {
         mt = MemoryType::HOST_PINNED;
      }
      workspace.SetSize(nblocks, mt);
      auto work = workspace.HostWrite();
      red.work = work;
      forall_2D(nblocks, block_size, 1, std::move(red));
      // wait for results
      MFEM_DEVICE_SYNC;
      for (int i = 0; i < nblocks; ++i)
      {
         reducer.Join(res, work[i]);
      }
      return;
   }
#endif

   for (int i = 0; i < N; ++i)
   {
      body(i, res);
   }
}

} // namespace mfem

#endif // MFEM_REDUCERS_HPP
