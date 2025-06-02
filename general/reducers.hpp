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

#include "forall.hpp"

#include <climits>
#include <cmath>
#include <cstdint>
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

} // namespace mfem

#endif
