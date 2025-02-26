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

/// a += b
template <class T> struct SumReducer
{
   using value_type = T;
   static MFEM_HOST_DEVICE void Join(value_type &a, const value_type &b)
   {
      a += b;
   }

   static MFEM_HOST_DEVICE void SetInitialValue(value_type &a) { a = T(0); }
};

/// a *= b
template <class T> struct MultReducer
{
   using value_type = T;
   static MFEM_HOST_DEVICE void Join(value_type &a, const value_type &b)
   {
      a *= b;
   }

   static MFEM_HOST_DEVICE void SetInitialValue(value_type &a) { a = T(1); }
};

/// a &= b
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

/// a = min(a,b)
template <class T> struct MinReducer;

/// @brief a = max(a,b)
template <class T> struct MaxReducer;

/// @brief a = minmax(a,b)
template <class T> struct MinMaxReducer;

/// @brief i = argmin(a[i], a[j])
///
/// Note: for ties the returned index can correspond to any min entry, not
/// necesarily the first one
template <class T, class I> struct ArgMinReducer;

/// @brief i = argmax(a[i], a[j])
///
/// Note: for ties the returned index can correspond to any min entry, not
/// necesarily the first one.
template <class T, class I> struct ArgMaxReducer;

/// @brief i = argminmax(a[i], a[j])
///
/// Note: for ties the returned indices can correspond to any min/max entry, not
/// necesarily the first one.
template <class T, class I> struct ArgMinMaxReducer;

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

#define MFEM_STAMP_MIN_REDUCER(type, val)                                      \
  template <> struct MinReducer<type> {                                        \
    using value_type = type;                                                   \
    static MFEM_HOST_DEVICE void Join(value_type &a, value_type b) {           \
      if (b < a) {                                                             \
        a = b;                                                                 \
      }                                                                        \
    }                                                                          \
    static MFEM_HOST_DEVICE void SetInitialValue(value_type &a) {              \
      a = static_cast<type>(val);                                              \
    }                                                                          \
  }

MFEM_STAMP_MIN_REDUCER(bool, true);
MFEM_STAMP_MIN_REDUCER(char, CHAR_MAX);
MFEM_STAMP_MIN_REDUCER(signed char, SCHAR_MAX);
MFEM_STAMP_MIN_REDUCER(unsigned char, UCHAR_MAX);
MFEM_STAMP_MIN_REDUCER(wchar_t, WCHAR_MAX);
MFEM_STAMP_MIN_REDUCER(char16_t, UINT_LEAST16_MAX);
MFEM_STAMP_MIN_REDUCER(char32_t, UINT_LEAST32_MAX);
MFEM_STAMP_MIN_REDUCER(short, SHRT_MAX);
MFEM_STAMP_MIN_REDUCER(unsigned short, USHRT_MAX);
MFEM_STAMP_MIN_REDUCER(int, INT_MAX);
MFEM_STAMP_MIN_REDUCER(unsigned int, UINT_MAX);
MFEM_STAMP_MIN_REDUCER(long, LONG_MAX);
MFEM_STAMP_MIN_REDUCER(unsigned long, ULONG_MAX);
MFEM_STAMP_MIN_REDUCER(long long, LLONG_MAX);
MFEM_STAMP_MIN_REDUCER(unsigned long long, ULLONG_MAX);

#undef MFEM_STAMP_MIN_REDUCER

#define MFEM_STAMP_ARGMIN_REDUCER(type, val)                                   \
  template <class I> struct ArgMinReducer<type, I> {                           \
    using value_type = DevicePair<type, I>;                                    \
    static MFEM_HOST_DEVICE void Join(value_type &a, const value_type &b) {    \
      if (b.first <= a.first) {                                                \
        a = b;                                                                 \
      }                                                                        \
    }                                                                          \
    static MFEM_HOST_DEVICE void SetInitialValue(value_type &a) {              \
      a = value_type{static_cast<type>(val), I{0}};                            \
    }                                                                          \
  }

MFEM_STAMP_ARGMIN_REDUCER(bool, true);
MFEM_STAMP_ARGMIN_REDUCER(char, CHAR_MAX);
MFEM_STAMP_ARGMIN_REDUCER(signed char, SCHAR_MAX);
MFEM_STAMP_ARGMIN_REDUCER(unsigned char, UCHAR_MAX);
MFEM_STAMP_ARGMIN_REDUCER(wchar_t, WCHAR_MAX);
MFEM_STAMP_ARGMIN_REDUCER(char16_t, UINT_LEAST16_MAX);
MFEM_STAMP_ARGMIN_REDUCER(char32_t, UINT_LEAST32_MAX);
MFEM_STAMP_ARGMIN_REDUCER(short, SHRT_MAX);
MFEM_STAMP_ARGMIN_REDUCER(unsigned short, USHRT_MAX);
MFEM_STAMP_ARGMIN_REDUCER(int, INT_MAX);
MFEM_STAMP_ARGMIN_REDUCER(unsigned int, UINT_MAX);
MFEM_STAMP_ARGMIN_REDUCER(long, LONG_MAX);
MFEM_STAMP_ARGMIN_REDUCER(unsigned long, ULONG_MAX);
MFEM_STAMP_ARGMIN_REDUCER(long long, LLONG_MAX);
MFEM_STAMP_ARGMIN_REDUCER(unsigned long long, ULLONG_MAX);
// also use this for floats and doubles since we need the index as well
MFEM_STAMP_ARGMIN_REDUCER(float, HUGE_VALF);
MFEM_STAMP_ARGMIN_REDUCER(double, HUGE_VAL);

#undef MFEM_STAMP_ARGMIN_REDUCER

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

#define MFEM_STAMP_MAX_REDUCER(type, val)                                      \
  template <> struct MaxReducer<type> {                                        \
    using value_type = type;                                                   \
    static MFEM_HOST_DEVICE void Join(value_type &a, value_type b) {           \
      if (a < b) {                                                             \
        a = b;                                                                 \
      }                                                                        \
    }                                                                          \
    static MFEM_HOST_DEVICE void SetInitialValue(value_type &a) {              \
      a = static_cast<type>(val);                                              \
    }                                                                          \
  }

MFEM_STAMP_MAX_REDUCER(bool, false);
MFEM_STAMP_MAX_REDUCER(char, CHAR_MIN);
MFEM_STAMP_MAX_REDUCER(signed char, SCHAR_MIN);
MFEM_STAMP_MAX_REDUCER(unsigned char, 0);
MFEM_STAMP_MAX_REDUCER(wchar_t, WCHAR_MIN);
MFEM_STAMP_MAX_REDUCER(char16_t, 0);
MFEM_STAMP_MAX_REDUCER(char32_t, 0);
MFEM_STAMP_MAX_REDUCER(short, SHRT_MIN);
MFEM_STAMP_MAX_REDUCER(unsigned short, 0);
MFEM_STAMP_MAX_REDUCER(int, INT_MIN);
MFEM_STAMP_MAX_REDUCER(unsigned int, 0);
MFEM_STAMP_MAX_REDUCER(long, LONG_MIN);
MFEM_STAMP_MAX_REDUCER(unsigned long, 0);
MFEM_STAMP_MAX_REDUCER(long long, LLONG_MIN);
MFEM_STAMP_MAX_REDUCER(unsigned long long, 0);

#undef MFEM_STAMP_MAX_REDUCER

#define MFEM_STAMP_ARGMAX_REDUCER(type, val)                                   \
  template <class I> struct ArgMaxReducer<type, I> {                           \
    using value_type = DevicePair<type, I>;                                    \
    static MFEM_HOST_DEVICE void Join(value_type &a, const value_type &b) {    \
      if (a.first <= b.first) {                                                \
        a = b;                                                                 \
      }                                                                        \
    }                                                                          \
    static MFEM_HOST_DEVICE void SetInitialValue(value_type &a) {              \
      a = value_type{static_cast<type>(val), I{0}};                            \
    }                                                                          \
  }

MFEM_STAMP_ARGMAX_REDUCER(bool, false);
MFEM_STAMP_ARGMAX_REDUCER(char, CHAR_MIN);
MFEM_STAMP_ARGMAX_REDUCER(signed char, SCHAR_MIN);
MFEM_STAMP_ARGMAX_REDUCER(unsigned char, 0);
MFEM_STAMP_ARGMAX_REDUCER(wchar_t, WCHAR_MIN);
MFEM_STAMP_ARGMAX_REDUCER(char16_t, 0);
MFEM_STAMP_ARGMAX_REDUCER(char32_t, 0);
MFEM_STAMP_ARGMAX_REDUCER(short, SHRT_MIN);
MFEM_STAMP_ARGMAX_REDUCER(unsigned short, 0);
MFEM_STAMP_ARGMAX_REDUCER(int, INT_MIN);
MFEM_STAMP_ARGMAX_REDUCER(unsigned int, 0);
MFEM_STAMP_ARGMAX_REDUCER(long, LONG_MIN);
MFEM_STAMP_ARGMAX_REDUCER(unsigned long, 0);
MFEM_STAMP_ARGMAX_REDUCER(long long, LLONG_MIN);
MFEM_STAMP_ARGMAX_REDUCER(unsigned long long, 0);
// also use this for floats and doubles since we need the index as well
MFEM_STAMP_ARGMAX_REDUCER(float, -HUGE_VALF);
MFEM_STAMP_ARGMAX_REDUCER(double, -HUGE_VAL);

#undef MFEM_STAMP_ARGMAX_REDUCER

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

#define MFEM_STAMP_MINMAX_REDUCER(type, min_val, max_val)                      \
  template <> struct MinMaxReducer<type> {                                     \
    using value_type = DevicePair<type, type>;                                 \
    static MFEM_HOST_DEVICE void Join(value_type &a, const value_type &b) {    \
      if (b.first < a.first) {                                                 \
        a.first = b.first;                                                     \
      }                                                                        \
      if (b.second > a.second) {                                               \
        a.second = b.second;                                                   \
      }                                                                        \
    }                                                                          \
                                                                               \
    static MFEM_HOST_DEVICE void SetInitialValue(value_type &a) {              \
      a = value_type{static_cast<type>(max_val), static_cast<type>(min_val)};  \
    }                                                                          \
  }

MFEM_STAMP_MINMAX_REDUCER(bool, false, true);
MFEM_STAMP_MINMAX_REDUCER(char, CHAR_MIN, CHAR_MAX);
MFEM_STAMP_MINMAX_REDUCER(signed char, SCHAR_MIN, SCHAR_MAX);
MFEM_STAMP_MINMAX_REDUCER(unsigned char, 0, UCHAR_MAX);
MFEM_STAMP_MINMAX_REDUCER(wchar_t, WCHAR_MIN, WCHAR_MAX);
MFEM_STAMP_MINMAX_REDUCER(char16_t, 0, UINT_LEAST16_MAX);
MFEM_STAMP_MINMAX_REDUCER(char32_t, 0, UINT_LEAST32_MAX);
MFEM_STAMP_MINMAX_REDUCER(short, SHRT_MIN, SHRT_MAX);
MFEM_STAMP_MINMAX_REDUCER(unsigned short, 0, USHRT_MAX);
MFEM_STAMP_MINMAX_REDUCER(int, INT_MIN, INT_MAX);
MFEM_STAMP_MINMAX_REDUCER(unsigned int, 0, UINT_MAX);
MFEM_STAMP_MINMAX_REDUCER(long, LONG_MIN, LONG_MAX);
MFEM_STAMP_MINMAX_REDUCER(unsigned long, 0, ULONG_MAX);
MFEM_STAMP_MINMAX_REDUCER(long long, LLONG_MIN, LLONG_MAX);
MFEM_STAMP_MINMAX_REDUCER(unsigned long long, 0, ULLONG_MAX);

#undef MFEM_STAMP_MINMAX_REDUCER

#define MFEM_STAMP_ARGMINMAX_REDUCER(type, min_val_, max_val_)                 \
  template <class I> struct ArgMinMaxReducer<type, I> {                        \
    using value_type = MinMaxLocScalar<type, I>;                               \
    static MFEM_HOST_DEVICE void Join(value_type &a, const value_type &b) {    \
      if (b.min_val <= a.min_val) {                                            \
        a.min_val = b.min_val;                                                 \
        a.min_loc = b.min_loc;                                                 \
      }                                                                        \
      if (b.max_val >= a.max_val) {                                            \
        a.max_val = b.max_val;                                                 \
        a.max_loc = b.max_loc;                                                 \
      }                                                                        \
    }                                                                          \
                                                                               \
    static MFEM_HOST_DEVICE void SetInitialValue(value_type &a) {              \
      a = value_type{static_cast<type>(max_val_), static_cast<type>(min_val_), \
                     I(0), I(0)};                                              \
    }                                                                          \
  }

MFEM_STAMP_ARGMINMAX_REDUCER(bool, false, true);
MFEM_STAMP_ARGMINMAX_REDUCER(char, CHAR_MIN, CHAR_MAX);
MFEM_STAMP_ARGMINMAX_REDUCER(signed char, SCHAR_MIN, SCHAR_MAX);
MFEM_STAMP_ARGMINMAX_REDUCER(unsigned char, 0, UCHAR_MAX);
MFEM_STAMP_ARGMINMAX_REDUCER(wchar_t, WCHAR_MIN, WCHAR_MAX);
MFEM_STAMP_ARGMINMAX_REDUCER(char16_t, 0, UINT_LEAST16_MAX);
MFEM_STAMP_ARGMINMAX_REDUCER(char32_t, 0, UINT_LEAST32_MAX);
MFEM_STAMP_ARGMINMAX_REDUCER(short, SHRT_MIN, SHRT_MAX);
MFEM_STAMP_ARGMINMAX_REDUCER(unsigned short, 0, USHRT_MAX);
MFEM_STAMP_ARGMINMAX_REDUCER(int, INT_MIN, INT_MAX);
MFEM_STAMP_ARGMINMAX_REDUCER(unsigned int, 0, UINT_MAX);
MFEM_STAMP_ARGMINMAX_REDUCER(long, LONG_MIN, LONG_MAX);
MFEM_STAMP_ARGMINMAX_REDUCER(unsigned long, 0, ULONG_MAX);
MFEM_STAMP_ARGMINMAX_REDUCER(long long, LLONG_MIN, LLONG_MAX);
MFEM_STAMP_ARGMINMAX_REDUCER(unsigned long long, 0, ULLONG_MAX);
MFEM_STAMP_ARGMINMAX_REDUCER(float, -HUGE_VALF, HUGE_VALF);
MFEM_STAMP_ARGMINMAX_REDUCER(double, -HUGE_VAL, HUGE_VAL);

#undef MFEM_STAMP_ARGMINMAX_REDUCER

} // namespace mfem

#endif
