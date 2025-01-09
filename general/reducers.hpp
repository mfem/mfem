// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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

#include <type_traits>

namespace mfem {

/** @brief pair of values which can be used in device code */
template <class A, class B> struct DevicePair {
  A first;
  B second;
};

/** @brief two pairs for the min/max values and their location indices */
template <class A, class B> struct MinMaxLocScalar {
  A min_val, max_val;
  B min_loc, max_loc;
};

/** @brief a += b */
template <class T> struct SumReducer {
  using value_type = T;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    a += b;
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const { a = T(0); }
};

/** @brief a *= b */
template <class T> struct MultReducer {
  using value_type = T;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    a *= b;
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const { a = T(1); }
};

/** @brief a &= b */
template <class T> struct BAndReducer {
  static_assert(std::is_integral<T>::value, "Only works for integral types");
  using value_type = T;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    a &= b;
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    // hopefully this will set all bits in a
    a = T(-1);
  }
};

/** @brief a |= b */
template <class T> struct BOrReducer {
  static_assert(std::is_integral<T>::value, "Only works for integral types");
  using value_type = T;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    a |= b;
  }

  MFEM_HOST_DEVICE void init_val(T &a) const { a = T(0); }
};

/** @brief a = min(a,b) */
template <class T> struct MinReducer;

/** @brief a = max(a,b) */
template <class T> struct MaxReducer;

/** @brief a = minmax(a,b) */
template <class T> struct MinMaxReducer;

/** @brief i = argmin(a[i], a[j]) */
template <class T, class I> struct ArgMinReducer;

/** @brief i = argmax(a[i], a[j]) */
template <class T, class I> struct ArgMaxReducer;

// i = argminmax(a[i], a[j])
template <class T, class I> struct ArgMinMaxReducer;

template <> struct MinReducer<float> {
  using value_type = float;
  MFEM_HOST_DEVICE void join(value_type &a, value_type b) const {
    a = fmin(a, b);
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const { a = HUGE_VALF; }
};

template <> struct MinReducer<double> {
  using value_type = double;
  MFEM_HOST_DEVICE void join(value_type &a, value_type b) const {
    a = fmin(a, b);
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const { a = HUGE_VAL; }
};

template <> struct MinReducer<int8_t> {
  using value_type = int8_t;
  MFEM_HOST_DEVICE void join(value_type &a, value_type b) const {
    if (b < a) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const { a = value_type(0x7f); }
};

template <> struct MinReducer<uint8_t> {
  using value_type = uint8_t;
  MFEM_HOST_DEVICE void join(value_type &a, value_type b) const {
    if (b < a) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const { a = value_type(0xffu); }
};

template <> struct MinReducer<int16_t> {
  using value_type = int16_t;
  MFEM_HOST_DEVICE void join(value_type &a, value_type b) const {
    if (b < a) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type(0x7fff);
  }
};

template <> struct MinReducer<uint16_t> {
  using value_type = uint16_t;
  MFEM_HOST_DEVICE void join(value_type &a, value_type b) const {
    if (b < a) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type(0xffffu);
  }
};

template <> struct MinReducer<int32_t> {
  using value_type = int32_t;
  MFEM_HOST_DEVICE void join(int32_t &a, int32_t b) const {
    if (b < a) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(int32_t &a) const { a = int32_t(0x7fffffff); }
};

template <> struct MinReducer<uint32_t> {
  using value_type = uint32_t;
  MFEM_HOST_DEVICE void join(uint32_t &a, uint32_t b) const {
    if (b < a) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(uint32_t &a) const {
    a = uint32_t(0xffffffffu);
  }
};

template <> struct MinReducer<int64_t> {
  using value_type = int64_t;
  MFEM_HOST_DEVICE void join(int64_t &a, int64_t b) const {
    if (b < a) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(int64_t &a) const {
    a = int64_t(0x7fffffffffffffffll);
  }
};

template <> struct MinReducer<uint64_t> {
  using value_type = uint64_t;
  MFEM_HOST_DEVICE void join(uint64_t &a, uint64_t b) const {
    if (b < a) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(uint64_t &a) const {
    a = uint64_t(0xffffffffffffffffull);
  }
};

template <class I> struct ArgMinReducer<float, I> {
  using value_type = DevicePair<float, I>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.first <= a.first) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{HUGE_VALF, I(0)};
  }
};

template <class I> struct ArgMinReducer<double, I> {
  using value_type = DevicePair<double, I>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.first <= a.first) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{HUGE_VAL, I(0)};
  }
};

template <class I> struct ArgMinReducer<int8_t, I> {
  using value_type = DevicePair<int8_t, I>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.first <= a.first) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{int8_t(0x7f), I(0)};
  }
};

template <class I> struct ArgMinReducer<uint8_t, I> {
  using value_type = DevicePair<uint8_t, I>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.first <= a.first) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{uint8_t(0xffu), I(0)};
  }
};

template <class I> struct ArgMinReducer<int16_t, I> {
  using value_type = DevicePair<int16_t, I>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.first <= a.first) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{int16_t(0x7fff), I(0)};
  }
};

template <class I> struct ArgMinReducer<uint16_t, I> {
  using value_type = DevicePair<uint16_t, I>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.first <= a.first) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{uint16_t(0xffffu), I(0)};
  }
};

template <class I> struct ArgMinReducer<int32_t, I> {
  using value_type = DevicePair<int32_t, I>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.first <= a.first) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{int32_t(0x7fffffff), I(0)};
  }
};

template <class I> struct ArgMinReducer<uint32_t, I> {
  using value_type = DevicePair<uint32_t, I>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.first <= a.first) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{uint32_t(0xffffffffu), I(0)};
  }
};

template <class I> struct ArgMinReducer<int64_t, I> {
  using value_type = DevicePair<int64_t, I>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.first <= a.first) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{int64_t(0x7fffffffffffffffll), I(0)};
  }
};

template <class I> struct ArgMinReducer<uint64_t, I> {
  using value_type = DevicePair<uint64_t, I>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.first <= a.first) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{uint64_t(0xffffffffffffffffull), I(0)};
  }
};

template <> struct MaxReducer<float> {
  using value_type = float;
  MFEM_HOST_DEVICE void join(value_type &a, value_type b) const {
    a = fmax(a, b);
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const { a = -HUGE_VALF; }
};

template <> struct MaxReducer<double> {
  using value_type = double;
  MFEM_HOST_DEVICE void join(value_type &a, value_type b) const {
    a = fmax(a, b);
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const { a = -HUGE_VAL; }
};

template <> struct MaxReducer<int8_t> {
  using value_type = int8_t;
  MFEM_HOST_DEVICE void join(value_type &a, value_type b) const {
    if (a < b) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const { a = value_type(-0x80); }
};

template <> struct MaxReducer<uint8_t> {
  using value_type = uint8_t;
  MFEM_HOST_DEVICE void join(uint8_t &a, uint8_t b) const {
    if (a < b) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(uint8_t &a) const { a = uint8_t(0); }
};

template <> struct MaxReducer<int16_t> {
  using value_type = int16_t;
  MFEM_HOST_DEVICE void join(int16_t &a, int16_t b) const {
    if (a < b) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(int16_t &a) const { a = int16_t(-0x8000); }
};

template <> struct MaxReducer<uint16_t> {
  using value_type = uint16_t;
  MFEM_HOST_DEVICE void join(uint16_t &a, uint16_t b) const {
    if (a < b) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(uint16_t &a) const { a = uint16_t(0); }
};

template <> struct MaxReducer<int32_t> {
  using value_type = int32_t;
  MFEM_HOST_DEVICE void join(int32_t &a, int32_t b) const {
    if (a < b) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(int32_t &a) const { a = int32_t(-0x80000000); }
};

template <> struct MaxReducer<uint32_t> {
  using value_type = uint32_t;
  MFEM_HOST_DEVICE void join(uint32_t &a, uint32_t b) const {
    if (a < b) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(uint32_t &a) const { a = uint32_t(0); }
};

template <> struct MaxReducer<int64_t> {
  using value_type = int64_t;
  MFEM_HOST_DEVICE void join(int64_t &a, int64_t b) const {
    if (a < b) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(int64_t &a) const {
    a = int64_t(-0x8000000000000000ll);
  }
};

template <> struct MaxReducer<uint64_t> {
  using value_type = uint64_t;
  MFEM_HOST_DEVICE void join(uint64_t &a, uint64_t b) const {
    if (a < b) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(uint64_t &a) const { a = uint64_t(0); }
};

template <class I> struct ArgMaxReducer<float, I> {
  using value_type = DevicePair<float, I>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.first >= a.first) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{-HUGE_VALF, I(0)};
  }
};

template <class I> struct ArgMaxReducer<double, I> {
  using value_type = DevicePair<double, I>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.first >= a.first) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{-HUGE_VAL, I(0)};
  }
};

template <class I> struct ArgMaxReducer<int8_t, I> {
  using value_type = DevicePair<int8_t, I>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.first >= a.first) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{int8_t(-0x80), I(0)};
  }
};

template <class I> struct ArgMaxReducer<uint8_t, I> {
  using value_type = DevicePair<uint8_t, I>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.first >= a.first) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{uint8_t(0), I(0)};
  }
};

template <class I> struct ArgMaxReducer<int16_t, I> {
  using value_type = DevicePair<int16_t, I>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.first >= a.first) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{int16_t(-0x8000), I(0)};
  }
};

template <class I> struct ArgMaxReducer<uint16_t, I> {
  using value_type = DevicePair<uint16_t, I>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.first >= a.first) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{uint16_t(0), I(0)};
  }
};

template <class I> struct ArgMaxReducer<int32_t, I> {
  using value_type = DevicePair<int32_t, I>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.first >= a.first) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{int32_t(-0x80000000), I(0)};
  }
};

template <class I> struct ArgMaxReducer<uint32_t, I> {
  using value_type = DevicePair<uint32_t, I>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.first >= a.first) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{uint32_t(0), I(0)};
  }
};

template <class I> struct ArgMaxReducer<int64_t, I> {
  using value_type = DevicePair<int64_t, I>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.first >= a.first) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{int64_t(-0x8000000000000000ll), I(0)};
  }
};

template <class I> struct ArgMaxReducer<uint64_t, I> {
  using value_type = DevicePair<uint64_t, I>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.first >= a.first) {
      a = b;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{uint64_t(0), I(0)};
  }
};

template <> struct MinMaxReducer<float> {
  using value_type = DevicePair<float, float>;
  MFEM_HOST_DEVICE void join(value_type &a, value_type b) const {
    a.first = fmin(a.first, b.first);
    a.second = fmax(a.second, b.second);
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{HUGE_VALF, -HUGE_VALF};
  }
};

template <> struct MinMaxReducer<double> {
  using value_type = DevicePair<double, double>;
  MFEM_HOST_DEVICE void join(value_type &a, value_type b) const {
    a.first = fmin(a.first, b.first);
    a.second = fmax(a.second, b.second);
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{HUGE_VAL, -HUGE_VAL};
  }
};

template <> struct MinMaxReducer<int8_t> {
  using value_type = DevicePair<int8_t, int8_t>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    // assume a.first <= a.second and b.first <= b.second
    if (b.first < a.first) {
      a.first = b.first;
    }
    if (b.second > a.second) {
      a.second = b.second;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{int8_t(0x7f), int8_t(-0x80)};
  }
};

template <> struct MinMaxReducer<uint8_t> {
  using value_type = DevicePair<uint8_t, uint8_t>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.first < a.first) {
      a.first = b.first;
    }
    if (b.second > a.second) {
      a.second = b.second;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{uint8_t(0xffu), uint8_t(0)};
  }
};

template <> struct MinMaxReducer<int16_t> {
  using value_type = DevicePair<int16_t, int16_t>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.first < a.first) {
      a.first = b.first;
    }
    if (b.second > a.second) {
      a.second = b.second;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{int16_t(0x7fff), int16_t(-0x8000)};
  }
};

template <> struct MinMaxReducer<uint16_t> {
  using value_type = DevicePair<uint16_t, uint16_t>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.first < a.first) {
      a.first = b.first;
    }
    if (b.second > a.second) {
      a.second = b.second;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{uint16_t(0xffffu), uint16_t(0x0)};
  }
};

template <> struct MinMaxReducer<int32_t> {
  using value_type = DevicePair<int32_t, int32_t>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.first < a.first) {
      a.first = b.first;
    }
    if (b.second > a.second) {
      a.second = b.second;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{int32_t(0x7fffffff), int32_t(-0x80000000)};
  }
};

template <> struct MinMaxReducer<uint32_t> {
  using value_type = DevicePair<uint32_t, uint32_t>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.first < a.first) {
      a.first = b.first;
    }
    if (b.second > a.second) {
      a.second = b.second;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{uint32_t(0xffffffffu), uint32_t(0)};
  }
};

template <> struct MinMaxReducer<int64_t> {
  using value_type = DevicePair<int64_t, int64_t>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.first < a.first) {
      a.first = b.first;
    }
    if (b.second > a.second) {
      a.second = b.second;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{int64_t(0x7fffffffffffffffll),
                   int64_t(-0x8000000000000000ll)};
  }
};

template <> struct MinMaxReducer<uint64_t> {
  using value_type = DevicePair<uint64_t, uint64_t>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.first < a.first) {
      a.first = b.first;
    }
    if (b.second > a.second) {
      a.second = b.second;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{uint64_t(0xffffffffffffffffull), uint64_t(0)};
  }
};

template <class I> struct ArgMinMaxReducer<float, I> {
  using value_type = MinMaxLocScalar<float, I>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.min_val <= a.min_val) {
      a.min_val = b.min_val;
      a.min_loc = a.min_loc;
    }
    if (b.max_val >= a.max_val) {
      a.max_val = b.max_val;
      a.max_loc = b.max_loc;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{HUGE_VALF, -HUGE_VALF, I(0), I(0)};
  }
};

template <class I> struct ArgMinMaxReducer<double, I> {
  using value_type = MinMaxLocScalar<double, I>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.min_val <= a.min_val) {
      a.min_val = b.min_val;
      a.min_loc = a.min_loc;
    }
    if (b.max_val >= a.max_val) {
      a.max_val = b.max_val;
      a.max_loc = b.max_loc;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{HUGE_VAL, -HUGE_VAL, I(0), I(0)};
  }
};

template <class I> struct ArgMinMaxReducer<int8_t, I> {
  using value_type = MinMaxLocScalar<int8_t, I>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.min_val <= a.min_val) {
      a.min_val = b.min_val;
      a.min_loc = a.min_loc;
    }
    if (b.max_val >= a.max_val) {
      a.max_val = b.max_val;
      a.max_loc = b.max_loc;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{int8_t(0x7f), int8_t(-0x80), I(0), I(0)};
  }
};

template <class I> struct ArgMinMaxReducer<uint8_t, I> {
  using value_type = MinMaxLocScalar<uint8_t, I>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.min_val <= a.min_val) {
      a.min_val = b.min_val;
      a.min_loc = a.min_loc;
    }
    if (b.max_val >= a.max_val) {
      a.max_val = b.max_val;
      a.max_loc = b.max_loc;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{uint8_t(0xffu), uint8_t(0), I(0), I(0)};
  }
};

template <class I> struct ArgMinMaxReducer<int16_t, I> {
  using value_type = MinMaxLocScalar<int16_t, I>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.min_val <= a.min_val) {
      a.min_val = b.min_val;
      a.min_loc = a.min_loc;
    }
    if (b.max_val >= a.max_val) {
      a.max_val = b.max_val;
      a.max_loc = b.max_loc;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{int16_t(0x7ffff), int16_t(-0x8000), I(0), I(0)};
  }
};

template <class I> struct ArgMinMaxReducer<uint16_t, I> {
  using value_type = MinMaxLocScalar<uint16_t, I>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.min_val <= a.min_val) {
      a.min_val = b.min_val;
      a.min_loc = a.min_loc;
    }
    if (b.max_val >= a.max_val) {
      a.max_val = b.max_val;
      a.max_loc = b.max_loc;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{uint16_t(0xffffu), uint16_t(0), I(0), I(0)};
  }
};

template <class I> struct ArgMinMaxReducer<int32_t, I> {
  using value_type = MinMaxLocScalar<int32_t, I>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.min_val <= a.min_val) {
      a.min_val = b.min_val;
      a.min_loc = a.min_loc;
    }
    if (b.max_val >= a.max_val) {
      a.max_val = b.max_val;
      a.max_loc = b.max_loc;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{int32_t(0x7fffffff), int32_t(-0x80000000), I(0), I(0)};
  }
};

template <class I> struct ArgMinMaxReducer<uint32_t, I> {
  using value_type = MinMaxLocScalar<uint32_t, I>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.min_val <= a.min_val) {
      a.min_val = b.min_val;
      a.min_loc = a.min_loc;
    }
    if (b.max_val >= a.max_val) {
      a.max_val = b.max_val;
      a.max_loc = b.max_loc;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{uint32_t(0xffffffffu), uint32_t(0), I(0), I(0)};
  }
};

template <class I> struct ArgMinMaxReducer<int64_t, I> {
  using value_type = MinMaxLocScalar<int64_t, I>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.min_val <= a.min_val) {
      a.min_val = b.min_val;
      a.min_loc = a.min_loc;
    }
    if (b.max_val >= a.max_val) {
      a.max_val = b.max_val;
      a.max_loc = b.max_loc;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{int64_t(0x7fffffffffffffffll),
                   int64_t(-0x8000000000000000ll), I(0), I(0)};
  }
};

template <class I> struct ArgMinMaxReducer<uint64_t, I> {
  using value_type = MinMaxLocScalar<uint64_t, I>;
  MFEM_HOST_DEVICE void join(value_type &a, const value_type &b) const {
    if (b.min_val <= a.min_val) {
      a.min_val = b.min_val;
      a.min_loc = a.min_loc;
    }
    if (b.max_val >= a.max_val) {
      a.max_val = b.max_val;
      a.max_loc = b.max_loc;
    }
  }

  MFEM_HOST_DEVICE void init_val(value_type &a) const {
    a = value_type{uint64_t(0xffffffffffffffffull), uint64_t(0), I(0), I(0)};
  }
};

} // namespace mfem

#endif
