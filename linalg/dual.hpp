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

/**
 * @file dual.hpp
 *
 * @brief This file contains the declaration of a dual number class
 */

#pragma once

#include <type_traits> // for is_arithmetic
#include <cmath>
#include "../general/backends.hpp"

namespace mfem
{
namespace future
{

/**
 * @brief Dual number struct (value plus gradient)
 * @tparam gradient_type The type of the gradient (should support addition,
 * scalar multiplication/division, and unary negation operators)
 */
template <typename value_type, typename gradient_type>
struct dual
{
   /// the actual numerical value
   value_type value;
   /// the partial derivatives of value w.r.t. some other quantity
   gradient_type gradient;

   /** @brief assignment of a real_t to a value of a dual. Promotes a real_t to
    * a dual with a zero gradient value. */
   MFEM_HOST_DEVICE
   auto operator=(real_t a) -> dual<value_type, gradient_type>&
   {
      value = a;
      gradient = {};
      return *this;
   }
};

/** @brief class for checking if a type is a dual number or not */
template <typename T>
struct is_dual_number
{
   /// whether or not type T is a dual number
   static constexpr bool value = false;
};

/** @brief class for checking if a type is a dual number or not */
template <typename value_type, typename gradient_type>
struct is_dual_number<dual<value_type, gradient_type> >
{
   static constexpr bool value = true;  ///< whether or not type T is a dual number
};

/** @brief addition of a dual number and a non-dual number */
template <typename other_type, typename value_type, typename gradient_type,
          typename = typename std::enable_if<
             std::is_arithmetic<other_type>::value ||
             is_dual_number<other_type>::value>::type>
MFEM_HOST_DEVICE
constexpr auto operator+(dual<value_type, gradient_type> a,
                         other_type b) -> dual<value_type, gradient_type>
{
   return {a.value + b, a.gradient};
}

// C++17 version of the above
//
// template <typename value_type, typename gradient_type>
// constexpr auto operator+(dual<value_type, gradient_type> a, value_type b)
// {
//    return dual{a.value + b, a.gradient};
// }

/** @brief addition of a dual number and a non-dual number */
template <typename other_type, typename value_type, typename gradient_type,
          typename = typename std::enable_if<
             std::is_arithmetic<other_type>::value ||
             is_dual_number<other_type>::value>::type>
MFEM_HOST_DEVICE
constexpr auto operator+(other_type a,
                         dual<value_type, gradient_type> b) -> dual<value_type, gradient_type>
{
   return {a + b.value, b.gradient};
}

/** @brief addition of two dual numbers */
template <typename value_type_a, typename gradient_type_a, typename value_type_b, typename gradient_type_b>
MFEM_HOST_DEVICE
constexpr auto operator+(dual<value_type_a, gradient_type_a> a,
                         dual<value_type_b, gradient_type_b> b) -> dual<decltype(a.value + b.value),
                              decltype(a.gradient + b.gradient)>
{
   return {a.value + b.value, a.gradient + b.gradient};
}

/** @brief unary negation of a dual number */
template <typename value_type, typename gradient_type> MFEM_HOST_DEVICE
constexpr auto operator-(dual<value_type, gradient_type> x) ->
dual<value_type, gradient_type>
{
   return {-x.value, -x.gradient};
}

/** @brief subtraction of a non-dual number from a dual number */
template <typename value_type, typename gradient_type> MFEM_HOST_DEVICE
constexpr auto operator-(dual<value_type, gradient_type> a,
                         real_t b) -> dual<value_type, gradient_type>
{
   return {a.value - b, a.gradient};
}

/** @brief subtraction of a dual number from a non-dual number */
template <typename value_type, typename gradient_type> MFEM_HOST_DEVICE
constexpr auto operator-(real_t a,
                         dual<value_type, gradient_type> b) -> dual<value_type, gradient_type>
{
   return {a - b.value, -b.gradient};
}

/** @brief subtraction of two dual numbers */
template <typename value_type_a, typename gradient_type_a, typename value_type_b, typename gradient_type_b>
MFEM_HOST_DEVICE
constexpr auto operator-(dual<value_type_a, gradient_type_a> a,
                         dual<value_type_b, gradient_type_b> b) -> dual<decltype(a.value - b.value),
                              decltype(a.gradient - b.gradient)>
{
   return {a.value - b.value, a.gradient - b.gradient};
}

/** @brief multiplication of a dual number and a non-dual number */
template <typename value_type, typename gradient_type> MFEM_HOST_DEVICE
constexpr auto operator*(const dual<value_type, gradient_type>& a,
                         real_t b) -> dual<decltype(a.value * b), decltype(a.gradient * b)>
{
   return {a.value * b, a.gradient * b};
}

/** @brief multiplication of a dual number and a non-dual number */
template <typename value_type, typename gradient_type> MFEM_HOST_DEVICE
constexpr auto operator*(real_t a,
                         const dual<value_type, gradient_type>& b) ->
dual<decltype(a * b.value), decltype(a * b.gradient)>
{
   return {a * b.value, a * b.gradient};
}

/** @brief multiplication of two dual numbers */
template <typename value_type_a, typename gradient_type_a, typename value_type_b, typename gradient_type_b>
MFEM_HOST_DEVICE
constexpr auto operator*(dual<value_type_a, gradient_type_a> a,
                         dual<value_type_b, gradient_type_b> b) -> dual<decltype(a.value * b.value),
                              decltype(b.value * a.gradient + a.value * b.gradient)>
{
   return {a.value * b.value, b.value * a.gradient + a.value * b.gradient};
}

/** @brief division of a dual number by a non-dual number */
template <typename value_type, typename gradient_type> MFEM_HOST_DEVICE
constexpr auto operator/(const dual<value_type, gradient_type>& a,
                         real_t b) -> dual<decltype(a.value / b), decltype(a.gradient / b)>
{
   return {a.value / b, a.gradient / b};
}

/** @brief division of a non-dual number by a dual number */
template <typename value_type, typename gradient_type> MFEM_HOST_DEVICE
constexpr auto operator/(real_t a,
                         const dual<value_type, gradient_type>& b) -> dual<decltype(a / b.value),
                               decltype(-(a / (b.value * b.value)) * b.gradient)>
{
   return {a / b.value, -(a / (b.value * b.value)) * b.gradient};
}

/** @brief division of two dual numbers */
template <typename value_type_a, typename gradient_type_a, typename value_type_b, typename gradient_type_b>
MFEM_HOST_DEVICE
constexpr auto operator/(dual<value_type_a, gradient_type_a> a,
                         dual<value_type_b, gradient_type_b> b) -> dual<decltype(a.value / b.value),
                              decltype((a.gradient / b.value) -
                                       (a.value * b.gradient) /
                                       (b.value * b.value))>
{
   return {a.value / b.value, (a.gradient / b.value) - (a.value * b.gradient) / (b.value * b.value)};
}

/**
 * @brief Generates const + non-const overloads for a binary comparison operator
 * Comparisons are conducted against the "value" part of the dual number
 * @param[in] x The comparison operator to overload
 */
#define mfem_binary_comparator_overload(x)                      \
  template <typename value_type, typename gradient_type>        \
  MFEM_HOST_DEVICE constexpr bool operator x(                   \
     const dual<value_type, gradient_type>& a,                  \
     real_t b)                                                  \
  {                                                             \
    return a.value x b;                                         \
  }                                                             \
                                                                \
  template <typename value_type, typename gradient_type>        \
  MFEM_HOST_DEVICE constexpr bool operator x(                   \
     real_t a,                                                  \
     const dual<value_type, gradient_type>& b)                  \
  {                                                             \
    return a x b.value;                                         \
  }                                                             \
                                                                \
  template <typename value_type_a,                              \
            typename gradient_type_a,                           \
            typename value_type_b,                              \
            typename gradient_type_b> MFEM_HOST_DEVICE          \
  constexpr bool operator x(                                    \
     const dual<value_type_a, gradient_type_a>& a,              \
     const dual<value_type_b, gradient_type_b>& b)              \
  {                                                             \
    return a.value x b.value;                                   \
  }

mfem_binary_comparator_overload(<)   ///< implement operator<  for dual numbers
mfem_binary_comparator_overload(<=)  ///< implement operator<= for dual numbers
mfem_binary_comparator_overload(==)  ///< implement operator== for dual numbers
mfem_binary_comparator_overload(>=)  ///< implement operator>= for dual numbers
mfem_binary_comparator_overload(>)   ///< implement operator>  for dual numbers

#undef mfem_binary_comparator_overload

/** @brief compound assignment (+) for dual numbers */
template <typename value_type, typename gradient_type> MFEM_HOST_DEVICE
dual<value_type, gradient_type>& operator+=(dual<value_type, gradient_type>& a,
                                            const dual<value_type, gradient_type>& b)
{
   a.value += b.value;
   a.gradient += b.gradient;
   return a;
}

/** @brief compound assignment (-) for dual numbers */
template <typename value_type, typename gradient_type> MFEM_HOST_DEVICE
dual<value_type, gradient_type>& operator-=(dual<value_type, gradient_type>& a,
                                            const dual<value_type, gradient_type>& b)
{
   a.value -= b.value;
   a.gradient -= b.gradient;
   return a;
}

/** @brief compound assignment (+) for dual numbers with `real_t` righthand side */
template <typename value_type, typename gradient_type> MFEM_HOST_DEVICE
dual<value_type, gradient_type>& operator+=(dual<value_type, gradient_type>& a,
                                            real_t b)
{
   a.value += b;
   return a;
}

/** @brief compound assignment (-) for dual numbers with `real_t` righthand side */
template <typename value_type, typename gradient_type> MFEM_HOST_DEVICE
dual<value_type, gradient_type>& operator-=(dual<value_type, gradient_type>& a,
                                            real_t b)
{
   a.value -= b;
   return a;
}

/** @brief implementation of absolute value function for dual numbers */
template <typename value_type, typename gradient_type> MFEM_HOST_DEVICE
dual<value_type, gradient_type> abs(dual<value_type, gradient_type> x)
{
   return (x.value >= 0) ? x : -x;
}

/** @brief implementation of square root for dual numbers */
template <typename value_type, typename gradient_type> MFEM_HOST_DEVICE
dual<value_type, gradient_type> sqrt(dual<value_type, gradient_type> x)
{
   using std::sqrt;
   return {sqrt(x.value), x.gradient / (2 * sqrt(x.value))};
}

/** @brief implementation of cosine for dual numbers */
template <typename value_type, typename gradient_type> MFEM_HOST_DEVICE
dual<value_type, gradient_type> cos(dual<value_type, gradient_type> a)
{
   using std::cos;
   using std::sin;
   return {cos(a.value), -a.gradient * sin(a.value)};
}

/** @brief implementation of sine for dual numbers */
template <typename value_type, typename gradient_type> MFEM_HOST_DEVICE
dual<value_type, gradient_type> sin(dual<value_type, gradient_type> a)
{
   using std::sin;
   using std::cos;
   return {sin(a.value), a.gradient * cos(a.value)};
}

/** @brief implementation of sinh for dual numbers */
template <typename value_type, typename gradient_type> MFEM_HOST_DEVICE
dual<value_type, gradient_type> sinh(dual<value_type, gradient_type> a)
{
   using std::sinh;
   using std::cosh;
   return {sinh(a.value), a.gradient * cosh(a.value)};
}

/** @brief implementation of acos for dual numbers */
template <typename value_type, typename gradient_type> MFEM_HOST_DEVICE
dual<value_type, gradient_type> acos(dual<value_type, gradient_type> a)
{
   using std::sqrt;
   using std::acos;
   return {acos(a.value), -a.gradient / sqrt(value_type{1} - a.value * a.value)};
}

/** @brief implementation of asin for dual numbers */
template <typename value_type, typename gradient_type> MFEM_HOST_DEVICE
dual<value_type, gradient_type> asin(dual<value_type, gradient_type> a)
{
   using std::sqrt;
   using std::asin;
   return {asin(a.value), a.gradient / sqrt(value_type{1} - a.value * a.value)};
}

/** @brief implementation of tan for dual numbers */
template <typename value_type, typename gradient_type> MFEM_HOST_DEVICE
dual<value_type, gradient_type> tan(dual<value_type, gradient_type> a)
{
   using std::tan;
   value_type f = tan(a.value);
   return {f, a.gradient * (value_type{1} + f * f)};
}

/** @brief implementation of atan for dual numbers */
template <typename value_type, typename gradient_type> MFEM_HOST_DEVICE
dual<value_type, gradient_type> atan(dual<value_type, gradient_type> a)
{
   using std::atan;
   return {atan(a.value), a.gradient / (value_type{1} + a.value * a.value)};
}

/** @brief implementation of exponential function for dual numbers */
template <typename value_type, typename gradient_type> MFEM_HOST_DEVICE
dual<value_type, gradient_type> exp(dual<value_type, gradient_type> a)
{
   using std::exp;
   return {exp(a.value), exp(a.value) * a.gradient};
}

/** @brief implementation of the natural logarithm function for dual numbers */
template <typename value_type, typename gradient_type> MFEM_HOST_DEVICE
dual<value_type, gradient_type> log(dual<value_type, gradient_type> a)
{
   using std::log;
   return {log(a.value), a.gradient / a.value};
}

/** @brief implementation of `a` (dual) raised to the `b` (dual) power */
template <typename value_type, typename gradient_type> MFEM_HOST_DEVICE
dual<value_type, gradient_type> pow(dual<value_type, gradient_type> a,
                                    dual<value_type, gradient_type> b)
{
   using std::log;
   using std::pow;
   value_type value = pow(a.value, b.value);
   return {value, value * (a.gradient * (b.value / a.value) + b.gradient * log(a.value))};
}

/** @brief implementation of `a` (non-dual) raised to the `b` (dual) power */
template <typename value_type, typename gradient_type> MFEM_HOST_DEVICE
dual<value_type, gradient_type> pow(real_t a, dual<value_type, gradient_type> b)
{
   using std::pow;
   using std::log;
   value_type value = pow(a, b.value);
   return {value, value * b.gradient * log(a)};
}

/** @brief implementation of `a` (non-dual) raised to the `b` (non-dual) power */
template <typename value_type > MFEM_HOST_DEVICE
value_type pow(value_type a, value_type b)
{
   using std::pow;
   return pow(a, b);
}

/** @brief implementation of `a` (dual) raised to the `b` (non-dual) power */
template <typename value_type, typename gradient_type> MFEM_HOST_DEVICE
dual<value_type, gradient_type> pow(dual<value_type, gradient_type> a, real_t b)
{
   using std::pow;
   value_type value = pow(a.value, b);
   return {value, value * a.gradient * b / a.value};
}

/** @brief overload of operator<< for `dual` to work with work with standard output streams */
template <typename value_type, typename gradient_type, int... n>
std::ostream& operator<<(std::ostream& os, dual<value_type, gradient_type> A)
{
   os << '(' << A.value << ' ' << A.gradient << ')';
   return os;
}

/** @brief promote a value to a dual number of the appropriate type */
MFEM_HOST_DEVICE constexpr dual<real_t, real_t> make_dual(real_t x) { return {x, 1.0}; }

/** @brief return the "value" part from a given type. For non-dual types, this is just the identity function */
template <typename T> MFEM_HOST_DEVICE T get_value(const T& arg) { return arg; }

/** @brief return the "value" part from a dual number type */
template <typename value_type, typename gradient_type>
MFEM_HOST_DEVICE gradient_type get_value(dual<value_type, gradient_type> arg)
{
   return arg.value;
}

/** @brief return the "gradient" part from a dual number type */
template <typename value_type, typename gradient_type>
MFEM_HOST_DEVICE gradient_type get_gradient(dual<value_type, gradient_type> arg)
{
   return arg.gradient;
}

} // namespace future
} // namespace mfem
