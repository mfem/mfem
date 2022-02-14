// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
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

#include <cmath>
#include "../general/backends.hpp"

namespace mfem
{
namespace internal
{

/**
 * @brief Dual number struct (value plus gradient)
 * @tparam gradient_type The type of the gradient (should support addition,
 * scalar multiplication/division, and unary negation operators)
 */
template <typename gradient_type>
struct dual
{
   /// the actual numerical value
   double value;
   /// the partial derivatives of value w.r.t. some other quantity
   gradient_type gradient;
};

/** @brief addition of a dual number and a non-dual number */
template <typename gradient_type>
constexpr auto operator+(dual<gradient_type> a, double b) -> dual<gradient_type>
{
   return {a.value + b, a.gradient};
}

// C++17 version of the above
//
// template <typename gradient_type>
// constexpr auto operator+(dual<gradient_type> a, double b)
// {
//    return dual{a.value + b, a.gradient};
// }

/** @brief addition of a dual number and a non-dual number */
template <typename gradient_type>
constexpr auto operator+(double a, dual<gradient_type> b) -> dual<gradient_type>
{
   return {a + b.value, b.gradient};
}

/** @brief addition of two dual numbers */
template <typename gradient_type_a, typename gradient_type_b>
constexpr auto operator+(dual<gradient_type_a> a,
                         dual<gradient_type_b> b) -> dual<decltype(a.gradient + b.gradient)>
{
   return {a.value + b.value, a.gradient + b.gradient};
}

/** @brief unary negation of a dual number */
template <typename gradient_type>
constexpr auto operator-(dual<gradient_type> x) -> dual<gradient_type>
{
   return {-x.value, -x.gradient};
}

/** @brief subtraction of a non-dual number from a dual number */
template <typename gradient_type>
constexpr auto operator-(dual<gradient_type> a, double b) -> dual<gradient_type>
{
   return {a.value - b, a.gradient};
}

/** @brief subtraction of a dual number from a non-dual number */
template <typename gradient_type>
constexpr auto operator-(double a, dual<gradient_type> b) -> dual<gradient_type>
{
   return {a - b.value, -b.gradient};
}

/** @brief subtraction of two dual numbers */
template <typename gradient_type_a, typename gradient_type_b>
constexpr auto operator-(dual<gradient_type_a> a,
                         dual<gradient_type_b> b) -> dual<decltype(a.gradient - b.gradient)>
{
   return {a.value - b.value, a.gradient - b.gradient};
}

/** @brief multiplication of a dual number and a non-dual number */
template <typename gradient_type>
constexpr auto operator*(const dual<gradient_type>& a,
                         double b) -> dual<decltype(a.gradient * b)>
{
   return {a.value * b, a.gradient * b};
}

/** @brief multiplication of a dual number and a non-dual number */
template <typename gradient_type>
constexpr auto operator*(double a,
                         const dual<gradient_type>& b) -> dual<decltype(a * b.gradient)>
{
   return {a * b.value, a * b.gradient};
}

/** @brief multiplication of two dual numbers */
template <typename gradient_type_a, typename gradient_type_b>
constexpr auto operator*(dual<gradient_type_a> a,
                         dual<gradient_type_b> b) -> dual<decltype(b.value * a.gradient + a.value *
                                                                   b.gradient)>
{
   return {a.value * b.value, b.value * a.gradient + a.value * b.gradient};
}

/** @brief division of a dual number by a non-dual number */
template <typename gradient_type>
constexpr auto operator/(const dual<gradient_type>& a,
                         double b) -> dual<decltype(a.gradient / b)>
{
   return {a.value / b, a.gradient / b};
}

/** @brief division of a non-dual number by a dual number */
template <typename gradient_type>
constexpr auto operator/(double a,
                         const dual<gradient_type>& b) -> dual<decltype(-(a / (b.value * b.value)) *
                                                                        b.gradient)>
{
   return {a / b.value, -(a / (b.value * b.value)) * b.gradient};
}

/** @brief division of two dual numbers */
template <typename gradient_type_a, typename gradient_type_b>
constexpr auto operator/(dual<gradient_type_a> a,
                         dual<gradient_type_b> b) -> dual<decltype((a.gradient / b.value) -
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
#define binary_comparator_overload(x)                           \
  template <typename T>                                         \
  constexpr bool operator x(const dual<T>& a, double b)         \
  {                                                             \
    return a.value x b;                                         \
  }                                                             \
                                                                \
  template <typename T>                                         \
  constexpr bool operator x(double a, const dual<T>& b)         \
  {                                                             \
    return a x b.value;                                         \
  };                                                            \
                                                                \
  template <typename T, typename U>                             \
  constexpr bool operator x(const dual<T>& a, const dual<U>& b) \
  {                                                             \
    return a.value x b.value;                                   \
  };

binary_comparator_overload(<);   ///< implement operator<  for dual numbers
binary_comparator_overload(<=);  ///< implement operator<= for dual numbers
binary_comparator_overload(==);  ///< implement operator== for dual numbers
binary_comparator_overload(>=);  ///< implement operator>= for dual numbers
binary_comparator_overload(>);   ///< implement operator>  for dual numbers

#undef binary_comparator_overload

/** @brief compound assignment (+) for dual numbers */
template <typename gradient_type>
dual<gradient_type>& operator+=(dual<gradient_type>& a,
                                const dual<gradient_type>& b)
{
   a.value += b.value;
   a.gradient += b.gradient;
   return a;
}

/** @brief compound assignment (-) for dual numbers */
template <typename gradient_type>
dual<gradient_type>& operator-=(dual<gradient_type>& a,
                                const dual<gradient_type>& b)
{
   a.value -= b.value;
   a.gradient -= b.gradient;
   return a;
}

/** @brief compound assignment (+) for dual numbers with `double` righthand side */
template <typename gradient_type>
dual<gradient_type>& operator+=(dual<gradient_type>& a, double b)
{
   a.value += b;
   return a;
}

/** @brief compound assignment (-) for dual numbers with `double` righthand side */
template <typename gradient_type>
dual<gradient_type>& operator-=(dual<gradient_type>& a, double b)
{
   a.value -= b;
   return a;
}

/** @brief implementation of absolute value function for dual numbers */
template <typename gradient_type>
dual<gradient_type> abs(dual<gradient_type> x)
{
   return (x.value >= 0) ? x : -x;
}

/** @brief implementation of square root for dual numbers */
template <typename gradient_type>
dual<gradient_type> sqrt(dual<gradient_type> x)
{
   return {std::sqrt(x.value), x.gradient / (2.0 * std::sqrt(x.value))};
}

/** @brief implementation of cosine for dual numbers */
template <typename gradient_type>
dual<gradient_type> cos(dual<gradient_type> a)
{
   return {std::cos(a.value), -a.gradient * std::sin(a.value)};
}

/** @brief implementation of sine for dual numbers */
template <typename gradient_type>
dual<gradient_type> sin(dual<gradient_type> a)
{
   return {std::sin(a.value), a.gradient * std::cos(a.value)};
}

/** @brief implementation of exponential function for dual numbers */
template <typename gradient_type>
dual<gradient_type> exp(dual<gradient_type> a)
{
   return {std::exp(a.value), std::exp(a.value) * a.gradient};
}

/** @brief implementation of the natural logarithm function for dual numbers */
template <typename gradient_type>
dual<gradient_type> log(dual<gradient_type> a)
{
   return {std::log(a.value), a.gradient / a.value};
}

/** @brief implementation of `a` (dual) raised to the `b` (dual) power */
template <typename gradient_type>
dual<gradient_type> pow(dual<gradient_type> a, dual<gradient_type> b)
{
   double value = std::pow(a.value, b.value);
   return {value, value * (a.gradient * (b.value / a.value) + b.gradient * std::log(a.value))};
}

/** @brief implementation of `a` (non-dual) raised to the `b` (dual) power */
template <typename gradient_type>
dual<gradient_type> pow(double a, dual<gradient_type> b)
{
   double value = std::pow(a, b.value);
   return {value, value * b.gradient * std::log(a)};
}

/** @brief implementation of `a` (dual) raised to the `b` (non-dual) power */
template <typename gradient_type>
dual<gradient_type> pow(dual<gradient_type> a, double b)
{
   double value = std::pow(a.value, b);
   return {value, value * a.gradient * b / a.value};
}

/** @brief overload of operator<< for `dual` to work with work with standard output streams */
template <typename T, int... n>
std::ostream& operator<<(std::ostream& out, dual<T> A)
{
   out << '(' << A.value << ' ' << A.gradient << ')';
   return out;
}

/** @brief promote a value to a dual number of the appropriate type */
constexpr dual<double> make_dual(double x) { return {x, 1.0}; }

/** @brief return the "value" part from a given type. For non-dual types, this is just the identity function */
template <typename T>
MFEM_HOST_DEVICE T get_value(const T& arg)
{
   return arg;
}

/** @brief return the "value" part from a dual number type */
template <typename T>
MFEM_HOST_DEVICE T get_value(dual<T> arg)
{
   return arg.value;
}

/** @brief return the "gradient" part from a dual number type */
template <typename gradient_type>
MFEM_HOST_DEVICE gradient_type get_gradient(dual<gradient_type> arg)
{
   return arg.gradient;
}

/** @brief class for checking if a type is a dual number or not */
template <typename T>
struct is_dual_number
{
   /// whether or not type T is a dual number
   static constexpr bool value = false;
};

/** @brief class for checking if a type is a dual number or not */
template <typename T>
struct is_dual_number<dual<T> >
{
   static constexpr bool value = true;  ///< whether or not type T is a dual number
};

} // namespace internal
} // namespace mfem