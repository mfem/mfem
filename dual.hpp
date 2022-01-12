// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file dual.hpp
 *
 * @brief This file contains the declaration of a dual number class
 */

#pragma once

#include <iostream>

#include <cmath>

#include "metaprogramming.hpp"

namespace serac {

/**
 * @brief Dual number struct (value plus gradient)
 * @tparam gradient_type The type of the gradient (should support addition, scalar multiplication/division, and unary
 * negation operators)
 */
template <typename gradient_type>
struct dual {
  double        value;     ///< the actual numerical value
  gradient_type gradient;  ///< the partial derivatives of value w.r.t. some other quantity
};

/**
 * @brief class template argument deduction guide for type `dual`.
 *
 * @note this lets users write
 * \code{.cpp} dual something{my_value, my_gradient}; \endcode
 * instead of explicitly writing the template parameter
 * \code{.cpp} dual< decltype(my_gradient) > something{my_value, my_gradient}; \endcode
 */
template <typename T>
dual(double, T) -> dual<T>;

/** @brief addition of a dual number and a non-dual number */
template <typename gradient_type>
constexpr auto operator+(dual<gradient_type> a, double b)
{
  return dual{a.value + b, a.gradient};
}

/** @brief addition of a dual number and a non-dual number */
template <typename gradient_type>
constexpr auto operator+(double a, dual<gradient_type> b)
{
  return dual{a + b.value, b.gradient};
}

/** @brief addition of two dual numbers */
template <typename gradient_type_a, typename gradient_type_b>
constexpr auto operator+(dual<gradient_type_a> a, dual<gradient_type_b> b)
{
  return dual{a.value + b.value, a.gradient + b.gradient};
}

/** @brief unary negation of a dual number */
template <typename gradient_type>
constexpr auto operator-(dual<gradient_type> x)
{
  return dual{-x.value, -x.gradient};
}

/** @brief subtraction of a non-dual number from a dual number */
template <typename gradient_type>
constexpr auto operator-(dual<gradient_type> a, double b)
{
  return dual{a.value - b, a.gradient};
}

/** @brief subtraction of a dual number from a non-dual number */
template <typename gradient_type>
constexpr auto operator-(double a, dual<gradient_type> b)
{
  return dual{a - b.value, -b.gradient};
}

/** @brief subtraction of two dual numbers */
template <typename gradient_type_a, typename gradient_type_b>
constexpr auto operator-(dual<gradient_type_a> a, dual<gradient_type_b> b)
{
  return dual{a.value - b.value, a.gradient - b.gradient};
}

/** @brief multiplication of a dual number and a non-dual number */
template <typename gradient_type>
constexpr auto operator*(const dual<gradient_type>& a, double b)
{
  return dual{a.value * b, a.gradient * b};
}

/** @brief multiplication of a dual number and a non-dual number */
template <typename gradient_type>
constexpr auto operator*(double a, const dual<gradient_type>& b)
{
  return dual{a * b.value, a * b.gradient};
}

/** @brief multiplication of two dual numbers */
template <typename gradient_type_a, typename gradient_type_b>
constexpr auto operator*(dual<gradient_type_a> a, dual<gradient_type_b> b)
{
  return dual{a.value * b.value, b.value * a.gradient + a.value * b.gradient};
}

/** @brief division of a dual number by a non-dual number */
template <typename gradient_type>
constexpr auto operator/(const dual<gradient_type>& a, double b)
{
  return dual{a.value / b, a.gradient / b};
}

/** @brief division of a non-dual number by a dual number */
template <typename gradient_type>
constexpr auto operator/(double a, const dual<gradient_type>& b)
{
  return dual{a / b.value, -(a / (b.value * b.value)) * b.gradient};
}

/** @brief division of two dual numbers */
template <typename gradient_type_a, typename gradient_type_b>
constexpr auto operator/(dual<gradient_type_a> a, dual<gradient_type_b> b)
{
  return dual{a.value / b.value, (a.gradient / b.value) - (a.value * b.gradient) / (b.value * b.value)};
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
constexpr auto& operator+=(dual<gradient_type>& a, const dual<gradient_type>& b)
{
  a.value += b.value;
  a.gradient += b.gradient;
  return a;
}

/** @brief compound assignment (-) for dual numbers */
template <typename gradient_type>
constexpr auto& operator-=(dual<gradient_type>& a, const dual<gradient_type>& b)
{
  a.value -= b.value;
  a.gradient -= b.gradient;
  return a;
}

/** @brief compound assignment (+) for dual numbers with `double` righthand side */
template <typename gradient_type>
constexpr auto& operator+=(dual<gradient_type>& a, double b)
{
  a.value += b;
  return a;
}

/** @brief compound assignment (-) for dual numbers with `double` righthand side */
template <typename gradient_type>
constexpr auto& operator-=(dual<gradient_type>& a, double b)
{
  a.value -= b;
  return a;
}

/** @brief implementation of absolute value function for dual numbers */
template <typename gradient_type>
auto abs(dual<gradient_type> x)
{
  return (x.value >= 0) ? x : -x;
}

/** @brief implementation of square root for dual numbers */
template <typename gradient_type>
auto sqrt(dual<gradient_type> x)
{
  using std::sqrt;
  return dual<gradient_type>{sqrt(x.value), x.gradient / (2.0 * sqrt(x.value))};
}

/** @brief implementation of cosine for dual numbers */
template <typename gradient_type>
auto cos(dual<gradient_type> a)
{
  using std::cos, std::sin;
  return dual<gradient_type>{cos(a.value), -a.gradient * sin(a.value)};
}

/** @brief implementation of exponential function for dual numbers */
template <typename gradient_type>
auto exp(dual<gradient_type> a)
{
  using std::exp;
  return dual<gradient_type>{exp(a.value), exp(a.value) * a.gradient};
}

/** @brief implementation of the natural logarithm function for dual numbers */
template <typename gradient_type>
auto log(dual<gradient_type> a)
{
  using std::log;
  return dual<gradient_type>{log(a.value), a.gradient / a.value};
}

/** @brief implementation of `a` (dual) raised to the `b` (dual) power */
template <typename gradient_type>
auto pow(dual<gradient_type> a, dual<gradient_type> b)
{
  using std::pow, std::log;
  double value = pow(a.value, b.value);
  return dual<gradient_type>{value, value * (a.gradient * (b.value / a.value) + b.gradient * log(a.value))};
}

/** @brief implementation of `a` (non-dual) raised to the `b` (dual) power */
template <typename gradient_type>
auto pow(double a, dual<gradient_type> b)
{
  using std::pow, std::log;
  double value = pow(a, b.value);
  return dual<gradient_type>{value, value * b.gradient * log(a)};
}

/** @brief implementation of `a` (dual) raised to the `b` (non-dual) power */
template <typename gradient_type>
auto pow(dual<gradient_type> a, double b)
{
  using std::pow;
  double value = pow(a.value, b);
  return dual<gradient_type>{value, value * a.gradient * b / a.value};
}

/** @brief overload of operator<< for `dual` to work with `std::cout` and other `std::ostream`s */
template <typename T, int... n>
auto& operator<<(std::ostream& out, dual<T> A)
{
  out << '(' << A.value << ' ' << A.gradient << ')';
  return out;
}

/** @brief promote a value to a dual number of the appropriate type */
constexpr auto make_dual(double x) { return dual{x, 1.0}; }

/** @brief return the "value" part from a given type. For non-dual types, this is just the identity function */
template <typename T>
SERAC_HOST_DEVICE auto get_value(const T& arg)
{
  return arg;
}

/** @brief return the "value" part from a dual number type */
template <typename T>
SERAC_HOST_DEVICE auto get_value(dual<T> arg)
{
  return arg.value;
}

/** @brief return the "gradient" part from a dual number type */
template <typename gradient_type>
SERAC_HOST_DEVICE auto get_gradient(dual<gradient_type> arg)
{
  return arg.gradient;
}

/** @brief class for checking if a type is a dual number or not */
template <typename T>
struct is_dual_number {
  static constexpr bool value = false;  ///< whether or not type T is a dual number
};

/** @brief class for checking if a type is a dual number or not */
template <typename T>
struct is_dual_number<dual<T> > {
  static constexpr bool value = true;  ///< whether or not type T is a dual number
};

}  // namespace serac