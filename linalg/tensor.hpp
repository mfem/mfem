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
 * @file tensor.hpp
 *
 * @brief Implementation of the tensor class
 */

#pragma once

#include "../general/backends.hpp"
#include "dual.hpp"
#include <limits>
#include <type_traits> // for std::false_type

namespace mfem
{
namespace future
{

template <typename T, int... n>
struct tensor;

/// The implementation can be drastically generalized by using concepts of the
/// c++17 standard.

template < typename T >
struct tensor<T>
{
   using type = T;
   static constexpr int ndim      = 1;
   static constexpr int first_dim = 0;
   MFEM_HOST_DEVICE T& operator[](int /*unused*/) { return values; }
   MFEM_HOST_DEVICE const T& operator[](int /*unused*/) const { return values; }
   MFEM_HOST_DEVICE T& operator()(int /*unused*/) { return values; }
   MFEM_HOST_DEVICE const T& operator()(int /*unused*/) const { return values; }
   MFEM_HOST_DEVICE operator T() const { return values; }
   T values;
};

template < typename T, int n0 >
struct tensor<T, n0>
{
   using type = T;
   static constexpr int ndim      = 1;
   static constexpr int first_dim = n0;
   MFEM_HOST_DEVICE T& operator[](int i) { return values[i]; }
   MFEM_HOST_DEVICE const T& operator[](int i) const { return values[i]; }
   MFEM_HOST_DEVICE T& operator()(int i) { return values[i]; }
   MFEM_HOST_DEVICE const T& operator()(int i) const { return values[i]; }
   T values[n0];
};

template < typename T >
struct tensor<T, 0>
{
   using type = T;
   static constexpr int ndim      = 1;
   static constexpr int first_dim = 0;
   MFEM_HOST_DEVICE T& operator[](int /*unused*/) { return values; }
   MFEM_HOST_DEVICE const T& operator[](int /*unused*/) const { return values; }
   MFEM_HOST_DEVICE T& operator()(int /*unused*/) { return values; }
   MFEM_HOST_DEVICE const T& operator()(int /*unused*/) const { return values; }
   T values;
};

template < typename T, int n0, int n1 >
struct tensor<T, n0, n1>
{
   using type = T;
   static constexpr int ndim      = 2;
   static constexpr int first_dim = n0;
   MFEM_HOST_DEVICE tensor< T, n1 >& operator[](int i) { return values[i]; }
   MFEM_HOST_DEVICE const tensor< T, n1 >& operator[](int i) const { return values[i]; }
   MFEM_HOST_DEVICE tensor< T, n1 >& operator()(int i) { return values[i]; }
   MFEM_HOST_DEVICE const tensor< T, n1 >& operator()(int i) const { return values[i]; }
   MFEM_HOST_DEVICE T& operator()(int i, int j) { return values[i][j]; }
   MFEM_HOST_DEVICE const T& operator()(int i, int j) const { return values[i][j]; }
   tensor < T, n1 > values[n0];
};

template < typename T, int n1 >
struct tensor<T, 0, n1>
{
   using type = T;
   static constexpr int ndim      = 2;
   static constexpr int first_dim = 0;
   MFEM_HOST_DEVICE tensor< T, n1 >& operator[](int /*unused*/) { return values; }
   MFEM_HOST_DEVICE const tensor< T, n1 >& operator[](int /*unused*/) const { return values; }
   MFEM_HOST_DEVICE tensor< T, n1 >& operator()(int /*unused*/) { return values; }
   MFEM_HOST_DEVICE const tensor< T, n1 >& operator()(int /*unused*/) const { return values; }
   MFEM_HOST_DEVICE T& operator()(int /*unused*/, int j) { return values[j]; }
   MFEM_HOST_DEVICE const T& operator()(int /*unused*/, int j) const { return values[j]; }
   tensor < T, n1 > values;
};

template < typename T, int n0, int n1, int n2 >
struct tensor<T, n0, n1, n2>
{
   using type = T;
   static constexpr int ndim      = 3;
   static constexpr int first_dim = n0;
   MFEM_HOST_DEVICE tensor< T, n1, n2 >& operator[](int i) { return values[i]; }
   MFEM_HOST_DEVICE const tensor< T, n1, n2 >& operator[](int i) const { return values[i]; }
   MFEM_HOST_DEVICE tensor< T, n1, n2 >& operator()(int i) { return values[i]; }
   MFEM_HOST_DEVICE const tensor< T, n1, n2 >& operator()(int i) const { return values[i]; }
   MFEM_HOST_DEVICE tensor< T, n2 >& operator()(int i, int j) { return values[i][j]; }
   MFEM_HOST_DEVICE const tensor< T, n2 >& operator()(int i, int j) const { return values[i][j]; }
   MFEM_HOST_DEVICE T& operator()(int i, int j, int k) { return values[i][j][k]; }
   MFEM_HOST_DEVICE const T& operator()(int i, int j, int k) const { return values[i][j][k]; }
   tensor < T, n1, n2 > values[n0];
};

template < typename T, int n0, int n1, int n2, int n3 >
struct tensor<T, n0, n1, n2, n3>
{
   using type = T;
   static constexpr int ndim      = 4;
   static constexpr int first_dim = n0;
   MFEM_HOST_DEVICE tensor< T, n1, n2, n3 >& operator[](int i) { return values[i]; }
   MFEM_HOST_DEVICE const tensor< T, n1, n2, n3 >& operator[](int i) const { return values[i]; }
   MFEM_HOST_DEVICE tensor< T, n1, n2, n3 >& operator()(int i) { return values[i]; }
   MFEM_HOST_DEVICE const tensor< T, n1, n2, n3 >& operator()(int i) const { return values[i]; }
   MFEM_HOST_DEVICE tensor< T, n2, n3 >& operator()(int i, int j) { return values[i][j]; }
   MFEM_HOST_DEVICE const tensor< T, n2, n3 >& operator()(int i, int j) const { return values[i][j]; }
   MFEM_HOST_DEVICE tensor< T, n3 >& operator()(int i, int j, int k) { return values[i][j][k]; }
   MFEM_HOST_DEVICE const tensor< T, n3 >& operator()(int i, int j, int k) const { return values[i][j][k]; }
   MFEM_HOST_DEVICE T& operator()(int i, int j, int k, int l) { return values[i][j][k][l]; }
   MFEM_HOST_DEVICE const T&  operator()(int i, int j, int k, int l) const { return values[i][j][k][l]; }
   tensor < T, n1, n2, n3 > values[n0];
};

template < typename T, int n0, int n1, int n2, int n3, int n4 >
struct tensor<T, n0, n1, n2, n3, n4>
{
   using type = T;
   static constexpr int ndim      = 5;
   static constexpr int first_dim = n0;
   MFEM_HOST_DEVICE tensor< T, n1, n2, n3, n4 >& operator[](int i) { return values[i]; }
   MFEM_HOST_DEVICE const tensor< T, n1, n2, n3, n4 >& operator[](int i) const { return values[i]; }
   MFEM_HOST_DEVICE tensor< T, n1, n2, n3, n4 >& operator()(int i) { return values[i]; }
   MFEM_HOST_DEVICE const tensor< T, n1, n2, n3, n4 >& operator()(int i) const { return values[i]; }
   MFEM_HOST_DEVICE tensor< T, n2, n3, n4 >& operator()(int i, int j) { return values[i][j]; }
   MFEM_HOST_DEVICE const tensor< T, n2, n3, n4 >& operator()(int i,
                                                              int j) const { return values[i][j]; }
   MFEM_HOST_DEVICE tensor< T, n3, n4>& operator()(int i, int j, int k) { return values[i][j][k]; }
   MFEM_HOST_DEVICE const tensor< T, n3, n4>& operator()(int i, int j,
                                                         int k) const { return values[i][j][k]; }
   MFEM_HOST_DEVICE tensor< T, n4 >& operator()(int i, int j, int k, int l) { return values[i][j][k][l]; }
   MFEM_HOST_DEVICE const tensor< T, n4 >& operator()(int i, int j, int k,
                                                      int l) const { return values[i][j][k][l]; }
   MFEM_HOST_DEVICE T& operator()(int i, int j, int k, int l, int m) { return values[i][j][k][l][m]; }
   MFEM_HOST_DEVICE const T& operator()(int i, int j, int k, int l, int m) const { return values[i][j][k][l][m]; }
   tensor < T, n1, n2, n3, n4 > values[n0];
};

/**
 * @brief A sentinel struct for eliding no-op tensor operations
 */
struct zero
{
   /** @brief `zero` is implicitly convertible to real_t with value 0.0 */
   MFEM_HOST_DEVICE operator real_t() { return 0.0; }

   /** @brief `zero` is implicitly convertible to a tensor of any shape */
   template <typename T, int... n>
   MFEM_HOST_DEVICE operator tensor<T, n...>()
   {
      return tensor<T, n...> {};
   }

   /** @brief `zero` can be accessed like a multidimensional array */
   template <typename... T>
   MFEM_HOST_DEVICE zero operator()(T...)
   {
      return zero{};
   }

   /** @brief anything assigned to `zero` does not change its value and returns `zero` */
   template <typename T>
   MFEM_HOST_DEVICE zero operator=(T)
   {
      return zero{};
   }
};

/** @brief checks if a type is `zero` */
template <typename T>
struct is_zero : std::false_type
{
};

/** @overload */
template <>
struct is_zero<zero> : std::true_type
{
};

/** @brief the sum of two `zero`s is `zero` */
MFEM_HOST_DEVICE constexpr zero operator+(zero, zero) { return zero{}; }

/** @brief the sum of `zero` with something non-`zero` just returns the other value */
template <typename T>
MFEM_HOST_DEVICE constexpr T operator+(zero, T other)
{
   return other;
}

/** @brief the sum of `zero` with something non-`zero` just returns the other value */
template <typename T>
MFEM_HOST_DEVICE constexpr T operator+(T other, zero)
{
   return other;
}

/////////////////////////////////////////////////

/** @brief the unary negation of `zero` is `zero` */
MFEM_HOST_DEVICE constexpr zero operator-(zero) { return zero{}; }

/** @brief the difference of two `zero`s is `zero` */
MFEM_HOST_DEVICE constexpr zero operator-(zero, zero) { return zero{}; }

/** @brief the difference of `zero` with something else is the unary negation of the other thing */
template <typename T>
MFEM_HOST_DEVICE constexpr T operator-(zero, T other)
{
   return -other;
}

/** @brief the difference of something else with `zero` is the other thing itself */
template <typename T>
MFEM_HOST_DEVICE constexpr T operator-(T other, zero)
{
   return other;
}

/////////////////////////////////////////////////

/** @brief the product of two `zero`s is `zero` */
MFEM_HOST_DEVICE constexpr zero operator*(zero, zero) { return zero{}; }

/** @brief the product `zero` with something else is also `zero` */
template <typename T>
MFEM_HOST_DEVICE constexpr zero operator*(zero, T /*other*/)
{
   return zero{};
}

/** @brief the product `zero` with something else is also `zero` */
template <typename T>
MFEM_HOST_DEVICE constexpr zero operator*(T /*other*/, zero)
{
   return zero{};
}

/** @brief `zero` divided by something is `zero` */
template <typename T>
MFEM_HOST_DEVICE constexpr zero operator/(zero, T /*other*/)
{
   return zero{};
}

/** @brief `zero` plus `zero` is `zero */
MFEM_HOST_DEVICE constexpr zero operator+=(zero, zero) { return zero{}; }

/** @brief `zero` minus `zero` is `zero */
MFEM_HOST_DEVICE constexpr zero operator-=(zero, zero) { return zero{}; }

/** @brief let `zero` be accessed like a tuple */
template <int i>
MFEM_HOST_DEVICE zero& get(zero& x)
{
   return x;
}

/** @brief the dot product of anything with `zero` is `zero` */
template <typename T>
MFEM_HOST_DEVICE zero dot(const T&, zero)
{
   return zero{};
}

/** @brief the dot product of anything with `zero` is `zero` */
template <typename T>
MFEM_HOST_DEVICE zero dot(zero, const T&)
{
   return zero{};
}

/**
 * @brief Removes 1s from tensor dimensions
 * For example, a tensor<T, 1, 10> is equivalent to a tensor<T, 10>
 * @tparam T The scalar type of the tensor
 * @tparam n1 The first dimension
 * @tparam n2 The second dimension
 */
template <typename T, int n1, int n2 = 1>
using reduced_tensor = typename std::conditional<
                       (n1 == 1 && n2 == 1), T,
                       typename std::conditional<n1 == 1, tensor<T, n2>,
                       typename std::conditional<n2 == 1, tensor<T, n1>, tensor<T, n1, n2>
                       >::type
                       >::type
                       >::type;

/**
 * @brief Creates a tensor of requested dimension by subsequent calls to a functor
 * Can be thought of as analogous to @p std::transform in that the set of possible
 * indices for dimensions @p n are transformed into the values of the tensor by @a f
 * @tparam lambda_type The type of the functor
 * @param[in] f The functor to generate the tensor values from
 *
 * @note the different cases of 0D, 1D, 2D, 3D, and 4D are implemented separately
 *       to work around a limitation in nvcc involving __host__ __device__ lambdas with `auto` parameters.
 */
template <typename lambda_type>
MFEM_HOST_DEVICE constexpr auto make_tensor(lambda_type f) ->
tensor<decltype(f())>
{
   return {f()};
}

/**
 * @brief Creates a tensor of requested dimension by subsequent calls to a functor
 *
 * @tparam n1 The dimension of the tensor
 * @tparam lambda_type The type of the functor
 * @param[in] f The functor to generate the tensor values from
 * @pre @a f must accept @p n1 arguments of type @p int
 *
 * @note the different cases of 0D, 1D, 2D, 3D, and 4D are implemented separately
 *       to work around a limitation in nvcc involving __host__ __device__ lambdas with `auto` parameters.
 */
template <int n1, typename lambda_type>
MFEM_HOST_DEVICE auto make_tensor(lambda_type f) ->
tensor<decltype(f(n1)), n1>
{
   using T = decltype(f(n1));
   tensor<T, n1> A{};
   for (int i = 0; i < n1; i++)
   {
      A(i) = f(i);
   }
   return A;
}

/**
 * @brief Creates a tensor of requested dimension by subsequent calls to a functor
 *
 * @tparam n1 The first dimension of the tensor
 * @tparam n2 The second dimension of the tensor
 * @tparam lambda_type The type of the functor
 * @param[in] f The functor to generate the tensor values from
 * @pre @a f must accept @p n1 x @p n2 arguments of type @p int
 *
 * @note the different cases of 0D, 1D, 2D, 3D, and 4D are implemented separately
 *       to work around a limitation in nvcc involving __host__ __device__ lambdas with `auto` parameters.
 */
template <int n1, int n2, typename lambda_type>
MFEM_HOST_DEVICE auto make_tensor(lambda_type f) ->
tensor<decltype(f(n1, n2)), n1, n2>
{
   using T = decltype(f(n1, n2));
   tensor<T, n1, n2> A{};
   for (int i = 0; i < n1; i++)
   {
      for (int j = 0; j < n2; j++)
      {
         A(i, j) = f(i, j);
      }
   }
   return A;
}

/**
 * @brief Creates a tensor of requested dimension by subsequent calls to a functor
 *
 * @tparam n1 The first dimension of the tensor
 * @tparam n2 The second dimension of the tensor
 * @tparam n3 The third dimension of the tensor
 * @tparam lambda_type The type of the functor
 * @param[in] f The functor to generate the tensor values from
 * @pre @a f must accept @p n1 x @p n2 x @p n3 arguments of type @p int
 *
 * @note the different cases of 0D, 1D, 2D, 3D, and 4D are implemented separately
 *       to work around a limitation in nvcc involving __host__ __device__ lambdas with `auto` parameters.
 */
template <int n1, int n2, int n3, typename lambda_type>
MFEM_HOST_DEVICE auto make_tensor(lambda_type f) ->
tensor<decltype(f(n1, n2, n3)), n1, n2, n3>
{
   using T = decltype(f(n1, n2, n3));
   tensor<T, n1, n2, n3> A{};
   for (int i = 0; i < n1; i++)
   {
      for (int j = 0; j < n2; j++)
      {
         for (int k = 0; k < n3; k++)
         {
            A(i, j, k) = f(i, j, k);
         }
      }
   }
   return A;
}

/**
 * @brief Creates a tensor of requested dimension by subsequent calls to a functor
 *
 * @tparam n1 The first dimension of the tensor
 * @tparam n2 The second dimension of the tensor
 * @tparam n3 The third dimension of the tensor
 * @tparam n4 The fourth dimension of the tensor
 * @tparam lambda_type The type of the functor
 * @param[in] f The functor to generate the tensor values from
 * @pre @a f must accept @p n1 x @p n2 x @p n3 x @p n4 arguments of type @p int
 *
 * @note the different cases of 0D, 1D, 2D, 3D, and 4D are implemented separately
 *       to work around a limitation in nvcc involving __host__ __device__ lambdas with `auto` parameters.
 */
template <int n1, int n2, int n3, int n4, typename lambda_type>
MFEM_HOST_DEVICE auto make_tensor(lambda_type f) ->
tensor<decltype(f(n1, n2, n3, n4)), n1, n2, n3, n4>
{
   using T = decltype(f(n1, n2, n3, n4));
   tensor<T, n1, n2, n3, n4> A{};
   for (int i = 0; i < n1; i++)
   {
      for (int j = 0; j < n2; j++)
      {
         for (int k = 0; k < n3; k++)
         {
            for (int l = 0; l < n4; l++)
            {
               A(i, j, k, l) = f(i, j, k, l);
            }
         }
      }
   }
   return A;
}

// needs to be generalized
template <typename T, int m, int n> MFEM_HOST_DEVICE
tensor<T, n> get_col(tensor<T, m, n> A, int j)
{
   tensor<T, n> c{};
   c(0) = A[0][j];
   c(1) = A[1][j];
   return c;
}

/// @overload
template <typename T> MFEM_HOST_DEVICE
tensor<T, 1> get_col(tensor<T, 1, 1> A, int j)
{
   return tensor<T, 1> {A[0][0]};
}

/**
 * @brief return the sum of two tensors
 * @tparam S the underlying type of the lefthand argument
 * @tparam T the underlying type of the righthand argument
 * @tparam n integers describing the tensor shape
 * @param[in] A The lefthand operand
 * @param[in] B The righthand operand
 */
template <typename S, typename T, int... n>
MFEM_HOST_DEVICE auto operator+(const tensor<S, n...>& A,
                                const tensor<T, n...>& B) ->
tensor<decltype(S {} + T{}), n...>
{
   tensor<decltype(S{} + T{}), n...> C{};
   for (int i = 0; i < tensor<T, n...>::first_dim; i++)
   {
      C[i] = A[i] + B[i];
   }
   return C;
}

/**
 * @brief return the unary negation of a tensor
 * @tparam T the underlying type of the righthand argument
 * @tparam n integers describing the tensor shape
 * @param[in] A The tensor to negate
 */
template <typename T, int... n>
MFEM_HOST_DEVICE tensor<T, n...> operator-(const tensor<T, n...>& A)
{
   tensor<T, n...> B{};
   for (int i = 0; i < tensor<T, n...>::first_dim; i++)
   {
      B[i] = -A[i];
   }
   return B;
}

/**
 * @brief return the difference of two tensors
 * @tparam S the underlying type of the lefthand argument
 * @tparam T the underlying type of the righthand argument
 * @tparam n integers describing the tensor shape
 * @param[in] A The lefthand operand
 * @param[in] B The righthand operand
 */
template <typename S, typename T, int... n>
MFEM_HOST_DEVICE auto operator-(const tensor<S, n...>& A,
                                const tensor<T, n...>& B) ->
tensor<decltype(S {} + T{}), n...>
{
   tensor<decltype(S{} + T{}), n...> C{};
   for (int i = 0; i < tensor<T, n...>::first_dim; i++)
   {
      C[i] = A[i] - B[i];
   }
   return C;
}

/**
 * @brief multiply a tensor by a scalar value
 * @tparam S the scalar value type. Must be arithmetic (e.g. float, real_t, int) or a dual number
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam n integers describing the tensor shape
 * @param[in] scale The scaling factor
 * @param[in] A The tensor to be scaled
 */
template <typename S, typename T, int... n,
          typename = typename std::enable_if<std::is_arithmetic<S>::value ||
                                             is_dual_number<S>::value>::type>
MFEM_HOST_DEVICE auto operator*(S scale, const tensor<T, n...>& A) ->
tensor<decltype(S {} * T{}), n...>
{
   tensor<decltype(S{} * T{}), n...> C{};
   for (int i = 0; i < tensor<T, n...>::first_dim; i++)
   {
      C[i] = scale * A[i];
   }
   return C;
}

/**
 * @brief multiply a tensor by a scalar value
 * @tparam S the scalar value type. Must be arithmetic (e.g. float, real_t, int) or a dual number
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam n integers describing the tensor shape
 * @param[in] A The tensor to be scaled
 * @param[in] scale The scaling factor
 */
template <typename S, typename T, int... n,
          typename = typename std::enable_if<std::is_arithmetic<S>::value ||
                                             is_dual_number<S>::value>::type>
MFEM_HOST_DEVICE auto operator*(const tensor<T, n...>& A, S scale) ->
tensor<decltype(T {} * S{}), n...>
{
   tensor<decltype(T{} * S{}), n...> C{};
   for (int i = 0; i < tensor<T, n...>::first_dim; i++)
   {
      C[i] = A[i] * scale;
   }
   return C;
}

/**
 * @brief divide a scalar by each element in a tensor
 * @tparam S the scalar value type. Must be arithmetic (e.g. float, real_t, int) or a dual number
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam n integers describing the tensor shape
 * @param[in] scale The numerator
 * @param[in] A The tensor of denominators
 */
template <typename S, typename T, int... n,
          typename = typename std::enable_if<std::is_arithmetic<S>::value ||
                                             is_dual_number<S>::value>::type>
MFEM_HOST_DEVICE auto operator/(S scale, const tensor<T, n...>& A) ->
tensor<decltype(S {} * T{}), n...>
{
   tensor<decltype(S{} * T{}), n...> C{};
   for (int i = 0; i < tensor<T, n...>::first_dim; i++)
   {
      C[i] = scale / A[i];
   }
   return C;
}

/**
 * @brief divide a tensor by a scalar
 * @tparam S the scalar value type. Must be arithmetic (e.g. float, real_t, int) or a dual number
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam n integers describing the tensor shape
 * @param[in] A The tensor of numerators
 * @param[in] scale The denominator
 */
template <typename S, typename T, int... n,
          typename = typename std::enable_if<std::is_arithmetic<S>::value ||
                                             is_dual_number<S>::value>::type>
MFEM_HOST_DEVICE auto operator/(const tensor<T, n...>& A, S scale) ->
tensor<decltype(T {} * S{}), n...>
{
   tensor<decltype(T{} * S{}), n...> C{};
   for (int i = 0; i < tensor<T, n...>::first_dim; i++)
   {
      C[i] = A[i] / scale;
   }
   return C;
}

/**
 * @brief compound assignment (+) on tensors
 * @tparam S the underlying type of the tensor (lefthand) argument
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam n integers describing the tensor shape
 * @param[in] A The lefthand tensor
 * @param[in] B The righthand tensor
 */
template <typename S, typename T, int... n> MFEM_HOST_DEVICE
tensor<S, n...>& operator+=(tensor<S, n...>& A,
                            const tensor<T, n...>& B)
{
   for (int i = 0; i < tensor<S, n...>::first_dim; i++)
   {
      A[i] += B[i];
   }
   return A;
}

/**
 * @brief compound assignment (+) on tensors
 * @tparam T the underlying type of the tensor argument
 * @param[in] A The lefthand tensor
 * @param[in] B The righthand tensor
 */
template <typename T> MFEM_HOST_DEVICE
tensor<T>& operator+=(tensor<T>& A, const T& B)
{
   return A.values += B;
}

/**
 * @brief compound assignment (+) on tensors
 * @tparam T the underlying type of the tensor argument
 * @param[in] A The lefthand tensor
 * @param[in] B The righthand tensor
 */
template <typename T> MFEM_HOST_DEVICE
tensor<T, 1>& operator+=(tensor<T, 1>& A, const T& B)
{
   return A.values += B;
}

/**
 * @brief compound assignment (+) on tensors
 * @tparam T the underlying type of the tensor argument
 * @param[in] A The lefthand tensor
 * @param[in] B The righthand tensor
 */
template <typename T> MFEM_HOST_DEVICE
tensor<T, 1, 1>& operator+=(tensor<T, 1, 1>& A, const T& B)
{
   return A.values += B;
}

/**
 * @brief compound assignment (+) between a tensor and zero (no-op)
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam n integers describing the tensor shape
 * @param[in] A The lefthand tensor
 */
template <typename T, int... n> MFEM_HOST_DEVICE
tensor<T, n...>& operator+=(tensor<T, n...>& A, zero)
{
   return A;
}

/**
 * @brief compound assignment (-) on tensors
 * @tparam S the underlying type of the tensor (lefthand) argument
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam n integers describing the tensor shape
 * @param[in] A The lefthand tensor
 * @param[in] B The righthand tensor
 */
template <typename S, typename T, int... n> MFEM_HOST_DEVICE
tensor<S, n...>& operator-=(tensor<S, n...>& A, const tensor<T, n...>& B)
{
   for (int i = 0; i < tensor<S, n...>::first_dim; i++)
   {
      A[i] -= B[i];
   }
   return A;
}

/**
 * @brief compound assignment (-) between a tensor and zero (no-op)
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam n integers describing the tensor shape
 * @param[in] A The lefthand tensor
 */
template <typename T, int... n> MFEM_HOST_DEVICE
constexpr tensor<T, n...>& operator-=(tensor<T, n...>& A, zero)
{
   return A;
}

/**
 * @brief compute the outer product of two tensors
 * @tparam S the type of the lefthand argument
 * @tparam T the type of the righthand argument
 * @param[in] A The lefthand argument
 * @param[in] B The righthand argument
 *
 * @note this overload implements the special case where both arguments are scalars
 */
template <typename S, typename T> MFEM_HOST_DEVICE
auto outer(S A, T B) -> decltype(A * B)
{
   static_assert(std::is_arithmetic<S>::value && std::is_arithmetic<T>::value,
                 "outer product types must be tensor or arithmetic_type");
   return A * B;
}

template <typename T, int n, int m> MFEM_HOST_DEVICE
tensor<T, n + m> flatten(tensor<T, n, m> A)
{
   tensor<T, n + m> B{};
   for (int i = 0; i < n; i++)
   {
      for (int j = 0; j < m; j++)
      {
         B(i + j * m) = A(i, j);
      }
   }
   return B;
}

/**
 * @overload
 * @note this overload implements the case where the left argument is a scalar, and the right argument is a tensor
 */
template <typename S, typename T, int n> MFEM_HOST_DEVICE
tensor<decltype(S{} * T{}), n> outer(S A, tensor<T, n> B)
{
   static_assert(std::is_arithmetic<S>::value,
                 "outer product types must be tensor or arithmetic_type");
   tensor<decltype(S{} * T{}), n> AB{};
   for (int i = 0; i < n; i++)
   {
      AB[i] = A * B[i];
   }
   return AB;
}

/**
 * @overload
 * @note this overload implements the case where the left argument is a tensor, and the right argument is a scalar
 */
template <typename S, typename T, int m> MFEM_HOST_DEVICE
tensor<decltype(S{} * T{}), m> outer(const tensor<S, m>& A, T B)
{
   static_assert(std::is_arithmetic<T>::value,
                 "outer product types must be tensor or arithmetic_type");
   tensor<decltype(S{} * T{}), m> AB{};
   for (int i = 0; i < m; i++)
   {
      AB[i] = A[i] * B;
   }
   return AB;
}

/**
 * @overload
 * @note this overload implements the case where the left argument is `zero`, and the right argument is a tensor
 */
template <typename T, int n> MFEM_HOST_DEVICE
zero outer(zero, const tensor<T, n>&)
{
   return zero{};
}

/**
 * @overload
 * @note this overload implements the case where the left argument is a tensor, and the right argument is `zero`
 */
template <typename T, int n> MFEM_HOST_DEVICE
zero outer(const tensor<T, n>&, zero)
{
   return zero{};
}

/**
 * @overload
 * @note this overload implements the case where the left argument is a scalar,
 * and the right argument is a tensor
 */
template <typename S, typename T, int m, int n> MFEM_HOST_DEVICE
tensor<decltype(S{} * T{}), m, n> outer(S A, const tensor<T, m, n>& B)
{
   static_assert(std::is_arithmetic<S>::value,
                 "outer product types must be tensor or arithmetic_type");
   tensor<decltype(S{} * T{}), m, n> AB{};
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < n; j++)
      {
         AB[i][j] = A * B[i][j];
      }
   }
   return AB;
}

/**
 * @overload
 * @note this overload implements the case where both arguments are vectors
 */
template <typename S, typename T, int m, int n> MFEM_HOST_DEVICE
tensor<decltype(S{} * T{}), m, n> outer(const tensor<S, m>& A,
                                        const tensor<T, n>& B)
{
   tensor<decltype(S{} * T{}), m, n> AB{};
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < n; j++)
      {
         AB[i][j] = A[i] * B[j];
      }
   }
   return AB;
}

/**
 * @overload
 * @note this overload implements the case where the left argument is a 2nd order tensor, and the right argument is a
 * scalar
 */
template <typename S, typename T, int m, int n> MFEM_HOST_DEVICE
tensor<decltype(S{} * T{}), m, n> outer(const tensor<S, m, n>& A, T B)
{
   static_assert(std::is_arithmetic<T>::value,
                 "outer product types must be tensor or arithmetic_type");
   tensor<decltype(S{} * T{}), m, n> AB{};
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < n; j++)
      {
         AB[i][j] = A[i][j] * B;
      }
   }
   return AB;
}

/**
 * @overload
 * @note this overload implements the case where the left argument is a 2nd order tensor, and the right argument is a
 * first order tensor
 */
template <typename S, typename T, int m, int n, int p> MFEM_HOST_DEVICE
tensor<decltype(S{} * T{}), m, n, p> outer(const tensor<S, m, n>& A,
                                           const tensor<T, p>& B)
{
   tensor<decltype(S{} * T{}), m, n, p> AB{};
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < n; j++)
      {
         for (int k = 0; k < p; k++)
         {
            AB[i][j][k] = A[i][j] * B[k];
         }
      }
   }
   return AB;
}

/**
 * @overload
 * @note this overload implements the case where the left argument is a 1st order tensor, and the right argument is a
 * 2nd order tensor
 */
template <typename S, typename T, int m, int n, int p> MFEM_HOST_DEVICE
tensor<decltype(S{} * T{}), m, n, p> outer(const tensor<S, m>& A,
                                           const tensor<T, n, p>& B)
{
   tensor<decltype(S{} * T{}), m, n, p> AB{};
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < n; j++)
      {
         for (int k = 0; k < p; k++)
         {
            AB[i][j][k] = A[i] * B[j][k];
         }
      }
   }
   return AB;
}

/**
 * @overload
 * @note this overload implements the case where both arguments are second order tensors
 */
template <typename S, typename T, int m, int n, int p, int q> MFEM_HOST_DEVICE
tensor<decltype(S{} * T{}), m, n, p, q> outer(const tensor<S, m, n>& A,
                                              const tensor<T, p, q>& B)
{
   tensor<decltype(S{} * T{}), m, n, p, q> AB{};
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < n; j++)
      {
         for (int k = 0; k < p; k++)
         {
            for (int l = 0; l < q; l++)
            {
               AB[i][j][k][l] = A[i][j] * B[k][l];
            }
         }
      }
   }
   return AB;
}

/**
 * @brief this function contracts over all indices of the two tensor arguments
 * @tparam S the underlying type of the tensor (lefthand) argument
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam m the number of rows
 * @tparam n the number of columns
 * @param[in] A The lefthand tensor
 * @param[in] B The righthand tensor
 */
template <typename S, typename T, int m, int n> MFEM_HOST_DEVICE
auto inner(const tensor<S, m, n>& A, const tensor<T, m, n>& B) ->
decltype(S {} * T{})
{
   decltype(S{} * T{}) sum{};
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < n; j++)
      {
         sum += A[i][j] * B[i][j];
      }
   }
   return sum;
}

/**
 * @brief this function contracts over the "middle" index of the two tensor
 * arguments. E.g. returns tensor C, such that C_ij = sum_kl A_ijkl B_kl.
 * @tparam S the underlying type of the tensor (lefthand) argument
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam n integers describing the tensor shape
 * @param[in] A The lefthand tensor
 * @param[in] B The righthand tensor
 */
template <typename S, typename T, int m, int n, int p> MFEM_HOST_DEVICE
auto dot(const tensor<S, m, n>& A,
         const tensor<T, n, p>& B) ->
tensor<decltype(S {} * T{}), m, p>
{
   tensor<decltype(S{} * T{}), m, p> AB{};
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < p; j++)
      {
         for (int k = 0; k < n; k++)
         {
            AB[i][j] = AB[i][j] + A[i][k] * B[k][j];
         }
      }
   }
   return AB;
}

/**
 * @overload
 * @note vector . matrix
 */
template <typename S, typename T, int m, int n> MFEM_HOST_DEVICE
auto dot(const tensor<S, m>& A, const tensor<T, m, n>& B) ->
tensor<decltype(S {} * T{}), n>
{
   tensor<decltype(S{} * T{}), n> AB{};
   for (int i = 0; i < n; i++)
   {
      for (int j = 0; j < m; j++)
      {
         AB[i] = AB[i] + A[j] * B[j][i];
      }
   }
   return AB;
}

/**
 * @overload
 * @note matrix . vector
 */
template <typename S, typename T, int m, int n> MFEM_HOST_DEVICE
auto dot(const tensor<S, m, n>& A, const tensor<T, n>& B) ->
tensor<decltype(S {} * T{}), m>
{
   tensor<decltype(S{} * T{}), m> AB{};
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < n; j++)
      {
         AB[i] = AB[i] + A[i][j] * B[j];
      }
   }
   return AB;
}

/**
 * @overload
 * @note 3rd-order-tensor . vector
 */
template <typename S, typename T, int m, int n, int p> MFEM_HOST_DEVICE
auto dot(const tensor<S, m, n, p>& A, const tensor<T, p>& B) ->
tensor<decltype(S {} * T{}), m, n>
{
   tensor<decltype(S{} * T{}), m, n> AB{};
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < n; j++)
      {
         for (int k = 0; k < p; k++)
         {
            AB[i][j] += A[i][j][k] * B[k];
         }
      }
   }
   return AB;
}

// /**
//  * @brief Dot product of a vector . vector and vector . tensor
//  *
//  * @tparam S the underlying type of the tensor (lefthand) argument
//  * @tparam T the underlying type of the tensor (righthand) argument
//  * @tparam m the dimension of the first tensor
//  * @tparam n the parameter pack of dimensions of the second tensor
//  * @param A The lefthand tensor
//  * @param B The righthand tensor
//  * @return The computed dot product
//  */
// template <typename S, typename T, int m, int... n>
// auto dot(const tensor<S, m>& A, const tensor<T, m, n...>& B)
// {
//    // this dot product function includes the vector * vector implementation and
//    // the vector * tensor one, since clang emits an error about ambiguous
//    // overloads if they are separate functions. The `if constexpr` expression avoids
//    // using an `else` because that confuses nvcc (11.2) into thinking there's not
//    // a return statement
//    if constexpr (sizeof...(n) == 0)
//    {
//       decltype(S{} * T{}) AB{};
//       for (int i = 0; i < m; i++)
//       {
//          AB += A[i] * B[i];
//       }
//       return AB;
//    }

//    if constexpr (sizeof...(n) > 0)
//    {
//       constexpr int                     dimensions[] = {n...};
//       tensor<decltype(S{} * T{}), n...> AB{};
//       for (int i = 0; i < dimensions[0]; i++)
//       {
//          for (int j = 0; j < m; j++)
//          {
//             AB[i] = AB[i] + A[j] * B[j][i];
//          }
//       }
//       return AB;
//    }
// }

template <typename S, typename T, int m> MFEM_HOST_DEVICE
auto dot(const tensor<S, m>& A, const tensor<T, m>& B) ->
decltype(S {} * T{})
{
   decltype(S{} * T{}) AB{};
   for (int i = 0; i < m; i++)
   {
      AB += A[i] * B[i];
   }
   return AB;
}

template <typename T, int m> MFEM_HOST_DEVICE
auto dot(const tensor<T, m>& A, const tensor<T, m>& B) ->
decltype(T {})
{
   decltype(T{}) AB{};
   for (int i = 0; i < m; i++)
   {
      AB += A[i] * B[i];
   }
   return AB;
}

template <typename S, typename T, int m, int n0, int n1, int... n>
MFEM_HOST_DEVICE
auto dot(const tensor<S, m>& A, const tensor<T, m, n0, n1, n...>& B) ->
tensor<decltype(S {} * T{}), n0, n1, n...>
{
   tensor<decltype(S{} * T{}), n0, n1, n...> AB{};
   for (int i = 0; i < n0; i++)
   {
      for (int j = 0; j < m; j++)
      {
         AB[i] = AB[i] + A[j] * B[j][i];
      }
   }
   return AB;
}

/**
 * @overload
 * @note vector . matrix . vector
 */
template <typename S, typename T, typename U, int m, int n> MFEM_HOST_DEVICE
auto dot(const tensor<S, m>& u, const tensor<T, m, n>& A,
         const tensor<U, n>& v) ->
decltype(S {} * T{} * U{})
{
   decltype(S{} * T{} * U{}) uAv{};
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < n; j++)
      {
         uAv += u[i] * A[i][j] * v[j];
      }
   }
   return uAv;
}

/**
 * @brief real_t dot product, contracting over the two "middle" indices
 * @tparam S the underlying type of the tensor (lefthand) argument
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam m first dimension of A
 * @tparam n second dimension of A
 * @tparam p third dimension of A, first dimensions of B
 * @tparam q fourth dimension of A, second dimensions of B
 * @param[in] A The lefthand tensor
 * @param[in] B The righthand tensor
 */
template <typename S, typename T, int m, int n, int p, int q> MFEM_HOST_DEVICE
auto ddot(const tensor<S, m, n, p, q>& A, const tensor<T, p, q>& B) ->
tensor<decltype(S {} * T{}), m, n>
{
   tensor<decltype(S{} * T{}), m, n> AB{};
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < n; j++)
      {
         for (int k = 0; k < p; k++)
         {
            for (int l = 0; l < q; l++)
            {
               AB[i][j] += A[i][j][k][l] * B[k][l];
            }
         }
      }
   }
   return AB;
}

/**
 * @overload
 * @note 3rd-order-tensor : 2nd-order-tensor. Returns vector C, such that C_i =
 * sum_jk A_ijk B_jk.
 */
template <typename S, typename T, int m, int n, int p> MFEM_HOST_DEVICE
auto ddot(const tensor<S, m, n, p>& A, const tensor<T, n, p>& B) ->
tensor<decltype(S {} * T{}), m>
{
   tensor<decltype(S{} * T{}), m> AB{};
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < n; j++)
      {
         for (int k = 0; k < p; k++)
         {
            AB[i] += A[i][j][k] * B[j][k];
         }
      }
   }
   return AB;
}

/**
 * @overload
 * @note 2nd-order-tensor : 2nd-order-tensor, like inner()
 */
template <typename S, typename T, int m, int n> MFEM_HOST_DEVICE
auto ddot(const tensor<S, m, n>& A, const tensor<T, m, n>& B) ->
decltype(S {} * T{})
{
   decltype(S{} * T{}) AB{};
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < n; j++)
      {
         AB += A[i][j] * B[i][j];
      }
   }
   return AB;
}

/**
 * @brief this is a shorthand for dot(A, B)
 */
template <typename S, typename T, int... m, int... n> MFEM_HOST_DEVICE
auto operator*(const tensor<S, m...>& A, const tensor<T, n...>& B) ->
decltype(dot(A, B))
{
   return dot(A, B);
}

/**
 * @brief Returns the squared Frobenius norm of the tensor
 * @param[in] A The tensor to obtain the squared norm from
 */
template <typename T, int m> MFEM_HOST_DEVICE
T sqnorm(const tensor<T, m>& A)
{
   T total{};
   for (int i = 0; i < m; i++)
   {
      total += A[i] * A[i];
   }
   return total;
}

/**
 * @overload
 * @brief Returns the squared Frobenius norm of the tensor
 */
template <typename T, int m, int n> MFEM_HOST_DEVICE
T sqnorm(const tensor<T, m, n>& A)
{
   T total{};
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < n; j++)
      {
         total += A[i][j] * A[i][j];
      }
   }
   return total;
}

/**
 * @brief Returns the Frobenius norm of the tensor
 * @param[in] A The tensor to obtain the norm from
 */
template <typename T, int... n> MFEM_HOST_DEVICE
T norm(const tensor<T, n...>& A)
{
   return std::sqrt(sqnorm(A));
}

template <typename T, int n, int m> MFEM_HOST_DEVICE
T weight(const tensor<T, n, m>& A)
{
   static_assert((n == m) || ((n == 2) && (m == 1)) || ((n == 3) && (m == 1)) ||
                 ((n == 3) && (m == 2)), "unsupported combination of n and m");
   if constexpr (n == m)
   {
      return det(A);
   }
   if constexpr (((n == 2) && (m == 1)) ||
                 ((n == 3) && (m == 1)))
   {
      return norm(A);
   }
   else if constexpr ((n == 3) && (m == 2))
   {
      T E = A[0][0] * A[0][0] + A[1][0] * A[1][0] + A[2][0] * A[2][0];
      T G = A[0][1] * A[0][1] + A[1][1] * A[1][1] + A[2][1] * A[2][1];
      T F = A[0][0] * A[0][1] + A[1][0] * A[1][1] + A[2][0] * A[2][1];
      return std::sqrt(E * G - F * F);
   }
   // Never reached because of the static_assert, but avoids compiler warning.
   return T{};
}

/**
 * @brief Normalizes the tensor
 * Each element is divided by the Frobenius norm of the tensor, @see norm
 * @param[in] A The tensor to normalize
 */
template <typename T, int... n> MFEM_HOST_DEVICE
auto normalize(const tensor<T, n...>& A) ->
decltype(A / norm(A))
{
   return A / norm(A);
}

/**
 * @brief Returns the trace of a square matrix
 * @param[in] A The matrix to compute the trace of
 * @return The sum of the elements on the main diagonal
 */
template <typename T, int n> MFEM_HOST_DEVICE
T tr(const tensor<T, n, n>& A)
{
   T trA{};
   for (int i = 0; i < n; i++)
   {
      trA = trA + A[i][i];
   }
   return trA;
}

/**
 * @brief Returns the symmetric part of a square matrix
 * @param[in] A The matrix to obtain the symmetric part of
 * @return (1/2) * (A + A^T)
 */
template <typename T, int n> MFEM_HOST_DEVICE
tensor<T, n, n> sym(const tensor<T, n, n>& A)
{
   tensor<T, n, n> symA{};
   for (int i = 0; i < n; i++)
   {
      for (int j = 0; j < n; j++)
      {
         symA[i][j] = 0.5 * (A[i][j] + A[j][i]);
      }
   }
   return symA;
}

/**
 * @brief Calculates the deviator of a matrix (rank-2 tensor)
 * @param[in] A The matrix to calculate the deviator of
 * In the context of stress tensors, the deviator is obtained by
 * subtracting the mean stress (average of main diagonal elements)
 * from each element on the main diagonal
 */
template <typename T, int n> MFEM_HOST_DEVICE
tensor<T, n, n> dev(const tensor<T, n, n>& A)
{
   auto devA = A;
   auto trA  = tr(A);
   for (int i = 0; i < n; i++)
   {
      devA[i][i] -= trA / n;
   }
   return devA;
}

/**
 * @brief Obtains the identity matrix of the specified dimension
 * @return I_dim
 */
template <int dim>
MFEM_HOST_DEVICE tensor<real_t, dim, dim> IdentityMatrix()
{
   tensor<real_t, dim, dim> I{};
   for (int i = 0; i < dim; i++)
   {
      for (int j = 0; j < dim; j++)
      {
         I[i][j] = (i == j);
      }
   }
   return I;
}

/**
 * @brief Returns the transpose of the matrix
 * @param[in] A The matrix to obtain the transpose of
 */
template <typename T, int m, int n> MFEM_HOST_DEVICE
tensor<T, n, m> transpose(const tensor<T, m, n>& A)
{
   tensor<T, n, m> AT{};
   for (int i = 0; i < n; i++)
   {
      for (int j = 0; j < m; j++)
      {
         AT[i][j] = A[j][i];
      }
   }
   return AT;
}

/**
 * @brief Returns the determinant of a matrix
 * @param[in] A The matrix to obtain the determinant of
 */
template <typename T> MFEM_HOST_DEVICE
T det(const tensor<T, 1, 1>& A)
{
   return A[0][0];
}
/// @overload
template <typename T> MFEM_HOST_DEVICE
T det(const tensor<T, 2, 2>& A)
{
   return A[0][0] * A[1][1] - A[0][1] * A[1][0];
}
/// @overload
template <typename T> MFEM_HOST_DEVICE
T det(const tensor<T, 3, 3>& A)
{
   return A[0][0] * A[1][1] * A[2][2] + A[0][1] * A[1][2] * A[2][0] + A[0][2] *
          A[1][0] * A[2][1] -
          A[0][0] * A[1][2] * A[2][1] - A[0][1] * A[1][0] * A[2][2] - A[0][2] * A[1][1] *
          A[2][0];
}

template <typename T> MFEM_HOST_DEVICE
std::tuple<tensor<T, 1>, tensor<T, 1, 1>> eig(tensor<T, 1, 1> &A)
{
   return {tensor<T, 1>{A[0][0]}, tensor<T, 1, 1>{{{1.0}}}};
}

template <typename T> MFEM_HOST_DEVICE
std::tuple<tensor<T, 2>, tensor<T, 2, 2>> eig(tensor<T, 2, 2> &A)
{
   tensor<T, 2> e;
   tensor<T, 2, 2> v;

   real_t d0 = A(0, 0);
   real_t d2 = A(0, 1);
   real_t d3 = A(1, 1);
   real_t c, s;

   if (d2 == 0.0)
   {
      c = 1.0;
      s = 0.0;
   }
   else
   {
      real_t t;
      const real_t zeta = (d3 - d0) / (2.0 * d2);
      const real_t azeta = fabs(zeta);
      if (azeta < std::sqrt(1.0/std::numeric_limits<T>::epsilon()))
      {
         t = copysign(1./(azeta + std::sqrt(1. + zeta*zeta)), zeta);
      }
      else
      {
         t = copysign(0.5/azeta, zeta);
      }
      c = std::sqrt(1./(1. + t*t));
      s = c*t;
      t *= d2;
      d0 -= t;
      d3 += t;
   }

   if (d0 <= d3)
   {
      e(0) = d0;
      e(1) = d3;
      v(0, 0) =  c;
      v(1, 0) = -s;
      v(0, 1) =  s;
      v(1, 1) =  c;
   }
   else
   {
      e(0) = d3;
      e(1) = d0;
      v(0, 0) =  s;
      v(1, 0) =  c;
      v(0, 1) =  c;
      v(1, 1) = -s;
   }

   return {e, v};
}

template <typename T> MFEM_HOST_DEVICE
void GetScalingFactor(const T &d_max, T &mult)
{
   int d_exp;
   if (d_max > 0.)
   {
      mult = frexp(d_max, &d_exp);
      if (d_exp == std::numeric_limits<T>::max_exponent)
      {
         mult *= std::numeric_limits<T>::radix;
      }
      mult = d_max/mult;
   }
   else
   {
      mult = 1.;
   }
}

template <typename T> MFEM_HOST_DEVICE
T calcsv(const tensor<T, 1, 1> A, const int i)
{
   return A[0][0];
}

/**
 * @brief Compute the i-th singular value of a 2x2 matrix A
 */
template <typename T> MFEM_HOST_DEVICE
T calcsv(const tensor<T, 2, 2> A, const int i)
{
   real_t mult;
   real_t d0, d1, d2, d3;
   d0 = A(0, 0);
   d1 = A(1, 0);
   d2 = A(0, 1);
   d3 = A(1, 1);

   real_t d_max = fabs(d0);
   if (d_max < fabs(d1)) { d_max = fabs(d1); }
   if (d_max < fabs(d2)) { d_max = fabs(d2); }
   if (d_max < fabs(d3)) { d_max = fabs(d3); }

   GetScalingFactor(d_max, mult);

   d0 /= mult;
   d1 /= mult;
   d2 /= mult;
   d3 /= mult;

   real_t t = 0.5*((d0+d2)*(d0-d2)+(d1-d3)*(d1+d3));
   real_t s = d0*d2 + d1*d3;
   s = std::sqrt(0.5*(d0*d0 + d1*d1 + d2*d2 + d3*d3) + std::sqrt(t*t + s*s));

   if (s == 0.0)
   {
      return 0.0;
   }
   t = fabs(d0*d3 - d1*d2) / s;
   if (t > s)
   {
      if (i == 0)
      {
         return t*mult;
      }
      return s*mult;
   }
   if (i == 0)
   {
      return s*mult;
   }
   return t*mult;
}


/**
 * @brief Return whether a square rank 2 tensor is symmetric
 *
 * @tparam n The height of the tensor
 * @param A The square rank 2 tensor
 * @param abs_tolerance The absolute tolerance to check for symmetry
 * @return Whether the square rank 2 tensor (matrix) is symmetric
 */
template <int n> MFEM_HOST_DEVICE
bool is_symmetric(tensor<real_t, n, n> A, real_t abs_tolerance = 1.0e-8_r)
{
   for (int i = 0; i < n; ++i)
   {
      for (int j = i + 1; j < n; ++j)
      {
         if (std::abs(A(i, j) - A(j, i)) > abs_tolerance)
         {
            return false;
         }
      }
   }
   return true;
}

/**
 * @brief Return whether a matrix is symmetric and positive definite
 * This check uses Sylvester's criterion, checking that each upper left subtensor has a
 * determinant greater than zero.
 *
 * @param A The matrix to test for positive definiteness
 * @return Whether the matrix is positive definite
 */
inline MFEM_HOST_DEVICE
bool is_symmetric_and_positive_definite(tensor<real_t, 2, 2> A)
{
   if (!is_symmetric(A))
   {
      return false;
   }
   if (A(0, 0) < 0.0)
   {
      return false;
   }
   if (det(A) < 0.0)
   {
      return false;
   }
   return true;
}
/// @overload
inline MFEM_HOST_DEVICE
bool is_symmetric_and_positive_definite(tensor<real_t, 3, 3> A)
{
   if (!is_symmetric(A))
   {
      return false;
   }
   if (det(A) < 0.0)
   {
      return false;
   }
   auto subtensor = make_tensor<2, 2>([A](int i, int j) { return A(i, j); });
   if (!is_symmetric_and_positive_definite(subtensor))
   {
      return false;
   }
   return true;
}

/**
 * @brief Solves Ax = b for x using Gaussian elimination with partial pivoting
 * @param[in] A The coefficient matrix A
 * @param[in] b The righthand side vector b
 * @note @a A and @a b are by-value as they are mutated as part of the elimination
 */
template <typename T, int n> MFEM_HOST_DEVICE
tensor<T, n> linear_solve(tensor<T, n, n> A, const tensor<T, n> b)
{
   auto abs  = [](real_t x) { return (x < 0) ? -x : x; };
   auto swap_vector = [](tensor<T, n>& x, tensor<T, n>& y)
   {
      auto tmp = x;
      x        = y;
      y        = tmp;
   };
   auto swap_scalar = [](T& x, T& y)
   {
      auto tmp = x;
      x        = y;
      y        = tmp;
   };


   tensor<real_t, n> x{};

   for (int i = 0; i < n; i++)
   {
      // Search for maximum in this column
      real_t max_val = abs(A[i][i]);

      int max_row = i;
      for (int j = i + 1; j < n; j++)
      {
         if (abs(A[j][i]) > max_val)
         {
            max_val = abs(A[j][i]);
            max_row = j;
         }
      }

      swap_scalar(b[max_row], b[i]);
      swap_vector(A[max_row], A[i]);

      // zero entries below in this column
      for (int j = i + 1; j < n; j++)
      {
         real_t c = -A[j][i] / A[i][i];
         A[j] += c * A[i];
         b[j] += c * b[i];
         A[j][i] = 0;
      }
   }

   // Solve equation Ax=b for an upper triangular matrix A
   for (int i = n - 1; i >= 0; i--)
   {
      x[i] = b[i] / A[i][i];
      for (int j = i - 1; j >= 0; j--)
      {
         b[j] -= A[j][i] * x[i];
      }
   }

   return x;
}

/**
 * @brief Inverts a matrix
 * @param[in] A The matrix to invert
 * @note Uses a shortcut for inverting a 1x1, 2x2 and 3x3 matrix
 */
template <typename T>
inline MFEM_HOST_DEVICE tensor<T, 1, 1> inv(const tensor<T, 1, 1>& A)
{
   return tensor<T, 1, 1> {{{T{1.0} / A[0][0]}}};
}

template <typename T>
inline MFEM_HOST_DEVICE tensor<T, 2, 2> inv(const tensor<T, 2, 2>& A)
{
   T inv_detA(1.0_r / det(A));

   tensor<T, 2, 2> invA{};

   invA[0][0] = A[1][1] * inv_detA;
   invA[0][1] = -A[0][1] * inv_detA;
   invA[1][0] = -A[1][0] * inv_detA;
   invA[1][1] = A[0][0] * inv_detA;

   return invA;
}

/**
 * @overload
 * @note Uses a shortcut for inverting a 3-by-3 matrix
 */
template <typename T>
inline MFEM_HOST_DEVICE tensor<T, 3, 3> inv(const tensor<T, 3, 3>& A)
{
   T inv_detA(1.0_r / det(A));

   tensor<T, 3, 3> invA{};

   invA[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) * inv_detA;
   invA[0][1] = (A[0][2] * A[2][1] - A[0][1] * A[2][2]) * inv_detA;
   invA[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * inv_detA;
   invA[1][0] = (A[1][2] * A[2][0] - A[1][0] * A[2][2]) * inv_detA;
   invA[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) * inv_detA;
   invA[1][2] = (A[0][2] * A[1][0] - A[0][0] * A[1][2]) * inv_detA;
   invA[2][0] = (A[1][0] * A[2][1] - A[1][1] * A[2][0]) * inv_detA;
   invA[2][1] = (A[0][1] * A[2][0] - A[0][0] * A[2][1]) * inv_detA;
   invA[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) * inv_detA;

   return invA;
}
/**
 * @overload
 * @note For N-by-N matrices with N > 3, requires Gaussian elimination
 * with partial pivoting
 */
template <typename T, int n>
MFEM_HOST_DEVICE
typename std::enable_if<(n > 3), tensor<T, n, n>>::type
                                               inv(const tensor<T, n, n>& A)
{
   auto abs  = [](T x) { return (x < 0) ? -x : x; };
   auto swap = [](tensor<T, n>& x, tensor<T, n>& y)
   {
      auto tmp = x;
      x        = y;
      y        = tmp;
   };

   tensor<T, n, n> B = IdentityMatrix<n>();

   for (int i = 0; i < n; i++)
   {
      // Search for maximum in this column
      T max_val = abs(A[i][i]);

      int max_row = i;
      for (int j = i + 1; j < n; j++)
      {
         if (abs(A[j][i]) > max_val)
         {
            max_val = abs(A[j][i]);
            max_row = j;
         }
      }

      swap(B[max_row], B[i]);
      swap(A[max_row], A[i]);

      // zero entries below in this column
      for (int j = i + 1; j < n; j++)
      {
         if (A[j][i] != 0.0)
         {
            T c = -A[j][i] / A[i][i];
            A[j] += c * A[i];
            B[j] += c * B[i];
            A[j][i] = 0;
         }
      }
   }

   // upper triangular solve
   for (int i = n - 1; i >= 0; i--)
   {
      B[i] = B[i] / A[i][i];
      for (int j = i - 1; j >= 0; j--)
      {
         if (A[j][i] != 0.0)
         {
            B[j] -= A[j][i] * B[i];
         }
      }
   }

   return B;
}

/**
 * @overload
 * @note when inverting a tensor of dual numbers,
 * hardcode the analytic derivative of the
 * inverse of a square matrix, rather than
 * apply Gauss elimination directly on the dual number types
 *
 * TODO: compare performance of this hardcoded implementation to just using inv() directly
 */
template <typename value_type, typename gradient_type, int n> MFEM_HOST_DEVICE
dual<value_type, gradient_type> inv(
   tensor<dual<value_type, gradient_type>, n, n> A)
{
   auto invA = inv(get_value(A));
   return make_tensor<n, n>([&](int i, int j)
   {
      auto          value = invA[i][j];
      gradient_type gradient{};
      for (int k = 0; k < n; k++)
      {
         for (int l = 0; l < n; l++)
         {
            gradient -= invA[i][k] * A[k][l].gradient * invA[l][j];
         }
      }
      return dual<value_type, gradient_type> {value, gradient};
   });
}

/**
 * @brief recursively serialize the entries in a tensor to an output stream.
 * Output format uses braces and comma separators to mimic C syntax for multidimensional array
 * initialization.
 *
 * @param[in] os The stream to work with standard output streams
 * @param[in] A The tensor to write out
 */
template <typename T, int... n>
std::ostream& operator<<(std::ostream& os, const tensor<T, n...>& A)
{
   os << '{' << A[0];
   for (int i = 1; i < tensor<T, n...>::first_dim; i++)
   {
      os << ", " << A[i];
   }
   os << '}';
   return os;
}

/**
 * @brief replace all entries in a tensor satisfying |x| < 1.0e-10 by literal zero
 * @param[in] A The tensor to "chop"
 */
template <int n> MFEM_HOST_DEVICE
tensor<real_t, n> chop(const tensor<real_t, n>& A)
{
   auto copy = A;
   for (int i = 0; i < n; i++)
   {
      if (copy[i] * copy[i] < 1.0e-20)
      {
         copy[i] = 0.0;
      }
   }
   return copy;
}

/// @overload
template <int m, int n> MFEM_HOST_DEVICE
tensor<real_t, m, n> chop(const tensor<real_t, m, n>& A)
{
   auto copy = A;
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < n; j++)
      {
         if (copy[i][j] * copy[i][j] < 1.0e-20)
         {
            copy[i][j] = 0.0;
         }
      }
   }
   return copy;
}

/// @cond
namespace detail
{
template <typename T1, typename T2>
struct outer_prod;

template <int... m, int... n>
struct outer_prod<tensor<real_t, m...>, tensor<real_t, n...>>
{
   using type = tensor<real_t, m..., n...>;
};

template <int... n>
struct outer_prod<real_t, tensor<real_t, n...>>
{
   using type = tensor<real_t, n...>;
};

template <int... n>
struct outer_prod<tensor<real_t, n...>, real_t>
{
   using type = tensor<real_t, n...>;
};

template <>
struct outer_prod<real_t, real_t>
{
   using type = tensor<real_t>;
};

template <typename T>
struct outer_prod<zero, T>
{
   using type = zero;
};

template <typename T>
struct outer_prod<T, zero>
{
   using type = zero;
};

}  // namespace detail
/// @endcond

/**
 * @brief a type function that returns the tensor type of an outer product of two tensors
 * @tparam T1 the first argument to the outer product
 * @tparam T2 the second argument to the outer product
 */
template <typename T1, typename T2>
using outer_product_t = typename detail::outer_prod<T1, T2>::type;

/**
 * @brief Retrieves the gradient component of a real_t (which is nothing)
 * @return The sentinel, @see zero
 */
inline MFEM_HOST_DEVICE zero get_gradient(real_t /* arg */) { return zero{}; }

/**
 * @brief get the gradient of type `tensor` (note: since its stored type is not a dual
 * number, the derivative term is identically zero)
 * @return The sentinel, @see zero
 */
template <int... n>
MFEM_HOST_DEVICE zero get_gradient(const tensor<real_t, n...>& /* arg */)
{
   return zero{};
}

/**
 * @brief evaluate the change (to first order) in a function, f, given a small change in the input argument, dx.
 */
inline MFEM_HOST_DEVICE zero chain_rule(const zero /* df_dx */,
                                        const zero /* dx */) { return zero{}; }

/**
 * @overload
 * @note this overload implements a no-op for the case where the gradient w.r.t. an input argument is identically zero
 */
template <typename T>
MFEM_HOST_DEVICE zero chain_rule(const zero /* df_dx */,
                                 const T /* dx */)
{
   return zero{};
}

/**
 * @overload
 * @note this overload implements a no-op for the case where the small change is identically zero
 */
template <typename T>
MFEM_HOST_DEVICE zero chain_rule(const T /* df_dx */,
                                 const zero /* dx */)
{
   return zero{};
}

/**
 * @overload
 * @note for a scalar-valued function of a scalar, the chain rule is just multiplication
 */
inline MFEM_HOST_DEVICE real_t chain_rule(const real_t df_dx,
                                          const real_t dx) { return df_dx * dx; }

/**
 * @overload
 * @note for a tensor-valued function of a scalar, the chain rule is just scalar multiplication
 */
template <int... n>
MFEM_HOST_DEVICE auto chain_rule(const tensor<real_t, n...>& df_dx,
                                 const real_t dx) ->
decltype(df_dx * dx)
{
   return df_dx * dx;
}

template <int n> struct always_false : std::false_type { };

template <typename T, int... n> struct isotropic_tensor;

template <typename T, int n>
struct isotropic_tensor<T, n>
{
   static_assert(always_false<n> {},
                 "error: there is no such thing as a rank-1 isotropic tensor!");
};

// rank-2 isotropic tensors are just identity matrices
template <typename T, int m>
struct isotropic_tensor<T, m, m>
{
   MFEM_HOST_DEVICE constexpr T operator()(int i, int j) const
   {
      return (i == j) * value;
   }
   T value;
};

template <int m>
MFEM_HOST_DEVICE constexpr isotropic_tensor<real_t, m, m> IsotropicIdentity()
{
   return isotropic_tensor<real_t, m, m> {1.0};
}

template <typename S, typename T, int m> MFEM_HOST_DEVICE constexpr
auto operator*(S scale,
               const isotropic_tensor<T, m, m> & I)
-> isotropic_tensor<decltype(S {} * T{}), m, m>
{
   return {I.value * scale};
}

template <typename S, typename T, int m> MFEM_HOST_DEVICE constexpr
auto operator*(const isotropic_tensor<T, m, m> & I,
               const S scale)
-> isotropic_tensor<decltype(S {}, T{}), m, m>
{
   return {I.value * scale};
}

template <typename S, typename T, int m> MFEM_HOST_DEVICE constexpr
auto operator+(const isotropic_tensor<S, m, m>& I1,
               const isotropic_tensor<T, m, m>& I2)
-> isotropic_tensor<decltype(S {} + T{}), m, m>
{
   return {I1.value + I2.value};
}

template <typename S, typename T, int m> MFEM_HOST_DEVICE constexpr
auto operator-(const isotropic_tensor<S, m, m>& I1,
               const isotropic_tensor<T, m, m>& I2)
-> isotropic_tensor<decltype(S {} - T{}), m, m>
{
   return {I1.value - I2.value};
}

template <typename S, typename T, int m> MFEM_HOST_DEVICE //constexpr
auto operator+(const isotropic_tensor<S, m, m>& I,
               const tensor<T, m, m>& A)
-> tensor<decltype(S {} + T{}), m, m>
{
   tensor<decltype(S{} + T{}), m, m> output{};
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < m; j++)
      {
         output[i][j] = I.value * (i == j) + A[i][j];
      }
   }
   return output;
}

template <typename S, typename T, int m> MFEM_HOST_DEVICE //constexpr
auto operator+(const tensor<S, m, m>& A,
               const isotropic_tensor<T, m, m>& I)
-> tensor<decltype(S {} + T{}), m, m>
{
   tensor<decltype(S{} + T{}), m, m> output{};
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < m; j++)
      {
         output[i][j] = A[i][j] + I.value * (i == j);
      }
   }
   return output;
}

template <typename S, typename T, int m> MFEM_HOST_DEVICE //constexpr
auto operator-(const isotropic_tensor<S, m, m>& I,
               const tensor<T, m, m>& A)
-> tensor<decltype(S {} - T{}), m, m>
{
   tensor<decltype(S{} - T{}), m, m> output{};
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < m; j++)
      {
         output[i][j] = I.value * (i == j) - A[i][j];
      }
   }
   return output;
}

template <typename S, typename T, int m> MFEM_HOST_DEVICE // constexpr
auto operator-(const tensor<S, m, m>& A,
               const isotropic_tensor<T, m, m>& I)
-> tensor<decltype(S {} - T{}), m, m>
{
   tensor<decltype(S{} - T{}), m, m> output{};
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < m; j++)
      {
         output[i][j] = A[i][j] - I.value * (i == j);
      }
   }
   return output;
}

template <typename S, typename T, int m, int... n> MFEM_HOST_DEVICE constexpr
auto dot(const isotropic_tensor<S, m, m>& I,
         const tensor<T, m, n...>& A)
-> tensor<decltype(S {} * T{}), m, n...>
{
   return I.value * A;
}

template <typename S, typename T, int m, int... n> MFEM_HOST_DEVICE //constexpr
auto dot(const tensor<S, n...>& A,
         const isotropic_tensor<T, m, m> & I)
-> tensor<decltype(S {} * T{}), n...>
{
   constexpr int dimensions[sizeof...(n)] = {n...};
   static_assert(dimensions[sizeof...(n) - 1] == m, "n-1 != m");
   return A * I.value;
}

template <typename S, typename T, int m, int... n> MFEM_HOST_DEVICE constexpr
auto ddot(const isotropic_tensor<S, m, m>& I,
          const tensor<T, m, m>& A)
-> decltype(S {} * T{})
{
   return I.value * tr(A);
}

template <typename T, int m> MFEM_HOST_DEVICE constexpr
auto sym(const isotropic_tensor<T, m, m>& I) -> isotropic_tensor<T, m, m>
{
   return I;
}

template <typename T, int m> MFEM_HOST_DEVICE constexpr
auto antisym(const isotropic_tensor<T, m, m>&) -> zero
{
   return zero{};
}

template <typename T, int m> MFEM_HOST_DEVICE constexpr
auto tr(const isotropic_tensor<T, m, m>& I) -> decltype(T {} * m)
{
   return I.value * m;
}

template <typename T, int m> MFEM_HOST_DEVICE constexpr
auto transpose(const isotropic_tensor<T, m, m>& I) -> isotropic_tensor<T, m, m>
{
   return I;
}

template <typename T, int m> MFEM_HOST_DEVICE constexpr
auto det(const isotropic_tensor<T, m, m>& I) -> T
{
   return std::pow(I.value, m);
}

template <typename T, int m> MFEM_HOST_DEVICE constexpr
auto norm(const isotropic_tensor<T, m, m>& I) -> T
{
   return sqrt(I.value * I.value * m);
}

template <typename T, int m> MFEM_HOST_DEVICE constexpr
auto sqnorm(const isotropic_tensor<T, m, m>& I) -> T
{
   return I.value * I.value * m;
}

// rank-3 isotropic tensors are just the alternating symbol
template <typename T>
struct isotropic_tensor<T, 3, 3, 3>
{
   MFEM_HOST_DEVICE constexpr T operator()(int i, int j, int k) const
   {
      return 0.5 * (i - j) * (j - k) * (k - i) * value;
   }
   T value;
};

// there are 3 linearly-independent rank-4 isotropic tensors,
// so the general one will be some linear combination of them
template <typename T, int m>
struct isotropic_tensor<T, m, m, m, m>
{
   T c1, c2, c3;

   MFEM_HOST_DEVICE constexpr T operator()(int i, int j, int k, int l) const
   {
      return c1 * (i == j) * (k == l)
             + c2 * ((i == k) * (j == l) + (i == l) * (j == k)) * 0.5
             + c3 * ((i == k) * (j == l) - (i == l) * (j == k)) * 0.5;
   }
};

template <int m> MFEM_HOST_DEVICE constexpr
auto SymmetricIdentity() -> isotropic_tensor<real_t, m, m, m, m>
{
   return {0.0, 1.0, 0.0};
}

template <int m>MFEM_HOST_DEVICE constexpr
auto AntisymmetricIdentity() -> isotropic_tensor<real_t, m, m, m, m>
{
   return {0.0, 0.0, 1.0};
}

template <typename S, typename T, int m> MFEM_HOST_DEVICE constexpr
auto operator*(S scale,
               isotropic_tensor<T, m, m, m, m> I)
-> isotropic_tensor<decltype(S {} * T{}), m, m, m, m>
{
   return {I.c1 * scale, I.c2 * scale, I.c3 * scale};
}

template <typename S, typename T, int m> MFEM_HOST_DEVICE constexpr
auto operator*(isotropic_tensor<S, m, m, m, m> I,
               T scale)
-> isotropic_tensor<decltype(S {} * T{}), m, m, m, m>
{
   return {I.c1 * scale, I.c2 * scale, I.c3 * scale};
}

template <typename S, typename T, int m> MFEM_HOST_DEVICE constexpr
auto operator+(isotropic_tensor<S, m, m, m, m> I1,
               isotropic_tensor<T, m, m, m, m> I2)
-> isotropic_tensor<decltype(S {} + T{}), m, m, m, m>
{
   return {I1.c1 + I2.c1, I1.c2 + I2.c2, I1.c3 + I2.c3};
}

template <typename S, typename T, int m> MFEM_HOST_DEVICE constexpr
auto operator-(isotropic_tensor<S, m, m, m, m> I1,
               isotropic_tensor<T, m, m, m, m> I2)
-> isotropic_tensor<decltype(S {} - T{}), m, m, m, m>
{
   return {I1.c1 - I2.c1, I1.c2 - I2.c2, I1.c3 - I2.c3};
}

template <typename S, typename T, int m, int... n> MFEM_HOST_DEVICE constexpr
auto ddot(const isotropic_tensor<S, m, m, m, m>& I,
          const tensor<T, m, m>& A)
-> tensor<decltype(S {} * T{}), m, m>
{
   return I.c1 * tr(A) * IdentityMatrix<m>() + I.c2 * sym(A) + I.c3 * antisym(A);
}

} // namespace future
} // namespace mfem
