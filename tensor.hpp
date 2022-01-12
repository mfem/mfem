// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file tensor.hpp
 *
 * @brief Implementation of the tensor class used by Functional
 */

#pragma once

// #include "serac/infrastructure/accelerator.hpp"

// #include "dual.hpp"

// #include "detail/metaprogramming.hpp"

#include "mfem.hpp"
#include "dual.hpp"

namespace serac {

/// @cond
namespace detail {
template <typename T, typename i0_t>
SERAC_HOST_DEVICE constexpr auto get(const T& values, i0_t i0)
{
  return values[i0];
}

template <typename T, typename i0_t, typename i1_t>
SERAC_HOST_DEVICE constexpr auto get(const T& values, i0_t i0, i1_t i1)
{
  return values[i0][i1];
}

template <typename T, typename i0_t, typename i1_t, typename i2_t>
SERAC_HOST_DEVICE constexpr auto get(const T& values, i0_t i0, i1_t i1, i2_t i2)
{
  return values[i0][i1][i2];
}

template <typename T, typename i0_t, typename i1_t, typename i2_t, typename i3_t>
SERAC_HOST_DEVICE constexpr auto get(const T& values, i0_t i0, i1_t i1, i2_t i2, i3_t i3)
{
  return values[i0][i1][i2][i3];
}

template <typename T, typename i0_t>
SERAC_HOST_DEVICE constexpr auto& get(T& values, i0_t i0)
{
  return values[i0];
}

template <typename T, typename i0_t, typename i1_t>
SERAC_HOST_DEVICE constexpr auto& get(T& values, i0_t i0, i1_t i1)
{
  return values[i0][i1];
}

template <typename T, typename i0_t, typename i1_t, typename i2_t>
SERAC_HOST_DEVICE constexpr auto& get(T& values, i0_t i0, i1_t i1, i2_t i2)
{
  return values[i0][i1][i2];
}

template <typename T, typename i0_t, typename i1_t, typename i2_t, typename i3_t>
SERAC_HOST_DEVICE constexpr auto& get(T& values, i0_t i0, i1_t i1, i2_t i2, i3_t i3)
{
  return values[i0][i1][i2][i3];
}

template <int n>
using always_int = int;

}  // namespace detail
/// @endcond

/// @cond
template <typename T, int... n>
struct tensor;

template <typename T>
struct tensor<T> {
  using type                                  = T;
  static constexpr int              ndim      = 0;
  static constexpr int              first_dim = 1;
  SERAC_HOST_DEVICE constexpr auto& operator[](int) { return value; }
  SERAC_HOST_DEVICE constexpr auto  operator[](int) const { return value; }

  template <typename... S>
  SERAC_HOST_DEVICE constexpr auto& operator()(S...)
  {
    return value;
  }

  template <typename... S>
  SERAC_HOST_DEVICE constexpr auto operator()(S...) const
  {
    return value;
  }

  SERAC_HOST_DEVICE constexpr tensor() : value{} {}
  SERAC_HOST_DEVICE constexpr tensor(T v) : value(v) {}
  SERAC_HOST_DEVICE constexpr operator T() { return value; }
  T                           value;
};

template <typename T>
struct tensor<T, 1> {
  using type                     = T;
  static constexpr int ndim      = 1;
  static constexpr int first_dim = 1;

  template <typename S>
  SERAC_HOST_DEVICE constexpr auto& operator()(S)
  {
    return value;
  }

  template <typename S>
  SERAC_HOST_DEVICE constexpr auto operator()(S) const
  {
    return value;
  }

  SERAC_HOST_DEVICE constexpr auto& operator[](int) { return value; };
  SERAC_HOST_DEVICE constexpr auto  operator[](int) const { return value; };

  SERAC_HOST_DEVICE constexpr operator T() { return value; }
  SERAC_HOST_DEVICE constexpr tensor() : value() {}
  SERAC_HOST_DEVICE constexpr tensor(T v) : value(v) {}
  T value;
};

template <typename T>
struct tensor<T, 1, 1> {
  using type                     = T;
  static constexpr int ndim      = 2;
  static constexpr int first_dim = 1;

  template <typename S>
  SERAC_HOST_DEVICE constexpr auto& operator()(S, S)
  {
    return value;
  }

  template <typename S>
  SERAC_HOST_DEVICE constexpr auto operator()(S, S) const
  {
    return value;
  }

  SERAC_HOST_DEVICE constexpr auto& operator[](int) { return value; };
  SERAC_HOST_DEVICE constexpr auto  operator[](int) const { return value; };

  operator tensor<T, 1>() { return value; }
  tensor() : value() {}
  tensor(T v) : value(v) {}
  tensor(tensor<T, 1> v) : value(v) {}

  tensor<T, 1> value;
};

template <typename T, int n>
struct tensor<T, n> {
  using type                     = T;
  static constexpr int ndim      = 1;
  static constexpr int first_dim = n;

  template <typename S>
  SERAC_HOST_DEVICE constexpr auto& operator()(S i)
  {
    return detail::get(value, i);
  }

  template <typename S>
  SERAC_HOST_DEVICE constexpr auto operator()(S i) const
  {
    return detail::get(value, i);
  }

  SERAC_HOST_DEVICE constexpr auto& operator[](int i) { return value[i]; };
  SERAC_HOST_DEVICE constexpr auto  operator[](int i) const { return value[i]; };
  T                                 value[n];
};
/// @endcond

/**
 * @brief Arbitrary-rank tensor class
 * @tparam T The scalar type of the tensor
 * @tparam first The leading dimension of the tensor
 * @tparam last The parameter pack of the remaining dimensions
 */
template <typename T, int first, int... rest>
struct tensor<T, first, rest...> {
  /**
   * @brief The scalar type
   */
  using type = T;
  /**
   * @brief The rank of the tensor
   */
  static constexpr int ndim = 1 + sizeof...(rest);
  /**
   * @brief The array of dimensions containing the shape (not the data itself)
   * Similar to numpy.ndarray.shape
   */
  static constexpr int first_dim = first;

  /**
   * @brief Retrieves the sub-tensor corresponding to the indices provided in the pack @a i
   * @param[in] i The pack of indices
   */
  template <typename... S>
  SERAC_HOST_DEVICE constexpr auto& operator()(S... i)
  {
    // FIXME: Compile-time check for <= 4 indices??
    return detail::get(value, i...);
  };
  /// @overload
  template <typename... S>
  SERAC_HOST_DEVICE constexpr auto operator()(S... i) const
  {
    return detail::get(value, i...);
  };

  /**
   * @brief Retrieves the "row" of the tensor corresponding to index @a i
   * @param[in] i The index to retrieve a rank - 1 tensor from
   */
  SERAC_HOST_DEVICE constexpr auto& operator[](int i) { return value[i]; };
  /// @overload
  SERAC_HOST_DEVICE constexpr auto operator[](int i) const { return value[i]; };

  /**
   * @brief The actual tensor data
   */
  tensor<T, rest...> value[first];
};

/**
 * @brief class template argument deduction guide for type `tensor`.
 *
 * @note this lets users write
 * \code{.cpp} tensor A = {{0.0, 1.0, 2.0}}; \endcode
 * instead of explicitly writing the template parameters
 * \code{.cpp} tensor< double, 3 > A = {{1.0, 2.0, 3.0}}; \endcode
 */
template <typename T, int n1>
tensor(const T (&data)[n1]) -> tensor<T, n1>;

/**
 * @brief class template argument deduction guide for type `tensor`.
 *
 * @note this lets users write
 * \code{.cpp} tensor A = {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}}; \endcode
 * instead of explicitly writing the template parameters
 * \code{.cpp} tensor< double, 3, 3 > A = {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}}; \endcode
 */
template <typename T, int n1, int n2>
tensor(const T (&data)[n1][n2]) -> tensor<T, n1, n2>;

/**
 * @brief A sentinel struct for eliding no-op tensor operations
 */
struct zero {
  /** @brief `zero` is implicitly convertible to double with value 0.0 */
  SERAC_HOST_DEVICE operator double() { return 0.0; }

  /** @brief `zero` is implicitly convertible to a tensor of any shape */
  template <typename T, int... n>
  SERAC_HOST_DEVICE operator tensor<T, n...>()
  {
    return tensor<T, n...>{};
  }

  /** @brief `zero` can be accessed like a multidimensional array */
  template <typename... T>
  SERAC_HOST_DEVICE auto operator()(T...)
  {
    return zero{};
  }

  /** @brief anything assigned to `zero` does not change its value and returns `zero` */
  template <typename T>
  SERAC_HOST_DEVICE auto operator=(T)
  {
    return zero{};
  }
};

/** @brief checks if a type is `zero` */
template <typename T>
struct is_zero : std::false_type {
};

/** @overload */
template <>
struct is_zero<zero> : std::true_type {
};

/** @brief the sum of two `zero`s is `zero` */
SERAC_HOST_DEVICE constexpr auto operator+(zero, zero) { return zero{}; }

/** @brief the sum of `zero` with something non-`zero` just returns the other value */
template <typename T>
SERAC_HOST_DEVICE constexpr auto operator+(zero, T other)
{
  return other;
}

/** @brief the sum of `zero` with something non-`zero` just returns the other value */
template <typename T>
SERAC_HOST_DEVICE constexpr auto operator+(T other, zero)
{
  return other;
}

/////////////////////////////////////////////////

/** @brief the unary negation of `zero` is `zero` */
SERAC_HOST_DEVICE constexpr auto operator-(zero) { return zero{}; }

/** @brief the difference of two `zero`s is `zero` */
SERAC_HOST_DEVICE constexpr auto operator-(zero, zero) { return zero{}; }

/** @brief the difference of `zero` with something else is the unary negation of the other thing */
template <typename T>
SERAC_HOST_DEVICE constexpr auto operator-(zero, T other)
{
  return -other;
}

/** @brief the difference of something else with `zero` is the other thing itself */
template <typename T>
SERAC_HOST_DEVICE constexpr auto operator-(T other, zero)
{
  return other;
}

/////////////////////////////////////////////////

/** @brief the product of two `zero`s is `zero` */
SERAC_HOST_DEVICE constexpr auto operator*(zero, zero) { return zero{}; }

/** @brief the product `zero` with something else is also `zero` */
template <typename T>
SERAC_HOST_DEVICE constexpr auto operator*(zero, T /*other*/)
{
  return zero{};
}

/** @brief the product `zero` with something else is also `zero` */
template <typename T>
SERAC_HOST_DEVICE constexpr auto operator*(T /*other*/, zero)
{
  return zero{};
}

/** @brief let `zero` be accessed like a tuple */
template <int i>
zero& get(zero& x)
{
  return x;
}

/**
 * @brief Removes 1s from tensor dimensions
 * For example, a tensor<T, 1, 10> is equivalent to a tensor<T, 10>
 * @tparam T The scalar type of the tensor
 * @tparam n1 The first dimension
 * @tparam n2 The second dimension
 */
template <typename T, int n1, int n2 = 1>
using reduced_tensor = std::conditional_t<
    (n1 == 1 && n2 == 1), double,
    std::conditional_t<n1 == 1, tensor<T, n2>, std::conditional_t<n2 == 1, tensor<T, n1>, tensor<T, n1, n2>>>>;

/**
 * @brief Creates a tensor given the dimensions in a @p std::integer_sequence
 * @see std::integer_sequence
 * @tparam n The parameter pack of integer dimensions
 */
template <typename T, int... n>
SERAC_HOST_DEVICE constexpr auto tensor_with_shape(std::integer_sequence<int, n...>)
{
  return tensor<T, n...>{};
}

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
SERAC_SUPPRESS_NVCC_HOSTDEVICE_WARNING
template <typename lambda_type>
SERAC_HOST_DEVICE constexpr auto make_tensor(lambda_type f)
{
  using T = decltype(f());
  return tensor<T>{f()};
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
SERAC_SUPPRESS_NVCC_HOSTDEVICE_WARNING
template <int n1, typename lambda_type>
SERAC_HOST_DEVICE constexpr auto make_tensor(lambda_type f)
{
  using T = decltype(f(n1));
  tensor<T, n1> A{};
  for (int i = 0; i < n1; i++) {
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
SERAC_SUPPRESS_NVCC_HOSTDEVICE_WARNING
template <int n1, int n2, typename lambda_type>
SERAC_HOST_DEVICE constexpr auto make_tensor(lambda_type f)
{
  using T = decltype(f(n1, n2));
  tensor<T, n1, n2> A{};
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
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
SERAC_SUPPRESS_NVCC_HOSTDEVICE_WARNING
template <int n1, int n2, int n3, typename lambda_type>
SERAC_HOST_DEVICE constexpr auto make_tensor(lambda_type f)
{
  using T = decltype(f(n1, n2, n3));
  tensor<T, n1, n2, n3> A{};
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      for (int k = 0; k < n3; k++) {
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
SERAC_SUPPRESS_NVCC_HOSTDEVICE_WARNING
template <int n1, int n2, int n3, int n4, typename lambda_type>
SERAC_HOST_DEVICE constexpr auto make_tensor(lambda_type f)
{
  using T = decltype(f(n1, n2, n3, n4));
  tensor<T, n1, n2, n3, n4> A{};
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      for (int k = 0; k < n3; k++) {
        for (int l = 0; l < n4; l++) {
          A(i, j, k, l) = f(i, j, k, l);
        }
      }
    }
  }
  return A;
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
SERAC_HOST_DEVICE constexpr auto operator+(const tensor<S, n...>& A, const tensor<T, n...>& B)
{
  tensor<decltype(S{} + T{}), n...> C{};
  for (int i = 0; i < tensor<T, n...>::first_dim; i++) {
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
SERAC_HOST_DEVICE constexpr auto operator-(const tensor<T, n...>& A)
{
  tensor<T, n...> B{};
  for (int i = 0; i < tensor<T, n...>::first_dim; i++) {
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
SERAC_HOST_DEVICE constexpr auto operator-(const tensor<S, n...>& A, const tensor<T, n...>& B)
{
  tensor<decltype(S{} + T{}), n...> C{};
  for (int i = 0; i < tensor<T, n...>::first_dim; i++) {
    C[i] = A[i] - B[i];
  }
  return C;
}

/**
 * @brief multiply a tensor by a scalar value
 * @tparam S the scalar value type. Must be arithmetic (e.g. float, double, int) or a dual number
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam n integers describing the tensor shape
 * @param[in] scale The scaling factor
 * @param[in] A The tensor to be scaled
 */
template <typename S, typename T, int... n,
          typename = std::enable_if_t<std::is_arithmetic_v<S> || is_dual_number<S>::value>>
SERAC_HOST_DEVICE constexpr auto operator*(S scale, const tensor<T, n...>& A)
{
  tensor<decltype(S{} * T{}), n...> C{};
  for (int i = 0; i < tensor<T, n...>::first_dim; i++) {
    C[i] = scale * A[i];
  }
  return C;
}

/**
 * @brief multiply a tensor by a scalar value
 * @tparam S the scalar value type. Must be arithmetic (e.g. float, double, int) or a dual number
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam n integers describing the tensor shape
 * @param[in] A The tensor to be scaled
 * @param[in] scale The scaling factor
 */
template <typename S, typename T, int... n,
          typename = std::enable_if_t<std::is_arithmetic_v<S> || is_dual_number<S>::value>>
SERAC_HOST_DEVICE constexpr auto operator*(const tensor<T, n...>& A, S scale)
{
  tensor<decltype(T{} * S{}), n...> C{};
  for (int i = 0; i < tensor<T, n...>::first_dim; i++) {
    C[i] = A[i] * scale;
  }
  return C;
}

/**
 * @brief divide a scalar by each element in a tensor
 * @tparam S the scalar value type. Must be arithmetic (e.g. float, double, int) or a dual number
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam n integers describing the tensor shape
 * @param[in] scale The numerator
 * @param[in] A The tensor of denominators
 */
template <typename S, typename T, int... n,
          typename = std::enable_if_t<std::is_arithmetic_v<S> || is_dual_number<S>::value>>
constexpr auto operator/(S scale, const tensor<T, n...>& A)
{
  tensor<decltype(S{} * T{}), n...> C{};
  for (int i = 0; i < tensor<T, n...>::first_dim; i++) {
    C[i] = scale / A[i];
  }
  return C;
}

/**
 * @brief divide a tensor by a scalar
 * @tparam S the scalar value type. Must be arithmetic (e.g. float, double, int) or a dual number
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam n integers describing the tensor shape
 * @param[in] A The tensor of numerators
 * @param[in] scale The denominator
 */
template <typename S, typename T, int... n,
          typename = std::enable_if_t<std::is_arithmetic_v<S> || is_dual_number<S>::value>>
constexpr auto operator/(const tensor<T, n...>& A, S scale)
{
  tensor<decltype(T{} * S{}), n...> C{};
  for (int i = 0; i < tensor<T, n...>::first_dim; i++) {
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
template <typename S, typename T, int... n>
constexpr auto& operator+=(tensor<S, n...>& A, const tensor<T, n...>& B)
{
  for (int i = 0; i < tensor<S, n...>::first_dim; i++) {
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
template <typename T>
constexpr auto& operator+=(tensor<T>& A, const T& B)
{
  return A.value += B;
}

/**
 * @brief compound assignment (+) on tensors
 * @tparam T the underlying type of the tensor argument
 * @param[in] A The lefthand tensor
 * @param[in] B The righthand tensor
 */
template <typename T>
constexpr auto& operator+=(tensor<T, 1>& A, const T& B)
{
  return A.value += B;
}

/**
 * @brief compound assignment (+) on tensors
 * @tparam T the underlying type of the tensor argument
 * @param[in] A The lefthand tensor
 * @param[in] B The righthand tensor
 */
template <typename T>
constexpr auto& operator+=(tensor<T, 1, 1>& A, const T& B)
{
  return A.value += B;
}

/**
 * @brief compound assignment (+) between a tensor and zero (no-op)
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam n integers describing the tensor shape
 * @param[in] A The lefthand tensor
 */
template <typename T, int... n>
constexpr auto& operator+=(tensor<T, n...>& A, zero)
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
template <typename S, typename T, int... n>
constexpr auto& operator-=(tensor<S, n...>& A, const tensor<T, n...>& B)
{
  for (int i = 0; i < tensor<S, n...>::first_dim; i++) {
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
template <typename T, int... n>
constexpr auto& operator-=(tensor<T, n...>& A, zero)
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
template <typename S, typename T>
constexpr auto outer(S A, T B)
{
  static_assert(std::is_arithmetic_v<S> && std::is_arithmetic_v<T>,
                "outer product types must be tensor or arithmetic_type");
  return A * B;
}

/**
 * @overload
 * @note this overload implements the case where the left argument is a scalar, and the right argument is a tensor
 */
template <typename S, typename T, int n>
constexpr auto outer(S A, tensor<T, n> B)
{
  static_assert(std::is_arithmetic_v<S>, "outer product types must be tensor or arithmetic_type");
  tensor<decltype(S{} * T{}), n> AB{};
  for (int i = 0; i < n; i++) {
    AB[i] = A * B[i];
  }
  return AB;
}

/**
 * @overload
 * @note this overload implements the case where the left argument is a tensor, and the right argument is a scalar
 */
template <typename S, typename T, int m>
constexpr auto outer(const tensor<S, m>& A, T B)
{
  static_assert(std::is_arithmetic_v<T>, "outer product types must be tensor or arithmetic_type");
  tensor<decltype(S{} * T{}), m> AB{};
  for (int i = 0; i < m; i++) {
    AB[i] = A[i] * B;
  }
  return AB;
}

/**
 * @overload
 * @note this overload implements the case where the left argument is `zero`, and the right argument is a tensor
 */
template <typename T, int n>
constexpr auto outer(zero, const tensor<T, n>&)
{
  return zero{};
}

/**
 * @overload
 * @note this overload implements the case where the left argument is a tensor, and the right argument is `zero`
 */
template <typename T, int n>
constexpr auto outer(const tensor<T, n>&, zero)
{
  return zero{};
}

/**
 * @overload
 * @note this overload implements the case where the left argument is a tensor, and the right argument is `zero`
 */
template <typename S, typename T, int m, int n>
constexpr auto outer(S A, const tensor<T, m, n>& B)
{
  static_assert(std::is_arithmetic_v<S>, "outer product types must be tensor or arithmetic_type");
  tensor<decltype(S{} * T{}), m, n> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      AB[i][j] = A * B[i][j];
    }
  }
  return AB;
}

/**
 * @overload
 * @note this overload implements the case where both arguments are vectors
 */
template <typename S, typename T, int m, int n>
constexpr auto outer(const tensor<S, m>& A, const tensor<T, n>& B)
{
  tensor<decltype(S{} * T{}), m, n> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
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
template <typename S, typename T, int m, int n>
constexpr auto outer(const tensor<S, m, n>& A, T B)
{
  static_assert(std::is_arithmetic_v<T>, "outer product types must be tensor or arithmetic_type");
  tensor<decltype(S{} * T{}), m, n> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
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
template <typename S, typename T, int m, int n, int p>
constexpr auto outer(const tensor<S, m, n>& A, const tensor<T, p>& B)
{
  tensor<decltype(S{} * T{}), m, n, p> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < p; k++) {
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
template <typename S, typename T, int m, int n, int p>
constexpr auto outer(const tensor<S, m>& A, const tensor<T, n, p>& B)
{
  tensor<decltype(S{} * T{}), m, n, p> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < p; k++) {
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
template <typename S, typename T, int m, int n, int p, int q>
constexpr auto outer(const tensor<S, m, n>& A, const tensor<T, p, q>& B)
{
  tensor<decltype(S{} * T{}), m, n, p, q> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < p; k++) {
        for (int l = 0; l < q; l++) {
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
template <typename S, typename T, int m, int n>
constexpr auto inner(const tensor<S, m, n>& A, const tensor<T, m, n>& B)
{
  decltype(S{} * T{}) sum{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      sum += A[i][j] * B[i][j];
    }
  }
  return sum;
}

/**
 * @brief this function contracts over the "middle" index of the two tensor arguments
 * @tparam S the underlying type of the tensor (lefthand) argument
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam n integers describing the tensor shape
 * @param[in] A The lefthand tensor
 * @param[in] B The righthand tensor
 */
template <typename S, typename T, int m, int n, int p>
constexpr auto dot(const tensor<S, m, n>& A, const tensor<T, n, p>& B)
{
  tensor<decltype(S{} * T{}), m, p> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < p; j++) {
      for (int k = 0; k < n; k++) {
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
template <typename S, typename T, int m, int n>
constexpr auto dot(const tensor<S, m>& A, const tensor<T, m, n>& B)
{
  tensor<decltype(S{} * T{}), n> AB{};
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      AB[i] = AB[i] + A[j] * B[j][i];
    }
  }
  return AB;
}

/**
 * @overload
 * @note matrix . vector
 */
template <typename S, typename T, int m, int n>
constexpr auto dot(const tensor<S, m, n>& A, const tensor<T, n>& B)
{
  tensor<decltype(S{} * T{}), m> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      AB[i] = AB[i] + A[i][j] * B[j];
    }
  }
  return AB;
}

/**
 * @overload
 * @note 3rd-order-tensor . vector
 */
template <typename S, typename T, int m, int n, int p>
constexpr auto dot(const tensor<S, m, n, p>& A, const tensor<T, p>& B)
{
  tensor<decltype(S{} * T{}), m, n> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < p; k++) {
        AB[i][j] += A[i][j][k] * B[k];
      }
    }
  }
  return AB;
}

/**
 * @brief Dot product of a vector . vector and vector . tensor
 *
 * @tparam S the underlying type of the tensor (lefthand) argument
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam m the dimension of the first tensor
 * @tparam n the parameter pack of dimensions of the second tensor
 * @param A The lefthand tensor
 * @param B The righthand tensor
 * @return The computed dot product
 */
template <typename S, typename T, int m, int... n>
constexpr auto dot(const tensor<S, m>& A, const tensor<T, m, n...>& B)
{
  // this dot product function includes the vector * vector implementation and
  // the vector * tensor one, since clang emits an error about ambiguous
  // overloads if they are separate functions. The `if constexpr` expression avoids
  // using an `else` because that confuses nvcc (11.2) into thinking there's not
  // a return statement
  if constexpr (sizeof...(n) == 0) {
    decltype(S{} * T{}) AB{};
    for (int i = 0; i < m; i++) {
      AB += A[i] * B[i];
    }
    return AB;
  }

  if constexpr (sizeof...(n) > 0) {
    constexpr int                     dimensions[] = {n...};
    tensor<decltype(S{} * T{}), n...> AB{};
    for (int i = 0; i < dimensions[0]; i++) {
      for (int j = 0; j < m; j++) {
        AB[i] = AB[i] + A[j] * B[j][i];
      }
    }
    return AB;
  }
}

/**
 * @overload
 * @note vector . matrix . vector
 */
template <typename S, typename T, typename U, int m, int n>
constexpr auto dot(const tensor<S, m>& u, const tensor<T, m, n>& A, const tensor<U, n>& v)
{
  decltype(S{} * T{} * U{}) uAv{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      uAv += u[i] * A[i][j] * v[j];
    }
  }
  return uAv;
}

/**
 * @brief double dot product, contracting over the two "middle" indices
 * @tparam S the underlying type of the tensor (lefthand) argument
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam m first dimension of A
 * @tparam n second dimension of A
 * @tparam p third dimension of A, first dimensions of B
 * @tparam q fourth dimension of A, second dimensions of B
 * @param[in] A The lefthand tensor
 * @param[in] B The righthand tensor
 */
template <typename S, typename T, int m, int n, int p, int q>
constexpr auto ddot(const tensor<S, m, n, p, q>& A, const tensor<T, p, q>& B)
{
  tensor<decltype(S{} * T{}), m, n> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < p; k++) {
        for (int l = 0; l < q; l++) {
          AB[i][j] += A[i][j][k][l] * B[k][l];
        }
      }
    }
  }
  return AB;
}

/**
 * @overload
 * @note 3rd-order-tensor : 2nd-order-tensor
 */
template <typename S, typename T, int m, int n, int p>
constexpr auto ddot(const tensor<S, m, n, p>& A, const tensor<T, n, p>& B)
{
  tensor<decltype(S{} * T{}), m> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < p; k++) {
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
template <typename S, typename T, int m, int n>
constexpr auto ddot(const tensor<S, m, n>& A, const tensor<T, m, n>& B)
{
  decltype(S{} * T{}) AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      AB += A[i][j] * B[i][j];
    }
  }
  return AB;
}

/**
 * @brief this is a shorthand for dot(A, B)
 */
template <typename S, typename T, int... m, int... n>
constexpr auto operator*(const tensor<S, m...>& A, const tensor<T, n...>& B)
{
  return dot(A, B);
}

/**
 * @brief Returns the squared Frobenius norm of the tensor
 * @param[in] A The tensor to obtain the squared norm from
 */
template <typename T, int m>
constexpr auto sqnorm(const tensor<T, m>& A)
{
  T total{};
  for (int i = 0; i < m; i++) {
    total += A[i] * A[i];
  }
  return total;
}

/// @overload
template <typename T, int m, int n>
constexpr auto sqnorm(const tensor<T, m, n>& A)
{
  T total{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      total += A[i][j] * A[i][j];
    }
  }
  return total;
}

/// @overload
template <typename T, int... n>
constexpr auto sqnorm(const tensor<T, n...>& A)
{
  T total{};
  for_constexpr<n...>([&](auto... i) { total += A(i...) * A(i...); });
  return total;
}

/**
 * @brief Returns the Frobenius norm of the tensor
 * @param[in] A The tensor to obtain the norm from
 */
template <typename T, int... n>
auto norm(const tensor<T, n...>& A)
{
  using std::sqrt;
  return sqrt(sqnorm(A));
}

/**
 * @brief Normalizes the tensor
 * Each element is divided by the Frobenius norm of the tensor, @see norm
 * @param[in] A The tensor to normalize
 */
template <typename T, int... n>
auto normalize(const tensor<T, n...>& A)
{
  return A / norm(A);
}

/**
 * @brief Returns the trace of a square matrix
 * @param[in] A The matrix to compute the trace of
 * @return The sum of the elements on the main diagonal
 */
template <typename T, int n>
constexpr auto tr(const tensor<T, n, n>& A)
{
  T trA{};
  for (int i = 0; i < n; i++) {
    trA = trA + A[i][i];
  }
  return trA;
}

/**
 * @brief Returns the symmetric part of a square matrix
 * @param[in] A The matrix to obtain the symmetric part of
 * @return (1/2) * (A + A^T)
 */
template <typename T, int n>
constexpr auto sym(const tensor<T, n, n>& A)
{
  tensor<T, n, n> symA{};
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
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
template <typename T, int n>
constexpr auto dev(const tensor<T, n, n>& A)
{
  auto devA = A;
  auto trA  = tr(A);
  for (int i = 0; i < n; i++) {
    devA[i][i] -= trA / n;
  }
  return devA;
}

/**
 * @brief Obtains the identity matrix of the specified dimension
 * @return I_dim
 */
template <int dim>
SERAC_HOST_DEVICE constexpr tensor<double, dim, dim> Identity()
{
  tensor<double, dim, dim> I{};
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      I[i][j] = (i == j);
    }
  }
  return I;
}

/**
 * @brief Returns the transpose of the matrix
 * @param[in] A The matrix to obtain the transpose of
 */
template <typename T, int m, int n>
constexpr auto transpose(const tensor<T, m, n>& A)
{
  tensor<T, n, m> AT{};
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      AT[i][j] = A[j][i];
    }
  }
  return AT;
}

/**
 * @brief Returns the determinant of a matrix
 * @param[in] A The matrix to obtain the determinant of
 */
template <typename T>
constexpr auto det(const tensor<T, 2, 2>& A)
{
  return A[0][0] * A[1][1] - A[0][1] * A[1][0];
}
/// @overload
template <typename T>
constexpr auto det(const tensor<T, 3, 3>& A)
{
  return A[0][0] * A[1][1] * A[2][2] + A[0][1] * A[1][2] * A[2][0] + A[0][2] * A[1][0] * A[2][1] -
         A[0][0] * A[1][2] * A[2][1] - A[0][1] * A[1][0] * A[2][2] - A[0][2] * A[1][1] * A[2][0];
}

/**
 * @brief Return whether a square rank 2 tensor is symmetric
 *
 * @tparam n The height of the tensor
 * @param A The square rank 2 tensor
 * @param tolerance The tolerance to check for symmetry
 * @return Whether the square rank 2 tensor (matrix) is symmetric
 */
template <int n>
bool is_symmetric(tensor<double, n, n> A, double tolerance = 1.0e-8)
{
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      if (std::abs(A(i, j) - A(j, i)) > tolerance) {
        return false;
      };
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
bool is_symmetric_and_positive_definite(tensor<double, 2, 2> A)
{
  if (!is_symmetric(A)) {
    return false;
  }
  if (A(0, 0) < 0.0) {
    return false;
  }
  if (det(A) < 0.0) {
    return false;
  }
  return true;
}
/// @overload
bool is_symmetric_and_positive_definite(tensor<double, 3, 3> A)
{
  if (!is_symmetric(A)) {
    return false;
  }
  if (det(A) < 0.0) {
    return false;
  }
  auto subtensor = make_tensor<2, 2>([A](int i, int j) { return A(i, j); });
  if (!is_symmetric_and_positive_definite(subtensor)) {
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
template <typename T, int n>
constexpr tensor<T, n> linear_solve(tensor<T, n, n> A, const tensor<T, n> b)
{
  constexpr auto abs  = [](double x) { return (x < 0) ? -x : x; };
  constexpr auto swap = [](auto& x, auto& y) {
    auto tmp = x;
    x        = y;
    y        = tmp;
  };

  tensor<double, n> x{};

  for (int i = 0; i < n; i++) {
    // Search for maximum in this column
    double max_val = abs(A[i][i]);

    int max_row = i;
    for (int j = i + 1; j < n; j++) {
      if (abs(A[j][i]) > max_val) {
        max_val = abs(A[j][i]);
        max_row = j;
      }
    }

    swap(b[max_row], b[i]);
    swap(A[max_row], A[i]);

    // zero entries below in this column
    for (int j = i + 1; j < n; j++) {
      double c = -A[j][i] / A[i][i];
      A[j] += c * A[i];
      b[j] += c * b[i];
      A[j][i] = 0;
    }
  }

  // Solve equation Ax=b for an upper triangular matrix A
  for (int i = n - 1; i >= 0; i--) {
    x[i] = b[i] / A[i][i];
    for (int j = i - 1; j >= 0; j--) {
      b[j] -= A[j][i] * x[i];
    }
  }

  return x;
}

/**
 * @brief Inverts a matrix
 * @param[in] A The matrix to invert
 * @note Uses a shortcut for inverting a 2-by-2 matrix
 */
constexpr tensor<double, 2, 2> inv(const tensor<double, 2, 2>& A)
{
  double inv_detA(1.0 / det(A));

  tensor<double, 2, 2> invA{};

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
constexpr tensor<double, 3, 3> inv(const tensor<double, 3, 3>& A)
{
  double inv_detA(1.0 / det(A));

  tensor<double, 3, 3> invA{};

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
constexpr tensor<T, n, n> inv(const tensor<T, n, n>& A)
{
  constexpr auto abs  = [](double x) { return (x < 0) ? -x : x; };
  constexpr auto swap = [](auto& x, auto& y) {
    auto tmp = x;
    x        = y;
    y        = tmp;
  };

  tensor<double, n, n> B = Identity<n>();

  for (int i = 0; i < n; i++) {
    // Search for maximum in this column
    double max_val = abs(A[i][i]);

    int max_row = i;
    for (int j = i + 1; j < n; j++) {
      if (abs(A[j][i]) > max_val) {
        max_val = abs(A[j][i]);
        max_row = j;
      }
    }

    swap(B[max_row], B[i]);
    swap(A[max_row], A[i]);

    // zero entries below in this column
    for (int j = i + 1; j < n; j++) {
      if (A[j][i] != 0.0) {
        double c = -A[j][i] / A[i][i];
        A[j] += c * A[i];
        B[j] += c * B[i];
        A[j][i] = 0;
      }
    }
  }

  // upper triangular solve
  for (int i = n - 1; i >= 0; i--) {
    B[i] = B[i] / A[i][i];
    for (int j = i - 1; j >= 0; j--) {
      if (A[j][i] != 0.0) {
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
 * apply gauss elimination directly on the dual number types
 *
 * TODO: compare performance of this hardcoded implementation to just using inv() directly
 */
template <typename gradient_type, int n>
auto inv(tensor<dual<gradient_type>, n, n> A)
{
  auto invA = inv(get_value(A));
  return make_tensor<n, n>([&](int i, int j) {
    auto          value = invA[i][j];
    gradient_type gradient{};
    for (int k = 0; k < n; k++) {
      for (int l = 0; l < n; l++) {
        gradient -= invA[i][k] * A[k][l].gradient * invA[l][j];
      }
    }
    return dual<gradient_type>{value, gradient};
  });
}

/**
 * @brief recursively serialize the entries in a tensor to an ostream.
 * Output format uses braces and comma separators to mimic C syntax for multidimensional array
 * initialization.
 *
 * @param[in] out the std::ostream to write to (e.g. std::cout or std::ofstream)
 * @param[in] A The tensor to write out
 */
template <typename T, int... n>
auto& operator<<(std::ostream& out, const tensor<T, n...>& A)
{
  out << '{' << A[0];
  for (int i = 1; i < tensor<T, n...>::first_dim; i++) {
    out << ", " << A[i];
  }
  out << '}';
  return out;
}

/**
 * @brief replace all entries in a tensor satisfying |x| < 1.0e-10 by literal zero
 * @param[in] A The tensor to "chop"
 */
template <int n>
constexpr auto chop(const tensor<double, n>& A)
{
  auto copy = A;
  for (int i = 0; i < n; i++) {
    if (copy[i] * copy[i] < 1.0e-20) {
      copy[i] = 0.0;
    }
  }
  return copy;
}

/// @overload
template <int m, int n>
constexpr auto chop(const tensor<double, m, n>& A)
{
  auto copy = A;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (copy[i][j] * copy[i][j] < 1.0e-20) {
        copy[i][j] = 0.0;
      }
    }
  }
  return copy;
}

/**
 * @brief Constructs a tensor of dual numbers from a tensor of values
 * @param[in] A The tensor of values
 * @note a d-order tensor's gradient will be initialized to the (2*d)-order identity tensor
 */
template <int... n>
constexpr auto make_dual(const tensor<double, n...>& A)
{
  tensor<dual<tensor<double, n...>>, n...> A_dual{};
  for_constexpr<n...>([&](auto... i) {
    A_dual(i...).value          = A(i...);
    A_dual(i...).gradient(i...) = 1.0;
  });
  return A_dual;
}

/// @cond
namespace detail {

template <typename T1, typename T2>
struct outer_prod;

template <int... m, int... n>
struct outer_prod<tensor<double, m...>, tensor<double, n...>> {
  using type = tensor<double, m..., n...>;
};

template <int... n>
struct outer_prod<double, tensor<double, n...>> {
  using type = tensor<double, n...>;
};

template <int... n>
struct outer_prod<tensor<double, n...>, double> {
  using type = tensor<double, n...>;
};

template <>
struct outer_prod<double, double> {
  using type = tensor<double>;
};

template <typename T>
struct outer_prod<zero, T> {
  using type = zero;
};

template <typename T>
struct outer_prod<T, zero> {
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
 * @brief Retrieves a value tensor from a tensor of dual numbers
 * @param[in] arg The tensor of dual numbers
 */
template <typename T, int... n>
SERAC_HOST_DEVICE auto get_value(const tensor<dual<T>, n...>& arg)
{
  tensor<double, n...> value{};
  for_constexpr<n...>([&](auto... i) { value(i...) = arg(i...).value; });
  return value;
}

/**
 * @brief Retrieves the gradient component of a double (which is nothing)
 * @return The sentinel, @see zero
 */
SERAC_HOST_DEVICE auto get_gradient(double /* arg */) { return zero{}; }

/**
 * @brief Retrieves a gradient tensor from a tensor of dual numbers
 * @param[in] arg The tensor of dual numbers
 */
template <int... n>
SERAC_HOST_DEVICE auto get_gradient(const tensor<dual<double>, n...>& arg)
{
  tensor<double, n...> g{};
  for_constexpr<n...>([&](auto... i) { g(i...) = arg(i...).gradient; });
  return g;
}

/// @overload
template <int... n, int... m>
SERAC_HOST_DEVICE auto get_gradient(const tensor<dual<tensor<double, m...>>, n...>& arg)
{
  tensor<double, n..., m...> g{};
  for_constexpr<n...>([&](auto... i) { g(i...) = arg(i...).gradient; });
  return g;
}

/**
 * @brief evaluate the change (to first order) in a function, f, given a small change in the input argument, dx.
 */
SERAC_HOST_DEVICE constexpr auto chain_rule(const zero /* df_dx */, const zero /* dx */) { return zero{}; }

/**
 * @overload
 * @note this overload implements a no-op for the case where the gradient w.r.t. an input argument is identically zero
 */
template <typename T>
SERAC_HOST_DEVICE constexpr auto chain_rule(const zero /* df_dx */, const T /* dx */)
{
  return zero{};
}

/**
 * @overload
 * @note this overload implements a no-op for the case where the small change is indentically zero
 */
template <typename T>
SERAC_HOST_DEVICE constexpr auto chain_rule(const T /* df_dx */, const zero /* dx */)
{
  return zero{};
}

/**
 * @overload
 * @note for a scalar-valued function of a scalar, the chain rule is just multiplication
 */
SERAC_HOST_DEVICE constexpr auto chain_rule(const double df_dx, const double dx) { return df_dx * dx; }

/**
 * @overload
 * @note for a tensor-valued function of a scalar, the chain rule is just scalar multiplication
 */
template <int... n>
SERAC_HOST_DEVICE constexpr auto chain_rule(const tensor<double, n...>& df_dx, const double dx)
{
  return df_dx * dx;
}

/**
 * @overload
 * @note for a scalar-valued function of a tensor, the chain rule is the inner product
 */
template <int... n>
SERAC_HOST_DEVICE constexpr auto chain_rule(const tensor<double, n...>& df_dx, const tensor<double, n...>& dx)
{
  double total{};
  for_constexpr<n...>([&](auto... i) { total += df_dx(i...) * dx(i...); });
  return total;
}

/**
 * @overload
 * @note for a vector-valued function of a tensor, the chain rule contracts over all indices of dx
 */
template <int m, int... n>
SERAC_HOST_DEVICE constexpr auto chain_rule(const tensor<double, m, n...>& df_dx, const tensor<double, n...>& dx)
{
  tensor<double, m> total{};
  for (int i = 0; i < m; i++) {
    total[i] = chain_rule(df_dx[i], dx);
  }
  return total;
}

/**
 * @overload
 * @note for a matrix-valued function of a tensor, the chain rule contracts over all indices of dx
 */
template <int m, int n, int... p>
SERAC_HOST_DEVICE auto chain_rule(const tensor<double, m, n, p...>& df_dx, const tensor<double, p...>& dx)
{
  tensor<double, m, n> total{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      total[i][j] = chain_rule(df_dx[i][j], dx);
    }
  }
  return total;
}

/**
 * @brief Recast the shape of a tensor <m,n>
 */
template <int m, int n, typename T>
auto convert_to_tensor_with_shape(const tensor<T, m, n>& A)
{
  return A;
}

/**
 * @brief Recast a double as a tensor
 */
template <int m, int n, typename T>
auto convert_to_tensor_with_shape(T value)
{
  tensor<T, m, n> A;
  A[0][0] = value;
  return A;
}

/**
 * @brief Recast the shape of a tensor<m, n, o>
 */
template <int m, int n, int o, typename T>
auto convert_to_tensor_with_shape(const tensor<T, m, n, o>& A)
{
  return A;
}

/**
 * @brief Recast a double as a tensor
 */
template <int m, int n, int o, typename T>
auto convert_to_tensor_with_shape(T value)
{
  tensor<T, m, n, o> A;
  A[0][0][0] = value;
  return A;
}

}  // namespace serac