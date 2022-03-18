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
 * @file tensor_isotropic.hpp
 *
 * @brief Implementation of the isotropic tensor class
 */

#include <type_traits> // for std::false_type

namespace mfem
{

namespace internal
{

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
MFEM_HOST_DEVICE constexpr isotropic_tensor<double, m, m> IsotropicIdentity()
{
   return isotropic_tensor<double, m, m> {1.0};
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
auto SymmetricIdentity() -> isotropic_tensor<double, m, m, m, m>
{
   return {0.0, 1.0, 0.0};
}

template <int m>MFEM_HOST_DEVICE constexpr
auto AntisymmetricIdentity() -> isotropic_tensor<double, m, m, m, m>
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
   return I.c1 * tr(A) * Identity<m>() + I.c2 * sym(A) + I.c3 * antisym(A);
}

} // namespace internal

} // namespace mfem
