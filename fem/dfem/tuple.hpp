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
#pragma once

#include <ostream>
#include "../../config/config.hpp"
#include <utility>
#include <type_traits>
#include <tuple>

namespace mfem::future
{

// Forward declaration
template <typename... T>
struct tuple;

// Implementation detail: storage using multiple inheritance from tuple_leaf,
// which lets the tuple be defined for an arbitrary number of elements.
// Structured bindings come from the std::tuple_size / std::tuple_element / get
// specializations at the bottom of this file, not from the layout.
namespace detail
{
/**
 * @brief Trait that is true when @a U is a single argument that is (a reference
 * to) @a Self
 *
 * A variadic constructor taking @c "U&&..." is a better match than the copy
 * constructor for a non-const lvalue of its own type; this is used to constrain
 * it out of those overload sets.
 */
template <typename Self, typename... U>
struct is_self_arg : std::false_type {};

/// @overload
template <typename Self, typename U>
struct is_self_arg<Self, U> : std::is_same<Self, std::decay_t<U>> {};

/// SFINAE guard enabling a constructor for every @a U except @a Self itself
template <typename Self, typename... U>
using disable_if_self_t = std::enable_if_t<!is_self_arg<Self, U...>::value>;

/**
 * @brief A single tuple element storage
 * @tparam I The index of this element in the tuple
 * @tparam T The type stored in this element
 */
template <size_t I, typename T>
struct tuple_leaf
{
   T value;  ///< The stored value

   /// Default constructor
   MFEM_HOST_DEVICE constexpr tuple_leaf() = default;

   /// Construct from value
   template <typename U, typename = disable_if_self_t<tuple_leaf, U>>
   MFEM_HOST_DEVICE constexpr explicit tuple_leaf(U&& v) :
      value(std::forward<U>(v)) {}
};

/**
 * @brief Implementation of tuple storage using multiple inheritance
 * @tparam Indices Index sequence for tuple elements
 * @tparam T The types stored in the tuple
 *
 * This uses multiple inheritance from tuple_leaf base classes so that a single
 * definition covers any number of elements, while keeping the storage layout
 * (and the trivial copyability that device kernels rely on) of a plain struct.
 */
template <typename Indices, typename... T>
struct tuple_impl;

/// Specialization that inherits from all tuple_leaf instances
template <size_t... I, typename... T>
struct tuple_impl<std::index_sequence<I...>, T...> : tuple_leaf<I, T>...
{
   /// Default constructor
   MFEM_HOST_DEVICE constexpr tuple_impl() = default;

   /**
    * @brief Construct from values
    * @param args The values to store in the tuple
    *
    * @note the arguments are perfectly forwarded, so that constructing a tuple
    * from lvalues costs exactly one copy per element (taking them by value
    * would add a copy plus a move).
    */
   template <typename... U, typename = disable_if_self_t<tuple_impl, U...>>
   MFEM_HOST_DEVICE
   constexpr explicit tuple_impl(U&&... args)
      : tuple_leaf<I, T>(std::forward<U>(args))... {}
   };

/**
 * @brief Element-wise constructibility check, only instantiated once the
 * argument list is known to have the right length
 * @tparam Viable whether the arity and self-argument checks have passed
 * @tparam Tuple the @p tuple being constructed
 * @tparam U the constructor argument types
 */
template <bool Viable, typename Tuple, typename... U>
struct is_constructible_from : std::false_type {};

/// @overload
template <typename... T, typename... U>
struct is_constructible_from<true, tuple<T...>, U...>
: std::bool_constant<(std::is_constructible_v<T, U&&> && ...)> {};

/**
 * @brief Trait that is true when @a Tuple can be constructed element-wise from
 * the argument list @a U
 * @tparam Tuple the @p tuple being constructed
 * @tparam U the constructor argument types
 */
template <typename Tuple, typename... U>
struct is_elementwise_constructible : std::false_type {};

/// @overload
template <typename... T, typename... U>
struct is_elementwise_constructible<tuple<T...>, U...>
   : is_constructible_from<sizeof...(U) == sizeof...(T) && sizeof...(U) != 0 &&
  !is_self_arg<tuple<T...>, U...>::value, tuple<T...>, U...> {};

/// SFINAE guard for the element-wise constructor of @p tuple
template <typename Tuple, typename... U>
using enable_elementwise_t =
   std::enable_if_t<is_elementwise_constructible<Tuple, U...>::value>;
}

/**
 * @tparam T the types stored in the tuple
 * @brief This is a class that mimics most of std::tuple's interface,
 * except that it is usable in CUDA kernels and admits some arithmetic operator overloads.
 *
 * see https://en.cppreference.com/w/cpp/utility/tuple for more information about std::tuple
 */
template <typename... T>
struct tuple : detail::tuple_impl<std::index_sequence_for<T...>, T...>
{
   using base_type = detail::tuple_impl<std::index_sequence_for<T...>, T...>;

   /// Default constructor
   MFEM_HOST_DEVICE
   constexpr tuple() = default;

   /**
    * @brief Construct tuple from values
    * @param args The values to store
    *
    * @note this constructor is deliberately *not* explicit, so that the
    * copy-list-initialization forms that worked when @p tuple was an aggregate
    * (@c "tuple<A,B> t = {a,b};", @c "return {a,b};", passing @c "{a,b}" to a
    * function) keep working.
    */
   template <typename... U, typename = detail::enable_elementwise_t<tuple, U...>>
   MFEM_HOST_DEVICE
   constexpr tuple(U&&... args) : base_type(std::forward<U>(args)...) {}

   /// Copy constructor
   MFEM_HOST_DEVICE
   constexpr tuple(const tuple&) = default;

   /// Move constructor
   MFEM_HOST_DEVICE
   constexpr tuple(tuple&&) = default;

   /// Copy assignment operator
   MFEM_HOST_DEVICE
   constexpr tuple& operator=(const tuple&) = default;

   /// Move assignment operator
   MFEM_HOST_DEVICE
   constexpr tuple& operator=(tuple&&) = default;
};

/**
 * @brief Specialization for empty tuple
 */
template <>
struct tuple<>
{
   /// Default constructor
   MFEM_HOST_DEVICE constexpr tuple() = default;
};

/**
 * @brief Class template argument deduction rule for tuples
 * @tparam T the variadic template parameter for tuple types
 */
template <typename... T>
MFEM_HOST_DEVICE
tuple(T...) -> tuple<T...>;

/**
 * @brief helper function for combining a list of values into a tuple
 * @tparam T types of the values to be tuple-d
 * @param args the actual values to be put into a tuple
 */
template <typename... T>
MFEM_HOST_DEVICE constexpr tuple<T...> make_tuple(const T&... args)
{
   return tuple<T...> {args...};
}

/**
 * @brief Get the size of a tuple type
 * @tparam Types the types in the tuple
 */
template <class... Types>
struct tuple_size;

template <class... Types>
struct tuple_size<tuple<Types...>> :
                                std::integral_constant<std::size_t, sizeof...(Types)>
{
};

/**
 * @brief a struct used to determine the type at index I of a tuple
 *
 * @note see: https://en.cppreference.com/w/cpp/utility/tuple/tuple_element
 *
 * @tparam I the index of the desired type
 * @tparam T a tuple of different types
 */
template <size_t I, class T>
struct tuple_element;

// recursive case
/// @overload
template <size_t I, class Head, class... Tail>
struct tuple_element<I, tuple<Head, Tail...>> : tuple_element<I - 1,
                                              tuple<Tail...>>
{
};

// base case
/// @overload
template <class Head, class... Tail>
struct tuple_element<0, tuple<Head, Tail...>>
{
   using type = Head;  ///< the type at the specified index
};

namespace detail
{
/**
 * @brief Get implementation for tuple_leaf - non-const lvalue reference
 * @tparam I the index of the tuple element
 * @tparam T the type of the tuple element
 * @param leaf the tuple_leaf containing the value
 * @return reference to the value
 *
 * @note @a T is deduced from the (unique) @p tuple_leaf base class of the
 * argument, so callers only have to supply the index @a I.
 */
template <size_t I, typename T>
MFEM_HOST_DEVICE constexpr T& get_impl(tuple_leaf<I, T>& leaf)
{
   return leaf.value;
}

/**
 * @brief Get implementation for tuple_leaf - const lvalue reference
 * @tparam I the index of the tuple element
 * @tparam T the type of the tuple element
 * @param leaf the tuple_leaf containing the value
 * @return const reference to the value
 */
template <size_t I, typename T>
MFEM_HOST_DEVICE constexpr const T& get_impl(const tuple_leaf<I, T>& leaf)
{
   return leaf.value;
}

/**
 * @brief Get implementation for tuple_leaf - non-const rvalue reference
 * @tparam I the index of the tuple element
 * @tparam T the type of the tuple element
 * @param leaf the tuple_leaf containing the value
 * @return rvalue reference to the value
 */
template <size_t I, typename T>
MFEM_HOST_DEVICE constexpr T&& get_impl(tuple_leaf<I, T>&& leaf)
{
   return static_cast<T&&>(leaf.value);
}

/**
 * @brief Get implementation for tuple_leaf - const rvalue reference
 * @tparam I the index of the tuple element
 * @tparam T the type of the tuple element
 * @param leaf the tuple_leaf containing the value
 * @return const rvalue reference to the value
 */
template <size_t I, typename T>
MFEM_HOST_DEVICE constexpr const T&& get_impl(const tuple_leaf<I, T>&& leaf)
{
   return static_cast<const T&&>(leaf.value);
}
}  // namespace detail

/**
 * @tparam I the tuple index to access
 * @tparam T the types stored in the tuple
 * @brief return a reference to the ith tuple entry
 * @param t the tuple to access
 */
template <size_t I, typename... T>
MFEM_HOST_DEVICE constexpr auto& get(tuple<T...>& t)
{
   static_assert(I < sizeof...(T), "Tuple index out of bounds");
   return detail::get_impl<I>(t);
}

/**
 * @tparam I the tuple index to access
 * @tparam T the types stored in the tuple
 * @brief return a const reference to the ith tuple entry
 * @param t the tuple to access
 */
template <size_t I, typename... T>
MFEM_HOST_DEVICE constexpr const auto& get(const tuple<T...>& t)
{
   static_assert(I < sizeof...(T), "Tuple index out of bounds");
   return detail::get_impl<I>(t);
}

/**
 * @tparam I the tuple index to access
 * @tparam T the types stored in the tuple
 * @brief return an rvalue reference to the ith tuple entry
 * @param t the tuple to access
 */
template <size_t I, typename... T>
MFEM_HOST_DEVICE constexpr auto&& get(tuple<T...>&& t)
{
   static_assert(I < sizeof...(T), "Tuple index out of bounds");
   return detail::get_impl<I>(std::move(t));
}

/**
 * @tparam I the tuple index to access
 * @tparam T the types stored in the tuple
 * @brief return a const rvalue reference to the ith tuple entry
 * @param t the tuple to access
 */
template <size_t I, typename... T>
MFEM_HOST_DEVICE constexpr const auto&& get(const tuple<T...>&& t)
{
   static_assert(I < sizeof...(T), "Tuple index out of bounds");
   return detail::get_impl<I>(std::move(t));
}

/**
 * @brief a function intended to be used for extracting the ith type from a tuple.
 *
 * @note type<i>(my_tuple) returns a value, whereas get<i>(my_tuple) returns a reference
 *
 * @tparam I the index of the tuple to query
 * @tparam T the types stored in the tuple
 * @param t the tuple of values
 * @return a copy of the ith entry of the input
 */
template <size_t I, typename... T>
MFEM_HOST_DEVICE constexpr auto type(const tuple<T...>& t)
{
   static_assert(I < sizeof...(T), "Tuple index out of bounds");
   return get<I>(t);
}

/**
 * @brief Helper for applying binary operations element-wise
 *
 * @tparam Op The binary operation type
 * @tparam S the types stored in the tuple x
 * @tparam T the types stored in the tuple y
 * @tparam I The integer sequence for indexing
 * @param x first tuple of values
 * @param y second tuple of values
 * @param op the binary operation to apply
 * @return tuple containing the result of applying op to each element pair
 */
template <typename Op, typename... S, typename... T, size_t... I>
MFEM_HOST_DEVICE constexpr auto apply_op_helper(
   const tuple<S...>& x,
   const tuple<T...>& y,
   Op op,
   std::index_sequence<I...>)
{
   return tuple{op(get<I>(x), get<I>(y))...};
}

/**
 * @tparam S the types stored in the tuple x
 * @tparam T the types stored in the tuple y
 * @param x a tuple of values
 * @param y a tuple of values
 * @brief return a tuple of values defined by elementwise sum of x and y
 */
template <typename... S, typename... T>
MFEM_HOST_DEVICE constexpr auto operator+(const tuple<S...>& x,
                                          const tuple<T...>& y)
{
   static_assert(sizeof...(S) == sizeof...(T), "tuples must have same size");
   return apply_op_helper(x, y, [](const auto& a, const auto& b) { return a + b; },
   std::make_index_sequence<sizeof...(S)> {});
}

/**
 * @tparam S the types stored in the tuple x
 * @tparam T the types stored in the tuple y
 * @param x a tuple of values
 * @param y a tuple of values
 * @brief return a tuple of values defined by elementwise difference of x and y
 */
template <typename... S, typename... T>
MFEM_HOST_DEVICE constexpr auto operator-(const tuple<S...>& x,
                                          const tuple<T...>& y)
{
   static_assert(sizeof...(S) == sizeof...(T), "tuples must have same size");
   return apply_op_helper(x, y, [](const auto& a, const auto& b) { return a - b; },
   std::make_index_sequence<sizeof...(S)> {});
}

/**
 * @tparam S the types stored in the tuple x
 * @tparam T the types stored in the tuple y
 * @param x a tuple of values
 * @param y a tuple of values
 * @brief return a tuple of values defined by elementwise multiplication of x and y
 */
template <typename... S, typename... T>
MFEM_HOST_DEVICE constexpr auto operator*(const tuple<S...>& x,
                                          const tuple<T...>& y)
{
   static_assert(sizeof...(S) == sizeof...(T), "tuples must have same size");
   return apply_op_helper(x, y, [](const auto& a, const auto& b) { return a * b; },
   std::make_index_sequence<sizeof...(S)> {});
}

/**
 * @tparam S the types stored in the tuple x
 * @tparam T the types stored in the tuple y
 * @param x a tuple of values
 * @param y a tuple of values
 * @brief return a tuple of values defined by elementwise division of x by y
 */
template <typename... S, typename... T>
MFEM_HOST_DEVICE constexpr auto operator/(const tuple<S...>& x,
                                          const tuple<T...>& y)
{
   static_assert(sizeof...(S) == sizeof...(T), "tuples must have same size");
   return apply_op_helper(x, y, [](const auto& a, const auto& b) { return a / b; },
   std::make_index_sequence<sizeof...(S)> {});
}

/**
 * @brief A helper function for the += operator of tuples
 *
 * @tparam T the types stored in the tuples x and y
 * @tparam I integer sequence used to index the tuples
 * @param x tuple of values to be incremented
 * @param y tuple of increment values
 */
template <typename... T, size_t... I>
MFEM_HOST_DEVICE constexpr void inplace_add_helper(
   tuple<T...>& x,
   const tuple<T...>& y,
   std::index_sequence<I...>)
{
   ((get<I>(x) += get<I>(y)), ...);
}

/**
 * @tparam T the types stored in the tuples x and y
 * @param x a tuple of values
 * @param y a tuple of values
 * @brief add values contained in y, to the tuple x
 */
template <typename... T>
MFEM_HOST_DEVICE constexpr tuple<T...>& operator+=(tuple<T...>& x,
                                                   const tuple<T...>& y)
{
   inplace_add_helper(x, y, std::make_index_sequence<sizeof...(T)> {});
   return x;
}

/**
 * @brief A helper function for the -= operator of tuples
 *
 * @tparam T the types stored in the tuples x and y
 * @tparam I integer sequence used to index the tuples
 * @param x tuple of values to be subtracted from
 * @param y tuple of values to subtract from x
 */
template <typename... T, size_t... I>
MFEM_HOST_DEVICE constexpr void inplace_sub_helper(
   tuple<T...>& x,
   const tuple<T...>& y,
   std::index_sequence<I...>)
{
   ((get<I>(x) -= get<I>(y)), ...);
}

/**
 * @tparam T the types stored in the tuples x and y
 * @param x a tuple of values
 * @param y a tuple of values
 * @brief subtract values contained in y from the tuple x
 */
template <typename... T>
MFEM_HOST_DEVICE constexpr tuple<T...>& operator-=(tuple<T...>& x,
                                                   const tuple<T...>& y)
{
   inplace_sub_helper(x, y, std::make_index_sequence<sizeof...(T)> {});
   return x;
}

/**
 * @brief A helper function for the unary - operator of tuples
 *
 * @tparam T the types stored in the tuple x
 * @tparam I The integer sequence for indexing
 * @param x tuple of values
 * @return the returned tuple with negated values
 */
template <typename... T, size_t... I>
MFEM_HOST_DEVICE constexpr auto unary_minus_helper(
   const tuple<T...>& x,
   std::index_sequence<I...>)
{
   return tuple{-get<I>(x)...};
}

/**
 * @tparam T the types stored in the tuple x
 * @param x a tuple of values
 * @brief return a tuple of values defined by applying the unary minus operator to each element of x
 */
template <typename... T>
MFEM_HOST_DEVICE constexpr auto operator-(const tuple<T...>& x)
{
   return unary_minus_helper(x, std::make_index_sequence<sizeof...(T)> {});
}

/**
 * @brief A helper function for the * operator of tuples with scalar
 *
 * @tparam T the types stored in the tuple x
 * @tparam I The integer sequence for indexing
 * @param a a constant multiplier
 * @param x tuple of values
 * @return the returned tuple product
 */
template <typename... T, size_t... I>
MFEM_HOST_DEVICE constexpr auto scalar_mult_helper(
   real_t a,
   const tuple<T...>& x,
   std::index_sequence<I...>)
{
   return tuple{a * get<I>(x)...};
}

/**
 * @tparam T the types stored in the tuple
 * @param a a scaling factor
 * @param x the tuple object
 * @brief multiply each component of x by the value a on the left
 */
template <typename... T>
MFEM_HOST_DEVICE constexpr auto operator*(real_t a, const tuple<T...>& x)
{
   return scalar_mult_helper(a, x, std::make_index_sequence<sizeof...(T)> {});
}

/**
 * @tparam T the types stored in the tuple
 * @param x the tuple object
 * @param a a scaling factor
 * @brief multiply each component of x by the value a on the right
 */
template <typename... T>
MFEM_HOST_DEVICE constexpr auto operator*(const tuple<T...>& x, real_t a)
{
   return a * x;
}

/**
 * @brief A helper function for the / operator of tuples with scalar denominator
 *
 * @tparam T the types stored in the tuple x
 * @tparam I The integer sequence for indexing
 * @param x tuple of values
 * @param a the constant denominator
 * @return the returned tuple ratio
 */
template <typename... T, size_t... I>
MFEM_HOST_DEVICE constexpr auto scalar_div_helper(
   const tuple<T...>& x,
   real_t a,
   std::index_sequence<I...>)
{
   return tuple{get<I>(x) / a...};
}

/**
 * @tparam T the types stored in the tuple x
 * @param x a tuple of numerator values
 * @param a a denominator
 * @brief return a tuple of values defined by elementwise division of x by a
 */
template <typename... T>
MFEM_HOST_DEVICE constexpr auto operator/(const tuple<T...>& x, real_t a)
{
   return scalar_div_helper(x, a, std::make_index_sequence<sizeof...(T)> {});
}

/**
 * @brief A helper function for the / operator with scalar numerator
 *
 * @tparam T the types stored in the tuple x
 * @tparam I The integer sequence for indexing
 * @param a the constant numerator
 * @param x tuple of values
 * @return the returned tuple ratio
 */
template <typename... T, size_t... I>
MFEM_HOST_DEVICE constexpr auto scalar_div_inv_helper(
   real_t a,
   const tuple<T...>& x,
   std::index_sequence<I...>)
{
   return tuple{a / get<I>(x)...};
}

/**
 * @tparam T the types stored in the tuple x
 * @param a the numerator
 * @param x a tuple of denominator values
 * @brief return a tuple of values defined by division of a by the elements of x
 */
template <typename... T>
MFEM_HOST_DEVICE constexpr auto operator/(real_t a, const tuple<T...>& x)
{
   return scalar_div_inv_helper(a, x, std::make_index_sequence<sizeof...(T)> {});
}

/**
 * @tparam T the types stored in the tuple
 * @tparam I a list of indices used to access each element of the tuple
 * @param out the ostream to write the output to
 * @param t the tuple of values
 * @brief helper used to implement printing a tuple of values
 */
template <typename... T, size_t... I>
auto& print_helper(std::ostream& out, const tuple<T...>& t,
                   std::index_sequence<I...>)
{
   out << "tuple{";
   (..., (out << (I == 0 ? "" : ", ") << get<I>(t)));
   out << "}";
   return out;
}

/**
 * @tparam T the types stored in the tuple
 * @param out the ostream to write the output to
 * @param t the tuple of values
 * @brief print a tuple of values
 */
template <typename... T>
auto& operator<<(std::ostream& out, const tuple<T...>& t)
{
   return print_helper(out, t, std::make_index_sequence<sizeof...(T)> {});
}

/**
 * @brief A helper to apply a lambda to a tuple
 *
 * @tparam F The functor type
 * @tparam T The tuple types
 * @tparam I The integer sequence for indexing
 * @param f The functor to apply to the tuple
 * @param args The input tuple
 * @return The functor output
 */
template <typename F, typename... T, size_t... I>
MFEM_HOST_DEVICE auto apply_helper(F&& f, tuple<T...>& args,
                                   std::index_sequence<I...>)
{
   return std::forward<F>(f)(get<I>(args)...);
}

/**
 * @tparam F a callable type
 * @tparam T the types of arguments to be passed in to f
 * @param f the callable object
 * @param args a tuple of arguments
 * @brief a way of passing an n-tuple to a function that expects n separate arguments
 *
 *   e.g. foo(bar, baz) is equivalent to apply(foo, mfem::tuple(bar,baz));
 */
template <typename F, typename... T>
MFEM_HOST_DEVICE auto apply(F&& f, tuple<T...>& args)
{
   return apply_helper(std::forward<F>(f), args,
                       std::make_index_sequence<sizeof...(T)> {});
}

/**
 * @overload
 */
template <typename F, typename... T, size_t... I>
MFEM_HOST_DEVICE auto apply_helper(F&& f, const tuple<T...>& args,
                                   std::index_sequence<I...>)
{
   return std::forward<F>(f)(get<I>(args)...);
}

/**
 * @tparam F a callable type
 * @tparam T the types of arguments to be passed in to f
 * @param f the callable object
 * @param args a tuple of arguments
 * @brief a way of passing an n-tuple to a function that expects n separate arguments
 *
 *   e.g. foo(bar, baz) is equivalent to apply(foo, mfem::tuple(bar,baz));
 */
template <typename F, typename... T>
MFEM_HOST_DEVICE auto apply(F&& f, const tuple<T...>& args)
{
   return apply_helper(std::forward<F>(f), args,
                       std::make_index_sequence<sizeof...(T)> {});
}

/**
 * @brief Trait for checking if a type is a @p mfem::tuple
 */
template <typename T>
struct is_tuple : std::false_type
{
};

/// @overload
template <typename... T>
struct is_tuple<tuple<T...>> : std::true_type
{
};

/**
 * @brief Trait for checking if a type if a @p mfem::tuple containing only @p mfem::tuple
 */
template <typename T>
struct is_tuple_of_tuples : std::false_type
{
};

/**
 * @brief Trait for checking if a type if a @p mfem::tuple containing only @p mfem::tuple
 */
template <typename... T>
struct is_tuple_of_tuples<tuple<T...>>
{
   static constexpr bool value = (is_tuple<T>::value &&
                                  ...);  ///< true/false result of type check
};

/** @brief Auxiliary template function that merges (concatenates) two
    mfem::future::tuple types into a single std::tuple that is empty, i.e. it is
    value initialized. */
template <typename... T1s, typename... T2s>
constexpr auto merge_mfem_tuples_as_empty_std_tuple(
   const tuple<T1s...>&,
   const tuple<T2s...>&)
{
   return std::tuple<T1s..., T2s...> {};
}

}  // namespace mfem::future

// Enable structured bindings for mfem::future::tuple
namespace std
{
/**
 * @brief Specialization of std::tuple_size for mfem::future::tuple
 * @tparam T The types in the mfem::future::tuple
 */
template <typename... T>
struct tuple_size<mfem::future::tuple<T...>>
                                          : integral_constant<size_t, sizeof...(T)> {};

/**
 * @brief Specialization of std::tuple_element for mfem::future::tuple
 * @tparam I The index of the element
 * @tparam T The types in the mfem::future::tuple
 */
template <size_t I, typename... T>
struct tuple_element<I, mfem::future::tuple<T...>>
{
   using type = typename
                mfem::future::tuple_element<I, mfem::future::tuple<T...>>::type;
};
}
