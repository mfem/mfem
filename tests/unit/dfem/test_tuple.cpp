// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../unit_tests.hpp"
#include "mfem.hpp"
#ifndef MFEM_USE_MPI
#include "../../../fem/dfem/tuple.hpp"
#endif

using namespace mfem;
using namespace mfem::future;

namespace tuple_test
{

// A payload that is not a scalar, mimicking what dFEM kernels actually store.
using vec3 = tensor<real_t, 3>;
using tuple3 = tuple<real_t, int, vec3>;

// mfem::future::tuple is no longer an aggregate: it derives from tuple_leaf
// bases so that it can be defined for an arbitrary number of elements. These
// checks pin down the properties that the aggregate used to provide for free
// and that device kernels (which capture tuples by value) depend on.
static_assert(std::is_trivially_copyable<tuple3>::value,
              "tuple must be trivially copyable to be captured by value in device kernels");
static_assert(std::is_trivially_destructible<tuple3>::value,
              "tuple must be trivially destructible");
static_assert(std::is_trivially_default_constructible<tuple3>::value,
              "tuple must be trivially default constructible");
static_assert(std::is_trivially_copy_assignable<tuple3>::value,
              "tuple must be trivially copy assignable");
static_assert(sizeof(tuple3) == sizeof(real_t) + sizeof(int) + sizeof(vec3) +
              (alignof(real_t) - sizeof(int)),
              "tuple must not be larger than the sum of its (padded) members");

// Size and element types, both through mfem::future and through the std
// specializations that drive structured bindings.
static_assert(tuple_size<tuple3>::value == 3, "");
static_assert(std::tuple_size<tuple3>::value == 3, "");
static_assert(std::is_same<tuple_element<0, tuple3>::type, real_t>::value, "");
static_assert(std::is_same<tuple_element<1, tuple3>::type, int>::value, "");
static_assert(std::is_same<tuple_element<2, tuple3>::type, vec3>::value, "");
static_assert(std::is_same<std::tuple_element_t<0, tuple3>, real_t>::value, "");
static_assert(std::is_same<std::tuple_element_t<2, tuple3>, vec3>::value, "");

// get must preserve the value category and constness of its argument.
static_assert(std::is_same<decltype(get<1>(std::declval<tuple3&>())),
              int&>::value, "get on an lvalue must return an lvalue reference");
static_assert(std::is_same<decltype(get<1>(std::declval<const tuple3&>())),
              const int&>::value,
              "get on a const lvalue must return a const lvalue reference");
static_assert(std::is_same<decltype(get<1>(std::declval<tuple3&&>())),
              int&&>::value, "get on an rvalue must return an rvalue reference");
static_assert(std::is_same<decltype(get<1>(std::declval<const tuple3&&>())),
              const int&&>::value,
              "get on a const rvalue must return a const rvalue reference");

// += and -= must return a reference, not a copy of the whole tuple.
using tuple2 = tuple<real_t, vec3>;
static_assert(std::is_same<decltype(std::declval<tuple2&>() +=
                                       std::declval<const tuple2&>()), tuple2&>::value,
              "operator+= must return a reference");
static_assert(std::is_same<decltype(std::declval<tuple2&>() -=
                                       std::declval<const tuple2&>()), tuple2&>::value,
              "operator-= must return a reference");

// The element-wise constructor must stay implicit, so that the
// copy-list-initialization forms that worked with the aggregate keep working.
static_assert(std::is_convertible<int, tuple<int>>::value,
              "tuple's element-wise constructor must not be explicit");

// Constructing from an incompatible type must SFINAE out rather than hard-error,
// so that the constructor does not poison type traits.
struct not_a_number { };
static_assert(!std::is_constructible<tuple<int, int>, int, not_a_number>::value,
              "");
static_assert(!std::is_constructible<tuple<int, int>, int>::value,
              "arity mismatch must not be constructible");

// Usable at compile time.
constexpr tuple<int, real_t> const_tuple {2, 3.0};
static_assert(get<0>(const_tuple) == 2, "");

// Copy-list-initialization in a return statement (broken by an explicit ctor).
tuple<int, real_t> returns_braced_init_list() { return {7, 8.0}; }

} // namespace tuple_test

using namespace tuple_test;

TEST_CASE("dFEM tuple structured bindings", "[dFEM]")
{
   tuple3 t {1.0, 2, vec3{{3.0, 4.0, 5.0}}};

   SECTION("binding by reference writes through")
   {
      auto &[a, b, c] = t;
      a = 10.0;
      b = 20;
      c(0) = 30.0;
      REQUIRE(get<0>(t) == 10.0_r);
      REQUIRE(get<1>(t) == 20);
      REQUIRE(get<2>(t)(0) == 30.0_r);
   }

   SECTION("binding by value copies")
   {
      auto [a, b, c] = t;
      a = 10.0;
      b = 20;
      c(0) = 30.0;
      REQUIRE(get<0>(t) == 1.0_r);
      REQUIRE(get<1>(t) == 2);
      REQUIRE(get<2>(t)(0) == 3.0_r);
   }

   SECTION("binding to const")
   {
      const auto &[a, b, c] = t;
      REQUIRE(a == 1.0_r);
      REQUIRE(b == 2);
      REQUIRE(c(2) == 5.0_r);
      static_assert(std::is_same<decltype(a), const real_t>::value, "");
      static_assert(std::is_same<decltype(c), const vec3>::value, "");
   }

   SECTION("the bindings alias the tuple storage")
   {
      auto &[a, b, c] = t;
      REQUIRE(&a == &get<0>(t));
      REQUIRE(&b == &get<1>(t));
      REQUIRE(&c == &get<2>(t));
   }
}

TEST_CASE("dFEM tuple construction", "[dFEM]")
{
   SECTION("copy-list-initialization")
   {
      tuple<int, real_t> a = {1, 2.0};
      REQUIRE(get<0>(a) == 1);
      REQUIRE(get<1>(a) == 2.0_r);

      const auto b = returns_braced_init_list();
      REQUIRE(get<0>(b) == 7);
      REQUIRE(get<1>(b) == 8.0_r);
   }

   SECTION("direct initialization and CTAD")
   {
      tuple c {1, 2.0_r, vec3{{1.0, 2.0, 3.0}}};
      static_assert(std::is_same<decltype(c), tuple<int, real_t, vec3>>::value,
                    "CTAD must decay the arguments");
      REQUIRE(get<1>(c) == 2.0_r);
   }

   SECTION("make_tuple")
   {
      const auto d = make_tuple(1, 2.0_r);
      static_assert(std::is_same<decltype(d), const tuple<int, real_t>>::value, "");
      REQUIRE(get<0>(d) == 1);
   }

   SECTION("copy and move construction preserve values")
   {
      tuple3 t {1.0, 2, vec3{{3.0, 4.0, 5.0}}};
      tuple3 copy(t);
      tuple3 moved(std::move(t));
      REQUIRE(get<1>(copy) == 2);
      REQUIRE(get<2>(moved)(1) == 4.0_r);
   }

   SECTION("value initialization zeroes trivial members")
   {
      tuple<int, real_t> z {};
      REQUIRE(get<0>(z) == 0);
      REQUIRE(get<1>(z) == 0.0_r);
   }
}

TEST_CASE("dFEM tuple arithmetic", "[dFEM]")
{
   const tuple2 x {1.0, vec3{{1.0, 2.0, 3.0}}};
   const tuple2 y {2.0, vec3{{4.0, 5.0, 6.0}}};

   SECTION("element-wise binary operators")
   {
      const auto sum = x + y;
      REQUIRE(get<0>(sum) == 3.0_r);
      REQUIRE(get<1>(sum)(2) == 9.0_r);

      const auto diff = y - x;
      REQUIRE(get<0>(diff) == 1.0_r);
      REQUIRE(get<1>(diff)(0) == 3.0_r);
   }

   SECTION("compound assignment mutates in place and returns a reference")
   {
      tuple2 z = x;
      auto &ref = (z += y);
      REQUIRE(&ref == &z);
      REQUIRE(get<0>(z) == 3.0_r);
      REQUIRE(get<1>(z)(1) == 7.0_r);

      auto &ref2 = (z -= y);
      REQUIRE(&ref2 == &z);
      REQUIRE(get<0>(z) == 1.0_r);
      REQUIRE(get<1>(z)(1) == 2.0_r);
   }

   SECTION("scalar operators and unary minus")
   {
      const auto scaled = 2.0_r * x;
      REQUIRE(get<0>(scaled) == 2.0_r);
      REQUIRE(get<1>(scaled)(2) == 6.0_r);

      const auto halved = x / 2.0_r;
      REQUIRE(get<0>(halved) == 0.5_r);

      const auto negated = -x;
      REQUIRE(get<0>(negated) == -1.0_r);
      REQUIRE(get<1>(negated)(0) == -1.0_r);
   }

   SECTION("apply")
   {
      const auto s = apply([](const real_t &a, const vec3 &b) { return a + b(0); },
      x);
      REQUIRE(s == 2.0_r);
   }
}

// The tuples are captured by value in device kernels, so exercise a round trip
// through device memory: construct, mutate through structured bindings and read
// back on the device.
TEST_CASE("dFEM tuple on device", "[dFEM][GPU]")
{
   Vector res(4);
   auto d_res = res.Write();

   forall(1, [=] MFEM_HOST_DEVICE (int)
   {
      tuple3 t {1.0, 2, vec3{{3.0, 4.0, 5.0}}};
      auto &[a, b, c] = t;
      a += static_cast<real_t>(b);
      c(0) = a;

      tuple2 u {get<0>(t), get<2>(t)};
      u += tuple2 {1.0, vec3{{1.0, 1.0, 1.0}}};

      d_res[0] = get<0>(u);
      d_res[1] = get<1>(u)(0);
      d_res[2] = get<1>(u)(1);
      d_res[3] = static_cast<real_t>(get<1>(t));

      tuple2 v1{0_r, vec3{0_r, 0_r, 0_r}};
      tuple2 v2{0_r, vec3{0_r, 0_r, 0_r}};
      [[maybe_unused]] auto v = v1 + v2;
   });

   res.HostRead();
   REQUIRE(std::as_const(res)(0) == 4.0_r);
   REQUIRE(std::as_const(res)(1) == 4.0_r);
   REQUIRE(std::as_const(res)(2) == 5.0_r);
   REQUIRE(std::as_const(res)(3) == 2.0_r);
}
