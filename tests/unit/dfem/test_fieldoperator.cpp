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

#include "../unit_tests.hpp"

#include "mfem.hpp"

#include "../../../fem/dfem/util.hpp"

#include <sstream>
#include <type_traits>

using namespace mfem;
using namespace mfem::future;

namespace
{

constexpr int U = 0;
constexpr int V = 1;

using derivative_outputs_t = decltype(
                                make_first_derivative_outputs<U,
                                Inputs<Value<U>, Curl<U>, Div<V>>>());

static_assert(is_curl_fop_v<Curl<U>>);
static_assert(!is_curl_fop_v<Div<U>>);
static_assert(is_div_fop_v<Div<U>>);
static_assert(!is_div_fop_v<Curl<U>>);
static_assert(Curl<U>::GetFieldId() == U);
static_assert(Div<V>::GetFieldId() == V);
static_assert(std::is_same_v<derivative_outputs_t,
              tuple<Value<U>, Curl<U>>>);

void CheckVectorFieldMetadata(int dim)
{
   Mesh mesh = dim == 2
               ? Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL)
               : Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON);

   H1_FECollection h1_fec(2, dim);
   ND_FECollection nd_fec(2, dim);
   RT_FECollection rt_fec(1, dim);
   FiniteElementSpace h1_fes(&mesh, &h1_fec);
   FiniteElementSpace nd_fes(&mesh, &nd_fec);
   FiniteElementSpace rt_fes(&mesh, &rt_fec);

   const FieldDescriptor h1_fd{U, &h1_fes};
   const FieldDescriptor nd_fd{U, &nd_fes};
   const FieldDescriptor rt_fd{U, &rt_fes};

   REQUIRE(IsCompatible<Entity::Element, Value<U>>(h1_fd));
   REQUIRE(IsCompatible<Entity::Element, Gradient<U>>(h1_fd));
   REQUIRE_FALSE(IsCompatible<Entity::Element, Curl<U>>(h1_fd));
   REQUIRE_FALSE(IsCompatible<Entity::Element, Div<U>>(h1_fd));

   REQUIRE(IsCompatible<Entity::Element, Value<U>>(nd_fd));
   REQUIRE(IsCompatible<Entity::Element, Curl<U>>(nd_fd));
   REQUIRE_FALSE(IsCompatible<Entity::Element, Gradient<U>>(nd_fd));
   REQUIRE_FALSE(IsCompatible<Entity::Element, Div<U>>(nd_fd));

   REQUIRE(IsCompatible<Entity::Element, Value<U>>(rt_fd));
   REQUIRE(IsCompatible<Entity::Element, Div<U>>(rt_fd));
   REQUIRE_FALSE(IsCompatible<Entity::Element, Gradient<U>>(rt_fd));
   REQUIRE_FALSE(IsCompatible<Entity::Element, Curl<U>>(rt_fd));

   REQUIRE(GetSizeOnQP<Entity::Element>(Value<U>{}, nd_fd) == dim);
   REQUIRE(GetSizeOnQP<Entity::Element>(Curl<U>{}, nd_fd) ==
           (dim == 2 ? 1 : 3));
   REQUIRE(GetSizeOnQP<Entity::Element>(Value<U>{}, rt_fd) == dim);
   REQUIRE(GetSizeOnQP<Entity::Element>(Div<U>{}, rt_fd) == 1);
}

} // namespace

TEST_CASE("dFEM Curl and Div field operators", "[dFEM][FieldOperator]")
{
   SECTION("2D metadata") { CheckVectorFieldMetadata(2); }
   SECTION("3D metadata") { CheckVectorFieldMetadata(3); }
}