// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mfem.hpp"
using namespace mfem;

#include "unit_tests.hpp"

//You typically want to start by testing things one object at a time.
TEST_CASE("IntegrationRules of Different Types", "[IntegrationRules]")
{
   //This code is automatically re-executed for all of the sections.
   IntegrationRules gauss_intrules(0, Quadrature1D::GaussLegendre);
   IntegrationRules lobatto_intrules(0, Quadrature1D::GaussLobatto);
   IntegrationRules oes_intrules(0, Quadrature1D::OpenUniform);
   IntegrationRules ces_intrules(0, Quadrature1D::ClosedUniform);



   //The tests will be reported in these sections.
   //Each REQUIRE counts as an assertion.
   //true = pass, false = fail
   SECTION("Expected Number of points for GaussLegendre exactness")
   {
      // polynomial degree we want to exactly integrate
      int exact = 4;
      // exact for 2*np - 1
      int pts_needed = 3;

      const IntegrationRule *ir;
      ir = &gauss_intrules.Get(Geometry::SEGMENT, exact);
      REQUIRE(ir->Size() == pts_needed);
   }
   SECTION("Expected Number of points for GaussLobatto exactness")
   {
      // polynomial degree we want to exactly integrate
      int exact = 4;
      // exact for 2*np - 3
      int pts_needed = 4;

      const IntegrationRule *ir;
      ir = &lobatto_intrules.Get(Geometry::SEGMENT, exact);
      REQUIRE(ir->Size() == pts_needed);
   }
}
