// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "mfem.hpp"
using namespace mfem;

#include "catch.hpp"

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
