// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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
#include "unit_tests.hpp"

using namespace mfem;

void VerifyOrdering(NodalFiniteElement &el)
{
   int order = el.GetOrder();
   Geometry::Type geom = el.GetGeomType();
   const Array<int> &p = el.GetLexicographicOrdering();

   GeometryRefiner refiner;
   refiner.SetType(BasisType::GaussLobatto);
   RefinedGeometry *ref_geom = refiner.Refine(geom, order);

   double error = 0.0;

   for (int i=0; i<el.GetDof(); ++i)
   {
      int pi = (p.Size() > 0) ? p[i] : i;
      error += std::fabs(el.GetNodes()[pi].x - ref_geom->RefPts[i].x);
      error += std::fabs(el.GetNodes()[pi].y - ref_geom->RefPts[i].y);
      error += std::fabs(el.GetNodes()[pi].z - ref_geom->RefPts[i].z);
   }

   REQUIRE(error == MFEM_Approx(0.0));
}

template <typename T> void VerifyOrdering(int order)
{
   T el(order, BasisType::GaussLobatto);
   Geometry::Type geom = el.GetGeomType();
   INFO("order " << order << " " << Geometry::Name[geom]);
   VerifyOrdering(el);
}

TEST_CASE("Lexicographic Ordering", "[FiniteElement,Geometry]")
{
   auto order = GENERATE(1, 2, 3, 4, 5, 6);
   VerifyOrdering<H1_SegmentElement>(order);
   VerifyOrdering<H1_TriangleElement>(order);
   VerifyOrdering<H1_QuadrilateralElement>(order);
   VerifyOrdering<H1_TetrahedronElement>(order);
   VerifyOrdering<H1_HexahedronElement>(order);
   VerifyOrdering<H1_WedgeElement>(order);

   VerifyOrdering<L2_SegmentElement>(order);
   VerifyOrdering<L2_TriangleElement>(order);
   VerifyOrdering<L2_QuadrilateralElement>(order);
   VerifyOrdering<L2_TetrahedronElement>(order);
   VerifyOrdering<L2_HexahedronElement>(order);
   VerifyOrdering<L2_WedgeElement>(order);
}
