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

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

void CompareFE(const FiniteElement &fe1, const FiniteElement &fe2)
{
   REQUIRE(fe1.GetDim()               == fe2.GetDim());
   REQUIRE(fe1.GetRangeDim()          == fe2.GetRangeDim());
   REQUIRE(fe1.GetCurlDim()           == fe2.GetCurlDim());
   REQUIRE(fe1.GetGeomType()          == fe2.GetGeomType());
   REQUIRE(fe1.GetDof()               == fe2.GetDof());
   REQUIRE(fe1.GetOrder()             == fe2.GetOrder());
   REQUIRE(fe1.GetRangeType()         == fe2.GetRangeType());
   REQUIRE(fe1.GetDerivRangeType()    == fe2.GetDerivRangeType());
   REQUIRE(fe1.GetMapType()           == fe2.GetMapType());
   REQUIRE(fe1.GetDerivType()         == fe2.GetDerivType());
   REQUIRE(fe1.GetDerivMapType()      == fe2.GetDerivMapType());

   REQUIRE(fe1.HasAnisotropicOrders() == fe2.HasAnisotropicOrders());
   REQUIRE(fe1.Space()                == fe2.Space());

   // Get a uniform grid or integration points
   const int res = 4;
   RefinedGeometry* ref = GlobGeometryRefiner.Refine( fe1.GetGeomType(), res);
   const IntegrationRule& intRule = ref->RefPts;
   int npoints = intRule.GetNPoints();

   if (fe1.GetRangeType() == FiniteElement::RangeType::SCALAR)
   {
      Vector s1(fe1.GetDof());
      Vector s2(fe2.GetDof());

      for (int i=0; i < npoints; i++)
      {
         // Get the current integration point from intRule
         IntegrationPoint ip = intRule.IntPoint(i);

         CAPTURE(ip.x, ip.y, ip.z);

         fe1.CalcShape(ip, s1);
         fe2.CalcShape(ip, s2);

         s2 -= s1;

         REQUIRE(s2.Norml2() == MFEM_Approx(0.));
      }
   }
   if (fe1.GetRangeType() == FiniteElement::RangeType::VECTOR)
   {
      DenseMatrix s1(fe1.GetDof(), fe1.GetRangeDim());
      DenseMatrix s2(fe2.GetDof(), fe2.GetRangeDim());

      for (int i=0; i < npoints; i++)
      {
         // Get the current integration point from intRule
         IntegrationPoint ip = intRule.IntPoint(i);

         CAPTURE(ip.x, ip.y, ip.z);

         fe1.CalcVShape(ip, s1);
         fe2.CalcVShape(ip, s2);

         s2 -= s1;

         REQUIRE(s2.FNorm2() == MFEM_Approx(0.));
      }
   }
   if (fe1.GetDerivType() == FiniteElement::DerivType::GRAD)
   {
      DenseMatrix s1(fe1.GetDof(), fe1.GetDim());
      DenseMatrix s2(fe2.GetDof(), fe2.GetDim());

      for (int i=0; i < npoints; i++)
      {
         // Get the current integration point from intRule
         IntegrationPoint ip = intRule.IntPoint(i);

         CAPTURE(ip.x, ip.y, ip.z);

         fe1.CalcDShape(ip, s1);
         fe2.CalcDShape(ip, s2);

         s2 -= s1;

         REQUIRE(s2.FNorm2() == MFEM_Approx(0.));
      }
   }
   if (fe1.GetDerivType() == FiniteElement::DerivType::CURL)
   {
      DenseMatrix s1(fe1.GetDof(), fe1.GetCurlDim());
      DenseMatrix s2(fe2.GetDof(), fe2.GetCurlDim());

      for (int i=0; i < npoints; i++)
      {
         // Get the current integration point from intRule
         IntegrationPoint ip = intRule.IntPoint(i);

         CAPTURE(ip.x, ip.y, ip.z);

         fe1.CalcCurlShape(ip, s1);
         fe2.CalcCurlShape(ip, s2);

         s2 -= s1;

         REQUIRE(s2.FNorm2() == MFEM_Approx(0.));
      }
   }
   if (fe1.GetDerivType() == FiniteElement::DerivType::DIV)
   {
      Vector s1(fe1.GetDof());
      Vector s2(fe2.GetDof());

      for (int i=0; i < npoints; i++)
      {
         // Get the current integration point from intRule
         IntegrationPoint ip = intRule.IntPoint(i);

         CAPTURE(ip.x, ip.y, ip.z);

         fe1.CalcDivShape(ip, s1);
         fe2.CalcDivShape(ip, s2);

         s2 -= s1;

         REQUIRE(s2.Norml2() == MFEM_Approx(0.));
      }
   }
}

TEST_CASE("Fixed Order Finite Elements",
          "[LinearPyramidFiniteElement]"
          "[Nedelec1PyrFiniteElement]"
          // "[Nedelec2PyrFiniteElement]"
          "[RT0PyrFiniteElement]"
          "[P0PyrFiniteElement]")
{
   SECTION("H1 Order 1")
   {
      LinearPyramidFiniteElement fo;
      H1_FuentesPyramidElement ao(1);

      CompareFE(fo, ao);
   }
   SECTION("Nedelec Order 1")
   {
      Nedelec1PyrFiniteElement fo;
      ND_FuentesPyramidElement ao(1);

      CompareFE(fo, ao);
   }
   SECTION("Raviart-Thomas Order 0")
   {
      RT0PyrFiniteElement fo(false);
      RT_FuentesPyramidElement ao(0);

      CompareFE(fo, ao);
   }
   SECTION("L2 Order 0")
   {
      P0PyrFiniteElement fo;
      L2_FuentesPyramidElement ao(0);

      CompareFE(fo, ao);
   }
   /*
   /// The following comparison fails because these two sets of basis functions
   /// define the interior functions differently
   SECTION("Nedelec Order 2")
   {
     Nedelec2PyrFiniteElement fo;
     ND_FuentesPyramidElement ao(2);

     CompareFE(fo, ao);
   }
   */
}
