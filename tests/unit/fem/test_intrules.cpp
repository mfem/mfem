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
using namespace mfem;

#include "unit_tests.hpp"

// You typically want to start by testing things one object at a time.
TEST_CASE("Integration rule container with no refinement", "[IntegrationRules]")
{
   // This code is automatically re-executed for all of the sections.
   IntegrationRules my_intrules(0, Quadrature1D::GaussLegendre);
   IntegrationRule single_point_rule(1);
   const IntegrationRule *ir;

   // The tests will be reported in these sections.
   // Each REQUIRE counts as an assertion.
   // true = pass, false = fail
   SECTION("can set int rules in empty container")
   {
      single_point_rule[0].weight = 1.0;
      my_intrules.SetOwnRules(0);

      my_intrules.Set(Geometry::SEGMENT, 0, single_point_rule);
      ir = &my_intrules.Get(Geometry::SEGMENT, 0);

      REQUIRE(ir->Size() == 1);
      REQUIRE(ir->IntPoint(0).weight == 1.0);
   }

   SECTION("user set int rules really are owned by the user")
   {
      // Set up a custom int rule and put it in the container
      single_point_rule[0].weight = 1.0;
      my_intrules.SetOwnRules(0);
      my_intrules.Set(Geometry::SEGMENT, 0, single_point_rule);

      // Alter the int rule
      single_point_rule[0].weight = 2.0;

      // Test ownership by making sure that the int rule has changed
      ir = &my_intrules.Get(Geometry::SEGMENT, 0);
      REQUIRE(ir->IntPoint(0).weight == 2.0);
   }

   SECTION("point int rules 0, 1 accessible")
   {
      ir = &my_intrules.Get(Geometry::POINT, 0);
      REQUIRE(ir->Size() == 1);

      ir = &my_intrules.Get(Geometry::POINT, 1);
      REQUIRE(ir->Size() == 1);
   }

   // Can't really unit test these because it will crash due to
   // a null pointer dereference if it doesn't work.  This will
   // force the crash in the unit tests though.
   SECTION("resize of the SEGMENT int rule array")
   {
      ir = &my_intrules.Get(Geometry::SEGMENT, 100);
      REQUIRE(true);
   }

   SECTION("setting the integration point index works")
   {
      ir = &my_intrules.Get(Geometry::CUBE, 5);
      for (int i = 0; i < ir->Size(); i++)
      {
         REQUIRE(ir->IntPoint(i).index == i);
      }
   }

   SECTION("intrules up to order 16 accessible")
   {
      for (int order = 0; order <= 16; order ++)
      {
         // Do this in reverse the usual order to make sure that
         // the higher dimension cases are causing their constituent
         // lower dimension cases to lazily create properly.
         my_intrules.Get(Geometry::CUBE,        order);
         my_intrules.Get(Geometry::TETRAHEDRON, order);
         my_intrules.Get(Geometry::SQUARE,      order);
         my_intrules.Get(Geometry::TRIANGLE,    order);
         my_intrules.Get(Geometry::SEGMENT,     order);
      }
      REQUIRE(true);
   }
}


TEST_CASE("Integration rule order initialization", "[IntegrationRules]")
{
   constexpr int refined = 0;
   IntegrationRules intrules(refined, Quadrature1D::GaussLegendre);

   SECTION("Segment rule constructed by accessing square rule")
   {
      auto &quad5_ir = intrules.Get(Geometry::SQUARE, 5);
      REQUIRE(quad5_ir.GetOrder() >= 5);
      // The segment integration rule of order 5 is lazy constructed when we get
      // the square integration rule of order 5. Make sure its order was
      // properly set:
      auto &line5_ir = intrules.Get(Geometry::SEGMENT, 5);
      REQUIRE(line5_ir.GetOrder() >= 5);
   }

   SECTION("Segment rule constructed by accessing cube rule")
   {
      auto &hex7_ir = intrules.Get(Geometry::CUBE, 7);
      REQUIRE(hex7_ir.GetOrder() >= 7);
      // The segment integration rule of order 7 is lazy constructed when we get
      // the cube integration rule of order 7. Make sure its order was properly
      // set:
      auto &line7_ir = intrules.Get(Geometry::SEGMENT, 7);
      REQUIRE(line7_ir.GetOrder() >= 7);
   }

   SECTION("Segment and triangle rules constructed by accessing prism rule")
   {
      auto &prism3_ir = intrules.Get(Geometry::PRISM, 3);
      REQUIRE(prism3_ir.GetOrder() >= 3);
      // The segment integration rule of order 3 is lazy constructed when we get
      // the prism integration rule of order 3. Make sure its order was properly
      // set:
      auto &line3_ir = intrules.Get(Geometry::SEGMENT, 3);
      REQUIRE(line3_ir.GetOrder() >= 3);
      // The triangle integration rule of order 3 is lazy constructed when we
      // get the prism integration rule of order 3. Make sure its order was
      // properly set:
      auto &tri3_ir = intrules.Get(Geometry::TRIANGLE, 3);
      REQUIRE(tri3_ir.GetOrder() >= 3);
   }
}


TEST_CASE("Integration rule weights",
          "[IntegrationRules]")
{
   // This code is automatically re-executed for all of the sections.
   IntegrationRules my_intrules(0, Quadrature1D::GaussLegendre);
   const IntegrationRule *ir;

   auto geom  = GENERATE(Geometry::SEGMENT,
                         Geometry::TRIANGLE, Geometry::SQUARE,
                         Geometry::TETRAHEDRON, Geometry::CUBE,
                         Geometry::PRISM, Geometry::PYRAMID);
   auto order = GENERATE(1, 2, 3, 4, 5);

   CAPTURE(geom);
   CAPTURE(order);

   ir = &my_intrules.Get(geom, 2*order - 1);

   real_t weight_sum = 0.0;
   for (int j = 0; j < ir->GetNPoints(); j++)
   {
      const IntegrationPoint &ip = ir->IntPoint(j);
      weight_sum += ip.weight;
   }
   REQUIRE(Geometry::Volume[geom] == MFEM_Approx(weight_sum));
}

double poly2d(const IntegrationPoint &ip, int m, int n)
{
   return pow(ip.x, m)*pow(ip.y, n);
   // exact integral over the reference triangle is
   // m!n!/(m+n+2)! = 1/binom(m+n,m)/(m+n+1)/(m+n+2)
}

double apoly2d(const IntegrationPoint &ip, int i, int j, int k)
{
   return pow(1. - ip.x - ip.y, i)*pow(ip.x, j)*pow(ip.y, k);
   // exact integral over the reference triangle is (with p = i+j+k)
   // i!j!k!/(p+2)! = 1/binom(p,i+j)/binom(i+j,i)/(p+1)/(p+2)
}

double poly3d(const IntegrationPoint &ip, int l, int m, int n)
{
   return pow(ip.x, l)*pow(ip.y, m)*pow(ip.z, n);
   // exact integral over the reference tetrahedron is (with p = l+m+n)
   // l!m!n!/(p+3)! = 1/binom(p,l+m)/binom(l+m,l)/(p+1)/(p+2)/(p+3)
}

TEST_CASE("Simplex integration rules", "[SimplexRules]")
{
   // This code is automatically re-executed for all of the sections.
   IntegrationRules my_intrules(0, Quadrature1D::GaussLegendre);
   IntegrationRule single_point_rule(1);

   const int maxn = 32;
   int binom[maxn+1][maxn+1];
   for (int n = 0; n <= maxn; n++)
   {
      binom[n][0] = binom[n][n] = 1;
      for (int k = 1; k < n; k++)
      {
         binom[n][k] = binom[n-1][k] + binom[n-1][k-1];
      }
   }

   SECTION("low triangle integration error on reference element for f=x^m y^n, where m+n <= p")
   {
      for (int order = 0; order <= 25; order++)
      {
         const IntegrationRule &ir = IntRules.Get(Geometry::TRIANGLE, order);

         // using the monomial basis: x^m y^n, 0 <= m+n <= order
         for (int p = 0; p <= order; p++)
         {
            for (int m = p; m >= 0; m--)
            {
               int n = p - m;

               double integral = 0.0;
               for (int i = 0; i < ir.GetNPoints(); i++)
               {
                  const IntegrationPoint &ip = ir.IntPoint(i);
                  integral += ip.weight*poly2d(ip, m, n);
               }

               double exact = 1.0/binom[p][m]/(p + 1)/(p + 2);
               double relerr = 1. - integral/exact;

               // If a test fails any INFO statements preceding the REQUIRE are displayed
               INFO("p=" << p << ", m=" << m << ", n=" << n);
               REQUIRE(fabs(relerr) < 1e-11);
            }
         }
      }
   }

   SECTION("low tet integration error on reference element for f=x^l y^m z^n, where l+m+n <= p")
   {
      for (int order = 0; order <= 21; order++)
      {
         const IntegrationRule &ir = IntRules.Get(Geometry::TETRAHEDRON, order);

         for (int p = 0; p <= order; p++)
         {
            for (int l = p; l >= 0; l--)
            {
               for (int m = p - l; m >= 0; m--)
               {
                  int n = p - l - m;

                  double integral = 0.0;
                  for (int i = 0; i < ir.GetNPoints(); i++)
                  {
                     const IntegrationPoint &ip = ir.IntPoint(i);
                     integral += ip.weight*poly3d(ip, l, m, n);
                  }

                  double exact = 1.0/binom[p][l+m]/binom[l+m][l]/(p+1)/(p+2)/(p+3);
                  double relerr = 1. - integral/exact;

                  // If a test fails any INFO statements preceding the REQUIRE are displayed
                  INFO("p=" << p << ", l=" << l << ", m=" << m << ", n=" << n);
                  REQUIRE(fabs(relerr) < 1e-11);
               }
            }
         }
      }
   }
}


// WV monomial exactness is tested by [SimplexRules] above, which now uses
// WV rules by default. The tests below verify WV-specific properties:
// weight positivity, stability, and interior point placement.

TEST_CASE("Witherden-Vincent weight positivity", "[WVRules]")
{
   IntegrationRules wv_rules(0, Quadrature1D::GaussLegendre,
                             SimplexQuadrature::WitherdenVincent);

   SECTION("WV triangle rules have all positive weights for orders 0-20")
   {
      for (int order = 0; order <= 20; order++)
      {
         const IntegrationRule &ir = wv_rules.Get(Geometry::TRIANGLE, order);
         for (int i = 0; i < ir.GetNPoints(); i++)
         {
            INFO("order=" << order << ", point=" << i);
            REQUIRE(ir.IntPoint(i).weight > 0.0);
         }
      }
   }

   SECTION("WV tet rules have all positive weights for orders 0-20")
   {
      for (int order = 0; order <= 20; order++)
      {
         const IntegrationRule &ir =
            wv_rules.Get(Geometry::TETRAHEDRON, order);
         for (int i = 0; i < ir.GetNPoints(); i++)
         {
            INFO("order=" << order << ", point=" << i);
            REQUIRE(ir.IntPoint(i).weight > 0.0);
         }
      }
   }
}


TEST_CASE("Witherden-Vincent rules advantage over legacy", "[WVRules]")
{
   IntegrationRules wv_rules(0, Quadrature1D::GaussLegendre,
                             SimplexQuadrature::WitherdenVincent);
   IntegrationRules legacy_rules(0, Quadrature1D::GaussLegendre,
                                 SimplexQuadrature::Legacy);

   // Weight stability metric: sum(|w|) / sum(w).
   // For rules with all positive weights this equals 1.0 exactly.
   // For rules with negative weights this exceeds 1.0, indicating
   // cancellation that amplifies floating-point error.
   auto weight_stability = [](const IntegrationRule &ir)
   {
      double sum_w = 0.0, sum_abs_w = 0.0;
      for (int i = 0; i < ir.GetNPoints(); i++)
      {
         sum_w += ir.IntPoint(i).weight;
         sum_abs_w += fabs(ir.IntPoint(i).weight);
      }
      return sum_abs_w / sum_w;
   };

   SECTION("WV triangle weight stability is 1.0 for all orders 0-20")
   {
      for (int order = 0; order <= 20; order++)
      {
         const IntegrationRule &ir = wv_rules.Get(Geometry::TRIANGLE, order);
         double ratio = weight_stability(ir);
         INFO("order=" << order << ", stability=" << ratio);
         REQUIRE(ratio == MFEM_Approx(1.0));
      }
   }

   SECTION("WV tet weight stability is 1.0 for orders 0-20")
   {
      for (int order = 0; order <= 20; order++)
      {
         const IntegrationRule &ir =
            wv_rules.Get(Geometry::TETRAHEDRON, order);
         double ratio = weight_stability(ir);
         INFO("order=" << order << ", stability=" << ratio);
         REQUIRE(ratio == MFEM_Approx(1.0));
      }
   }

   SECTION("legacy rules have worse weight stability at some orders")
   {
      // At least one legacy triangle or tet rule should have stability > 1.0
      // (negative weights), demonstrating the advantage of WV rules.
      int legacy_unstable_count = 0;
      for (int order = 0; order <= 25; order++)
      {
         double ratio = weight_stability(
                           legacy_rules.Get(Geometry::TRIANGLE, order));
         if (ratio > 1.0 + 1e-14) { legacy_unstable_count++; }
      }
      for (int order = 0; order <= 21; order++)
      {
         double ratio = weight_stability(
                           legacy_rules.Get(Geometry::TETRAHEDRON, order));
         if (ratio > 1.0 + 1e-14) { legacy_unstable_count++; }
      }
      INFO("legacy_unstable_count=" << legacy_unstable_count);
      REQUIRE(legacy_unstable_count > 0);
   }

   SECTION("WV triangle points are strictly interior for orders 0-20")
   {
      for (int order = 0; order <= 20; order++)
      {
         const IntegrationRule &ir = wv_rules.Get(Geometry::TRIANGLE, order);
         for (int i = 0; i < ir.GetNPoints(); i++)
         {
            const IntegrationPoint &ip = ir.IntPoint(i);
            INFO("order=" << order << ", pt=" << i
                 << ", x=" << ip.x << ", y=" << ip.y);
            REQUIRE(ip.x > -1e-15);
            REQUIRE(ip.y > -1e-15);
            REQUIRE(ip.x + ip.y < 1.0 + 1e-15);
         }
      }
   }

   SECTION("WV tet points are strictly interior for orders 0-20")
   {
      for (int order = 0; order <= 20; order++)
      {
         const IntegrationRule &ir =
            wv_rules.Get(Geometry::TETRAHEDRON, order);
         for (int i = 0; i < ir.GetNPoints(); i++)
         {
            const IntegrationPoint &ip = ir.IntPoint(i);
            INFO("order=" << order << ", pt=" << i
                 << ", x=" << ip.x << ", y=" << ip.y << ", z=" << ip.z);
            REQUIRE(ip.x > -1e-15);
            REQUIRE(ip.y > -1e-15);
            REQUIRE(ip.z > -1e-15);
            REQUIRE(ip.x + ip.y + ip.z < 1.0 + 1e-15);
         }
      }
   }

   SECTION("WV rules use no more total quadrature points than legacy")
   {
      int wv_total_tri = 0, leg_total_tri = 0;
      int wv_total_tet = 0, leg_total_tet = 0;
      for (int order = 0; order <= 20; order++)
      {
         wv_total_tri += wv_rules.Get(Geometry::TRIANGLE, order).GetNPoints();
         leg_total_tri += legacy_rules.Get(Geometry::TRIANGLE, order).GetNPoints();
         wv_total_tet +=
            wv_rules.Get(Geometry::TETRAHEDRON, order).GetNPoints();
         leg_total_tet +=
            legacy_rules.Get(Geometry::TETRAHEDRON, order).GetNPoints();
      }
      INFO("triangle total: WV=" << wv_total_tri << " legacy=" << leg_total_tri);
      INFO("tet total: WV=" << wv_total_tet << " legacy=" << leg_total_tet);
      REQUIRE(wv_total_tri <= leg_total_tri);
      REQUIRE(wv_total_tet <= leg_total_tet);
   }
}
