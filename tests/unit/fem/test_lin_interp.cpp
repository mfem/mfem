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
#include "unit_tests.hpp"

using namespace mfem;

namespace lin_interp
{

double f1(const Vector & x) { return 2.345 * x[0]; }
double Grad_f1(const Vector & x) { return 2.345; }

double f2(const Vector & x) { return 2.345 * x[0] + 3.579 * x[1]; }
void F2(const Vector & x, Vector & v)
{
   v.SetSize(2);
   v[0] = 1.234 * x[0] - 2.357 * x[1];
   v[1] = 3.572 * x[0] + 4.321 * x[1];
}

void Grad_f2(const Vector & x, Vector & df)
{
   df.SetSize(2);
   df[0] = 2.345;
   df[1] = 3.579;
}
double CurlF2(const Vector & x) { return 3.572 + 2.357; }
double DivF2(const Vector & x) { return 1.234 + 4.321; }

double f3(const Vector & x)
{ return 2.345 * x[0] + 3.579 * x[1] + 4.680 * x[2]; }
void F3(const Vector & x, Vector & v)
{
   v.SetSize(3);
   v[0] =  1.234 * x[0] - 2.357 * x[1] + 3.572 * x[2];
   v[1] =  2.537 * x[0] + 4.321 * x[1] - 1.234 * x[2];
   v[2] = -2.572 * x[0] + 1.321 * x[1] + 3.234 * x[2];
}

void Grad_f3(const Vector & x, Vector & df)
{
   df.SetSize(3);
   df[0] = 2.345;
   df[1] = 3.579;
   df[2] = 4.680;
}
void CurlF3(const Vector & x, Vector & df)
{
   df.SetSize(3);
   df[0] = 1.321 + 1.234;
   df[1] = 3.572 + 2.572;
   df[2] = 2.537 + 2.357;
}
double DivF3(const Vector & x)
{ return 1.234 + 4.321 + 3.234; }

double g3(const Vector & x)
{ return 4.234 * x[0] + 3.357 * x[1] + 1.572 * x[2]; }

void G3(const Vector & x, Vector & v)
{
   v.SetSize(3);
   v[0] = 4.234 * x[0] + 3.357 * x[1] + 1.572 * x[2];
   v[1] = 4.537 * x[0] + 1.321 * x[1] + 2.234 * x[2];
   v[2] = 1.572 * x[0] + 2.321 * x[1] + 3.234 * x[2];
}

double fg3(const Vector & x) { return f3(x) * g3(x); }
void   fG3(const Vector & x, Vector & v) { G3(x, v); v *= f3(x); }
void   Fg3(const Vector & x, Vector & v) { F3(x, v); v *= g3(x); }

void FcrossG3(const Vector & x, Vector & FxG)
{
   Vector F; F3(x, F);
   Vector G; G3(x, G);
   FxG.SetSize(3);
   FxG(0) = F(1) * G(2) - F(2) * G(1);
   FxG(1) = F(2) * G(0) - F(0) * G(2);
   FxG(2) = F(0) * G(1) - F(1) * G(0);
}

double FdotG3(const Vector & x)
{
   Vector F; F3(x, F);
   Vector G; G3(x, G);
   return F * G;
}

TEST_CASE("1D Identity Linear Interpolators",
          "[IdentityInterpolator]")
{
   int order_h1 = 1, order_l2 = 1, n = 3, dim = 1;
   double tol = 1e-9;

   FunctionCoefficient     fCoef(f1);

   for (int type = (int)Element::SEGMENT;
        type <= (int)Element::SEGMENT; type++)
   {
      Mesh mesh(n, (Element::Type)type, 1, 2.0);

      SECTION("Operators on H1 for element type " + std::to_string(type))
      {
         H1_FECollection    fec_h1(order_h1, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1);

         GridFunction f0(&fespace_h1);
         f0.ProjectCoefficient(fCoef);

         SECTION("Mapping to L2")
         {
            L2_FECollection    fec_l2(order_l2, dim);
            FiniteElementSpace fespace_l2(&mesh, &fec_l2);

            GridFunction f1(&fespace_l2);

            DiscreteLinearOperator Op(&fespace_h1,&fespace_l2);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f0,f1);

            REQUIRE( f1.ComputeL2Error(fCoef) < tol );
         }
         SECTION("Mapping to L2 (INTEGRAL)")
         {
            L2_FECollection    fec_l2(order_l2, dim,
                                      BasisType::GaussLegendre,
                                      FiniteElement::INTEGRAL);
            FiniteElementSpace fespace_l2(&mesh, &fec_l2);

            GridFunction f1(&fespace_l2);

            DiscreteLinearOperator Op(&fespace_h1,&fespace_l2);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f0,f1);

            REQUIRE( f1.ComputeL2Error(fCoef) < tol );
         }
      }
   }
}

TEST_CASE("2D Identity Linear Interpolators",
          "[IdentityInterpolator]")
{
   int order_h1 = 1, order_nd = 2, order_rt = 1, order_l2 = 1, n = 3, dim = 2;
   double tol = 1e-9;

   FunctionCoefficient     fCoef(f2);
   VectorFunctionCoefficient FCoef(dim, F2);

   for (int type = (int)Element::TRIANGLE;
        type <= (int)Element::QUADRILATERAL; type++)
   {
      Mesh mesh(n, n, (Element::Type)type, 1, 2.0, 3.0);

      SECTION("Operators on H1 for element type " + std::to_string(type))
      {
         H1_FECollection    fec_h1(order_h1, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1);

         GridFunction f0(&fespace_h1);
         f0.ProjectCoefficient(fCoef);

         SECTION("Mapping to H1")
         {
            H1_FECollection    fec_h1p(order_h1+1, dim);
            FiniteElementSpace fespace_h1p(&mesh, &fec_h1p);

            GridFunction f0p(&fespace_h1p);

            DiscreteLinearOperator Op(&fespace_h1,&fespace_h1p);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f0,f0p);

            REQUIRE( f0p.ComputeL2Error(fCoef) < tol );
         }
         SECTION("Mapping to L2")
         {
            L2_FECollection    fec_l2(order_l2, dim);
            FiniteElementSpace fespace_l2(&mesh, &fec_l2);

            GridFunction f1(&fespace_l2);

            DiscreteLinearOperator Op(&fespace_h1,&fespace_l2);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f0,f1);

            REQUIRE( f1.ComputeL2Error(fCoef) < tol );
         }
         SECTION("Mapping to L2 (INTEGRAL)")
         {
            L2_FECollection    fec_l2(order_l2, dim,
                                      BasisType::GaussLegendre,
                                      FiniteElement::INTEGRAL);
            FiniteElementSpace fespace_l2(&mesh, &fec_l2);

            GridFunction f1(&fespace_l2);

            DiscreteLinearOperator Op(&fespace_h1,&fespace_l2);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f0,f1);

            REQUIRE( f1.ComputeL2Error(fCoef) < tol );
         }
      }
      SECTION("Operators on L2 for element type " + std::to_string(type))
      {
         L2_FECollection    fec_l2(order_l2, dim);
         FiniteElementSpace fespace_l2(&mesh, &fec_l2);

         GridFunction f0(&fespace_l2);
         f0.ProjectCoefficient(fCoef);

         SECTION("Mapping to L2")
         {
            L2_FECollection    fec_l2p(order_l2+1, dim);
            FiniteElementSpace fespace_l2p(&mesh, &fec_l2p);

            GridFunction f1(&fespace_l2p);

            DiscreteLinearOperator Op(&fespace_l2,&fespace_l2p);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f0,f1);

            REQUIRE( f1.ComputeL2Error(fCoef) < tol );
         }
         SECTION("Mapping to L2 (INTEGRAL)")
         {
            L2_FECollection    fec_l2p(order_l2+1, dim,
                                       BasisType::GaussLegendre,
                                       FiniteElement::INTEGRAL);
            FiniteElementSpace fespace_l2p(&mesh, &fec_l2p);

            GridFunction f1(&fespace_l2p);

            DiscreteLinearOperator Op(&fespace_l2,&fespace_l2p);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f0,f1);

            REQUIRE( f1.ComputeL2Error(fCoef) < tol );
         }
      }
      SECTION("Operators on L2 (INTEGRAL) for element type " +
              std::to_string(type))
      {
         L2_FECollection    fec_l2(order_l2, dim,
                                   BasisType::GaussLegendre,
                                   FiniteElement::INTEGRAL);
         FiniteElementSpace fespace_l2(&mesh, &fec_l2);

         GridFunction f0(&fespace_l2);
         f0.ProjectCoefficient(fCoef);

         SECTION("Mapping to L2")
         {
            L2_FECollection    fec_l2p(order_l2+1, dim);
            FiniteElementSpace fespace_l2p(&mesh, &fec_l2p);

            GridFunction f1(&fespace_l2p);

            DiscreteLinearOperator Op(&fespace_l2,&fespace_l2p);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f0,f1);

            REQUIRE( f1.ComputeL2Error(fCoef) < tol );
         }
         SECTION("Mapping to L2 (INTEGRAL)")
         {
            L2_FECollection    fec_l2p(order_l2+1, dim,
                                       BasisType::GaussLegendre,
                                       FiniteElement::INTEGRAL);
            FiniteElementSpace fespace_l2p(&mesh, &fec_l2p);

            GridFunction f1(&fespace_l2p);

            DiscreteLinearOperator Op(&fespace_l2,&fespace_l2p);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f0,f1);

            REQUIRE( f1.ComputeL2Error(fCoef) < tol );
         }
      }
      SECTION("Operators on HCurl for element type " + std::to_string(type))
      {
         ND_FECollection    fec_nd(order_nd, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         GridFunction f1(&fespace_nd);
         f1.ProjectCoefficient(FCoef);

         SECTION("Mapping to HCurl")
         {
            ND_FECollection    fec_ndp(order_nd+1, dim);
            FiniteElementSpace fespace_ndp(&mesh, &fec_ndp);

            GridFunction f1p(&fespace_ndp);

            DiscreteLinearOperator Op(&fespace_nd,&fespace_ndp);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f1,f1p);

            REQUIRE( f1p.ComputeL2Error(FCoef) < tol );
         }
         SECTION("Mapping to L2^d")
         {
            L2_FECollection    fec_l2(order_l2, dim);
            FiniteElementSpace fespace_l2(&mesh, &fec_l2, dim);

            GridFunction f2d(&fespace_l2);

            DiscreteLinearOperator Op(&fespace_nd,&fespace_l2);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f1,f2d);

            REQUIRE( f2d.ComputeL2Error(FCoef) < tol );
         }
         SECTION("Mapping to L2^d (INTEGRAL)")
         {
            L2_FECollection    fec_l2(order_l2, dim,
                                      BasisType::GaussLegendre,
                                      FiniteElement::INTEGRAL);
            FiniteElementSpace fespace_l2(&mesh, &fec_l2, dim);

            GridFunction f2d(&fespace_l2);

            DiscreteLinearOperator Op(&fespace_nd,&fespace_l2);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f1,f2d);

            REQUIRE( f2d.ComputeL2Error(FCoef) < tol );
         }
      }
      SECTION("Operators on HDiv for element type " + std::to_string(type))
      {
         RT_FECollection    fec_rt(order_rt, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         GridFunction f2(&fespace_rt);
         f2.ProjectCoefficient(FCoef);

         SECTION("Mapping to HDiv")
         {
            RT_FECollection    fec_rtp(order_rt+1, dim);
            FiniteElementSpace fespace_rtp(&mesh, &fec_rtp);

            GridFunction f2p(&fespace_rtp);

            DiscreteLinearOperator Op(&fespace_rt,&fespace_rtp);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f2,f2p);

            REQUIRE( f2p.ComputeL2Error(FCoef) < tol );
         }
         SECTION("Mapping to L2^d")
         {
            L2_FECollection    fec_l2(order_l2, dim);
            FiniteElementSpace fespace_l2(&mesh, &fec_l2, dim);

            GridFunction f2d(&fespace_l2);

            DiscreteLinearOperator Op(&fespace_rt,&fespace_l2);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f2,f2d);

            REQUIRE( f2d.ComputeL2Error(FCoef) < tol );
         }
         SECTION("Mapping to L2^d (INTEGRAL)")
         {
            L2_FECollection    fec_l2(order_l2, dim,
                                      BasisType::GaussLegendre,
                                      FiniteElement::INTEGRAL);
            FiniteElementSpace fespace_l2(&mesh, &fec_l2, dim);

            GridFunction f2d(&fespace_l2);

            DiscreteLinearOperator Op(&fespace_rt,&fespace_l2);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f2,f2d);

            REQUIRE( f2d.ComputeL2Error(FCoef) < tol );
         }
      }
      SECTION("Operators on H1^d for element type " + std::to_string(type))
      {
         H1_FECollection    fec_h1(order_h1, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1, dim);

         GridFunction f0(&fespace_h1);
         f0.ProjectCoefficient(FCoef);

         SECTION("Mapping to HCurl")
         {
            ND_FECollection    fec_ndp(order_nd, dim);
            FiniteElementSpace fespace_ndp(&mesh, &fec_ndp);

            GridFunction f1(&fespace_ndp);

            DiscreteLinearOperator Op(&fespace_h1,&fespace_ndp);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f0,f1);

            REQUIRE( f1.ComputeL2Error(FCoef) < tol );
         }
         SECTION("Mapping to HDiv")
         {
            RT_FECollection    fec_rtp(order_rt, dim);
            FiniteElementSpace fespace_rtp(&mesh, &fec_rtp);

            GridFunction f2(&fespace_rtp);

            DiscreteLinearOperator Op(&fespace_h1,&fespace_rtp);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f0,f2);

            REQUIRE( f2.ComputeL2Error(FCoef) < tol );
         }
         /// The following tests would fail.  The reason for the
         /// failure would not be obvious from the user's point of
         /// view.  I recommend keeping these tests here as a reminder
         /// that we should consider supporting this, or a very
         /// similar, usage.
         /*
              SECTION("Mapping to L2^d")
              {
                 L2_FECollection    fec_l2(order_l2, dim);
                 FiniteElementSpace fespace_l2(&mesh, &fec_l2, dim);

                 GridFunction f2d(&fespace_l2);

                 DiscreteLinearOperator Op(&fespace_h1,&fespace_l2);
                 Op.AddDomainInterpolator(new IdentityInterpolator());
                 Op.Assemble();

                 Op.Mult(f0,f2d);

                 REQUIRE( f2d.ComputeL2Error(FCoef) < tol );
              }
              SECTION("Mapping to L2^d (INTEGRAL)")
              {
                 L2_FECollection    fec_l2(order_l2, dim,
                                           BasisType::GaussLegendre,
                                           FiniteElement::INTEGRAL);
                 FiniteElementSpace fespace_l2(&mesh, &fec_l2, dim);

                 GridFunction f2d(&fespace_l2);

                 DiscreteLinearOperator Op(&fespace_h1,&fespace_l2);
                 Op.AddDomainInterpolator(new IdentityInterpolator());
                 Op.Assemble();

                 Op.Mult(f0,f2d);

                 REQUIRE( f2d.ComputeL2Error(FCoef) < tol );
              }
         */
      }
   }
}

TEST_CASE("3D Identity Linear Interpolators",
          "[IdentityInterpolator]")
{
   int order_h1 = 1, order_nd = 2, order_rt = 1, order_l2 = 1, n = 3, dim = 3;
   double tol = 1e-9;

   FunctionCoefficient     fCoef(f3);
   VectorFunctionCoefficient FCoef(dim, F3);

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      if (type == Element::TETRAHEDRON)
      {
         mesh.ReorientTetMesh();
      }

      SECTION("Operators on H1 for element type " + std::to_string(type))
      {
         H1_FECollection    fec_h1(order_h1, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1);

         GridFunction f0(&fespace_h1);
         f0.ProjectCoefficient(fCoef);

         SECTION("Mapping to H1")
         {
            H1_FECollection    fec_h1p(order_h1+1, dim);
            FiniteElementSpace fespace_h1p(&mesh, &fec_h1p);

            GridFunction f0p(&fespace_h1p);

            DiscreteLinearOperator Op(&fespace_h1,&fespace_h1p);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f0,f0p);

            REQUIRE( f0p.ComputeL2Error(fCoef) < tol );
         }
         SECTION("Mapping to L2")
         {
            L2_FECollection    fec_l2(order_l2, dim);
            FiniteElementSpace fespace_l2(&mesh, &fec_l2);

            GridFunction f1(&fespace_l2);

            DiscreteLinearOperator Op(&fespace_h1,&fespace_l2);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f0,f1);

            REQUIRE( f1.ComputeL2Error(fCoef) < tol );
         }
         SECTION("Mapping to L2 (INTEGRAL)")
         {
            L2_FECollection    fec_l2(order_l2, dim,
                                      BasisType::GaussLegendre,
                                      FiniteElement::INTEGRAL);
            FiniteElementSpace fespace_l2(&mesh, &fec_l2);

            GridFunction f1(&fespace_l2);

            DiscreteLinearOperator Op(&fespace_h1,&fespace_l2);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f0,f1);

            REQUIRE( f1.ComputeL2Error(fCoef) < tol );
         }
      }
      SECTION("Operators on L2 for element type " + std::to_string(type))
      {
         L2_FECollection    fec_l2(order_l2, dim);
         FiniteElementSpace fespace_l2(&mesh, &fec_l2);

         GridFunction f0(&fespace_l2);
         f0.ProjectCoefficient(fCoef);

         SECTION("Mapping to L2")
         {
            L2_FECollection    fec_l2p(order_l2+1, dim);
            FiniteElementSpace fespace_l2p(&mesh, &fec_l2p);

            GridFunction f1(&fespace_l2p);

            DiscreteLinearOperator Op(&fespace_l2,&fespace_l2p);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f0,f1);

            REQUIRE( f1.ComputeL2Error(fCoef) < tol );
         }
         SECTION("Mapping to L2 (INTEGRAL)")
         {
            L2_FECollection    fec_l2p(order_l2+1, dim,
                                       BasisType::GaussLegendre,
                                       FiniteElement::INTEGRAL);
            FiniteElementSpace fespace_l2p(&mesh, &fec_l2p);

            GridFunction f1(&fespace_l2p);

            DiscreteLinearOperator Op(&fespace_l2,&fespace_l2p);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f0,f1);

            REQUIRE( f1.ComputeL2Error(fCoef) < tol );
         }
      }
      SECTION("Operators on L2 (INTEGRAL) for element type " +
              std::to_string(type))
      {
         L2_FECollection    fec_l2(order_l2, dim,
                                   BasisType::GaussLegendre,
                                   FiniteElement::INTEGRAL);
         FiniteElementSpace fespace_l2(&mesh, &fec_l2);

         GridFunction f0(&fespace_l2);
         f0.ProjectCoefficient(fCoef);

         SECTION("Mapping to L2")
         {
            L2_FECollection    fec_l2p(order_l2+1, dim);
            FiniteElementSpace fespace_l2p(&mesh, &fec_l2p);

            GridFunction f1(&fespace_l2p);

            DiscreteLinearOperator Op(&fespace_l2,&fespace_l2p);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f0,f1);

            REQUIRE( f1.ComputeL2Error(fCoef) < tol );
         }
         SECTION("Mapping to L2 (INTEGRAL)")
         {
            L2_FECollection    fec_l2p(order_l2+1, dim,
                                       BasisType::GaussLegendre,
                                       FiniteElement::INTEGRAL);
            FiniteElementSpace fespace_l2p(&mesh, &fec_l2p);

            GridFunction f1(&fespace_l2p);

            DiscreteLinearOperator Op(&fespace_l2,&fespace_l2p);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f0,f1);

            REQUIRE( f1.ComputeL2Error(fCoef) < tol );
         }
      }
      SECTION("Operators on HCurl for element type " + std::to_string(type))
      {
         ND_FECollection    fec_nd(order_nd, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         GridFunction f1(&fespace_nd);
         f1.ProjectCoefficient(FCoef);

         SECTION("Mapping to HCurl")
         {
            ND_FECollection    fec_ndp(order_nd+1, dim);
            FiniteElementSpace fespace_ndp(&mesh, &fec_ndp);

            GridFunction f1p(&fespace_ndp);

            DiscreteLinearOperator Op(&fespace_nd,&fespace_ndp);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f1,f1p);

            REQUIRE( f1p.ComputeL2Error(FCoef) < tol );
         }
         SECTION("Mapping to L2^d")
         {
            L2_FECollection    fec_l2(order_l2, dim);
            FiniteElementSpace fespace_l2(&mesh, &fec_l2, dim);

            GridFunction f2d(&fespace_l2);

            DiscreteLinearOperator Op(&fespace_nd,&fespace_l2);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f1,f2d);

            REQUIRE( f2d.ComputeL2Error(FCoef) < tol );
         }
         SECTION("Mapping to L2^d (INTEGRAL)")
         {
            L2_FECollection    fec_l2(order_l2, dim,
                                      BasisType::GaussLegendre,
                                      FiniteElement::INTEGRAL);
            FiniteElementSpace fespace_l2(&mesh, &fec_l2, dim);

            GridFunction f2d(&fespace_l2);

            DiscreteLinearOperator Op(&fespace_nd,&fespace_l2);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f1,f2d);

            REQUIRE( f2d.ComputeL2Error(FCoef) < tol );
         }
      }
      SECTION("Operators on HDiv for element type " + std::to_string(type))
      {
         RT_FECollection    fec_rt(order_rt, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         GridFunction f2(&fespace_rt);
         f2.ProjectCoefficient(FCoef);

         SECTION("Mapping to HDiv")
         {
            RT_FECollection    fec_rtp(order_rt+1, dim);
            FiniteElementSpace fespace_rtp(&mesh, &fec_rtp);

            GridFunction f2p(&fespace_rtp);

            DiscreteLinearOperator Op(&fespace_rt,&fespace_rtp);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f2,f2p);

            REQUIRE( f2p.ComputeL2Error(FCoef) < tol );
         }
         SECTION("Mapping to L2^d")
         {
            L2_FECollection    fec_l2(order_l2, dim);
            FiniteElementSpace fespace_l2(&mesh, &fec_l2, dim);

            GridFunction f2d(&fespace_l2);

            DiscreteLinearOperator Op(&fespace_rt,&fespace_l2);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f2,f2d);

            REQUIRE( f2d.ComputeL2Error(FCoef) < tol );
         }
         SECTION("Mapping to L2^d (INTEGRAL)")
         {
            L2_FECollection    fec_l2(order_l2, dim,
                                      BasisType::GaussLegendre,
                                      FiniteElement::INTEGRAL);
            FiniteElementSpace fespace_l2(&mesh, &fec_l2, dim);

            GridFunction f2d(&fespace_l2);

            DiscreteLinearOperator Op(&fespace_rt,&fespace_l2);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f2,f2d);

            REQUIRE( f2d.ComputeL2Error(FCoef) < tol );
         }
      }
      SECTION("Operators on H1^d for element type " + std::to_string(type))
      {
         H1_FECollection    fec_h1(order_h1, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1, dim);

         GridFunction f0(&fespace_h1);
         f0.ProjectCoefficient(FCoef);

         SECTION("Mapping to HCurl")
         {
            ND_FECollection    fec_ndp(order_nd, dim);
            FiniteElementSpace fespace_ndp(&mesh, &fec_ndp);

            GridFunction f1(&fespace_ndp);

            DiscreteLinearOperator Op(&fespace_h1,&fespace_ndp);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f0,f1);

            REQUIRE( f1.ComputeL2Error(FCoef) < tol );
         }
         SECTION("Mapping to HDiv")
         {
            RT_FECollection    fec_rtp(order_rt, dim);
            FiniteElementSpace fespace_rtp(&mesh, &fec_rtp);

            GridFunction f2(&fespace_rtp);

            DiscreteLinearOperator Op(&fespace_h1,&fespace_rtp);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f0,f2);

            REQUIRE( f2.ComputeL2Error(FCoef) < tol );
         }
         /// The following tests would fail.  The reason for the
         /// failure would not be obvious from the user's point of
         /// view.  I recommend keeping these tests here as a reminder
         /// that we should consider supporting this, or a very
         /// similar, usage.
         /*
              SECTION("Mapping to L2^d")
              {
                 L2_FECollection    fec_l2(order_l2, dim);
                 FiniteElementSpace fespace_l2(&mesh, &fec_l2, dim);

                 GridFunction f2d(&fespace_l2);

                 DiscreteLinearOperator Op(&fespace_h1,&fespace_l2);
                 Op.AddDomainInterpolator(new IdentityInterpolator());
                 Op.Assemble();

                 Op.Mult(f0,f2d);

                 REQUIRE( f2d.ComputeL2Error(FCoef) < tol );
              }
              SECTION("Mapping to L2^d (INTEGRAL)")
              {
                 L2_FECollection    fec_l2(order_l2, dim,
                                           BasisType::GaussLegendre,
                                           FiniteElement::INTEGRAL);
                 FiniteElementSpace fespace_l2(&mesh, &fec_l2, dim);

                 GridFunction f2d(&fespace_l2);

                 DiscreteLinearOperator Op(&fespace_h1,&fespace_l2);
                 Op.AddDomainInterpolator(new IdentityInterpolator());
                 Op.Assemble();

                 Op.Mult(f0,f2d);

                 REQUIRE( f2d.ComputeL2Error(FCoef) < tol );
              }
         */
      }
   }
}

TEST_CASE("1D Derivative Linear Interpolators",
          "[GradientInterpolator]")
{
   int order_h1 = 1, order_l2 = 0, n = 3, dim = 1;
   double tol = 1e-9;

   FunctionCoefficient     fCoef(f1);
   FunctionCoefficient GradfCoef(Grad_f1);


   for (int type = (int)Element::SEGMENT;
        type <= (int)Element::SEGMENT; type++)
   {
      Mesh mesh(n, (Element::Type)type, 1, 2.0);

      SECTION("Operators on H1 for element type " + std::to_string(type))
      {
         H1_FECollection    fec_h1(order_h1, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1);

         GridFunction f0(&fespace_h1);
         f0.ProjectCoefficient(fCoef);

         SECTION("Mapping to L2")
         {
            L2_FECollection    fec_l2(order_l2, dim);
            FiniteElementSpace fespace_l2(&mesh, &fec_l2);

            GridFunction df0(&fespace_l2);

            DiscreteLinearOperator Op(&fespace_h1,&fespace_l2);
            Op.AddDomainInterpolator(new GradientInterpolator());
            Op.Assemble();

            Op.Mult(f0,df0);

            REQUIRE( df0.ComputeL2Error(GradfCoef) < tol );
         }
         SECTION("Mapping to L2 (INTEGRAL)")
         {
            L2_FECollection    fec_l2(order_l2, dim,
                                      BasisType::GaussLegendre,
                                      FiniteElement::INTEGRAL);
            FiniteElementSpace fespace_l2(&mesh, &fec_l2);

            GridFunction df0(&fespace_l2);

            DiscreteLinearOperator Op(&fespace_h1,&fespace_l2);
            Op.AddDomainInterpolator(new GradientInterpolator());
            Op.Assemble();

            Op.Mult(f0,df0);

            REQUIRE( df0.ComputeL2Error(GradfCoef) < tol );
         }
      }
   }
}

TEST_CASE("2D Derivative Linear Interpolators",
          "[GradientInterpolator]"
          "[CurlInterpolator]"
          "[DivergenceInterpolator]")
{
   int order_h1 = 1, order_nd = 1, order_rt = 0, order_l2 = 0, n = 3, dim = 2;
   double tol = 1e-9;

   FunctionCoefficient       fCoef(f2);
   VectorFunctionCoefficient FCoef(dim, F2);

   VectorFunctionCoefficient GradfCoef(dim, Grad_f2);
   FunctionCoefficient       CurlFCoef(CurlF2);
   FunctionCoefficient       DivFCoef(DivF2);

   for (int type = (int)Element::TRIANGLE;
        type <= (int)Element::QUADRILATERAL; type++)
   {
      Mesh mesh(n, n, (Element::Type)type, 1, 2.0, 3.0);

      SECTION("Operators on H1 for element type " + std::to_string(type))
      {
         H1_FECollection    fec_h1(order_h1, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1);

         GridFunction f0(&fespace_h1);
         f0.ProjectCoefficient(fCoef);

         SECTION("Mapping to HCurl")
         {
            ND_FECollection    fec_nd(order_nd, dim);
            FiniteElementSpace fespace_nd(&mesh, &fec_nd);

            GridFunction df0(&fespace_nd);

            DiscreteLinearOperator Op(&fespace_h1,&fespace_nd);
            Op.AddDomainInterpolator(new GradientInterpolator());
            Op.Assemble();

            Op.Mult(f0,df0);

            REQUIRE( df0.ComputeL2Error(GradfCoef) < tol );
         }
      }
      SECTION("Operators on HCurl for element type " + std::to_string(type))
      {
         ND_FECollection    fec_nd(order_nd, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         GridFunction F1(&fespace_nd);
         F1.ProjectCoefficient(FCoef);

         SECTION("Mapping to L2 (INTEGRAL)")
         {
            L2_FECollection    fec_l2(order_l2, dim,
                                      BasisType::GaussLegendre,
                                      FiniteElement::INTEGRAL);
            FiniteElementSpace fespace_l2(&mesh, &fec_l2);

            GridFunction dF1(&fespace_l2);

            DiscreteLinearOperator Op(&fespace_nd,&fespace_l2);
            Op.AddDomainInterpolator(new CurlInterpolator());
            Op.Assemble();

            Op.Mult(F1,dF1);

            REQUIRE( dF1.ComputeL2Error(CurlFCoef) < tol );
         }
      }
      SECTION("Operators on HDiv for element type " + std::to_string(type))
      {
         RT_FECollection    fec_rt(order_rt, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         GridFunction F2(&fespace_rt);
         F2.ProjectCoefficient(FCoef);

         SECTION("Mapping to L2")
         {
            L2_FECollection    fec_l2(order_l2, dim);
            FiniteElementSpace fespace_l2(&mesh, &fec_l2);

            GridFunction dF2(&fespace_l2);

            DiscreteLinearOperator Op(&fespace_rt,&fespace_l2);
            Op.AddDomainInterpolator(new DivergenceInterpolator());
            Op.Assemble();

            Op.Mult(F2,dF2);

            REQUIRE( dF2.ComputeL2Error(DivFCoef) < tol );
         }
         SECTION("Mapping to L2 (INTEGRAL)")
         {
            L2_FECollection    fec_l2(order_l2, dim,
                                      BasisType::GaussLegendre,
                                      FiniteElement::INTEGRAL);
            FiniteElementSpace fespace_l2(&mesh, &fec_l2);

            GridFunction dF2(&fespace_l2);

            DiscreteLinearOperator Op(&fespace_rt,&fespace_l2);
            Op.AddDomainInterpolator(new DivergenceInterpolator());
            Op.Assemble();

            Op.Mult(F2,dF2);

            REQUIRE( dF2.ComputeL2Error(DivFCoef) < tol );
         }
      }
   }
}

TEST_CASE("3D Derivative Linear Interpolators",
          "[GradientInterpolator]"
          "[CurlInterpolator]"
          "[DivergenceInterpolator]")
{
   int order_h1 = 1, order_nd = 1, order_rt = 0, order_l2 = 0, n = 3, dim = 3;
   double tol = 1e-9;

   FunctionCoefficient       fCoef(f3);
   VectorFunctionCoefficient FCoef(dim, F3);

   VectorFunctionCoefficient GradfCoef(dim, Grad_f3);
   VectorFunctionCoefficient CurlFCoef(dim, CurlF3);
   FunctionCoefficient       DivFCoef(DivF3);

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      if (type == Element::TETRAHEDRON)
      {
         mesh.ReorientTetMesh();
      }

      SECTION("Operators on H1 for element type " + std::to_string(type))
      {
         H1_FECollection    fec_h1(order_h1, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1);

         GridFunction f0(&fespace_h1);
         f0.ProjectCoefficient(fCoef);

         SECTION("Mapping to HCurl")
         {
            ND_FECollection    fec_nd(order_nd, dim);
            FiniteElementSpace fespace_nd(&mesh, &fec_nd);

            GridFunction df0(&fespace_nd);

            DiscreteLinearOperator Op(&fespace_h1,&fespace_nd);
            Op.AddDomainInterpolator(new GradientInterpolator());
            Op.Assemble();

            Op.Mult(f0,df0);

            REQUIRE( df0.ComputeL2Error(GradfCoef) < tol );
         }
      }
      SECTION("Operators on HCurl for element type " + std::to_string(type))
      {
         ND_FECollection    fec_nd(order_nd, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         GridFunction F1(&fespace_nd);
         F1.ProjectCoefficient(FCoef);

         SECTION("Mapping to HDiv")
         {
            RT_FECollection    fec_rt(order_rt, dim);
            FiniteElementSpace fespace_rt(&mesh, &fec_rt);

            GridFunction dF1(&fespace_rt);

            DiscreteLinearOperator Op(&fespace_nd,&fespace_rt);
            Op.AddDomainInterpolator(new CurlInterpolator());
            Op.Assemble();

            Op.Mult(F1,dF1);

            REQUIRE( dF1.ComputeL2Error(CurlFCoef) < tol );
         }
      }
      SECTION("Operators on HDiv for element type " + std::to_string(type))
      {
         RT_FECollection    fec_rt(order_rt, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         GridFunction F2(&fespace_rt);
         F2.ProjectCoefficient(FCoef);

         SECTION("Mapping to L2")
         {
            L2_FECollection    fec_l2(order_l2, dim);
            FiniteElementSpace fespace_l2(&mesh, &fec_l2);

            GridFunction dF2(&fespace_l2);

            DiscreteLinearOperator Op(&fespace_rt,&fespace_l2);
            Op.AddDomainInterpolator(new DivergenceInterpolator());
            Op.Assemble();

            Op.Mult(F2,dF2);

            REQUIRE( dF2.ComputeL2Error(DivFCoef) < tol );
         }
         SECTION("Mapping to L2 (INTEGRAL)")
         {
            L2_FECollection    fec_l2(order_l2, dim,
                                      BasisType::GaussLegendre,
                                      FiniteElement::INTEGRAL);
            FiniteElementSpace fespace_l2(&mesh, &fec_l2);

            GridFunction dF2(&fespace_l2);

            DiscreteLinearOperator Op(&fespace_rt,&fespace_l2);
            Op.AddDomainInterpolator(new DivergenceInterpolator());
            Op.Assemble();

            Op.Mult(F2,dF2);

            REQUIRE( dF2.ComputeL2Error(DivFCoef) < tol );
         }
      }
   }
}

TEST_CASE("3D Product Linear Interpolators",
          "[ScalarProductInterpolator]"
          "[VectorScalarProductInterpolator]"
          "[ScalarVectorProductInterpolator]"
          "[VectorCrossProductInterpolator]"
          "[VectorInnerProductInterpolator]")
{
   int order_h1 = 1, order_nd = 2, order_rt = 2, n = 3, dim = 3;
   double tol = 1e-9;

   FunctionCoefficient       fCoef(f3);
   VectorFunctionCoefficient FCoef(dim, F3);

   FunctionCoefficient       gCoef(g3);
   VectorFunctionCoefficient GCoef(dim, G3);

   FunctionCoefficient        fgCoef(fg3);
   FunctionCoefficient        FGCoef(FdotG3);
   VectorFunctionCoefficient  fGCoef(dim, fG3);
   VectorFunctionCoefficient  FgCoef(dim, Fg3);
   VectorFunctionCoefficient FxGCoef(dim, FcrossG3);

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      if (type == Element::TETRAHEDRON)
      {
         mesh.ReorientTetMesh();
      }

      SECTION("Operators on H1 for element type " + std::to_string(type))
      {
         H1_FECollection    fec_h1(order_h1, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1);

         GridFunction g0(&fespace_h1);
         g0.ProjectCoefficient(gCoef);

         SECTION("Mapping H1 to H1")
         {
            GridFunction f0(&fespace_h1);
            f0.ProjectCoefficient(fCoef);

            H1_FECollection    fec_h1p(2*order_h1, dim);
            FiniteElementSpace fespace_h1p(&mesh, &fec_h1p);

            DiscreteLinearOperator Opf0(&fespace_h1,&fespace_h1p);
            Opf0.AddDomainInterpolator(new ScalarProductInterpolator(fCoef));
            Opf0.Assemble();

            GridFunction fg0(&fespace_h1p);
            Opf0.Mult(g0,fg0);

            REQUIRE( fg0.ComputeL2Error(fgCoef) < tol );
         }
         SECTION("Mapping to HCurl")
         {
            ND_FECollection    fec_nd(order_nd, dim);
            FiniteElementSpace fespace_nd(&mesh, &fec_nd);

            GridFunction F1(&fespace_nd);
            F1.ProjectCoefficient(FCoef);

            ND_FECollection    fec_ndp(order_h1+order_nd, dim);
            FiniteElementSpace fespace_ndp(&mesh, &fec_ndp);

            DiscreteLinearOperator OpF1(&fespace_h1,&fespace_ndp);
            OpF1.AddDomainInterpolator(new VectorScalarProductInterpolator(FCoef));
            OpF1.Assemble();

            GridFunction Fg1(&fespace_ndp);
            OpF1.Mult(g0,Fg1);

            REQUIRE( Fg1.ComputeL2Error(FgCoef) < tol );
         }
      }
      SECTION("Operators on HCurl for element type " + std::to_string(type))
      {
         ND_FECollection    fec_nd(order_nd, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         GridFunction G1(&fespace_nd);
         G1.ProjectCoefficient(GCoef);

         SECTION("Mapping HCurl to HCurl")
         {
            H1_FECollection    fec_h1(order_h1, dim);
            FiniteElementSpace fespace_h1(&mesh, &fec_h1);

            GridFunction f0(&fespace_h1);
            f0.ProjectCoefficient(fCoef);

            ND_FECollection    fec_ndp(order_nd+order_h1, dim);
            FiniteElementSpace fespace_ndp(&mesh, &fec_ndp);

            DiscreteLinearOperator Opf0(&fespace_nd,&fespace_ndp);
            Opf0.AddDomainInterpolator(new ScalarVectorProductInterpolator(fCoef));
            Opf0.Assemble();

            GridFunction fG1(&fespace_ndp);
            Opf0.Mult(G1,fG1);

            REQUIRE( fG1.ComputeL2Error(fGCoef) < tol );
         }
         SECTION("Mapping to HDiv")
         {
            GridFunction F1(&fespace_nd);
            F1.ProjectCoefficient(FCoef);

            RT_FECollection    fec_rtp(2*order_nd, dim);
            FiniteElementSpace fespace_rtp(&mesh, &fec_rtp);

            DiscreteLinearOperator OpF1(&fespace_nd,&fespace_rtp);
            OpF1.AddDomainInterpolator(new VectorCrossProductInterpolator(FCoef));
            OpF1.Assemble();

            GridFunction FxG2(&fespace_rtp);
            OpF1.Mult(G1,FxG2);

            REQUIRE( FxG2.ComputeL2Error(FxGCoef) < tol );
         }
         SECTION("Mapping to L2")
         {
            RT_FECollection    fec_rt(order_rt, dim);
            FiniteElementSpace fespace_rt(&mesh, &fec_rt);

            GridFunction F2(&fespace_rt);
            F2.ProjectCoefficient(FCoef);

            L2_FECollection    fec_l2p(order_nd+order_rt, dim);
            FiniteElementSpace fespace_l2p(&mesh, &fec_l2p);

            DiscreteLinearOperator OpF2(&fespace_nd,&fespace_l2p);
            OpF2.AddDomainInterpolator(new VectorInnerProductInterpolator(FCoef));
            OpF2.Assemble();

            GridFunction FG3(&fespace_l2p);
            OpF2.Mult(G1,FG3);

            REQUIRE( FG3.ComputeL2Error(FGCoef) < tol );
         }
      }
      SECTION("Operators on HDiv for element type " + std::to_string(type))
      {
         RT_FECollection    fec_rt(order_rt, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         GridFunction G2(&fespace_rt);
         G2.ProjectCoefficient(GCoef);

         SECTION("Mapping to L2")
         {
            ND_FECollection    fec_nd(order_nd, dim);
            FiniteElementSpace fespace_nd(&mesh, &fec_nd);

            GridFunction F1(&fespace_nd);
            F1.ProjectCoefficient(FCoef);

            L2_FECollection    fec_l2p(order_nd+order_rt, dim);
            FiniteElementSpace fespace_l2p(&mesh, &fec_l2p);

            DiscreteLinearOperator OpF1(&fespace_rt,&fespace_l2p);
            OpF1.AddDomainInterpolator(new VectorInnerProductInterpolator(FCoef));
            OpF1.Assemble();

            GridFunction FG3(&fespace_l2p);
            OpF1.Mult(G2,FG3);

            REQUIRE( FG3.ComputeL2Error(FGCoef) < tol );
         }
      }
   }
}

} // namespace lin_interp
