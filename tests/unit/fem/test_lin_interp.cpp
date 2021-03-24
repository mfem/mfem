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

namespace lin_interp
{

double f1(const Vector & x) { return 2.345 * x[0]; }
double grad_f1(const Vector & x) { return 2.345; }
void Grad_f1(const Vector & x, Vector & df)
{
   df.SetSize(1);
   df[0] = grad_f1(x);
}

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
double curlF2(const Vector & x) { return 3.572 + 2.357; }
void CurlF2(const Vector & x, Vector & v)
{
   v.SetSize(1);
   v[0] = curlF2(x);
}
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

double g1(const Vector & x) { return 4.234 * x[0]; }
double g2(const Vector & x) { return 4.234 * x[0] + 3.357 * x[1]; }
double g3(const Vector & x)
{ return 4.234 * x[0] + 3.357 * x[1] + 1.572 * x[2]; }

void G2(const Vector & x, Vector & v)
{
   v.SetSize(2);
   v[0] = 4.234 * x[0] + 3.357 * x[1];
   v[1] = 4.537 * x[0] + 1.321 * x[1];
}
void G3(const Vector & x, Vector & v)
{
   v.SetSize(3);
   v[0] = 4.234 * x[0] + 3.357 * x[1] + 1.572 * x[2];
   v[1] = 4.537 * x[0] + 1.321 * x[1] + 2.234 * x[2];
   v[2] = 1.572 * x[0] + 2.321 * x[1] + 3.234 * x[2];
}

double fg1(const Vector & x) { return f1(x) * g1(x); }

double fg2(const Vector & x) { return f2(x) * g2(x); }
void   fG2(const Vector & x, Vector & v) { G2(x, v); v *= f2(x); }
void   Fg2(const Vector & x, Vector & v) { F2(x, v); v *= g2(x); }

void FcrossG2(const Vector & x, Vector & FxG)
{
   Vector F; F2(x, F);
   Vector G; G2(x, G);
   FxG.SetSize(1);
   FxG(0) = F(0) * G(1) - F(1) * G(0);
}

double FdotG2(const Vector & x)
{
   Vector F; F2(x, F);
   Vector G; G2(x, G);
   return F * G;
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

TEST_CASE("Identity Linear Interpolators",
          "[IdentityInterpolator]")
{
   int order_h1 = 1, order_nd = 2, order_rt = 1, order_l2 = 1, n = 3, dim = -1;
   double tol = 1e-9;

   for (int type = (int)Element::SEGMENT;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh *mesh = NULL;

      if (type < (int)Element::TRIANGLE)
      {
         dim = 1;
         mesh = new Mesh(n, (Element::Type)type, 1, 2.0);

      }
      else if (type < (int)Element::TETRAHEDRON)
      {
         dim = 2;
         mesh = new Mesh(n, n, (Element::Type)type, 1, 2.0, 3.0);
      }
      else
      {
         dim = 3;
         mesh = new Mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

         if (type == Element::TETRAHEDRON)
         {
            mesh->ReorientTetMesh();
         }
      }

      FunctionCoefficient        fCoef((dim==1) ? f1 :
                                       ((dim==2) ? f2 : f3));
      VectorFunctionCoefficient dfCoef(dim,
                                       (dim==1) ? Grad_f1 :
                                       ((dim==2)? Grad_f2 : Grad_f3));

      SECTION("Operators on H1 for element type " + std::to_string(type))
      {
         H1_FECollection    fec_h1(order_h1, dim);
         FiniteElementSpace fespace_h1(mesh, &fec_h1);

         GridFunction f0(&fespace_h1);
         f0.ProjectCoefficient(fCoef);

         SECTION("Mapping to H1")
         {
            H1_FECollection    fec_h1p(order_h1+1, dim);
            FiniteElementSpace fespace_h1p(mesh, &fec_h1p);

            GridFunction f0p(&fespace_h1p);

            DiscreteLinearOperator Op(&fespace_h1,&fespace_h1p);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f0,f0p);

            REQUIRE( f0p.ComputeH1Error(&fCoef, &dfCoef) < tol );
         }
         SECTION("Mapping to L2")
         {
            L2_FECollection    fec_l2(order_l2, dim);
            FiniteElementSpace fespace_l2(mesh, &fec_l2);

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
            FiniteElementSpace fespace_l2(mesh, &fec_l2);

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
         FiniteElementSpace fespace_l2(mesh, &fec_l2);

         GridFunction f0(&fespace_l2);
         f0.ProjectCoefficient(fCoef);

         SECTION("Mapping to L2")
         {
            L2_FECollection    fec_l2p(order_l2+1, dim);
            FiniteElementSpace fespace_l2p(mesh, &fec_l2p);

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
            FiniteElementSpace fespace_l2p(mesh, &fec_l2p);

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
         FiniteElementSpace fespace_l2(mesh, &fec_l2);

         GridFunction f0(&fespace_l2);
         f0.ProjectCoefficient(fCoef);

         SECTION("Mapping to L2")
         {
            L2_FECollection    fec_l2p(order_l2+1, dim);
            FiniteElementSpace fespace_l2p(mesh, &fec_l2p);

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
            FiniteElementSpace fespace_l2p(mesh, &fec_l2p);

            GridFunction f1(&fespace_l2p);

            DiscreteLinearOperator Op(&fespace_l2,&fespace_l2p);
            Op.AddDomainInterpolator(new IdentityInterpolator());
            Op.Assemble();

            Op.Mult(f0,f1);

            REQUIRE( f1.ComputeL2Error(fCoef) < tol );
         }
      }
      if (dim > 1)
      {
         VectorFunctionCoefficient     FCoef(dim,
                                             (dim==2) ? F2 : F3);
         VectorFunctionCoefficient curlFCoef(dim,
                                             (dim==2) ? CurlF2 : CurlF3);
         FunctionCoefficient        divFCoef((dim==2) ? DivF2 : DivF3);

         SECTION("Operators on HCurl for element type " + std::to_string(type))
         {
            ND_FECollection    fec_nd(order_nd, dim);
            FiniteElementSpace fespace_nd(mesh, &fec_nd);

            GridFunction f1(&fespace_nd);
            f1.ProjectCoefficient(FCoef);

            SECTION("Mapping to HCurl")
            {
               ND_FECollection    fec_ndp(order_nd+1, dim);
               FiniteElementSpace fespace_ndp(mesh, &fec_ndp);

               GridFunction f1p(&fespace_ndp);

               DiscreteLinearOperator Op(&fespace_nd,&fespace_ndp);
               Op.AddDomainInterpolator(new IdentityInterpolator());
               Op.Assemble();

               Op.Mult(f1,f1p);

               REQUIRE( f1p.ComputeHCurlError(&FCoef, &curlFCoef) < tol );
            }
            SECTION("Mapping to L2^d")
            {
               L2_FECollection    fec_l2(order_l2, dim);
               FiniteElementSpace fespace_l2(mesh, &fec_l2, dim);

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
               FiniteElementSpace fespace_l2(mesh, &fec_l2, dim);

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
            FiniteElementSpace fespace_rt(mesh, &fec_rt);

            GridFunction f2(&fespace_rt);
            f2.ProjectCoefficient(FCoef);

            SECTION("Mapping to HDiv")
            {
               RT_FECollection    fec_rtp(order_rt+1, dim);
               FiniteElementSpace fespace_rtp(mesh, &fec_rtp);

               GridFunction f2p(&fespace_rtp);

               DiscreteLinearOperator Op(&fespace_rt,&fespace_rtp);
               Op.AddDomainInterpolator(new IdentityInterpolator());
               Op.Assemble();

               Op.Mult(f2,f2p);

               REQUIRE( f2p.ComputeHDivError(&FCoef, &divFCoef) < tol );
            }
            SECTION("Mapping to L2^d")
            {
               L2_FECollection    fec_l2(order_l2, dim);
               FiniteElementSpace fespace_l2(mesh, &fec_l2, dim);

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
               FiniteElementSpace fespace_l2(mesh, &fec_l2, dim);

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
            FiniteElementSpace fespace_h1(mesh, &fec_h1, dim);

            GridFunction f0(&fespace_h1);
            f0.ProjectCoefficient(FCoef);

            SECTION("Mapping to HCurl")
            {
               ND_FECollection    fec_ndp(order_nd, dim);
               FiniteElementSpace fespace_ndp(mesh, &fec_ndp);

               GridFunction f1(&fespace_ndp);

               DiscreteLinearOperator Op(&fespace_h1,&fespace_ndp);
               Op.AddDomainInterpolator(new IdentityInterpolator());
               Op.Assemble();

               Op.Mult(f0,f1);

               REQUIRE( f1.ComputeHCurlError(&FCoef, &curlFCoef) < tol );
            }
            SECTION("Mapping to HDiv")
            {
               RT_FECollection    fec_rtp(order_rt, dim);
               FiniteElementSpace fespace_rtp(mesh, &fec_rtp);

               GridFunction f2(&fespace_rtp);

               DiscreteLinearOperator Op(&fespace_h1,&fespace_rtp);
               Op.AddDomainInterpolator(new IdentityInterpolator());
               Op.Assemble();

               Op.Mult(f0,f2);

               REQUIRE( f2.ComputeHDivError(&FCoef, &divFCoef) < tol );
            }
         }
         /// The following tests would fail. The reason for the failure would
         /// not be obvious from the user's point of view. I recommend keeping
         /// these tests here as a reminder that we should consider supporting
         /// this, or a very similar, usage.
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
      delete mesh;
   }
}

TEST_CASE("Derivative Linear Interpolators",
          "[GradientInterpolator]"
          "[CurlInterpolator]"
          "[DivergenceInterpolator]")
{
   int order_h1 = 1, order_nd = 1, order_rt = 0, order_l2 = 0, n = 3, dim = -1;
   double tol = 1e-9;

   for (int type = (int)Element::SEGMENT;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh *mesh = NULL;

      if (type < (int)Element::TRIANGLE)
      {
         dim = 1;
         mesh = new Mesh(n, (Element::Type)type, 1, 2.0);

      }
      else if (type < (int)Element::TETRAHEDRON)
      {
         dim = 2;
         mesh = new Mesh(n, n, (Element::Type)type, 1, 2.0, 3.0);
      }
      else
      {
         dim = 3;
         mesh = new Mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

         if (type == Element::TETRAHEDRON)
         {
            mesh->ReorientTetMesh();
         }
      }

      FunctionCoefficient        fCoef((dim==1) ? f1 :
                                       ((dim==2) ? f2 : f3));
      FunctionCoefficient       gradfCoef(grad_f1);
      VectorFunctionCoefficient GradfCoef(dim,
                                          (dim==1) ? Grad_f1 :
                                          ((dim==2)? Grad_f2 : Grad_f3));

      SECTION("Operators on H1 for element type " + std::to_string(type))
      {
         H1_FECollection    fec_h1(order_h1, dim);
         FiniteElementSpace fespace_h1(mesh, &fec_h1);

         GridFunction f0(&fespace_h1);
         f0.ProjectCoefficient(fCoef);

         if (dim ==1)
         {
            SECTION("Mapping to L2")
            {
               L2_FECollection    fec_l2(order_l2, dim);
               FiniteElementSpace fespace_l2(mesh, &fec_l2);

               GridFunction df0(&fespace_l2);

               DiscreteLinearOperator Op(&fespace_h1,&fespace_l2);
               Op.AddDomainInterpolator(new GradientInterpolator());
               Op.Assemble();

               Op.Mult(f0,df0);

               REQUIRE( df0.ComputeL2Error(gradfCoef) < tol );
            }
            SECTION("Mapping to L2 (INTEGRAL)")
            {
               L2_FECollection    fec_l2(order_l2, dim,
                                         BasisType::GaussLegendre,
                                         FiniteElement::INTEGRAL);
               FiniteElementSpace fespace_l2(mesh, &fec_l2);

               GridFunction df0(&fespace_l2);

               DiscreteLinearOperator Op(&fespace_h1,&fespace_l2);
               Op.AddDomainInterpolator(new GradientInterpolator());
               Op.Assemble();

               Op.Mult(f0,df0);

               REQUIRE( df0.ComputeL2Error(gradfCoef) < tol );
            }

         }
         else
         {
            SECTION("Mapping to HCurl")
            {
               ND_FECollection    fec_nd(order_nd, dim);
               FiniteElementSpace fespace_nd(mesh, &fec_nd);

               GridFunction df0(&fespace_nd);

               DiscreteLinearOperator Op(&fespace_h1,&fespace_nd);
               Op.AddDomainInterpolator(new GradientInterpolator());
               Op.Assemble();

               Op.Mult(f0,df0);

               REQUIRE( df0.ComputeL2Error(GradfCoef) < tol );
            }
         }
      }
      if (dim > 1)
      {
         VectorFunctionCoefficient     FCoef(dim,
                                             (dim==2) ? F2 : F3);
         FunctionCoefficient       curlFCoef(curlF2);
         VectorFunctionCoefficient CurlFCoef(dim,
                                             (dim==2) ? CurlF2 : CurlF3);
         FunctionCoefficient        DivFCoef((dim==2) ? DivF2 : DivF3);

         SECTION("Operators on HCurl for element type " + std::to_string(type))
         {
            ND_FECollection    fec_nd(order_nd, dim);
            FiniteElementSpace fespace_nd(mesh, &fec_nd);

            GridFunction F1(&fespace_nd);
            F1.ProjectCoefficient(FCoef);

            if (dim == 2)
            {
               SECTION("Mapping to L2")
               {
                  L2_FECollection    fec_l2(order_l2, dim);
                  FiniteElementSpace fespace_l2(mesh, &fec_l2);

                  GridFunction dF1(&fespace_l2);

                  DiscreteLinearOperator Op(&fespace_nd,&fespace_l2);
                  Op.AddDomainInterpolator(new CurlInterpolator());
                  Op.Assemble();

                  Op.Mult(F1,dF1);

                  REQUIRE( dF1.ComputeL2Error(curlFCoef) < tol );
               }
               SECTION("Mapping to L2 (INTEGRAL)")
               {
                  L2_FECollection    fec_l2(order_l2, dim,
                                            BasisType::GaussLegendre,
                                            FiniteElement::INTEGRAL);
                  FiniteElementSpace fespace_l2(mesh, &fec_l2);

                  GridFunction dF1(&fespace_l2);

                  DiscreteLinearOperator Op(&fespace_nd,&fespace_l2);
                  Op.AddDomainInterpolator(new CurlInterpolator());
                  Op.Assemble();

                  Op.Mult(F1,dF1);

                  REQUIRE( dF1.ComputeL2Error(curlFCoef) < tol );
               }
            }
            else
            {
               SECTION("Mapping to HDiv")
               {
                  RT_FECollection    fec_rt(order_rt, dim);
                  FiniteElementSpace fespace_rt(mesh, &fec_rt);

                  GridFunction dF1(&fespace_rt);

                  DiscreteLinearOperator Op(&fespace_nd,&fespace_rt);
                  Op.AddDomainInterpolator(new CurlInterpolator());
                  Op.Assemble();

                  Op.Mult(F1,dF1);

                  REQUIRE( dF1.ComputeL2Error(CurlFCoef) < tol );
               }
            }
         }
         SECTION("Operators on HDiv for element type " + std::to_string(type))
         {
            RT_FECollection    fec_rt(order_rt, dim);
            FiniteElementSpace fespace_rt(mesh, &fec_rt);

            GridFunction F2(&fespace_rt);
            F2.ProjectCoefficient(FCoef);

            SECTION("Mapping to L2")
            {
               L2_FECollection    fec_l2(order_l2, dim);
               FiniteElementSpace fespace_l2(mesh, &fec_l2);

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
               FiniteElementSpace fespace_l2(mesh, &fec_l2);

               GridFunction dF2(&fespace_l2);

               DiscreteLinearOperator Op(&fespace_rt,&fespace_l2);
               Op.AddDomainInterpolator(new DivergenceInterpolator());
               Op.Assemble();

               Op.Mult(F2,dF2);

               REQUIRE( dF2.ComputeL2Error(DivFCoef) < tol );
            }
         }
      }
      delete mesh;
   }
}

TEST_CASE("Product Linear Interpolators",
          "[ScalarProductInterpolator]"
          "[VectorScalarProductInterpolator]"
          "[ScalarVectorProductInterpolator]"
          "[ScalarCrossProductInterpolator]"
          "[VectorCrossProductInterpolator]"
          "[VectorInnerProductInterpolator]")
{
   int order_h1 = 1, order_nd = 2, order_rt = 2, n = 3, dim = -1;
   double tol = 1e-9;

   for (int type = (int)Element::SEGMENT;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh *mesh = NULL;

      if (type < (int)Element::TRIANGLE)
      {
         dim = 1;
         mesh = new Mesh(n, (Element::Type)type, 1, 2.0);

      }
      else if (type < (int)Element::TETRAHEDRON)
      {
         dim = 2;
         mesh = new Mesh(n, n, (Element::Type)type, 1, 2.0, 3.0);
      }
      else
      {
         dim = 3;
         mesh = new Mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

         if (type == Element::TETRAHEDRON)
         {
            mesh->ReorientTetMesh();
         }
      }

      FunctionCoefficient        fCoef((dim==1) ? f1 :
                                       ((dim==2) ? f2 : f3));
      FunctionCoefficient        gCoef((dim==1) ? g1 :
                                       ((dim==2) ? g2 : g3));
      FunctionCoefficient        fgCoef((dim==1) ? fg1 :
                                        ((dim==2) ? fg2 : fg3));

      VectorFunctionCoefficient   FCoef(dim,
                                        (dim==2) ? F2 : F3);
      VectorFunctionCoefficient   GCoef(dim,
                                        (dim==2) ? G2 : G3);

      FunctionCoefficient        FGCoef((dim==2) ? FdotG2 : FdotG3);
      VectorFunctionCoefficient  fGCoef(dim, (dim==2) ? fG2 : fG3);
      VectorFunctionCoefficient  FgCoef(dim, (dim==2) ? Fg2 : Fg3);
      VectorFunctionCoefficient FxGCoef(dim, (dim==2) ? FcrossG2 : FcrossG3);

      SECTION("Operators on H1 for element type " + std::to_string(type))
      {
         H1_FECollection    fec_h1(order_h1, dim);
         FiniteElementSpace fespace_h1(mesh, &fec_h1);

         GridFunction g0(&fespace_h1);
         g0.ProjectCoefficient(gCoef);

         SECTION("Mapping H1 to H1")
         {
            H1_FECollection    fec_h1p(2*order_h1, dim);
            FiniteElementSpace fespace_h1p(mesh, &fec_h1p);

            DiscreteLinearOperator Opf0(&fespace_h1,&fespace_h1p);
            Opf0.AddDomainInterpolator(
               new ScalarProductInterpolator(fCoef));
            Opf0.Assemble();

            GridFunction fg0(&fespace_h1p);
            Opf0.Mult(g0,fg0);

            REQUIRE( fg0.ComputeL2Error(fgCoef) < tol );
         }
         if (dim > 1)
         {
            SECTION("Mapping to HCurl")
            {
               ND_FECollection    fec_nd(order_nd, dim);
               FiniteElementSpace fespace_nd(mesh, &fec_nd);

               ND_FECollection    fec_ndp(order_h1+order_nd, dim);
               FiniteElementSpace fespace_ndp(mesh, &fec_ndp);

               DiscreteLinearOperator OpF1(&fespace_h1,&fespace_ndp);
               OpF1.AddDomainInterpolator(
                  new VectorScalarProductInterpolator(FCoef));
               OpF1.Assemble();

               GridFunction Fg1(&fespace_ndp);
               OpF1.Mult(g0,Fg1);

               REQUIRE( Fg1.ComputeL2Error(FgCoef) < tol );
            }
         }
      }
      if (dim > 1)
      {
         SECTION("Operators on HCurl for element type " + std::to_string(type))
         {
            ND_FECollection    fec_nd(order_nd, dim);
            FiniteElementSpace fespace_nd(mesh, &fec_nd);

            GridFunction G1(&fespace_nd);
            G1.ProjectCoefficient(GCoef);

            SECTION("Mapping HCurl to HCurl")
            {
               H1_FECollection    fec_h1(order_h1, dim);
               FiniteElementSpace fespace_h1(mesh, &fec_h1);

               ND_FECollection    fec_ndp(order_nd+order_h1, dim);
               FiniteElementSpace fespace_ndp(mesh, &fec_ndp);

               DiscreteLinearOperator Opf0(&fespace_nd,&fespace_ndp);
               Opf0.AddDomainInterpolator(
                  new ScalarVectorProductInterpolator(fCoef));
               Opf0.Assemble();

               GridFunction fG1(&fespace_ndp);
               Opf0.Mult(G1,fG1);

               REQUIRE( fG1.ComputeL2Error(fGCoef) < tol );
            }
            if (dim == 2)
            {
               SECTION("Mapping to L2")
               {
                  L2_FECollection    fec_l2p(2*order_nd-1, dim);
                  FiniteElementSpace fespace_l2p(mesh, &fec_l2p);

                  DiscreteLinearOperator OpF1(&fespace_nd,&fespace_l2p);
                  OpF1.AddDomainInterpolator(
                     new ScalarCrossProductInterpolator(FCoef));
                  OpF1.Assemble();

                  GridFunction FxG2(&fespace_l2p);
                  OpF1.Mult(G1,FxG2);

                  REQUIRE( FxG2.ComputeL2Error(FxGCoef) < tol );
               }
            }
            else
            {
               SECTION("Mapping to HDiv")
               {
                  RT_FECollection    fec_rtp(2*order_nd-1, dim);
                  FiniteElementSpace fespace_rtp(mesh, &fec_rtp);

                  DiscreteLinearOperator OpF1(&fespace_nd,&fespace_rtp);
                  OpF1.AddDomainInterpolator(
                     new VectorCrossProductInterpolator(FCoef));
                  OpF1.Assemble();

                  GridFunction FxG2(&fespace_rtp);
                  OpF1.Mult(G1,FxG2);

                  REQUIRE( FxG2.ComputeL2Error(FxGCoef) < tol );
               }
            }
            SECTION("Mapping to L2")
            {
               RT_FECollection    fec_rt(order_rt, dim);
               FiniteElementSpace fespace_rt(mesh, &fec_rt);

               L2_FECollection    fec_l2p(order_nd+order_rt, dim);
               FiniteElementSpace fespace_l2p(mesh, &fec_l2p);

               DiscreteLinearOperator OpF2(&fespace_nd,&fespace_l2p);
               OpF2.AddDomainInterpolator(
                  new VectorInnerProductInterpolator(FCoef));
               OpF2.Assemble();

               GridFunction FG3(&fespace_l2p);
               OpF2.Mult(G1,FG3);

               REQUIRE( FG3.ComputeL2Error(FGCoef) < tol );
            }
         }
         SECTION("Operators on HDiv for element type " + std::to_string(type))
         {
            RT_FECollection    fec_rt(order_rt, dim);
            FiniteElementSpace fespace_rt(mesh, &fec_rt);

            GridFunction G2(&fespace_rt);
            G2.ProjectCoefficient(GCoef);

            SECTION("Mapping to L2")
            {
               ND_FECollection    fec_nd(order_nd, dim);
               FiniteElementSpace fespace_nd(mesh, &fec_nd);

               L2_FECollection    fec_l2p(order_nd+order_rt, dim);
               FiniteElementSpace fespace_l2p(mesh, &fec_l2p);

               DiscreteLinearOperator OpF1(&fespace_rt,&fespace_l2p);
               OpF1.AddDomainInterpolator(
                  new VectorInnerProductInterpolator(FCoef));
               OpF1.Assemble();

               GridFunction FG3(&fespace_l2p);
               OpF1.Mult(G2,FG3);

               REQUIRE( FG3.ComputeL2Error(FGCoef) < tol );
            }
         }
      }
   }
}

} // namespace lin_interp
