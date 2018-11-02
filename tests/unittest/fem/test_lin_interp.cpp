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
#include "catch.hpp"

using namespace mfem;

namespace lin_interp
{

double f3(const Vector & x)
{ return 2.345 * x[0] + 3.579 * x[1] + 4.680 * x[2]; }
void F3(const Vector & x, Vector & v)
{
   v.SetSize(3);
   v[0] =  1.234 * x[0] - 2.357 * x[1] + 3.572 * x[2];
   v[1] =  2.537 * x[0] + 4.321 * x[1] - 1.234 * x[2];
   v[2] = -2.572 * x[0] + 1.321 * x[1] + 3.234 * x[2];
}

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

TEST_CASE("Linear Interpolators")
{
   int order_h1 = 1, order_nd = 2, order_rt = 2, n = 3, dim = 3;
   double tol = 1e-9;

   Mesh mesh(n, n, n, Element::HEXAHEDRON, 1, 2.0, 3.0, 5.0);

   FunctionCoefficient       fCoef(f3);
   VectorFunctionCoefficient FCoef(dim, F3);

   FunctionCoefficient       gCoef(g3);
   VectorFunctionCoefficient GCoef(dim, G3);

   FunctionCoefficient        fgCoef(fg3);
   FunctionCoefficient        FGCoef(FdotG3);
   VectorFunctionCoefficient  fGCoef(dim, fG3);
   VectorFunctionCoefficient  FgCoef(dim, Fg3);
   VectorFunctionCoefficient FxGCoef(dim, FcrossG3);

   SECTION("Operators on H1")
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
   SECTION("Operators on HCurl")
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
   SECTION("Operators on HDiv")
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

} // namespace lin_interp
