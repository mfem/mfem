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
#include "catch.hpp"

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <random>

using namespace mfem;

namespace
{
// String used to define a single element mesh, with a c-shaped quad element
std::string mesh_str =
   "MFEM mesh v1.0"                    "\n\n"
   "dimension"                           "\n"
   "2"                                 "\n\n"
   "elements"                            "\n"
   "1"                                   "\n"
   "1 3 0 1 2 3"                       "\n\n"
   "boundary"                            "\n"
   "0"                                 "\n\n"
   "vertices"                            "\n"
   "4"                                 "\n\n"
   "nodes"                               "\n"
   "FiniteElementSpace"                  "\n"
   "FiniteElementCollection: Quadratic"  "\n"
   "VDim: 2"                             "\n"
   "Ordering: 1"                         "\n"
   "0 0"                                 "\n"
   "0 2"                                 "\n"
   "0 6"                                 "\n"
   "0 8"                                 "\n"
   "0 1"                                 "\n"
   "-6 4"                                "\n"
   "0 7"                                 "\n"
   "-8 4"                                "\n"
   "-7 4"                                "\n";

double scalar_func(const Vector &x)
{
   double q = 0;
   for (int i = 0; i < x.Size(); ++i)
   {
      q += pow(x(i), 2);
   }
   return q;
}

void scalar_funcRevDiff(const mfem::Vector &x, const double q_bar, mfem::Vector &x_bar)
{
   for (int i = 0; i < x.Size(); ++i)
   {
      x_bar(i) += q_bar * 2 * x(i);
   }
}

double scalar_func2(const Vector &x)
{
   double q = 0;
   for (int i = 0; i < x.Size(); ++i)
   {
      q += x(i);
   }
   return q;
}

void scalar_func2RevDiff(const mfem::Vector &x, const double q_bar, mfem::Vector &x_bar)
{
   for (int i = 0; i < x.Size(); ++i)
   {
      x_bar(i) += q_bar;
   }
}

void func2D(const Vector &x, Vector &y)
{
   y.SetSize(2);
   y(0) = x(0)*x(0) - x(1);
   y(1) = x(0) * exp(x(1));
}

void func2DRevDiff(const Vector &x, const Vector &v_bar, Vector &x_bar)
{
   x_bar(0) = v_bar(0) * 2*x(0) + v_bar(1) * exp(x(1));
   x_bar(1) = -v_bar(0) + v_bar(1) * x(0) * exp(x(1));
}

void func3D(const Vector &x, Vector &y)
{
   y.SetSize(3);
   y(0) = x(0)*x(0) - x(1);
   y(1) = x(0) * exp(x(1));
   y(2) = x(2)*x(0) - x(1);
}

void func3DRevDiff(const Vector &x, const Vector &v_bar, Vector &x_bar)
{
   x_bar(0) = v_bar(0) * 2*x(0) + v_bar(1) * exp(x(1)) + v_bar(2)*x(2);
   x_bar(1) = -v_bar(0) + v_bar(1) * x(0) * exp(x(1)) - v_bar(2);
   x_bar(2) = v_bar(2) * x(0);
}

void runScalarTest(Mesh &mesh, Coefficient &q)
{
   constexpr double eps_fd = 1e-5;
   std::default_random_engine generator;
   std::uniform_real_distribution<double> distribution(-1.0,1.0);

   for (int p = 1; p <= 4; ++p)
   {
      const int dim = mesh.Dimension();
      H1_FECollection fec(p, dim);
      FiniteElementSpace fes(&mesh, &fec);

      const FiniteElement &el = *fes.GetFE(0);
      IsoparametricTransformation trans;
      mesh.GetElementTransformation(0, &trans);

      DenseMatrix &coords = trans.GetPointMat();
      DenseMatrix coords_bar(coords.Height(), coords.Width());

      double Q_bar = distribution(generator);

      int order = trans.OrderW() + 2 * el.GetOrder();
      const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), order);
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint (&ip);

         // reverse-mode differentiation of Eval
         coords_bar = 0.0;
         q.EvalRevDiff(Q_bar, trans, ip, coords_bar);

         // get the weighted derivatives using finite difference method
         for (int n = 0; n < coords.Width(); ++n)
         {
            for (int di = 0; di < coords.Height(); ++di)
            {
               coords(di, n) += eps_fd;
               double Q_fd = q.Eval(trans, ip);
               coords(di, n) -= 2.0*eps_fd;
               Q_fd -= q.Eval(trans, ip);
               Q_fd /= (2.0 * eps_fd);
               coords(di, n) += eps_fd;
               double x_bar_fd = Q_bar * Q_fd;

               REQUIRE(coords_bar(di, n) == Approx(x_bar_fd));
            }
         }
      }
   }
}

void runVectorTest(Mesh &mesh, VectorCoefficient &vc)
{
   constexpr double eps_fd = 1e-5;
   std::default_random_engine generator;
   std::uniform_real_distribution<double> distribution(-1.0,1.0);

   for (int p = 1; p <= 4; ++p)
   {
      const int dim = mesh.Dimension();
      ND_FECollection fec(p, dim);
      FiniteElementSpace fes(&mesh, &fec);

      const FiniteElement &el = *fes.GetFE(0);
      IsoparametricTransformation trans;
      mesh.GetElementTransformation(0, &trans);

      DenseMatrix &coords = trans.GetPointMat();
      DenseMatrix coords_bar(coords.Height(), coords.Width());

      Vector V_bar(dim), V_fd(dim), V_pert(dim);
      for (int i = 0; i < V_bar.Size(); ++i)
      {
         V_bar(i) = distribution(generator);
      }

      int order = trans.OrderW() + 2 * el.GetOrder();
      const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), order);
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint (&ip);

         // reverse-mode differentiation of Eval
         coords_bar = 0.0;
         vc.EvalRevDiff(V_bar, trans, ip, coords_bar);

         // get the weighted derivatives using finite difference method
         for (int n = 0; n < coords.Width(); ++n)
         {
            for (int di = 0; di < coords.Height(); ++di)
            {
               coords(di, n) += eps_fd;
               vc.Eval(V_fd, trans, ip);
               coords(di, n) -= 2.0*eps_fd;
               vc.Eval(V_pert, trans, ip);
               V_fd -= V_pert;
               V_fd /= (2.0 * eps_fd);
               coords(di, n) += eps_fd;
               double x_bar_fd = V_bar * V_fd;

               REQUIRE(coords_bar(di, n) == Approx(x_bar_fd));
            }
         }
      }
   }
}

} // anonymous namespace

namespace coeff_revdiff
{

TEST_CASE("CoeffRevDiff::FunctionCoefficient::EvalRevDiff_2D")
{
   // Create quadratic mesh with single C-shaped quadrilateral
   std::stringstream meshStr;
   meshStr << mesh_str;
   Mesh mesh2D(meshStr);

   REQUIRE( mesh2D.GetNE() == 1 );
   REQUIRE( mesh2D.GetNodes() != NULL );

   FunctionCoefficient c1(scalar_func, scalar_funcRevDiff);
   runScalarTest(mesh2D, c1);

   FunctionCoefficient c2(scalar_func2, scalar_func2RevDiff);
   runScalarTest(mesh2D, c2);
}

TEST_CASE("CoeffRevDiff::ProductCoefficient::EvalRevDiff_2D 1 Coeff")
{
   // Create quadratic mesh with single C-shaped quadrilateral
   std::stringstream meshStr;
   meshStr << mesh_str;
   Mesh mesh2D(meshStr);

   REQUIRE( mesh2D.GetNE() == 1 );
   REQUIRE( mesh2D.GetNodes() != NULL );

   FunctionCoefficient c1(scalar_func, scalar_funcRevDiff);
   ProductCoefficient prod(2.0, c1);

   runScalarTest(mesh2D, prod);
}

TEST_CASE("CoeffRevDiff::ProductCoefficient::EvalRevDiff_2D 2 Coeffs")
{
   // Create quadratic mesh with single C-shaped quadrilateral
   std::stringstream meshStr;
   meshStr << mesh_str;
   Mesh mesh2D(meshStr);

   REQUIRE( mesh2D.GetNE() == 1 );
   REQUIRE( mesh2D.GetNodes() != NULL );

   FunctionCoefficient c1(scalar_func, scalar_funcRevDiff);
   FunctionCoefficient c2(scalar_func2, scalar_func2RevDiff);

   ProductCoefficient prod(c1, c2);

   runScalarTest(mesh2D, prod);
}


TEST_CASE("CoeffRevDiff::VectorFunctionCoefficient::EvalRevDiff_2D")
{
   // Create quadratic mesh with single C-shaped quadrilateral
   std::stringstream meshStr;
   meshStr << mesh_str;
   Mesh mesh2D(meshStr);

   REQUIRE( mesh2D.GetNE() == 1 );
   REQUIRE( mesh2D.GetNodes() != NULL );

   VectorFunctionCoefficient vc2D(2, func2D, func2DRevDiff);

   runVectorTest(mesh2D, vc2D);
}

TEST_CASE("CoeffRevDiff::VectorFunctionCoefficient::EvalRevDiff_3D")
{
   auto mesh3D = Mesh::MakeCartesian3D(2, 2, 2,
                                       Element::TETRAHEDRON,
                                       2.0, 1.0, 3.0, true);
   mesh3D.EnsureNodes();

   VectorFunctionCoefficient vc3D(3, func3D, func3DRevDiff);

   runVectorTest(mesh3D, vc3D);
}

TEST_CASE("CoeffRevDiff::ScalarVectorProductCoefficient::EvalRevDiff_3D")
{
   auto mesh3D = Mesh::MakeCartesian3D(2, 2, 2,
                                       Element::TETRAHEDRON,
                                       2.0, 1.0, 3.0, true);
   mesh3D.EnsureNodes();

   VectorFunctionCoefficient vfc(3, func3D, func3DRevDiff);

   ScalarVectorProductCoefficient vc(2.0, vfc);

   runVectorTest(mesh3D, vc);
}

} // namespace coeff_revdiff
